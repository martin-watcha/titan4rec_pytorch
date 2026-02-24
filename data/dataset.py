from pathlib import Path

import lightning as L
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset

from .preprocess import data_load

_DIR = Path(__file__).parent


def _worker_init_fn(worker_id):
    """Seed each DataLoader worker independently to avoid duplicate samples."""
    np.random.seed((torch.initial_seed() + worker_id) % (2**32))


def data_split(dataset_name: str):
    """Leave-one-out split following SASRec.

    Returns:
        (user_num, item_num, seq_train, seq_val, seq_test)
    """
    data_path = _DIR / "processed" / f"{dataset_name}_data.txt"

    user_items = defaultdict(list)
    user_num, item_num = 0, 0

    with open(data_path, "r") as f:
        for line in f:
            uid, iid = line.strip().split()
            uid, iid = int(uid), int(iid)
            user_items[uid].append(iid)
            user_num = max(user_num, uid)
            item_num = max(item_num, iid)

    seq_train = defaultdict(list)
    seq_val = defaultdict(list)
    seq_test = defaultdict(list)

    for uid in range(1, user_num + 1):
        items = user_items[uid]
        if len(items) < 3:
            seq_train[uid] = items
            seq_val[uid] = []
            seq_test[uid] = []
        else:
            seq_train[uid] = items[:-2]
            seq_val[uid] = [items[-2]]
            seq_test[uid] = [items[-1]]

    return user_num, item_num, seq_train, seq_val, seq_test


class SeqRecTrainDataset(Dataset):
    """Training dataset that samples (seq, pos, neg) per __getitem__."""

    def __init__(self, seq_train, user_num, item_num, max_len, num_batches_per_epoch, batch_size):
        self.seq_train = seq_train
        self.item_num = item_num
        self.max_len = max_len
        self.epoch_size = num_batches_per_epoch * batch_size
        self.valid_users = np.array(
            [uid for uid in range(1, user_num + 1) if len(seq_train[uid]) >= 3]
        )

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        uid = np.random.choice(self.valid_users)
        seq = np.zeros(self.max_len, dtype=np.int64)
        pos = np.zeros(self.max_len, dtype=np.int64)
        neg = np.zeros(self.max_len, dtype=np.int64)

        nxt = self.seq_train[uid][-1]
        write_idx = self.max_len - 1
        item_set = set(self.seq_train[uid])

        for item in reversed(self.seq_train[uid][:-1]):
            seq[write_idx] = item
            pos[write_idx] = nxt
            neg_item = np.random.randint(1, self.item_num + 1)
            while neg_item in item_set:
                neg_item = np.random.randint(1, self.item_num + 1)
            neg[write_idx] = neg_item
            nxt = item
            write_idx -= 1
            if write_idx < 0:
                break

        return torch.from_numpy(seq), torch.from_numpy(pos), torch.from_numpy(neg)


class BERT4RecTrainDataset(Dataset):
    """Training dataset for BERT4Rec: returns (masked_seq, labels)."""

    def __init__(self, seq_train, user_num, item_num, max_len,
                 num_batches_per_epoch, batch_size, mask_prob=0.15):
        self.seq_train = seq_train
        self.item_num = item_num
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = item_num + 1
        self.epoch_size = num_batches_per_epoch * batch_size
        self.valid_users = np.array(
            [uid for uid in range(1, user_num + 1) if len(seq_train[uid]) >= 3]
        )

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        uid = np.random.choice(self.valid_users)
        items = self.seq_train[uid]

        # Build left-padded sequence
        seq = np.zeros(self.max_len, dtype=np.int64)
        start = max(0, self.max_len - len(items))
        end = self.max_len
        src_start = max(0, len(items) - self.max_len)
        seq[start:end] = items[src_start:]

        # Create masked version and labels
        masked_seq = seq.copy()
        labels = np.zeros(self.max_len, dtype=np.int64)

        # Collect non-padding positions
        non_pad = np.where(seq != 0)[0]
        if len(non_pad) == 0:
            return torch.from_numpy(masked_seq), torch.from_numpy(labels)

        # Random masking with mask_prob, ensure at least 1 masked
        mask_flags = np.random.random(len(non_pad)) < self.mask_prob
        if not mask_flags.any():
            mask_flags[np.random.randint(len(non_pad))] = True

        for i, pos in enumerate(non_pad):
            if mask_flags[i]:
                labels[pos] = seq[pos]
                masked_seq[pos] = self.mask_token

        return torch.from_numpy(masked_seq), torch.from_numpy(labels)


class SeqRecDataModule(L.LightningDataModule):
    def __init__(self, dataset_name, max_len, batch_size, num_workers,
                 num_batches_per_epoch, model_name="sasrec", mask_prob=0.15):
        super().__init__()
        self.dataset_name = dataset_name
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_batches_per_epoch = num_batches_per_epoch
        self.model_name = model_name
        self.mask_prob = mask_prob

        self.user_num = None
        self.item_num = None
        self.dataset = None  # [seq_train, seq_val, seq_test, user_num, item_num]

    def setup(self, stage=None):
        if self.dataset is not None:
            return
        data_load(self.dataset_name)
        user_num, item_num, seq_train, seq_val, seq_test = data_split(self.dataset_name)
        self.user_num = user_num
        self.item_num = item_num
        self.dataset = [seq_train, seq_val, seq_test, user_num, item_num]

        if self.model_name == "bert4rec":
            self.train_ds = BERT4RecTrainDataset(
                seq_train, user_num, item_num, self.max_len,
                self.num_batches_per_epoch, self.batch_size,
                mask_prob=self.mask_prob,
            )
        else:
            self.train_ds = SeqRecTrainDataset(
                seq_train, user_num, item_num, self.max_len,
                self.num_batches_per_epoch, self.batch_size,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            worker_init_fn=_worker_init_fn,
        )

    def val_dataloader(self):
        # Dummy single-batch loader to trigger validation epoch
        dummy = torch.zeros(1, 1)
        return DataLoader(torch.utils.data.TensorDataset(dummy), batch_size=1)
