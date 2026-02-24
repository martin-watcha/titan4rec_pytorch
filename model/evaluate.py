import math
import random

import numpy as np
import torch

EVAL_BATCH = 256  # users per inference batch


def evaluate(model, dataset, max_len, is_test=False, seed=42):
    """Evaluate model using 101-candidate ranking (SASRec protocol).

    Args:
        model: must have .predict(seqs, item_indices) -> (B, N) scores.
               item_indices can be 1-D (N,) [shared] or 2-D (B, N) [per-user].
        dataset: [seq_train, seq_val, seq_test, user_num, item_num]
        max_len: int
        is_test: bool
        seed: int — fixed seed for deterministic user sampling and negatives.

    Returns:
        (ndcg@10, hit@10)
    """
    model.eval()
    train, val, test, user_num, item_num = dataset

    # Use local RNG instances for deterministic evaluation across epochs
    py_rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    users = py_rng.sample(range(1, user_num + 1), min(10000, user_num))

    # --- Pre-build all valid users' sequences and candidate item lists ---
    all_seqs = []
    all_items = []

    for u in users:
        if len(train[u]) < 1:
            continue
        target_seq = test if is_test else val
        if len(target_seq[u]) < 1:
            continue

        seq = np.zeros(max_len, dtype=np.int64)
        idx = max_len - 1

        if is_test:
            seq[idx] = val[u][0]
            idx -= 1

        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx < 0:
                break

        rated = set(train[u])
        rated.add(0)
        if is_test and len(val[u]) > 0:
            rated.add(val[u][0])
        target = target_seq[u][0]
        rated.add(target)  # prevent target from appearing as a negative

        # 101 candidates: 1 positive + 100 negatives
        item_idx = [target]
        for _ in range(100):
            t = np_rng.randint(1, item_num + 1)
            while t in rated:
                t = np_rng.randint(1, item_num + 1)
            item_idx.append(t)

        all_seqs.append(seq)
        all_items.append(item_idx)

    # --- Batched inference + vectorized ranking on GPU ---
    all_ranks = []
    device = model.device

    with torch.no_grad():
        for start in range(0, len(all_seqs), EVAL_BATCH):
            seqs_np = np.stack(all_seqs[start : start + EVAL_BATCH])   # (B, T)
            items_np = np.array(all_items[start : start + EVAL_BATCH]) # (B, 101)

            seqs_t  = torch.from_numpy(seqs_np).long().to(device)
            items_t = torch.from_numpy(items_np).long().to(device)

            scores = model.predict(seqs_t, items_t)  # (B, 101)

            # Vectorized rank computation on GPU
            # rank[i] = position of index 0 (positive item) in descending sort
            ranks = scores.argsort(dim=-1, descending=True).argsort(dim=-1)[:, 0]  # (B,)

            # Filter degenerate users: non-finite or constant scores
            finite_mask = scores.isfinite().all(dim=-1)                    # (B,)
            not_constant = scores.max(dim=-1).values != scores.min(dim=-1).values  # (B,)
            valid_mask = finite_mask & not_constant

            # Set invalid ranks to a value >= k so they contribute 0 to metrics
            ranks = ranks.where(valid_mask, torch.tensor(999, device=device))
            all_ranks.append(ranks)

    if not all_ranks:
        return 0.0, 0.0

    ranks = torch.cat(all_ranks)  # (total_users,)
    valid_mask = ranks < 999
    valid_count = valid_mask.sum().item()

    if valid_count == 0:
        return 0.0, 0.0

    # Vectorized NDCG@10 and HR@10 on GPU
    k = 10
    hit_mask = (ranks < k) & valid_mask
    hr = hit_mask.sum().item() / valid_count
    # ndcg = 1 / log2(rank + 2) for ranks < k, 0 otherwise
    ndcg_vals = torch.where(
        hit_mask,
        1.0 / torch.log2(ranks.float() + 2.0),
        torch.tensor(0.0, device=ranks.device),
    )
    ndcg = ndcg_vals.sum().item() / valid_count

    return ndcg, hr
