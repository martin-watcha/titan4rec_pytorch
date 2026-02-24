import torch
import torch.nn as nn
import torch.nn.functional as F

from . import RMSNorm
from .embedding import Titan4RecEmbedding
from .mac_block import MACBlock


class Titan4Rec(nn.Module):
    def __init__(
        self,
        num_items,
        d_model=64,
        num_blocks=2,
        num_heads=2,
        segment_size=20,
        memory_depth=2,
        memory_heads=2,
        expansion_factor=4,
        num_persistent=4,
        max_len=200,
        max_lr=0.01,
        dropout_rate=0.2,
        tbptt_k=3,
        **kwargs,
    ):
        super().__init__()
        self.segment_size = segment_size
        self.max_len = max_len
        self.num_items = num_items
        self.tbptt_k = tbptt_k

        # Positional embedding covers the max padded sequence length
        max_padded_len = ((max_len + segment_size - 1) // segment_size) * segment_size
        self.embedding = Titan4RecEmbedding(
            num_items, d_model, max_padded_len, dropout_rate
        )

        self.mac_blocks = nn.ModuleList(
            [
                MACBlock(
                    d_model,
                    num_heads,
                    num_persistent,
                    memory_depth,
                    memory_heads,
                    expansion_factor,
                    max_lr,
                    dropout_rate,
                )
                for _ in range(num_blocks)
            ]
        )

        self.final_norm = RMSNorm(d_model)

    @property
    def device(self):
        return self.embedding.item_emb.weight.device

    def log2feats(self, input_seq):
        """
        Args: input_seq (B, T) LongTensor
        Returns: (B, T, d_model)
        """
        B = input_seq.shape[0]
        C = self.segment_size

        # 1. Item Embedding
        x = self.embedding(input_seq)
        seq_len = x.shape[1]

        # 2. Pad to multiple of C (left-pad)
        remainder = seq_len % C
        if remainder != 0:
            pad_len = C - remainder
            x = F.pad(x, (0, 0, pad_len, 0), value=0.0)
        else:
            pad_len = 0

        padded_len = x.shape[1]
        num_segments = padded_len // C

        # 3. Padding mask (0 = padding)
        input_padded = (
            F.pad(input_seq, (pad_len, 0), value=0) if pad_len > 0 else input_seq
        )
        token_mask = input_padded != 0  # (B, padded_len)

        # 4. Absolute positional encoding (full sequence, then zero padding)
        pos_enc = self.embedding.get_position_encoding(padded_len, device=x.device)
        x = x + pos_enc.unsqueeze(0)
        x = x * token_mask.unsqueeze(-1).float()

        # 5. Split into segments
        seg_masks = token_mask.view(B, num_segments, C)
        segments = x.view(B, num_segments, C, -1)

        # 6. Initialize memory states per block
        memory_states, momentum_states = [], []
        for block in self.mac_blocks:
            ms, mom = block.ltm.init_memory_state(B)
            memory_states.append(ms)
            momentum_states.append(mom)

        # 7. Process segments sequentially (TBPTT: detach memory beyond last tbptt_k segments)
        outputs = []
        for t in range(num_segments):
            seg = segments[:, t, :, :]
            seg_mask = seg_masks[:, t, :]
            seg_mask_float = seg_mask.unsqueeze(-1).float()  # (B, C, 1) — once per segment
            for i, block in enumerate(self.mac_blocks):
                seg, memory_states[i], momentum_states[i] = block(
                    seg, memory_states[i], momentum_states[i], padding_mask=seg_mask
                )
                # Zero out padding after each block (like SASRec).
                # Without this, garbage at padding positions propagates to the
                # next block's LTM retrieval and attention prefix.
                seg = seg * seg_mask_float
            outputs.append(seg)
            # Detach memory state for segments outside the TBPTT window
            steps_remaining = num_segments - 1 - t
            if steps_remaining >= self.tbptt_k:
                for i in range(len(self.mac_blocks)):
                    memory_states[i] = {k: v.detach() for k, v in memory_states[i].items()}
                    momentum_states[i] = {k: v.detach() for k, v in momentum_states[i].items()}

        # 8. Concat + remove padding
        out = torch.cat(outputs, dim=1)
        if pad_len > 0:
            out = out[:, pad_len:, :]

        return self.final_norm(out)

    def forward(self, seqs, pos_seqs, neg_seqs):
        """Training forward pass.
        Args: seqs, pos_seqs, neg_seqs - LongTensor (B, T)
        Returns: (pos_logits, neg_logits) - Tensors (B, T)
        """
        log_feats = self.log2feats(seqs)

        pos_emb = self.embedding.item_emb(pos_seqs)
        neg_emb = self.embedding.item_emb(neg_seqs)

        pos_logits = (log_feats * pos_emb).sum(dim=-1)
        neg_logits = (log_feats * neg_emb).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self, seqs, item_indices):
        """Inference: score candidate items.
        Args:
            seqs: LongTensor (B, T)
            item_indices: LongTensor (N,) or (B, N) - shared or per-user candidates
        Returns: (B, N) scores
        """
        log_feats = self.log2feats(seqs)
        final_feat = log_feats[:, -1, :]           # (B, d_model)
        item_embs = self.embedding.item_emb(item_indices)
        if item_indices.dim() == 2:
            # Per-user candidates: item_embs is (B, N, d_model)
            return (final_feat.unsqueeze(1) * item_embs).sum(-1)  # (B, N)
        return final_feat @ item_embs.t()          # (B, N) shared
