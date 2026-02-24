import torch
import torch.nn as nn
import torch.nn.functional as F


class BERT4Rec(nn.Module):
    """BERT4Rec: Bidirectional Transformer for sequential recommendation.

    Training: masked item prediction (Cloze task) with CrossEntropyLoss.
    Inference: append [MASK] token and use last position's output.

    forward(masked_seqs, labels) → loss   (different from SASRec)
    predict(seqs, item_indices) → scores  (same interface as SASRec)
    """

    def __init__(
        self,
        item_num,
        hidden_units=64,
        max_len=200,
        num_blocks=2,
        num_heads=2,
        dropout_rate=0.2,
        **kwargs,
    ):
        super().__init__()
        self.item_num = item_num
        self.max_len = max_len
        self.mask_token = item_num + 1  # [MASK] token ID

        # +2: 0=pad, 1..item_num=items, item_num+1=[MASK]
        self.item_emb = nn.Embedding(item_num + 2, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)

        # Bidirectional Transformer blocks (no causal mask)
        self.attention_layernorms = nn.ModuleList(
            [nn.LayerNorm(hidden_units) for _ in range(num_blocks)]
        )
        self.attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    hidden_units, num_heads, dropout_rate, batch_first=True
                )
                for _ in range(num_blocks)
            ]
        )
        self.ffn_layernorms = nn.ModuleList(
            [nn.LayerNorm(hidden_units) for _ in range(num_blocks)]
        )
        self.ffn_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_units, hidden_units * 4),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(hidden_units * 4, hidden_units),
                    nn.Dropout(dropout_rate),
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_units)

        # Prediction head bias (over item vocab, not including MASK)
        self.output_bias = nn.Parameter(torch.zeros(item_num + 1))

    @property
    def device(self):
        return self.item_emb.weight.device

    def encode(self, seqs):
        """Bidirectional encoding (no causal mask).
        Args: seqs (B, T) LongTensor — may contain [MASK] tokens
        Returns: (B, T, H)
        """
        x = self.item_emb(seqs)
        x = x * (self.item_emb.embedding_dim ** 0.5)
        positions = torch.arange(seqs.shape[1], device=seqs.device)
        x = x + self.pos_emb(positions)
        x = self.emb_dropout(x)

        # Padding mask only — no causal mask (bidirectional)
        padding_mask = seqs == 0  # (B, T)

        for ln_attn, attn, ln_ffn, ffn in zip(
            self.attention_layernorms,
            self.attention_layers,
            self.ffn_layernorms,
            self.ffn_layers,
        ):
            q = ln_attn(x)
            x = x + attn(
                q, q, q,
                key_padding_mask=padding_mask, need_weights=False,
            )[0]
            x = x + ffn(ln_ffn(x))

        return self.final_norm(x)

    def forward(self, masked_seqs, labels):
        """Training forward: masked item prediction.
        Args:
            masked_seqs: (B, T) — sequence with some items replaced by [MASK]
            labels: (B, T) — original item IDs at masked positions, 0 elsewhere
        Returns:
            loss: scalar CrossEntropyLoss
        """
        hidden = self.encode(masked_seqs)  # (B, T, H)

        # Project to item vocabulary (shared embedding weights, excluding MASK row)
        item_weights = self.item_emb.weight[: self.item_num + 1]  # (item_num+1, H)
        logits = F.linear(hidden, item_weights, self.output_bias)  # (B, T, item_num+1)

        # Only compute loss at masked positions
        mask = labels != 0  # (B, T)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        logits_masked = logits[mask]  # (num_masked, item_num+1)
        labels_masked = labels[mask]  # (num_masked,)

        loss = F.cross_entropy(logits_masked, labels_masked)
        return loss

    def predict(self, seqs, item_indices):
        """Inference: score candidate items at the last position.

        Appends [MASK] at the end (shifting sequence left by 1) so the model
        predicts the next item using bidirectional context.

        Args:
            seqs: LongTensor (B, T) — user history (left-padded)
            item_indices: LongTensor (N,) or (B, N) — shared or per-user candidates
        Returns: (B, N) scores
        """
        B, T = seqs.shape
        # Shift left by 1, append [MASK] token at the end
        mask_col = torch.full(
            (B, 1), self.mask_token, dtype=seqs.dtype, device=seqs.device
        )
        masked_seqs = torch.cat([seqs[:, 1:], mask_col], dim=1)  # (B, T)

        hidden = self.encode(masked_seqs)
        final_feat = hidden[:, -1, :]  # (B, H)
        item_embs = self.item_emb(item_indices)
        if item_indices.dim() == 2:
            return (final_feat.unsqueeze(1) * item_embs).sum(-1)  # (B, N)
        return final_feat @ item_embs.t()
