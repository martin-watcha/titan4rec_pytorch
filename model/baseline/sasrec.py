import torch
import torch.nn as nn


class SASRec(nn.Module):
    def __init__(
        self,
        item_num,
        hidden_units=64,
        max_len=200,
        num_blocks=2,
        num_heads=1,
        dropout_rate=0.2,
        **kwargs,
    ):
        super().__init__()
        self.item_num = item_num
        self.max_len = max_len

        self.item_emb = nn.Embedding(item_num + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)

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
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(hidden_units * 4, hidden_units),
                    nn.Dropout(dropout_rate),
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_units)

    @property
    def device(self):
        return self.item_emb.weight.device

    def log2feats(self, seqs):
        """
        Args: seqs (B, T) LongTensor
        Returns: (B, T, H)
        """
        x = self.item_emb(seqs)
        x = x * (self.item_emb.embedding_dim**0.5)
        positions = torch.arange(seqs.shape[1], device=seqs.device)
        x = x + self.pos_emb(positions)
        x = self.emb_dropout(x)

        T = seqs.shape[1]
        attn_mask = torch.triu(
            torch.ones(T, T, device=seqs.device), diagonal=1
        ).bool()

        # Zero out padding positions to avoid NaN from causal mask + padding
        # interaction (when all keys are masked, softmax produces NaN).
        pad_mask = (seqs != 0).unsqueeze(-1).float()  # (B, T, 1)
        x = x * pad_mask

        for ln_attn, attn, ln_ffn, ffn in zip(
            self.attention_layernorms,
            self.attention_layers,
            self.ffn_layernorms,
            self.ffn_layers,
        ):
            q = ln_attn(x)
            x = x + attn(
                q, q, q, attn_mask=attn_mask, need_weights=False,
            )[0]
            x = x + ffn(ln_ffn(x))
            x = x * pad_mask

        return self.final_norm(x)

    def forward(self, seqs, pos_seqs, neg_seqs):
        """Training forward pass.
        Args: seqs, pos_seqs, neg_seqs - LongTensor (B, T)
        Returns: (pos_logits, neg_logits) - Tensors (B, T)
        """
        log_feats = self.log2feats(seqs)

        pos_emb = self.item_emb(pos_seqs)
        neg_emb = self.item_emb(neg_seqs)

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
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(item_indices)
        if item_indices.dim() == 2:
            return (final_feat.unsqueeze(1) * item_embs).sum(-1)  # (B, N)
        return final_feat @ item_embs.t()
