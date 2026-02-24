import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelectiveSSM(nn.Module):
    """Selective State Space Model (S6) — pure PyTorch, for-loop scan."""

    def __init__(self, d_inner, d_state=16, dt_rank=None):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank or math.ceil(d_inner / 16)

        # x → (dt, B, C) projections
        self.x_proj = nn.Linear(d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner)

        # A: structured as log for stability
        log_A = torch.log(
            torch.arange(1, d_state + 1, dtype=torch.float32)
            .unsqueeze(0)
            .expand(d_inner, -1)
        )
        self.log_A = nn.Parameter(log_A)  # (d_inner, d_state)
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x, pad_mask=None):
        """
        Args:
            x: (B, T, d_inner)
            pad_mask: (B, T) bool — True for padding positions
        Returns:
            y: (B, T, d_inner)
        """
        B, T, D = x.shape

        # Project x → dt, B_t, C_t
        xp = self.x_proj(x)  # (B, T, dt_rank + 2*d_state)
        dt = F.softplus(self.dt_proj(xp[..., : self.dt_rank]))  # (B, T, d_inner)
        B_t = xp[..., self.dt_rank : self.dt_rank + self.d_state]  # (B, T, d_state)
        C_t = xp[..., self.dt_rank + self.d_state :]  # (B, T, d_state)

        A = -torch.exp(self.log_A)  # (d_inner, d_state)

        # Sequential scan
        h = torch.zeros(B, D, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            dt_t = dt[:, t].unsqueeze(-1)  # (B, D, 1)
            dA = torch.exp(A * dt_t)  # (B, D, d_state)
            dB = dt_t * B_t[:, t].unsqueeze(1)  # (B, D, d_state)
            x_t = x[:, t].unsqueeze(-1)  # (B, D, 1)
            h = dA * h + dB * x_t  # (B, D, d_state)
            y_t = (h * C_t[:, t].unsqueeze(1)).sum(-1) + self.D * x[:, t]  # (B, D)

            if pad_mask is not None:
                mask_t = (~pad_mask[:, t]).unsqueeze(-1).float()  # (B, 1)
                y_t = y_t * mask_t
                h = h * mask_t.unsqueeze(-1)

            ys.append(y_t)

        return torch.stack(ys, dim=1)  # (B, T, D)


class MambaBlock(nn.Module):
    """Single Mamba block: norm → in_proj → (conv + SSM ⊗ gate) → out_proj + residual."""

    def __init__(self, hidden_dim, d_state=16, expand=2, d_conv=4, dropout=0.0):
        super().__init__()
        d_inner = hidden_dim * expand

        self.norm = nn.LayerNorm(hidden_dim)
        self.in_proj = nn.Linear(hidden_dim, d_inner * 2, bias=False)
        # Causal depthwise conv1d
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner,
        )
        self.ssm = SelectiveSSM(d_inner, d_state=d_state)
        self.out_proj = nn.Linear(d_inner, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):
        """
        Args:
            x: (B, T, H)
            pad_mask: (B, T) bool — True for padding
        Returns:
            (B, T, H)
        """
        residual = x
        T = x.shape[1]
        x = self.norm(x)

        xz = self.in_proj(x)  # (B, T, 2*d_inner)
        x_branch, z = xz.chunk(2, dim=-1)  # each (B, T, d_inner)

        # Causal conv1d (channels-first)
        x_branch = x_branch.transpose(1, 2)  # (B, d_inner, T)
        x_branch = self.conv1d(x_branch)[..., :T]  # causal: trim future
        x_branch = x_branch.transpose(1, 2)  # (B, T, d_inner)
        x_branch = F.silu(x_branch)

        # SSM
        y = self.ssm(x_branch, pad_mask=pad_mask)

        # Gate
        y = y * F.silu(z)

        out = self.dropout(self.out_proj(y))
        return residual + out


class Mamba4Rec(nn.Module):
    """Mamba4Rec: Mamba-based sequential recommendation model.

    Same interface as SASRec: forward(seqs, pos, neg) and predict(seqs, items).
    """

    def __init__(
        self,
        item_num,
        hidden_units=64,
        max_len=200,
        num_blocks=2,
        d_state=16,
        expand=2,
        d_conv=4,
        dropout_rate=0.2,
        **kwargs,
    ):
        super().__init__()
        self.item_num = item_num
        self.max_len = max_len

        self.item_emb = nn.Embedding(item_num + 1, hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout_rate)

        self.blocks = nn.ModuleList(
            [
                MambaBlock(hidden_units, d_state=d_state, expand=expand,
                           d_conv=d_conv, dropout=dropout_rate)
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
        x = self.emb_dropout(x)

        pad_mask = seqs == 0  # (B, T)

        # Zero out padding embeddings
        x = x * (~pad_mask).unsqueeze(-1).float()

        for block in self.blocks:
            x = block(x, pad_mask=pad_mask)
            # Re-mask after each block to prevent recurrence leakage
            x = x * (~pad_mask).unsqueeze(-1).float()

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
