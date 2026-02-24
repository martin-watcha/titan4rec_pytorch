import torch
import torch.nn as nn
import torch.nn.functional as F

from . import RMSNorm
from .long_term_memory import NeuralLongTermMemory
from .attention import CausalAttention


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout_rate=0.2):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout2(self.linear2(self.dropout1(F.silu(self.linear1(x)))))


class PersistentMemory(nn.Module):
    def __init__(self, num_persistent, d_model):
        super().__init__()
        self.persistent_tokens = nn.Parameter(
            torch.randn(num_persistent, d_model) * 0.02
        )

    def forward(self, batch_size):
        """Returns: (B, N_p, d_model)"""
        return self.persistent_tokens.unsqueeze(0).expand(batch_size, -1, -1)


class MACBlock(nn.Module):
    """Memory as a Context block (Titans Section 4.1).

    Each block owns an independent LTM, Attention, FFN, and PersistentMemory.
    """

    def __init__(
        self,
        d_model,
        num_heads,
        num_persistent,
        memory_depth,
        memory_heads,
        expansion_factor,
        max_lr,
        dropout_rate,
    ):
        super().__init__()
        self.ltm = NeuralLongTermMemory(
            d_model,
            num_layers=memory_depth,
            expansion_factor=expansion_factor,
            num_heads=memory_heads,
            max_lr=max_lr,
        )
        self.persistent_memory = PersistentMemory(num_persistent, d_model)
        self.attention = CausalAttention(d_model, num_heads, dropout_rate)
        self.ffn = PositionwiseFFN(d_model, dropout_rate=dropout_rate)

        self.pre_attn_norm = RMSNorm(d_model)
        self.pre_ffn_norm = RMSNorm(d_model)
        self.gate_norm_y = RMSNorm(d_model)
        self.gate_norm_z = RMSNorm(d_model)
        # Gate bias: sigmoid(2.0) ≈ 0.88, starting near pass-through.
        # Without this, sigmoid ≈ 0.5 halves the signal per block.
        self.gate_bias = nn.Parameter(torch.full((1, 1, d_model), 2.0))

        self.num_persistent = num_persistent
        # Pre-allocated buffer: expanded to batch size at runtime via expand()
        self.register_buffer(
            "_persistent_valid",
            torch.ones(1, num_persistent, dtype=torch.bool),
        )

    def forward(self, segment, memory_state, momentum_state, padding_mask=None):
        """
        Args:
            segment: (B, C, d_model)
            memory_state, momentum_state: dicts from LTM
            padding_mask: (B, C) bool, True=valid
        Returns:
            (output, updated_memory_state, updated_momentum_state)
        """
        B, C, D = segment.shape

        # 1. LTM Retrieval
        h_t = self.ltm.retrieve(segment, memory_state)

        # 2. Context Construction: [persistent | LTM retrieved | segment]
        persistent = self.persistent_memory(B)
        context = torch.cat([persistent, h_t, segment], dim=1)
        prefix_len = self.num_persistent + C

        # 3. Full padding mask for context
        # Bug fix: LTM retrieved tokens from padding queries are garbage —
        # mark them invalid so valid segment tokens don't attend to them.
        # Only persistent tokens are unconditionally valid.
        if padding_mask is not None:
            persistent_valid = self._persistent_valid.expand(B, -1)
            full_padding_mask = torch.cat(
                [persistent_valid, padding_mask, padding_mask], dim=1
            )
        else:
            full_padding_mask = None

        # 4. Pre-norm + Attention + Residual
        context = context + self.attention(
            self.pre_attn_norm(context),
            prefix_len=prefix_len,
            padding_mask=full_padding_mask,
        )

        # 5. Pre-norm + FFN + Residual
        context = context + self.ffn(self.pre_ffn_norm(context))

        # 6. Extract segment portion
        y_t = context[:, prefix_len:, :]

        # 7. LTM Update (pass padding_mask to avoid updating on padding tokens)
        memory_state, momentum_state = self.ltm.update(
            y_t, memory_state, momentum_state, padding_mask=padding_mask
        )

        # 8. Sigmoid-gated Output (gate_bias starts at 2.0 → sigmoid≈0.88)
        z_t = self.ltm.retrieve(y_t, memory_state)
        o_t = self.gate_norm_y(y_t) * torch.sigmoid(
            self.gate_norm_z(z_t) + self.gate_bias
        )

        # Inter-block residual: preserves gradient highway from input to output
        # (like standard Transformer: output = input + sublayer(input)).
        # Without this, the multiplicative gate attenuates signal ~12% per block.
        return segment + o_t, memory_state, momentum_state
