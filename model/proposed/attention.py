import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate=0.2):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_rate = dropout_rate

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self._cached_mask = None
        self._cached_mask_key = None

    def forward(self, x, prefix_len=0, padding_mask=None):
        """
        Args:
            x: (B, T, d_model) - [persistent | LTM retrieved | segment]
            prefix_len: int - N_p + C (fully visible prefix length)
            padding_mask: (B, T) bool - True = valid, False = padding
        Returns: (B, T, d_model)
        """
        B, T, D = x.shape

        Q = self.W_Q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Build combined bool mask: (1,1,T,T) mac + (B,1,1,T) key padding
        # F.scaled_dot_product_attention: True = attend, False = ignore
        attn_mask = self._get_mac_mask(T, prefix_len, x.device)   # (T, T)
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)            # (1, 1, T, T)
        if padding_mask is not None:
            # mask keys that are padding; prefix_valid is always True (no NaN risk)
            attn_mask = attn_mask & padding_mask.unsqueeze(1).unsqueeze(2)  # (B,1,T,T)

        dropout_p = self.dropout_rate if self.training else 0.0
        out = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=attn_mask, dropout_p=dropout_p
        )
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_O(out)

    def _get_mac_mask(self, total_len, prefix_len, device):
        """Return cached MAC mask, rebuilding only when shape/device changes."""
        key = (total_len, prefix_len, device)
        if self._cached_mask_key != key:
            self._cached_mask = self._build_mac_mask(total_len, prefix_len, device)
            self._cached_mask_key = key
        return self._cached_mask

    @staticmethod
    def _build_mac_mask(total_len, prefix_len, device):
        """MAC mask: prefix fully visible, segment causal.

        (T, T) bool tensor. mask[i, j] = True means position i can attend to j.
        """
        mask = torch.zeros(total_len, total_len, dtype=torch.bool, device=device)
        mask[:, :prefix_len] = True
        causal_full = torch.tril(
            torch.ones(total_len, total_len, dtype=torch.bool, device=device)
        )
        mask[:, prefix_len:] = causal_full[:, prefix_len:]
        return mask
