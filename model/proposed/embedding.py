import math

import torch
import torch.nn as nn

from . import RMSNorm


class Titan4RecEmbedding(nn.Module):
    def __init__(self, num_items, d_model, max_pos_len, dropout_rate=0.2):
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_pos_len, d_model)
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.d_model = d_model

    def forward(self, input_seq):
        """Item embedding only. Positional encoding added separately in log2feats.
        Args: input_seq (B, T) LongTensor
        Returns: (B, T, d_model)
        """
        x = self.item_emb(input_seq) * math.sqrt(self.d_model)
        x = self.norm(x)
        x = self.dropout(x)
        return x

    def get_position_encoding(self, length, device=None):
        """Returns: (length, d_model)"""
        positions = torch.arange(length, device=device)
        return self.pos_emb(positions)
