import torch
import torch.nn as nn

if hasattr(nn, "RMSNorm"):
    RMSNorm = nn.RMSNorm
else:

    class RMSNorm(nn.Module):
        def __init__(self, normalized_shape, eps=1e-8):
            super().__init__()
            if isinstance(normalized_shape, int):
                normalized_shape = (normalized_shape,)
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.eps = eps

        def forward(self, x):
            rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
            return (x * rms).to(x.dtype) * self.weight.to(x.dtype)
