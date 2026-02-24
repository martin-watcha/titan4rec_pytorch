import torch
import torch.nn as nn
import torch.nn.functional as F

from . import RMSNorm


def _gelu_grad(x):
    """Derivative of GELU activation: GELU'(x) = Φ(x) + x·φ(x)"""
    cdf = 0.5 * (1.0 + torch.erf(x * 0.7071067811865476))  # Φ(x)
    pdf = torch.exp(-0.5 * x * x) * 0.3989422804014327      # φ(x)
    return cdf + x * pdf


class NeuralLongTermMemory(nn.Module):
    """Neural Long-Term Memory using explicit bmm-based gradient computation.

    Replaces vmap(grad)+functional_call with pure batched matrix multiplications.
    memory_state: {f'W{i}': (B*H, d_in, d_out)} — per-sample weight tensors.
    Forward/backward through memory MLP are implemented explicitly with bmm,
    enabling full outer-loop gradient flow without vmap overhead.
    """

    def __init__(
        self, d_model, num_layers=2, expansion_factor=4, num_heads=2, max_lr=0.01
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_lr = max_lr
        self.num_layers = num_layers

        # Weight shapes: W_i maps d_in → d_out, stored as (d_in, d_out)
        hidden_dim = self.head_dim * expansion_factor
        dims = [self.head_dim] + [hidden_dim] * (num_layers - 1) + [self.head_dim]
        self._weight_shapes = list(zip(dims[:-1], dims[1:]))

        # Learnable initial memory weights (outer-loop parameters)
        self.memory_init_weights = nn.ParameterList([
            nn.Parameter(torch.empty(d_in, d_out))
            for d_in, d_out in self._weight_shapes
        ])
        for w in self.memory_init_weights:
            nn.init.xavier_uniform_(w)

        # Outer loop projections
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)

        # Norms
        self.k_norm = RMSNorm(self.head_dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.store_norm = RMSNorm(d_model)
        self.retrieve_norm = RMSNorm(d_model)

        # Data-dependent gates: per-token, per-head scalar
        self.to_alpha = nn.Linear(d_model, num_heads)  # decay
        self.to_theta = nn.Linear(d_model, num_heads)  # learning rate
        self.to_eta = nn.Linear(d_model, num_heads)    # momentum

        # Gate bias init: prevent aggressive decay/lr at start
        nn.init.zeros_(self.to_alpha.weight)
        nn.init.constant_(self.to_alpha.bias, -5.0)  # sigmoid(-5)~0.007 → 99.3% retention
        nn.init.zeros_(self.to_theta.weight)
        nn.init.constant_(self.to_theta.bias, -3.0)  # sigmoid(-3)~0.047 * max_lr
        nn.init.zeros_(self.to_eta.weight)
        nn.init.constant_(self.to_eta.bias, 0.0)     # sigmoid(0)=0.5 momentum

    def _split_heads(self, x):
        """(B, S, d_model) -> (B*H, S, head_dim)"""
        B, S, _ = x.shape
        x = x.view(B, S, self.num_heads, self.head_dim)
        x = x.permute(0, 2, 1, 3).reshape(B * self.num_heads, S, self.head_dim)
        return x

    def _merge_heads(self, x, batch_size):
        """(B*H, S, head_dim) -> (B, S, d_model)"""
        x = x.view(batch_size, self.num_heads, -1, self.head_dim)
        x = x.permute(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)
        return x

    def init_memory_state(self, batch_size):
        """Returns: (memory_state, momentum_state)
        memory_state: {f'W{i}': (B*H, d_in, d_out)}
        """
        bh = batch_size * self.num_heads
        ms, mom = {}, {}
        for i, w in enumerate(self.memory_init_weights):
            ms[f'W{i}'] = w.unsqueeze(0).expand(bh, -1, -1).clone()
            mom[f'W{i}'] = torch.zeros_like(ms[f'W{i}'])
        return ms, mom

    @staticmethod
    def _rmsnorm(h):
        """Parameter-free RMSNorm: h / rms(h). Returns (h_norm, rms)."""
        rms = (h.square().mean(dim=-1, keepdim=True) + 1e-8).sqrt()
        return h / rms, rms

    @staticmethod
    def _rmsnorm_backward(d_out, h_norm, rms):
        """Backward through parameter-free RMSNorm.
        d_out: gradient w.r.t. normalized output (BH, S, D)
        Returns: gradient w.r.t. h (input before normalization)
        """
        D = d_out.shape[-1]
        dy_dot_y = (d_out * h_norm).sum(dim=-1, keepdim=True)  # (BH, S, 1)
        return (d_out - h_norm * dy_dot_y / D) / rms

    def _memory_forward(self, x, memory_state):
        """Forward pass through memory MLP using bmm.

        Args:
            x: (B*H, S, head_dim)
            memory_state: {f'W{i}': (B*H, d_in, d_out)}
        Returns:
            (B*H, S, head_dim) — RMSNorm(MLP(x)) + x
        """
        h = x
        for i in range(self.num_layers):
            W = memory_state[f'W{i}']       # (BH, d_in, d_out)
            h_new = torch.bmm(h, W)         # (BH, S, d_out)
            if i < self.num_layers - 1:
                h_new = F.gelu(h_new)
            h = h_new
        # Replace NaN/Inf with 0 (GPU-native, no CPU sync).
        # When h==0, _rmsnorm returns 0 → output = 0 + x = x (same as original fallback).
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        h_norm, _ = self._rmsnorm(h)        # normalize before residual
        return h_norm + x

    def _memory_grad(self, keys, values, theta, memory_state):
        """Compute gradient of weighted MSE loss w.r.t. memory weights.

        Implements explicit backpropagation through the memory MLP using bmm.
        memory_state weights are detached to avoid second-order graph growth.
        Outer-loop gradients (w.r.t. W_K, W_V) flow via keys/values, which
        remain attached. The (1-alpha)*w path in update() carries the gradient
        for memory_init_weights separately.

        Loss: sum_t theta_t * ||RMSNorm(M(k_t)) + k_t - v_t||^2

        Args:
            keys:   (B*H, C, head_dim)
            values: (B*H, C, head_dim)
            theta:  (B*H, C)   — per-token learning rate weights
        Returns:
            dict {f'W{i}': (B*H, d_in, d_out)} — gradients w.r.t. memory weights
        """
        # Detach memory weights: prevents second-order graph through memory chain.
        # W_K/W_V gradients still flow via keys/values (kept attached below).
        det_ms = {k: v.detach() for k, v in memory_state.items()}

        # --- Forward pass, storing activations and pre-activations ---
        activations = [keys]   # activations[0] = input
        pre_acts = []
        h = keys
        for i in range(self.num_layers):
            W = det_ms[f'W{i}']
            pre = torch.bmm(h, W)
            pre_acts.append(pre)
            if i < self.num_layers - 1:
                h = F.gelu(pre)
            else:
                h = pre
            activations.append(h)

        # RMSNorm on last layer output (same as _memory_forward)
        h_last = activations[-1]
        h_norm, rms = self._rmsnorm(h_last)
        pred = h_norm + keys                # residual: (BH, C, head_dim)

        # --- Gradient of loss w.r.t. pred ---
        # d/d(pred) sum_t theta_t * ||pred_t - v_t||^2
        d_pred = 2.0 * theta.unsqueeze(-1) * (pred - values)  # (BH, C, head_dim)

        # --- Backward through RMSNorm (pred = h_norm + keys, d_keys ignored here) ---
        d_h = self._rmsnorm_backward(d_pred, h_norm, rms)

        # --- Backward pass through MLP layers (W's detached) ---
        grads = {}
        for i in range(self.num_layers - 1, -1, -1):
            h_in = activations[i]                                         # (BH, C, d_in)
            grads[f'W{i}'] = torch.bmm(h_in.transpose(-2, -1), d_h)     # (BH, d_in, d_out)
            if i > 0:
                W = det_ms[f'W{i}']                                       # detached
                d_pre = torch.bmm(d_h, W.transpose(-2, -1))              # (BH, C, d_in_i)
                d_h = d_pre * _gelu_grad(pre_acts[i - 1])                # through GELU

        return grads

    @staticmethod
    def _clip_grad_norm(grads, max_norm=10.0):
        """Per-sample gradient norm clipping across all memory weight matrices.

        Computes the total gradient norm per sample (across all W_i),
        then scales down gradients for samples exceeding max_norm.
        The clip coefficient is computed under no_grad to avoid creating
        complex cross-dependencies in the autograd graph.
        """
        with torch.no_grad():
            sq_norms = None
            for g in grads.values():
                sn = g.reshape(g.shape[0], -1).pow(2).sum(dim=1)  # (BH,)
                sq_norms = sn if sq_norms is None else sq_norms + sn
            total_norm = sq_norms.sqrt()  # (BH,)
            clip_coef = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
            clip_coef = clip_coef.view(-1, 1, 1)
        return {k: g * clip_coef for k, g in grads.items()}

    def retrieve(self, query, memory_state):
        """Forward pass without weight update.
        Args: query (B, S, d_model), memory_state dict
        Returns: (B, S, d_model)
        """
        B = query.shape[0]
        q = self.W_Q(self.retrieve_norm(query))
        q = self._split_heads(q)
        q = self.q_norm(q)
        out = self._memory_forward(q, memory_state)
        return self._merge_heads(out, B)

    def update(self, x, memory_state, momentum_state, padding_mask=None):
        """Chunk-level memory update using explicit bmm gradient computation.

        Args:
            x: (B, S, d_model)
            memory_state: dict {f'W{i}': (B*H, d_in, d_out)}
            momentum_state: dict, same shape as memory_state
            padding_mask: (B, S) bool, True=valid. If None, all positions valid.
        """
        B = x.shape[0]
        x_n = self.store_norm(x)

        keys = self.W_K(x_n)
        keys = self._split_heads(keys)  # (B*H, C, head_dim)
        keys = self.k_norm(keys)

        values = self.W_V(x_n)
        values = self._split_heads(values)  # (B*H, C, head_dim)

        # Gates
        alpha = torch.sigmoid(self.to_alpha(x_n))       # (B, C, H)
        theta = torch.sigmoid(self.to_theta(x_n)) * self.max_lr
        eta   = torch.sigmoid(self.to_eta(x_n))

        if padding_mask is not None:
            mask = padding_mask.unsqueeze(-1).float()
            alpha = alpha * mask
            theta = theta * mask
            eta   = eta   * mask

        # (B, C, H) -> (B*H, C)
        alpha = alpha.permute(0, 2, 1).reshape(B * self.num_heads, -1)
        theta = theta.permute(0, 2, 1).reshape(B * self.num_heads, -1)
        eta   = eta.permute(0, 2, 1).reshape(B * self.num_heads, -1)

        # Explicit bmm-based gradient (replaces vmap+grad)
        grads = self._memory_grad(keys, values, theta, memory_state)

        # Per-sample gradient norm clipping to prevent inner-loop gradient explosion.
        # Without this, divergent memory weights cause Inf gradients that trigger the
        # isfinite fallback (which uses w.detach() and silently breaks gradient flow).
        grads = self._clip_grad_norm(grads, max_norm=10.0)

        # Chunk-averaged gates (valid positions only)
        if padding_mask is not None:
            valid_counts = padding_mask.float().sum(dim=1)  # (B,)
            valid_counts = (
                valid_counts.unsqueeze(1)
                .expand(B, self.num_heads)
                .reshape(B * self.num_heads)
                .clamp(min=1.0)
            )
            alpha_mean = alpha.sum(dim=1) / valid_counts
            eta_mean   = eta.sum(dim=1)   / valid_counts
        else:
            alpha_mean = alpha.mean(dim=1)
            eta_mean   = eta.mean(dim=1)

        new_ms, new_mom = {}, {}
        for name in memory_state:
            g = grads[name]
            m = momentum_state[name]
            w = memory_state[name]
            shape = [alpha_mean.shape[0]] + [1] * (g.ndim - 1)
            a = alpha_mean.view(shape)
            e = eta_mean.view(shape)
            new_m = e * m - g            # momentum update
            new_w = (1 - a) * w + new_m  # decay + momentum
            # Guard against NaN/Inf from inner-loop gradient explosion.
            # Revert weight to previous; reset momentum to zero (safest fallback).
            new_w = torch.where(torch.isfinite(new_w), new_w, w.detach())
            new_m = torch.where(torch.isfinite(new_m), new_m, 0.0)
            new_ms[name]  = new_w
            new_mom[name] = new_m

        return new_ms, new_mom
