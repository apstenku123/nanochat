import torch
import torch.nn as nn

from nanochat.common import print0


class ManifoldBranchMixer(nn.Module):
    """
    Constrained branch mixer with Sinkhorn-normalized weights.

    Input: list[(B, T, C)]
    Output: (B, T, C)
    """

    def __init__(
        self,
        n_embd,
        sinkhorn_iters=5,
        temperature=1.0,
        epsilon=1e-6,
        blend_alpha=1.0,
        max_branches=0,
    ):
        super().__init__()
        hidden = max(8, min(256, n_embd // 4))
        self.score_proj = nn.Linear(n_embd, hidden, bias=False)
        self.score_out = nn.Linear(hidden, 1, bias=False)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.temperature = float(temperature)
        self.epsilon = float(epsilon)
        self.blend_alpha = float(blend_alpha)
        self.max_branches = int(max_branches)
        self._warned_fallback = False

    def _sinkhorn(self, raw_matrix):
        eps = self.epsilon
        m = raw_matrix - raw_matrix.amax(dim=(-2, -1), keepdim=True)
        transport = torch.exp(m).clamp_min(eps)
        for _ in range(max(0, self.sinkhorn_iters)):
            transport = transport / (transport.sum(dim=-1, keepdim=True) + eps)
            transport = transport / (transport.sum(dim=-2, keepdim=True) + eps)
        transport = transport / (transport.sum(dim=-1, keepdim=True) + eps)
        return transport

    def forward(self, branches):
        if not branches:
            raise ValueError("ManifoldBranchMixer requires at least one branch")
        if len(branches) == 1:
            return branches[0]
        if self.max_branches > 0 and len(branches) > self.max_branches:
            raise ValueError(f"Too many branches: got {len(branches)}, max_branches={self.max_branches}")

        ref_shape = branches[0].shape
        for branch in branches[1:]:
            if branch.shape != ref_shape:
                raise ValueError(f"Branch shape mismatch: expected {ref_shape}, got {branch.shape}")

        # (B, T, N, C)
        stacked = torch.stack(branches, dim=2)
        pooled = stacked.mean(dim=1)  # (B, N, C)
        logits = self.score_out(torch.tanh(self.score_proj(pooled))).squeeze(-1)  # (B, N)
        temperature = max(self.temperature, self.epsilon)
        raw_matrix = (logits.unsqueeze(-1) + logits.unsqueeze(-2)) / temperature
        transport = self._sinkhorn(raw_matrix.float())

        # Compute weights from transport matrix, using torch.where for NaN/Inf
        # fallback instead of data-dependent Python `if` (which forces host-device
        # sync on XLA/TPU, causing 37x slowdown).
        weights = transport.mean(dim=1)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + self.epsilon)
        n_branches = stacked.size(2)
        uniform = torch.full_like(weights, 1.0 / n_branches)
        is_valid = torch.isfinite(weights).all(dim=-1, keepdim=True)
        weights = torch.where(is_valid, weights, uniform).to(dtype=stacked.dtype)

        mixed = (stacked * weights[:, None, :, None]).sum(dim=2)
        alpha = min(max(self.blend_alpha, 0.0), 1.0)
        if alpha == 1.0:
            return mixed
        return branches[0] + alpha * (mixed - branches[0])
