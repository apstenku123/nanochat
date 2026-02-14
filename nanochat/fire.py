"""
Plasticity Toolkit: FIRE + DASH + ReDo for nanochat.

FIRE (Frobenius-Isometry REinitialization): Inter-phase weight orthogonalization
via Newton-Schulz iteration. Applied ONCE between training phases.

DASH (Direction-Aware SHrinking): Per-neuron weight shrinking based on
cosine similarity with gradients. Applied periodically DURING training.

ReDo (Recycling Dormant Neurons): Detection and reinitialization of
dead neurons in MLP layers. Applied periodically DURING training.

Reference: Han et al., "FIRE: Frobenius-Isometry Reinitialization for
Balancing the Stability-Plasticity Tradeoff", arXiv:2602.08040v1, Feb 2026.
"""
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Set, Dict, Tuple, Callable, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# FIRE: Newton-Schulz Orthogonalization
# =============================================================================

@torch.no_grad()
def newton_schulz(W: torch.Tensor, iters: int = 15) -> torch.Tensor:
    """Approximate polar decomposition via Newton-Schulz iteration.

    Projects W onto the nearest orthogonal matrix (minimizes ||W - W_tilde||_F
    subject to W_tilde^T W_tilde = I). Preserves the original Frobenius norm
    to avoid breaking custom initialization scales.

    Args:
        W: 2D weight matrix (d_out, d_in)
        iters: Number of Newton-Schulz iterations (15 for reliable convergence;
               cubic iteration needs ~15 iters for condition numbers up to 500)

    Returns:
        Orthogonalized matrix with same Frobenius norm as input.
    """
    assert W.dim() == 2, f"newton_schulz requires 2D tensor, got {W.dim()}D"

    # Work in float32 for numerical stability (bf16 doesn't support linalg ops)
    orig_dtype = W.dtype
    W_f32 = W.float()

    # Normalize by spectral norm so all singular values are in (0, 1].
    # This ensures the cubic NS iteration converges (basin is (0, sqrt(3))).
    spectral_norm = torch.linalg.matrix_norm(W_f32, ord=2).clamp(min=1e-8)
    X = W_f32 / spectral_norm

    # Handle wide matrices (d_out < d_in): transpose so rows >= cols
    # Newton-Schulz converges only when the matrix has full column rank
    is_wide = W.shape[0] < W.shape[1]
    if is_wide:
        X = X.T

    # Standard Newton-Schulz coefficients (not Muon's tuned quintic)
    # These converge to true orthogonal matrix, not approximate
    a, b = 1.5, -0.5
    for _ in range(iters):
        A = X.T @ X
        X = a * X + b * (X @ A)

    if is_wide:
        X = X.T

    # Return the orthogonal matrix (all singular values ≈ 1.0).
    # Scaling for signal variance preservation is handled by apply_fire,
    # not here, because the correct scale depends on the model architecture.
    return X.to(orig_dtype)


@torch.no_grad()
def apply_fire(
    model: nn.Module,
    target_keywords: Optional[List[str]] = None,
    skip_keywords: Optional[List[str]] = None,
    iters: int = 15,
) -> Set[nn.Parameter]:
    """Apply FIRE orthogonalization to model weights.

    Iterates over all named parameters. Orthogonalizes 2D weight matrices
    that match target_keywords (if specified) and don't match skip_keywords.
    Skips all non-2D tensors (Mamba A_log, dt_bias, D, conv1d.weight).

    Args:
        model: The model to orthogonalize.
        target_keywords: If set, only params whose names contain any keyword.
                         If None, all 2D params (except skipped) are targeted.
        skip_keywords: Param names containing these are always skipped.
                       Default: ['wte', 'lm_head'] (embeddings).
        iters: Newton-Schulz iterations.

    Returns:
        Set of modified nn.Parameter objects (for selective optimizer reset).
    """
    if skip_keywords is None:
        skip_keywords = ['wte', 'lm_head']

    modified = set()
    for name, param in model.named_parameters():
        # Only 2D matrices (protects Mamba 1D/3D params)
        if param.dim() != 2:
            continue
        # Skip embeddings / output head
        if any(sk in name for sk in skip_keywords):
            continue
        # Target filter
        if target_keywords is not None:
            if not any(kw in name for kw in target_keywords):
                continue

        param.data.copy_(newton_schulz(param.data, iters))
        modified.add(param)

    logger.info(f"[FIRE] Orthogonalized {len(modified)} 2D matrices, {iters} NS iterations")
    return modified


# =============================================================================
# Optimizer State Reset (selective, not global)
# =============================================================================

@torch.no_grad()
def reset_optimizer_states_for_fired_params(
    optimizers: List[torch.optim.Optimizer],
    modified_params: Set[nn.Parameter],
) -> int:
    """Selectively reset optimizer states for params that were FIRE'd.

    Removes exp_avg, exp_avg_sq (AdamW) and momentum_buffer (Muon/SGD)
    ONLY for the parameters that were orthogonalized. Embeddings, lm_head,
    and Mamba 1D params keep their accumulated momentum.

    Returns:
        Number of optimizer states reset.
    """
    reset_count = 0
    for opt in optimizers:
        keys_to_pop = []
        for p in modified_params:
            if p in opt.state:
                keys_to_pop.append(p)
        for p in keys_to_pop:
            opt.state.pop(p)
            reset_count += 1
    logger.info(f"[FIRE] Selectively reset {reset_count} optimizer states")
    return reset_count


# =============================================================================
# DASH: Direction-Aware Shrinking (per-neuron)
# =============================================================================

@torch.no_grad()
def dash_step(
    W_or_model,
    grad_or_alpha: float = 0.05,
    alpha: float = 0.05,
    shrink_rate: float = 0.01,
    muon_params: Optional[Set[nn.Parameter]] = None,
):
    """Apply per-neuron DASH shrinking.

    Can be called in two ways:
    1. Standalone on tensors: dash_step(W, grad, alpha=0.05, shrink_rate=0.01) -> Tensor
    2. On full model: dash_step(model, alpha=0.05, shrink_rate=0.01, muon_params=...)

    For each 2D weight, computes per-row (per-neuron) cosine similarity
    between weight and gradient. Neurons with cos_sim > alpha are shrunk.
    """
    # Dispatch: tensor mode vs model mode
    if isinstance(W_or_model, torch.Tensor):
        W = W_or_model
        grad = grad_or_alpha
        if not isinstance(grad, torch.Tensor):
            raise ValueError("In tensor mode, second arg must be gradient tensor")
        if W.dim() != 2:
            raise ValueError(f"dash_step requires 2D tensors, got {W.dim()}D")

        cos_sim = F.cosine_similarity(W, grad, dim=1)
        penalty = torch.clamp(cos_sim - alpha, min=0.0).unsqueeze(1)
        shrink_factor = torch.clamp(1.0 - shrink_rate * penalty, min=0.5, max=1.0)
        return W * shrink_factor

    # Model mode
    model = W_or_model
    alpha_val = grad_or_alpha if isinstance(grad_or_alpha, (int, float)) else alpha
    if muon_params is None:
        muon_params = set()

    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue
        if param.dim() != 2:
            continue
        if param in muon_params:
            continue

        cos_sim = F.cosine_similarity(param.data, param.grad.data, dim=1)
        penalty = torch.clamp(cos_sim - alpha_val, min=0.0).unsqueeze(1)
        shrink_factor = torch.clamp(1.0 - shrink_rate * penalty, min=0.5, max=1.0)
        param.data.mul_(shrink_factor)


# =============================================================================
# ReDo: Dormant Neuron Detection and Recycling
# =============================================================================

class ReDoDiagnostics:
    """Monitors neuron activity in MLP layers via forward hooks.

    Attaches hooks to c_fc (or c_gate for SwiGLU) modules to track
    post-activation mean absolute values per neuron. Uses EMA for
    stable estimation across batches.

    The act_fn callable is applied inside the hook to simulate the
    functional activation (since relu^2 is not an nn.Module).
    """

    def __init__(self):
        self.stats: Dict[str, torch.Tensor] = {}
        self._hooks = []

    def attach(
        self,
        model: nn.Module,
        target_names: Optional[List[str]] = None,
        act_fn: Optional[Callable] = None,
    ) -> int:
        """Attach hooks to target modules by name matching.

        Args:
            model: The model.
            target_names: List of substrings to match module names.
                          Default: ['c_fc'] for standard MLP.
            act_fn: Activation function applied after the hooked module.
                    Default: lambda x: F.relu(x).square() (Primer relu^2).

        Returns:
            Number of hooks attached.
        """
        self.detach()

        if target_names is None:
            target_names = ['c_fc']
        if act_fn is None:
            act_fn = lambda x: F.relu(x).square()

        count = 0
        for name, module in model.named_modules():
            if any(name.endswith(t) for t in target_names):
                hook = module.register_forward_hook(self._make_hook(name, act_fn))
                self._hooks.append(hook)
                count += 1

        return count

    def attach_hooks(
        self,
        model: Optional[nn.Module],
        target_modules: Optional[List[nn.Module]] = None,
        act_fn: Optional[Callable] = None,
    ) -> int:
        """Attach hooks directly to specific module objects.

        Args:
            model: Ignored (for API compat). Can be None.
            target_modules: List of nn.Module instances to hook.
            act_fn: Activation function applied after the hooked module.

        Returns:
            Number of hooks attached.
        """
        self.detach()

        if target_modules is None:
            return 0
        if act_fn is None:
            act_fn = lambda x: F.relu(x).square()

        count = 0
        for i, module in enumerate(target_modules):
            name = f"module_{i}"
            hook = module.register_forward_hook(self._make_hook(name, act_fn))
            self._hooks.append(hook)
            count += 1

        return count

    def detach(self):
        """Remove all hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def remove_hooks(self):
        """Alias for detach()."""
        self.detach()

    def get_stats(self) -> Dict[str, torch.Tensor]:
        """Return collected stats dict."""
        return self.stats

    def _make_hook(self, name: str, act_fn: Callable):
        def hook(module, inp, out):
            with torch.no_grad():
                activated = act_fn(out)
                # Mean absolute activation per neuron: [hidden_dim]
                mean_abs = activated.abs().mean(dim=tuple(range(activated.dim() - 1)))

                if name not in self.stats:
                    self.stats[name] = mean_abs.clone()
                else:
                    # Exponential moving average (alpha=0.1)
                    self.stats[name] = 0.9 * self.stats[name] + 0.1 * mean_abs
        return hook

    def get_dormant_ratio(self, tau: float = 0.025) -> Dict[str, float]:
        """Get fraction of dormant neurons per monitored layer.

        A neuron is dormant if its normalized activity score < tau.
        """
        ratios = {}
        for name, stats in self.stats.items():
            layer_mean = stats.mean().clamp(min=1e-8)
            scores = stats / layer_mean
            n_dormant = (scores < tau).sum().item()
            ratios[name] = n_dormant / stats.numel()
        return ratios


@torch.no_grad()
def recycle_dormant_neurons(
    fc_in_or_layer_map,
    fc_out_or_stats = None,
    stats_or_tau = None,
    tau: float = 0.025,
) -> int:
    """Reinitialize dormant neurons in MLP layers.

    Uses torch.where (not boolean indexing) for XLA compatibility.

    Can be called in two ways:
    1. Simple: recycle_dormant_neurons(fc_in, fc_out, stats, tau=0.025) -> int
    2. Dict:   recycle_dormant_neurons(layer_map, redo_stats, tau=0.025) -> int

    Returns:
        Total number of recycled neurons.
    """
    # Dispatch: simple mode vs dict mode
    if isinstance(fc_in_or_layer_map, nn.Module):
        # Simple mode: (fc_in, fc_out, stats_tensor, tau)
        fc_in = fc_in_or_layer_map
        fc_out = fc_out_or_stats
        stats = stats_or_tau
        if stats is None or not isinstance(stats, torch.Tensor):
            raise ValueError("Simple mode: recycle_dormant_neurons(fc_in, fc_out, stats_tensor, tau)")
        return _recycle_single(fc_in, fc_out, stats, tau)

    # Dict mode: (layer_map, redo_stats_dict, tau)
    layer_map = fc_in_or_layer_map
    redo_stats = fc_out_or_stats
    if stats_or_tau is not None and isinstance(stats_or_tau, (int, float)):
        tau = stats_or_tau

    total = 0
    for name, modules in layer_map.items():
        if name not in redo_stats:
            continue

        if isinstance(modules, tuple) and len(modules) == 2:
            in_modules, out_module = modules
        else:
            raise ValueError(f"layer_map['{name}'] must be (in_module(s), out_module)")

        if not isinstance(in_modules, (list, tuple)):
            in_modules = [in_modules]

        stats = redo_stats[name]
        total += _recycle_core(in_modules, out_module, stats, tau, redo_stats, name)

    if total > 0:
        logger.info(f"[ReDo] Recycled {total} dormant MLP neurons")
    return total


def _recycle_single(fc_in: nn.Module, fc_out: nn.Module, stats: torch.Tensor, tau: float) -> int:
    """Recycle dormant neurons for a single fc_in -> fc_out pair."""
    return _recycle_core([fc_in], fc_out, stats, tau)


def _recycle_core(
    in_modules: List[nn.Module],
    out_module: nn.Module,
    stats: torch.Tensor,
    tau: float,
    redo_stats: Optional[Dict] = None,
    name: Optional[str] = None,
) -> int:
    """Core recycling logic using torch.where (XLA-safe)."""
    layer_mean = stats.mean().clamp(min=1e-8)
    is_dormant = (stats / layer_mean) < tau

    n_dormant = is_dormant.sum().item()
    if n_dormant == 0:
        return 0

    for in_mod in in_modules:
        std_in = in_mod.weight.std().item()
        new_w = torch.empty_like(in_mod.weight).normal_(0, max(std_in, 1e-4))
        mask = is_dormant.unsqueeze(1).expand_as(in_mod.weight)
        in_mod.weight.copy_(torch.where(mask, new_w, in_mod.weight))

        if hasattr(in_mod, 'bias') and in_mod.bias is not None:
            in_mod.bias.copy_(torch.where(is_dormant, torch.zeros_like(in_mod.bias), in_mod.bias))

    std_out = out_module.weight.std().item()
    new_w = torch.empty_like(out_module.weight).normal_(0, max(std_out * 0.1, 1e-5))
    mask = is_dormant.unsqueeze(0).expand_as(out_module.weight)
    out_module.weight.copy_(torch.where(mask, new_w, out_module.weight))

    if redo_stats is not None and name is not None:
        redo_stats[name] = torch.where(is_dormant, layer_mean, stats)

    return n_dormant
