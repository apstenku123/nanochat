"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration (local build for GB10/SM121)
- Apple Cut Cross Entropy (CCE) for memory-efficient loss computation
"""

import math
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch 2.9 checkpoint calls getattr(torch, device_type) which fails for 'xla'
# because torch_xla doesn't register as torch.xla. Fix: register it explicitly.
try:
    import torch_xla
    if not hasattr(torch, 'xla'):
        torch.xla = torch_xla
except ImportError:
    pass

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW

# =============================================================================
# Flash Attention with automatic FA3/SDPA fallback
# =============================================================================
# Uses local FA3 build for SM90 (Hopper) and SM121 (GB10/DGX Spark)
# Falls back to PyTorch SDPA for other GPUs (Ada SM89, Blackwell SM100, etc.)
from nanochat.flash_attention import flash_attn_func, flash_attn_with_kvcache

# =============================================================================
# Kernel backend for loss computation (CCE recommended for best performance)
# =============================================================================
from nanochat import kernels

# =============================================================================
# Precision plan (BF16 default; TE/NVFP4/FP8 stubs for compatibility)
# =============================================================================
from contextlib import nullcontext
from typing import Any, Optional

from nanochat.engram import EngramBranch
from nanochat.mhc import ManifoldBranchMixer
from nanochat.sparse_attention import DeepSeekSparseAttention

@dataclass
class PrecisionPlan:
    name: str
    recipe: Optional[Any]
    use_te: bool

def select_precision(target: str = "auto", disable_rht: bool = True, disable_sr: bool = True) -> PrecisionPlan:
    """Select precision plan. Without TE, always returns BF16."""
    return PrecisionPlan("PyTorch BF16", None, False)

def make_autocast_ctx(plan: PrecisionPlan, device_type: str = "cuda"):
    """Create autocast context factory."""
    if device_type == "cuda":
        return lambda: torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
    return nullcontext


@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (half context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "L"
    # Optional Engram branch
    engram_enabled: bool = False
    engram_layers: str = ""
    engram_ngram_orders: str = "2,3,4"
    engram_bottleneck_dim: int = 0
    engram_dropout: float = 0.0
    # Optional mHC branch mixer
    mhc_enabled: bool = False
    mhc_num_branches: int = 0
    mhc_sinkhorn_iters: int = 5
    mhc_temperature: float = 1.0
    mhc_epsilon: float = 1e-6
    mhc_blend_alpha: float = 1.0
    # Optional Multi-Token Prediction (DeepSeek-V3 style)
    mtp_enabled: bool = False
    mtp_lambda: float = 0.3       # MTP loss weight (DeepSeek uses 0.3 early, 0.1 later)
    # Optional DeepSeek Sparse Attention (DSA)
    dsa_enabled: bool = False
    dsa_start_layer: int = 7      # first layer to use sparse attention (0-indexed)
    dsa_top_k_ratio: float = 0.5  # fraction of tokens to select per query
    dsa_local_window: int = 128   # local window always included
    dsa_indexer_heads: int = 16   # number of lightweight indexer heads
    # Gradient checkpointing (saves memory by recomputing activations during backward)
    gradient_checkpointing: bool = False
    dsa_indexer_dim: int = 32     # dimension per indexer head
    # Reserved for optional auxiliary objectives
    aux_loss_weight: float = 0.0
    # Mamba-2 hybrid layers
    mamba_enabled: bool = False
    mamba_pattern: str = ""            # A=attention, M=mamba, tiled across layers. Empty=all attention.
    mamba_d_state: int = 64            # SSM state dimension
    mamba_d_conv: int = 4              # depthwise conv kernel width
    mamba_expand: int = 2              # expansion factor (d_inner = expand * n_embd)
    mamba_headdim: int = 128           # head dimension for SSD
    mamba_ngroups: int = 1             # number of groups for B/C (GQA-like)
    mamba_chunk_size: int = 256        # chunk size for SSD scan
    # Mamba-3 upgrades (Phase 2, all default off)
    mamba3_qknorm: bool = False        # QK-norm on B/C
    mamba3_bias: bool = False          # learnable B/C bias
    mamba3_complex_rope: bool = False  # complex RoPE on B/C
    # Mamba-3 Phase 3
    mamba3_trapezoidal: bool = False   # trapezoidal discretization
    # Attention window sizing (0 = use sequence_len)
    window_long: int = 0
    window_short: int = 0
    # RoPE
    rope_theta: float = 10000.0


def _parse_csv_ints(value: str) -> list[int]:
    if not value:
        return []
    values = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        values.append(int(raw))
    return values


def _parse_engram_layers(layer_spec: str, n_layer: int) -> set[int]:
    layers = set()
    for layer_idx in _parse_csv_ints(layer_spec):
        # Allow negative indices for convenience (-1 means final layer).
        if layer_idx < 0:
            layer_idx += n_layer
        assert 0 <= layer_idx < n_layer, f"Invalid engram layer index {layer_idx}, expected [0, {n_layer - 1}]"
        layers.add(layer_idx)
    return layers


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm

        # Attention with Flash Attention 3
        # FA3 handles GQA automatically when n_kv_heads < n_heads
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx, engram_layers):
        super().__init__()
        # Determine layer type: Mamba > DSA > CSA (mutually exclusive priority)
        m_pattern = getattr(config, 'mamba_pattern', '').upper()
        if getattr(config, 'mamba_enabled', False) and not m_pattern:
            m_pattern = "AAM"
        self.is_mamba = bool(m_pattern) and m_pattern[layer_idx % len(m_pattern)] == 'M'
        use_dsa = bool(getattr(config, 'dsa_enabled', False)) and layer_idx >= getattr(config, 'dsa_start_layer', 7)

        if self.is_mamba:
            from nanochat.mamba2 import Mamba2Layer
            self.attn = Mamba2Layer(config, layer_idx)
        elif use_dsa:
            self.attn = DeepSeekSparseAttention(
                config, layer_idx,
                dsa_top_k_ratio=config.dsa_top_k_ratio,
                dsa_local_window=config.dsa_local_window,
                dsa_indexer_heads=config.dsa_indexer_heads,
                dsa_indexer_dim=config.dsa_indexer_dim,
            )
        else:
            self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
        self.engram = None
        self.mhc = None
        self.use_engram = bool(config.engram_enabled) and layer_idx in engram_layers
        self.use_mhc = bool(config.mhc_enabled)
        if self.use_engram:
            self.engram = EngramBranch(
                n_embd=config.n_embd,
                ngram_orders=config.engram_ngram_orders,
                bottleneck_dim=config.engram_bottleneck_dim,
                dropout=config.engram_dropout,
            )
        if self.use_mhc:
            self.mhc = ManifoldBranchMixer(
                n_embd=config.n_embd,
                sinkhorn_iters=config.mhc_sinkhorn_iters,
                temperature=config.mhc_temperature,
                epsilon=config.mhc_epsilon,
                blend_alpha=config.mhc_blend_alpha,
                max_branches=config.mhc_num_branches,
            )

    def forward(self, x, cos_sin, window_size, kv_cache):
        x_attn = x + self.attn(norm(x), cos_sin, window_size, kv_cache)
        baseline_out = x_attn + self.mlp(norm(x_attn))

        engram_out = None
        if self.engram is not None:
            engram_out = baseline_out + self.engram(norm(x))

        if self.mhc is None:
            return engram_out if engram_out is not None else baseline_out

        branches = [baseline_out, x]
        if engram_out is not None:
            branches.append(engram_out)
        return self.mhc(branches)


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        self.engram_layers = _parse_engram_layers(config.engram_layers, config.n_layer) if config.engram_enabled else set()
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab size to multiple of 64 for tensor core efficiency (significant speedup for lm_head matmul).
        # Trade-off: During training, the softmax denominator includes padding logits (initialized to ~0),
        # adding ~(padding_count) to the denominator. Impact is tiny (<0.01% for 47 extra tokens vs 50K vocab)
        # and consistent across training. Inference correctly slices to vocab_size.
        # Default vocab_size=50304 is already aligned, so padding only affects custom unaligned vocab sizes.
        # Set pad_vocab_size_to=1 to disable padding if exact loss values are critical.
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.padded_vocab_size = padded_vocab_size
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx, self.engram_layers) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        # Optional MTP head (DeepSeek-V3 Multi-Token Prediction)
        self.mtp = None
        if config.mtp_enabled:
            from nanochat.mtp import MTPModule
            self.mtp = MTPModule(config)
            self.mtp_lambda = config.mtp_lambda
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            # 1. Init sequence mixer (Mamba or Attention)
            if getattr(block, "is_mamba", False):
                torch.nn.init.uniform_(block.attn.in_proj.weight, -s, s)
                torch.nn.init.zeros_(block.attn.out_proj.weight)
                # Meta-init safety: re-init per-head params that were set in constructor
                with torch.no_grad():
                    A = torch.empty(block.attn.nheads, device=block.attn.A_log.device).uniform_(1, 16)
                    block.attn.A_log.data.copy_(torch.log(A))
                    dt = torch.exp(torch.rand(block.attn.nheads, device=block.attn.dt_bias.device)
                                   * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=0.001)
                    block.attn.dt_bias.data.copy_(dt + torch.log(-torch.expm1(-dt)))
                    block.attn.D.data.fill_(1.0)
                    if hasattr(block.attn, 'B_bias') and block.attn.B_bias is not None:
                        block.attn.B_bias.data.zero_()
                        block.attn.C_bias.data.zero_()
            else:
                torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
                torch.nn.init.zeros_(block.attn.c_proj.weight)
            # 2. Init MLP (always)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            # 3. Engram
            if block.engram is not None:
                torch.nn.init.uniform_(block.engram.in_proj.weight, -s, s)
                for mix in block.engram.order_mix:
                    torch.nn.init.uniform_(mix.weight, -s, s)
                torch.nn.init.zeros_(block.engram.out_proj.weight)
            # 4. mHC
            if block.mhc is not None:
                torch.nn.init.uniform_(block.mhc.score_proj.weight, -s, s)
                torch.nn.init.zeros_(block.mhc.score_out.weight)
            # 5. DSA indexer (additional, not replacement for c_q/c_k/c_v/c_proj)
            if type(block.attn).__name__ == "DeepSeekSparseAttention":
                torch.nn.init.uniform_(block.attn.indexer.q_proj.weight, -s, s)
                torch.nn.init.uniform_(block.attn.indexer.k_proj.weight, -s, s)
                torch.nn.init.uniform_(block.attn.indexer.w_proj.weight, -s * 0.1, s * 0.1)

        # MTP module initialization
        if self.mtp is not None:
            torch.nn.init.zeros_(self.mtp.proj.weight)  # conservative: start near zero
            blk = self.mtp.block
            torch.nn.init.uniform_(blk.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(blk.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(blk.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(blk.attn.c_proj.weight)
            torch.nn.init.uniform_(blk.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(blk.mlp.c_proj.weight)

        # Per-layer scalars
        with torch.no_grad():
            self.resid_lambdas.fill_(1.0)   # 1.0 => typical residual connections at init
            self.x0_lambdas.fill_(0.0)      # 0.0 => skip connection to input is disabled at init

        # Rotary embeddings (use configurable rope_theta for long-context support)
        head_dim = self.config.n_embd // self.config.n_head
        rope_theta = float(getattr(self.config, 'rope_theta', 10000.0))
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim, base=rope_theta)
        self.cos, self.sin = cos, sin

        # Cast entire model to bf16 for optimal performance with CCE
        # This is important for achieving ~20k tok/s on GB10
        if self.transformer.wte.weight.device.type == "cuda":
            self.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter, or None for Mamba layers:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)
        - None: Mamba layer (ignores window_size)

        Pattern string is tiled across layers. Last *attention* layer always gets L (full context).
        Characters: L=long (full context), S=short (half context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."

        # Determine Mamba layer pattern
        m_pattern = getattr(config, 'mamba_pattern', '').upper()
        if getattr(config, 'mamba_enabled', False) and not m_pattern:
            m_pattern = "AAM"

        # Window sizes: use dedicated fields if set, else derive from sequence_len
        long_window = getattr(config, 'window_long', 0) or config.sequence_len
        short_window = getattr(config, 'window_short', 0) or (long_window // 2)
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers, with None for Mamba layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            if m_pattern and m_pattern[layer_idx % len(m_pattern)] == 'M':
                window_sizes.append(None)  # Mamba layers ignore window_size
            else:
                char = pattern[layer_idx % len(pattern)]
                window_sizes.append(char_to_window[char])
        # Last attention layer always gets full context
        if window_sizes[-1] is None:
            print0(f"WARNING: Last layer ({config.n_layer-1}) is Mamba. Consider a pattern ending in 'A'.")
        else:
            window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        nparams_exclude = self.transformer.wte.weight.numel() + self.resid_lambdas.numel() + self.x0_lambdas.numel()
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window and DSA
        attn_flops = 0
        for i, window_size in enumerate(self.window_sizes):
            if window_size is None:
                # Mamba layer: SSD chunked scan FLOPs per token (fwd+bwd = 3x fwd)
                d_inner = getattr(self.config, 'mamba_expand', 2) * self.config.n_embd
                d_state = getattr(self.config, 'mamba_d_state', 64)
                headdim = getattr(self.config, 'mamba_headdim', 128)
                chunk_size = getattr(self.config, 'mamba_chunk_size', 256)
                nheads_m = d_inner // headdim
                attn_flops += 6 * chunk_size * nheads_m * d_state   # CB contraction
                attn_flops += 6 * chunk_size * nheads_m * headdim   # y_local matmul
                attn_flops += 12 * nheads_m * d_state * headdim     # cross-chunk
                continue
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            # DSA layers attend to fewer tokens (top_k_ratio fraction)
            if self.config.dsa_enabled and i >= self.config.dsa_start_layer:
                effective_seq = int(effective_seq * self.config.dsa_top_k_ratio)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return all of the parameters, same as Chinchilla paper.
        Kaplan et al. did not include embedding parameters and said that this led to cleaner scaling laws.
        But Kaplan et al. also had a bug in their results (as pointed out by Chinchilla).
        My own experiments in nanochat confirm the Chinchilla approach gives the much cleaner scaling law.
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper <- good).
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper <- bad)
        """
        nparams = sum(p.numel() for p in self.parameters())
        return nparams

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # Route parameters: Muon gets 2D matrix weights only; AdamW gets everything else.
        # Mamba introduces non-2D params (conv1d.weight=3D, A_log/dt_bias/D=1D, B_bias/C_bias=2D bias)
        # that would crash Muon's Newton-Schulz orthogonalization.
        matrix_params = []
        mamba_adam_params = []  # non-2D params + 2D bias params → AdamW
        wte_params = []
        lm_head_params = []
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        # On XLA/TPU, DSA indexer params are unused (DSA falls back to full attention
        # since mask-based sparse attention is O(T^2) memory). Exclude them from all optimizers.
        device_type = str(next(self.parameters()).device).split(':')[0]
        dsa_indexer_params_list = []
        if device_type == 'xla' and self.config.dsa_enabled:
            for block in self.transformer.h:
                if hasattr(block.attn, 'indexer'):
                    dsa_indexer_params_list.extend(list(block.attn.indexer.parameters()))
            if dsa_indexer_params_list:
                print0(f"DSA: excluded {len(dsa_indexer_params_list)} indexer params from optimizer (unused on XLA)")
        dsa_ids = {id(p) for p in dsa_indexer_params_list}
        for name, p in self.named_parameters():
            if id(p) in dsa_ids or id(p) == id(self.resid_lambdas) or id(p) == id(self.x0_lambdas):
                continue
            if "wte" in name:
                wte_params.append(p)
            elif "lm_head" in name:
                lm_head_params.append(p)
            elif p.ndim != 2 or name.endswith('.bias') or name.endswith('_bias'):
                mamba_adam_params.append(p)
            else:
                matrix_params.append(p)
        n_optim = len(matrix_params) + len(mamba_adam_params) + len(wte_params) + len(lm_head_params) + len(resid_params) + len(x0_params) + len(dsa_indexer_params_list)
        assert len(list(self.parameters())) == n_optim, f"Parameter count mismatch: {len(list(self.parameters()))} != {n_optim}"
        # Create the AdamW optimizer for the embedding, lm_head, per-layer scalars, and Mamba non-matrix params
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=wte_params, lr=embedding_lr * dmodel_lr_scale),
            dict(params=resid_params, lr=scalar_lr * 0.01), # these are a lot more sensitive because they accumulate in the residual stream
            dict(params=x0_params, lr=scalar_lr),
        ]
        if mamba_adam_params:
            adam_groups.append(dict(params=mamba_adam_params, lr=embedding_lr * dmodel_lr_scale))
        adamw_kwargs = dict(betas=adam_betas, eps=1e-10, weight_decay=0.0) # NOTE: weight decay is hardcoded to 0.0 for AdamW, only used in Muon
        # fused=True not supported on XLA/TPU devices
        use_fused = device_type != 'xla'  # fused only works on CUDA, CPU, MPS
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=use_fused)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # Create the Muon optimizer for 2D matrix weights only (no LR scaling for Muon)
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95, weight_decay=weight_decay)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        # Combine the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x  # save initial normalized embedding for x0 residual
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            if self.training and self.config.gradient_checkpointing:
                if idx.device.type == 'xla':
                    # XLA-specific checkpoint that uses optimization barriers
                    # to force the compiler to respect checkpoint boundaries.
                    # PyTorch's checkpoint does NOT save memory on XLA.
                    from torch_xla.utils.checkpoint import checkpoint as xla_checkpoint
                    x = xla_checkpoint(block, x, cos_sin, self.window_sizes[i], kv_cache,
                                       preserve_rng_state=False)
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        block, x, cos_sin, self.window_sizes[i], kv_cache,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
            else:
                x = block(x, cos_sin, self.window_sizes[i], kv_cache)

        # Advance KV cache position ONCE after all layers, regardless of layer type.
        # Previously this was inside CausalSelfAttention/DSA per-layer, which broke
        # when the last layer was Mamba (advance never fired). See Bug #2 in design log.
        if kv_cache is not None:
            kv_cache.advance(T)

        x = norm(x)

        # Softcap: smoothly cap the logits to the range [-softcap, softcap]
        softcap = 15

        if targets is not None:
            # Training: use fused linear + cross entropy (CCE recommended)
            # CCE avoids materializing the huge logits tensor (B*T*V), saving ~8GB for large vocabs.
            # Note: lm_head.weight may have padded_vocab_size rows (see __init__ comment for trade-off).
            main_loss = kernels.fused_linear_cross_entropy(
                x.to(torch.bfloat16),
                self.lm_head.weight.to(torch.bfloat16),
                targets,
                ignore_index=-1,
                softcap=softcap,
                reduction=loss_reduction,
            )
            # Multi-Token Prediction: predict token at position i+2
            if self.mtp is not None and loss_reduction == 'mean':
                mtp_loss = self.mtp(
                    x, targets, self.transformer.wte,
                    self.lm_head.weight, cos_sin, softcap=softcap,
                )
                return main_loss + self.mtp_lambda * mtp_loss
            return main_loss
        else:
            # Inference: compute full logits
            logits = self.lm_head(x) # (B, T, padded_vocab_size)
            logits = logits[..., :self.config.vocab_size] # slice to remove padding
            logits = logits.float() # switch to fp32 for logit softcap
            logits = softcap * torch.tanh(logits / softcap) # squash the logits
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
