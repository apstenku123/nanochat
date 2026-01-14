"""
GPT model using TransformerEngine fused layers for maximum performance.

Notable features:
- te.TransformerLayer for fused attention + MLP kernels
- rotary embeddings (RoPE)
- QK norm (RMSNorm)
- untied weights for token embedding and lm_head
- srelu (squared relu) activation in MLP
- norm after token embedding
- RMSNorm (no learnable gamma in main norm, but TE uses learnable)
- no bias in linear layers
- Group-Query Attention (GQA) support
- NVFP4/FP8 training via Transformer Engine autocast
- Sliding window attention support
"""

import math
from contextlib import nullcontext
from functools import partial
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW
from nanochat import kernels

# ==============================================================================
# Transformer Engine (required for NVFP4/FP8 on GB10/SM121)
# ==============================================================================
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format as TE_Format
from transformer_engine.common.recipe import DelayedScaling as TE_DelayedScaling
from transformer_engine.common.recipe import NVFP4BlockScaling as TE_NVFP4BlockScaling


def check_te_capability(feature: str) -> Tuple[bool, str]:
    """Check if TE supports a feature on current hardware."""
    fn_name = f"is_{feature}_available"
    if not hasattr(te, fn_name):
        return False, f"te.{fn_name} missing"
    try:
        res = getattr(te, fn_name)(return_reason=True)
        if isinstance(res, tuple):
            return bool(res[0]), str(res[1])
        return bool(res), ""
    except Exception:
        try:
            return bool(getattr(te, fn_name)()), ""
        except Exception:
            return False, "Check crashed"


@dataclass
class PrecisionPlan:
    """Describes the precision strategy for training."""
    name: str
    recipe: Optional[Any]
    use_te: bool


def select_precision(
    target: str = "auto",
    disable_rht: bool = True,
    disable_sr: bool = True,  # Must be True for SM121/GB10 (PTX instruction issue)
) -> PrecisionPlan:
    """
    Select precision plan based on hardware capabilities.

    Args:
        target: "auto", "nvfp4", "fp8", or "bf16"
        disable_rht: Disable Random Hadamard Transform (required for SM121/GB10)
        disable_sr: Disable Stochastic Rounding (required for SM121/GB10)

    Returns:
        PrecisionPlan with recipe configuration
    """
    nvfp4_ok, nv4_r = check_te_capability("nvfp4")
    fp8_ok, fp8_r = check_te_capability("fp8")

    print0(f"HW Capability: NVFP4={nvfp4_ok} ({nv4_r}) | FP8={fp8_ok}")

    # 1. NVFP4 Strategy (Blackwell)
    if target in ["auto", "nvfp4"] and nvfp4_ok and TE_NVFP4BlockScaling:
        try:
            # CRITICAL: override_linear_precision=(False, False, True)
            # (Fwd=FP4, Bwd=FP4, WGrad=BF16) - keeps weight updates clean
            recipe = TE_NVFP4BlockScaling(
                fp4_format=getattr(TE_Format, "E2M1", None),
                override_linear_precision=(False, False, True),
                disable_rht=disable_rht,
                disable_stochastic_rounding=disable_sr,
            )
            name = "NVFP4 (E2M1) + WGrad BF16"
            if disable_sr:
                name += " [no SR]"
        except TypeError:
            # Fallback for older TE versions
            recipe = TE_NVFP4BlockScaling(fp4_format=getattr(TE_Format, "E2M1", None))
            name = "NVFP4 (E2M1)"
        return PrecisionPlan(name, recipe, True)

    # 2. FP8 Strategy (Hopper)
    if target in ["auto", "fp8", "nvfp4"] and fp8_ok:
        fmt = getattr(TE_Format, "HYBRID", getattr(TE_Format, "E4M3", None))
        recipe = TE_DelayedScaling(fp8_format=fmt)
        return PrecisionPlan("FP8 (Hybrid)", recipe, True)

    # 3. Fallback (BF16)
    return PrecisionPlan("PyTorch BF16", None, False)


def make_autocast_ctx(plan: PrecisionPlan, device_type: str = "cuda"):
    """
    Create autocast context factory for the precision plan.
    Returns a callable that creates a fresh context each time.
    """
    if plan.use_te and plan.recipe is not None:
        def te_ctx():
            return te.autocast(enabled=True, recipe=plan.recipe)
        return te_ctx
    elif device_type == "cuda":
        return lambda: torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
    else:
        return nullcontext


@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6  # number of query heads
    n_kv_head: int = 6  # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (half context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "L"


def norm(x):
    # Purely functional rmsnorm with no learnable params
    # Uses optimized kernel when available (liger/triton backend)
    return kernels.rms_norm(x)


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config

        # Compute per-layer window sizes for sliding window attention
        self.window_sizes = self._compute_window_sizes(config)

        # Pad vocab for efficiency (DDP, tensor cores)
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.padded_vocab_size = padded_vocab_size

        # Token embedding
        self.wte = nn.Embedding(padded_vocab_size, config.n_embd, device='cuda')

        # Transformer blocks using TE's fused TransformerLayer
        self.blocks = nn.ModuleList([
            te.TransformerLayer(
                hidden_size=config.n_embd,
                ffn_hidden_size=4 * config.n_embd,
                num_attention_heads=config.n_head,
                num_gqa_groups=config.n_kv_head,
                layernorm_epsilon=1e-6,
                hidden_dropout=0.0,
                attention_dropout=0.0,
                bias=False,
                activation='srelu',  # squared relu
                normalization='RMSNorm',
                qk_norm_type='RMSNorm',
                qk_norm_eps=1e-6,
                self_attn_mask_type='causal',
                attn_input_format='bshd',  # (B, T, H) format
                apply_residual_connection_post_layernorm=False,
                device='cuda',
                layer_number=layer_idx + 1,  # TE uses 1-indexed layer numbers
            )
            for layer_idx in range(config.n_layer)
        ])

        # LM head (untied from embedding)
        self.lm_head = te.Linear(config.n_embd, padded_vocab_size, bias=False, device='cuda')

        # Per-layer learnable scalars (applied externally to TE blocks)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer, device='cuda'))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer, device='cuda'))

        # Precompute rotary embeddings in TE format: (T, 1, 1, head_dim)
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000):
        """Precompute rotary embeddings in TE format: (T, 1, 1, head_dim)"""
        device = 'cuda'
        # Compute frequencies
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # Compute position embeddings
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)  # (T, head_dim/2)
        # Compute cos/sin and expand to full head_dim
        cos = freqs.cos()
        sin = freqs.sin()
        # TE expects (T, 1, 1, head_dim) with cos/sin interleaved or concatenated
        cos = torch.cat([cos, cos], dim=-1)  # (T, head_dim)
        sin = torch.cat([sin, sin], dim=-1)  # (T, head_dim)
        # Add batch and head dimensions: (T, 1, 1, head_dim)
        cos = cos.unsqueeze(1).unsqueeze(1).bfloat16()
        sin = sin.unsqueeze(1).unsqueeze(1).bfloat16()
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for TE's window_size parameter.
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."

        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }

        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def init_weights(self):
        """Initialize model weights."""
        n_embd = self.config.n_embd

        # Token embedding: normal with std=1.0
        torch.nn.init.normal_(self.wte.weight, mean=0.0, std=1.0)
        self.wte.to(dtype=torch.bfloat16)

        # LM head: normal with small std
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # TE TransformerLayer weights
        # TE uses fused QKV projection: layernorm_qkv contains the fused weights
        # We need to initialize them to match our original scheme
        s = 3**0.5 * n_embd**-0.5  # Uniform bound for same std as normal

        for block in self.blocks:
            # TE's self_attention has layernorm_qkv (fused LayerNorm + QKV projection)
            # and proj (output projection)
            attn = block.self_attention

            # QKV weights: uniform init
            if hasattr(attn, 'layernorm_qkv') and hasattr(attn.layernorm_qkv, 'weight'):
                # This is the fused QKV weight
                torch.nn.init.uniform_(attn.layernorm_qkv.weight, -s, s)

            # Output projection: zeros
            if hasattr(attn, 'proj') and hasattr(attn.proj, 'weight'):
                torch.nn.init.zeros_(attn.proj.weight)

            # MLP weights (TE has layernorm_mlp with fused weights)
            mlp = block.layernorm_mlp
            if hasattr(mlp, 'fc1_weight'):
                torch.nn.init.uniform_(mlp.fc1_weight, -s, s)
            if hasattr(mlp, 'fc2_weight'):
                torch.nn.init.zeros_(mlp.fc2_weight)
            # Alternative: direct weight access
            if hasattr(mlp, 'weight'):
                # fc1 portion: uniform, fc2 portion: zeros
                # This is more complex with fused weights, so just use uniform for now
                torch.nn.init.uniform_(mlp.weight, -s, s)

        # Per-layer scalars
        with torch.no_grad():
            self.resid_lambdas.fill_(1.0)
            self.x0_lambdas.fill_(0.0)

        # Convert all TE layers to bfloat16
        for block in self.blocks:
            block.to(dtype=torch.bfloat16)
        self.lm_head.to(dtype=torch.bfloat16)

    def get_device(self):
        return self.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        nparams_exclude = self.wte.weight.numel() + self.resid_lambdas.numel() + self.x0_lambdas.numel()
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """Return all parameters (Chinchilla counting)."""
        return sum(p.numel() for p in self.parameters())

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate parameters into groups
        # TE blocks contain the matrix params (attention + MLP weights)
        matrix_params = list(self.blocks.parameters())
        embedding_params = list(self.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]

        # Scale LR by 1/sqrt(dmodel)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
            dict(params=resid_params, lr=scalar_lr * 0.01),
            dict(params=x0_params, lr=scalar_lr),
        ]
        adamw_kwargs = dict(betas=adam_betas, eps=1e-10, weight_decay=0.0)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)

        # Muon for matrix params
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95, weight_decay=weight_decay)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)

        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # Get rotary embeddings for current sequence
        assert T <= self.cos.size(0), f"Sequence too long: {T} > {self.cos.size(0)}"
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos = self.cos[T0:T0+T]  # (T, 1, 1, head_dim)
        sin = self.sin[T0:T0+T]
        rotary_pos_emb = (cos, sin)

        # Forward through embedding + norm
        x = self.wte(idx)
        x = norm(x)
        x0 = x  # Save for x0 residual

        # Forward through TE transformer blocks
        for i, block in enumerate(self.blocks):
            # Apply per-layer scaling before block
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            # TE TransformerLayer forward
            x = block(x, rotary_pos_emb=rotary_pos_emb, window_size=self.window_sizes[i])

        x = norm(x)

        softcap = 15  # Smoothly cap logits to [-softcap, softcap]

        if targets is not None:
            # Training: compute loss
            backend = kernels.get_kernel_backend()
            if backend in ["liger", "triton"] and kernels.LIGER_AVAILABLE:
                # Fused path: hidden_states -> loss
                lm_head_weight = self.lm_head.weight[:self.config.vocab_size, :]
                loss = kernels.fused_linear_cross_entropy(
                    x, lm_head_weight, targets,
                    ignore_index=-1, softcap=softcap
                )
                return loss
            else:
                # Standard path
                logits = self.lm_head(x)
                logits = logits[..., :self.config.vocab_size]
                logits = logits.float()
                logits = softcap * torch.tanh(logits / softcap)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
                return loss
        else:
            # Inference
            logits = self.lm_head(x)
            logits = logits[..., :self.config.vocab_size]
            logits = logits.float()
            logits = softcap * torch.tanh(logits / softcap)
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """Autoregressive generation."""
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            logits = self.forward(ids)
            logits = logits[:, -1, :]
            if top_k is not None:
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
