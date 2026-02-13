# Mamba Integration Design Log

## Overview

Adding hybrid Mamba-2 / Mamba-3 layers to nanochat's dense Transformer,
inspired by NVIDIA Nemotron-3 Nano's hybrid architecture.

**Target**: 24-layer model with AAM pattern (16 attention + 8 mamba layers).

---

## Research Sources

- **Mamba-3 Paper**: "MAMBA-3: IMPROVED SEQUENCE MODELING USING STATE..." (ICLR 2026, OpenReview HwCvaJOiCj)
- **Official Mamba**: github.com/state-spaces/mamba (Mamba2Simple class)
- **Nemotron-3 Nano**: NVIDIA technical report, 52-layer hybrid with ~28 Mamba-2 + ~9 Attention layers

---

## Architecture Decisions

### Layer Pattern: AAM (Attention-Attention-Mamba)

```
Layer  0: Attention+MLP    Layer 12: Attention+MLP
Layer  1: Attention+MLP    Layer 13: Attention+MLP
Layer  2: Mamba-2+MLP      Layer 14: Mamba-2+MLP
Layer  3: Attention+MLP    Layer 15: Attention+MLP
...                        ...
Layer 11: Mamba-2+MLP      Layer 23: Mamba-2+MLP
```

Rationale: Attention acts as "routing hubs" for exact recall/ICL.
Mamba acts as O(1) context compressors between attention layers.

### Module Compatibility Matrix

| Module     | Works with Attention? | Works with Mamba? | Notes |
|------------|----------------------|-------------------|-------|
| MLP        | Yes | Yes, unchanged | Pure feedforward |
| Engram     | Yes | Yes, unchanged | Operates on block input norm(x) |
| mHC        | Yes | Yes, unchanged | Combines (B,T,C) branches |
| DSA        | Yes (IS attention) | Mutually exclusive | Per-layer choice |
| MTP        | Yes | Keep as attention | Internal plain Block |

### Key Insight

Block.forward doesn't need branching. Mamba2Layer matches
CausalSelfAttention's signature: forward(x, cos_sin, window_size, kv_cache)
— it simply ignores cos_sin, window_size, and kv_cache.

---

## Implementation: nanochat/mamba2.py

### Mamba2Layer Class

Drop-in replacement for CausalSelfAttention using SSD (State Space Duality).

**Constructor params from GPTConfig**:
```python
mamba_d_state: int = 64         # SSM state dimension
mamba_d_conv: int = 4           # depthwise conv width
mamba_expand: int = 2           # expansion factor (d_inner = expand * d_model)
mamba_headdim: int = 128        # head dimension
mamba_ngroups: int = 1          # B,C groups (like GQA groups)
mamba_chunk_size: int = 256     # SSD chunk size
```

**Internal structure** (matching official Mamba2Simple):
```
in_proj: Linear(d_model -> [z, x, B, C, dt])
  where dt dimension = nheads (NOT d_model)
conv1d: depthwise Conv1d on [x, B, C] concat
A_log: Parameter(nheads,) — stored as log, A = -exp(A_log)
dt_bias: Parameter(nheads,) — added before softplus
D: Parameter(nheads,) — skip connection
out_proj: Linear(d_inner -> d_model)
```

**Forward path**:
```
in_proj → conv1d → SiLU → split(x,B,C) → SSD scan → gated_rmsnorm(y,z) → out_proj
```

### Three Compute Paths

1. **CUDA training**: `mamba_chunk_scan_combined` Triton kernel (fast)
2. **TPU/XLA training**: `_ssd_scan_ref` chunked reference (matmul-based)
3. **Inference decode (L=1)**: `_ssd_step_ref` O(1) recurrence

### SSD Reference Implementation (_ssd_scan_ref)

Must include BOTH components of the SSD dual:

```python
# 1. Cross-chunk: states accumulated from previous chunks
#    Sequential over nchunks (= L/256 ≈ 8), not L
chunk_states = einsum('bclhn,bclhd->bchnd', B*decay, x*dt)
for c in range(nchunks):  # only 8 iterations, not 2048
    running_state = running_state * chunk_decay + chunk_states[:, c]
y_cross = einsum('bclhn,bchnd->bclhd', C, prev_states) * decay

# 2. Within-chunk: attention-like quadratic (the SSD "dual")
CB = einsum('bclhn,bcshn->bclsh', C, B)  # attention matrix
decay_mat = exp(cumsum[l] - cumsum[s]) * causal_mask
y_local = einsum('bclsh,bcshd->bclhd', CB * decay_mat, x*dt)

y = y_local + y_cross + x * D  # combine
```

CRITICAL: Without within-chunk (y_local), tokens can only see info
compressed into chunk boundary states. Local context mixing is lost.

### O(1) Decode Step (_ssd_step_ref)

Maintains two state buffers per layer in InferenceParams:
- `conv_state`: (B, conv_dim, d_conv) — shift register for causal conv
- `ssm_state`: (B, nheads, headdim, d_state) — recurrent SSM state

```python
# 1. Conv shift register
conv_state = roll(conv_state, -1)
conv_state[:, :, -1] = xBC_raw
xBC = sum(conv_state * conv1d.weight, dim=-1) + bias
xBC = silu(xBC)

# 2. SSM recurrence
dA = exp(dt * A)                              # (B, H)
dBx = (dt * B).unsqueeze(2) * x.unsqueeze(-1) # (B, H, D, N)
ssm_state = dA * ssm_state + dBx              # (B, H, D, N)

# 3. Output
y = (C * ssm_state).sum(-1) + D * x           # (B, H, D)
```

---

## Initialization

### A_log: Match official Mamba2Simple

```python
# CORRECT (official):
A = torch.empty(nheads).uniform_(1, 16)    # A magnitude in [1, 16]
A_log = nn.Parameter(torch.log(A))          # A_log in [0, 2.77]
# Result: A = -exp(A_log) in [-16, -1]

# WRONG (earlier proposals):
# nn.init.uniform_(A_log, -log(64), -log(1))  # A in [-64, -1], 4x too wide
# Large negative A = fast decay = state forgets within ~1 token
```

### conv1d: Leave at PyTorch default

```python
# WRONG: uniform_(-s, s) with s = sqrt(3)/sqrt(n_embd) ≈ 0.048
#   Depthwise conv has fan_in=d_conv=4, kaiming gives s ≈ 0.866
#   Linear-style init is 18x too small, conv starts as near-zero
# CORRECT: Don't re-init. PyTorch's Conv1d.__init__ uses kaiming_uniform.
```

### Block MLP: Always init regardless of layer type

```python
for block in self.transformer.h:
    if block.is_mamba:
        init.uniform_(block.attn.in_proj.weight, -s, s)
        init.zeros_(block.attn.out_proj.weight)
        # conv1d: leave at default kaiming
    elif isinstance(block.attn, DeepSeekSparseAttention):
        # ... existing DSA init ...
    else:
        init.uniform_(block.attn.c_q.weight, -s, s)
        # ... etc ...

    # CRITICAL: Always init MLP
    init.uniform_(block.mlp.c_fc.weight, -s, s)
    init.zeros_(block.mlp.c_proj.weight)
```

---

## Optimizer Routing

### Problem: Muon crashes on non-2D tensors

Muon uses Newton-Schulz orthogonalization — requires 2D matrices.
Mamba introduces 1D params (A_log, dt_bias, D, conv1d.bias) and
3D params (conv1d.weight: (conv_dim, 1, d_conv)).

### Solution: `p.ndim != 2` filter

```python
matrix_params = []       # 2D → Muon
mamba_adam_params = []    # non-2D → AdamW

for name, p in self.transformer.h.named_parameters():
    if p.ndim != 2:
        mamba_adam_params.append(p)
    else:
        matrix_params.append(p)
```

### Preserve LR separation (CRITICAL)

The existing code has carefully tuned separate LRs:
```python
lm_head:    0.004 * scale   # small — output projection is sensitive
wte:        0.2   * scale   # large — embedding can learn fast  
resid:      0.005           # very small — accumulates in residual
x0:         0.5             # moderate
```

NEVER merge these into one group. Mamba non-2D params get their own group:
```python
adam_groups = [
    dict(params=lm_head_params,    lr=unembedding_lr * scale),  # 0.004
    dict(params=wte_params,        lr=embedding_lr * scale),    # 0.2
    dict(params=mamba_adam_params,  lr=embedding_lr * scale),    # 0.2 (new)
    dict(params=resid_params,      lr=scalar_lr * 0.01),        # 0.005
    dict(params=x0_params,         lr=scalar_lr),               # 0.5
]
```

---

## Mamba-3 Incremental Upgrades

Applied surgically to SSD pre-processing. The scan kernel itself is unchanged.

### Phase 2a: QK-Norm on B,C (2 lines)

```python
B_ssm = F.rms_norm(B_ssm, (d_state,))  # parameterless, matching nanochat style
C_ssm = F.rms_norm(C_ssm, (d_state,))
```

Decision: Use `F.rms_norm` (functional, no learnable params) to match
nanochat's existing QK-norm style in CausalSelfAttention:
```python
q, k = norm(q), norm(k)  # also parameterless
```

Do NOT use `nn.RMSNorm` (learnable) — it adds orphan parameters if you
then call `F.rms_norm` in forward instead. Pick one or the other.

### Phase 2b: Learnable B,C bias (2 lines + 2 params)

```python
# __init__:
self.B_bias = nn.Parameter(torch.zeros(ngroups, d_state))
self.C_bias = nn.Parameter(torch.zeros(ngroups, d_state))

# forward (after norm):
B_ssm = B_ssm + self.B_bias.view(1, 1, ngroups, d_state)
C_ssm = C_ssm + self.C_bias.view(1, 1, ngroups, d_state)
```

### Phase 2c: Complex RoPE on B,C

Multi-frequency rotation (NOT single-frequency):
```python
# Per-dimension frequencies (like RoPE's theta^{-2i/d}):
inv_freq = 1.0 / (10000 ** (arange(0, d_state, 2) / d_state))
# NOT: single scalar angle per position (collapses to 1 frequency)

# Cumulative angle from dt:
cumsum_dt = cumsum(dt.mean(dim=-1), dim=1)  # (B, L)
angles = cumsum_dt.unsqueeze(-1) * inv_freq  # (B, L, d_state//2)
```

### Phase 2d: Trapezoidal Discretization

NOT a simple dt averaging. The trapezoidal rule couples adjacent timesteps:
```
h_t = decay(dt) * h_{t-1} + (dt/2) * [B_t*x_t + B_{t-1}*x_{t-1}]
```

This creates a data-dependent width-2 convolution on B*x that cannot be
passed through mamba_chunk_scan_combined (which computes B*x internally).
Requires either a modified SSD mask (per Proposition 4 in the paper)
or a custom Triton kernel. Deferred to Phase 3.

---

## KVCache Integration (engine.py)

```python
# Add to KVCache.__init__:
self.mamba_params = None
if InferenceParams is not None:
    self.mamba_params = InferenceParams(max_seqlen=seq_len, max_batch_size=batch_size)

# InferenceParams.key_value_memory_dict stores per-layer:
#   layer_idx -> {"conv_state": (B, conv_dim, d_conv),
#                 "ssm_state": (B, nheads, headdim, d_state)}
```

---

## Bug Tracker: Issues Found Across Review Iterations

### Iterations 1-4 (early proposals — summary only)

**Iteration 1**: Mamba3Block scan was passthrough (y = x_inner), wrong
constructor (Mamba2 vs Mamba2Simple), init_weights skip included MLP,
missing layer_idx, TPU not addressed, conv1d.weight to Muon (crash),
x_proj dt dimension wrong (d_model vs nheads), Block.forward branching.

**Iteration 2**: Trapezoidal = just avg dt (wrong), B_bias broadcasting
crash (nheads vs ngroups), Complex RoPE single frequency, several
iteration 1 bugs still present.

**Iteration 3**: _ssd_scan_ref missing within-chunk attention, _ssd_step_ref
was placeholder `return x * D`, A_log range [-64,-1] (4x too wide),
optimizer LR merge (lm_head 50x too high), conv1d init 18x too small.

**Iteration 4**: Variable name typo all_states vs all_prev_states (crash),
dead nn.RMSNorm params, conv1d init still 18x too small, MTP fragility,
estimate_flops not updated.

---

### Iteration 5: Full Implementation Draft (Annotated Code)

This was the first complete end-to-end code draft. It fixed many iter 1-4
bugs but introduced 3 critical new bugs and left 2 moderate issues.

#### Proposed mamba2.py (iteration 5) — full code with bug annotations

```python
"""
Mamba-2 Token Mixer with incremental Mamba-3 Upgrades.
Drop-in replacement for CausalSelfAttention in nanochat.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
except ImportError:
    mamba_chunk_scan_combined = None

class Mamba2Layer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.d_model = config.n_embd
        self.d_state = getattr(config, "mamba_d_state", 64)
        self.d_conv = getattr(config, "mamba_d_conv", 4)
        self.expand = getattr(config, "mamba_expand", 2)
        self.headdim = getattr(config, "mamba_headdim", 128)
        self.ngroups = getattr(config, "mamba_ngroups", 1)
        self.chunk_size = getattr(config, "mamba_chunk_size", 256)

        self.d_inner = self.expand * self.d_model
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim

        self.mamba3_qknorm = getattr(config, 'mamba3_qknorm', False)
        self.mamba3_bias = getattr(config, 'mamba3_bias', False)
        self.mamba3_complex_rope = getattr(config, 'mamba3_complex_rope', False)

        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False)

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim, out_channels=conv_dim, kernel_size=self.d_conv,
            groups=conv_dim, padding=self.d_conv - 1, bias=True
        )  # ✓ CORRECT: left at PyTorch default kaiming_uniform

        if self.mamba3_bias:
            self.B_bias = nn.Parameter(torch.zeros(self.ngroups, self.d_state))
            self.C_bias = nn.Parameter(torch.zeros(self.ngroups, self.d_state))

        self.A_log = nn.Parameter(torch.empty(self.nheads))
        self.dt_bias = nn.Parameter(torch.empty(self.nheads))
        self.D = nn.Parameter(torch.ones(self.nheads))

        if self.mamba3_complex_rope:
            inv_freq = 1.0 / (10000 ** (torch.arange(0, self.d_state, 2).float() / self.d_state))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

        # ✓ CORRECT: Official A_log init range
        A = torch.empty(self.nheads).uniform_(1, 16)
        self.A_log.data.copy_(torch.log(A))

        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=0.001)
        self.dt_bias.data.copy_(dt + torch.log(-torch.expm1(-dt)))

    def _apply_complex_rope(self, tensor, dt_soft, inference_params=None):
        B_sz, L, G, N = tensor.shape
        dt_avg = dt_soft.view(B_sz, L, G, -1).mean(dim=-1)

        if inference_params is not None and L == 1:
            key = f"rope_angle_{self.layer_idx}"
            rope_angle = inference_params.key_value_memory_dict.setdefault(
                key, torch.zeros(B_sz, G, device=tensor.device, dtype=tensor.dtype)
            )
            rope_angle = rope_angle + dt_avg.squeeze(1)
            inference_params.key_value_memory_dict[key] = rope_angle
            angles = rope_angle.unsqueeze(1).unsqueeze(-1) * self.inv_freq.view(1, 1, 1, N//2)
        else:
            cumsum_dt = torch.cumsum(dt_avg, dim=1)
            angles = cumsum_dt.unsqueeze(-1) * self.inv_freq.view(1, 1, 1, N//2)
            #
            # ██ BUG (MODERATE): Prefill→decode angle discontinuity
            # After prefill (L > 1), the final cumulative angle is NOT stored.
            # When autoregressive decode starts (L=1), rope_angle initializes
            # from zeros via setdefault above. The decode angles are offset
            # from their correct values by the entire prefill's cumulative dt.
            # FIX: add here:
            #   if inference_params is not None:
            #       key = f"rope_angle_{self.layer_idx}"
            #       inference_params.key_value_memory_dict[key] = cumsum_dt[:, -1]
            #
        x1, x2 = tensor[..., :N//2], tensor[..., N//2:]
        cos, sin = torch.cos(angles), torch.sin(angles)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def forward(self, x, cos_sin=None, window_size=None, kv_cache=None):
        B_sz, L, _ = x.shape
        inference_params = getattr(kv_cache, 'mamba_params', None) if kv_cache is not None else None

        if inference_params is not None and L == 1:
            return self._ssd_step_ref(x, inference_params)

        zxbcdt = self.in_proj(x)
        z = zxbcdt[..., :self.d_inner]
        xBC_raw = zxbcdt[..., self.d_inner : self.d_inner + self.d_inner + 2*self.ngroups*self.d_state]
        dt = zxbcdt[..., -self.nheads:]

        xBC = xBC_raw.transpose(1, 2)

        if inference_params is not None:
            states = inference_params.key_value_memory_dict.setdefault(self.layer_idx, {})
            conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
            if "conv_state" not in states:
                states["conv_state"] = torch.zeros(B_sz, conv_dim, self.d_conv, device=x.device, dtype=x.dtype)
            conv_state = states["conv_state"]
            if L >= self.d_conv:
                conv_state.copy_(xBC[:, :, -self.d_conv:])
            else:
                conv_state.copy_(torch.roll(conv_state, shifts=-L, dims=-1))
                conv_state[:, :, -L:] = xBC

        xBC = self.conv1d(xBC)[..., :L].transpose(1, 2)
        xBC = F.silu(xBC)

        x_ssm = xBC[..., :self.d_inner].view(B_sz, L, self.nheads, self.headdim)
        B_ssm = xBC[..., self.d_inner : self.d_inner + self.ngroups*self.d_state].view(B_sz, L, self.ngroups, self.d_state)
        C_ssm = xBC[..., -self.ngroups*self.d_state:].view(B_sz, L, self.ngroups, self.d_state)

        # ✓ CORRECT: Using F.rms_norm (parameterless), no dead nn.RMSNorm
        if self.mamba3_qknorm:
            B_ssm = F.rms_norm(B_ssm, (self.d_state,))
            C_ssm = F.rms_norm(C_ssm, (self.d_state,))
        if self.mamba3_bias:
            B_ssm = B_ssm + self.B_bias.view(1, 1, self.ngroups, self.d_state)
            C_ssm = C_ssm + self.C_bias.view(1, 1, self.ngroups, self.d_state)

        dt_soft = F.softplus(dt + self.dt_bias)

        if self.mamba3_complex_rope:
            B_ssm = self._apply_complex_rope(B_ssm, dt_soft)
            C_ssm = self._apply_complex_rope(C_ssm, dt_soft)

        A = -torch.exp(self.A_log)

        if mamba_chunk_scan_combined is not None and x.device.type == "cuda" and inference_params is None:
            y = mamba_chunk_scan_combined(x_ssm, dt_soft, A, B_ssm, C_ssm, chunk_size=self.chunk_size, D=self.D)
        else:
            y, final_states = self._ssd_scan_ref(x_ssm, dt_soft, A, B_ssm, C_ssm, self.D)
            if inference_params is not None:
                inference_params.key_value_memory_dict[self.layer_idx]["ssm_state"] = final_states

        y = y.view(B_sz, L, self.d_inner)
        y = F.rms_norm(y, (self.d_inner,)) * F.silu(z)
        return self.out_proj(y)

    def _ssd_step_ref(self, x, inference_params):
        """True O(1) Autoregressive Decode Step."""
        B_sz = x.shape[0]
        zxbcdt = self.in_proj(x.squeeze(1))

        z = zxbcdt[..., :self.d_inner]
        xBC_raw = zxbcdt[..., self.d_inner : self.d_inner + self.d_inner + 2*self.ngroups*self.d_state]
        dt = zxbcdt[..., -self.nheads:]

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        states = inference_params.key_value_memory_dict.setdefault(self.layer_idx, {})

        if "conv_state" not in states:
            states["conv_state"] = torch.zeros(B_sz, conv_dim, self.d_conv, device=x.device, dtype=x.dtype)
        if "ssm_state" not in states:
            states["ssm_state"] = torch.zeros(B_sz, self.nheads, self.d_state, self.headdim, device=x.device, dtype=x.dtype)

        conv_state = states["conv_state"]
        ssm_state = states["ssm_state"]

        conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
        conv_state[:, :, -1] = xBC_raw

        xBC_conv = torch.sum(conv_state * self.conv1d.weight.squeeze(1), dim=-1)
        if self.conv1d.bias is not None:
            xBC_conv += self.conv1d.bias
        xBC_conv = F.silu(xBC_conv)

        x_ssm = xBC_conv[..., :self.d_inner].view(B_sz, self.nheads, self.headdim)
        B_ssm = xBC_conv[..., self.d_inner : self.d_inner + self.ngroups*self.d_state].view(B_sz, self.ngroups, self.d_state)
        C_ssm = xBC_conv[..., -self.ngroups*self.d_state:].view(B_sz, self.ngroups, self.d_state)

        if self.mamba3_qknorm:
            B_ssm = F.rms_norm(B_ssm, (self.d_state,))
            C_ssm = F.rms_norm(C_ssm, (self.d_state,))
        if self.mamba3_bias:
            B_ssm = B_ssm + self.B_bias.view(1, self.ngroups, self.d_state)
            C_ssm = C_ssm + self.C_bias.view(1, self.ngroups, self.d_state)

        dt_soft = F.softplus(dt + self.dt_bias)
        A = -torch.exp(self.A_log)

        if self.mamba3_complex_rope:
            B_ssm = self._apply_complex_rope(B_ssm.unsqueeze(1), dt_soft.unsqueeze(1), inference_params).squeeze(1)
            C_ssm = self._apply_complex_rope(C_ssm.unsqueeze(1), dt_soft.unsqueeze(1), inference_params).squeeze(1)

        heads_per_group = self.nheads // self.ngroups
        B_ssm = B_ssm.repeat_interleave(heads_per_group, dim=1)
        C_ssm = C_ssm.repeat_interleave(heads_per_group, dim=1)

        dA = torch.exp(dt_soft * A)
        dBx = (dt_soft.unsqueeze(-1) * B_ssm).unsqueeze(-1) * x_ssm.unsqueeze(-2)

        ssm_state.copy_(ssm_state * dA.view(B_sz, self.nheads, 1, 1) + dBx)

        y = (C_ssm.unsqueeze(-1) * ssm_state).sum(dim=-2) + self.D.view(1, self.nheads, 1) * x_ssm

        y = y.view(B_sz, 1, self.d_inner)
        y = F.rms_norm(y, (self.d_inner,)) * F.silu(z.unsqueeze(1))
        return self.out_proj(y)

    def _ssd_scan_ref(self, x, dt, A, B, C, D):
        """Chunked PyTorch Reference for TPU (Cross-Chunk + Within-Chunk)"""
        B_sz, L, H, D_head = x.shape
        _, _, G, N = B.shape
        cs = self.chunk_size

        pad = (cs - L % cs) % cs
        if pad > 0:
            #
            # ████████████████████████████████████████████████████████████████
            # ██ BUG #1 (CRITICAL): F.pad pads the BATCH dimension
            # ██
            # ██ F.pad processes dimensions LAST to FIRST. For a 4D tensor
            # ██ (B, L, H, D), 8 values map to:
            # ██   (D_left, D_right, H_left, H_right, L_left, L_right, B_left, B_right)
            # ██
            # ██ So (0,0, 0,0, 0,0, 0,pad) pads dim 0 (BATCH), not dim 1 (L).
            # ██ This silently adds `pad` phantom batch elements filled with
            # ██ zeros. The subsequent .view(B_sz, nchunks, ...) reshapes
            # ██ corrupted data without any error.
            # ██
            # ██ FIX: Use 6 values to pad up to dim 1:
            # ██   x  = F.pad(x,  (0,0, 0,0, 0,pad))  # pads dim 1 (L)
            # ██   dt = F.pad(dt, (0,0, 0,pad))         # pads dim 1 (L)
            # ██   B  = F.pad(B,  (0,0, 0,0, 0,pad))   # pads dim 1 (L)
            # ██   C  = F.pad(C,  (0,0, 0,0, 0,pad))   # pads dim 1 (L)
            # ████████████████████████████████████████████████████████████████
            #
            x = F.pad(x, (0,0, 0,0, 0,0, 0,pad))      # ◄◄◄ WRONG: pads dim 0
            dt = F.pad(dt, (0,0, 0,pad))                # ✓ OK: 4 values on 3D
            B = F.pad(B, (0,0, 0,0, 0,0, 0,pad))       # ◄◄◄ WRONG: pads dim 0
            C = F.pad(C, (0,0, 0,0, 0,0, 0,pad))       # ◄◄◄ WRONG: pads dim 0

        nchunks = x.shape[1] // cs
        x_c = x.view(B_sz, nchunks, cs, H, D_head)
        dt_c = dt.view(B_sz, nchunks, cs, H)
        B_c = B.view(B_sz, nchunks, cs, G, N)
        C_c = C.view(B_sz, nchunks, cs, G, N)

        heads_per_group = H // G
        B_h = B_c.repeat_interleave(heads_per_group, dim=3)
        C_h = C_c.repeat_interleave(heads_per_group, dim=3)

        dA_c = dt_c * A.view(1, 1, 1, H)
        dA_cumsum = torch.cumsum(dA_c, dim=2)

        decay_to_end = torch.exp(dA_cumsum[:, :, -1:] - dA_cumsum)
        x_dt = x_c * dt_c.unsqueeze(-1)
        chunk_states = torch.einsum('bclhn,bclhd->bchnd', B_h * decay_to_end.unsqueeze(-1), x_dt)

        chunk_decay = torch.exp(dA_cumsum[:, :, -1])
        running_state = torch.zeros(B_sz, H, N, D_head, device=x.device, dtype=x.dtype)

        all_prev_states = []
        for c in range(nchunks):
            all_prev_states.append(running_state.clone())
            running_state = running_state * chunk_decay[:, c].view(B_sz, H, 1, 1) + chunk_states[:, c]

        prev_states = torch.stack(all_prev_states, dim=1)  # ✓ CORRECT name

        cross_decay = torch.exp(dA_cumsum).unsqueeze(-1)
        y_cross = torch.einsum('bclhn,bchnd->bclhd', C_h, prev_states) * cross_decay

        CB = torch.einsum('bclhn,bcshn->bclsh', C_h, B_h)
        diff = dA_cumsum.unsqueeze(3) - dA_cumsum.unsqueeze(2)
        causal_mask = torch.tril(torch.ones(cs, cs, device=x.device, dtype=torch.bool))

        decay_mat = torch.exp(diff.masked_fill(~causal_mask.view(1, 1, cs, cs, 1), float('-inf')))
        attn = CB * decay_mat
        y_local = torch.einsum('bclsh,bcshd->bclhd', attn, x_dt)

        y = y_local + y_cross + x_c * D.view(1, 1, 1, H, 1)
        y = y.reshape(B_sz, nchunks * cs, H, D_head)

        if pad > 0:
            y = y[:, :L]
        return y, running_state
```

#### Proposed gpt.py changes (iteration 5) — with bug annotations

##### setup_optimizers (proposed replacement)

```python
def setup_optimizers(self, ...):
    # ... (routing loop) ...
    for name, p in self.named_parameters():
        if id(p) in dsa_ids: continue
        if id(p) == id(self.resid_lambdas) or id(p) == id(self.x0_lambdas): continue

        if "wte" in name:
            wte_params.append(p)
        elif "lm_head" in name:
            lm_head_params.append(p)
        elif p.ndim != 2 or 'bias' in name:
            mamba_adam_params.append(p)
        else:
            matrix_params.append(p)
    # ✓ CORRECT: Separate LR groups preserved
    adam_groups = [
        dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
        dict(params=wte_params,     lr=embedding_lr * dmodel_lr_scale),
        dict(params=resid_params,   lr=scalar_lr * 0.01),
        dict(params=x0_params,      lr=scalar_lr),
    ]
    if mamba_adam_params:
        adam_groups.append(dict(params=mamba_adam_params, lr=embedding_lr * dmodel_lr_scale))
```

##### init_weights (proposed replacement)

```python
for block in self.transformer.h:
    if getattr(block, "is_mamba", False):
        torch.nn.init.uniform_(block.attn.in_proj.weight, -s, s)
        torch.nn.init.zeros_(block.attn.out_proj.weight)
    elif hasattr(block.attn, "indexer"):
        #
        # ██████████████████████████████████████████████████████████
        # ██ BUG #3 (CRITICAL): DSA attention weights not initialized
        # ██
        # ██ `pass` means c_q, c_k, c_v, c_proj stay at meta-device
        # ██ garbage. DSA (DeepSeekSparseAttention) has ALL the same
        # ██ attention weights as CausalSelfAttention PLUS an indexer.
        # ██
        # ██ Current working code in gpt.py:346-366:
        # ██   for block in self.transformer.h:
        # ██       # ALL blocks get this (including DSA):
        # ██       init.uniform_(block.attn.c_q.weight, -s, s)
        # ██       init.uniform_(block.attn.c_k.weight, -s, s)
        # ██       init.uniform_(block.attn.c_v.weight, -s, s)
        # ██       init.zeros_(block.attn.c_proj.weight)
        # ██       # THEN DSA blocks get additional indexer init:
        # ██       if isinstance(block.attn, DeepSeekSparseAttention):
        # ██           init.uniform_(block.attn.indexer.q_proj.weight, -s, s)
        # ██           init.uniform_(block.attn.indexer.k_proj.weight, -s, s)
        # ██           init.uniform_(block.attn.indexer.w_proj.weight, ...)
        # ██
        # ██ FIX: DSA gets standard attention init (else branch),
        # ██ then indexer init is ADDITIONAL (separate if after else):
        # ██   if getattr(block, "is_mamba", False):
        # ██       init mamba...
        # ██   else:
        # ██       init c_q, c_k, c_v, c_proj  (ALL non-mamba blocks)
        # ██   if isinstance(block.attn, DeepSeekSparseAttention):
        # ██       init indexer...  (ADDITIONAL, not replacement)
        # ██████████████████████████████████████████████████████████
        #
        pass  # ◄◄◄ WRONG: skips c_q/c_k/c_v/c_proj init for DSA blocks
    else:
        torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
        torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
        torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
        torch.nn.init.zeros_(block.attn.c_proj.weight)

    torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
    torch.nn.init.zeros_(block.mlp.c_proj.weight)
```

##### kv_cache.advance() — the invisible bug

This bug exists NOT in any proposed code but in the EXISTING code
that the proposal fails to modify. The current CausalSelfAttention
in gpt.py:196-199:

```python
# gpt.py lines 196-199 (EXISTING CODE):
    # Advance position after last layer processes
    if self.layer_idx == kv_cache.n_layers - 1:
        kv_cache.advance(T)
```

With AAM pattern on 24 layers, layer 23 is 'M' (Mamba).
Mamba2Layer.forward() ignores kv_cache entirely — it uses
inference_params from kv_cache.mamba_params instead.

No attention layer has layer_idx == 23, so advance() NEVER fires.

```
# ████████████████████████████████████████████████████████████████████
# ██ BUG #2 (CRITICAL): kv_cache.advance(T) never called
# ██
# ██ Consequences during autoregressive inference:
# ██   1. kv_cache.get_pos() returns 0 forever
# ██   2. In GPT.forward (line 543):
# ██        T0 = 0 if kv_cache is None else kv_cache.get_pos()
# ██        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]
# ██      → Rotary embeddings always use position 0
# ██   3. Attention layers overwrite cache slot 0 every decode step
# ██   4. Model produces garbage after the first token
# ██
# ██ FIX: Move advance() to GPT.forward, after the block loop:
# ██   # After line 567 (after the for block loop), before x = norm(x):
# ██   if kv_cache is not None:
# ██       kv_cache.advance(T)
# ██
# ██ And REMOVE the advance calls from:
# ██   - CausalSelfAttention.forward (gpt.py lines 198-199)
# ██   - DeepSeekSparseAttention._full_attention (sparse_attention.py lines 142-143)
# ██
# ██ This is cleaner architecture: the model owns position tracking,
# ██ not individual layers. Works regardless of layer type mix.
# ████████████████████████████████████████████████████████████████████
```

---

### Iteration 6: Surgical Fixes (Annotated Code)

This iteration claimed "100% Scorecard Achieved". It fixed 5 issues
from iteration 4 but left all 3 critical bugs from iteration 5
unaddressed, and introduced a new moderate bug in estimate_flops.

#### What iteration 6 correctly fixed

**Fix A: all_prev_states typo** (was all_states → NameError)
```python
# Was (iteration 4): torch.stack(all_states, dim=1)
# Now (iteration 6): torch.stack(all_prev_states, dim=1)  ✓
```

**Fix B: Dead nn.RMSNorm removed** — no longer creates nn.RMSNorm
objects then calls F.rms_norm in forward. Uses F.rms_norm throughout. ✓

**Fix C: conv1d not re-initialized** — correctly left at PyTorch's
kaiming_uniform default (fan_in=d_conv=4, not dense layer's fan_in=n_embd). ✓

**Fix D: MTP mamba_enabled=False** — proposed in mtp.py:
```python
plain_config = replace(
    config,
    engram_enabled=False,
    mhc_enabled=False,
    dsa_enabled=False,
    mamba_enabled=False,  # ◄ NEW: prevents MTP block becoming Mamba
)
# Note: this REQUIRES mamba_enabled to exist as a field in GPTConfig,
# otherwise dataclasses.replace() raises TypeError. Not yet added.
```

**Fix E: MLP always initialized** ✓

#### What iteration 6 left broken

Bugs #1, #2, #3 from iteration 5 were NOT ADDRESSED — the iteration
did not even mention them. See annotations in iteration 5 code above.

#### New bug introduced in iteration 6: estimate_flops

```python
def estimate_flops(self):
    # ...
    for i, window_size in enumerate(self.window_sizes):
        if window_size is None:
            # Mamba FLOPs: Conv1D + SSD Scan
            # Note: Parameter matmuls (in_proj, out_proj) are ALREADY counted by the
            # 6 * (nparams - nparams_exclude) base equation.
            d_inner = getattr(self.config, 'mamba_expand', 2) * self.config.n_embd
            d_state = getattr(self.config, 'mamba_d_state', 64)
            ngroups = getattr(self.config, 'mamba_ngroups', 1)

            conv_dim = d_inner + 2 * ngroups * d_state
            d_conv = getattr(self.config, 'mamba_d_conv', 4)

            #
            # ██████████████████████████████████████████████████████████████
            # ██ BUG #4 (MODERATE): Double-counting + wrong units
            # ██
            # ██ Problem 1: conv1d.weight IS a model parameter counted
            # ██ in nparams. Its FLOPs are already in 6 * (nparams - ...).
            # ██ Adding them again double-counts.
            # ██
            # ██ Problem 2: The accumulator attn_flops collects PER-TOKEN
            # ██ non-weight FLOPs. The attention formula 12*h*q*s is
            # ██ per-token (each token attends to s others).
            # ██ But 2*t*conv_dim*d_conv and 2*t*d_inner*d_state multiply
            # ██ by t, giving PER-SEQUENCE totals. This inflates Mamba's
            # ██ contribution by a factor of t (e.g. 2048x).
            # ██
            # ██ For our config (d_inner=1536, d_state=64, t=2048):
            # ██   Proposed conv1d:  2*2048*1664*4  = 27.3M  (should be 0)
            # ██   Proposed SSD:     2*2048*1536*64 = 402.7M (should be ~4.7M)
            # ██   Total per Mamba layer: ~430M (should be ~4.7M, ~91x too high)
            # ██
            # ██ FIX: Remove conv1d (already counted). Compute SSD FLOPs
            # ██ per-token from chunk_size (see "estimate_flops derivation
            # ██ for SSD" in Open Design Questions section).
            # ██████████████████████████████████████████████████████████████
            #
            attn_flops += 2 * t * conv_dim * d_conv     # ◄◄◄ WRONG: double-counted + per-seq
            attn_flops += 2 * t * d_inner * d_state      # ◄◄◄ WRONG: per-seq not per-token
            continue

        window = window_size[0]
        effective_seq = t if window < 0 else min(window, t)
        if getattr(self.config, 'dsa_enabled', False) and i >= getattr(self.config, 'dsa_start_layer', 7):
            effective_seq = int(effective_seq * getattr(self.config, 'dsa_top_k_ratio', 0.5))
        attn_flops += 12 * h * q * effective_seq

    return 6 * (nparams - nparams_exclude) + attn_flops
```

---

### Cumulative Bug Status Table (after iteration 6)

| # | Severity | Description | Introduced | Fixed? |
|---|----------|-------------|-----------|--------|
| 1 | CRITICAL | F.pad 8-val on 4D pads batch dim in _ssd_scan_ref | Iter 5 | NO |
| 2 | CRITICAL | kv_cache.advance() not called when last layer is Mamba | Iter 5 | NO |
| 3 | CRITICAL | init_weights `elif pass` skips DSA c_q/c_k/c_v/c_proj | Iter 5 | NO |
| 4 | MODERATE | estimate_flops double-counts conv1d + wrong units | Iter 6 | — |
| 5 | MODERATE | Complex RoPE angle lost at prefill→decode boundary | Iter 5 | NO |
| 6 | MODERATE | Missing: Block.__init__, KVCache, CLI, GPTConfig fields | Iter 5 | NO |

---

### Corrected Code for All 6 Open Bugs

What follows is the exact corrected code for each bug. The next
iteration MUST incorporate all of these.

#### Bug #1 Fix: F.pad dimension count

```python
# In _ssd_scan_ref, replace the padding block:
pad = (cs - L % cs) % cs
if pad > 0:
    # F.pad processes dimensions LAST to FIRST.
    # 6 values for 4D tensor: (dim3_left, dim3_right, dim2_left, dim2_right, dim1_left, dim1_right)
    # This pads dim 1 (sequence length L), leaving dims 0,2,3 untouched.
    x  = F.pad(x,  (0, 0, 0, 0, 0, pad))  # (B, L, H, D) → pads L
    dt = F.pad(dt, (0, 0, 0, pad))          # (B, L, H)    → pads L
    B  = F.pad(B,  (0, 0, 0, 0, 0, pad))  # (B, L, G, N) → pads L
    C  = F.pad(C,  (0, 0, 0, 0, 0, pad))  # (B, L, G, N) → pads L
```

#### Bug #2 Fix: Move kv_cache.advance() to GPT.forward

```python
# In GPT.forward, REPLACE lines 550-567:
#
# BEFORE (existing code):
#   for i, block in enumerate(self.transformer.h):
#       x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
#       ... checkpoint or block call ...
#   x = norm(x)
#
# AFTER (with advance moved here):
    for i, block in enumerate(self.transformer.h):
        x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        if self.training and self.config.gradient_checkpointing:
            if idx.device.type == 'xla':
                from torch_xla.utils.checkpoint import checkpoint as xla_checkpoint
                x = xla_checkpoint(block, x, cos_sin, self.window_sizes[i], kv_cache,
                                   preserve_rng_state=False)
            else:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, cos_sin, self.window_sizes[i], kv_cache,
                    use_reentrant=False, preserve_rng_state=False)
        else:
            x = block(x, cos_sin, self.window_sizes[i], kv_cache)

    # ✓ NEW: Advance KV cache position AFTER all layers, regardless of layer type.
    # Previously this was inside CausalSelfAttention.forward (line 198) and
    # DeepSeekSparseAttention._full_attention (line 142), triggered only when
    # layer_idx == n_layers-1. With Mamba layers, the last layer may not be
    # attention, so advance() would never fire.
    if kv_cache is not None:
        kv_cache.advance(T)

    x = norm(x)


# ALSO: Remove these lines from CausalSelfAttention.forward (gpt.py:196-199):
#   # Advance position after last layer processes
#   if self.layer_idx == kv_cache.n_layers - 1:
#       kv_cache.advance(T)
#
# AND from DeepSeekSparseAttention._full_attention (sparse_attention.py:140-143):
#   if self.layer_idx == kv_cache.n_layers - 1:
#       kv_cache.advance(T)
```

#### Bug #3 Fix: init_weights for DSA blocks

```python
# In GPT.init_weights, REPLACE the block loop:
    for block in self.transformer.h:
        # 1. Init Sequence Mixer
        if getattr(block, "is_mamba", False):
            # Mamba: init projections, leave conv1d at PyTorch kaiming default
            torch.nn.init.uniform_(block.attn.in_proj.weight, -s, s)
            torch.nn.init.zeros_(block.attn.out_proj.weight)
        else:
            # Standard attention init — works for BOTH CSA and DSA
            # (DSA has c_q/c_k/c_v/c_proj just like CSA)
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)

        # 2. Init MLP (ALL blocks, always)
        torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
        torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # 3. Init Engram (if present on this block)
        if block.engram is not None:
            torch.nn.init.uniform_(block.engram.in_proj.weight, -s, s)
            for mix in block.engram.order_mix:
                torch.nn.init.uniform_(mix.weight, -s, s)
            torch.nn.init.zeros_(block.engram.out_proj.weight)

        # 4. Init mHC (if present on this block)
        if block.mhc is not None:
            torch.nn.init.uniform_(block.mhc.score_proj.weight, -s, s)
            torch.nn.init.zeros_(block.mhc.score_out.weight)

        # 5. DSA indexer: ADDITIONAL init (not a replacement for c_q/c_k/c_v/c_proj)
        if isinstance(block.attn, DeepSeekSparseAttention):
            torch.nn.init.uniform_(block.attn.indexer.q_proj.weight, -s, s)
            torch.nn.init.uniform_(block.attn.indexer.k_proj.weight, -s, s)
            torch.nn.init.uniform_(block.attn.indexer.w_proj.weight, -s * 0.1, s * 0.1)
```

#### Bug #4 Fix: estimate_flops correct derivation

```python
# In GPT.estimate_flops, the Mamba branch inside the window_size loop:
    for i, window_size in enumerate(self.window_sizes):
        if window_size is None:
            # SSD chunked scan: non-weight FLOPs per token (forward+backward = 3x forward)
            # Conv1d weight FLOPs are already counted in 6*(nparams-nparams_exclude).
            # in_proj/out_proj weight FLOPs are already counted there too.
            # What remains: the SSD einsum operations (analogous to QK^T and AV in attention).
            d_inner = getattr(self.config, 'mamba_expand', 2) * self.config.n_embd
            d_state = getattr(self.config, 'mamba_d_state', 64)
            headdim = getattr(self.config, 'mamba_headdim', 128)
            chunk_size = getattr(self.config, 'mamba_chunk_size', 256)
            nheads_m = d_inner // headdim

            # Per-token forward FLOPs:
            #   CB contraction:    2 * chunk_size * nheads_m * d_state
            #   y_local matmul:    2 * chunk_size * nheads_m * headdim
            #   y_cross + states:  4 * nheads_m * d_state * headdim
            # Forward+backward = 3x forward:
            attn_flops += 6 * chunk_size * nheads_m * d_state
            attn_flops += 6 * chunk_size * nheads_m * headdim
            attn_flops += 12 * nheads_m * d_state * headdim
            continue

        # ... existing attention FLOPs code ...
```

#### Bug #5 Fix: Complex RoPE angle preservation

```python
# In _apply_complex_rope, the prefill branch (else clause):
    else:
        cumsum_dt = torch.cumsum(dt_avg, dim=1)
        angles = cumsum_dt.unsqueeze(-1) * self.inv_freq.view(1, 1, 1, N//2)

        # ✓ FIX: Store final angle for decode continuation
        if inference_params is not None:
            key = f"rope_angle_{self.layer_idx}"
            inference_params.key_value_memory_dict[key] = cumsum_dt[:, -1]  # (B, G)
```

#### Bug #6 Fix: Missing integration code

##### GPTConfig new fields (add after existing fields in gpt.py):

```python
@dataclass
class GPTConfig:
    # ... existing fields ...
    # Mamba-2 hybrid layers
    mamba_enabled: bool = False
    mamba_pattern: str = "A"          # A=attention, M=mamba, tiled across layers
    mamba_d_state: int = 64           # SSM state dimension
    mamba_d_conv: int = 4             # conv1d kernel size
    mamba_expand: int = 2             # expansion factor (d_inner = expand * n_embd)
    mamba_headdim: int = 128          # head dimension for SSD
    mamba_ngroups: int = 1            # number of groups for B/C (GQA-like)
    mamba_chunk_size: int = 256       # chunk size for SSD scan
    # Mamba-3 upgrades (Phase 2, all default off)
    mamba3_qknorm: bool = False       # QK-norm on B/C
    mamba3_bias: bool = False         # learnable B/C bias
    mamba3_complex_rope: bool = False # complex RoPE on B/C
```

##### Block.__init__ Mamba dispatch (replace in gpt.py):

```python
class Block(nn.Module):
    def __init__(self, config, layer_idx, engram_layers):
        super().__init__()
        # Determine layer type: Mamba > DSA > CSA (mutually exclusive priority)
        m_pattern = getattr(config, 'mamba_pattern', 'A').upper() if getattr(config, 'mamba_enabled', False) else None
        self.is_mamba = m_pattern is not None and m_pattern[layer_idx % len(m_pattern)] == 'M'

        if self.is_mamba:
            from nanochat.mamba2 import Mamba2Layer
            self.attn = Mamba2Layer(config, layer_idx)
        elif bool(config.dsa_enabled) and layer_idx >= config.dsa_start_layer:
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
        # ... rest unchanged (engram, mhc) ...
```

##### KVCache changes (engine.py):

```python
class MambaInferenceParams:
    """Lightweight stand-in for mamba_ssm's InferenceParams.
    Stores per-layer conv_state and ssm_state keyed by layer_idx."""
    def __init__(self):
        self.key_value_memory_dict = {}

class KVCache:
    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers,
                 device, dtype=torch.bfloat16, has_mamba=False):
        # ... existing k_cache/v_cache/cache_seqlens init ...
        self.mamba_params = MambaInferenceParams() if has_mamba else None

    def reset(self):
        self.cache_seqlens.zero_()
        if self.mamba_params is not None:
            self.mamba_params.key_value_memory_dict.clear()

    # advance() stays here but is now called from GPT.forward, not per-layer
```

Engine.generate must pass `has_mamba=True` when creating KVCache if
the model config has `mamba_enabled=True`.

##### MTP mamba_enabled=False (mtp.py):

```python
# In MTPModule.__init__:
    plain_config = replace(
        config,
        engram_enabled=False,
        mhc_enabled=False,
        dsa_enabled=False,
        mamba_enabled=False,  # ◄ prevents MTP block becoming Mamba
    )
```

##### CLI args (base_train.py):

```python
# Add to argparse:
parser.add_argument("--mamba_enabled", type=int, default=0)
parser.add_argument("--mamba_pattern", type=str, default="A")
parser.add_argument("--mamba_d_state", type=int, default=64)
parser.add_argument("--mamba_d_conv", type=int, default=4)
parser.add_argument("--mamba_expand", type=int, default=2)
parser.add_argument("--mamba_headdim", type=int, default=128)
parser.add_argument("--mamba_ngroups", type=int, default=1)
parser.add_argument("--mamba_chunk_size", type=int, default=256)
parser.add_argument("--mamba3_qknorm", type=int, default=0)
parser.add_argument("--mamba3_bias", type=int, default=0)
parser.add_argument("--mamba3_complex_rope", type=int, default=0)
```

---

## Open Design Questions

### kv_cache.advance() ownership

Current design: advance() called from within individual attention modules
when `layer_idx == n_layers - 1`. This is fragile — any non-attention last
layer breaks it. Recommended: move to GPT.forward after the block loop.
This requires removing advance calls from CausalSelfAttention (line 198)
and DeepSeekSparseAttention._full_attention (line 142).

### estimate_flops derivation for SSD

The `attn_flops` accumulator adds per-token non-weight FLOPs per layer.
For attention: `12 * h * q * s` (forward 4x + backward 8x).

For SSD chunked scan, the per-token non-weight operations are:
```
CB contraction (einsum 'bclhn,bcshn→bclsh'):
  contracts over d_state (N), each token has chunk_size pairs
  → 2 * chunk_size * nheads * d_state per token

y_local matmul (einsum 'bclsh,bcshd→bclhd'):
  contracts over chunk_size
  → 2 * chunk_size * nheads * headdim per token

y_cross (einsum 'bclhn,bchnd→bclhd'):
  contracts over d_state
  → 2 * nheads * d_state * headdim per token

chunk_states (einsum, amortized):
  → 2 * nheads * d_state * headdim per token

Forward total per token:
  2*cs*H*N + 2*cs*H*D + 4*H*N*D

With 3x fwd+bwd multiplier:
  6*cs*H*N + 6*cs*H*D + 12*H*N*D

Example (cs=256, H=12, N=64, D=128):
  1.18M + 2.36M + 1.18M ≈ 4.7M per token per layer
  vs attention (s=2048): 37.7M per token per layer
  Ratio: Mamba ≈ 8x cheaper (matches chunk_size/seq_len ratio)
```

### DSA + Mamba mutual exclusivity

A layer is either Mamba, DSA, or standard CSA. Priority: Mamba > DSA > CSA.
In Block.__init__, the Mamba check must happen BEFORE the DSA check.
DSA blocks still have c_q/c_k/c_v/c_proj and need standard attention init.

### B_bias/C_bias are 2D tensors

B_bias is (ngroups, d_state) = 2D. `p.ndim != 2` alone won't catch it.
The optimizer routing uses `'bias' in name` to also route these to AdamW.
This is correct but fragile — any future param with "bias" in its name
(e.g. `unbiased_proj`) would be misrouted. Tighter filter:
  `name.endswith('.bias') or name.endswith('_bias')`

---

### Iteration 7: Two Proposals Reviewed

Two proposals submitted simultaneously. **Proposal A** ("Final 100%
complete") continues our Mamba2Layer drop-in design. **Proposal B**
("Another look") is a separate Mamba3/hybrid design. Both reviewed
against actual source files.

---

#### Proposal A: Iteration 7 — Status of Previously-Open Bugs

**Bug #1 (F.pad)** — ✅ FIXED. `_ssd_scan_ref` now uses 6-value padding:
```python
x  = F.pad(x,  (0, 0, 0, 0, 0, pad))  # 4D (B,L,H,D): pads dim 1
dt = F.pad(dt, (0, 0, 0, pad))          # 3D (B,L,H):   pads dim 1
B  = F.pad(B,  (0, 0, 0, 0, 0, pad))  # 4D (B,L,G,N): pads dim 1
C  = F.pad(C,  (0, 0, 0, 0, 0, pad))  # 4D (B,L,G,N): pads dim 1
```

**Bug #2 (kv_cache.advance)** — ✅ FIXED. Moved to GPT.forward after
block loop. Reminder: must also REMOVE advance calls from
CausalSelfAttention.forward (gpt.py:197-199) and
DeepSeekSparseAttention._full_attention (sparse_attention.py:142-143).

**Bug #3 (init_weights DSA)** — ✅ FIXED. Structure is now:
```python
if is_mamba:    → init mamba projections
else:          → init c_q/c_k/c_v/c_proj (ALL non-mamba, including DSA)
if hasattr(block.attn, "indexer"):  → ADDITIONAL indexer init
```

**Bug #4 (estimate_flops)** — ✅ FIXED. Uses SSD per-token formula
matching our derivation exactly.

**Bug #5 (Complex RoPE angle)** — ✅ FIXED. After prefill, stores
`cumsum_dt[:, -1]` for decode continuation.

**Bug #6 (Missing integration code)** — ✅ FIXED. All pieces provided:
GPTConfig fields, Block.__init__ dispatch, KVCache, MTP, CLI args.

**.clone() removal** — ✅ SAFE. `running_state = running_state * ... + ...`
creates a new tensor (Python `*` and `+` are not in-place), so the old
reference in `all_prev_states` is not corrupted.

**Bias filter** — ✅ CORRECT. Tightened from `'bias' in name` to
`name.endswith('.bias') or name.endswith('_bias')`.

---

#### Proposal A: NEW Bugs Found in Iteration 7

##### Bug #7 (CRITICAL): KVCache.prefill() mamba state type mismatch

The proposed prefill code:
```python
self.mamba_params.key_value_memory_dict = {
    k: v.clone() if isinstance(v, torch.Tensor) else tuple(t.clone() for t in v)
    for k, v in other.mamba_params.key_value_memory_dict.items()
}
```

But Mamba2Layer stores states as **dicts**, not tensors or tuples:
```python
# In Mamba2Layer.forward:
states = inference_params.key_value_memory_dict.setdefault(self.layer_idx, {})
states["conv_state"] = torch.zeros(...)  # value is a dict of tensors
states["ssm_state"] = final_states
```

So `key_value_memory_dict[layer_idx]` is a `dict`. The isinstance checks:
- `isinstance(v, torch.Tensor)` → False (it's a dict)
- `tuple(t.clone() for t in v)` → iterates over dict KEYS (strings
  "conv_state", "ssm_state"), tries `.clone()` on strings
  → **AttributeError crash**

Additionally, complex RoPE stores `f"rope_angle_{layer_idx}"` keyed by
STRING → `torch.Tensor`. So the dict has mixed key types (int for layer
states, str for RoPE angles) and mixed value types (dict for states,
Tensor for RoPE). Neither case is handled correctly.

**FIX**: Deep-copy respecting actual types:
```python
# In KVCache.prefill():
if self.mamba_params is not None and other.mamba_params is not None:
    self.mamba_params.seqlen_offset = other.mamba_params.seqlen_offset
    new_dict = {}
    for k, v in other.mamba_params.key_value_memory_dict.items():
        if isinstance(v, torch.Tensor):
            # RoPE angle tensors (str keys): expand batch if needed
            if v.size(0) == 1 and self.batch_size > 1:
                new_dict[k] = v.expand(self.batch_size, *v.shape[1:]).clone()
            else:
                new_dict[k] = v.clone()
        elif isinstance(v, dict):
            # Per-layer state dicts (int keys): deep-clone + batch expand
            new_dict[k] = {
                sk: (sv.expand(self.batch_size, *sv.shape[1:]).clone()
                     if sv.size(0) == 1 and self.batch_size > 1
                     else sv.clone())
                for sk, sv in v.items()
            }
        else:
            raise TypeError(f"Unexpected mamba state type for key {k}: {type(v)}")
    self.mamba_params.key_value_memory_dict = new_dict
```

##### Bug #8 (CRITICAL): Mamba state batch expansion missing in prefill

Engine.generate creates prefill cache with batch_size=1, then a decode
cache with batch_size=num_samples. KVCache.prefill() copies KV tensors
from (1,T,H,D) → (N,T,H,D) via broadcast. But Proposal A's mamba state
copy just clones without expansion:

Prefill (batch=1) produces:
- `conv_state`: shape (1, conv_dim, d_conv)
- `ssm_state`: shape (1, H, N, D)
- `rope_angle_*`: shape (1, G)

After prefill into decode cache (batch=num_samples), tensors are still
(1, ...). When `_ssd_step_ref` runs with batch=N:
```python
conv_state[:, :, -1] = xBC_raw  # xBC_raw is (N, conv_dim), but
                                  # conv_state[:,:,-1] is (1, conv_dim)
                                  # → Shape mismatch crash
```

**FIX**: Included in Bug #7 fix above — the `.expand().clone()` pattern.
Proposal B correctly identifies this issue.

##### Bug #9 (MODERATE): Triton prefill not used during inference

The code guards Triton kernel with `inference_params is None`:
```python
if mamba_chunk_scan_combined is not None and x.device.type == "cuda" and inference_params is None:
    y = mamba_chunk_scan_combined(...)
```

This means CUDA inference prefill (L>1, inference_params is not None)
falls through to `_ssd_scan_ref` Python loops (~8 iterations). This is
**correct** but slow. Triton should also be used for prefill with
`return_final_states=True` to get the SSM state:

```python
if mamba_chunk_scan_combined is not None and x.device.type == "cuda":
    if inference_params is not None:
        # Prefill: use Triton AND capture final states
        y, final_states = mamba_chunk_scan_combined(
            x_ssm, dt_soft, A, B_ssm, C_ssm,
            chunk_size=self.chunk_size, D=self.D,
            return_final_states=True,
        )
        inference_params.key_value_memory_dict.setdefault(
            self.layer_idx, {}
        )["ssm_state"] = final_states
    else:
        # Training: no states needed
        y = mamba_chunk_scan_combined(
            x_ssm, dt_soft, A, B_ssm, C_ssm,
            chunk_size=self.chunk_size, D=self.D,
        )
else:
    y, final_states = self._ssd_scan_ref(...)
    if inference_params is not None:
        inference_params.key_value_memory_dict.setdefault(
            self.layer_idx, {}
        )["ssm_state"] = final_states
```

##### Bug #10 (MODERATE): Engine.generate KVCache not shown

The proposal says "update instantiations of KVCache to include
has_mamba=" but doesn't show actual code. Both KVCache constructors in
Engine.generate need `has_mamba=getattr(self.model.config, 'mamba_enabled', False)`.

Engine.generate creates two caches:
1. `kv_cache_prefill` (batch=1, seq_len=prompt_len)
2. `kv_cache_decode` (batch=num_samples, seq_len=prompt_len + max_tokens)

Both must have `has_mamba=True` for Mamba state dicts to be allocated.

##### Bug #11 (LOW): MTP hasattr guard unnecessary

```python
if hasattr(config, 'mamba_enabled'):
    kwargs['mamba_enabled'] = False
```

Since mamba_enabled is added to GPTConfig as a dataclass field, it
always exists. Match existing style in mtp.py (line 42):
```python
plain_config = replace(config,
    engram_enabled=False, mhc_enabled=False,
    dsa_enabled=False, mamba_enabled=False,
)
```

---

#### Proposal B: "Another Look" (Mamba3/Hybrid) — Full Review

**Overall verdict: DO NOT ADOPT.** Violates multiple core design
principles. But contains 5 valuable ideas to extract.

##### Critical Violations

**Violation 1: Breaks drop-in replacement architecture**

Creates a separate `MambaBlock` class with a different forward()
signature (`inference_params=` kwarg instead of positional kv_cache).
Requires isinstance branching in GPT.__init__ and maintaining two
parallel Block classes. Our design uses `Block.attn = Mamba2Layer(...)`.

```python
# Proposal B (WRONG — two different block types):
if t == "A":
    h.append(Block(config, layer_idx, engram_layers))
elif t == "M":
    h.append(MambaBlock(config, layer_idx, engram_layers))

# Our design (CORRECT — one Block class, swapped mixer):
# Block.__init__:
if self.is_mamba:
    self.attn = Mamba2Layer(config, layer_idx)
else:
    self.attn = CausalSelfAttention(config, layer_idx)
```

**Violation 2: O(L) Python loop makes training unusable**

`Mamba3Mixer.forward` loops `for t in range(L)` — 2048 sequential
Python iterations per layer per batch. Our `_ssd_scan_ref` uses chunked
scan with ~8 iterations (L/chunk_size = 2048/256). For 8 Mamba layers:
- Proposal B: 8 * 2048 = 16,384 Python loop iterations
- Proposal A: 8 * 8 = 64 Python loop iterations
That's a 256x Python overhead difference.

**Violation 3: Breaks residual scaling**

```python
# Existing formula (gpt.py:549-553):
x = resid_lambdas[i] * x + x0_lambdas[i] * x0
x = block(x, ...)

# Proposal B changes to:
x_next = block(x, ...)
x = x_next + resid_lambdas[i] * (x_next - x) + x0_lambdas[i] * x0
```

These are algebraically different. The existing formula scales the INPUT,
not the output. Changing this changes model behavior for all layers.

**Violation 4: conv1d re-init 18x too small (iteration 3 regression)**

`reset_parameters()` re-initializes conv1d with `uniform_(-s, s)` where
`s = 1/sqrt(d_model) ≈ 0.036`. PyTorch's kaiming_uniform with
fan_in=d_conv=4 gives `s ≈ 0.866`. This is 24x too small — the same
bug from iteration 3 that we already fixed.

**Violation 5: Double init wastes work**

`init_weights` manually initializes A_log/dt_bias/D, then at the end
calls `mixer.reset_parameters()` which re-initializes EVERYTHING
including in_proj and out_proj (overwriting the manual init). The manual
init is wasted. Also, reset_parameters uses wrong conv1d init (see V4).

**Violation 6: _compute_window_sizes signature change**

Takes a new `layer_types` parameter, breaking the existing call site
in GPT.__init__ which calls `self._compute_window_sizes(config)` with
no second argument.

**Violation 7: Attribute name inconsistency**

Uses `kv_cache.ssm_params` instead of our `kv_cache.mamba_params`.
Inconsistent naming across the codebase.

**Violation 8: Misnamed "Mamba3"**

Only implements trapezoidal discretization. Missing QK-norm on B/C,
learnable B/C bias, complex RoPE — the actual Mamba-3 upgrades we
designed in Phases 2a-2c. Our Mamba2Layer + toggle flags is the correct
approach for incremental Mamba-3 upgrades.

##### Valuable Ideas to Extract from Proposal B

**Idea 1: fp32 SSM state accumulation** (ADOPT — Phase 1)

**Problem**: Our Proposal A initializes `running_state` with `x.dtype`
(bf16 in training). The chunked scan accumulates across ~8 chunks via a
geometric series:
```python
running_state = running_state * chunk_decay[:, c].view(B, H, 1, 1) + chunk_states[:, c]
```

In bf16 (7-bit mantissa, range ±65504), this accumulation can lose
precision when `chunk_decay` ≈ 1 (small A*dt). After 8 multiplications,
the least-significant bits of the state are lost. Proposal B avoids this
by using fp32 for the state:

```python
ssm_state = torch.zeros(B, self.nheads, self.headdim, self.d_state,
                         device=device, dtype=torch.float32)
```

**What to change in `_ssd_scan_ref`**:
```python
def _ssd_scan_ref(self, x, dt, A, B, C, D):
    B_sz, L, H, D_head = x.shape
    _, _, G, N = B.shape
    cs = self.chunk_size

    # ... padding and chunking (unchanged) ...

    # FP32 CHANGE 1: running_state in fp32 for precision
    running_state = torch.zeros(B_sz, H, N, D_head,
                                device=x.device, dtype=torch.float32)

    all_prev_states = []
    for c in range(nchunks):
        all_prev_states.append(running_state)
        # Both chunk_decay and chunk_states are bf16, but running_state
        # is fp32 — PyTorch auto-promotes to fp32 via mixed precision
        running_state = running_state * chunk_decay[:, c].view(B_sz, H, 1, 1) + chunk_states[:, c]

    prev_states = torch.stack(all_prev_states, dim=1)
    # prev_states is fp32 → y_cross will be fp32 after einsum
    # This is fine: y_local and D*x_c are bf16, but PyTorch promotes
    # the addition to fp32 automatically

    # ... y_local, y_cross computation (unchanged) ...

    y = y_local + y_cross + x_c * D.view(1, 1, 1, H, 1)
    y = y.reshape(B_sz, nchunks * cs, H, D_head)

    if pad > 0:
        y = y[:, :L]
    # FP32 CHANGE 2: cast y back to input dtype for downstream norm/gate
    return y.to(x.dtype), running_state
```

**What to change in `_ssd_step_ref`** (decode path):
```python
def _ssd_step_ref(self, x, inference_params):
    B_sz = x.shape[0]
    # ... projections, conv, B/C extraction (unchanged) ...

    states = inference_params.key_value_memory_dict.setdefault(self.layer_idx, {})
    if "ssm_state" not in states:
        # FP32 CHANGE 3: decode ssm_state in fp32
        # This accumulates over potentially thousands of tokens,
        # making precision even more critical than training
        states["ssm_state"] = torch.zeros(B_sz, self.nheads,
            self.d_state, self.headdim,
            device=x.device, dtype=torch.float32)
    ssm_state = states["ssm_state"]

    # ... dA, dBx computation (unchanged) ...
    # dA is bf16, ssm_state is fp32 → auto-promoted
    ssm_state.copy_(ssm_state * dA.view(B_sz, self.nheads, 1, 1) + dBx)

    y = (C_ssm.unsqueeze(-1) * ssm_state).sum(dim=-2) + self.D.view(1, self.nheads, 1) * x_ssm
    y = y.view(B_sz, 1, self.d_inner)
    # FP32 CHANGE 4: cast before norm/gate (bf16 expected downstream)
    y = y.to(x.dtype)
    y = F.rms_norm(y, (self.d_inner,)) * F.silu(z.unsqueeze(1))
    return self.out_proj(y)
```

**Why fp32 matters numerically**: The SSM state h_t follows the recurrence
`h_t = exp(Δ*A) * h_{t-1} + Δ*B*x`. When exp(Δ*A) ≈ 0.99 (small decay),
the state acts as a leaky integrator. After 2048 decode steps:
- bf16 (7-bit mantissa): distinguishes ~128 levels → state quantizes
- fp32 (23-bit mantissa): distinguishes ~8M levels → smooth gradient

The Triton kernel (`mamba_chunk_scan_combined`) already uses fp32
internally for its scan, so this change only affects the fallback path.

**Memory cost**: Negligible. `running_state` is (B, H, N, D) =
(B, 12, 64, 128) = 96K elements. At fp32 vs bf16 that's +192KB total,
independent of sequence length.

---

**Idea 2: Batch expansion in SSM prefill** (ADOPT — already in Bug #7/#8 fix)

**Problem**: Engine.generate creates a prefill cache (batch=1), runs the
prompt through, then creates a decode cache (batch=num_samples) and calls
`decode_cache.prefill(prefill_cache)`. For KV attention, this works via
tensor broadcasting. For Mamba states, the batch dimension must be
explicitly expanded.

**The state shapes that need expansion** (per Mamba layer):

| State key | Prefill shape (batch=1) | Decode shape (batch=N) |
|-----------|------------------------|----------------------|
| `conv_state` | (1, conv_dim, d_conv) | (N, conv_dim, d_conv) |
| `ssm_state` | (1, H, N_state, D) | (N, H, N_state, D) |
| `rope_angle_*` | (1, G) | (N, G) |

**The expansion pattern** (from our Bug #7 fix):
```python
# For Tensor values (rope angles):
if v.size(0) == 1 and self.batch_size > 1:
    new_dict[k] = v.expand(self.batch_size, *v.shape[1:]).clone()

# For dict values (per-layer state dicts):
elif isinstance(v, dict):
    new_dict[k] = {
        sk: (sv.expand(self.batch_size, *sv.shape[1:]).clone()
             if sv.size(0) == 1 and self.batch_size > 1
             else sv.clone())
        for sk, sv in v.items()
    }
```

**Why `.expand().clone()` not `.repeat()`**: `.expand()` creates a view
with stride=0 on the batch dim (no memory copy). The `.clone()` then
materializes it. This is equivalent to `.repeat(N, 1, ...)` but avoids
computing the repeat counts for arbitrary dimensionality. Both are O(N *
numel_per_sample).

**Already incorporated** in Bug #7 fix code (lines 1572-1593 above).

---

**Idea 3: Trapezoidal discretization** (DEFER — Phase 3 reference)

Proposal B provides the full Mamba-3 trapezoidal update equations. These
are documented here as the Phase 3 implementation spec.

**Equations** (from Mamba-3 paper):
```
h_t = α_t h_{t-1} + β_t (B_{t-1} ⊗ x_{t-1}) + γ_t (B_t ⊗ x_t)

where:
  α_t = exp(Δ_t * A)                         ← same as Mamba-2
  β_t = (1 - λ_t) * Δ_t * exp(Δ_t * A)      ← backward Euler term
  γ_t = λ_t * Δ_t                            ← forward Euler term
  λ_t = σ(u_t)                               ← trapezoidal gate (learned)
  u_t = W_λ @ token_t                        ← linear projection per-head
```

When λ_t = 0: purely backward Euler (like Mamba-2, uses B_{t-1}*x_{t-1})
When λ_t = 1: purely forward Euler (uses B_t*x_t)
When λ_t = 0.5: classic trapezoidal rule (equal weight to both)

**What changes from Mamba-2 to enable this**:

1. **New parameter**: `W_lambda` projection (nheads output dims) added to
   `in_proj`. Changes `d_in_proj` from:
   ```python
   # Mamba-2:
   d_in_proj = 2*d_inner + 2*ngroups*d_state + nheads
   # Mamba-3 trapezoidal:
   d_in_proj = 2*d_inner + 2*ngroups*d_state + nheads + nheads  # +nheads for λ
   ```

2. **New cached state**: `prev_B` and `prev_x` per layer (for B_{t-1}*x_{t-1}).
   During decode:
   ```python
   # At end of each step:
   states["prev_B"] = B_ssm_expanded  # (B, H, N) after repeat_interleave
   states["prev_x"] = x_ssm           # (B, H, D)
   ```

3. **Modified recurrence** in `_ssd_step_ref`:
   ```python
   # Extract λ from projection
   lam = torch.sigmoid(zxbcdt[..., -self.nheads:])  # (B, H)

   # Trapezoidal coefficients
   alpha = torch.exp(dt_soft * A)                               # same
   beta = (1.0 - lam) * dt_soft * alpha                         # backward term
   gamma = lam * dt_soft                                        # forward term

   # Previous token's outer product (from cache)
   prev_B = states.get("prev_B", torch.zeros_like(B_ssm_expanded))
   prev_x = states.get("prev_x", torch.zeros_like(x_ssm))
   dBx_prev = prev_B.unsqueeze(-1) * prev_x.unsqueeze(-2)      # (B,H,N,D)
   dBx_curr = B_ssm_expanded.unsqueeze(-1) * x_ssm.unsqueeze(-2)

   # Trapezoidal update
   ssm_state.copy_(
       ssm_state * alpha.view(B_sz, self.nheads, 1, 1)
       + dBx_prev * beta.view(B_sz, self.nheads, 1, 1)
       + dBx_curr * gamma.view(B_sz, self.nheads, 1, 1)
   )

   # Cache for next step
   states["prev_B"] = B_ssm_expanded
   states["prev_x"] = x_ssm
   ```

4. **Training path problem**: The chunked SSD scan (`_ssd_scan_ref`)
   computes B*x internally via `torch.einsum('bclhn,bclhd->bchnd', B_h *
   decay, x_dt)`. The trapezoidal rule needs BOTH chunk c's B*x AND chunk
   c-1's last-position B*x for the cross-chunk boundary. This requires
   either:

   a. **Modified SSD mask** per Proposition 4 in the Mamba-3 paper: encode
      the trapezoidal weights into the structured mask matrix. This is the
      mathematically clean approach but requires understanding Proposition 4
      deeply and modifying the mask construction.

   b. **Custom Triton kernel**: Write a fused trapezoidal scan that handles
      both Euler terms in one pass. Requires kernel development effort.

   c. **Naive O(L) loop** (Proposal B's approach): Works for correctness
      validation but 256x slower than chunked scan. NOT suitable for training.

   **Decision**: Defer trapezoidal to Phase 3. Phases 2a-2c (QK-norm,
   learnable B/C bias, complex RoPE) are independent toggles that work
   with the existing Mamba-2 chunked scan without modification.

---

**Idea 4: checkpoint_manager.py compatibility patching** (ADOPT — Phase 1)

When loading checkpoints saved before Mamba fields existed, the config
will be missing mamba-related keys. The existing `_patch_missing_config_keys`
function in `checkpoint_manager.py` uses `dict.setdefault()` to add
missing keys with backward-compatible defaults.

**Current pattern** (from checkpoint_manager.py):
```python
def _patch_missing_config_keys(model_config_kwargs):
    model_config_kwargs.setdefault('engram_enabled', False)
    model_config_kwargs.setdefault('mhc_enabled', False)
    model_config_kwargs.setdefault('dsa_enabled', False)
    model_config_kwargs.setdefault('mtp_enabled', False)
    # ... etc
```

**What to add** (all Mamba config fields with their GPTConfig defaults):
```python
    # Mamba-2 hybrid integration
    model_config_kwargs.setdefault('mamba_enabled', False)
    model_config_kwargs.setdefault('mamba_pattern', 'A')
    model_config_kwargs.setdefault('mamba_d_state', 64)
    model_config_kwargs.setdefault('mamba_d_conv', 4)
    model_config_kwargs.setdefault('mamba_expand', 2)
    model_config_kwargs.setdefault('mamba_headdim', 128)
    model_config_kwargs.setdefault('mamba_ngroups', 1)
    model_config_kwargs.setdefault('mamba_chunk_size', 256)
    # Mamba-3 toggles
    model_config_kwargs.setdefault('mamba3_qknorm', False)
    model_config_kwargs.setdefault('mamba3_bias', False)
    model_config_kwargs.setdefault('mamba3_complex_rope', False)
```

**Why this matters**: Without these defaults, loading a pre-mamba checkpoint
will fail at `GPTConfig(**model_config_kwargs)` because the saved dict
won't contain mamba fields. The `_build_gpt_config` function filters to
only known GPTConfig fields, so these defaults ensure backward compat:
- Old checkpoint (no mamba keys) → defaults applied → `mamba_enabled=False`
- New checkpoint (has mamba keys) → `.setdefault()` is a no-op

**Also needed in `_patch_missing_keys`** (for state_dict compatibility):
When Mamba layers are present in the model but not in a checkpoint's
state_dict, PyTorch's `load_state_dict(strict=False)` will skip them.
This is already the correct behavior — no additional patching needed for
state_dict keys, only for config keys.

---

**Idea 5: prev_B/prev_x state caching strategy for trapezoidal** (DEFER — Phase 3 spec)

The trapezoidal update's β_t term needs `B_{t-1} ⊗ x_{t-1}` — the
outer product from the PREVIOUS timestep. Two caching strategies exist:

**Strategy A: Cache the product** (precomputed)
```python
# Cache: prev_Bx of shape (B, H, N, D)
states["prev_Bx"] = B_ssm_expanded.unsqueeze(-1) * x_ssm.unsqueeze(-2)
```
Memory per layer: B * H * N * D * sizeof(dtype)
= 1 * 12 * 64 * 128 * 2 = 192KB (bf16)

**Strategy B: Cache factors separately** (Proposal B's approach)
```python
# Cache: prev_B of shape (B, H, N), prev_x of shape (B, H, D)
states["prev_B"] = B_ssm_expanded   # (B, H, N) = 1*12*64 = 768 elements
states["prev_x"] = x_ssm            # (B, H, D) = 1*12*128 = 1536 elements
```
Memory per layer: B * H * (N + D) * sizeof(dtype)
= 1 * 12 * (64 + 128) * 2 = 4.5KB (bf16)

**Comparison**:
- Strategy A: 192KB/layer × 8 layers = 1.5MB total. No recomputation.
- Strategy B: 4.5KB/layer × 8 layers = 36KB total. Requires one
  outer product recomputation per decode step per layer (trivially cheap).

**Decision**: Strategy B is 42x more memory-efficient with negligible
compute overhead. The outer product `B.unsqueeze(-1) * x.unsqueeze(-2)`
is a single fused multiply on modern GPUs.

**Integration into Mamba2Layer state dict structure**:
```python
# In _ssd_step_ref, at the end of each step (Phase 3 only):
if self.mamba3_trapezoidal:  # Phase 3 toggle
    states["prev_B"] = B_ssm  # after repeat_interleave: (B, H, N)
    states["prev_x"] = x_ssm  # (B, H, D)

# In KVCache.prefill(), these are just additional tensor entries in the
# per-layer state dict, handled by the Bug #7 fix's isinstance(v, dict)
# branch — no additional prefill code needed.
```

**First-token initialization**: At the first decode step after prefill
(or at sequence start), `prev_B` and `prev_x` don't exist yet. Use
zero initialization:
```python
prev_B = states.get("prev_B", torch.zeros(B_sz, self.nheads,
                     self.d_state, device=x.device, dtype=x.dtype))
prev_x = states.get("prev_x", torch.zeros(B_sz, self.nheads,
                     self.headdim, device=x.device, dtype=x.dtype))
```

This is semantically correct: at t=0, there is no previous token, so
`B_{-1} * x_{-1} = 0`. The β_t term contributes nothing on the first
step, and the γ_t term (forward Euler, like Mamba-2) handles initialization.

---

#### Cumulative Bug Status Table (after iteration 7)

| # | Severity | Description | Introduced | Fixed? |
|---|----------|-------------|-----------|--------|
| 1 | CRITICAL | F.pad 8-val on 4D pads batch dim | Iter 5 | ✅ Iter 7 |
| 2 | CRITICAL | kv_cache.advance() never called (last layer Mamba) | Iter 5 | ✅ Iter 7 |
| 3 | CRITICAL | init_weights skips DSA c_q/c_k/c_v/c_proj | Iter 5 | ✅ Iter 7 |
| 4 | MODERATE | estimate_flops double-counts + wrong units | Iter 6 | ✅ Iter 7 |
| 5 | MODERATE | Complex RoPE angle lost at prefill→decode | Iter 5 | ✅ Iter 7 |
| 6 | MODERATE | Missing Block.__init__, KVCache, CLI, GPTConfig | Iter 5 | ✅ Iter 7 |
| 7 | CRITICAL | KVCache.prefill() mamba state type mismatch | Iter 7 | **NO** |
| 8 | CRITICAL | Mamba state batch expansion missing in prefill | Iter 7 | **NO** |
| 9 | MODERATE | Triton kernel not used during inference prefill | Iter 7 | **NO** |
| 10 | MODERATE | Engine.generate KVCache constructors not shown | Iter 7 | **NO** |
| 11 | LOW | MTP hasattr guard unnecessary (use replace directly) | Iter 7 | **NO** |

Bugs #1-6 from iterations 5-6 are all **FIXED** in iteration 7.
Bugs #7-8 are **CRITICAL** new regressions in KVCache prefill.
Bugs #9-11 are moderate/low improvements needed.

---

#### Corrected Code for All 5 Open Bugs (#7-#11)

##### Bug #7 + #8 Fix: KVCache.prefill() mamba states

```python
# In KVCache.prefill(), REPLACE the mamba state copy block:
    if self.mamba_params is not None and other.mamba_params is not None:
        self.mamba_params.seqlen_offset = other.mamba_params.seqlen_offset
        new_dict = {}
        for k, v in other.mamba_params.key_value_memory_dict.items():
            if isinstance(v, torch.Tensor):
                # String-keyed entries (e.g. "rope_angle_3"): Tensor
                if v.size(0) == 1 and self.batch_size > 1:
                    new_dict[k] = v.expand(self.batch_size, *v.shape[1:]).clone()
                else:
                    new_dict[k] = v.clone()
            elif isinstance(v, dict):
                # Int-keyed entries (layer_idx): dict of Tensors
                new_dict[k] = {}
                for sk, sv in v.items():
                    if sv.size(0) == 1 and self.batch_size > 1:
                        new_dict[k][sk] = sv.expand(self.batch_size, *sv.shape[1:]).clone()
                    else:
                        new_dict[k][sk] = sv.clone()
            else:
                raise TypeError(f"Unexpected mamba state type for key {k}: {type(v)}")
        self.mamba_params.key_value_memory_dict = new_dict
```

##### Bug #9 Fix: Triton prefill with return_final_states

```python
# In Mamba2Layer.forward(), REPLACE the Triton guard:
    if mamba_chunk_scan_combined is not None and x.device.type == "cuda":
        if inference_params is not None:
            y, final_states = mamba_chunk_scan_combined(
                x_ssm, dt_soft, A, B_ssm, C_ssm,
                chunk_size=self.chunk_size, D=self.D,
                return_final_states=True,
            )
            inference_params.key_value_memory_dict.setdefault(
                self.layer_idx, {}
            )["ssm_state"] = final_states
        else:
            y = mamba_chunk_scan_combined(
                x_ssm, dt_soft, A, B_ssm, C_ssm,
                chunk_size=self.chunk_size, D=self.D,
            )
    else:
        y, final_states = self._ssd_scan_ref(x_ssm, dt_soft, A, B_ssm, C_ssm, self.D)
        if inference_params is not None:
            inference_params.key_value_memory_dict.setdefault(
                self.layer_idx, {}
            )["ssm_state"] = final_states
```

##### Bug #10 Fix: Engine.generate KVCache with has_mamba

```python
# In Engine.generate, BOTH KVCache constructors need has_mamba:
    has_mamba = getattr(self.model.config, 'mamba_enabled', False)

    kv_cache = KVCache(
        batch_size=1, num_heads=..., seq_len=..., head_dim=...,
        num_layers=..., device=device, has_mamba=has_mamba,
    )

    # ... and the decode cache:
    decode_cache = KVCache(
        batch_size=num_samples, num_heads=..., seq_len=..., head_dim=...,
        num_layers=..., device=device, has_mamba=has_mamba,
    )
```

##### Bug #11 Fix: MTP mamba_enabled=False (direct replace)

```python
# In MTPModule.__init__, match existing style:
    plain_config = replace(config,
        engram_enabled=False,
        mhc_enabled=False,
        dsa_enabled=False,
        mamba_enabled=False,
    )
```

---

#### Additional Improvement: fp32 SSM State Accumulation

Extracted from Proposal B's idea. Apply to both scan paths:

```python
# In _ssd_scan_ref, use fp32 for running_state:
    running_state = torch.zeros(B_sz, H, N, D_head,
        device=x.device, dtype=torch.float32)

# In _ssd_step_ref, use fp32 for ssm_state init:
    if "ssm_state" not in states:
        states["ssm_state"] = torch.zeros(B_sz, self.nheads,
            self.d_state, self.headdim,
            device=x.device, dtype=torch.float32)
```

This prevents precision loss during geometric series accumulation
in bf16 over 8 chunks or many decode steps.

---

#### checkpoint_manager.py Compatibility Patching

```python
# Add to _patch_missing_config_keys in checkpoint_manager.py:
    model_config_kwargs.setdefault('mamba_enabled', False)
    model_config_kwargs.setdefault('mamba_pattern', 'A')
    model_config_kwargs.setdefault('mamba_d_state', 64)
    model_config_kwargs.setdefault('mamba_d_conv', 4)
    model_config_kwargs.setdefault('mamba_expand', 2)
    model_config_kwargs.setdefault('mamba_headdim', 128)
    model_config_kwargs.setdefault('mamba_ngroups', 1)
    model_config_kwargs.setdefault('mamba_chunk_size', 256)
    model_config_kwargs.setdefault('mamba3_qknorm', False)
    model_config_kwargs.setdefault('mamba3_bias', False)
    model_config_kwargs.setdefault('mamba3_complex_rope', False)
```

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `nanochat/mamba2.py` | **Create** — Mamba2Layer class |
| `nanochat/gpt.py` | **Modify** — GPTConfig, Block dispatch, init_weights, setup_optimizers, estimate_flops, _compute_window_sizes |
| `nanochat/engine.py` | **Modify** — KVCache (add MambaInferenceParams), move advance() |
| `nanochat/mtp.py` | **Modify** — Add mamba_enabled=False to plain_config |
| `nanochat/checkpoint_manager.py` | **Modify** — Add mamba config defaults to compat patching |
| `base_train.py` | **Modify** — CLI args for mamba |
