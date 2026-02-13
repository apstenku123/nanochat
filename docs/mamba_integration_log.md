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

---

### Iteration 8 Review (Gemini "Ultimate Refinement" Proposal)

**Date**: 2026-02-13
**Source**: Gemini's self-described "final, complete, production-ready" code
**Verdict**: 5 previously-open bugs now FIXED. 6 new bugs found (2 CRITICAL).

#### What iteration 8 FIXED from iteration 7

| Bug | Status | How |
|-----|--------|-----|
| #7 KVCache.prefill type mismatch | ✅ FIXED | Now handles both Tensor and dict via isinstance |
| #8 Batch expansion missing | ✅ FIXED | `.expand(batch_size, ...).clone()` for nested dicts |
| #9 Triton not used in prefill | ✅ FIXED | `return_final_states=True` branch added |
| #10 has_mamba not passed | ✅ FIXED | Shown in both KVCache constructors |
| #11 MTP hasattr guard | ✅ FIXED | Direct `replace(config, mamba_enabled=False)` |

Also correctly incorporated:
- ✅ FP32 accumulation in `_ssd_scan_ref` (chunk loop uses fp32 running_state)
- ✅ FP32 init in `_ssd_step_ref` (new ssm_state initialized as fp32)
- ✅ ndim!=2 filter for Muon optimizer routing
- ✅ `.bias` / `_bias` suffix filter for 2D bias params
- ✅ checkpoint_manager.py compat patching (all 11 mamba fields)
- ✅ CLI args match existing `store_true` / named-param style
- ✅ `estimate_flops` SSD-based formula (no double-count)
- ✅ `kv_cache.advance(T)` centralized in GPT.forward

#### NEW Bugs Found in Iteration 8

##### Bug #12 (CRITICAL): Triton final_states shape transposed vs reference convention

`mamba_chunk_scan_combined` returns `final_states` with shape
**(B, nheads, headdim, d_state)** — this is the official mamba_ssm
convention where the state matrix per head is (headdim × d_state).

Our `_ssd_step_ref` initializes and expects ssm_state as
**(B, nheads, d_state, headdim)** — matching the einsum convention
`'bclhn,bclhd->bchnd'` which produces (B, chunks, H, N, D).

These are **transposed on the last two dims**. When prefill runs on
CUDA (Triton path) and decode uses `_ssd_step_ref`, the stored state
has wrong dimension order. With default config (headdim=128, d_state=64),
this causes a **silent shape mismatch** — the `.copy_()` in the step
function writes values into the wrong positions, producing garbage output.

**Location in proposal**:
```python
# forward() Triton branch stores directly:
states["ssm_state"] = final_states  # shape (B, H, D, N) — WRONG for step

# _ssd_step_ref expects:
ssm_state = states["ssm_state"]     # expects (B, H, N, D)
# Then does:
dBx = (dt * B).unsqueeze(-1) * x.unsqueeze(-2)  # produces (B, H, N, D)
ssm_state.copy_(ssm_state * dA + dBx)           # shape mismatch!
```

**Fix**: Transpose Triton output to match reference convention:
```python
# In forward(), after mamba_chunk_scan_combined with return_final_states:
    y, final_states = mamba_chunk_scan_combined(...)
    # Triton returns (B, H, headdim, d_state), we need (B, H, d_state, headdim)
    states["ssm_state"] = final_states.transpose(-1, -2).contiguous()
```

Alternatively, adopt the Triton convention (H, D, N) everywhere and
adjust `_ssd_scan_ref` and `_ssd_step_ref`. But transposing the Triton
output is a 1-line fix vs rewriting two functions.

**Note**: When headdim == d_state (e.g. both 64), the transpose is a
no-op on shape and the bug is **silent** — the model trains and runs
but with subtly wrong state propagation. This makes it extremely hard
to catch by testing alone.

---

##### Bug #13 (CRITICAL): fp32 state precision lost at prefill→decode boundary

The proposal correctly uses fp32 for state accumulation within each
function, but the precision is **discarded at the boundary**:

**Prefill → decode dtype flow**:

| Path | Prefill stores as | Decode retrieves as | Decode accumulates in |
|------|-------------------|--------------------|-----------------------|
| Triton | bf16 (kernel output) | bf16 | fp32 via `.float()` in copy_ |
| Ref scan | bf16 (`running_state.to(x.dtype)`) | bf16 | fp32 via `.float()` in copy_ |

The problem is in `_ssd_step_ref`:
```python
if "ssm_state" not in states:
    # Only this path creates fp32 state
    states["ssm_state"] = torch.zeros(..., dtype=torch.float32)
ssm_state = states["ssm_state"]
# If state exists from prefill, it's bf16!
# The .copy_() then truncates fp32 computation back to bf16
ssm_state.copy_(ssm_state * dA.float() + dBx.float())
```

After prefill, `"ssm_state"` IS in states (set by prefill path), so the
fp32 initialization is skipped. The existing bf16 tensor is used, and
`.copy_()` silently downcasts the fp32 result to bf16 every step.

**Fix**: Always upcast to fp32 after retrieval, and store as fp32:
```python
if "ssm_state" not in states:
    states["ssm_state"] = torch.zeros(B_sz, self.nheads,
        self.d_state, self.headdim, device=x.device, dtype=torch.float32)
else:
    # Upcast prefill state to fp32 for decode accumulation
    if states["ssm_state"].dtype != torch.float32:
        states["ssm_state"] = states["ssm_state"].float()
ssm_state = states["ssm_state"]
```

Also fix `_ssd_scan_ref` to return fp32 state for inference:
```python
# At end of _ssd_scan_ref:
    return y.to(x.dtype), running_state  # keep running_state as fp32, NOT .to(x.dtype)
```

Combined with Bug #12 fix:
```python
# In forward(), after Triton prefill:
    states["ssm_state"] = final_states.transpose(-1, -2).contiguous().float()
```

---

##### Bug #14 (MODERATE): Block.__init__ self-imports from nanochat.gpt

The proposal adds these imports inside `Block.__init__`:
```python
from nanochat.gpt import CausalSelfAttention
from nanochat.gpt import MLP
```

And in `Block.forward`:
```python
from nanochat.gpt import norm
```

But `Block` IS defined in `gpt.py`. These classes and the `norm()`
function are already in scope — `CausalSelfAttention` is defined at
line 152, `MLP` at line 207, and `norm()` at line 139 of the same file.

The self-imports are:
1. **Redundant** — Python resolves them via `sys.modules`, returning the
   already-loaded module, so they don't crash
2. **Wasteful** — `from X import Y` executes module lookup + attribute
   access on every Block construction (24× during init)
3. **Misleading** — suggests Block might be in a separate file, confusing
   future maintainers

**Fix**: Remove all `from nanochat.gpt import ...` inside Block. Use
`CausalSelfAttention`, `MLP`, and `norm` directly as in the actual code.

The only import that IS correct is `from nanochat.mamba2 import Mamba2Layer`
(cross-module, lazy import to avoid circular dependency).

---

##### Bug #15 (MODERATE): Muon LR scaling regression

**Proposal**:
```python
muon_kwargs = dict(lr=matrix_lr * dmodel_lr_scale, momentum=0.95, ...)
```

**Actual code** (gpt.py setup_optimizers):
```python
muon_kwargs = dict(lr=matrix_lr, momentum=0.95, ...)
```

The `dmodel_lr_scale = (model_dim / 768) ** -0.5` is only applied to
AdamW param groups (lm_head, wte, resid, x0, mamba_adam). Muon's LR
has been tuned experimentally at `matrix_lr=0.02` without this scaling.

Applying the scale factor changes training dynamics:
- At n_embd=768: scale=1.0, no difference
- At n_embd=1536: scale≈0.71, Muon LR drops to 0.014
- At n_embd=3072: scale=0.5, Muon LR drops to 0.01

This could silently degrade training quality at larger model sizes.

**Fix**: Remove `dmodel_lr_scale` from Muon kwargs:
```python
muon_kwargs = dict(lr=matrix_lr, momentum=0.95, weight_decay=weight_decay)
```

---

##### Bug #16 (MODERATE): _compute_window_sizes drops config fields

**Proposal**:
```python
char_to_window = {"L": (config.sequence_len, 0), "S": (config.sequence_len // 2, 0)}
```

**Actual code** (gpt.py:411-438):
```python
long_window = config.sequence_len
short_window = long_window // 2
char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
# Plus: assert char in "SL" validation
```

Two issues:
1. The actual GPTConfig has a separate `short_window` field (not derived
   from sequence_len). The proposal should use `config.short_window` for
   the "S" mapping and `config.sequence_len` for "L".
2. Missing `assert char in "SL"` validation — without it, a typo in
   `window_pattern` silently produces `None` window sizes.

**Fix**:
```python
def _compute_window_sizes(self, config):
    pattern = getattr(config, 'window_pattern', 'L').upper()
    m_pattern = getattr(config, 'mamba_pattern', 'A').upper() if getattr(config, 'mamba_enabled', False) else None

    long_window = config.sequence_len
    short_window = long_window // 2
    char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}

    window_sizes = []
    for layer_idx in range(config.n_layer):
        if m_pattern and m_pattern[layer_idx % len(m_pattern)] == 'M':
            window_sizes.append(None)
        else:
            char = pattern[layer_idx % len(pattern)]
            assert char in "SL", f"Invalid window pattern char '{char}' at position {layer_idx}"
            window_sizes.append(char_to_window[char])

    # Last attention layer always gets full context
    if window_sizes[-1] is None:
        from nanochat.common import print0
        print0(f"WARNING: Last layer ({config.n_layer-1}) is Mamba.")
    else:
        window_sizes[-1] = (long_window, 0)
    return window_sizes
```

---

##### Bug #17 (LOW): init_weights skips Mamba per-head params

The proposal's `init_weights` only initializes `in_proj.weight` and
`out_proj.weight` for Mamba layers:
```python
if getattr(block, "is_mamba", False):
    torch.nn.init.uniform_(block.attn.in_proj.weight, -s, s)
    torch.nn.init.zeros_(block.attn.out_proj.weight)
```

Missing: `conv1d.weight`, `conv1d.bias`, `A_log`, `dt_bias`, `D`,
and optionally `B_bias`/`C_bias`.

Currently these are initialized in `Mamba2Layer.__init__()`:
- `conv1d`: PyTorch default kaiming_uniform (fan_in=d_conv=4)
- `A_log`: `uniform_(1,16)` then `log()`
- `dt_bias`: inverse softplus of log-uniform dt
- `D`: `ones()`

This works because the model is constructed normally (not on meta device).
But if meta-init is ever introduced, constructor init would be lost.

**Improvement** (not a runtime bug today, but defensive):
```python
if getattr(block, "is_mamba", False):
    torch.nn.init.uniform_(block.attn.in_proj.weight, -s, s)
    torch.nn.init.zeros_(block.attn.out_proj.weight)
    # Conv1d: leave at kaiming default (fan_in=4, appropriate for depthwise)
    # A_log, dt_bias, D: re-init from constructor ranges
    A = torch.empty(block.attn.nheads, device=block.attn.A_log.device).uniform_(1, 16)
    block.attn.A_log.data.copy_(torch.log(A))
    dt = torch.exp(torch.rand(block.attn.nheads, device=block.attn.dt_bias.device)
                   * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=0.001)
    block.attn.dt_bias.data.copy_(dt + torch.log(-torch.expm1(-dt)))
    block.attn.D.data.fill_(1.0)
```

---

#### Cumulative Bug Status Table (after iteration 8)

| # | Severity | Description | Introduced | Fixed? |
|---|----------|-------------|-----------|--------|
| 1 | CRITICAL | F.pad 8-val on 4D pads batch dim | Iter 5 | ✅ Iter 7 |
| 2 | CRITICAL | kv_cache.advance() never called (last layer Mamba) | Iter 5 | ✅ Iter 7 |
| 3 | CRITICAL | init_weights skips DSA c_q/c_k/c_v/c_proj | Iter 5 | ✅ Iter 7 |
| 4 | MODERATE | estimate_flops double-counts + wrong units | Iter 6 | ✅ Iter 7 |
| 5 | MODERATE | Complex RoPE angle lost at prefill→decode | Iter 5 | ✅ Iter 7 |
| 6 | MODERATE | Missing Block.__init__, KVCache, CLI, GPTConfig | Iter 5 | ✅ Iter 7 |
| 7 | CRITICAL | KVCache.prefill() mamba state type mismatch | Iter 7 | ✅ Iter 8 |
| 8 | CRITICAL | Mamba state batch expansion missing in prefill | Iter 7 | ✅ Iter 8 |
| 9 | MODERATE | Triton kernel not used during inference prefill | Iter 7 | ✅ Iter 8 |
| 10 | MODERATE | Engine.generate KVCache constructors not shown | Iter 7 | ✅ Iter 8 |
| 11 | LOW | MTP hasattr guard unnecessary | Iter 7 | ✅ Iter 8 |
| 12 | **CRITICAL** | **Triton final_states shape (H,D,N) vs ref (H,N,D)** | **Iter 8** | **NO** |
| 13 | **CRITICAL** | **fp32 state lost at prefill→decode boundary** | **Iter 8** | **NO** |
| 14 | MODERATE | Block.__init__ self-imports from nanochat.gpt | Iter 8 | **NO** |
| 15 | MODERATE | Muon LR incorrectly scaled by dmodel_lr_scale | Iter 8 | **NO** |
| 16 | MODERATE | _compute_window_sizes drops short_window config | Iter 8 | **NO** |
| 17 | LOW | init_weights skips conv1d/A_log/dt_bias/D | Iter 8 | **NO** |

Bugs #1-11 from iterations 5-7 are all **FIXED** in iteration 8.
Bugs #12-13 are **CRITICAL** new issues in the Triton↔reference boundary.
Bugs #14-16 are moderate regressions vs actual code.
Bug #17 is a defensive improvement.

---

#### Corrected Code for All 6 Open Bugs (#12-#17)

##### Bug #12 + #13 Combined Fix: Triton shape + fp32 boundary

```python
# In Mamba2Layer.forward(), Triton prefill branch:
    if mamba_chunk_scan_combined is not None and x.device.type == "cuda":
        if inference_params is not None:
            y, final_states = mamba_chunk_scan_combined(
                x_ssm, dt_soft, A, B_ssm, C_ssm,
                chunk_size=self.chunk_size, D=self.D,
                return_final_states=True,
            )
            states = inference_params.key_value_memory_dict.setdefault(self.layer_idx, {})
            # BUG 12: Transpose from Triton (B,H,D,N) to our (B,H,N,D)
            # BUG 13: Cast to fp32 for decode accumulation precision
            states["ssm_state"] = final_states.transpose(-1, -2).contiguous().float()
        else:
            y = mamba_chunk_scan_combined(
                x_ssm, dt_soft, A, B_ssm, C_ssm,
                chunk_size=self.chunk_size, D=self.D,
            )
    else:
        y, final_states = self._ssd_scan_ref(x_ssm, dt_soft, A, B_ssm, C_ssm, self.D)
        if inference_params is not None:
            states = inference_params.key_value_memory_dict.setdefault(self.layer_idx, {})
            # BUG 13: Keep fp32 — _ssd_scan_ref already returns fp32 running_state
            states["ssm_state"] = final_states

# In _ssd_scan_ref, return fp32 running_state WITHOUT downcasting:
    return y.to(x.dtype), running_state  # NOT running_state.to(x.dtype)

# In _ssd_step_ref, upcast existing state if needed:
    if "ssm_state" not in states:
        states["ssm_state"] = torch.zeros(B_sz, self.nheads,
            self.d_state, self.headdim, device=x.device, dtype=torch.float32)
    ssm_state = states["ssm_state"]
    if ssm_state.dtype != torch.float32:
        states["ssm_state"] = ssm_state.float()
        ssm_state = states["ssm_state"]
```

##### Bug #14 Fix: Remove self-imports from Block

```python
# Block.__init__ — use classes directly (they're in the same file):
class Block(nn.Module):
    def __init__(self, config, layer_idx, engram_layers):
        super().__init__()
        self.layer_idx = layer_idx

        m_pattern = getattr(config, 'mamba_pattern', 'A').upper() if getattr(config, 'mamba_enabled', False) else "A"
        self.is_mamba = m_pattern[layer_idx % len(m_pattern)] == 'M'
        use_dsa = bool(getattr(config, 'dsa_enabled', False)) and layer_idx >= getattr(config, 'dsa_start_layer', 7)

        if self.is_mamba:
            from nanochat.mamba2 import Mamba2Layer  # ONLY cross-module import needed
            self.attn = Mamba2Layer(config, layer_idx)
        elif use_dsa:
            self.attn = DeepSeekSparseAttention(  # already imported at top of gpt.py
                config, layer_idx,
                dsa_top_k_ratio=config.dsa_top_k_ratio,
                dsa_local_window=config.dsa_local_window,
                dsa_indexer_heads=config.dsa_indexer_heads,
                dsa_indexer_dim=config.dsa_indexer_dim
            )
        else:
            self.attn = CausalSelfAttention(config, layer_idx)  # same file

        self.mlp = MLP(config)  # same file
        # ... engram/mhc unchanged ...

    def forward(self, x, cos_sin, window_size, kv_cache):
        x_attn = x + self.attn(norm(x), cos_sin, window_size, kv_cache)  # norm() is module-level
        baseline_out = x_attn + self.mlp(norm(x_attn))
        # ... engram/mhc unchanged ...
```

##### Bug #15 Fix: Muon LR without scaling

```python
# In setup_optimizers:
muon_kwargs = dict(lr=matrix_lr, momentum=0.95, weight_decay=weight_decay)
# NOT: lr=matrix_lr * dmodel_lr_scale
```

##### Bug #16 Fix: Use config fields for window sizes

```python
# In _compute_window_sizes:
    long_window = config.sequence_len
    short_window = long_window // 2
    char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
    # ...
    char = pattern[layer_idx % len(pattern)]
    assert char in "SL", f"Invalid window pattern char '{char}'"
    window_sizes.append(char_to_window[char])
```

##### Bug #17 Fix: Add Mamba per-head param init

```python
# In init_weights, inside the mamba block branch:
if getattr(block, "is_mamba", False):
    torch.nn.init.uniform_(block.attn.in_proj.weight, -s, s)
    torch.nn.init.zeros_(block.attn.out_proj.weight)
    # Defensive: re-init per-head params (survives meta-init)
    A = torch.empty(block.attn.nheads, device=block.attn.A_log.device).uniform_(1, 16)
    block.attn.A_log.data.copy_(torch.log(A))
    dt = torch.exp(torch.rand(block.attn.nheads, device=block.attn.dt_bias.device)
                   * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=0.001)
    block.attn.dt_bias.data.copy_(dt + torch.log(-torch.expm1(-dt)))
    block.attn.D.data.fill_(1.0)
    # conv1d: leave at PyTorch kaiming default (fan_in=d_conv=4, appropriate for depthwise)
```

---

---

### Iteration 9: Skipped (identical to iteration 8, same 6 open bugs)

---

### Iteration 10 Review

**Date**: 2026-02-13
**Source**: Gemini "exceptional review process" proposal
**Verdict**: 3 of 6 open bugs FIXED. Bug #13 partially fixed. 2 remaining
issues. 1 new regression. **Closest to implementation-ready yet.**

#### Status of Previously-Open Bugs (#12-17)

**Bug #12 (Triton shape transpose)** — ✅ FIXED. Code has:
```python
states["ssm_state"] = final_states.transpose(-1, -2).to(torch.float32)
```

**Bug #13 (fp32 state at prefill→decode boundary)** — PARTIALLY FIXED.
Two changes:

1. Triton path: `.transpose(-1,-2).to(torch.float32)` — ✅ stores fp32
2. Ref scan path: `states["ssm_state"] = final_states` — now returns
   `running_state.to(torch.float32)` at end of `_ssd_scan_ref` — ✅ FIXED

But `_ssd_step_ref` still only creates fp32 if state doesn't exist:
```python
if "ssm_state" not in states:
    states["ssm_state"] = torch.zeros(..., dtype=torch.float32)
```
Since both prefill paths now store fp32, this is fine in normal flow.
However, if the official `mamba_ssm.InferenceParams` is used instead of
our `MambaInferenceParams`, the dtype depends on what their code stores.
**Defensive upcast still recommended** but no longer a runtime crash.

**Bug #14 (Block self-imports)** — ✅ FIXED. No self-imports from
`nanochat.gpt`. Uses `CausalSelfAttention`, `MLP`, `norm` directly.
Only cross-module `from nanochat.mamba2 import Mamba2Layer` remains.
`from nanochat.sparse_attention import DeepSeekSparseAttention` is
present but DSA is already imported at top of gpt.py — redundant but
harmless since Python caches module imports.

**Bug #15 (Muon LR scaling)** — ✅ FIXED. Now uses:
```python
muon_kwargs = dict(lr=matrix_lr, momentum=0.95, weight_decay=weight_decay)
```
No `dmodel_lr_scale` applied to Muon. Matches actual code.

**Bug #16 (_compute_window_sizes)** — STILL PRESENT (LOW severity now).
Still uses `long_window // 2` for short windows. Matches actual code
pattern so not a regression — both derive from sequence_len.

**Bug #17 (init_weights skips per-head params)** — STILL PRESENT (LOW).
Only initializes `in_proj.weight` and `out_proj.weight`. Works today
since constructor handles A_log/dt_bias/D.

#### NEW Issues in Iteration 10

##### Bug #18 (MODERATE): GPT.forward has self-import of norm

```python
def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
    ...
    x = self.transformer.wte(idx)
    from nanochat.gpt import norm    # ◄◄◄ UNNECESSARY
    x = norm(x)
```

`norm()` is a module-level function defined at gpt.py:139. It's already
in scope inside `GPT.forward()` since GPT is defined in the same file.
This `from nanochat.gpt import norm` is the same self-import pattern
from Bug #14 — redundant and wasteful (executes on every forward pass).

**Fix**: Remove `from nanochat.gpt import norm`. Use `norm(x)` directly
as in the actual code.

##### Bug #19 (LOW): GPT.forward uses `self.config.mtp_lambda` instead of `self.mtp_lambda`

```python
return main_loss + getattr(self.config, "mtp_lambda", 0.3) * mtp_loss
```

The actual code at gpt.py:591 uses `self.mtp_lambda` (set in `GPT.__init__`
from `config.mtp_lambda`). This is a cosmetic difference — both resolve
to the same value — but accessing via config on every forward is slightly
less clean than the instance attribute.

#### What iteration 10 got RIGHT (improvements over iter 8/9)

1. ✅ `_ssd_scan_ref` returns `running_state.to(torch.float32)` — keeps
   fp32 for inference state. This FIXES the core of Bug #13.
2. ✅ Muon LR no longer scaled (Bug #15 fixed)
3. ✅ No self-imports in Block.__init__ (Bug #14 fixed for Block)
4. ✅ DSA `_full_attention` shown with advance() removed — previously
   only mentioned, now explicit code provided
5. ✅ `GPT.forward` shows full output logic (softcap, MTP, inference)
   instead of `# ... standard output matching ...` truncation
6. ✅ `from nanochat.sparse_attention import DeepSeekSparseAttention`
   inside Block is acceptable as lazy import (though redundant with
   top-of-file import, it doesn't cause issues)

#### Cumulative Bug Status Table (after iteration 10)

| # | Severity | Description | Introduced | Fixed? |
|---|----------|-------------|-----------|--------|
| 1 | CRITICAL | F.pad 8-val on 4D pads batch dim | Iter 5 | ✅ Iter 7 |
| 2 | CRITICAL | kv_cache.advance() never called (last layer Mamba) | Iter 5 | ✅ Iter 7 |
| 3 | CRITICAL | init_weights skips DSA c_q/c_k/c_v/c_proj | Iter 5 | ✅ Iter 7 |
| 4 | MODERATE | estimate_flops double-counts + wrong units | Iter 6 | ✅ Iter 7 |
| 5 | MODERATE | Complex RoPE angle lost at prefill→decode | Iter 5 | ✅ Iter 7 |
| 6 | MODERATE | Missing Block.__init__, KVCache, CLI, GPTConfig | Iter 5 | ✅ Iter 7 |
| 7 | CRITICAL | KVCache.prefill() mamba state type mismatch | Iter 7 | ✅ Iter 8 |
| 8 | CRITICAL | Mamba state batch expansion missing in prefill | Iter 7 | ✅ Iter 8 |
| 9 | MODERATE | Triton kernel not used during inference prefill | Iter 7 | ✅ Iter 8 |
| 10 | MODERATE | Engine.generate KVCache constructors not shown | Iter 7 | ✅ Iter 8 |
| 11 | LOW | MTP hasattr guard unnecessary | Iter 7 | ✅ Iter 8 |
| 12 | CRITICAL | Triton final_states shape (H,D,N) vs ref (H,N,D) | Iter 8 | ✅ Iter 10 |
| 13 | CRITICAL | fp32 state lost at prefill→decode boundary | Iter 8 | ✅ Iter 10 |
| 14 | MODERATE | Block.__init__ self-imports from nanochat.gpt | Iter 8 | ✅ Iter 10 |
| 15 | MODERATE | Muon LR incorrectly scaled by dmodel_lr_scale | Iter 8 | ✅ Iter 10 |
| 16 | LOW | _compute_window_sizes hardcodes short_window | Iter 8 | matches actual |
| 17 | LOW | init_weights skips conv1d/A_log/dt_bias/D | Iter 8 | defensive only |
| 18 | **MODERATE** | **GPT.forward self-import of norm** | **Iter 10** | **NO** |
| 19 | LOW | GPT.forward uses config.mtp_lambda vs self.mtp_lambda | Iter 10 | cosmetic |

**All CRITICAL bugs are now FIXED.** Remaining issues are moderate/low.

#### Implementation Readiness Assessment

This proposal is **ready for implementation** with these minor fixes
applied during coding:

1. Remove `from nanochat.gpt import norm` from GPT.forward (Bug #18)
2. Use `self.mtp_lambda` instead of `getattr(self.config, "mtp_lambda", 0.3)` (Bug #19)
3. Add defensive fp32 upcast in `_ssd_step_ref` for robustness (Bug #13 hardening):
   ```python
   ssm_state = states["ssm_state"]
   if ssm_state.dtype != torch.float32:
       states["ssm_state"] = ssm_state.float()
       ssm_state = states["ssm_state"]
   ```
4. Optionally add Mamba per-head param re-init in init_weights (Bug #17)

These are all trivial 1-3 line fixes that can be applied during
implementation without needing another review cycle.

---

---

### Alternative Proposal Review: Ideas to Merge

**Date**: 2026-02-13
**Source**: Independent review proposing Phase 1-3 implementation with
XLA scan, trapezoidal discretization, and Nemotron-style sliding windows

**Overall verdict**: DO NOT ADOPT as a whole (violates too many codebase
conventions — different GPTConfig fields, different init style, different
Block structure, RMSNorm as nn.Module, different optimizer API). But
contains **6 valuable ideas** to extract and merge.

---

#### Idea A: XLA scan for TPU training (ADOPT — Phase 1)

The alternative proposal uses `torch_xla.experimental.scan` to replace
the Python loop in `_ssd_scan_ref`. This is significant for TPU:

Our current `_ssd_scan_ref` has a Python loop over `nchunks ≈ 8`, which
is fine for CUDA (Triton handles it), but on XLA the loop prevents
torch_xla from fusing the entire scan into one HLO program.

**What to add** (new method in Mamba2Layer):
```python
def _ssd_scan_xla(self, x, dt, A, B, C, D):
    """XLA-optimized scan using torch_xla.experimental.scan."""
    try:
        from torch_xla.experimental.scan import scan as xla_scan
    except ImportError:
        return self._ssd_scan_ref(x, dt, A, B, C, D)

    # Use xla_scan with (state, input) -> (state, output) body
    # Body processes one chunk at a time (not one token)
    # This lets XLA compile the entire scan into a single While loop
    ...
```

**Key insight**: The proposal's O(T) Python loop version is WRONG for
training (too slow), but the idea of using `xla_scan` with our chunked
scan body (processing one chunk per iteration, ~8 iterations) is sound.
We should add this as a third compute path alongside Triton and ref scan.

**Decision**: Add `_ssd_scan_xla` method that wraps our existing chunked
scan logic inside `xla_scan`. Defer to Phase 1 implementation.

---

#### Idea B: `mamba3_trapezoidal` config toggle + lambda projection (ADOPT — Phase 3 prep)

The proposal adds `mamba3_trapezoidal: bool = False` as a GPTConfig field
and extends `d_in_proj` by `nheads` when enabled (for the lambda gate):

```python
d_lambda = self.nheads if self.mamba3_trapezoidal else 0
d_in_proj = 2 * d_inner + 2 * ngroups * d_state + nheads + d_lambda
```

This is exactly what our Phase 3 design spec requires. Adding the config
field and in_proj width change NOW (even if trapezoidal scan isn't
implemented yet) means:
1. Checkpoints save the toggle
2. Parameter count is correct when enabled
3. The lambda projection weights exist for future training

**What to add to GPTConfig**:
```python
mamba3_trapezoidal: bool = False
```

**What to add to Mamba2Layer.__init__**:
```python
d_lambda = self.nheads if self.mamba3_trapezoidal else 0
d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads + d_lambda
```

**What to add to forward() projection split**:
```python
if self.mamba3_trapezoidal:
    dt_raw, lam_raw = torch.split(tail, [self.nheads, self.nheads], dim=-1)
    lam = torch.sigmoid(lam_raw.to(torch.float32))
else:
    dt_raw = tail
    lam = None
```

**Decision**: Add config field and in_proj sizing in Phase 1. Trapezoidal
scan body deferred to Phase 3. Also add to checkpoint_manager defaults.

---

#### Idea C: `window_long` / `window_short` explicit config fields (ADOPT)

The proposal introduces `window_long: int = 0` and `window_short: int = 0`
as explicit GPTConfig fields, decoupling window sizes from `sequence_len`:

```python
window_long: int = 0   # 0 => use sequence_len (backward compatible)
window_short: int = 0  # 0 => use window_long//2 (backward compatible)
```

This solves our Bug #16 properly — instead of always deriving short
window from `sequence_len // 2`, users can set explicit window sizes.
The `0` default preserves backward compatibility.

**What to add to GPTConfig**:
```python
window_long: int = 0    # 0 = use sequence_len, -1 = full context
window_short: int = 0   # 0 = window_long // 2
```

**What to change in _compute_window_sizes**:
```python
long_window = config.window_long if config.window_long != 0 else config.sequence_len
short_window = config.window_short if config.window_short != 0 else long_window // 2
```

**Decision**: Adopt. Cleaner than hardcoding `sequence_len // 2`. Add to
checkpoint_manager defaults too.

---

#### Idea D: `scan_layers` safety check for hybrid blocks (ADOPT)

The proposal identifies that `torch_xla.experimental.scan_layers`
assumes homogeneous blocks (same compiled body). With hybrid A+M blocks,
`scan_layers` would compile attention and Mamba into the same trace,
which fails because they have different parameter shapes.

**What to add to base_train.py** (where `scan_layers` is called):
```python
if args.use_scan and device_type == 'xla':
    pat = getattr(config, 'mamba_pattern', 'A').upper()
    is_hybrid = getattr(config, 'mamba_enabled', False) and 'A' in pat and 'M' in pat
    if is_hybrid:
        print0("WARNING: scan_layers disabled for hybrid attention+mamba stack")
    else:
        model.transformer.h = scan_layers(model.transformer.h)
```

**Decision**: Adopt. One-line guard prevents a hard-to-debug XLA crash.

---

#### Idea E: `rope_theta` config field for complex RoPE base (ADOPT)

The proposal parameterizes the complex RoPE theta via config:
```python
rope_theta = float(getattr(config, "rope_theta", 10000.0))
inv_freq = 1.0 / (rope_theta ** (arange(0, d_state, 2) / d_state))
```

Our current implementation hardcodes `10000` in `Mamba2Layer.__init__`.
Making it configurable via `rope_theta` (already present in GPTConfig
for attention RoPE) allows tuning for longer contexts.

**What to change in Mamba2Layer.__init__**:
```python
rope_theta = float(getattr(config, 'rope_theta', 10000.0))
inv_freq = 1.0 / (rope_theta ** (torch.arange(0, self.d_state, 2).float() / self.d_state))
```

**Decision**: Adopt. Trivial change, enables 1M+ context experiments.

---

#### Idea F: `prev_B`/`prev_x` state initialization in `_get_or_create_state_dict` (ADOPT — Phase 3)

The proposal's state dict creation handles trapezoidal states cleanly:
```python
if self.mamba3_trapezoidal:
    if "prev_B" not in states:
        states["prev_B"] = torch.zeros(B, ngroups, d_state, ...)
    if "prev_x" not in states:
        states["prev_x"] = torch.zeros(B, nheads, headdim, ...)
```

This matches our Phase 3 spec from the Idea 5 section (iteration 7
Proposal B ideas). The proposal also correctly stores `prev_B` in fp32
and `prev_x` in input dtype, which matches our Strategy B analysis
(4.5KB/layer vs 192KB/layer for the precomputed product approach).

**Decision**: Adopt the state initialization pattern when implementing
Phase 3 trapezoidal scan.

---

#### Rejected Ideas (DO NOT ADOPT)

| Idea | Why rejected |
|------|-------------|
| Different GPTConfig fields | Uses `window_long/short`, `rope_base`, different field names. Our config has `window_pattern`, `sequence_len` derivation. Keep ours but adopt window_long/short. |
| `RMSNorm` as `nn.Module` | Codebase uses functional `F.rms_norm` everywhere (`norm()` at gpt.py:139). Adding `nn.Module` RMSNorm creates orphan learnable params. |
| Different `Block.__init__` structure | Uses `self.ln_1 = RMSNorm(...)` (not in actual code). Our blocks use `norm()` inline in forward. |
| O(T) Python loop fallback | `_ssm_scan_python_mamba2` loops over T=2048. Our `_ssd_scan_ref` does chunked scan over ~8 iterations. 256x slower. |
| Different `init_weights` scale | Uses `std=0.02` hardcoded. Actual code uses `s = sqrt(3) * n_embd^-0.5`. |
| Different optimizer API | Uses `muon.Muon` with `nesterov=True, backend="newtonschulz5"`. Our code uses `nanochat.muon.Muon` with `momentum=0.95`. |
| `Engram`/`MultiHeadContext` class names | Actual classes are `EngramBranch` and `ManifoldBranchMixer`. |
| Removing RoPE from attention layers | Nemotron-style "no RoPE in attention" is a fundamental architecture change, not a Mamba integration detail. |
| Different KVCache API | Uses `kv_cache.update(k, v)` instead of `flash_attn_with_kvcache`. |
| `checkpoint` import in forward | Uses `from torch.utils.checkpoint import checkpoint` inline. Actual code has the same function but different import path. |

---

#### Summary of Mergeable Ideas

| # | Idea | Phase | Effort |
|---|------|-------|--------|
| A | XLA scan for TPU training | 1 (impl) | Medium — new method in Mamba2Layer |
| B | `mamba3_trapezoidal` config + lambda in_proj | 1 (config) / 3 (scan) | Low — config field + in_proj math |
| C | `window_long`/`window_short` config fields | 1 | Low — 2 GPTConfig fields + _compute fix |
| D | `scan_layers` safety for hybrid blocks | 1 | Trivial — 5 lines in base_train.py |
| E | `rope_theta` for complex RoPE base | 1 | Trivial — 1 line in Mamba2Layer |
| F | `prev_B`/`prev_x` state init pattern | 3 | Low — state dict creation |

---

---

### Iteration 11: Gemini "Absolute Finish Line" Proposal

**Date**: 2026-02-13
**Verdict**: Fixes Bug #17 (meta-init) and Bug #18 (norm self-import).
Still has Bug #14 (self-imports in Block.__init__) and reintroduces
`import math` inside `init_weights`. Otherwise clean — ready for
implementation with the 6 ideas from the alternative proposal.

#### What iteration 11 FIXED from iteration 10

| Bug | Status | How |
|-----|--------|-----|
| #17 | ✅ FIXED | `init_weights` now re-inits A_log/dt_bias/D/B_bias/C_bias under `with torch.no_grad()` for meta-init safety |
| #18 | ✅ FIXED | `GPT.forward` no longer has `from nanochat.gpt import norm` — uses `norm(x)` directly |
| #19 | ✅ FIXED | Uses `self.mtp_lambda` instead of `getattr(self.config, "mtp_lambda", 0.3)` |
| #13 (defensive) | ✅ ADDED | `_ssd_step_ref` now has `if ssm_state.dtype != torch.float32: ssm_state = ssm_state.float()` |

#### Remaining issues in iteration 11

| Bug | Severity | Description |
|-----|----------|-------------|
| #14 | LOW | Block.__init__ still imports `from nanochat.gpt import CausalSelfAttention` and `from nanochat.gpt import MLP` — these are self-imports (Block IS in gpt.py). Fix during implementation: remove these lines. |
| NEW | TRIVIAL | `import math` added inside `init_weights` body — should be a module-level import (already present at top of gpt.py). |

#### Assessment: IMPLEMENTATION READY

All critical and moderate bugs are resolved. The only remaining issues
are cosmetic (self-imports, misplaced import statement) that take 2
seconds to fix during implementation.

---

### 6 Ideas to Adopt from Alternative Proposal (Detailed)

These ideas address real gaps in the current design. Each is described
with full production code, exact file locations, integration context,
and rationale. The next iteration should incorporate all of these into
the proposed code.

---

#### Idea A: XLA Scan for TPU Training

**Problem**: Our `_ssd_scan_ref` uses a chunked approach with a Python
`for c in range(nchunks)` loop (~8 iterations for L=2048, cs=256).
On TPU/XLA, each Python loop iteration generates a separate HLO
computation graph. The XLA compiler cannot fuse these into one While
loop, which increases compile time, prevents cross-iteration
optimization, and wastes HBM due to duplicated intermediate buffers.

**Impact**: For 8 Mamba layers x 8 chunks = 64 separate HLO bodies
compiled vs 8 (one per layer). Compilation overhead ~8x higher.

**Solution**: Add an alternative scan path that uses
`torch_xla.experimental.scan` — XLA's native scan primitive that
compiles the entire recurrence body once and executes it T times in a
single HLO While loop. This is NOT a replacement for the chunked scan
(which is more memory-efficient for training); it's an alternative
for when XLA is available.

**Where**: `nanochat/mamba2.py` — 3 changes:

**Change 1**: Add import at the top of `mamba2.py`:
```python
# At top of mamba2.py, after the mamba_ssm import:
try:
    from torch_xla.experimental.scan import scan as xla_scan
    _HAVE_XLA_SCAN = True
except ImportError:
    xla_scan = None
    _HAVE_XLA_SCAN = False
```

**Change 2**: Add new method `_ssd_scan_xla` to `Mamba2Layer`:
```python
def _ssd_scan_xla(self, x_ssm, dt, A, B_ssm, C_ssm, D, init_state):
    """
    SSM recurrence using torch_xla.experimental.scan.

    Unlike _ssd_scan_ref which chunks the sequence and loops over chunks,
    this scans token-by-token using XLA's native While loop. XLA compiles
    the body function once and executes it T times efficiently.

    Args:
        x_ssm: (B, T, H, D_head) input after conv+SiLU
        dt: (B, T, H) softplus'd timestep
        A: (H,) negative diagonal (float32)
        B_ssm: (B, T, G, N) input projection B
        C_ssm: (B, T, G, N) output projection C
        D: (H,) skip connection
        init_state: (B, H, N, D_head) initial SSM state (float32)

    Returns:
        y: (B, T, H, D_head) output
        final_state: (B, H, N, D_head) final SSM state (float32)
    """
    B_sz, T, H, D_head = x_ssm.shape
    G = self.ngroups
    Hpg = self.nheads // G
    N = self.d_state

    # Reshape to grouped form — avoids repeat_interleave inside the scan
    # by computing per-group and broadcasting across heads_per_group
    x_g = x_ssm.view(B_sz, T, G, Hpg, D_head)
    dt_g = dt.view(B_sz, T, G, Hpg)
    A_g = A.view(G, Hpg)
    D_g = D.view(G, Hpg)
    state0 = init_state.view(B_sz, G, Hpg, N, D_head).float()

    # XLA scan requires leading dimension = scan dimension (T)
    # All inputs must be time-major: (T, B, ...)
    xs = (
        x_g.transpose(0, 1).float(),    # (T, B, G, Hpg, D)
        dt_g.transpose(0, 1).float(),   # (T, B, G, Hpg)
        B_ssm.transpose(0, 1).float(),  # (T, B, G, N)
        C_ssm.transpose(0, 1).float(),  # (T, B, G, N)
    )

    def body(carry, inp):
        """Single-step SSM recurrence. XLA compiles this once."""
        x_t, dt_t, B_t, C_t = inp
        # carry: (B, G, Hpg, N, D) float32

        # Decay: alpha = exp(dt * A) per head
        alpha = torch.exp(dt_t * A_g.view(1, G, Hpg))  # (B, G, Hpg)

        # Input term: dt * B * x → outer product over (N, D)
        dtB = dt_t.unsqueeze(-1) * B_t.unsqueeze(-2)       # (B, G, Hpg, N)
        dBx = dtB.unsqueeze(-1) * x_t.unsqueeze(-2)        # (B, G, Hpg, N, D)

        # State update: h = alpha * h + dt * B * x
        new_carry = carry * alpha.unsqueeze(-1).unsqueeze(-1) + dBx

        # Output: y = C * h + D * x
        y_t = (C_t.unsqueeze(-2).unsqueeze(-1) * new_carry).sum(dim=-2)  # (B, G, Hpg, D)
        y_t = y_t + D_g.view(1, G, Hpg, 1) * x_t

        return new_carry, y_t

    # xla_scan(fn, init, xs) → (final_carry, stacked_outputs)
    # Scans over the leading dimension of xs (T)
    stateT, ys = xla_scan(body, state0, xs)

    # Reshape back to (B, T, H, D) and (B, H, N, D)
    y = ys.transpose(0, 1).reshape(B_sz, T, H, D_head)
    final_state = stateT.reshape(B_sz, H, N, D_head)

    return y.to(x_ssm.dtype), final_state
```

**Change 3**: Modify the dispatch in `Mamba2Layer.forward()`. Replace
the current two-way branch (Triton vs chunked-ref) with a three-way:
```python
        # In forward(), replace the scan dispatch block:
        A = -torch.exp(self.A_log)

        # Compute initial state for inference prefill
        if inference_params is not None:
            states = inference_params.key_value_memory_dict.setdefault(self.layer_idx, {})
            if "ssm_state" in states:
                init_state = states["ssm_state"]
                # Ensure fp32 for accumulation
                if init_state.dtype != torch.float32:
                    init_state = init_state.float()
            else:
                init_state = torch.zeros(B_sz, self.nheads, self.d_state, self.headdim,
                                         device=x.device, dtype=torch.float32)
        else:
            init_state = torch.zeros(B_sz, self.nheads, self.d_state, self.headdim,
                                     device=x.device, dtype=torch.float32)

        # Three-way dispatch: XLA scan > Triton kernel > chunked reference
        use_xla = (x.device.type == 'xla') and _HAVE_XLA_SCAN
        use_triton = (mamba_chunk_scan_combined is not None) and (x.device.type == "cuda")

        if use_xla:
            y, final_states = self._ssd_scan_xla(
                x_ssm, dt_soft, A, B_ssm, C_ssm, self.D, init_state,
            )
        elif use_triton:
            if inference_params is not None:
                y, final_states_raw = mamba_chunk_scan_combined(
                    x_ssm, dt_soft, A, B_ssm, C_ssm,
                    chunk_size=self.chunk_size, D=self.D, return_final_states=True,
                )
                # Triton returns (B,H,headdim,d_state) — transpose to our (B,H,d_state,headdim)
                final_states = final_states_raw.transpose(-1, -2).to(torch.float32)
            else:
                y = mamba_chunk_scan_combined(
                    x_ssm, dt_soft, A, B_ssm, C_ssm,
                    chunk_size=self.chunk_size, D=self.D,
                )
                final_states = None
        else:
            y, final_states = self._ssd_scan_ref(x_ssm, dt_soft, A, B_ssm, C_ssm, self.D)

        # Store final state for inference
        if inference_params is not None and final_states is not None:
            states["ssm_state"] = final_states.to(torch.float32)
```

**Why this matters**: On TPU v6e x8, the Python loop in `_ssd_scan_ref`
forces XLA to compile 8 separate HLO bodies per Mamba layer. With
`xla_scan`, it compiles 1 body and executes it as a native While loop.
Expected compile time reduction: ~4-8x. Training throughput improvement:
~15-30% for the Mamba layer portion (Mamba is ~8/24 = 33% of layers,
so overall improvement ~5-10%).

**Tradeoff**: The XLA scan operates token-by-token (O(T) sequential
steps) while the chunked ref scan operates chunk-by-chunk (O(T/cs)
steps but each step is a matmul over cs tokens). For training with
cs=256, the chunked approach is actually more computationally efficient
because it parallelizes within each chunk. The XLA scan's advantage
is purely compilation efficiency. For Phase 1, keep both paths and
let the dispatch choose based on device type.

---

#### Idea B: `mamba3_trapezoidal` Config Field + Lambda Projection

**Problem**: Phase 3 trapezoidal discretization changes the `in_proj`
output width — it adds `nheads` extra dimensions for the λ (lambda)
gate that interpolates between forward and backward Euler. If we train
a model without this field and later enable trapezoidal, the saved
checkpoint's `in_proj.weight` has shape `(d_model, old_width)` which
doesn't match `(d_model, old_width + nheads)`. The model won't load.

**Solution**: Add the config field and conditional `in_proj` width
computation NOW, defaulting to `False`. When `False`, the width is
identical to current code. When `True`, it adds `nheads` dims.

**Where**: 4 files need changes:

**File 1 — `nanochat/gpt.py` GPTConfig**:
```python
@dataclass
class GPTConfig:
    # ... existing fields ...
    mamba3_qknorm: bool = False
    mamba3_bias: bool = False
    mamba3_complex_rope: bool = False
    mamba3_trapezoidal: bool = False   # ← ADD THIS
```

**File 2 — `nanochat/mamba2.py` Mamba2Layer.__init__**:
```python
    def __init__(self, config, layer_idx):
        # ... existing setup ...

        # Phase 3 toggle
        self.mamba3_trapezoidal = getattr(config, 'mamba3_trapezoidal', False)

        # in_proj width: z + xBC + dt + (optional) lambda
        d_lambda = self.nheads if self.mamba3_trapezoidal else 0
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads + d_lambda
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False)

        # ... rest unchanged ...
```

And in `forward()`, the projection split changes:
```python
        zxbcdt = self.in_proj(x)
        z = zxbcdt[..., :self.d_inner]
        xBC_raw = zxbcdt[..., self.d_inner : self.d_inner + self.d_inner + 2*self.ngroups*self.d_state]

        if self.mamba3_trapezoidal:
            dt = zxbcdt[..., -(self.nheads + self.nheads) : -self.nheads]
            lam_raw = zxbcdt[..., -self.nheads:]
            lam = torch.sigmoid(lam_raw.float())  # (B, L, H) in [0, 1]
        else:
            dt = zxbcdt[..., -self.nheads:]
            lam = None
```

**File 3 — `nanochat/checkpoint_manager.py`**:
```python
    model_config_kwargs.setdefault('mamba3_trapezoidal', False)
```

**File 4 — `scripts/base_train.py`**:
```python
    parser.add_argument("--mamba3_trapezoidal", action="store_true",
                        help="enable Mamba-3 trapezoidal discretization (Phase 3)")

    # In model_config_kwargs:
    mamba3_trapezoidal=args.mamba3_trapezoidal,
```

**Why now vs later**: Adding the config field + conditional in_proj
width costs zero runtime when disabled. But it means ALL checkpoints
saved from Phase 1 forward will have the field, so enabling trapezoidal
later is a non-breaking config change rather than a checkpoint migration.

---

#### Idea C: `window_long` / `window_short` Config Fields

**Problem**: Window sizes are currently derived from `sequence_len`:
```python
# Current code in _compute_window_sizes:
long_window = config.sequence_len      # e.g. 2048
short_window = long_window // 2        # e.g. 1024
```

This has three problems:
1. Training at seq_len=65536 but wanting 128k windows — impossible
2. Inference at different lengths than training — wrong window size
3. Full context (`-1` in flash-attn) — can't express it

**Solution**: Two new config fields with backward-compatible defaults.

**File 1 — `nanochat/gpt.py` GPTConfig**:
```python
@dataclass
class GPTConfig:
    # ... existing fields ...
    window_pattern: str = "L"
    window_long: int = 0    # 0 = use sequence_len (backward compatible)
    window_short: int = 0   # 0 = window_long // 2 (backward compatible)
```

**File 2 — `nanochat/gpt.py` `_compute_window_sizes`**:
```python
    def _compute_window_sizes(self, config):
        pattern = getattr(config, 'window_pattern', 'L').upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}."

        m_pattern = getattr(config, 'mamba_pattern', '').upper()
        if getattr(config, 'mamba_enabled', False) and not m_pattern:
            m_pattern = "AAM"

        # Resolve window sizes from config or fallback to sequence_len
        long_window = getattr(config, 'window_long', 0) or config.sequence_len
        short_window = getattr(config, 'window_short', 0) or (long_window // 2)

        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}

        window_sizes = []
        for layer_idx in range(config.n_layer):
            if m_pattern and m_pattern[layer_idx % len(m_pattern)] == 'M':
                window_sizes.append(None)
            else:
                char = pattern[layer_idx % len(pattern)]
                window_sizes.append(char_to_window[char])

        # Last attention layer always gets full context
        if window_sizes[-1] is None:
            from nanochat.common import print0
            print0(f"WARNING: Last layer ({config.n_layer-1}) is Mamba.")
        else:
            window_sizes[-1] = (long_window, 0)
        return window_sizes
```

**File 3 — `nanochat/checkpoint_manager.py`**:
```python
    model_config_kwargs.setdefault('window_long', 0)
    model_config_kwargs.setdefault('window_short', 0)
```

**File 4 — `scripts/base_train.py`** (optional CLI args):
```python
    parser.add_argument("--window_long", type=int, default=0,
                        help="long window size (0=sequence_len)")
    parser.add_argument("--window_short", type=int, default=0,
                        help="short window size (0=window_long//2)")
```

**Example usage**: Training at seq_len=65536 with 128k sliding windows:
```bash
python base_train.py --sequence_len 65536 --window_long 131072 --window_short 65536
```

---

#### Idea D: `scan_layers` Safety Check for Hybrid Blocks

**Problem**: On TPU, `base_train.py` uses
`scan_layers(model.transformer.h)` to compile all layers as one reused
HLO body. This assumes all blocks have identical structure
(homogeneous). With hybrid A+M patterns, `transformer.h` contains both
`CausalSelfAttention` and `Mamba2Layer` blocks — `scan_layers` would
either crash (different parameter shapes) or silently produce wrong
results (treating Mamba params as attention params).

**Solution**: Add a guard before the `scan_layers` call that detects
hybrid patterns and skips the optimization with a warning.

**Where**: `scripts/base_train.py`, near the existing `scan_layers`
call. Find the section that looks like:
```python
if args.use_scan and device_type == "xla":
    from torch_xla.experimental.scan import scan_layers
    model.transformer.h = scan_layers(model.transformer.h)
```

**Replace with**:
```python
if args.use_scan and device_type == "xla":
    # scan_layers requires homogeneous blocks (same compiled body).
    # Hybrid A+M patterns have different block structures — skip.
    pat = getattr(config, "mamba_pattern", "") or ""
    is_hybrid = (
        getattr(config, "mamba_enabled", False)
        and pat.upper() != ""
        and "A" in pat.upper()
        and "M" in pat.upper()
    )
    if is_hybrid:
        print0("WARNING: --use_scan disabled because transformer.h "
               "contains heterogeneous blocks (attention + mamba). "
               "Each block type will be compiled separately.")
    else:
        try:
            from torch_xla.experimental.scan import scan_layers
            model.transformer.h = scan_layers(model.transformer.h)
            print0(f"Enabled XLA scan_layers ({config.n_layer} layers "
                   f"-> 1 compiled block)")
        except ImportError:
            print0("WARNING: --use_scan requires torch_xla scan_layers")
```

**Why this matters**: Without this guard, running
`--mamba --mamba_pattern AAM --use_scan` on TPU would crash with a
shape mismatch error (attention params ≠ mamba params) or silently
produce numerically wrong gradients. The guard is trivial to add and
prevents a confusing failure mode.

---

#### Idea E: Configurable `rope_theta` for Complex RoPE

**Problem**: The complex RoPE frequencies in `Mamba2Layer.__init__`
use a hardcoded base theta of 10000:
```python
inv_freq = 1.0 / (10000 ** (torch.arange(0, self.d_state, 2).float() / self.d_state))
```

The standard attention RoPE in `CausalSelfAttention` also uses
theta=10000 by default (gpt.py `_precompute_rotary_embeddings`
line 394: `base=10000`). For long context models (128k-1M tokens),
higher theta values are needed to spread the rotation frequencies
across the longer sequence range. Nemotron uses theta=1M for their
1M-context models.

**Solution**: Read `rope_theta` from config instead of hardcoding:

**Where**: `nanochat/mamba2.py`, in `Mamba2Layer.__init__`, the block
that creates `inv_freq`:

```python
        # BEFORE (hardcoded):
        if self.mamba3_complex_rope:
            inv_freq = 1.0 / (10000 ** (torch.arange(0, self.d_state, 2).float() / self.d_state))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        # AFTER (configurable):
        if self.mamba3_complex_rope:
            rope_theta = float(getattr(config, 'rope_theta', 10000.0))
            inv_freq = 1.0 / (rope_theta ** (torch.arange(0, self.d_state, 2).float() / self.d_state))
            self.register_buffer("inv_freq", inv_freq, persistent=False)
```

**Note**: We don't need a separate `mamba_rope_theta` field. The
attention RoPE theta is set via `_precompute_rotary_embeddings(base=)`
which currently hardcodes 10000. If the user wants different theta for
attention vs Mamba, they can add `mamba_rope_theta` later. For now,
sharing `rope_theta` across both is the right default — Nemotron uses
the same theta for both their attention and Mamba layers.

**No additional file changes needed** — this is a 1-line change in
`mamba2.py`. The config field doesn't need to exist in GPTConfig;
`getattr(config, 'rope_theta', 10000.0)` handles missing gracefully.

---

#### Idea F: `prev_B`/`prev_x` State Caching for Trapezoidal Decode

**Problem**: The trapezoidal discretization (Phase 3) computes:
```
h_t = α_t h_{t-1} + β_t (B_{t-1} ⊗ x_{t-1}) + γ_t (B_t ⊗ x_t)
```

The β_t term needs `B_{t-1}` and `x_{t-1}` from the PREVIOUS token.
During autoregressive decode, each call to `_ssd_step_ref` processes
one token, so the previous token's B and x must be cached in the state
dict between calls.

**Design decision**: Cache `prev_B` and `prev_x` separately (Strategy B
from Idea 5 in iteration 7) rather than their outer product:
- Strategy A (product): 192KB/layer in bf16
- Strategy B (factors): 4.5KB/layer in bf16

The outer product is trivially cheap to recompute from factors.

**Where**: Multiple locations in `nanochat/mamba2.py`, gated by
`self.mamba3_trapezoidal`:

**Change 1 — State dict creation** (in `_ssd_step_ref` or a helper):
```python
        # When mamba3_trapezoidal is True, add to state dict:
        if self.mamba3_trapezoidal:
            if "prev_B" not in states:
                states["prev_B"] = torch.zeros(
                    B_sz, self.ngroups, self.d_state,
                    device=x.device, dtype=torch.float32,
                )
            if "prev_x" not in states:
                states["prev_x"] = torch.zeros(
                    B_sz, self.nheads, self.headdim,
                    device=x.device, dtype=x.dtype,
                )
```

**Change 2 — Decode step** (replace standard SSM update in `_ssd_step_ref`):
```python
        if self.mamba3_trapezoidal:
            # λ from projection (already split in forward projection)
            lam = torch.sigmoid(lam_raw.float())  # (B, H)

            # Retrieve previous token's B and x
            prev_B = states["prev_B"]   # (B, G, N) float32
            prev_x = states["prev_x"]   # (B, H, D) x.dtype

            # Expand prev_B to match heads
            prev_B_h = prev_B.repeat_interleave(heads_per_group, dim=1)  # (B, H, N)

            # Trapezoidal coefficients (Eq. 4)
            alpha = dA                                          # exp(dt * A), (B, H)
            beta = (1.0 - lam) * dt_soft * alpha                # backward Euler weight
            gamma = lam * dt_soft                                # forward Euler weight

            # Previous term: β * B_{t-1} ⊗ x_{t-1}
            dBx_prev = (beta.unsqueeze(-1) * prev_B_h).unsqueeze(-1) \
                      * prev_x.to(torch.float32).unsqueeze(-2)  # (B, H, N, D)

            # Current term: γ * B_t ⊗ x_t
            dBx_curr = (gamma.unsqueeze(-1) * B_ssm).unsqueeze(-1) \
                      * x_ssm.to(torch.float32).unsqueeze(-2)   # (B, H, N, D)

            # State update (trapezoidal)
            ssm_state.copy_(
                ssm_state * alpha.view(B_sz, self.nheads, 1, 1).float()
                + dBx_prev + dBx_curr
            )

            # Cache current B and x for next token
            # Store B before repeat_interleave (grouped form, smaller)
            states["prev_B"].copy_(B_ssm_grouped.float())  # (B, G, N)
            states["prev_x"].copy_(x_ssm)                   # (B, H, D)
        else:
            # Standard Mamba-2 update (existing code)
            ssm_state.copy_(ssm_state * dA.view(B_sz, self.nheads, 1, 1).float() + dBx.float())
```

**Change 3 — Prefill boundary**: At the end of prefill (in `forward()`),
store the last token's B and x for decode continuation:
```python
        if self.mamba3_trapezoidal and inference_params is not None:
            # Store last token's B and x for trapezoidal decode
            states["prev_B"] = B_ssm[:, -1].float()  # (B, G, N)
            states["prev_x"] = x_ssm[:, -1]           # (B, H, D)
```

**KVCache.prefill compatibility**: The existing `prefill()` method's
`isinstance(v, dict)` branch handles `prev_B` and `prev_x` as regular
tensor entries in the per-layer state dict. The batch expansion logic
(`sv.expand(batch_size, ...).clone()`) works correctly for both shapes.
No additional prefill code is needed.

**First-token semantics**: At t=0, `prev_B` and `prev_x` are
zero-initialized. This is semantically correct — there is no previous
token, so `B_{-1} * x_{-1} = 0`. The β_t term contributes nothing,
and the γ_t term (forward Euler, like standard Mamba-2) handles the
first token naturally.

---

#### Cumulative Bug Status Table (after iteration 11)

| # | Severity | Description | Introduced | Fixed? |
|---|----------|-------------|-----------|--------|
| 1 | CRITICAL | F.pad 8-val on 4D pads batch dim | Iter 5 | ✅ Iter 7 |
| 2 | CRITICAL | kv_cache.advance() never called (last layer Mamba) | Iter 5 | ✅ Iter 7 |
| 3 | CRITICAL | init_weights skips DSA c_q/c_k/c_v/c_proj | Iter 5 | ✅ Iter 7 |
| 4 | MODERATE | estimate_flops double-counts + wrong units | Iter 6 | ✅ Iter 7 |
| 5 | MODERATE | Complex RoPE angle lost at prefill→decode | Iter 5 | ✅ Iter 7 |
| 6 | MODERATE | Missing Block.__init__, KVCache, CLI, GPTConfig | Iter 5 | ✅ Iter 7 |
| 7 | CRITICAL | KVCache.prefill() mamba state type mismatch | Iter 7 | ✅ Iter 8 |
| 8 | CRITICAL | Mamba state batch expansion missing in prefill | Iter 7 | ✅ Iter 8 |
| 9 | MODERATE | Triton kernel not used during inference prefill | Iter 7 | ✅ Iter 8 |
| 10 | MODERATE | Engine.generate KVCache constructors not shown | Iter 7 | ✅ Iter 8 |
| 11 | LOW | MTP hasattr guard unnecessary | Iter 7 | ✅ Iter 8 |
| 12 | CRITICAL | Triton final_states shape transposed | Iter 8 | ✅ Iter 8 |
| 13 | CRITICAL | fp32 state lost at prefill→decode boundary | Iter 8 | ✅ Iter 10+11 |
| 14 | LOW | Block.__init__ self-imports from gpt.py | Iter 8 | Still present (cosmetic) |
| 15 | MODERATE | Muon LR incorrectly scaled | Iter 8 | ✅ Iter 10 |
| 16 | LOW | _compute_window_sizes uses sequence_len//2 | Iter 8 | Addressed by Idea C |
| 17 | LOW | init_weights skips Mamba per-head params | Iter 8 | ✅ Iter 11 |
| 18 | MODERATE | norm self-import in GPT.forward | Iter 10 | ✅ Iter 11 |
| 19 | LOW | config.mtp_lambda vs self.mtp_lambda | Iter 10 | ✅ Iter 11 |

**All CRITICAL and MODERATE bugs are now FIXED.**
Only Bug #14 remains (cosmetic self-import, fix during implementation).

---

---

### Iteration 12 Review (Gemini "absolute final" + all 6 ideas adopted)

**Date**: 2026-02-13
**Verdict**: ALL 6 ideas from the alternative proposal successfully integrated.
1 old cosmetic bug persists (#14), 1 old bug regressed (#18), 1 new bug (#20).

#### Ideas Adoption Scorecard

| Idea | Status | How |
|------|--------|-----|
| A: XLA scan | ✅ ADOPTED | `_ssd_scan_xla()` with `torch_xla.experimental.scan`, 3-way dispatch |
| B: `mamba3_trapezoidal` | ✅ ADOPTED | GPTConfig field, `d_lambda` conditional in `d_in_proj`, trapezoidal `_ssd_step_ref` |
| C: `window_long/short` | ✅ ADOPTED | GPTConfig fields, `_compute_window_sizes` uses `getattr(config, 'window_long', 0) or config.sequence_len` |
| D: `scan_layers` safety | ✅ ADOPTED | Guard in base_train.py with `"A" in pat and "M" in pat` check |
| E: `rope_theta` config | ✅ ADOPTED | `rope_theta` in GPTConfig, used in Mamba2Layer and `_precompute_rotary_embeddings(base=...)` |
| F: `prev_B`/`prev_x` | ✅ ADOPTED | Full trapezoidal decode with `prev_B`/`prev_x` in `_ssd_step_ref` |

#### Previously-Fixed Bugs Verified

All critical bugs #1-13, #15, #17 remain fixed. Defensive fp32 cast
in `_ssd_step_ref` present. Meta-init safety for A_log/dt_bias/D present.
Muon LR uses `lr=matrix_lr` without `dmodel_lr_scale`. Triton state
`.transpose(-1,-2).to(fp32)` present.

#### Remaining Bugs

##### Bug #14 (LOW): Block.__init__ self-imports — STILL PRESENT

```python
# Block.__init__ has:
from nanochat.gpt import CausalSelfAttention  # same file
from nanochat.gpt import MLP                   # same file
```

These are redundant — `CausalSelfAttention` and `MLP` are defined in
gpt.py where Block lives. Only `from nanochat.mamba2 import Mamba2Layer`
is a legitimate cross-module lazy import.

**Fix**: Remove both imports, use classes directly.

##### Bug #18 (MODERATE): GPT.forward norm self-import — REGRESSED

```python
# GPT.forward has:
    x = self.transformer.wte(idx)
    from nanochat.gpt import norm    # ← WRONG: self-import, runs every forward
    x = norm(x)
```

This was fixed in iteration 11 but reintroduced in iteration 12.
`norm()` is a module-level function at gpt.py:139, always in scope.

**Fix**: Delete `from nanochat.gpt import norm`.

##### Bug #20 (MODERATE): Trapezoidal prev_B shape too large

```python
# In _ssd_step_ref trapezoidal branch:
    prev_B = states.setdefault("prev_B", torch.zeros_like(B_ssm))
    prev_x = states.setdefault("prev_x", torch.zeros_like(x_ssm))
```

At this point `B_ssm` has shape `(B, nheads, d_state)` — AFTER
`repeat_interleave(heads_per_group)`. Our design spec says `prev_B`
should be `(B, ngroups, d_state)` to save memory (42x smaller when
ngroups=1, nheads=12).

With `zeros_like(B_ssm)`, `prev_B` is `(B, 12, 64)` = 768 elements
instead of `(B, 1, 64)` = 64 elements per sample. Not a correctness
bug (the math still works with expanded B), but wastes memory and
diverges from the design spec.

**Fix**: Initialize before `repeat_interleave`:
```python
if self.mamba3_trapezoidal:
    # Store prev_B at group granularity (before repeat_interleave)
    B_ssm_grouped = xBC_conv[..., self.d_inner : self.d_inner + self.ngroups*self.d_state].view(B_sz, self.ngroups, self.d_state)
    prev_B = states.setdefault("prev_B", torch.zeros(B_sz, self.ngroups, self.d_state, device=x.device, dtype=torch.float32))
    prev_x = states.setdefault("prev_x", torch.zeros(B_sz, self.nheads, self.headdim, device=x.device, dtype=x.dtype))
    # ... after repeat_interleave for B_ssm:
    prev_B_expanded = prev_B.repeat_interleave(heads_per_group, dim=1)
```

#### Design Notes (not bugs)

**Trapezoidal training scan**: `_ssd_scan_ref` only implements Mamba-2
chunked scan. When `mamba3_trapezoidal=True`, training uses the
non-trapezoidal scan (equivalent to λ=1 forward Euler). Only decode
uses the full trapezoidal recurrence. This is correct per our Phase 3
design — the chunked trapezoidal scan requires Proposition 4 mask
modifications or a custom Triton kernel. A warning would be helpful
when `mamba3_trapezoidal=True` is used without the trapezoidal training
scan being available.

**XLA scan state not explicitly cast**: The XLA path stores
`states["ssm_state"] = final_states` where `_ssd_scan_xla` returns
fp32. This is correct but implicit — the Triton path explicitly casts
with `.to(torch.float32)`. Consider adding explicit cast for clarity.

#### Cumulative Bug Status Table (after iteration 12)

| # | Severity | Description | Introduced | Fixed? |
|---|----------|-------------|-----------|--------|
| 1-11 | various | (see iterations 5-8) | Iter 5-7 | ✅ All fixed |
| 12 | CRITICAL | Triton shape (H,D,N) vs ref (H,N,D) | Iter 8 | ✅ Iter 9 |
| 13 | CRITICAL | fp32 lost at prefill→decode | Iter 8 | ✅ Iter 10 |
| 14 | LOW | Block.__init__ self-imports | Iter 8 | **STILL OPEN** |
| 15 | MODERATE | Muon LR scaled by dmodel_lr_scale | Iter 8 | ✅ Iter 10 |
| 16 | LOW | window sizes hardcoded | Iter 8 | ✅ Iter 12 (Idea C) |
| 17 | LOW | init_weights skips per-head params | Iter 8 | ✅ Iter 11 |
| 18 | MODERATE | norm self-import in GPT.forward | Iter 8 | ✅ Iter 11, **REGRESSED Iter 12** |
| 19 | LOW | config mtp_lambda vs self.mtp_lambda | Iter 10 | ✅ Iter 12 |
| 20 | MODERATE | Trapezoidal prev_B shape (nheads vs ngroups) | Iter 12 | **NEW** |

**Summary**: 20 bugs total. 17 fixed. 3 remaining (1 LOW cosmetic, 2 MODERATE).
All CRITICAL bugs fixed. Ready to implement with 3 minor fixes during coding.

---

### Iteration 13: Bug Fixes #14, #18, #20

**Date**: 2026-02-13
**Verdict**: All 3 remaining bugs correctly fixed. 20/20 bugs resolved.

**Bug #14 FIXED**: Block.__init__ now uses `CausalSelfAttention` and `MLP`
directly (no self-imports). Only cross-module import `from nanochat.mamba2
import Mamba2Layer` retained.

**Bug #18 FIXED**: `from nanochat.gpt import norm` removed from GPT.forward.
`norm()` used directly as module-level function.

**Bug #20 FIXED**: Trapezoidal `prev_B` initialized at group level
`(B, ngroups, d_state)` via `torch.zeros_like(B_ssm)` BEFORE repeat_interleave.
Expanded dynamically to head level only for the step computation. Memory
savings confirmed: 4.5KB/layer (Strategy B) vs 54KB/layer (old).

**Minor observation**: `prev_x` remains at `(B, nheads, headdim)` since x_ssm
is inherently head-level — this is correct, no waste.

#### Final Bug Status: 20/20 FIXED

| # | Severity | Description | Fixed |
|---|----------|-------------|-------|
| 1-11 | CRITICAL-LOW | Iterations 5-8 bugs | ✅ |
| 12 | CRITICAL | Triton state shape transposed | ✅ Iter 10 |
| 13 | CRITICAL | fp32 lost at prefill→decode | ✅ Iter 10 |
| 14 | LOW | Block self-imports | ✅ **Iter 13** |
| 15 | MODERATE | Muon LR scaling | ✅ Iter 10 |
| 16 | LOW | Window sizes (addressed by Idea C) | ✅ Iter 12 |
| 17 | LOW | init_weights per-head params | ✅ Iter 11 |
| 18 | MODERATE | norm self-import in forward | ✅ **Iter 13** |
| 19 | LOW | config mtp_lambda access | ✅ Iter 11 |
| 20 | MODERATE | Trapezoidal prev_B memory waste | ✅ **Iter 13** |

---

### Patch Bundle Review (External Implementation)

**Date**: 2026-02-13
**Source**: Pre-built patch at `/home/dave/Downloads/nanochat_mamba_patch/`
**Files**: mamba2.py (634 lines), gpt.py (658 lines, truncated), engine.py,
checkpoint_manager.py, mtp.py, base_train.py

#### Status: DO NOT ADOPT wholesale — 7 bugs found. 1 excellent idea extracted.

**Bugs in the patch**:
1. `mamba2.py:105`: `self.qk_norm` should be `self.qknorm` — typo crashes at init
2. `gpt.py:624`: `get_ddp()` does not exist — should be `get_dist_info()`
3. `gpt.py`: File truncated at line 658 — missing `GPT.forward()`, rotary, generate
4. `base_train.py`: New CLI args defined but NOT wired into `model_config_kwargs`
5. `base_train.py:606`: Passes `adam_betas` to changed signature that doesn't accept it
6. `base_train.py:609`: Destructuring `adamw, muon = optimizers` reversed from return order
7. `sparse_attention.py`: Included but identical to original — advance() NOT removed

**Idea G: Dual-Scan Trapezoidal Decomposition** (ADOPT — Phase 3)

The patch decomposes trapezoidal discretization into TWO chunked SSD scans
instead of an O(T) Python loop. This is the key insight we were missing for
making Phase 3 training efficient.

**How it works**: The trapezoidal update is:
```
h_t = alpha_t * h_{t-1} + beta_t * B_{t-1}*x_{t-1} + gamma_t * B_t*x_t
```

This can be decomposed as two independent SSD scans:
```python
# 1. Current term: standard SSD with dt_input = gamma * dt
y_curr, state_curr = _ssd_scan_ref(x_ssm, dt_decay=dt_soft, dt_input=gamma_dt, ...)

# 2. Previous term: SSD on shifted inputs with dt_input = beta * dt
x_prev = F.pad(x_ssm, (0,0, 0,0, 1,0))[:, :-1]  # shift right by 1
B_prev = F.pad(B_ssm, (0,0, 0,0, 1,0))[:, :-1]
y_prev, state_prev = _ssd_scan_ref(x_prev, dt_decay=dt_soft, dt_input=beta_dt, ...)

# 3. Combine
y = y_curr + y_prev + x_ssm * D
final_state = state_curr + state_prev
```

**Why this matters**: Instead of an O(T) Python loop (2048 iterations per
layer), this uses two O(chunk_size) chunked scans (~8 iterations each).
The total cost is 2x a standard Mamba-2 scan — still 128x faster than
the naive Python loop.

**Implementation change needed in `_ssd_scan_ref`**: Split the `dt`
parameter into `dt_decay` (used for exp(dt*A) decay computation) and
`dt_input` (used for scaling B*x before accumulation). In standard
Mamba-2, both are identical. In trapezoidal mode, `dt_decay` stays as
dt_soft while `dt_input` becomes gamma*dt or beta*dt.

```python
def _ssd_scan_ref(self, x, dt_decay, dt_input, A, B, C, D):
    # dt_decay: used for exp(dt*A) state decay
    # dt_input: used for scaling the B*x input term
    # For Mamba-2: dt_decay == dt_input == dt_soft
    # For trapezoidal current: dt_input = gamma * dt
    # For trapezoidal previous: dt_input = beta * dt

    # ... chunking unchanged ...

    dA_c = dt_decay_c * A.view(1, 1, 1, H)  # decay uses dt_decay
    x_dt = x_c * dt_input_c.unsqueeze(-1)   # input scaling uses dt_input

    # ... rest of scan unchanged ...
```

**Other minor ideas noted but NOT adopted**:
- String-keyed state dict (`f'mamba_{layer_idx}'` vs integer) — no benefit
- `_InferParamsProxy` fallback — defensive but adds complexity
- `_get_or_create_causal_mask()` caching — minor optimization
- `exp/clamp_min` instead of `exp(diff)` for decay — numerically equivalent
- `conv1d.reset_parameters()` call — our meta-init in init_weights is cleaner
- `attn_long_window`/`attn_short_window` naming — we already have
  `window_long`/`window_short` (Idea C)

---

## Implementation Plan

The iteration 13 proposal with all 6 adopted ideas, all 20 bugs
fixed, plus Idea G (dual-scan trapezoidal) forms the complete
implementation specification.

**Files to create/modify (final)**:

| File | Action |
|------|--------|
| `nanochat/mamba2.py` | **Create** — Mamba2Layer class (Phase 1-2 + XLA scan + Phase 3 trapezoidal decode) |
| `nanochat/gpt.py` | **Modify** — GPTConfig (mamba + window_long/short + trapezoidal + rope_theta), Block dispatch, init_weights (meta-init safe), setup_optimizers, estimate_flops, _compute_window_sizes, forward |
| `nanochat/engine.py` | **Modify** — KVCache (add MambaInferenceParams), Engine.generate (has_mamba) |
| `nanochat/sparse_attention.py` | **Modify** — Remove advance() from _full_attention |
| `nanochat/mtp.py` | **Modify** — Add mamba_enabled=False to plain_config |
| `nanochat/checkpoint_manager.py` | **Modify** — Add mamba + window + trapezoidal + rope_theta defaults |
| `scripts/base_train.py` | **Modify** — CLI args for mamba + window + rope_theta, scan_layers safety check |
| `tests/test_mamba_integration.py` | **Create** — Composability tests for all phase/module combinations |
