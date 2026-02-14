# FIRE Integration Design Log

## Overview

Adding FIRE (Frobenius-Isometry REinitialization) to nanochat's training pipeline
for plasticity restoration at phase boundaries. Also adding optional SwiGLU
activation and ReDo dormant neuron diagnostics.

**Paper**: "FIRE: Frobenius-Isometry Reinitialization for Balancing the
Stability-Plasticity Tradeoff" (Han et al., GIST/KAIST, arXiv 2602.08040v1,
Feb 2026)

**Reference code**: https://github.com/isaac7778/FIRE (cloned at /home/dave/source/FIRE)

---

## What FIRE Does

FIRE solves a constrained optimization: project each weight matrix W onto the
nearest orthogonal matrix while minimizing change from the current weights.

```
minimize ||W - W_tilde||_F^2        (stability: stay close to trained weights)
subject to W_tilde^T W_tilde = I    (plasticity: isometric spectrum)
```

Solution (Orthogonal Procrustes): `W_tilde = W (W^T W)^{-1/2}`

Approximated via Newton-Schulz iteration (same math as Muon, but applied to
WEIGHTS once at phase boundary, not to GRADIENTS every step):

```python
X = W / ||W||_F
for _ in range(N):       # N=5 is enough
    A = X^T @ X
    X = 1.5 * X - 0.5 * X @ A
return X * sqrt(d_out / d_in)
```

**Key properties**:
- DfI(W_tilde) = 0 (all singular values = 1, maximum plasticity)
- SFE(W, W_tilde) is minimized (closest possible to trained weights)
- <1% training time overhead (applied once between phases)
- Only hyperparameter: N iterations (5 works, 10 is safer)
- Proven to: smooth loss landscape (Thm 2), increase feature rank (Thm 3),
  reduce dormant neurons (Thm 4)

---

## Why We Need It

Our training pipeline has clear phase boundaries where data distribution shifts:

| Transition                   | What changes                              | FIRE scope                  |
| ---------------------------- | ----------------------------------------- | --------------------------- |
| Pretrain phase N → phase N+1 | New data chunk                            | All 2D (if loss plateaued)  |
| Pretrain → SFT               | Format: raw code → instruction+tool       | All 2D                      |
| SFT → GSPO/RL                | Signal: teacher forcing → compiler reward | All 2D                      |
| 64K → 128K context           | Positional encoding (YaRN)                | Attention Q/K/V/c_proj only |

At each boundary, the model's weight spectrum has drifted from isotropy.
Singular values spread out, some grow large (memorized patterns), some collapse
(dead directions). FIRE projects back to isotropy while staying as close as
possible to the trained solution.

---

## Architecture Compatibility

### Parameter Classification

Our 877M model has these parameter types:

| Parameter              | Shape        | Optimizer   | FIRE?         | Why                         |
| ---------------------- | ------------ | ----------- | ------------- | --------------------------- |
| `wte.weight`           | (V, D)       | AdamW       | NO            | Embedding, not a projection |
| `lm_head.weight`       | (V, D)       | AdamW       | NO            | Output head, sensitive      |
| `resid_lambdas`        | (L,)         | AdamW       | NO            | 1D scalar                   |
| `x0_lambdas`           | (L,)         | AdamW       | NO            | 1D scalar                   |
| `attn.c_q.weight`      | (D, D)       | Muon        | YES           | Attention Q projection      |
| `attn.c_k.weight`      | (D_kv, D)    | Muon        | YES           | Attention K projection      |
| `attn.c_v.weight`      | (D_kv, D)    | Muon        | YES           | Attention V projection      |
| `attn.c_proj.weight`   | (D, D)       | Muon        | YES           | Attention output            |
| `mlp.c_fc.weight`      | (4D, D)      | Muon        | YES           | MLP up projection           |
| `mlp.c_proj.weight`    | (D, 4D)      | Muon        | YES           | MLP down projection         |
| `mlp.c_gate.weight`    | (4D, D)      | Muon        | YES           | SwiGLU gate (if enabled)    |
| `attn.in_proj.weight`  | (d_proj, D)  | Muon        | YES           | Mamba input projection      |
| `attn.out_proj.weight` | (D, D_inner) | Muon        | YES           | Mamba output projection     |
| `attn.A_log`           | (H,)         | AdamW       | NO            | 1D, SSM pole positions      |
| `attn.dt_bias`         | (H,)         | AdamW       | NO            | 1D, SSM timestep bias       |
| `attn.D`               | (H,)         | AdamW       | NO            | 1D, SSM skip connection     |
| `attn.conv1d.weight`   | (C, 1, K)    | AdamW       | NO            | 3D, depthwise conv          |
| `attn.conv1d.bias`     | (C,)         | AdamW       | NO            | 1D, conv bias               |
| `attn.B_bias`          | (G, N)       | AdamW       | NO            | 2D but bias, AdamW-routed   |
| `attn.C_bias`          | (G, N)       | AdamW       | NO            | 2D but bias, AdamW-routed   |
| `engram.*`             | various      | Muon/AdamW  | YES (2D only) | Engram projections          |
| `mhc.*`                | various      | Muon/AdamW  | YES (2D only) | mHC projections             |
| DSA `indexer.*`        | various      | AdamW (XLA) | YES (2D only) | DSA indexer                 |

**Rule**: FIRE applies to `param.dim() == 2` AND NOT in skip list
(`embed`, `head`, `bias`, `_bias`, `lambda`).

### Wide Matrix Handling

Newton-Schulz requires rows >= cols. Some of our matrices are "wide"
(d_out < d_in), e.g., `c_proj` when d_inner > d_model. Fix: transpose
before NS, transpose back after.

```python
is_wide = d_out < d_in
if is_wide: X = X.T
# ... NS iterations ...
if is_wide: X = X.T
```

### Scaling Factor

After orthogonalization, singular values = 1. Need to restore signal
variance for the residual stream. Use Modular Duality scaling:

```python
scale = sqrt(d_out / d_in)
```

This preserves `E[||Wx||^2] = ||x||^2 * d_out / d_in` which is what
random orthogonal initialization gives.

---

## Optimizer State Reset

CRITICAL: After FIRE, optimizer momentum/variance states are stale.

We have two optimizers:
- **Muon**: stores momentum buffer for 2D matrix params
- **AdamW**: stores exp_avg + exp_avg_sq for embeddings, head, scalars, Mamba 1D/3D

After FIRE on 2D matrices, we must reset:
1. Muon momentum for FIRE'd params (these are the same params Muon optimizes)
2. AdamW states for any FIRE'd params that happen to be in AdamW groups
   (e.g., Engram/mHC/DSA 2D params if they're in AdamW)

We must NOT reset:
- AdamW states for wte, lm_head, resid_lambdas, x0_lambdas (untouched by FIRE)
- AdamW states for Mamba 1D params (untouched by FIRE)

Implementation: iterate optimizer.state, clear only for params that were FIRE'd.

```python
def reset_optimizer_for_params(optimizers, fired_param_ids):
    for opt in optimizers:
        for group in opt.param_groups:
            for p in group['params']:
                if id(p) in fired_param_ids:
                    if p in opt.state:
                        del opt.state[p]
```

---

## SwiGLU Option

### Why

relu^2 (Primer, So et al. 2021) creates dormant neurons aggressively.
SwiGLU (Shazeer 2020, used by Llama/Mistral/Qwen/DeepSeek) uses SiLU
which is smooth and never exactly zero — no dormant neurons.

Adding SwiGLU as optional activation eliminates the need for ReDo
diagnostics entirely.

### Implementation

```python
# GPTConfig:
activation: str = "relu2"   # "relu2" | "swiglu"
mlp_hidden_mult: float = 4.0  # 4.0 for relu2, 8/3 for swiglu (Llama trick)

# MLP:
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = int(config.mlp_hidden_mult * config.n_embd)
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)
        self.use_swiglu = config.activation == "swiglu"
        if self.use_swiglu:
            self.c_gate = nn.Linear(config.n_embd, hidden, bias=False)

    def forward(self, x):
        if self.use_swiglu:
            return self.c_proj(F.silu(self.c_gate(x)) * self.c_fc(x))
        else:
            return self.c_proj(F.relu(self.c_fc(x)).square())
```

### FLOP Comparison (d_model=1536)

| Variant              | hidden | FLOP/token | Params/layer | GEMMs |
| -------------------- | ------ | ---------- | ------------ | ----- |
| relu^2 (current)     | 6144   | 37.7M      | 18.9M        | 2     |
| SwiGLU + Llama trick | 4096   | 37.7M      | 18.9M        | 3     |
| SwiGLU no trick      | 6144   | 56.6M      | 28.3M        | 3     |

Llama trick: equal FLOP, equal params, 3 smaller GEMMs vs 2 larger.
~3-5% wall-clock overhead from extra kernel dispatch.

### Init for SwiGLU

c_gate gets same init as c_fc (uniform_(-s, s)).
c_proj still zeros (matching our residual stream convention).

---

## ReDo Diagnostics

### When Needed

Only relevant for relu^2. If SwiGLU is enabled, skip this section.

### How It Works

Monitor neuron activity: `activity_j = E[|activation_j|] / mean(E[|activation_k|])`.
If `activity_j < tau` (default 0.025), neuron j is "dormant".

### Hook Placement for Our Architecture

relu^2 is functional (`F.relu(x).square()`), not an nn.Module.
Hook goes on `c_fc` output, apply activation in the hook:

```python
def hook(module, inp, out):
    act_out = F.relu(out).square()
    mean_abs = act_out.abs().mean(dim=(0, 1))  # [hidden_dim]
    # EMA update...
```

### Layer Map for Recycling

```python
layer_map = {}
for i, block in enumerate(model.transformer.h):
    name = f"transformer.h.{i}.mlp.c_fc"
    layer_map[name] = (block.mlp.c_fc, block.mlp.c_proj)
```

For SwiGLU, the map would be:
```python
    layer_map[name] = ([block.mlp.c_gate, block.mlp.c_fc], block.mlp.c_proj)
```

### XLA Safety

All neuron replacement via `torch.where` (static shapes), no boolean indexing.

---

## DASH (Direction-Aware Shrinking)

### When to Use

Only if loss plateaus WITHIN a phase (same data, loss stuck).
NOT between phases (that's FIRE's job).

### Conflict with Muon

CRITICAL: DASH modifies weights between backward() and optimizer.step().
Muon also modifies the gradient direction via NS iteration.
Applying DASH to Muon-managed params means:
1. DASH shrinks the weight
2. Muon orthogonalizes the gradient
3. Weight gets updated by orthogonalized gradient of a shrunk weight

This is unpredictable. Two options:

**Option A (Safe)**: DASH only on AdamW-managed 2D params (Engram, mHC, DSA
indexer). Skip all Muon params. This is very limited.

**Option B (Periodic)**: Apply DASH every K steps (K=2000+), not every 50.
At this frequency, the weight modification is small enough that Muon's
per-step NS doesn't fight it.

**Recommendation**: Start with Option B, K=2000, alpha=0.05, shrink_rate=0.005.
Monitor loss curve for instability. If unstable, switch to Option A or drop
DASH entirely (FIRE between phases may be sufficient).

### Implementation

Per-neuron cosine similarity (not scalar):

```python
def dash_step(W, grad, alpha=0.05, shrink_rate=0.005):
    cos_sim = F.cosine_similarity(W, grad, dim=1)  # [d_out]
    penalty = torch.clamp(cos_sim - alpha, min=0.0).unsqueeze(1)
    shrink_factor = torch.clamp(1.0 - shrink_rate * penalty, min=0.5, max=1.0)
    return W * shrink_factor
```

---

## Implementation Plan

### File: `nanochat/fire.py` (new)

Core FIRE algorithm + ReDo diagnostics + DASH.

Functions:
- `fire_reinitialize(model, iters=5, skip, target)` — apply FIRE to 2D params
- `reset_optimizer_states(optimizers, fired_params)` — selective state reset
- `newton_schulz(W, iters)` — the NS iteration (shared math with Muon)
- `measure_dfi(model)` — diagnostic: compute DfI per layer
- `measure_sfe(model, checkpoint)` — diagnostic: compute SFE vs saved weights
- `attach_dormancy_hooks(model, act_fn)` — ReDo monitoring
- `recycle_dormant(model, stats, tau)` — neuron recycling
- `dash_step(model, alpha, shrink_rate)` — DASH shrinking

### File: `nanochat/gpt.py` (modify)

- Add `activation: str = "relu2"` and `mlp_hidden_mult: float = 4.0` to GPTConfig
- Add `c_gate` to MLP when swiglu
- Add SwiGLU forward path
- Update `init_weights` for c_gate
- Update `estimate_flops` for 3-GEMM SwiGLU

### File: `scripts/base_train.py` (modify)

- Add `--fire_iters` (int, default 0 = disabled)
- Add `--fire_target` (str, choices=['all', 'attention', 'mlp'], default='all')
- Add `--activation` (str, choices=['relu2', 'swiglu'], default='relu2')
- Add `--mlp_hidden_mult` (float, default=0)
  - 0 means auto: 4.0 for relu2, 2.667 for swiglu
- Add `--dash_interval` (int, default 0 = disabled)
- Add `--redo_interval` (int, default 0 = disabled)
- Wire into training loop at phase boundaries

### File: `nanochat/checkpoint_manager.py` (modify)

- Add compat defaults: `activation="relu2"`, `mlp_hidden_mult=4.0`

### File: `tests/test_fire.py` (new)

- Test NS iteration converges to orthogonal matrix
- Test FIRE preserves param count and shapes
- Test FIRE reduces DfI to near-zero
- Test SFE is smaller than full reset
- Test wide matrix handling (transpose)
- Test skip_keywords work
- Test selective optimizer reset
- Test SwiGLU forward matches expected output
- Test SwiGLU + FIRE composition
- Test ReDo detects dormant neurons with relu^2
- Test DASH per-neuron shrinking

---

## Open Questions

1. **FIRE on B_bias/C_bias**: These are 2D (ngroups, d_state) but routed to
   AdamW via `_bias` suffix. Should FIRE skip them? YES — they're small
   learned biases, not projections. The `_bias` suffix skip handles this.

2. **FIRE on Engram/mHC**: These modules have 2D projections (in_proj,
   out_proj, score_proj, etc.). FIRE should apply to them. They're standard
   linear projections that benefit from spectral conditioning.

3. **FIRE iterations**: Paper says 5 is enough for LLMs (Table 10, Appendix E.2).
   Their GPT-0.1B used 5. Our 877M is ~9x larger but NS convergence doesn't
   depend on matrix size, only on condition number. Start with 5, increase if
   DfI diagnostic shows insufficient convergence.

4. **When to apply within pretrain**: Only at plateau. Monitor
   `val_loss_derivative < epsilon` for K consecutive evals. Don't apply
   if loss is still decreasing — FIRE is unnecessary and would slow
   convergence by disrupting a good optimization trajectory.

5. **mlp_hidden_mult for SwiGLU**: Llama uses 8/3 ≈ 2.667, but then rounds
   to nearest multiple of 256 for hardware efficiency. For d_model=1536:
   `int(8/3 * 1536) = 4096` which is already 256-aligned. Good.

---

## Bug Tracker

(No bugs yet — implementation not started)

| #   | Severity | Description | Status |
| --- | -------- | ----------- | ------ |

---

## Iteration 1: Design Review

Status: **DESIGN PHASE** — awaiting review before implementation.

### Decisions to confirm:

1. SwiGLU hidden = `int(8/3 * n_embd)` rounded to multiple of 256?
2. FIRE default scope = all 2D (not just attention)?
3. DASH frequency = every 2000 steps (not 50)?
4. ReDo only with relu^2, skip if SwiGLU?
5. Optimizer reset = selective (only FIRE'd params), not global clear?
