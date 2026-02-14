# Mamba-3 Follow-up Plan v4

Consolidated review of `nanochat/mamba2.py`, `docs/mamba_integration_log.md`,
an alternative "dual-scan" proposal, and PyTorch/XLA SPMD documentation.
All findings, fixes, and implementation status.

---

## Bugs Found and Fixed (not in design log's 20-bug tracker)

### Bug #21 (CRITICAL): Complex RoPE double-increment in decode — FIXED

**Location**: `nanochat/mamba2.py` `_apply_complex_rope` (old method, now removed).
Called twice per step (for B then C) in both `forward()` and `_ssd_step_ref`.

**Problem**: In decode mode (L==1), the method mutated the stored `rope_angle`
by adding `dt_avg`. First call (B) incremented angle from `a` to `a + dt`.
Second call (C) read `a + dt` and incremented to `a + 2*dt`. C got the wrong angle.

Never caught across 13 review iterations in the design log.

**Fix**: Split into `_compute_rope_angles` (single mutation per step) +
`_rotate_with_rope` (pure rotation, no state). Compute angle once, apply to both.

**Verification**: `tests/test_mamba_bugs.py::TestBug21RopeDoubleIncrement` (3 tests)

---

### Bug #22 (MODERATE): `prev_x` stored in model dtype, not fp32 — FIXED

**Location**: `nanochat/mamba2.py` prefill storage and decode init for `prev_x`.

**Problem**: `prev_B` was stored in fp32 but `prev_x` was stored in model dtype
(bf16). Precision lost at storage boundary over many decode steps.

**Fix**: Added `.float()` to all `prev_x` storage paths. Decode init uses
`dtype=torch.float32`.

**Verification**: `tests/test_mamba_bugs.py::TestBug22PrevXDtype` (2 tests)

---

### Bug #23 (LOW): Trapezoidal cross-chunk boundary term dropped — DOCUMENTED

**Location**: `nanochat/mamba2.py` `_ssd_scan_ref` trapezoidal section.

**Problem**: `beta_next[:, :, -1]` is zero, dropping coupling from last token
of chunk `c` to first token of chunk `c+1`. Affects 1/256 tokens.

**Decision**: Accepted as approximation. Dropped term is attenuated by
cross-chunk state decay. Comment added to code.

**Note**: With Idea G dual-scan now implemented in `forward()`, this inline
trapezoidal path in `_ssd_scan_ref` is only used as a reference implementation.

**Verification**: `tests/test_mamba_bugs.py::TestBug23CrossChunkBoundary`

---

## Idea G: Dual-Scan Trapezoidal Decomposition — IMPLEMENTED

### Mathematical Proof

The standard scan computes: `h_t = exp(dt_t*A) * h_{t-1} + dt_t * B_t * x_t`

The decay `exp(dt_t*A)` applies ONLY to the past state `h_{t-1}`. The new
input `dt_t * B_t * x_t` is injected WITHOUT any decay at step t. It only
experiences its first decay at step t+1.

Therefore, constructing `x_prev_mod = (1-lam) * exp(dt*A) * x_{t-1}` and
passing it as input at step t correctly computes the trapezoidal term without
double-decay. The mathematical proof and computational test confirm the
decomposition is exact.

**Verified computationally**: `tests/test_idea_g_math.py` proves the
decomposition is exact to fp64 precision over 32 steps.

### Implementation

The dual-scan trapezoidal is implemented in `forward()`. Two parallel standard
SSD scans compute trapezoidal math, enabling Triton and XLA backends natively
for Phase 3 training without custom kernels.

---

## Structural Improvements — IMPLEMENTED

### `_run_scan` Abstraction

Centralizes backend dispatch: XLA > Triton > ref. Handles fp32 state
conversion, Triton final-state transpose (Bug #12), XLA availability
warning, and `D=None` support for dual-scan.

### `D=None` Support

Added to both `_ssd_scan_ref` and `_ssd_scan_xla`. Required for Idea G
dual-scan where D skip-connection is applied outside the scans.

### XLA Scan Config Wiring

`mamba_xla_scan` config flag wired through `GPTConfig`, `checkpoint_manager.py`
compat defaults, and `scripts/base_train.py` CLI (`--mamba_xla_scan`).

### CLI Surfacing

`--rope_theta`, `--window_long`, `--window_short`, `--mamba_xla_scan` all
exposed in `scripts/base_train.py` and wired into model config.

---

## SPMD and Distributed Features — IMPLEMENTED

### Existing SPMD Infrastructure (already in codebase)

The project has extensive SPMD support:
- 1D/2D device mesh via `torch_xla.distributed.spmd.Mesh`
- Megatron-style tensor parallelism for attention Q/K/V/proj and MLP
- SPMD partition specs for Pallas flash attention
- Data-parallel input sharding via `xs.mark_sharding`
- `torch_xla.compile()` context for whole-graph XLA compilation
- `scan_layers` for single-block XLA compilation of transformer stack
- `torch.compile` deliberately avoided on TPU (causes OOM)

### What was added

**1. Mamba SPMD Sharding** (`scripts/base_train.py` `_apply_tensor_parallel_sharding`)

Mamba `in_proj` and `out_proj` were not covered by existing sharding annotations.
Added:
- `attn.in_proj.weight` -> `('model', None)` column-parallel
- `attn.out_proj.weight` -> `(None, 'model')` row-parallel

This matches the Megatron-style sharding used for attention layers. The conv1d
is left replicated (depthwise conv with `groups=conv_dim` doesn't benefit from
tensor parallelism — each channel is independent).

**2. scan_layers Hybrid Guard** (`scripts/base_train.py`)

`scan_layers` assumes all layers in the `nn.ModuleList` are identical. With
hybrid AAM pattern (alternating Attention and Mamba), layers are heterogeneous.
Added a guard that detects `is_mamba` heterogeneity across blocks and falls
back with a warning instead of silently producing wrong results.

**3. Distributed Checkpointing** (`nanochat/checkpoint_manager.py`)

Added `save_checkpoint_distributed()` and `load_checkpoint_distributed()` using
`torch.distributed.checkpoint` with `SPMDSavePlanner`/`SPMDLoadPlanner` from
torch_xla. Benefits:
- Save sharded model weights directly (no gather to rank 0)
- Load sharded weights directly (no broadcast)
- Resharding support when changing TP degree

Gated behind availability check — falls back to standard `torch.save`/`torch.load`
if `torch.distributed.checkpoint` or XLA planners are unavailable. Fully
backward-compatible with existing checkpoint format.

### What was NOT added (and why)

| Feature | Reason |
| --- | --- |
| `torch.compile(backend='openxla')` | Deliberately avoided in project — causes OOM during compilation on TPU. Project uses `torch_xla.compile()` context instead. |
| `XLAShardedTensor` explicit usage | SPMD sharding via `mark_sharding` is simpler and equivalent for this use case. |
| `HybridMesh` | Only needed for multi-pod TPU training (ICI + DCN). Single-pod v6e-4 uses standard `Mesh`. |
| Auto-sharding (`XLA_AUTO_SPMD_MESH`) | Manual sharding annotations give more control and are already comprehensive. |

---

## Alternative Proposal Review

### Adopted

| Feature | Description | Status |
| --- | --- | --- |
| `_run_scan` abstraction | Centralize backend dispatch | IMPLEMENTED |
| `D=None` support | Allow D outside scan for dual-scan | IMPLEMENTED |
| Global sequence shift | `F.pad(x, (0,0, 0,0, 1,0))[:, :-1]` | IMPLEMENTED |
| `prev_x`/`prev_B` injection | Set `x_prev[:, 0]` from cache at prefill boundary | IMPLEMENTED |
| Idea G dual-scan | Two standard scans compute trapezoidal | IMPLEMENTED |
| `ngroups` assertion | Assert nheads % ngroups == 0 | IMPLEMENTED |
| `heads_per_group` attribute | Store instead of recomputing | IMPLEMENTED |
| fp32 output contraction | `C.float() * state` before downcast | ALREADY HAD |

### Rejected

| Feature | Reason |
| --- | --- |
| Python loop in `_ssd_scan_ref` | Regression from parallel matmul; unrolls into XLA HLO |
| bf16 `prev_B`/`prev_x` storage | Precision loss over decode sequences |
| `ssm_state.to(x.dtype)` in output | Loses fp32 precision in C*state contraction |
| `getattr(self, "D", None)` | D is always created in `__init__`; dead defensive code |

### Earlier objection retracted

I claimed the dual-scan `x_prev_mod * dA` embedding caused double-decay.
This was wrong. The scan's decay applies to `h_{t-1}`, not to the new input
at step `t`. Verified computationally to fp64 precision.

---

## Test Results

### Local (CPU)

| Test Suite | Passed | Skipped | Failed |
| --- | --- | --- | --- |
| `tests/test_mamba_bugs.py` | 8 | 0 | 0 |
| `tests/test_mamba_integration.py` | 41 | 0 | 0 |
| `tests/test_idea_g_math.py` | 3 | 0 | 0 |
| `tests/test_mamba_parity.py` | 4 | 3 (CUDA) | 0 |
| **Total** | **56** | **3** | **0** |

### TPU v6e-4 (us-east5-a)

| Test Suite | Passed | Skipped | Failed |
| --- | --- | --- | --- |
| `tests/test_mamba_bugs.py` | 8 | 0 | 0 |
| `tests/test_mamba_integration.py` | 41 | 0 | 0 |
| `tests/test_idea_g_math.py` | 3 | 0 | 0 |
| `tests/test_mamba_parity.py` | 4 | 3 (CUDA) | 0 |
| **Total** | **56** | **3** | **0** |

---

## All Files Modified Across Sessions

| File | Changes |
| --- | --- |
| `nanochat/mamba2.py` | Bug #21/#22/#23 fixes, `_run_scan`, D=None, Idea G dual-scan |
| `nanochat/gpt.py` | `mamba_xla_scan` config field |
| `nanochat/checkpoint_manager.py` | `mamba_xla_scan` compat default, distributed checkpoint functions |
| `scripts/base_train.py` | 4 CLI args, Mamba SPMD sharding, scan_layers hybrid guard |
| `tests/test_mamba_bugs.py` | NEW: 8 bug reproduction tests |
| `tests/test_mamba_parity.py` | NEW: 7 parity + streaming stability tests |
| `tests/test_idea_g_math.py` | NEW: 3 mathematical verification tests |
| `scripts/bench_mamba_backends.py` | NEW: throughput/memory benchmark harness |
| `docs/mamba_review_followup_plan.md` | This document (v4) |

---

## Remaining Work

| Task | Priority | Notes |
| --- | --- | --- |
| Run small all-inclusive training (mamba3 + mhc + mtp + dsa + engram) | High | Verify convergence with all features enabled |
| CUDA Triton parity validation | Medium | Requires CUDA GPU to run skipped tests |
| Benchmark on CUDA GPU | Medium | `scripts/bench_mamba_backends.py` ready |
| Integrate distributed checkpoint into training loop | Low | Functions exist; need wiring in `base_train.py` |
| scan_layers for Mamba-only models | Low | Works when all layers are Mamba; guard skips hybrid |
