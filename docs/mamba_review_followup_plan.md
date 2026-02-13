# Mamba-3 Follow-up Plan v3

Consolidated review of `nanochat/mamba2.py`, `docs/mamba_integration_log.md`,
and an alternative "dual-scan" proposal. This document captures all findings,
concrete fixes with code, and implementation status.

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
trapezoidal path in `_ssd_scan_ref` is only used when `forward()` calls
`_ssd_scan_ref` with `lam` directly (which no longer happens for the main
trapezoidal training path). The `_ssd_scan_ref` trapezoidal code is retained
as a reference implementation and for direct testing.

**Verification**: `tests/test_mamba_bugs.py::TestBug23CrossChunkBoundary`

---

## Idea G: Dual-Scan Trapezoidal Decomposition — IMPLEMENTED

### Mathematical Proof

The alternative proposal's claim is **correct**. I was wrong about "Bug A"
(double-decay). Here's the proof:

The standard scan computes: `h_t = exp(dt_t*A) * h_{t-1} + dt_t * B_t * x_t`

The decay `exp(dt_t*A)` applies ONLY to the past state `h_{t-1}`. The new
input `dt_t * B_t * x_t` is injected WITHOUT any decay at step t. It only
experiences its first decay at step t+1.

Therefore, constructing `x_prev_mod = (1-lam) * exp(dt*A) * x_{t-1}` and
passing it as input at step t results in:

```
h_prev_t = exp(dt_t*A) * h_prev_{t-1} + dt_t * B_t * [(1-lam_t) * exp(dt_t*A) * x_{t-1}]
```

This correctly computes the `beta` trapezoidal term without double-decay.
The `exp(dt_t*A)` in `x_prev_mod` is the decay from `t-1` to `t` (the
trapezoidal coupling), while the scan's `exp(dt_t*A)` on `h_{t-1}` is the
state propagation — these are independent and don't interact.

**Verified computationally**: `tests/test_idea_g_math.py` proves the
decomposition is exact to fp64 precision over 32 steps.

### Implementation

The dual-scan trapezoidal is implemented in `forward()` at lines 267-297:

```python
if self.mamba3_trapezoidal:
    # Build shifted previous-step tensors (sequence-level shift)
    x_prev = F.pad(x_ssm, (0,0, 0,0, 1,0))[:, :-1]
    B_prev = F.pad(B_ssm, (0,0, 0,0, 1,0))[:, :-1]

    # Inject cached state from prior prefill/decode
    if inference_params is not None:
        if "prev_x" in states: x_prev[:, 0] = states["prev_x"].to(x.dtype)
        if "prev_B" in states: B_prev[:, 0] = states["prev_B"].to(x.dtype)

    # Embed trapezoidal weights into modified inputs
    x_curr_mod = x_ssm * lam.unsqueeze(-1)
    dA = torch.exp(dt_soft * A.view(1, 1, self.nheads))
    x_prev_mod = x_prev * ((1.0 - lam) * dA).unsqueeze(-1)

    # Two parallel standard SSD scans (enables Triton/XLA natively!)
    y_curr, st_curr = self._run_scan(x_curr_mod, dt_soft, A, B_ssm, C_ssm, D=None)
    y_prev, st_prev = self._run_scan(x_prev_mod, dt_soft, A, B_prev, C_ssm, D=None)

    y = y_curr + y_prev + x_ssm * self.D.view(1, 1, self.nheads, 1)
    final_state = st_curr + st_prev  # valid: linear superposition
```

**Key insight**: Both scans are standard Euler SSD scans — no modified kernels
needed. This means Triton `mamba_chunk_scan_combined` and XLA `xla_scan` can
compute trapezoidal math natively, eliminating the previous restriction that
gated Triton off for trapezoidal mode.

### What remains of the old trapezoidal code

The inline trapezoidal in `_ssd_scan_ref` (dt_eff / beta_next / diagonal
correction) is retained but no longer called by `forward()` for the main
training path. It serves as:
1. Reference implementation for numerical parity testing
2. Backward compatibility for any code calling `_ssd_scan_ref` directly

---

## Structural Improvements — IMPLEMENTED

### `_run_scan` Abstraction (lines 159-195)

Centralizes backend dispatch: XLA > Triton > ref. Handles:
- fp32 state conversion
- Triton final-state transpose (`Bug #12`)
- XLA availability warning
- `D=None` support for dual-scan

### `D=None` Support

Added to both `_ssd_scan_ref` and `_ssd_scan_xla`. Required for Idea G
dual-scan where D skip-connection is applied outside the scans.

---

## Alternative Proposal Review

### Adopted

| Feature                     | Description                                       | Status      |
| --------------------------- | ------------------------------------------------- | ----------- |
| `_run_scan` abstraction     | Centralize backend dispatch                       | IMPLEMENTED |
| `D=None` support            | Allow D outside scan for dual-scan                | IMPLEMENTED |
| Global sequence shift       | `F.pad(x, (0,0, 0,0, 1,0))[:, :-1]`               | IMPLEMENTED |
| `prev_x`/`prev_B` injection | Set `x_prev[:, 0]` from cache at prefill boundary | IMPLEMENTED |
| Idea G dual-scan            | Two standard scans compute trapezoidal            | IMPLEMENTED |
| `ngroups` assertion         | Assert nheads % ngroups == 0                      | IMPLEMENTED |
| `heads_per_group` attribute | Store instead of recomputing                      | IMPLEMENTED |
| `conv_dim` attribute        | Store instead of recomputing                      | ALREADY HAD |
| fp32 output contraction     | `C.float() * state` before downcast               | ALREADY HAD |

### Rejected

| Feature                           | Reason                                                 |
| --------------------------------- | ------------------------------------------------------ |
| Python loop in `_ssd_scan_ref`    | Regression from parallel matmul; unrolls into XLA HLO  |
| bf16 `prev_B`/`prev_x` storage    | Precision loss over decode sequences                   |
| `ssm_state.to(x.dtype)` in output | Loses fp32 precision in C*state contraction            |
| `getattr(self, "D", None)`        | D is always created in `__init__`; dead defensive code |

### My earlier objection retracted

I claimed the dual-scan `x_prev_mod * dA` embedding caused double-decay.
This was wrong. The scan's decay applies to `h_{t-1}`, not to the new input
at step `t`. The mathematical proof and computational test confirm the
decomposition is exact.

---

## Remaining Roadmap

### Phase 2: Integration polish

| Step | Task                                                             | File                                               | Status |
| ---- | ---------------------------------------------------------------- | -------------------------------------------------- | ------ |
| 1    | Wire XLA scan via config flag                                    | `gpt.py`, `base_train.py`, `checkpoint_manager.py` | TODO   |
| 2    | Add backend parity tests (ref vs Triton)                         | `tests/test_mamba_parity.py`                       | TODO   |
| 3    | Add prefill/decode drift smoke test                              | `tests/test_mamba_parity.py`                       | TODO   |
| 4    | Surface `--rope_theta`, `--window_long`, `--window_short` in CLI | `scripts/base_train.py`                            | TODO   |

### Phase 3: Performance validation

| Step | Task                                    | File                              | Status |
| ---- | --------------------------------------- | --------------------------------- | ------ |
| 5    | Add benchmark harness (tok/s, peak mem) | `scripts/bench_mamba_backends.py` | TODO   |
| 6    | Run small all-inclusive training        | (training script)                 | TODO   |
| 7    | TPU v6e validation                      | (remote)                          | TODO   |

---

## Definition of Done

| Test Suite                        | Count        | Status       |
| --------------------------------- | ------------ | ------------ |
| `tests/test_mamba_bugs.py`        | 8 tests      | ALL PASS     |
| `tests/test_mamba_integration.py` | 41 tests     | ALL PASS     |
| `tests/test_idea_g_math.py`       | 3 tests      | ALL PASS     |
| **Total**                         | **52 tests** | **ALL PASS** |

---

## Files Modified in This Session

| File                                 | Changes|            ------------------------------------------------------------------------------------------------------------------------------- |
| `nanochat/mamba2.py`                 | Bug #21 fix (RoPE split), Bug #22 fix (prev_x fp32), Bug #23 comment, `_run_scan` 
|                                      | abstraction, D=None support, Idea G dual-scan  
| `tests/test_mamba_bugs.py`           | NEW: 8 bug reproduction tests
| `tests/test_idea_g_math.py`          | NEW: 3 mathematical verification tests 
| `docs/mamba_review_followup_plan.md` | This document (v3) 
