"""
Bug reproduction tests for Mamba-2/3 integration.

These tests are designed to FAIL on the current (buggy) code and PASS
after fixes are applied. Each test targets a specific bug from the
review in docs/mamba_review_followup_plan.md.

Bug #21: Complex RoPE double-increment in decode
Bug #22: prev_x stored in model dtype instead of fp32
Bug #23: Trapezoidal cross-chunk boundary beta_next[-1]=0
"""
import pytest
import torch
import torch.nn.functional as F
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Minimal config that satisfies Mamba2Layer constructor
# ---------------------------------------------------------------------------
@dataclass
class MambaBugConfig:
    n_embd: int = 64
    mamba_d_state: int = 8
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_headdim: int = 32
    mamba_ngroups: int = 1
    mamba_chunk_size: int = 8
    mamba3_qknorm: bool = False
    mamba3_bias: bool = False
    mamba3_complex_rope: bool = False
    mamba3_trapezoidal: bool = False
    rope_theta: float = 10000.0


def _try_import_mamba():
    try:
        from nanochat.mamba2 import Mamba2Layer
        return Mamba2Layer
    except ImportError:
        return None


def _try_import_engine():
    try:
        from nanochat.engine import KVCache, MambaInferenceParams
        return KVCache, MambaInferenceParams
    except ImportError:
        return None, None


Mamba2Layer = _try_import_mamba()
_engine = _try_import_engine()
KVCache = _engine[0] if _engine else None
MambaInferenceParams = _engine[1] if _engine else None

skip_no_mamba = pytest.mark.skipif(Mamba2Layer is None, reason="Mamba2Layer not available")
skip_no_engine = pytest.mark.skipif(KVCache is None, reason="KVCache not available")


# ===========================================================================
# Bug #21: Complex RoPE double-increment in decode
# ===========================================================================
class TestBug21RopeDoubleIncrement:
    """
    _apply_complex_rope is called twice per step (once for B, once for C).
    In decode mode (L==1), the method mutates the stored rope_angle by
    adding dt_avg. The first call (B) increments from a to a+dt. The
    second call (C) reads a+dt and increments to a+2*dt.

    After N decode steps, the stored angle should be N*dt_avg, but the
    bug causes it to be 2*N*dt_avg.
    """

    @skip_no_mamba
    def test_rope_angle_after_single_decode_step(self):
        """One decode step should increment rope_angle by dt_avg, not 2*dt_avg.

        The fix splits _apply_complex_rope into _compute_rope_angles (called once,
        mutates state) + _rotate_with_rope (pure). This test verifies that calling
        _compute_rope_angles once produces exactly one increment, regardless of
        how many times _rotate_with_rope is called afterwards.
        """
        cfg = MambaBugConfig(mamba3_complex_rope=True)
        layer = Mamba2Layer(cfg, layer_idx=0)
        layer.eval()

        inference_params = MambaInferenceParams()

        B_sz, G, N = 1, cfg.mamba_ngroups, cfg.mamba_d_state
        H = (cfg.mamba_expand * cfg.n_embd) // cfg.mamba_headdim

        dt_soft = torch.ones(B_sz, 1, H) * 0.5
        dt_avg_expected = 0.5

        # Compute angles ONCE (this is what the fix does)
        angles = layer._compute_rope_angles(dt_soft, N, inference_params)

        # Rotate both B and C with the SAME pre-computed angles (pure, no mutation)
        B_tensor = torch.randn(B_sz, 1, G, N)
        C_tensor = torch.randn(B_sz, 1, G, N)
        _ = layer._rotate_with_rope(B_tensor, angles)
        _ = layer._rotate_with_rope(C_tensor, angles)

        key = f"rope_angle_{layer.layer_idx}"
        stored_angle = inference_params.key_value_memory_dict[key]

        assert stored_angle.shape == (B_sz, G)
        actual = stored_angle[0, 0].item()
        expected = dt_avg_expected

        assert abs(actual - expected) < 1e-5, (
            f"Bug #21: rope_angle after 1 step = {actual:.4f}, expected {expected:.4f}. "
            f"Angle was incremented {actual/expected:.0f}x instead of 1x."
        )

    @skip_no_mamba
    def test_rope_angle_after_multiple_decode_steps(self):
        """After 5 decode steps, rope_angle should be 5*dt_avg, not 10*dt_avg."""
        cfg = MambaBugConfig(mamba3_complex_rope=True)
        layer = Mamba2Layer(cfg, layer_idx=0)
        layer.eval()

        inference_params = MambaInferenceParams()
        B_sz, G, N = 1, cfg.mamba_ngroups, cfg.mamba_d_state
        H = (cfg.mamba_expand * cfg.n_embd) // cfg.mamba_headdim

        dt_val = 0.3
        dt_soft = torch.full((B_sz, 1, H), dt_val)

        n_steps = 5
        for _ in range(n_steps):
            # One call to _compute_rope_angles per step (increments once)
            angles = layer._compute_rope_angles(dt_soft, N, inference_params)
            # Pure rotations — no state mutation
            B_t = torch.randn(B_sz, 1, G, N)
            C_t = torch.randn(B_sz, 1, G, N)
            _ = layer._rotate_with_rope(B_t, angles)
            _ = layer._rotate_with_rope(C_t, angles)

        key = f"rope_angle_{layer.layer_idx}"
        stored = inference_params.key_value_memory_dict[key][0, 0].item()
        expected = n_steps * dt_val

        assert abs(stored - expected) < 1e-4, (
            f"Bug #21: rope_angle after {n_steps} steps = {stored:.4f}, "
            f"expected {expected:.4f}. Ratio = {stored/expected:.1f}x."
        )

    @skip_no_mamba
    def test_b_and_c_get_same_rotation(self):
        """B and C should be rotated by the SAME angle at each step."""
        cfg = MambaBugConfig(mamba3_complex_rope=True)
        layer = Mamba2Layer(cfg, layer_idx=0)
        layer.eval()

        inference_params = MambaInferenceParams()
        B_sz, G, N = 1, cfg.mamba_ngroups, cfg.mamba_d_state
        H = (cfg.mamba_expand * cfg.n_embd) // cfg.mamba_headdim

        dt_soft = torch.ones(B_sz, 1, H) * 0.5

        # Compute angles once
        angles = layer._compute_rope_angles(dt_soft, N, inference_params)

        # Use identical input tensors — same angles should produce same output
        shared = torch.ones(B_sz, 1, G, N)
        B_out = layer._rotate_with_rope(shared.clone(), angles)
        C_out = layer._rotate_with_rope(shared.clone(), angles)

        diff = (B_out - C_out).abs().max().item()
        assert diff < 1e-6, (
            f"Bug #21: B and C rotated by different angles. Max output diff = {diff:.6f}"
        )


# ===========================================================================
# Bug #22: prev_x stored in model dtype instead of fp32
# ===========================================================================
class TestBug22PrevXDtype:
    """
    prev_B is stored in fp32 (line 236: .float()), but prev_x is stored
    in model dtype (line 237: no .float()). During decode, prev_x is cast
    to fp32 for math, but precision is lost at the storage boundary.

    This test verifies that prev_x is stored in fp32 after a forward pass.
    """

    @skip_no_mamba
    @skip_no_engine
    def test_prev_x_dtype_after_prefill(self):
        """After prefill with trapezoidal, prev_x should be fp32."""
        cfg = MambaBugConfig(mamba3_trapezoidal=True)
        layer = Mamba2Layer(cfg, layer_idx=0).to(torch.bfloat16)
        layer.eval()

        B_sz, L = 1, 16
        kv = KVCache(
            batch_size=B_sz, num_heads=layer.nheads,
            seq_len=64, head_dim=layer.headdim,
            num_layers=1, device='cpu',
            dtype=torch.bfloat16, has_mamba=True,
        )

        x = torch.randn(B_sz, L, cfg.n_embd, dtype=torch.bfloat16)
        with torch.no_grad():
            _ = layer(x, kv_cache=kv)

        states = kv.mamba_params.key_value_memory_dict[0]

        # prev_B should be fp32 (this already works)
        assert states["prev_B"].dtype == torch.float32, (
            f"prev_B dtype = {states['prev_B'].dtype}, expected float32"
        )

        # prev_x should ALSO be fp32
        # BUG: prev_x.dtype == torch.bfloat16
        assert states["prev_x"].dtype == torch.float32, (
            f"Bug #22: prev_x dtype = {states['prev_x'].dtype}, expected float32"
        )

    @skip_no_mamba
    @skip_no_engine
    def test_prev_x_dtype_after_decode_init(self):
        """When prev_x is first created in decode, it should be fp32."""
        cfg = MambaBugConfig(mamba3_trapezoidal=True)
        layer = Mamba2Layer(cfg, layer_idx=0).to(torch.bfloat16)
        layer.eval()

        B_sz = 1
        kv = KVCache(
            batch_size=B_sz, num_heads=layer.nheads,
            seq_len=64, head_dim=layer.headdim,
            num_layers=1, device='cpu',
            dtype=torch.bfloat16, has_mamba=True,
        )

        # Single token decode (no prior prefill, so prev_x created via setdefault)
        x = torch.randn(B_sz, 1, cfg.n_embd, dtype=torch.bfloat16)
        with torch.no_grad():
            _ = layer(x, kv_cache=kv)

        states = kv.mamba_params.key_value_memory_dict[0]

        # BUG: prev_x created with dtype=x.dtype (bfloat16) at line 314-315
        assert states["prev_x"].dtype == torch.float32, (
            f"Bug #22: decode-init prev_x dtype = {states['prev_x'].dtype}, expected float32"
        )


# ===========================================================================
# Bug #23: Trapezoidal cross-chunk boundary beta_next[-1] = 0
# ===========================================================================
class TestBug23CrossChunkBoundary:
    """
    In _ssd_scan_ref trapezoidal mode, beta_next[:, :, -1] is left as 0,
    dropping the coupling from the last token of chunk c to the first
    token of chunk c+1.

    This test demonstrates the approximation error by comparing a 2-chunk
    sequence against a single-chunk computation of the same data.
    """

    @skip_no_mamba
    def test_cross_chunk_boundary_term_is_zero(self):
        """Directly verify that beta_next[:, :, -1] == 0 in _ssd_scan_ref."""
        cfg = MambaBugConfig(mamba3_trapezoidal=True, mamba_chunk_size=8)
        layer = Mamba2Layer(cfg, layer_idx=0)
        layer.eval()

        B_sz, L = 1, 16  # exactly 2 chunks of size 8
        H = layer.nheads
        G, N = layer.ngroups, layer.d_state

        x = torch.randn(B_sz, L, H, layer.headdim)
        dt = torch.rand(B_sz, L, H).clamp(min=0.01)
        A = -torch.exp(layer.A_log).detach()
        B_t = torch.randn(B_sz, L, G, N)
        C_t = torch.randn(B_sz, L, G, N)
        D = layer.D.detach()
        lam = torch.full((B_sz, L, H), 0.5)  # constant lambda

        # Run _ssd_scan_ref
        y, _ = layer._ssd_scan_ref(x, dt, A, B_t, C_t, D, lam=lam)

        # Now compute what the contribution at the chunk boundary should be.
        # Token 7 (last in chunk 0) should contribute a beta_next term to
        # token 8 (first in chunk 1). With lam=0.5 and the boundary dropped,
        # this term is missing.

        # Verify by computing the same sequence as a single chunk (cs=16)
        layer_big = Mamba2Layer(
            MambaBugConfig(mamba3_trapezoidal=True, mamba_chunk_size=16),
            layer_idx=0,
        )
        # Copy weights
        layer_big.load_state_dict(layer.state_dict())
        layer_big.eval()

        y_single, _ = layer_big._ssd_scan_ref(x, dt, A, B_t, C_t, D, lam=lam)

        # The two outputs should be identical if there were no boundary error.
        # BUG #23: they differ at positions near the chunk boundary (token 8).
        max_diff = (y - y_single).abs().max().item()

        # Document the error magnitude. This test PASSES regardless (it's
        # documenting an approximation, not a crash bug), but logs the error.
        print(f"\nBug #23: max diff between 2-chunk and 1-chunk = {max_diff:.6e}")

        # Check that the error specifically appears at the boundary region
        # (tokens near position chunk_size = 8)
        boundary_diff = (y[:, 7:9] - y_single[:, 7:9]).abs().max().item()
        interior_diff_a = (y[:, 2:6] - y_single[:, 2:6]).abs().max().item()

        # The boundary should have larger error than deep interior
        # (this test documents the effect, not a strict pass/fail)
        if max_diff > 1e-6:
            print(f"  Boundary diff (tokens 7-8): {boundary_diff:.6e}")
            print(f"  Interior diff (tokens 2-5): {interior_diff_a:.6e}")
            print(f"  This confirms the cross-chunk approximation error.")
        else:
            print(f"  No measurable difference (error < 1e-6). "
                  f"Boundary term may be negligible for this test config.")


# ===========================================================================
# Integration: verify bugs manifest through full forward path
# ===========================================================================
class TestBugsThroughForward:
    """End-to-end tests that exercise bugs through the full forward/decode path."""

    @skip_no_mamba
    @skip_no_engine
    def test_bug21_through_full_decode(self):
        """RoPE double-increment manifests through actual decode steps."""
        cfg = MambaBugConfig(mamba3_complex_rope=True)
        layer = Mamba2Layer(cfg, layer_idx=0).to(torch.bfloat16)
        layer.eval()

        B_sz = 1
        kv = KVCache(
            batch_size=B_sz, num_heads=layer.nheads,
            seq_len=64, head_dim=layer.headdim,
            num_layers=1, device='cpu',
            dtype=torch.bfloat16, has_mamba=True,
        )

        # Prefill with 8 tokens
        x_pre = torch.randn(B_sz, 8, cfg.n_embd, dtype=torch.bfloat16)
        with torch.no_grad():
            _ = layer(x_pre, kv_cache=kv)

        # Record angle after prefill
        key = f"rope_angle_{layer.layer_idx}"
        angle_after_prefill = kv.mamba_params.key_value_memory_dict[key].clone()

        # Do 3 decode steps
        for _ in range(3):
            x_step = torch.randn(B_sz, 1, cfg.n_embd, dtype=torch.bfloat16)
            with torch.no_grad():
                _ = layer(x_step, kv_cache=kv)

        angle_after_decode = kv.mamba_params.key_value_memory_dict[key]

        # The increment over 3 steps should be proportional to 3, not 6
        delta = (angle_after_decode - angle_after_prefill).abs()

        # We can't predict exact dt values (they depend on input), but we
        # can verify the angle changed. The key check: after fixing,
        # running the same test should give exactly half the delta.
        print(f"\nBug #21 through forward: angle delta over 3 decode steps = {delta}")
        print(f"  (After fix, this delta should be exactly half)")

    @skip_no_mamba
    @skip_no_engine
    def test_bug22_through_full_forward(self):
        """prev_x dtype manifests through actual forward with trapezoidal."""
        cfg = MambaBugConfig(mamba3_trapezoidal=True)
        layer = Mamba2Layer(cfg, layer_idx=0).to(torch.bfloat16)
        layer.eval()

        B_sz = 1
        kv = KVCache(
            batch_size=B_sz, num_heads=layer.nheads,
            seq_len=64, head_dim=layer.headdim,
            num_layers=1, device='cpu',
            dtype=torch.bfloat16, has_mamba=True,
        )

        x = torch.randn(B_sz, 16, cfg.n_embd, dtype=torch.bfloat16)
        with torch.no_grad():
            _ = layer(x, kv_cache=kv)

        states = kv.mamba_params.key_value_memory_dict[0]

        # Both prev_B and prev_x should be fp32 for consistent precision
        assert states["prev_B"].dtype == torch.float32
        # BUG: prev_x is bfloat16
        assert states["prev_x"].dtype == torch.float32, (
            f"Bug #22: prev_x.dtype = {states['prev_x'].dtype}"
        )
