"""
Backend parity tests: ref vs Triton (shared intermediates), and prefill/decode drift smoke test.

These tests verify:
1. _ssd_scan_ref and mamba_chunk_scan_combined produce matching outputs + final states
2. Long decode sequences don't drift or explode
3. Complex RoPE angles are consistent between prefill and incremental decode
"""
import pytest
import torch
from dataclasses import dataclass

from nanochat.mamba2 import Mamba2Layer, mamba_chunk_scan_combined
from nanochat.engine import KVCache


@dataclass
class ParityConfig:
    n_embd: int = 128
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_headdim: int = 32
    mamba_ngroups: int = 1
    mamba_chunk_size: int = 32
    mamba3_qknorm: bool = False
    mamba3_bias: bool = False
    mamba3_complex_rope: bool = False
    mamba3_trapezoidal: bool = False
    mamba_xla_scan: bool = False
    rope_theta: float = 10000.0


# ---------------------------------------------------------------------------
# Section 1: Backend parity (ref vs Triton) with shared intermediate tensors
# ---------------------------------------------------------------------------

class TestStandardScanParity:
    """Compare _ssd_scan_ref vs Triton mamba_chunk_scan_combined on identical inputs."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_ref_vs_triton_output_and_state(self):
        if mamba_chunk_scan_combined is None:
            pytest.skip("mamba_ssm Triton kernel unavailable")

        cfg = ParityConfig(mamba3_trapezoidal=False)
        layer = Mamba2Layer(cfg, layer_idx=0).cuda().bfloat16()
        layer.eval()

        B_sz, L = 2, 64
        # Generate shared intermediates that both backends consume
        x_ssm = torch.randn(B_sz, L, layer.nheads, layer.headdim, device="cuda", dtype=torch.bfloat16)
        dt = torch.rand(B_sz, L, layer.nheads, device="cuda", dtype=torch.bfloat16).clamp(min=0.01)
        A = -torch.exp(layer.A_log).detach()
        B = torch.randn(B_sz, L, layer.ngroups, layer.d_state, device="cuda", dtype=torch.bfloat16)
        C = torch.randn(B_sz, L, layer.ngroups, layer.d_state, device="cuda", dtype=torch.bfloat16)
        D = layer.D.detach()

        with torch.no_grad():
            # Reference path
            y_ref, s_ref = layer._ssd_scan_ref(x_ssm, dt, A, B, C, D)

            # Triton path
            y_tri, s_tri = mamba_chunk_scan_combined(
                x_ssm, dt, A, B, C,
                chunk_size=layer.chunk_size, D=D, return_final_states=True,
            )
            # Triton returns (B,H,headdim,d_state) -> our convention (B,H,d_state,headdim)
            s_tri = s_tri.transpose(-1, -2).contiguous()

        torch.testing.assert_close(y_ref.float(), y_tri.float(), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(s_ref.float(), s_tri.float(), atol=2e-2, rtol=2e-2)
        assert s_ref.dtype == torch.float32, f"Ref state dtype should be fp32, got {s_ref.dtype}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_ref_vs_triton_no_D(self):
        """Parity with D=None (used by Idea G dual-scan)."""
        if mamba_chunk_scan_combined is None:
            pytest.skip("mamba_ssm Triton kernel unavailable")

        cfg = ParityConfig()
        layer = Mamba2Layer(cfg, layer_idx=0).cuda().bfloat16()
        layer.eval()

        B_sz, L = 2, 64
        x_ssm = torch.randn(B_sz, L, layer.nheads, layer.headdim, device="cuda", dtype=torch.bfloat16)
        dt = torch.rand(B_sz, L, layer.nheads, device="cuda", dtype=torch.bfloat16).clamp(min=0.01)
        A = -torch.exp(layer.A_log).detach()
        B = torch.randn(B_sz, L, layer.ngroups, layer.d_state, device="cuda", dtype=torch.bfloat16)
        C = torch.randn(B_sz, L, layer.ngroups, layer.d_state, device="cuda", dtype=torch.bfloat16)

        with torch.no_grad():
            y_ref, s_ref = layer._ssd_scan_ref(x_ssm, dt, A, B, C, D=None)
            y_tri = mamba_chunk_scan_combined(
                x_ssm, dt, A, B, C,
                chunk_size=layer.chunk_size, D=None,
            )

        torch.testing.assert_close(y_ref.float(), y_tri.float(), atol=2e-2, rtol=2e-2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_run_scan_routes_to_triton_on_cuda(self):
        """Verify _run_scan uses Triton on CUDA and matches ref."""
        if mamba_chunk_scan_combined is None:
            pytest.skip("mamba_ssm Triton kernel unavailable")

        cfg = ParityConfig()
        layer = Mamba2Layer(cfg, layer_idx=0).cuda().bfloat16()
        layer.eval()

        B_sz, L = 2, 64
        x_ssm = torch.randn(B_sz, L, layer.nheads, layer.headdim, device="cuda", dtype=torch.bfloat16)
        dt = torch.rand(B_sz, L, layer.nheads, device="cuda", dtype=torch.bfloat16).clamp(min=0.01)
        A = -torch.exp(layer.A_log).detach()
        B = torch.randn(B_sz, L, layer.ngroups, layer.d_state, device="cuda", dtype=torch.bfloat16)
        C = torch.randn(B_sz, L, layer.ngroups, layer.d_state, device="cuda", dtype=torch.bfloat16)
        D = layer.D.detach()

        with torch.no_grad():
            y_scan, s_scan = layer._run_scan(x_ssm, dt, A, B, C, D=D, return_final_states=True)
            y_ref, s_ref = layer._ssd_scan_ref(x_ssm, dt, A, B, C, D)

        torch.testing.assert_close(y_scan.float(), y_ref.float(), atol=2e-2, rtol=2e-2)
        assert s_scan.dtype == torch.float32


# ---------------------------------------------------------------------------
# Section 2: Prefill/decode streaming stability
# ---------------------------------------------------------------------------

class TestStreamingStability:
    """Verify Mamba state doesn't explode over many decode steps."""

    def _make_kv_cache(self, layer, seq_len=256, batch_size=1, device="cpu"):
        return KVCache(
            batch_size=batch_size,
            num_heads=layer.nheads,
            seq_len=seq_len,
            head_dim=layer.headdim,
            num_layers=1,
            device=device,
            has_mamba=True,
        )

    def test_standard_prefill_then_decode_64_steps(self):
        """Standard (non-trapezoidal) 64-step decode after prefill."""
        cfg = ParityConfig(mamba3_trapezoidal=False)
        layer = Mamba2Layer(cfg, layer_idx=0).bfloat16()
        layer.eval()

        cache = self._make_kv_cache(layer)
        x_pre = torch.randn(1, 32, cfg.n_embd, dtype=torch.bfloat16)

        with torch.no_grad():
            y_pre = layer(x_pre, kv_cache=cache)
        assert torch.isfinite(y_pre).all(), "Prefill output has NaN/Inf"

        for i in range(64):
            x_step = torch.randn(1, 1, cfg.n_embd, dtype=torch.bfloat16)
            with torch.no_grad():
                y_step = layer(x_step, kv_cache=cache)
            assert torch.isfinite(y_step).all(), f"Decode step {i}: NaN/Inf"
            assert y_step.abs().max() < 100.0, f"Decode step {i}: output exploded ({y_step.abs().max():.1f})"

        state = cache.mamba_params.key_value_memory_dict[0]["ssm_state"]
        assert state.dtype == torch.float32
        assert torch.isfinite(state).all(), "Final state has NaN/Inf"

    def test_trapezoidal_prefill_then_decode_64_steps(self):
        """Trapezoidal + complex_rope 64-step decode."""
        cfg = ParityConfig(mamba3_trapezoidal=True, mamba3_complex_rope=True)
        layer = Mamba2Layer(cfg, layer_idx=0).bfloat16()
        layer.eval()

        cache = self._make_kv_cache(layer)
        x_pre = torch.randn(1, 32, cfg.n_embd, dtype=torch.bfloat16)

        with torch.no_grad():
            y_pre = layer(x_pre, kv_cache=cache)
        assert torch.isfinite(y_pre).all()

        for i in range(64):
            x_step = torch.randn(1, 1, cfg.n_embd, dtype=torch.bfloat16)
            with torch.no_grad():
                y_step = layer(x_step, kv_cache=cache)
            assert torch.isfinite(y_step).all(), f"Trap decode step {i}: NaN/Inf"
            assert y_step.abs().max() < 100.0, f"Trap decode step {i}: exploded"

        state = cache.mamba_params.key_value_memory_dict[0]["ssm_state"]
        assert state.dtype == torch.float32
        assert torch.isfinite(state).all()

        # Verify trapezoidal state caches exist and are fp32
        prev_x = cache.mamba_params.key_value_memory_dict[0]["prev_x"]
        prev_B = cache.mamba_params.key_value_memory_dict[0]["prev_B"]
        assert prev_x.dtype == torch.float32, f"prev_x dtype {prev_x.dtype}"
        assert prev_B.dtype == torch.float32, f"prev_B dtype {prev_B.dtype}"

    def test_rope_angle_prefill_vs_incremental(self):
        """Verify complex RoPE angle is consistent: full prefill vs partial prefill + decode steps."""
        cfg = ParityConfig(mamba3_complex_rope=True, mamba3_trapezoidal=False)
        layer = Mamba2Layer(cfg, layer_idx=0).bfloat16()
        layer.eval()

        tokens = torch.randn(1, 16, cfg.n_embd, dtype=torch.bfloat16)
        angle_key = f"rope_angle_{layer.layer_idx}"

        # Full prefill: 16 tokens at once
        cache_full = self._make_kv_cache(layer)
        with torch.no_grad():
            layer(tokens, kv_cache=cache_full)
        angle_full = cache_full.mamba_params.key_value_memory_dict[angle_key].clone()

        # Partial: 8 prefill + 8 decode steps
        cache_inc = self._make_kv_cache(layer)
        with torch.no_grad():
            layer(tokens[:, :8], kv_cache=cache_inc)
            for i in range(8, 16):
                layer(tokens[:, i:i+1], kv_cache=cache_inc)
        angle_inc = cache_inc.mamba_params.key_value_memory_dict[angle_key]

        # bf16 cumsum vs sequential fp32 accumulation causes small drift
        torch.testing.assert_close(angle_full.float(), angle_inc.float(), atol=5e-3, rtol=5e-3)

    def test_all_phase2_features_decode_stability(self):
        """All Phase 2 features enabled during decode."""
        cfg = ParityConfig(
            mamba3_qknorm=True,
            mamba3_bias=True,
            mamba3_complex_rope=True,
            mamba3_trapezoidal=True,
        )
        layer = Mamba2Layer(cfg, layer_idx=0).bfloat16()
        layer.eval()

        cache = self._make_kv_cache(layer)
        x_pre = torch.randn(1, 32, cfg.n_embd, dtype=torch.bfloat16)

        with torch.no_grad():
            layer(x_pre, kv_cache=cache)

        for i in range(32):
            x_step = torch.randn(1, 1, cfg.n_embd, dtype=torch.bfloat16)
            with torch.no_grad():
                y_step = layer(x_step, kv_cache=cache)
            assert torch.isfinite(y_step).all(), f"All-features decode step {i}: NaN/Inf"
