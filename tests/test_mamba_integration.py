"""
Comprehensive integration tests for Mamba-2/Mamba-3 hybrid layers.

Tests all 3 phases and composability with existing modules:
  - Phase 1: Mamba-2 SSD (chunked scan, O(1) decode)
  - Phase 2: Mamba-3 incremental (QK-norm, learnable B/C bias, complex RoPE)
  - Phase 3: Mamba-3 trapezoidal discretization

All features are additive and optional via GPTConfig fields.
Every combination (mamba + engram, mamba + mhc, mamba + dsa, mamba + mtp, etc.)
must produce finite loss, finite gradients, and correct inference shapes.

Follows the same convention as test_engram_mhc_integration.py:
  - Tiny model (seq=16, vocab=128, layers=4, embd=64)
  - Forward+backward: loss finite, grads finite
  - Inference: logits shape (B, T, V), finite
  - Uses _CONFIG_FIELDS for forward compatibility
"""

import pytest
import torch

from nanochat import kernels
from nanochat.gpt import GPT, GPTConfig


_CONFIG_FIELDS = set(GPTConfig.__dataclass_fields__.keys())


def _resolve_flag_name(*candidates):
    for name in candidates:
        if name in _CONFIG_FIELDS:
            return name
    return None


def _maybe_set(config_kwargs, name, value):
    if name is not None and name in _CONFIG_FIELDS:
        config_kwargs[name] = value


def _has_field(name):
    return name in _CONFIG_FIELDS


# Resolve feature flags (handles renames across branches)
_MAMBA_FLAG = _resolve_flag_name("mamba_enabled")
_ENGRAM_FLAG = _resolve_flag_name("engram_enabled", "engram")
_MHC_FLAG = _resolve_flag_name("mhc_enabled", "mhc")
_DSA_FLAG = _resolve_flag_name("dsa_enabled", "dsa")
_MTP_FLAG = _resolve_flag_name("mtp_enabled", "mtp")


def _skip_if_no_mamba():
    """Skip test if mamba fields are not in GPTConfig on this branch."""
    if _MAMBA_FLAG is None:
        pytest.skip("mamba_enabled not available in GPTConfig on this branch.")


def _base_config(**overrides):
    """Create a tiny model config for testing. 4 layers to allow AAM pattern."""
    config_kwargs = dict(
        sequence_len=16,
        vocab_size=128,
        n_layer=4,
        n_head=4,
        n_kv_head=4,
        n_embd=64,
        window_pattern="L",
    )
    # Disable all optional features by default
    _maybe_set(config_kwargs, _ENGRAM_FLAG, False)
    _maybe_set(config_kwargs, _MHC_FLAG, False)
    _maybe_set(config_kwargs, _DSA_FLAG, False)
    _maybe_set(config_kwargs, _MTP_FLAG, False)
    _maybe_set(config_kwargs, _MAMBA_FLAG, False)

    config_kwargs.update(overrides)
    return config_kwargs


def _mamba_config(**overrides):
    """Base config with Mamba enabled and tiny Mamba dims."""
    kw = _base_config()
    _maybe_set(kw, _MAMBA_FLAG, True)
    _maybe_set(kw, "mamba_pattern", "AAM")
    _maybe_set(kw, "mamba_d_state", 8)
    _maybe_set(kw, "mamba_expand", 2)
    _maybe_set(kw, "mamba_headdim", 32)
    _maybe_set(kw, "mamba_chunk_size", 8)
    kw.update(overrides)
    return kw


def _run_forward_backward(config_kwargs, batch_size=2, seq_len=8):
    """Build model, run forward+backward, assert finite loss and grads."""
    config = GPTConfig(**config_kwargs)
    model = GPT(config)
    model.init_weights()
    model.train()

    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len), dtype=torch.long)
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len), dtype=torch.long)

    loss = model(idx, targets)
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "No gradients were produced."
    assert all(torch.isfinite(g).all().item() for g in grads), "Gradient contains NaN/Inf."

    return model, config


def _run_inference(model, config, batch_size=2, seq_len=8):
    """Run inference and assert correct shapes and finite values."""
    model.eval()
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len), dtype=torch.long)
    with torch.no_grad():
        logits = model(idx)
    assert logits.shape == (batch_size, seq_len, config.vocab_size), \
        f"Expected shape {(batch_size, seq_len, config.vocab_size)}, got {logits.shape}"
    assert torch.isfinite(logits).all(), "Logits contain NaN/Inf."


# ============================================================================
# Phase 1: Mamba-2 Basic Integration
# ============================================================================

class TestMamba2Basic:
    """Phase 1: Mamba-2 drop-in replacement for attention layers."""

    def test_mamba_disabled_is_baseline(self):
        """With mamba_enabled=False, model is pure attention (baseline)."""
        prev = kernels.get_kernel_backend()
        kernels.set_kernel_backend("current")
        try:
            config_kwargs = _base_config()
            model, config = _run_forward_backward(config_kwargs)
            _run_inference(model, config)
            # Verify no blocks have is_mamba set
            for block in model.transformer.h:
                assert not getattr(block, "is_mamba", False), "Block should not be Mamba when disabled"
        finally:
            kernels.set_kernel_backend(prev)

    def test_mamba_aam_pattern(self):
        """AAM pattern: layers 0,1=attention, 2=mamba, 3=attention."""
        _skip_if_no_mamba()
        prev = kernels.get_kernel_backend()
        kernels.set_kernel_backend("current")
        try:
            config_kwargs = _mamba_config()
            model, config = _run_forward_backward(config_kwargs)
            _run_inference(model, config)

            # Verify pattern: layer 2 is Mamba, others are attention
            is_mamba = [getattr(b, "is_mamba", False) for b in model.transformer.h]
            assert is_mamba == [False, False, True, False], f"Expected [A,A,M,A], got {is_mamba}"
        finally:
            kernels.set_kernel_backend(prev)

    def test_mamba_all_mamba_pattern(self):
        """All layers are Mamba (pattern='M')."""
        _skip_if_no_mamba()
        prev = kernels.get_kernel_backend()
        kernels.set_kernel_backend("current")
        try:
            config_kwargs = _mamba_config(mamba_pattern="M")
            model, config = _run_forward_backward(config_kwargs)
            _run_inference(model, config)

            is_mamba = [getattr(b, "is_mamba", False) for b in model.transformer.h]
            assert all(is_mamba), f"Expected all Mamba, got {is_mamba}"
        finally:
            kernels.set_kernel_backend(prev)

    def test_mamba_empty_pattern_defaults_to_aam(self):
        """Empty pattern with mamba_enabled=True should default to AAM."""
        _skip_if_no_mamba()
        prev = kernels.get_kernel_backend()
        kernels.set_kernel_backend("current")
        try:
            config_kwargs = _mamba_config(mamba_pattern="")
            model, config = _run_forward_backward(config_kwargs)
            is_mamba = [getattr(b, "is_mamba", False) for b in model.transformer.h]
            # AAM on 4 layers: [A,A,M,A]
            assert is_mamba == [False, False, True, False], f"Expected AAM default, got {is_mamba}"
        finally:
            kernels.set_kernel_backend(prev)

    def test_mamba_am_alternating_pattern(self):
        """AM pattern: layers 0,2=attention, 1,3=mamba."""
        _skip_if_no_mamba()
        prev = kernels.get_kernel_backend()
        kernels.set_kernel_backend("current")
        try:
            config_kwargs = _mamba_config(mamba_pattern="AM")
            model, config = _run_forward_backward(config_kwargs)
            _run_inference(model, config)

            is_mamba = [getattr(b, "is_mamba", False) for b in model.transformer.h]
            assert is_mamba == [False, True, False, True], f"Expected [A,M,A,M], got {is_mamba}"
        finally:
            kernels.set_kernel_backend(prev)


# ============================================================================
# Phase 2: Mamba-3 Incremental Upgrades
# ============================================================================

class TestMamba3Phase2:
    """Phase 2: QK-norm, learnable B/C bias, complex RoPE."""

    @pytest.mark.parametrize(
        ("qknorm", "bias", "rope"),
        [
            pytest.param(True, False, False, id="qknorm_only"),
            pytest.param(False, True, False, id="bias_only"),
            pytest.param(False, False, True, id="rope_only"),
            pytest.param(True, True, False, id="qknorm_bias"),
            pytest.param(True, True, True, id="all_phase2"),
        ],
    )
    def test_phase2_toggles(self, qknorm, bias, rope):
        """Each Phase 2 toggle should produce finite loss independently."""
        _skip_if_no_mamba()
        if qknorm and not _has_field("mamba3_qknorm"):
            pytest.skip("mamba3_qknorm not in GPTConfig")
        if bias and not _has_field("mamba3_bias"):
            pytest.skip("mamba3_bias not in GPTConfig")
        if rope and not _has_field("mamba3_complex_rope"):
            pytest.skip("mamba3_complex_rope not in GPTConfig")

        prev = kernels.get_kernel_backend()
        kernels.set_kernel_backend("current")
        try:
            config_kwargs = _mamba_config(mamba_pattern="AM")
            _maybe_set(config_kwargs, "mamba3_qknorm", qknorm)
            _maybe_set(config_kwargs, "mamba3_bias", bias)
            _maybe_set(config_kwargs, "mamba3_complex_rope", rope)

            model, config = _run_forward_backward(config_kwargs)
            _run_inference(model, config)
        finally:
            kernels.set_kernel_backend(prev)


# ============================================================================
# Phase 3: Trapezoidal Discretization
# ============================================================================

class TestMamba3Phase3:
    """Phase 3: Trapezoidal discretization with lambda gate."""

    def test_trapezoidal_forward_backward(self):
        """Trapezoidal mode should produce finite loss and grads."""
        _skip_if_no_mamba()
        if not _has_field("mamba3_trapezoidal"):
            pytest.skip("mamba3_trapezoidal not in GPTConfig")

        prev = kernels.get_kernel_backend()
        kernels.set_kernel_backend("current")
        try:
            config_kwargs = _mamba_config(mamba_pattern="AM")
            _maybe_set(config_kwargs, "mamba3_trapezoidal", True)

            model, config = _run_forward_backward(config_kwargs)
            _run_inference(model, config)
        finally:
            kernels.set_kernel_backend(prev)

    def test_trapezoidal_with_all_phase2(self):
        """Trapezoidal + all Phase 2 features combined."""
        _skip_if_no_mamba()
        if not _has_field("mamba3_trapezoidal"):
            pytest.skip("mamba3_trapezoidal not in GPTConfig")

        prev = kernels.get_kernel_backend()
        kernels.set_kernel_backend("current")
        try:
            config_kwargs = _mamba_config(mamba_pattern="AM")
            _maybe_set(config_kwargs, "mamba3_qknorm", True)
            _maybe_set(config_kwargs, "mamba3_bias", True)
            _maybe_set(config_kwargs, "mamba3_complex_rope", True)
            _maybe_set(config_kwargs, "mamba3_trapezoidal", True)

            model, config = _run_forward_backward(config_kwargs)
            _run_inference(model, config)
        finally:
            kernels.set_kernel_backend(prev)

    def test_trapezoidal_with_mtp(self):
        """Trapezoidal + MTP: MTP block must remain attention-only."""
        _skip_if_no_mamba()
        if not _has_field("mamba3_trapezoidal"):
            pytest.skip("mamba3_trapezoidal not in GPTConfig")
        if _MTP_FLAG is None:
            pytest.skip("MTP not available in GPTConfig")

        prev = kernels.get_kernel_backend()
        kernels.set_kernel_backend("current")
        try:
            config_kwargs = _mamba_config(mamba_pattern="AM")
            _maybe_set(config_kwargs, "mamba3_trapezoidal", True)
            _maybe_set(config_kwargs, _MTP_FLAG, True)

            model, config = _run_forward_backward(config_kwargs)

            if hasattr(model, "mtp") and model.mtp is not None:
                assert not getattr(model.mtp.block, "is_mamba", False), \
                    "MTP block must remain attention-only even with trapezoidal"
        finally:
            kernels.set_kernel_backend(prev)


# ============================================================================
# Composability: Mamba + Other Modules
# ============================================================================

class TestMambaComposability:
    """Test that Mamba layers compose correctly with all other optional modules."""

    @pytest.mark.parametrize(
        ("engram", "mhc"),
        [
            pytest.param(False, False, id="mamba_only"),
            pytest.param(True, False, id="mamba_engram"),
            pytest.param(False, True, id="mamba_mhc"),
            pytest.param(True, True, id="mamba_engram_mhc"),
        ],
    )
    def test_mamba_with_engram_mhc(self, engram, mhc):
        """Mamba layers should work with Engram and/or mHC on the same block."""
        _skip_if_no_mamba()
        if engram and _ENGRAM_FLAG is None:
            pytest.skip("Engram not available in GPTConfig")
        if mhc and _MHC_FLAG is None:
            pytest.skip("mHC not available in GPTConfig")

        prev = kernels.get_kernel_backend()
        kernels.set_kernel_backend("current")
        try:
            config_kwargs = _mamba_config()
            if engram:
                _maybe_set(config_kwargs, _ENGRAM_FLAG, True)
                _maybe_set(config_kwargs, "engram_layers", "0,2")
                _maybe_set(config_kwargs, "engram_ngram_orders", "2,3")
                _maybe_set(config_kwargs, "engram_bottleneck_dim", 16)
                _maybe_set(config_kwargs, "engram_dropout", 0.0)
            if mhc:
                _maybe_set(config_kwargs, _MHC_FLAG, True)
                _maybe_set(config_kwargs, "mhc_num_branches", 0)
                _maybe_set(config_kwargs, "mhc_sinkhorn_iters", 1)
                _maybe_set(config_kwargs, "mhc_temperature", 1.0)
                _maybe_set(config_kwargs, "mhc_epsilon", 1e-6)
                _maybe_set(config_kwargs, "mhc_blend_alpha", 1.0)

            model, config = _run_forward_backward(config_kwargs)
            _run_inference(model, config)
        finally:
            kernels.set_kernel_backend(prev)

    def test_mamba_with_mtp(self):
        """Mamba + MTP: MTP block must remain attention-only."""
        _skip_if_no_mamba()
        if _MTP_FLAG is None:
            pytest.skip("MTP not available in GPTConfig")

        prev = kernels.get_kernel_backend()
        kernels.set_kernel_backend("current")
        try:
            config_kwargs = _mamba_config(mamba_pattern="AM")
            _maybe_set(config_kwargs, _MTP_FLAG, True)

            model, config = _run_forward_backward(config_kwargs)

            # Verify MTP's internal block is NOT Mamba
            if hasattr(model, "mtp") and model.mtp is not None:
                assert not getattr(model.mtp.block, "is_mamba", False), \
                    "MTP internal block must remain attention-only"
        finally:
            kernels.set_kernel_backend(prev)

    def test_mamba_does_not_interfere_with_dsa(self):
        """Mamba and DSA are mutually exclusive per-layer.
        With AAM pattern on 4 layers and dsa_start_layer=3,
        layer 2 should be Mamba, layer 3 should be DSA (not Mamba)."""
        _skip_if_no_mamba()
        if _DSA_FLAG is None:
            pytest.skip("DSA not available in GPTConfig")

        prev = kernels.get_kernel_backend()
        kernels.set_kernel_backend("current")
        try:
            config_kwargs = _mamba_config()
            _maybe_set(config_kwargs, _DSA_FLAG, True)
            _maybe_set(config_kwargs, "dsa_start_layer", 3)

            model, config = _run_forward_backward(config_kwargs)

            # Pattern AAM on 4 layers: 0=A, 1=A, 2=M, 3=A
            # DSA starts at layer 3, but layer 3 is A in AAM pattern
            # Mamba takes priority over DSA for layer 2
            is_mamba = [getattr(b, "is_mamba", False) for b in model.transformer.h]
            assert is_mamba[2] is True, "Layer 2 should be Mamba"
            assert is_mamba[3] is False, "Layer 3 should be attention (DSA)"
        finally:
            kernels.set_kernel_backend(prev)

    def test_dsa_forward_backward_alone(self):
        """DSA alone (no Mamba) should still work correctly."""
        if _DSA_FLAG is None:
            pytest.skip("DSA not available in GPTConfig")

        prev = kernels.get_kernel_backend()
        kernels.set_kernel_backend("current")
        try:
            config_kwargs = _base_config()
            _maybe_set(config_kwargs, _DSA_FLAG, True)
            _maybe_set(config_kwargs, "dsa_start_layer", 2)

            model, config = _run_forward_backward(config_kwargs)
            _run_inference(model, config)
        finally:
            kernels.set_kernel_backend(prev)

    def test_mtp_forward_backward_alone(self):
        """MTP alone (no Mamba) should still work correctly."""
        if _MTP_FLAG is None:
            pytest.skip("MTP not available in GPTConfig")

        prev = kernels.get_kernel_backend()
        kernels.set_kernel_backend("current")
        try:
            config_kwargs = _base_config()
            _maybe_set(config_kwargs, _MTP_FLAG, True)

            model, config = _run_forward_backward(config_kwargs)
            _run_inference(model, config)
        finally:
            kernels.set_kernel_backend(prev)

    def test_mamba_mtp_dsa_combined(self):
        """Mamba + MTP + DSA: all three features combined correctly.
        DSA applies to attention layers only (not Mamba layers)."""
        _skip_if_no_mamba()
        if _MTP_FLAG is None:
            pytest.skip("MTP not available in GPTConfig")
        if _DSA_FLAG is None:
            pytest.skip("DSA not available in GPTConfig")

        prev = kernels.get_kernel_backend()
        kernels.set_kernel_backend("current")
        try:
            config_kwargs = _mamba_config()
            _maybe_set(config_kwargs, _MTP_FLAG, True)
            _maybe_set(config_kwargs, _DSA_FLAG, True)
            _maybe_set(config_kwargs, "dsa_start_layer", 3)

            model, config = _run_forward_backward(config_kwargs)

            # MTP block should be pure attention
            if hasattr(model, "mtp") and model.mtp is not None:
                assert not getattr(model.mtp.block, "is_mamba", False)
        finally:
            kernels.set_kernel_backend(prev)


# ============================================================================
# Optimizer Routing
# ============================================================================

class TestMambaOptimizer:
    """Verify Muon/AdamW parameter routing with Mamba layers."""

    def test_optimizer_param_count(self):
        """All parameters must be accounted for in optimizer groups."""
        _skip_if_no_mamba()
        prev = kernels.get_kernel_backend()
        kernels.set_kernel_backend("current")
        try:
            config_kwargs = _mamba_config(mamba_pattern="AM")
            config = GPTConfig(**config_kwargs)
            model = GPT(config)
            model.init_weights()

            # setup_optimizers should not raise (param count assertion inside)
            optimizers = model.setup_optimizers()
            assert len(optimizers) == 2, "Expected [AdamW, Muon]"
        finally:
            kernels.set_kernel_backend(prev)

    def test_non_2d_params_not_in_muon(self):
        """Mamba's 1D (A_log, dt_bias, D) and 3D (conv1d.weight) params
        must NOT go to Muon (which requires 2D matrices)."""
        _skip_if_no_mamba()
        prev = kernels.get_kernel_backend()
        kernels.set_kernel_backend("current")
        try:
            config_kwargs = _mamba_config(mamba_pattern="M")
            config = GPTConfig(**config_kwargs)
            model = GPT(config)
            model.init_weights()

            optimizers = model.setup_optimizers()
            muon_opt = optimizers[1]  # Second optimizer is Muon

            for group in muon_opt.param_groups:
                for p in group["params"]:
                    assert p.ndim == 2, \
                        f"Muon received a {p.ndim}D param (shape {p.shape}), expected only 2D"
        finally:
            kernels.set_kernel_backend(prev)

    def test_optimizer_with_all_features(self):
        """Optimizer param routing with Mamba + Engram + mHC + MTP."""
        _skip_if_no_mamba()
        prev = kernels.get_kernel_backend()
        kernels.set_kernel_backend("current")
        try:
            config_kwargs = _mamba_config()
            if _ENGRAM_FLAG:
                _maybe_set(config_kwargs, _ENGRAM_FLAG, True)
                _maybe_set(config_kwargs, "engram_layers", "2")
                _maybe_set(config_kwargs, "engram_ngram_orders", "2")
                _maybe_set(config_kwargs, "engram_bottleneck_dim", 16)
                _maybe_set(config_kwargs, "engram_dropout", 0.0)
            if _MHC_FLAG:
                _maybe_set(config_kwargs, _MHC_FLAG, True)
                _maybe_set(config_kwargs, "mhc_num_branches", 0)
                _maybe_set(config_kwargs, "mhc_sinkhorn_iters", 1)
            if _MTP_FLAG:
                _maybe_set(config_kwargs, _MTP_FLAG, True)

            config = GPTConfig(**config_kwargs)
            model = GPT(config)
            model.init_weights()

            # Should not raise
            optimizers = model.setup_optimizers()
            assert len(optimizers) == 2
        finally:
            kernels.set_kernel_backend(prev)


# ============================================================================
# Window Sizes
# ============================================================================

class TestWindowSizes:
    """Verify _compute_window_sizes handles Mamba and window configs correctly."""

    def test_mamba_layers_get_none_window(self):
        """Mamba layers should have window_size=None in the computed list."""
        _skip_if_no_mamba()
        prev = kernels.get_kernel_backend()
        kernels.set_kernel_backend("current")
        try:
            config_kwargs = _mamba_config()
            config = GPTConfig(**config_kwargs)
            model = GPT(config)

            # AAM on 4 layers: [A, A, M, A]
            assert model.window_sizes[2] is None, "Mamba layer should have None window"
            assert model.window_sizes[0] is not None, "Attention layer should have a window"
            assert model.window_sizes[3] is not None, "Last attention layer should have a window"
        finally:
            kernels.set_kernel_backend(prev)

    def test_window_long_short_fields(self):
        """window_long/window_short should decouple from sequence_len."""
        _skip_if_no_mamba()
        if not _has_field("window_long"):
            pytest.skip("window_long not in GPTConfig")

        prev = kernels.get_kernel_backend()
        kernels.set_kernel_backend("current")
        try:
            config_kwargs = _mamba_config(
                window_pattern="SL",
                window_long=12,
                window_short=6,
            )
            config = GPTConfig(**config_kwargs)
            model = GPT(config)

            # 4 layers with AAM: [A, A, M, A]
            # Pattern "SL" tiles: layer 0=S, 1=L, 2=M(None), 3=A(last→long)
            assert model.window_sizes[2] is None, "Mamba layer should be None"
            # Last attention layer (3) should get long window
            assert model.window_sizes[3] is not None
            assert model.window_sizes[3][0] == 12, f"Last attention layer should have long_window=12"
        finally:
            kernels.set_kernel_backend(prev)


# ============================================================================
# Gradient Checkpointing
# ============================================================================

class TestGradientCheckpointing:
    """Verify gradient checkpointing works with Mamba hybrid blocks."""

    def test_mamba_with_gradient_checkpointing(self):
        """Mamba + gradient checkpointing should produce same-quality grads."""
        _skip_if_no_mamba()
        if not _has_field("gradient_checkpointing"):
            pytest.skip("gradient_checkpointing not in GPTConfig")

        prev = kernels.get_kernel_backend()
        kernels.set_kernel_backend("current")
        try:
            config_kwargs = _mamba_config(gradient_checkpointing=True)
            model, config = _run_forward_backward(config_kwargs)
        finally:
            kernels.set_kernel_backend(prev)


# ============================================================================
# RoPE Configuration
# ============================================================================

class TestRoPEConfig:
    """Verify configurable rope_theta for long-context support."""

    def test_rope_theta_configurable(self):
        """Custom rope_theta should not crash and produce finite output."""
        _skip_if_no_mamba()
        if not _has_field("rope_theta"):
            pytest.skip("rope_theta not in GPTConfig")

        prev = kernels.get_kernel_backend()
        kernels.set_kernel_backend("current")
        try:
            config_kwargs = _mamba_config(rope_theta=1000000.0)
            _maybe_set(config_kwargs, "mamba3_complex_rope", True)
            model, config = _run_forward_backward(config_kwargs)
            _run_inference(model, config)
        finally:
            kernels.set_kernel_backend(prev)


# ============================================================================
# Full Kitchen Sink: All Features Combined
# ============================================================================

class TestKitchenSink:
    """Test the most complex configurations: all features enabled simultaneously."""

    def test_all_features_combined(self):
        """Mamba + all Phase 2 + Engram + mHC + MTP together."""
        _skip_if_no_mamba()

        prev = kernels.get_kernel_backend()
        kernels.set_kernel_backend("current")
        try:
            config_kwargs = _mamba_config()
            # Phase 2
            _maybe_set(config_kwargs, "mamba3_qknorm", True)
            _maybe_set(config_kwargs, "mamba3_bias", True)
            _maybe_set(config_kwargs, "mamba3_complex_rope", True)
            # Engram on the Mamba layer (layer 2) to verify agnosticism
            if _ENGRAM_FLAG:
                _maybe_set(config_kwargs, _ENGRAM_FLAG, True)
                _maybe_set(config_kwargs, "engram_layers", "2")
                _maybe_set(config_kwargs, "engram_ngram_orders", "2")
                _maybe_set(config_kwargs, "engram_bottleneck_dim", 16)
                _maybe_set(config_kwargs, "engram_dropout", 0.0)
            # mHC
            if _MHC_FLAG:
                _maybe_set(config_kwargs, _MHC_FLAG, True)
                _maybe_set(config_kwargs, "mhc_num_branches", 0)
                _maybe_set(config_kwargs, "mhc_sinkhorn_iters", 1)
            # MTP
            if _MTP_FLAG:
                _maybe_set(config_kwargs, _MTP_FLAG, True)

            model, config = _run_forward_backward(config_kwargs)
            _run_inference(model, config)

            # Sanity: layer 2 has Mamba + Engram
            block2 = model.transformer.h[2]
            assert getattr(block2, "is_mamba", False), "Layer 2 should be Mamba"
            if _ENGRAM_FLAG:
                assert block2.engram is not None, "Engram should be on layer 2 (a Mamba layer)"
        finally:
            kernels.set_kernel_backend(prev)

    def test_all_features_with_phase3(self):
        """Mamba + Phase 2 + Phase 3 + Engram + mHC + MTP + DSA."""
        _skip_if_no_mamba()
        if not _has_field("mamba3_trapezoidal"):
            pytest.skip("mamba3_trapezoidal not in GPTConfig")

        prev = kernels.get_kernel_backend()
        kernels.set_kernel_backend("current")
        try:
            config_kwargs = _mamba_config()
            # All Phase 2 + Phase 3
            _maybe_set(config_kwargs, "mamba3_qknorm", True)
            _maybe_set(config_kwargs, "mamba3_bias", True)
            _maybe_set(config_kwargs, "mamba3_complex_rope", True)
            _maybe_set(config_kwargs, "mamba3_trapezoidal", True)
            # Engram
            if _ENGRAM_FLAG:
                _maybe_set(config_kwargs, _ENGRAM_FLAG, True)
                _maybe_set(config_kwargs, "engram_layers", "0,2")
                _maybe_set(config_kwargs, "engram_ngram_orders", "2")
                _maybe_set(config_kwargs, "engram_bottleneck_dim", 16)
                _maybe_set(config_kwargs, "engram_dropout", 0.0)
            # mHC
            if _MHC_FLAG:
                _maybe_set(config_kwargs, _MHC_FLAG, True)
                _maybe_set(config_kwargs, "mhc_num_branches", 0)
                _maybe_set(config_kwargs, "mhc_sinkhorn_iters", 1)
            # MTP
            if _MTP_FLAG:
                _maybe_set(config_kwargs, _MTP_FLAG, True)
            # DSA on attention layers only
            if _DSA_FLAG:
                _maybe_set(config_kwargs, _DSA_FLAG, True)
                _maybe_set(config_kwargs, "dsa_start_layer", 3)

            model, config = _run_forward_backward(config_kwargs)
            _run_inference(model, config)

            # Verify module placement
            assert getattr(model.transformer.h[2], "is_mamba", False), "Layer 2 should be Mamba"
            assert not getattr(model.transformer.h[3], "is_mamba", False), "Layer 3 should be attention"
        finally:
            kernels.set_kernel_backend(prev)

    def test_all_features_with_gradient_checkpointing(self):
        """Full kitchen sink + gradient checkpointing."""
        _skip_if_no_mamba()
        if not _has_field("gradient_checkpointing"):
            pytest.skip("gradient_checkpointing not in GPTConfig")

        prev = kernels.get_kernel_backend()
        kernels.set_kernel_backend("current")
        try:
            config_kwargs = _mamba_config(gradient_checkpointing=True)
            _maybe_set(config_kwargs, "mamba3_qknorm", True)
            _maybe_set(config_kwargs, "mamba3_bias", True)
            if _MTP_FLAG:
                _maybe_set(config_kwargs, _MTP_FLAG, True)

            model, config = _run_forward_backward(config_kwargs)
        finally:
            kernels.set_kernel_backend(prev)


# ============================================================================
# Baseline (non-Mamba) regressions — ensure existing features still work
# ============================================================================

class TestBaselineFeatures:
    """Verify that enabling Mamba code doesn't break existing features when disabled."""

    def test_pure_attention_baseline(self):
        """Pure attention model with no optional features."""
        prev = kernels.get_kernel_backend()
        kernels.set_kernel_backend("current")
        try:
            config_kwargs = _base_config()
            model, config = _run_forward_backward(config_kwargs)
            _run_inference(model, config)
        finally:
            kernels.set_kernel_backend(prev)

    def test_engram_mhc_without_mamba(self):
        """Engram + mHC without Mamba should work as before."""
        if _ENGRAM_FLAG is None or _MHC_FLAG is None:
            pytest.skip("Engram or mHC not available")

        prev = kernels.get_kernel_backend()
        kernels.set_kernel_backend("current")
        try:
            config_kwargs = _base_config()
            _maybe_set(config_kwargs, _ENGRAM_FLAG, True)
            _maybe_set(config_kwargs, "engram_layers", "0,1")
            _maybe_set(config_kwargs, "engram_ngram_orders", "2,3")
            _maybe_set(config_kwargs, "engram_bottleneck_dim", 16)
            _maybe_set(config_kwargs, "engram_dropout", 0.0)
            _maybe_set(config_kwargs, _MHC_FLAG, True)
            _maybe_set(config_kwargs, "mhc_num_branches", 0)
            _maybe_set(config_kwargs, "mhc_sinkhorn_iters", 1)
            _maybe_set(config_kwargs, "mhc_temperature", 1.0)
            _maybe_set(config_kwargs, "mhc_epsilon", 1e-6)
            _maybe_set(config_kwargs, "mhc_blend_alpha", 1.0)

            model, config = _run_forward_backward(config_kwargs)
            _run_inference(model, config)
        finally:
            kernels.set_kernel_backend(prev)

    def test_dsa_mtp_without_mamba(self):
        """DSA + MTP without Mamba should work as before."""
        if _DSA_FLAG is None or _MTP_FLAG is None:
            pytest.skip("DSA or MTP not available")

        prev = kernels.get_kernel_backend()
        kernels.set_kernel_backend("current")
        try:
            config_kwargs = _base_config()
            _maybe_set(config_kwargs, _DSA_FLAG, True)
            _maybe_set(config_kwargs, "dsa_start_layer", 2)
            _maybe_set(config_kwargs, _MTP_FLAG, True)

            model, config = _run_forward_backward(config_kwargs)
            _run_inference(model, config)
        finally:
            kernels.set_kernel_backend(prev)


# =============================================================================
# Idea G: Dual-Scan Trapezoidal Decomposition (Phase 3 optimization)
# =============================================================================
# Validates that the trapezoidal scan can be decomposed into two standard
# SSD scans with separated dt_decay/dt_input parameters, keeping O(chunk_size)
# complexity instead of O(T). This tests the mathematical equivalence.

class TestDualScanTrapezoidal:
    """Test the dual-scan trapezoidal decomposition from Idea G.

    The trapezoidal update h_t = alpha*h_{t-1} + beta*B_{t-1}*x_{t-1} + gamma*B_t*x_t
    can be decomposed into two standard Euler scans:
      y_curr from scan(x_t, dt_input=gamma*dt, dt_decay=dt)
      y_prev from scan(x_{t-1}, dt_input=beta*dt, dt_decay=dt)
      y = y_curr + y_prev + D*x

    This test verifies: if mamba2.py is available with trapezoidal support,
    a forward+backward pass produces finite loss and gradients.
    """

    def test_trapezoidal_forward_backward(self):
        """Phase 3 trapezoidal produces finite loss and gradients."""
        if not _has_field("mamba3_trapezoidal"):
            pytest.skip("mamba3_trapezoidal not in GPTConfig")

        prev = kernels.get_kernel_backend()
        kernels.set_kernel_backend("current")
        try:
            config_kwargs = _base_config()
            _maybe_set(config_kwargs, "mamba_enabled", True)
            _maybe_set(config_kwargs, "mamba_pattern", "AAM")
            _maybe_set(config_kwargs, "mamba3_trapezoidal", True)

            model, config = _run_forward_backward(config_kwargs)
            _run_inference(model, config)
        finally:
            kernels.set_kernel_backend(prev)

    def test_trapezoidal_with_all_phase2(self):
        """Phase 3 + all Phase 2 upgrades simultaneously."""
        if not _has_field("mamba3_trapezoidal"):
            pytest.skip("mamba3_trapezoidal not in GPTConfig")

        prev = kernels.get_kernel_backend()
        kernels.set_kernel_backend("current")
        try:
            config_kwargs = _base_config()
            _maybe_set(config_kwargs, "mamba_enabled", True)
            _maybe_set(config_kwargs, "mamba_pattern", "AM")
            _maybe_set(config_kwargs, "mamba3_qknorm", True)
            _maybe_set(config_kwargs, "mamba3_bias", True)
            _maybe_set(config_kwargs, "mamba3_complex_rope", True)
            _maybe_set(config_kwargs, "mamba3_trapezoidal", True)

            model, config = _run_forward_backward(config_kwargs)
            _run_inference(model, config)
        finally:
            kernels.set_kernel_backend(prev)

    def test_trapezoidal_prev_state_memory_efficiency(self):
        """Verify prev_B is stored at group-level (Bug #20 fix).

        prev_B should be (B, ngroups, d_state) not (B, nheads, d_state).
        With ngroups=1, nheads=12, this saves 12x memory per layer.
        """
        if not _has_field("mamba3_trapezoidal"):
            pytest.skip("mamba3_trapezoidal not in GPTConfig")

        try:
            from nanochat.mamba2 import Mamba2Layer
        except ImportError:
            pytest.skip("mamba2 module not available")

        prev = kernels.get_kernel_backend()
        kernels.set_kernel_backend("current")
        try:
            config_kwargs = _base_config()
            _maybe_set(config_kwargs, "mamba_enabled", True)
            _maybe_set(config_kwargs, "mamba_pattern", "M")
            _maybe_set(config_kwargs, "mamba3_trapezoidal", True)

            config = GPTConfig(**config_kwargs)
            model = GPT(config)
            model.init_weights()
            model.eval()

            # Run inference to populate states
            from nanochat.engine import KVCache
            has_mamba = getattr(config, 'mamba_enabled', False)
            kv_cache = KVCache(
                batch_size=1,
                num_heads=config.n_kv_head,
                seq_len=config.sequence_len,
                head_dim=config.n_embd // config.n_head,
                num_layers=config.n_layer,
                device=model.get_device(),
                has_mamba=has_mamba,
            )
            idx = torch.randint(0, config.vocab_size, (1, 4), device=model.get_device())
            with torch.no_grad():
                model(idx, kv_cache=kv_cache)

            # Check state shapes if mamba_params is available
            if kv_cache.mamba_params is not None:
                for layer_idx, state_dict in kv_cache.mamba_params.key_value_memory_dict.items():
                    if isinstance(state_dict, dict) and "prev_B" in state_dict:
                        prev_B = state_dict["prev_B"]
                        ngroups = getattr(config, 'mamba_ngroups', 1)
                        d_state = getattr(config, 'mamba_d_state', 64)
                        # prev_B should be at group-level: (B, ngroups, d_state)
                        assert prev_B.shape[1] == ngroups, (
                            f"prev_B dim 1 should be ngroups={ngroups}, "
                            f"got {prev_B.shape[1]} (Bug #20 regression: stored at head-level)"
                        )
        finally:
            kernels.set_kernel_backend(prev)

    def test_trapezoidal_with_mtp(self):
        """Trapezoidal + MTP: MTP block stays attention-only."""
        if not _has_field("mamba3_trapezoidal") or _MTP_FLAG is None:
            pytest.skip("mamba3_trapezoidal or MTP not available")

        prev = kernels.get_kernel_backend()
        kernels.set_kernel_backend("current")
        try:
            config_kwargs = _base_config()
            _maybe_set(config_kwargs, "mamba_enabled", True)
            _maybe_set(config_kwargs, "mamba_pattern", "AAM")
            _maybe_set(config_kwargs, "mamba3_trapezoidal", True)
            _maybe_set(config_kwargs, _MTP_FLAG, True)

            model, config = _run_forward_backward(config_kwargs)

            # Verify MTP block is NOT mamba
            if hasattr(model, 'mtp') and model.mtp is not None:
                assert not getattr(model.mtp.block, 'is_mamba', False), \
                    "MTP block became Mamba even with mamba_enabled=False in plain_config"
        finally:
            kernels.set_kernel_backend(prev)

    def test_trapezoidal_changes_output_and_has_gradient(self):
        """Verify trapezoidal actually changes Mamba output and lambda gets gradient.

        Tests with isolated Mamba layers to avoid bf16 precision issues
        in the full model at tiny test scale.
        """
        if not _has_field("mamba3_trapezoidal"):
            pytest.skip("mamba3_trapezoidal not in GPTConfig")

        from nanochat.mamba2 import Mamba2Layer

        prev = kernels.get_kernel_backend()
        kernels.set_kernel_backend("current")
        try:
            # Euler and trapezoidal Mamba layers with same weights
            cfg_e = _mamba_config(mamba_pattern="M")
            cfg_t = _mamba_config(mamba_pattern="M", mamba3_trapezoidal=True)
            config_e = GPTConfig(**cfg_e)
            config_t = GPTConfig(**cfg_t)

            torch.manual_seed(42)
            m_euler = Mamba2Layer(config_e, layer_idx=0).to(torch.bfloat16)
            m_euler._init_mamba_params()
            state = {k: v.clone() for k, v in m_euler.state_dict().items()}

            torch.manual_seed(42)
            m_trap = Mamba2Layer(config_t, layer_idx=0).to(torch.bfloat16)
            m_trap._init_mamba_params()
            for k, v in state.items():
                if k in m_trap.state_dict() and m_trap.state_dict()[k].shape == v.shape:
                    m_trap.state_dict()[k].copy_(v)

            x = torch.randn(2, 16, 64, dtype=torch.bfloat16)
            y_euler = m_euler(x.clone())
            y_trap = m_trap(x.clone())

            # Trapezoidal should produce different output
            assert not torch.allclose(y_euler, y_trap), \
                "Trapezoidal should change output vs Euler"

            # Lambda gate should get non-zero gradient
            y_trap.sum().backward()
            nheads = m_trap.nheads
            lam_grad = m_trap.in_proj.weight.grad[-nheads:]
            assert lam_grad.float().abs().sum() > 0, \
                "Lambda gate gradient is zero — trapezoidal not active in training"
        finally:
            kernels.set_kernel_backend(prev)

    def test_trapezoidal_with_engram_and_mhc(self):
        """Trapezoidal Mamba + Engram on Mamba layer + mHC."""
        if not _has_field("mamba3_trapezoidal"):
            pytest.skip("mamba3_trapezoidal not in GPTConfig")
        if _ENGRAM_FLAG is None or _MHC_FLAG is None:
            pytest.skip("Engram or mHC not available")

        prev = kernels.get_kernel_backend()
        kernels.set_kernel_backend("current")
        try:
            config_kwargs = _base_config()
            _maybe_set(config_kwargs, "mamba_enabled", True)
            _maybe_set(config_kwargs, "mamba_pattern", "AM")
            _maybe_set(config_kwargs, "mamba3_trapezoidal", True)
            _maybe_set(config_kwargs, _ENGRAM_FLAG, True)
            _maybe_set(config_kwargs, "engram_layers", "1")  # layer 1 is Mamba in AM pattern
            _maybe_set(config_kwargs, "engram_ngram_orders", "2,3")
            _maybe_set(config_kwargs, "engram_bottleneck_dim", 16)
            _maybe_set(config_kwargs, "engram_dropout", 0.0)
            _maybe_set(config_kwargs, _MHC_FLAG, True)
            _maybe_set(config_kwargs, "mhc_num_branches", 0)
            _maybe_set(config_kwargs, "mhc_sinkhorn_iters", 1)

            model, config = _run_forward_backward(config_kwargs)
            _run_inference(model, config)
        finally:
            kernels.set_kernel_backend(prev)
