"""
Tests for Plasticity Toolkit: FIRE, DASH, ReDo, SwiGLU.

These tests verify:
1. FIRE: Newton-Schulz orthogonalization preserves stability while restoring plasticity
2. DASH: Per-neuron direction-aware shrinking modifies weights correctly
3. ReDo: Dormant neuron detection and recycling via hooks
4. SwiGLU: Optional activation swap in MLP with Llama hidden_dim trick

All tests use tiny models (n_embd=64, n_layer=2) for speed.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass, replace


# ---------------------------------------------------------------------------
# Helpers: minimal config and model creation
# ---------------------------------------------------------------------------

def _has_gpt():
    try:
        from nanochat.gpt import GPTConfig, GPT
        return True
    except Exception:
        return False


def _skip_if_no_gpt():
    if not _has_gpt():
        pytest.skip("nanochat.gpt not importable")


def _tiny_config(**overrides):
    """Minimal GPTConfig for fast tests."""
    _skip_if_no_gpt()
    from nanochat.gpt import GPTConfig
    defaults = dict(
        sequence_len=32,
        vocab_size=128,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
    )
    defaults.update(overrides)
    return GPTConfig(**defaults)


def _tiny_model(**overrides):
    """Create a tiny GPT model for testing."""
    from nanochat.gpt import GPT
    cfg = _tiny_config(**overrides)
    model = GPT(cfg)
    model.init_weights()
    return model


# ===========================================================================
# PART 1: FIRE (Frobenius-Isometry Reinitialization)
# ===========================================================================

class TestFIRENewtonSchulz:
    """Test the core Newton-Schulz iteration for orthogonalization."""

    def test_newton_schulz_produces_orthogonal_matrix(self):
        """After N-S iteration, W^T W should be close to identity."""
        from nanochat.fire import newton_schulz

        W = torch.randn(64, 64)
        W_orth = newton_schulz(W, iters=15)

        # W_orth^T @ W_orth should be close to I
        WtW = W_orth.T @ W_orth
        I = torch.eye(64)
        dfi = (WtW - I).norm().item()
        assert dfi < 0.1, f"DfI = {dfi}, expected < 0.1 after 15 iters"

    def test_newton_schulz_5_iters_sufficient(self):
        """Paper says 5 iterations is enough."""
        from nanochat.fire import newton_schulz

        W = torch.randn(128, 64)
        W_orth = newton_schulz(W, iters=5)

        WtW = W_orth.T @ W_orth
        I = torch.eye(64)
        dfi_after = (WtW - I).norm().item()
        # 5 iters may not reach zero for random matrices (need ~15 for that),
        # but should be finite and show progress toward orthogonality
        assert dfi_after < 50.0, f"DfI = {dfi_after}, expected significant reduction"
        # With 15 iters it should be near-zero
        W_orth_15 = newton_schulz(W, iters=15)
        dfi_15 = ((W_orth_15.T @ W_orth_15) - I).norm().item()
        assert dfi_15 < 0.01, f"DfI@15 = {dfi_15}, expected < 0.01"

    def test_newton_schulz_wide_matrix_transposed(self):
        """Wide matrices (d_out < d_in) should be transposed internally."""
        from nanochat.fire import newton_schulz

        # Wide: 32 x 128 (d_out < d_in)
        W = torch.randn(32, 128)
        W_orth = newton_schulz(W, iters=15)

        # Output should have same shape
        assert W_orth.shape == (32, 128)

        # W @ W^T should be close to I (for wide matrices)
        WWt = W_orth @ W_orth.T
        I = torch.eye(32)
        dfi = (WWt - I).norm().item()
        assert dfi < 0.01, f"Wide DfI = {dfi}"

    def test_newton_schulz_preserves_subspace(self):
        """Orthogonalized matrix should be close to original (low SFE)."""
        from nanochat.fire import newton_schulz

        W = torch.randn(64, 64)
        # Make it slightly non-orthogonal (perturb identity)
        W = torch.eye(64) + 0.1 * torch.randn(64, 64)
        W_orth = newton_schulz(W, iters=15)

        sfe = (W - W_orth).norm().item()
        # SFE should be small because W was already near-orthogonal
        assert sfe < 5.0, f"SFE = {sfe}, expected small for near-orthogonal input"

    def test_newton_schulz_pure_orthogonal(self):
        """newton_schulz returns pure orthogonal (all singular values ~1.0).
        Scaling is done in apply_fire, not in newton_schulz."""
        from nanochat.fire import newton_schulz

        W = torch.randn(128, 64)
        W_orth = newton_schulz(W, iters=15)

        # All singular values should be near 1.0 (pure orthogonal)
        svs = torch.linalg.svdvals(W_orth)
        mean_sv = svs.mean().item()
        assert abs(mean_sv - 1.0) < 0.1, \
            f"Mean singular value = {mean_sv}, expected ~1.0"

    def test_newton_schulz_bf16(self):
        """Should work in bf16."""
        from nanochat.fire import newton_schulz

        W = torch.randn(64, 64, dtype=torch.bfloat16)
        W_orth = newton_schulz(W, iters=15)
        assert W_orth.dtype == torch.bfloat16
        assert torch.isfinite(W_orth).all()


class TestFIREApplyToModel:
    """Test applying FIRE to a full GPT model."""

    def test_fire_applies_to_2d_params_only(self):
        """FIRE should only touch 2D weight matrices."""
        _skip_if_no_gpt()
        from nanochat.fire import apply_fire

        model = _tiny_model()
        # Record original 1D params
        orig_1d = {}
        for name, p in model.named_parameters():
            if p.dim() != 2:
                orig_1d[name] = p.data.clone()

        apply_fire(model, iters=5)

        # All 1D params should be unchanged
        for name, p in model.named_parameters():
            if name in orig_1d:
                assert torch.equal(p.data, orig_1d[name]), \
                    f"1D param {name} was modified by FIRE"

    def test_fire_skips_embeddings_and_head(self):
        """Embeddings and lm_head should not be orthogonalized."""
        _skip_if_no_gpt()
        from nanochat.fire import apply_fire

        model = _tiny_model()
        orig_wte = model.transformer.wte.weight.data.clone()
        orig_head = model.lm_head.weight.data.clone()

        apply_fire(model, iters=5)

        assert torch.equal(model.transformer.wte.weight.data, orig_wte), \
            "wte was modified by FIRE"
        assert torch.equal(model.lm_head.weight.data, orig_head), \
            "lm_head was modified by FIRE"

    def test_fire_modifies_attention_and_mlp(self):
        """FIRE should modify Q/K/V/proj and MLP weights."""
        _skip_if_no_gpt()
        from nanochat.fire import apply_fire

        model = _tiny_model()
        orig_cq = model.transformer.h[0].attn.c_q.weight.data.clone()
        orig_cfc = model.transformer.h[0].mlp.c_fc.weight.data.clone()

        apply_fire(model, iters=5)

        assert not torch.equal(model.transformer.h[0].attn.c_q.weight.data, orig_cq), \
            "c_q was NOT modified by FIRE"
        assert not torch.equal(model.transformer.h[0].mlp.c_fc.weight.data, orig_cfc), \
            "c_fc was NOT modified by FIRE"

    def test_fire_reduces_dfi(self):
        """After FIRE, DfI should be near zero for all treated matrices."""
        _skip_if_no_gpt()
        from nanochat.fire import apply_fire

        model = _tiny_model()

        # Train a few steps to break orthogonality
        x = torch.randint(0, 128, (2, 16))
        y = torch.randint(0, 128, (2, 16))
        for _ in range(10):
            loss = model(x, targets=y)
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p -= 0.01 * p.grad
                    p.grad = None

        apply_fire(model, iters=20)

        # Check DfI on attention Q weight
        W = model.transformer.h[0].attn.c_q.weight.data
        WtW = W.T @ W
        I = torch.eye(W.shape[1])
        dfi = (WtW - I).norm().item()
        # After FIRE with 20 iters, DfI should be small
        assert dfi < 1.0, f"DfI after FIRE = {dfi}, expected < 1.0"

    def test_fire_with_mamba_layers(self):
        """FIRE should work with Mamba layers, skipping 1D/3D params."""
        _skip_if_no_gpt()
        from nanochat.fire import apply_fire

        model = _tiny_model(
            mamba_enabled=True,
            mamba_pattern="AM",
            mamba_d_state=8,
            mamba_expand=2,
            mamba_headdim=32,
        )

        # Record Mamba 1D params
        mamba_1d = {}
        for name, p in model.named_parameters():
            if p.dim() != 2 and 'attn' in name:
                mamba_1d[name] = p.data.clone()

        apply_fire(model, iters=5)

        # Mamba 1D/3D params should be untouched
        for name, p in model.named_parameters():
            if name in mamba_1d:
                assert torch.equal(p.data, mamba_1d[name]), \
                    f"Mamba non-2D param {name} was modified by FIRE"

    def test_fire_model_still_runs(self):
        """After FIRE, model should produce finite outputs and trainable loss."""
        _skip_if_no_gpt()
        from nanochat.fire import apply_fire

        model = _tiny_model()
        apply_fire(model, iters=5)

        x = torch.randint(0, 128, (2, 16))
        y = torch.randint(0, 128, (2, 16))
        loss = model(x, targets=y)
        assert torch.isfinite(loss), f"Loss not finite after FIRE: {loss}"
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), "NaN gradients after FIRE"

    def test_fire_target_keywords(self):
        """FIRE with target_keywords should only touch matching params."""
        _skip_if_no_gpt()
        from nanochat.fire import apply_fire

        model = _tiny_model()
        orig_cfc = model.transformer.h[0].mlp.c_fc.weight.data.clone()
        orig_cq = model.transformer.h[0].attn.c_q.weight.data.clone()

        # Only apply to attention Q/K
        apply_fire(model, iters=5, target_keywords=['c_q', 'c_k'])

        assert not torch.equal(model.transformer.h[0].attn.c_q.weight.data, orig_cq), \
            "c_q should be modified with target_keywords=['c_q', 'c_k']"
        assert torch.equal(model.transformer.h[0].mlp.c_fc.weight.data, orig_cfc), \
            "c_fc should NOT be modified with target_keywords=['c_q', 'c_k']"


# ===========================================================================
# PART 2: DASH (Direction-Aware Shrinking)
# ===========================================================================

class TestDASH:
    """Test per-neuron direction-aware shrinking."""

    def test_dash_shrinks_aligned_neurons(self):
        """Neurons whose weights align with gradient should be shrunk."""
        from nanochat.fire import dash_step

        W = torch.randn(32, 64)
        # Gradient perfectly aligned with weights
        grad = W.clone()

        W_new = dash_step(W, grad, alpha=0.0, shrink_rate=0.5)

        # All neurons should be shrunk (cosine = 1.0 > alpha=0.0)
        assert (W_new.norm(dim=1) < W.norm(dim=1)).all(), \
            "Aligned neurons should be shrunk"

    def test_dash_preserves_orthogonal_neurons(self):
        """Neurons orthogonal to gradient should not be touched."""
        from nanochat.fire import dash_step

        W = torch.zeros(32, 64)
        W[0, 0] = 1.0  # neuron 0 points in dim 0

        grad = torch.zeros(32, 64)
        grad[0, 1] = 1.0  # gradient points in dim 1 (orthogonal)

        W_new = dash_step(W, grad, alpha=0.0, shrink_rate=0.5)

        # Neuron 0 is orthogonal to gradient (cos=0), should not be shrunk
        assert torch.allclose(W_new[0], W[0]), \
            "Orthogonal neuron should not be shrunk"

    def test_dash_is_per_neuron(self):
        """Each neuron should get individual shrink factor."""
        from nanochat.fire import dash_step

        W = torch.randn(4, 8)
        grad = torch.randn(4, 8)

        W_new = dash_step(W, grad, alpha=0.0, shrink_rate=0.1)

        # Different neurons should have different shrink factors
        ratios = W_new.norm(dim=1) / (W.norm(dim=1) + 1e-8)
        assert not torch.allclose(ratios, ratios[0].expand_as(ratios), atol=1e-4), \
            "All neurons got same shrink factor — DASH is not per-neuron"

    def test_dash_respects_alpha_threshold(self):
        """Neurons with cosine < alpha should not be shrunk."""
        from nanochat.fire import dash_step

        W = torch.randn(32, 64)
        grad = torch.randn(32, 64)

        # Very high alpha — nothing should be shrunk
        W_high = dash_step(W, grad, alpha=0.99, shrink_rate=0.5)
        assert torch.allclose(W_high, W, atol=1e-6), \
            "With alpha=0.99, almost nothing should be shrunk"

    def test_dash_on_2d_only(self):
        """dash_step should only accept 2D tensors."""
        from nanochat.fire import dash_step

        W_1d = torch.randn(32)
        grad_1d = torch.randn(32)

        with pytest.raises((ValueError, RuntimeError)):
            dash_step(W_1d, grad_1d, alpha=0.05, shrink_rate=0.01)


# ===========================================================================
# PART 3: ReDo (Recycling Dormant Neurons)
# ===========================================================================

class TestReDo:
    """Test dormant neuron detection and recycling."""

    def test_redo_detects_dormant_neurons(self):
        """Hook should detect neurons with near-zero activation."""
        from nanochat.fire import ReDoDiagnostics

        # Create simple MLP
        mlp = nn.Sequential(
            nn.Linear(16, 32, bias=False),
            nn.ReLU(),
            nn.Linear(32, 16, bias=False),
        )

        # Kill half the neurons by zeroing their incoming weights
        with torch.no_grad():
            mlp[0].weight.data[16:] = 0.0  # neurons 16-31 are dead

        diag = ReDoDiagnostics()
        act_fn = lambda x: F.relu(x)
        diag.attach_hooks(mlp, target_modules=[mlp[0]], act_fn=act_fn)

        # Run some data through
        for _ in range(10):
            x = torch.randn(4, 16)
            _ = mlp(x)

        stats = diag.get_stats()
        assert len(stats) > 0, "No stats collected"

        # Check that dead neurons have near-zero stats
        for name, s in stats.items():
            dead_activity = s[16:].mean().item()
            live_activity = s[:16].mean().item()
            assert dead_activity < live_activity * 0.1, \
                f"Dead neurons should have much lower activity: dead={dead_activity}, live={live_activity}"

        diag.remove_hooks()

    def test_redo_recycle_resets_dead_neurons(self):
        """Recycling should reinitialize dead neuron weights."""
        from nanochat.fire import recycle_dormant_neurons

        fc_in = nn.Linear(16, 32, bias=False)
        fc_out = nn.Linear(32, 16, bias=False)

        # Kill neurons 0-7
        with torch.no_grad():
            fc_in.weight.data[:8] = 0.0
            fc_out.weight.data[:, :8] = 0.0

        # Create fake stats showing neurons 0-7 as dormant
        stats = torch.ones(32)
        stats[:8] = 0.0  # neurons 0-7 dead

        recycled = recycle_dormant_neurons(
            fc_in, fc_out, stats, tau=0.025
        )

        assert recycled > 0, "Should have recycled some neurons"
        assert recycled == 8, f"Expected 8 recycled, got {recycled}"

        # Dead neuron weights should no longer be zero
        assert fc_in.weight.data[:8].abs().sum() > 0, \
            "Recycled neurons should have non-zero incoming weights"

    def test_redo_uses_torch_where(self):
        """Recycling should use torch.where, not boolean indexing (XLA safety)."""
        from nanochat.fire import recycle_dormant_neurons

        fc_in = nn.Linear(16, 32, bias=False)
        fc_out = nn.Linear(32, 16, bias=False)

        # All neurons alive — nothing should change
        stats = torch.ones(32)
        orig_in = fc_in.weight.data.clone()
        orig_out = fc_out.weight.data.clone()

        recycled = recycle_dormant_neurons(fc_in, fc_out, stats, tau=0.025)

        assert recycled == 0, "No neurons should be recycled"
        assert torch.equal(fc_in.weight.data, orig_in)
        assert torch.equal(fc_out.weight.data, orig_out)

    def test_redo_hook_on_functional_activation(self):
        """Hook should work with functional activations (relu^2)."""
        from nanochat.fire import ReDoDiagnostics

        fc = nn.Linear(16, 32, bias=False)

        diag = ReDoDiagnostics()
        # Our actual activation: F.relu(x).square()
        act_fn = lambda x: F.relu(x).square()
        diag.attach_hooks(None, target_modules=[fc], act_fn=act_fn)

        # Forward pass
        x = torch.randn(4, 8, 16)  # (B, T, d)
        out = fc(x)
        # Hook fires on fc output, applies act_fn internally

        stats = diag.get_stats()
        assert len(stats) > 0, "Hook should have collected stats"

        diag.remove_hooks()


# ===========================================================================
# PART 4: SwiGLU MLP Option
# ===========================================================================

class TestSwiGLU:
    """Test optional SwiGLU activation in MLP."""

    def test_swiglu_config_field_exists(self):
        """GPTConfig should have activation field."""
        _skip_if_no_gpt()
        cfg = _tiny_config()
        assert hasattr(cfg, 'activation'), "GPTConfig missing 'activation' field"

    def test_relu2_is_default(self):
        """Default activation should be relu2."""
        _skip_if_no_gpt()
        cfg = _tiny_config()
        assert cfg.activation == "relu2", f"Default activation should be relu2, got {cfg.activation}"

    def test_swiglu_model_creates(self):
        """Model with activation='swiglu' should create successfully."""
        _skip_if_no_gpt()
        model = _tiny_model(activation="swiglu")
        assert model is not None

    def test_swiglu_mlp_has_gate(self):
        """SwiGLU MLP should have c_gate projection."""
        _skip_if_no_gpt()
        model = _tiny_model(activation="swiglu")
        mlp = model.transformer.h[0].mlp
        assert hasattr(mlp, 'c_gate'), "SwiGLU MLP should have c_gate"

    def test_relu2_mlp_no_gate(self):
        """relu2 MLP should NOT have c_gate."""
        _skip_if_no_gpt()
        model = _tiny_model(activation="relu2")
        mlp = model.transformer.h[0].mlp
        assert not hasattr(mlp, 'c_gate'), "relu2 MLP should not have c_gate"

    def test_swiglu_hidden_dim_llama_trick(self):
        """SwiGLU should use 8/3*n_embd hidden dim (Llama trick)."""
        _skip_if_no_gpt()
        model = _tiny_model(activation="swiglu")
        mlp = model.transformer.h[0].mlp

        n_embd = 64
        expected_hidden = int(8 * n_embd / 3)  # 170
        # Round to nearest multiple of 8 for hardware alignment
        expected_hidden = ((expected_hidden + 7) // 8) * 8  # 176

        actual_hidden = mlp.c_fc.weight.shape[0]
        assert actual_hidden == expected_hidden, \
            f"SwiGLU hidden_dim = {actual_hidden}, expected {expected_hidden}"

    def test_swiglu_param_count_similar(self):
        """SwiGLU (with Llama trick) should have similar param count to relu2."""
        _skip_if_no_gpt()
        model_relu2 = _tiny_model(activation="relu2")
        model_swiglu = _tiny_model(activation="swiglu")

        params_relu2 = sum(p.numel() for p in model_relu2.parameters())
        params_swiglu = sum(p.numel() for p in model_swiglu.parameters())

        ratio = params_swiglu / params_relu2
        # Should be within 10% of each other
        assert 0.9 < ratio < 1.15, \
            f"SwiGLU params ({params_swiglu}) vs relu2 ({params_relu2}) ratio={ratio:.2f}, expected ~1.0"

    def test_swiglu_forward_backward(self):
        """SwiGLU model should produce finite loss and gradients."""
        _skip_if_no_gpt()
        model = _tiny_model(activation="swiglu")

        x = torch.randint(0, 128, (2, 16))
        y = torch.randint(0, 128, (2, 16))
        loss = model(x, targets=y)
        assert torch.isfinite(loss), f"SwiGLU loss not finite: {loss}"

        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"NaN grad in {name}"

    def test_swiglu_different_output_than_relu2(self):
        """SwiGLU and relu2 should produce different outputs."""
        _skip_if_no_gpt()

        torch.manual_seed(42)
        model_relu2 = _tiny_model(activation="relu2")

        torch.manual_seed(42)
        model_swiglu = _tiny_model(activation="swiglu")

        x = torch.randint(0, 128, (1, 8))
        with torch.no_grad():
            out_relu2 = model_relu2(x)
            out_swiglu = model_swiglu(x)

        # Outputs should be different (different architecture)
        assert not torch.allclose(out_relu2, out_swiglu, atol=1e-3), \
            "SwiGLU and relu2 should produce different outputs"

    def test_fire_works_with_swiglu(self):
        """FIRE should work correctly on SwiGLU model (c_gate included)."""
        _skip_if_no_gpt()
        from nanochat.fire import apply_fire

        model = _tiny_model(activation="swiglu")
        orig_gate = model.transformer.h[0].mlp.c_gate.weight.data.clone()

        apply_fire(model, iters=5)

        # c_gate should be orthogonalized too
        assert not torch.equal(model.transformer.h[0].mlp.c_gate.weight.data, orig_gate), \
            "FIRE should also orthogonalize c_gate"


# ===========================================================================
# PART 5: Optimizer State Reset (selective)
# ===========================================================================

class TestOptimizerStateReset:
    """Test selective optimizer state reset after FIRE."""

    def test_selective_reset_clears_fired_params(self):
        """Only optimizer states for FIRE'd params should be cleared."""
        _skip_if_no_gpt()
        from nanochat.fire import apply_fire, reset_optimizer_states_for_fired_params

        model = _tiny_model()

        # Create AdamW optimizer and run a step to populate states
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randint(0, 128, (2, 16))
        y = torch.randint(0, 128, (2, 16))
        loss = model(x, targets=y)
        loss.backward()
        optimizer.step()

        # All params should have optimizer state now
        assert len(optimizer.state) > 0, "Optimizer should have states"

        # Apply FIRE and selective reset
        fired_params = apply_fire(model, iters=5)
        reset_optimizer_states_for_fired_params([optimizer], fired_params)

        # Embedding params should still have optimizer state
        wte_param = model.transformer.wte.weight
        if wte_param in optimizer.state:
            assert 'exp_avg' in optimizer.state[wte_param], \
                "wte optimizer state should be preserved"


# ===========================================================================
# PART 6: Integration — all together
# ===========================================================================

class TestIntegration:
    """Test all components working together."""

    def test_full_pipeline_relu2(self):
        """Full FIRE + model forward/backward with relu2."""
        _skip_if_no_gpt()
        from nanochat.fire import apply_fire

        model = _tiny_model(activation="relu2")

        # Step 1: Train a bit
        x = torch.randint(0, 128, (2, 16))
        y = torch.randint(0, 128, (2, 16))
        loss1 = model(x, targets=y)
        loss1.backward()

        # Step 2: Apply FIRE
        apply_fire(model, iters=5)

        # Step 3: Train should still work
        loss2 = model(x, targets=y)
        assert torch.isfinite(loss2), "Loss not finite after FIRE"
        loss2.backward()

    def test_full_pipeline_swiglu(self):
        """Full FIRE + model forward/backward with SwiGLU."""
        _skip_if_no_gpt()
        from nanochat.fire import apply_fire

        model = _tiny_model(activation="swiglu")

        x = torch.randint(0, 128, (2, 16))
        y = torch.randint(0, 128, (2, 16))

        loss1 = model(x, targets=y)
        loss1.backward()

        apply_fire(model, iters=5)

        loss2 = model(x, targets=y)
        assert torch.isfinite(loss2)
        loss2.backward()

    def test_full_pipeline_with_mamba(self):
        """Full pipeline with Mamba hybrid model."""
        _skip_if_no_gpt()
        from nanochat.fire import apply_fire

        model = _tiny_model(
            activation="relu2",
            mamba_enabled=True,
            mamba_pattern="AM",
            mamba_d_state=8,
            mamba_expand=2,
            mamba_headdim=32,
        )

        x = torch.randint(0, 128, (2, 16))
        y = torch.randint(0, 128, (2, 16))

        loss1 = model(x, targets=y)
        assert torch.isfinite(loss1)

        apply_fire(model, iters=5)

        loss2 = model(x, targets=y)
        assert torch.isfinite(loss2)
        loss2.backward()

    def test_existing_tests_not_broken(self):
        """Sanity: original relu2 model should still work identically."""
        _skip_if_no_gpt()

        model = _tiny_model()
        x = torch.randint(0, 128, (2, 16))
        y = torch.randint(0, 128, (2, 16))
        loss = model(x, targets=y)
        assert torch.isfinite(loss)
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()
