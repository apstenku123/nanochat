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


_ENGRAM_FLAG = _resolve_flag_name("engram_enabled", "engram")
_MHC_FLAG = _resolve_flag_name("mhc_enabled", "mhc")


def _maybe_set(config_kwargs, name, value):
    if name in _CONFIG_FIELDS:
        config_kwargs[name] = value


@pytest.mark.parametrize(
    ("engram_enabled", "mhc_enabled"),
    [
        pytest.param(False, False, id="baseline"),
        pytest.param(False, True, id="mhc_only"),
        pytest.param(True, False, id="engram_only"),
        pytest.param(True, True, id="engram_mhc"),
    ],
)
def test_tiny_gpt_forward_backward_across_modes(engram_enabled, mhc_enabled):
    if engram_enabled and _ENGRAM_FLAG is None:
        pytest.skip("Engram flag is not available in GPTConfig on this branch.")
    if mhc_enabled and _MHC_FLAG is None:
        pytest.skip("mHC flag is not available in GPTConfig on this branch.")

    prev_backend = kernels.get_kernel_backend()
    kernels.set_kernel_backend("current")

    try:
        torch.manual_seed(7)
        config_kwargs = dict(
            sequence_len=16,
            vocab_size=128,
            n_layer=2,
            n_head=4,
            n_kv_head=4,
            n_embd=64,
            window_pattern="L",
        )

        if _ENGRAM_FLAG is not None:
            config_kwargs[_ENGRAM_FLAG] = engram_enabled
        if _MHC_FLAG is not None:
            config_kwargs[_MHC_FLAG] = mhc_enabled

        if engram_enabled:
            _maybe_set(config_kwargs, "engram_layers", "0")
            _maybe_set(config_kwargs, "engram_ngram_orders", "2,3")
            _maybe_set(config_kwargs, "engram_bottleneck_dim", 16)
            _maybe_set(config_kwargs, "engram_dropout", 0.0)

        if mhc_enabled:
            _maybe_set(config_kwargs, "mhc_num_branches", 0)
            _maybe_set(config_kwargs, "mhc_sinkhorn_iters", 1)
            _maybe_set(config_kwargs, "mhc_temperature", 1.0)
            _maybe_set(config_kwargs, "mhc_epsilon", 1e-6)
            _maybe_set(config_kwargs, "mhc_blend_alpha", 1.0)

        config = GPTConfig(**config_kwargs)
        model = GPT(config)
        model.init_weights()
        model.train()

        batch_size = 2
        seq_len = 8
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len), dtype=torch.long)
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len), dtype=torch.long)

        loss = model(idx, targets)
        assert torch.isfinite(loss), "Loss contains NaN/Inf."
        loss.backward()

        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads, "No gradients were produced."
        assert all(torch.isfinite(g).all().item() for g in grads), "Gradient contains NaN/Inf."

        with torch.no_grad():
            logits = model(idx)
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert torch.isfinite(logits).all(), "Logits contain NaN/Inf."
    finally:
        kernels.set_kernel_backend(prev_backend)
