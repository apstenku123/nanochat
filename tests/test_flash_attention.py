import torch

from nanochat.flash_attention import flash_attn_func


def test_flash_attention_cpu_tensors_fallback():
    # Should always work on CPU tensors, even when CUDA/FA3 is present.
    q = torch.randn(2, 4, 3, 8)
    k = torch.randn(2, 4, 3, 8)
    v = torch.randn(2, 4, 3, 8)
    y = flash_attn_func(q, k, v, causal=True, window_size=(4, 0))
    assert y.shape == q.shape
