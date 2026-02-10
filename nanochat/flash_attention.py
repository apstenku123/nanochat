"""
Unified Flash Attention interface with automatic FA3/SDPA switching.

Exports `flash_attn` module that matches the FA3 API exactly, but falls back
to PyTorch SDPA on unsupported GPUs, MPS, and CPU.

Our local FA3 build supports SM121 (GB10/DGX Spark). For other GPUs, use SDPA.

Usage (drop-in replacement for FA3):
    from nanochat.flash_attention import flash_attn_func, flash_attn_with_kvcache

    # Training (no KV cache)
    y = flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # Inference (with KV cache)
    y = flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""
import torch
import torch.nn.functional as F


# =============================================================================
# Detection: Try to load local FA3 build
# =============================================================================
def _load_flash_attention():
    """Try to load Flash Attention (local build or SDPA fallback)."""
    if not torch.cuda.is_available():
        return None, "no_cuda"

    try:
        major, minor = torch.cuda.get_device_capability()
        sm = major * 10 + minor

        # Our local FA3 build supports:
        # - SM90 (Hopper: H100, H200)
        # - SM121 (GB10/DGX Spark) - custom build
        # Blackwell (SM100) and Ada (SM89) need SDPA fallback
        if sm in (90, 121):
            from flash_attn import flash_attn_func as fa3_func
            from flash_attn import flash_attn_with_kvcache as fa3_kvcache
            return (fa3_func, fa3_kvcache), f"fa3_sm{sm}"
        else:
            return None, f"sdpa_sm{sm}"

    except ImportError as e:
        return None, f"sdpa_import_error:{e}"
    except Exception as e:
        return None, f"sdpa_error:{e}"


_fa3_funcs, _backend_info = _load_flash_attention()
HAS_FA3 = _fa3_funcs is not None

# =============================================================================
# XLA Flash Attention (TPU Pallas kernels)
# =============================================================================
_xla_flash_attn = None
_xla_flash_enabled = False

def _load_xla_flash_attention():
    """Try to load XLA flash attention for TPU."""
    try:
        from torch_xla.experimental.custom_kernel import flash_attention
        return flash_attention
    except ImportError:
        return None

def enable_xla_flash_attention():
    """Enable XLA flash attention for TPU. Call before training."""
    global _xla_flash_attn, _xla_flash_enabled, _backend_info
    _xla_flash_attn = _load_xla_flash_attention()
    if _xla_flash_attn is not None:
        _xla_flash_enabled = True
        _backend_info = "xla_flash_pallas"
        print(f"XLA Flash Attention enabled (Pallas TPU kernels)")
    else:
        print("WARNING: XLA flash attention requested but not available, using SDPA")

# Print which backend is being used
def get_backend_info():
    return _backend_info

# Override for testing: set to 'fa3', 'sdpa', or None (auto)
_override_impl = None


def _use_xla_flash(tensor_device=None):
    """Check if we should use XLA flash attention."""
    if not _xla_flash_enabled or _xla_flash_attn is None:
        return False
    if tensor_device is not None and not str(tensor_device).startswith("xla"):
        return False
    return True


def _use_fa3(tensor_device=None):
    """Determine whether to use FA3 based on availability, override, and input device."""
    if tensor_device is not None and tensor_device.type != "cuda":
        return False
    if _override_impl == 'fa3':
        assert HAS_FA3, f"Cannot override to FA3: not available ({_backend_info})"
        return True
    if _override_impl == 'sdpa':
        return False
    return HAS_FA3  # auto


# =============================================================================
# SDPA helpers
# =============================================================================
# Check if enable_gqa is supported (PyTorch 2.5+)
def _sdpa_supports_gqa():
    try:
        # Try calling with enable_gqa to see if it's supported
        # Use tiny tensors to minimize overhead
        q = torch.zeros(1, 1, 1, 1, device='cpu')
        F.scaled_dot_product_attention(q, q, q, enable_gqa=False)
        return True
    except TypeError:
        return False
    except Exception:
        # Any other error means we can't tell, assume not supported
        return False

_HAS_SDPA_GQA = _sdpa_supports_gqa()


def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding window support.
    q, k, v are (B, H, T, D) format.
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Build kwargs dict - enable_gqa only supported in PyTorch 2.5+
    extra_kwargs = {}
    if _HAS_SDPA_GQA:
        extra_kwargs['enable_gqa'] = enable_gqa

    # Full context, same length
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, **extra_kwargs)

    # Single token generation
    if Tq == 1:
        if window >= 0 and window < Tk:
            # window is "left" tokens we need to include (window + 1) keys total
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, **extra_kwargs)

    # Need explicit mask for sliding window/chunk inference
    device = q.device
    # For chunk inference (Tq != Tk), is_causal is not aligned to cache position => build an explicit bool mask
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    # sliding window (left)
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, **extra_kwargs)


# =============================================================================
# Public API: Same interface as FA3
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    if _use_fa3(q.device):
        fa3_func, _ = _fa3_funcs
        return fa3_func(q, k, v, causal=causal, window_size=window_size)

    # XLA Flash Attention (TPU Pallas kernels) — O(n) memory
    # Note: does not support sliding window, use window_pattern=L for long context
    if _use_xla_flash(q.device):
        # XLA flash expects (B, H, T, D), our input is (B, T, H, D)
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        y = _xla_flash_attn(q_t, k_t, v_t, causal=causal)
        return y.transpose(1, 2)  # back to (B, T, H, D)

    # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)  # back to (B, T, H, D)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    Flash Attention with KV cache for inference.

    FA3 updates k_cache/v_cache in-place. Our SDPA fallback does the same.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    if _use_fa3(q.device):
        _, fa3_kvcache = _fa3_funcs
        return fa3_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )

    # SDPA fallback: manually manage KV cache
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()  # assume uniform position across batch

    # Insert new K, V into cache
    if k is not None and v is not None:
        k_cache[:, pos:pos + T_new] = k
        v_cache[:, pos:pos + T_new] = v

    # Build full K, V from cache up to current position
    T_total = pos + T_new
    k_full = k_cache[:, :T_total]
    v_full = v_cache[:, :T_total]

    # Transpose for SDPA: (B, T, H, D) -> (B, H, T, D)
    q_t = q.transpose(1, 2)
    k_t = k_full.transpose(1, 2)
    v_t = v_full.transpose(1, 2)

    enable_gqa = q_t.size(1) != k_t.size(1)
    y = _sdpa_attention(q_t, k_t, v_t, window_size, enable_gqa)
    return y.transpose(1, 2)  # back to (B, T, H, D)
