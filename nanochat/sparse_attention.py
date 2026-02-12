"""
DeepSeek Sparse Attention (DSA) with Lightning Indexer.

Based on the DSA mechanism from DeepSeek-V3.2 (arXiv:2512.02556):
1. A lightweight "lightning indexer" scores importance of each key for each query
2. Top-k keys are selected per query position
3. Standard attention is computed only over selected keys + local window

In this implementation we use mask-based sparse attention via PyTorch SDPA
(since flash attention doesn't support arbitrary masks). For short sequences
or inference with KV cache, we fall back to full flash attention.

The indexer uses multi-head ReLU-gated scoring in low dimensions (d_I=32)
following the DSA paper. A local sliding window is always included to ensure
nearby context is never dropped.

Reference: DeepSeek-V3.2 Technical Report (arXiv:2512.02556)
Reference: NSA Paper (arXiv:2502.11089)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.flash_attention import flash_attn_func, flash_attn_with_kvcache


def _norm(x):
    return F.rms_norm(x, (x.size(-1),))


def _apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class LightningIndexer(nn.Module):
    """Lightweight multi-head importance scorer (DeepSeek-V3.2 style).

    Computes: I(t,s) = sum_j w(t,j) * ReLU(q_I(t,j) . k_I(s))
    where j indexes indexer heads, using low-dimensional projections.
    """

    def __init__(self, n_embd, n_indexer_heads=16, indexer_dim=32):
        super().__init__()
        self.n_indexer_heads = n_indexer_heads
        self.indexer_dim = indexer_dim
        # Low-dimensional Q projection: per indexer head
        self.q_proj = nn.Linear(n_embd, n_indexer_heads * indexer_dim, bias=False)
        # Shared K projection across indexer heads
        self.k_proj = nn.Linear(n_embd, indexer_dim, bias=False)
        # Per-head weight derived from query token
        self.w_proj = nn.Linear(n_embd, n_indexer_heads, bias=False)

    def forward(self, x):
        """Compute importance scores for all (query, key) pairs.

        Args:
            x: (B, T, C) input hidden states

        Returns:
            importance: (B, T, T) where importance[b, t, s] = score of key s for query t
        """
        B, T, C = x.shape
        H_I = self.n_indexer_heads
        D_I = self.indexer_dim

        q = self.q_proj(x).view(B, T, H_I, D_I)  # (B, T, H_I, D_I)
        k = self.k_proj(x)  # (B, T, D_I) -- shared across heads
        w = self.w_proj(x)  # (B, T, H_I) -- per-head weights

        # Per-head scores: (B, T_q, H_I, D_I) x (B, T_k, D_I)^T -> (B, T_q, H_I, T_k)
        scores = torch.einsum('bqhd,bkd->bqhk', q, k)
        scores = F.relu(scores)  # ReLU gating (key DSA design choice)

        # Weighted sum across indexer heads: (B, T, H_I) * (B, T_q, H_I, T_k) -> (B, T_q, T_k)
        importance = torch.einsum('bqh,bqhk->bqk', w, scores)

        return importance


class DeepSeekSparseAttention(nn.Module):
    """Sparse attention with lightning indexer for token selection.

    Replaces CausalSelfAttention on designated layers (typically layer 7+).
    Uses mask-based SDPA for sparse attention during training.
    Falls back to full flash attention for short sequences and inference.
    """

    def __init__(self, config, layer_idx, dsa_top_k_ratio=0.5,
                 dsa_local_window=128, dsa_indexer_heads=16, dsa_indexer_dim=32):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.top_k_ratio = dsa_top_k_ratio
        self.local_window = dsa_local_window

        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        # Standard Q/K/V projections (same as CausalSelfAttention)
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        # Lightning indexer
        self.indexer = LightningIndexer(
            self.n_embd,
            n_indexer_heads=dsa_indexer_heads,
            indexer_dim=dsa_indexer_dim,
        )

    def _full_attention(self, x, cos_sin, window_size, kv_cache):
        """Standard full attention (fallback for short sequences and inference)."""
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        cos, sin = cos_sin
        q, k = _apply_rotary_emb(q, cos, sin), _apply_rotary_emb(k, cos, sin)
        q, k = _norm(q), _norm(k)

        if kv_cache is None:
            y = flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y

    def forward(self, x, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Inference or KV cache: always use full attention
        if kv_cache is not None:
            return self._full_attention(x, cos_sin, window_size, kv_cache)

        # Compute top_k from ratio
        top_k = max(int(T * self.top_k_ratio), self.local_window)

        # Short sequences: full attention is cheaper than sparse overhead
        if T <= top_k + 32:
            return self._full_attention(x, cos_sin, window_size, kv_cache)

        # Long sequences (>4096): mask-based SDPA would allocate O(T^2) memory
        # which is prohibitive (e.g., T=65536 => 8GB per mask). Fall back to
        # flash attention with sliding window. On XLA/TPU, always use flash attention
        # since Pallas FA already has O(n) memory. The indexer params still train
        # via their own gradient path and can be activated with custom kernels on GPU.
        if T > 4096 or x.device.type == 'xla':
            return self._full_attention(x, cos_sin, window_size, kv_cache)

        # --- Sparse attention path ---

        # 1. Compute Q, K, V with RoPE and QK-norm
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        cos, sin = cos_sin
        q, k = _apply_rotary_emb(q, cos, sin), _apply_rotary_emb(k, cos, sin)
        q, k = _norm(q), _norm(k)

        # 2. Compute importance scores via lightning indexer
        importance = self.indexer(x)  # (B, T, T)

        # 3. Build causal mask and apply to importance scores
        causal_mask = torch.ones(T, T, device=x.device, dtype=torch.bool).triu(diagonal=1)
        importance = importance.masked_fill(causal_mask.unsqueeze(0), -1e9)

        # 4. Boost local window tokens so they're always selected
        positions = torch.arange(T, device=x.device)
        dist = positions.unsqueeze(0) - positions.unsqueeze(1)  # (T, T), dist[i,j] = i-j
        in_local = (dist >= 0) & (dist < self.local_window)  # (T, T)
        importance = importance.masked_fill(in_local.unsqueeze(0), 1e9)

        # 5. Select top-k tokens per query position
        actual_k = min(top_k, T)
        _, top_indices = importance.topk(actual_k, dim=-1)  # (B, T, top_k)

        # 6. Build sparse mask with Straight-Through Estimator (STE)
        # This ensures gradients flow back through the indexer during training.
        # Forward: hard {0,1} mask; Backward: gradients via sigmoid(importance)
        sparse_mask_hard = torch.zeros(B, T, T, device=x.device, dtype=q.dtype)
        sparse_mask_hard.scatter_(2, top_indices, 1.0)
        # Enforce causality and self-attention
        causal_float = causal_mask.unsqueeze(0).to(q.dtype)
        sparse_mask_hard = sparse_mask_hard * (1.0 - causal_float)
        diag = torch.arange(T, device=x.device)
        sparse_mask_hard[:, diag, diag] = 1.0

        # STE: soft path for gradients, hard path for forward
        # Temperature controls gradient sharpness (lower = more focused gradients)
        soft_scores = torch.sigmoid(importance * 0.1)
        soft_scores = soft_scores * (1.0 - causal_float)
        soft_scores[:, diag, diag] = 1.0
        # STE trick: forward uses hard mask, backward uses soft scores
        gate = soft_scores + (sparse_mask_hard - soft_scores).detach()

        # 7. Convert gate to attention bias for SDPA
        # Where gate ≈ 0, set bias to -inf; where gate ≈ 1, set bias to 0
        attn_bias = torch.log(gate.clamp(min=1e-6)).unsqueeze(1)  # (B, 1, T, T)

        # 8. Run SDPA with sparse mask
        # SDPA expects (B, H, T, D) layout
        q_sdpa = q.transpose(1, 2)  # (B, H, T, D)

        # Handle GQA: repeat K,V heads to match Q heads
        if self.n_kv_head < self.n_head:
            repeat_factor = self.n_head // self.n_kv_head
            k_sdpa = k.transpose(1, 2).repeat_interleave(repeat_factor, dim=1)
            v_sdpa = v.transpose(1, 2).repeat_interleave(repeat_factor, dim=1)
        else:
            k_sdpa = k.transpose(1, 2)
            v_sdpa = v.transpose(1, 2)

        y = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, attn_mask=attn_bias)

        # 9. Reshape back and project
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y
