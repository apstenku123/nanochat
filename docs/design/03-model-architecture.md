# Model Architecture Design

**Status**: Draft v1
**Related bd epic**: `cpp-model-arch`
**Design doc**: `docs/design/03-model-architecture.md`

---

## Architecture: Decoder-Only Transformer (Llama-style)

Based on nanochat's GPT implementation with modifications for C++ specialist use case.

---

## Dense 3B Configuration

```json
{
    "architecture": "CppReasonForCausalLM",
    "hidden_size": 2560,
    "intermediate_size": 6912,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "hidden_act": "silu",
    "max_position_embeddings": 8192,
    "rms_norm_eps": 1e-05,
    "vocab_size": 32000,
    "rope_theta": 1000000.0,
    "attention_bias": false,
    "tie_word_embeddings": false
}
```

### Parameter Count Breakdown

| Component             | Parameters                     | Notes                           |
| --------------------- | ------------------------------ | ------------------------------- |
| Token embedding       | 32000 × 2560 = 82M             | Input embeddings                |
| Attention (per layer) | 4 × 2560² × (32+8+8)/32 = ~31M | Q, K, V, O projections with GQA |
| MLP (per layer)       | 3 × 2560 × 6912 = ~53M         | Gate, Up, Down (SwiGLU)         |
| Per-layer total       | ~84M                           | Attention + MLP                 |
| All layers            | 32 × 84M = 2,688M              |                                 |
| Output head           | 32000 × 2560 = 82M             | Unembedding (not tied)          |
| **Total**             | **~2,852M ≈ 3B**               |                                 |

### Key Design Choices

**GQA (Grouped Query Attention) 4:1**
- 32 query heads, 8 KV heads
- Reduces KV-cache by 4x → critical for long context inference
- 4:1 ratio proven effective in Llama 2/3

**SwiGLU Activation**
- `FFN(x) = (xW₁ ⊙ SiLU(xWg)) W₂`
- Better quality than GELU/ReLU at same FLOPs
- `intermediate_size = 2.7 × hidden_size` (standard for SwiGLU)

**RMSNorm (Pre-Norm)**
- No learnable parameters (following nanochat design)
- Pre-norm for stable deep network training:
  ```
  x = x + Attention(RMSNorm(x))
  x = x + MLP(RMSNorm(x))
  ```

**No Bias**
- All linear layers: no bias terms
- Reduces parameters, improves training stability

**Untied Embeddings**
- Input embedding and output head are separate
- Better for specialized vocab where input/output distributions differ

---

## Positional Encoding: RoPE

### Base Configuration

```python
rope_theta = 1_000_000  # High base frequency for extrapolation
```

Standard RoPE (theta=10000) struggles beyond training context length.
High theta (1M) enables extrapolation to 100k+ without retraining.

### YaRN Scaling (for Long Context)

For extending beyond training length (8k → 128k → 1M):

```python
def yarn_rope_scaling(max_seq_len, original_max=8192, beta_fast=32, beta_slow=1):
    """YaRN: Yet another RoPE extension."""
    scale = max_seq_len / original_max
    # Frequency-dependent interpolation
    # Low frequencies: interpolate (NTK-aware)
    # High frequencies: extrapolate
    # Middle: blend
    ...
```

YaRN allows context extension with minimal fine-tuning (just the LCFT stage).

---

## Attention Variants

### Standard: Causal Self-Attention + Flash Attention 3

```python
# Forward pass
q, k, v = self.qkv_proj(x).split(...)
q = apply_rope(q, positions)
k = apply_rope(k, positions)
output = flash_attn_func(q, k, v, causal=True)
```

### Sliding Window Attention (Optional)

From nanochat's `window_pattern` design:
- Pattern `"SSSL"`: 3 sliding window layers + 1 full attention layer
- Sliding window = 1024 tokens
- Reduces memory for very long sequences
- Full attention layers maintain global context

### MLA (Multi-Head Latent Attention) — Future

DeepSeek-V3's innovation for KV-cache compression:
- Compress KV into low-rank latent vectors
- 4-8x reduction in KV-cache memory
- Enables 1M context on reasonable hardware

---

## MoE Variant: 10-12B / 2B Active

### Configuration

```json
{
    "architecture": "CppReasonMoEForCausalLM",
    "hidden_size": 2048,
    "num_hidden_layers": 28,
    "num_experts": 16,
    "num_experts_per_tok": 2,
    "num_shared_experts": 2,
    "expert_intermediate_size": 4096,
    "shared_expert_intermediate_size": 4096
}
```

### Expert Routing

**Top-K Gating with Load Balancing**:
```python
gate_logits = self.gate(x)  # (batch, seq, num_experts)
top_k_logits, top_k_indices = gate_logits.topk(k=2)
weights = softmax(top_k_logits)

# Load balancing auxiliary loss
aux_loss = num_experts * (fraction_routed * fraction_probability).sum()
```

### Shared Experts (DeepSeek-style)

- 2 experts are always active (shared)
- K=2 additional experts selected by router
- Total active: 4 experts per token (2 shared + 2 routed)
- Shared experts learn universal C++ syntax
- Routed experts specialize (templates, STL, debugging, algorithms)

### Why MoE for C++ Specialist?

Even within C++ there are distinct "domains":
- Template metaprogramming
- Systems programming (pointers, memory)
- Algorithm implementation
- STL usage patterns
- Debugging/error analysis

Experts can naturally specialize in these sub-domains.

---

## FIM Training Objective

### Implementation in Forward Pass

```python
def forward(self, input_ids, targets=None):
    # Standard causal LM forward
    hidden = self.transformer(input_ids)
    logits = self.lm_head(hidden)

    if targets is not None:
        # CCE loss (never materializes full logits)
        loss = cce_loss(logits, targets)

    return loss
```

FIM is handled at the **data level** (see `02-data-pipeline.md`), not the model level. The model simply predicts next tokens — FIM examples have their tokens rearranged into PSM order.

---

## Special Token Handling

### Token IDs (reserved at start of vocab)

```python
SPECIAL_TOKEN_IDS = {
    "<BOS>": 0,
    "<EOS>": 1,
    "<PAD>": 2,
    "<UNK>": 3,
    "<CODE_START>": 4,
    "<CODE_END>": 5,
    "<THOUGHT_START>": 6,
    "<THOUGHT_END>": 7,
    "<QUERY_TOOL>": 8,
    "<INDEX>": 9,
    "<FIM_PREFIX>": 10,
    "<FIM_MIDDLE>": 11,
    "<FIM_SUFFIX>": 12,
    "<DEBUG_CONTEXT>": 13,
    "<FILE_SEP>": 14,
}
```

### Generation Stop Tokens

During inference, stop generation on:
- `<EOS>`: Normal end of generation
- `<CODE_END>`: End of code output
- `<QUERY_TOOL>`: Agent needs to handle tool request

---

## Nanochat Modifications Required

### `nanochat/gpt.py` Changes

1. **GPTConfig**: Update defaults for 3B architecture
2. **Vocab size**: 32000 instead of 50304
3. **RoPE theta**: 1M instead of 10000
4. **Add YaRN scaling**: Optional RoPE interpolation
5. **Keep**: RMSNorm, SwiGLU (ReLU² → SiLU), Flash Attention, residual lambdas

### `nanochat/kernels.py` Changes

- No changes needed — CCE loss works with any vocab size

### `nanochat/engine.py` Changes

1. **Stop tokens**: Add `<CODE_END>`, `<QUERY_TOOL>` to stop conditions
2. **Tool detection**: Parse output for `__AGENT_QUERY__` patterns
