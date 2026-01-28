# Long Context Strategy (128K → 1M Tokens)

**Status**: Draft v1
**Related bd features**: `cpp-model-arch` (Feature 3.2)
**Design doc**: `docs/design/06-long-context.md`

---

## Why 1M Context for C++?

A single large C++ project can contain:
- 10,000+ source files
- 1,000+ header files with type definitions
- Stack traces spanning 50+ frames
- Core dumps with thousands of memory regions

To be a useful agent, the model must process repository-level context, not just single files.

### Context Requirements by Task

| Task                     | Typical Context    | Tokens (estimated) |
| ------------------------ | ------------------ | ------------------ |
| Fix single function      | 1 file + headers   | 2K-8K              |
| Debug with stack trace   | 5-10 files + trace | 20K-50K            |
| Refactor class hierarchy | 20+ files          | 50K-200K           |
| Codebase-wide analysis   | 100+ files         | 200K-1M            |

---

## Approach 1: YaRN RoPE Scaling (Primary)

### How It Works

YaRN (Yet another RoPE extension) modifies RoPE frequency scaling to support longer sequences than training length.

**Key insight**: Different frequency components in RoPE have different effective ranges:
- High frequencies: encode local position (nearby tokens)
- Low frequencies: encode global position (distant tokens)

YaRN scales these differently:
- High frequencies: extrapolate (no change)
- Low frequencies: interpolate (compress to fit)
- Middle: smooth blend

### Implementation

```python
def yarn_find_correction_range(beta_fast, beta_slow, dim, base, max_seq_len):
    """Find the range of frequency indices to interpolate."""
    low = math.floor(dim * math.log(max_seq_len / (beta_fast * 2 * math.pi)) / (2 * math.log(base)))
    high = math.ceil(dim * math.log(max_seq_len / (beta_slow * 2 * math.pi)) / (2 * math.log(base)))
    return max(low, 0), min(high, dim - 1)

def yarn_get_mscale(scale):
    """Attention scaling factor for YaRN."""
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0
```

### Training Schedule

| Phase        | Context Length | Duration | Method                   |
| ------------ | -------------- | -------- | ------------------------ |
| Pretrain     | 8,192          | 140h     | Standard RoPE (theta=1M) |
| LCFT Stage 1 | 32,768         | 8h       | YaRN scale=4             |
| LCFT Stage 2 | 131,072        | 8h       | YaRN scale=16            |
| LCFT Stage 3 | 524,288        | 4h       | YaRN scale=64            |

Each stage uses progressively longer training sequences built from multi-file contexts.

---

## Approach 2: LongCodeZip (Inference Optimization)

### Problem

Even with 128K context, raw code + debug traces may exceed the window. Most of the content is low-information (boilerplate, standard patterns).

### Solution

LongCodeZip compresses code based on **conditional perplexity**: tokens that the model can easily predict (low perplexity) are removed.

**Result**: 5.6x effective context expansion without quality loss.

### Algorithm

```python
def longcodezip(model, code, target_ratio=0.2):
    """Compress code by removing predictable tokens."""
    tokens = tokenizer.encode(code)

    # Compute per-token perplexity
    with torch.no_grad():
        logits = model(tokens)
        probs = softmax(logits, dim=-1)
        token_perplexity = -log(probs[range(len(tokens)), tokens])

    # Keep only high-perplexity tokens (surprising/important)
    threshold = np.percentile(token_perplexity, target_ratio * 100)
    important_indices = token_perplexity > threshold

    # Reconstruct compressed code
    compressed = reconstruct_with_markers(tokens, important_indices)
    return compressed
```

### What Gets Removed

- Standard boilerplate (`#include <iostream>`, `using namespace std;`)
- Closing braces that match obvious opening braces
- Type declarations that are contextually obvious
- Repeated patterns (getter/setter boilerplate)

### What Gets Kept

- Function signatures (entry points)
- Conditional logic (if/else, switch)
- Pointer operations and memory management
- Error handling
- Comments explaining intent

---

## Approach 3: MLA (Multi-Head Latent Attention) — Future

### Problem

KV-cache for 1M tokens is enormous:
```
KV-cache size = 2 × num_layers × num_kv_heads × seq_len × head_dim × sizeof(bf16)
             = 2 × 32 × 8 × 1,000,000 × 80 × 2 bytes
             = 81.9 GB
```

This exceeds single GPU memory just for the KV-cache.

### Solution: MLA (from DeepSeek-V3)

Compress KV pairs into low-rank latent vectors:
```
Standard:  K, V ∈ R^(seq_len × kv_dim)
MLA:       C_kv ∈ R^(seq_len × latent_dim)   where latent_dim << kv_dim
           K = C_kv × W_k
           V = C_kv × W_v
```

With `latent_dim = kv_dim / 4`:
- KV-cache reduced from 82GB to ~20GB for 1M tokens
- Fits on single B200 (192GB)

### Implementation Priority

MLA is a significant architectural change. Implement only after Dense model is proven:
1. Train base model with standard GQA
2. Validate quality and compilation metrics
3. Add MLA for production deployment with 1M context

---

## Approach 4: Hybrid Architecture (Transformer + Mamba)

### Concept

Replace some Transformer layers with Mamba (State Space Model) layers:
- **Mamba layers**: O(N) complexity, process long sequences efficiently
- **Attention layers**: O(N²) but capture precise long-range dependencies

### Pattern: Alternating Layers

```
Layer 0:  Mamba    (efficient bulk processing)
Layer 1:  Mamba
Layer 2:  Mamba
Layer 3:  Attention (precise matching)
Layer 4:  Mamba
Layer 5:  Mamba
Layer 6:  Mamba
Layer 7:  Attention
...
```

Ratio: 3:1 Mamba:Attention (like Jamba model)

### For C++ Code Specifically

- **Mamba is good for**: Sequential code reading, log processing, boilerplate skipping
- **Attention is good for**: Matching braces, finding variable definitions, type checking
- **Combination**: Best of both worlds for code analysis

### Implementation Priority

Lower priority than YaRN. Consider only if context requirements exceed 512K and MLA is insufficient.

---

## Context Parallelism (Training)

### Problem

Training with 128K+ context doesn't fit in single GPU memory even with gradient checkpointing.

### Solution: Ring Attention / Context Parallelism

Split the sequence across GPUs:
```
GPU 0: tokens [0, 32K)
GPU 1: tokens [32K, 64K)
GPU 2: tokens [64K, 96K)
GPU 3: tokens [96K, 128K)
```

Each GPU computes attention for its local chunk and communicates KV blocks to neighbors in a ring pattern.

### Framework Support

- **TorchTitan**: Native Context Parallelism support
- **Megatron-LM**: Sequence Parallelism
- **DeepSpeed Ulysses**: Sequence-level partitioning

---

## Data for Long-Context Training

### Types of Long-Context Training Examples

| Type                    | Length   | Source                                |
| ----------------------- | -------- | ------------------------------------- |
| Full source files       | 2K-20K   | Single large `.cpp` files             |
| Header + implementation | 5K-40K   | `.h` + `.cpp` pairs                   |
| Multi-file context      | 20K-200K | Class with all dependencies           |
| Repository context      | 100K-1M  | Entire module with build context      |
| Debug session           | 50K-500K | Code + full stack trace + memory dump |

### Needle-in-Haystack for Code

Synthetic evaluation: Hide a specific function definition deep in a long context. Ask the model to use that definition to fix a bug.

```
Tokens 0-50K:     Irrelevant source files
Tokens 50K-50.1K: Target function definition (the "needle")
Tokens 50K-100K:  More irrelevant source files
Tokens 100K-101K: // TASK: Fix bug that uses the target function
```

Model must find and use the definition from 50K tokens ago.

---

## Recommended Implementation Order

1. **YaRN scaling** — highest ROI, easiest to implement (modify RoPE only)
2. **LongCodeZip** — inference-time optimization, no model changes needed
3. **MLA** — architectural change, implement for production 1M context
4. **Hybrid (Mamba)** — research direction, implement if pure Transformer is insufficient
5. **Context Parallelism** — infrastructure, needed for training with >32K context
