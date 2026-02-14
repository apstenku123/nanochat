# Tensor Parallelism on TPU (SPMD 2D Mesh)

## Overview

nanochat supports **Megatron-style tensor parallelism** on TPU via XLA SPMD sharding.
This splits model weights across multiple TPU chips, reducing per-chip HBM usage and
enabling training with larger models and longer sequences.

**Flag**: `--tensor_parallel=N` (where N divides the total chip count)

With 8 chips and `--tensor_parallel=4`, the SPMD mesh is `(dp=2, tp=4)`:
- **2-way data parallelism**: batch split across 2 groups
- **4-way tensor parallelism**: model weights sharded across 4 chips

## Active Run (Feb 14, 2026)

**Machine**: v6e-8 (8 TPU chips, 31.25 GB HBM each)
**IP**: 34.32.194.134

```bash
python3 -u -m scripts.base_train \
    --depth=24 --num_iterations=50000 \
    --tensor_parallel=4 \
    --device_batch_size=1 --max_seq_len=65536 --total_batch_size=524288 \
    --kernel=current --no_compile --xla_flash_attn \
    --window_pattern=L \
    --mamba --mamba_pattern=AAM \
    --mamba3_qknorm --mamba3_bias --mamba3_complex_rope --mamba3_trapezoidal \
    --engram --engram_layers=0,3,6 \
    --mhc --mtp --mtp_lambda=0.3 \
    --dsa --dsa_start_layer=7 \
    --gradient_checkpointing \
    --fim_rate=0.5 \
    --data_dir=/home/dave/data/parquet --streaming_data \
    --run=dummy \
    --core_metric_every=5000 --save_every=5000 --sample_every=5000
```

**Results** (early steps):
- 877M parameters (d=24, AAM hybrid, all features)
- 64K sequence length per chip
- 2-way data × 4-way tensor parallelism
- 4 gradient accumulation steps → 524K tokens per optimizer step
- ~200K tok/sec throughput
- ~2.5s per step (steady state)
- Loss: 13.5 → 5.1 in first 90 steps
- ETA: ~36 hours for 50K steps

## How Weight Sharding Works

The `_apply_tensor_parallel_sharding()` function in `scripts/base_train.py` marks
weight tensors with SPMD partition specs. XLA automatically inserts communication
(all-reduce, all-gather) as needed.

### Sharded Weights (Megatron-style)

| Layer         | Weight            | Logical Shape               | Partition Spec    | Pattern         |
| ------------- | ----------------- | --------------------------- | ----------------- | --------------- |
| Attention Q   | `c_q.weight`      | `[n_head*head_dim, n_embd]` | `('model', None)` | Column-parallel |
| Attention K   | `c_k.weight`      | `[n_head*head_dim, n_embd]` | `('model', None)` | Column-parallel |
| Attention V   | `c_v.weight`      | `[n_head*head_dim, n_embd]` | `('model', None)` | Column-parallel |
| Attention out | `c_proj.weight`   | `[n_embd, n_embd]`          | `(None, 'model')` | Row-parallel    |
| MLP up        | `c_fc.weight`     | `[4*n_embd, n_embd]`        | `('model', None)` | Column-parallel |
| MLP down      | `c_proj.weight`   | `[n_embd, 4*n_embd]`        | `(None, 'model')` | Row-parallel    |
| Mamba in      | `in_proj.weight`  | `[d_in_proj, n_embd]`       | `('model', None)` | Column-parallel |
| Mamba out     | `out_proj.weight` | `[n_embd, d_inner]`         | `(None, 'model')` | Row-parallel    |

### Replicated Weights (not sharded)

- `transformer.wte.weight` (token embeddings)
- `lm_head.weight` (output projection)
- Layer norms, RMSNorm
- Mamba per-head params: `conv1d.weight`, `A_log`, `dt_bias`, `D`, `B_bias`, `C_bias`
- Scalar params: `resid_lambdas`, `x0_lambdas`
- MTP projection

### Flash Attention SPMD

With TP, flash attention uses partition spec `('data', 'model', None, None)`:
- Batch dim sharded across data axis
- Head dim sharded across model axis
- Sequence and head_dim unsharded

## Critical Constraint: Head Divisibility

**`num_heads` must be evenly divisible by `tp_degree`.**

If not, the Q/K/V column-parallel sharding creates tensors with dimensions that
don't align with the attention head reshape. For example, d=24 has 12 heads:
12 / 8 = 1.5 → shape mismatch → `RuntimeError: mm(): cannot matrix-multiply`.

### TP Dimension Compatibility Table

| Depth  | model_dim | num_heads | head_dim | TP=2   | TP=4   | TP=8   |
| ------ | --------- | --------- | -------- | ------ | ------ | ------ |
| 8      | 512       | 4         | 128      | OK     | OK     | NO     |
| 12     | 768       | 6         | 128      | OK     | NO     | NO     |
| 16     | 1024      | 8         | 128      | OK     | OK     | OK     |
| 20     | 1280      | 10        | 128      | OK     | NO     | NO     |
| **24** | **1536**  | **12**    | **128**  | **OK** | **OK** | **NO** |
| 28     | 1792      | 14        | 128      | OK     | NO     | NO     |
| 32     | 2048      | 16        | 128      | OK     | OK     | OK     |
| 40     | 2560      | 20        | 128      | OK     | OK     | NO     |
| 48     | 3072      | 24        | 128      | OK     | OK     | OK     |

**Formula**: `num_heads = model_dim / head_dim`, must be divisible by `tp_degree`.

With `head_dim=64` (non-default), more configurations become TP=8 compatible:
| Depth | num_heads (hd=64) | TP=8 |
| ----- | ----------------- | ---- |
| 12    | 12                | NO   |
| 16    | 16                | OK   |
| 24    | 24                | OK   |
| 32    | 32                | OK   |

## Gradient Accumulation Calculations

With tensor parallelism, `dp_degree = num_devices / tp_degree`. Only the data-parallel
dimension contributes to world token count:

```
tokens_per_fwdbwd = device_batch_size × max_seq_len
dp_degree = num_devices / tp_degree
world_tokens = tokens_per_fwdbwd × dp_degree
grad_accum = total_batch_size / world_tokens
```

### Example Configurations (v6e-8, 8 chips)

| TP    | DP    | device_bs | seq_len   | world_tokens | total_batch | grad_accum |
| ----- | ----- | --------- | --------- | ------------ | ----------- | ---------- |
| 1     | 8     | 1         | 65536     | 524,288      | 524,288     | 1          |
| 1     | 8     | 2         | 16384     | 262,144      | 524,288     | 2          |
| 2     | 4     | 1         | 65536     | 262,144      | 524,288     | 2          |
| **4** | **2** | **1**     | **65536** | **131,072**  | **524,288** | **4**      |
| 4     | 2     | 1         | 65536     | 131,072      | 1,048,576   | 8          |
| 8     | 1     | 1         | 65536     | 65,536       | 524,288     | 8          |

## Memory Analysis

### Why TP=1 OOMs at 64K (d=24, 877M params)

With TP=1, each chip holds the full model:
- Model weights (bf16): ~1.7 GB
- Optimizer states (Adam+Muon): ~7 GB
- Activations at 65K seq: ~36 GB (even with gradient checkpointing)
- **Total: ~45 GB > 31.25 GB HBM per chip → OOM**

### With TP=4 (d=24, 877M params)

Each chip holds 1/4 of sharded weights:
- Sharded weights: ~0.4 GB
- Sharded optimizer: ~1.8 GB
- Sharded attention activations: ~9 GB (heads split 4-way)
- Replicated residual stream: ~2 GB (with gradient checkpointing)
- **Total: ~12-15 GB per chip → fits in 31.25 GB**

### Sequence Length Limits by TP Degree

| TP  | dp  | Max seq_len (d=24, bs=1) | Notes                |
| --- | --- | ------------------------ | -------------------- |
| 1   | 8   | ~16K                     | Original experiments |
| 2   | 4   | ~32K                     | Estimated            |
| 4   | 2   | **65K**                  | **Verified working** |

## Implementation Details

### Code Locations

| Feature              | File                          | Lines                               |
| -------------------- | ----------------------------- | ----------------------------------- |
| CLI flag             | `scripts/base_train.py`       | `--tensor_parallel` arg             |
| 2D mesh creation     | `scripts/base_train.py`       | `train()`, SPMD setup block         |
| Weight sharding      | `scripts/base_train.py`       | `_apply_tensor_parallel_sharding()` |
| Flash attention SPMD | `nanochat/flash_attention.py` | `set_spmd_mesh()`                   |
| Data sharding        | `scripts/base_train.py`       | `shard_data()`                      |

### Known Bug Fix (Feb 14, 2026)

The original Mesh creation used `device_ids.reshape(dp_degree, tp_degree)`:
```python
# BUG: len(2D_array) returns first dim, not total elements
spmd_mesh = Mesh(device_ids.reshape(dp_degree, tp_degree), ...)
```

Fix: pass flat `device_ids` and let `mesh_shape` handle the layout:
```python
# FIXED: flat array, mesh_shape defines topology
spmd_mesh = Mesh(device_ids, (dp_degree, tp_degree), ('data', 'model'))
```

### MFU Calculation Note

With TP, the reported MFU may exceed 100% because the FLOP estimator
(`model.estimate_flops()`) doesn't account for the communication overhead of
all-reduce operations. The raw tok/sec metric is more reliable for comparing
configurations.

## Future Work

- **Sequence parallelism**: Shard residual stream along sequence dimension to further
  reduce per-chip activation memory. Would enable larger models at 64K.
- **Vocabulary parallelism**: Shard embedding/lm_head across model axis.
  Minimal memory savings (~0.1 GB) but reduces all-reduce volume.
- **Pipeline parallelism**: Split layers across chips (not weights).
  More complex, requires micro-batching. Not needed for v6e-8 with TP.
