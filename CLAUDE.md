# Claude Code Instructions for nanochat

## GPU Performance Testing Rules

**ALWAYS use nsys profiler when testing GPU performance** to verify the correct kernels are being used:

```bash
nsys profile --stats=true -o /tmp/profile python script.py
```

Key things to verify in nsys output:
- Check which attention backend is used (nvte_flash_attn vs cunn_SoftMax)
- Verify NVFP4 kernels are active (cutlass3x_sm120_bstensorop, nvte_quantize)
- Look for unexpected kernel overhead (quantize/transpose should be <10%)

**Never trust throughput numbers alone** - always profile to confirm expected kernels are running.

## Required Environment Variables for Benchmarks

**ALWAYS set these before running benchmarks** to enable caching and avoid repeated autotune:

```bash
# Persistent inductor cache (avoids re-autotuning triton kernels)
export TORCHINDUCTOR_CACHE_DIR=/home/dave/Downloads/source/nanochat/.cache/torchinductor
export TORCHINDUCTOR_FX_GRAPH_CACHE=1

# GB10/SM121 fixes
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas  # SM121a needs CUDA 13+ ptxas
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1  # Force autotune on GB10 (48 SMs)
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Transformer Engine settings
export NVTE_NVFP4_DISABLE_RHT=1  # Required for torch.compile compatibility
export NVTE_FUSED_ATTN=1
export NVTE_FLASH_ATTN=1
```

**Why triton_mm autotune happens**: Liger's fused_linear_cross_entropy uses torch matmul for lm_head projection, which goes through inductor's triton backend. TE fused layers use cuBLAS/cuDNN. The autotune caches to TORCHINDUCTOR_CACHE_DIR when FX_GRAPH_CACHE=1.

**Quick benchmark command** (after env vars are set):
```bash
.venv/bin/python -m scripts.base_train --depth=20 --num_iterations=10 --warmup_steps=2
```

## Flash Attention 3 (Local Build)

We have a **local FA3 build** for GB10/SM121 (NOT from HuggingFace `kernels` module which lacks aarch64 support):

```python
# Import directly - available in venv via PyTorch build
from flash_attn import flash_attn_func, flash_attn_with_kvcache

# Location: /home/dave/PyTorch_cuda_13_1_main_4fd1b9b7/third_party/flash-attention/
# Binary: flash_attn_2_cuda.cpython-313-aarch64-linux-gnu.so
```

**DO NOT use** `from kernels import get_kernel` (HF Hub) - it lacks SM121/aarch64 builds.

## Performance Comparison (D20 Model, B=8, T=1024, GB10/SM121)

### Architecture Comparison (20 training steps)

| Architecture     | Kernel Backend    | Throughput       | Peak Memory | Notes                   |
| ---------------- | ----------------- | ---------------- | ----------- | ----------------------- |
| **Original FA3** | CCE               | **21,756 tok/s** | **8.59 GB** | **FASTEST - Use this!** |
| **Original FA3** | current (PyTorch) | 16,040 tok/s     | 9.65 GB     | Good baseline           |
| **Original FA3** | Liger             | 15,645 tok/s     | 17.06 GB    | Slower + 2x memory      |
| TE + BF16        | compiled          | 12,524 tok/s     | 17.79 GB    | TE overhead             |
| TE + NVFP4       | compiled          | 13,010 tok/s     | 17.19 GB    | 4% faster than TE BF16  |

### Kernel Backend Comparison (Original FA3 architecture)

| Backend           | Throughput   | Memory   | Speedup vs Baseline |
| ----------------- | ------------ | -------- | ------------------- |
| **CCE (Apple)**   | 21,756 tok/s | 8.59 GB  | **1.36x**           |
| current (PyTorch) | 16,040 tok/s | 9.65 GB  | 1.00x               |
| Liger             | 15,645 tok/s | 17.06 GB | 0.98x               |

**Winner: Apple Cut Cross Entropy (CCE)** - 36% faster, 1 GB less memory than baseline!

### Top Kernels (CCE + FA3 profile)

| Kernel                                  | Time % | Description              |
| --------------------------------------- | ------ | ------------------------ |
| `_cce_backward_kernel`                  | 5.8%   | CCE custom backward pass |
| `flash_bwd_dq_dk_dv_loop_seqk_parallel` | 4.8%   | Flash Attention backward |
| `triton_tem_fused__fused_rms_norm_mm_t` | 4.7%   | Fused RMSNorm + MatMul   |
| `_cce_lse_forward_kernel`               | 3.2%   | CCE log-sum-exp forward  |
| `flash_fwd_kernel`                      | 2.1%   | Flash Attention forward  |

### Key Findings

1. **Original FA3 beats TE** - Simple custom layers + flash_attn_func is 1.6x faster than te.TransformerLayer
2. **CCE is the best loss kernel** - Apple's Cut Cross Entropy saves memory without materializing logits
3. **Liger uses too much memory** - 17 GB vs 9 GB for PyTorch native on this model
4. **torch.compile matters more for FA3** - 2x speedup with compile vs only ~10% for TE

### Recommended Configuration

```python
# Best performance: Original FA3 + CCE
from nanochat import kernels
kernels.set_kernel_backend('cce')  # Apple Cut Cross Entropy

# In model forward:
# return kernels.fused_linear_cross_entropy(
#     hidden_states.to(torch.bfloat16),
#     lm_head.weight.to(torch.bfloat16),
#     targets
# )
```

### References

- [Apple Cut Cross Entropy](https://github.com/apple/ml-cross-entropy) - ICLR 2025 paper
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel) - LinkedIn's Triton kernels
- Flash Attention 3 - Local build at `/home/dave/PyTorch_cuda_13_1_main_4fd1b9b7/third_party/flash-attention/`

### TE Experiment Backup

The Transformer Engine refactor experiment (te.TransformerLayer) is archived at:
`/home/dave/Downloads/source/nanochat_te_experiment_backup/`

Contains:
- `BENCHMARK_REPORT.md` - Detailed benchmark results and analysis
- `gpt.py` - TE-refactored GPT model
- `kernels.py` - With CCE backend support
- Benchmark scripts used for testing

**Conclusion**: TE adds overhead for small models. Original FA3 + CCE is 1.7x faster.

## TPU Training (Google Cloud v6e)

For full TPU setup guide, see [docs/TPU_SETUP.md](docs/TPU_SETUP.md).

### Architecture: How PyTorch Uses TPU Hardware

Our TPU stack has three layers:

```
nanochat (pure PyTorch)
    ↓
torch_xla 2.9.0 (translates PyTorch ops → XLA HLO graphs via PJRT runtime)
    ↓
libtpu 0.0.21–0.0.23.1 (user-space TPU driver, shared library libtpu.so)
    ↓
TPU hardware (v6e Trillium / v5e)
```

**We do NOT import JAX directly anywhere in our code.** JAX is installed solely
because `torch_xla.experimental.custom_kernel.flash_attention` internally uses JAX's
Pallas kernel language to implement flash attention on TPU. This is an implementation
detail of torch_xla, not something we interact with.

**We do NOT use:** torchax, jax_import_guard, Pallas directly, or any JAX APIs.

### Package Version Matrix (verified Feb 14, 2026)

All v6e TPUs run a single Python 3.11 venv with these pinned versions:

| Package    | Version         | Why this exact version                                        |
| ---------- | --------------- | ------------------------------------------------------------- |
| Python     | **3.11**        | jaxlib 0.7.x+ requires >= 3.11; 3.12 tested but no benefit yet |
| torch      | **2.9.1**       | Stable release; torch_xla 2.9.0 is compatible                |
| torch_xla  | **2.9.0**       | Latest stable XLA backend (Nov 2025); 2.10 does NOT exist yet |
| libtpu     | **0.0.23.1**    | Max compatible with torch_xla 2.9.0 (range: 0.0.21–0.0.23.1) |
| jax        | **0.9.0**       | Required by torch_xla's Pallas flash attention kernel         |
| jaxlib     | **0.9.0**       | Must match jax version exactly                                |

> **Version pinning is critical.** Do NOT upgrade any of these independently:
> - `libtpu >= 0.0.24` breaks with PJRT_ExecuteOptions size mismatch (expects 112, torch_xla provides 80)
> - `jax` 0.7.0 through 0.9.0 all work with libtpu 0.0.23.1 (tested Feb 14, 2026)
> - `torch_xla 2.10.0` does not exist yet (PyTorch 2.10 released Jan 2026, but no XLA match)
> - Nightly torch_xla wheels stopped publishing Oct 2025 and matching PyTorch nightlies are purged

### libtpu Compatibility Testing (Feb 14, 2026)

Binary search results for libtpu + torch_xla 2.9.0 + flash attention:

| libtpu     | PJRT API | Flash Attention | Status              |
| ---------- | -------- | --------------- | ------------------- |
| 0.0.21     | OK       | PASSED          | Default (pinned)    |
| 0.0.21.1   | OK       | PASSED          | Compatible          |
| 0.0.22     | OK       | PASSED          | Compatible          |
| 0.0.23     | OK       | PASSED          | Compatible          |
| **0.0.23.1** | **OK** | **PASSED**      | **Max compatible**  |
| 0.0.24     | v0.78    | FAILED          | PJRT API mismatch   |
| 0.0.27     | v0.80    | FAILED          | PJRT API mismatch   |
| 0.0.29+    | v0.81+   | FAILED          | PJRT API mismatch   |
| 0.0.35     | v0.90    | FAILED          | PJRT API mismatch   |

The PJRT_ExecuteOptions struct size changed from 80→112 bytes between libtpu 0.0.23.1 and 0.0.24.
Note: basic TPU init (`xm.xla_device()`) passes with all versions due to XLA's lazy execution,
but actual computation (`xm.mark_step()`) fails on 0.0.24+.

### JAX Version Compatibility (tested Feb 14, 2026)

With libtpu 0.0.23.1 + torch_xla 2.9.0, all JAX versions from 0.7.0 through 0.9.0 pass
flash attention tests:

| JAX    | jaxlib | Flash Attention | Status |
| ------ | ------ | --------------- | ------ |
| 0.7.0  | 0.7.0  | PASSED          | Minimum compatible |
| 0.7.1  | 0.7.1  | PASSED          | Compatible |
| 0.7.2  | 0.7.2  | PASSED          | Compatible |
| 0.8.0  | 0.8.0  | PASSED          | Compatible |
| 0.8.3  | 0.8.3  | PASSED          | Compatible |
| **0.9.0** | **0.9.0** | **PASSED** | **Recommended (latest stable)** |

A/B performance test (20-step training, v6e-4, 16K seq_len, flash attention) confirmed
**zero performance regression**: jax 0.9.0 + libtpu 0.0.23.1 matches jax 0.7.0 + libtpu 0.0.21
within noise (both ~350-430K tok/sec steady state).

### Setup: v6e TPU from Scratch

```bash
# 1. Install Python 3.11 venv support
sudo apt-get update -qq && sudo apt-get install -y python3.11-venv python3.11-dev

# 2. Create venv
python3.11 -m venv ~/venv && source ~/venv/bin/activate && pip install --upgrade pip

# 3. Install PyTorch + XLA (auto-resolves compatible libtpu)
pip install 'torch~=2.9.0' 'torch_xla[tpu]~=2.9.0' \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html

# 4. Install JAX 0.9.0 (needed by torch_xla's flash attention kernel)
pip install 'jax==0.9.0' 'jaxlib==0.9.0' \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html

# 5. Re-pin libtpu (jax install may pull a newer incompatible version)
pip install 'libtpu==0.0.23.1' \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html

# 6. Install training dependencies
pip install pyarrow tokenizers wandb tqdm datasets transformers \
  psutil tabulate scipy regex tiktoken rustbpe
```

Automated setup: `bash scripts/tpu/setup_v6e.sh` (runs all steps above).

### Launch Training

```bash
source ~/venv/bin/activate && source ~/.tpu_env
cd ~/nanochat
export NANOCHAT_BASE_DIR=/home/dave/data
export XLA_NO_SPECIAL_SCALARS=1
export NANOCHAT_GCS_CHECKPOINT_BUCKET=gs://nanochat-training-data-2026/checkpoints/<experiment>

nohup python3 -u -m scripts.base_train \
    --depth=16 --num_iterations=50000 \
    --device_batch_size=2 --max_seq_len=16384 \
    --kernel=current --no_compile --xla_flash_attn \
    --data_dir=/home/dave/data/parquet --streaming_data \
    --run=dummy \
    --core_metric_every=2500 --save_every=2500 --sample_every=2500 \
    > ~/train.log 2>&1 &
```

### Current Experiments (v6e-4, 16K context, d16/270M params)

| Experiment   | Features                  | device_bs | Loss (latest) | MFU     |
| ------------ | ------------------------- | --------- | ------------- | ------- |
| longctx      | baseline                  | 2         | ~0.96         | ~66%    |
| engram       | --engram --engram_layers  | 2         | ~1.05         | ~47%    |
| mhc          | --mhc                     | 1         | ~1.02         | ~35%    |
| mhc+engram   | --mhc --engram            | 2         | **~0.91**     | ~49%    |

### v6e-8 Full Pipeline (64K context, d24/877M, TP=4)

**Active run** (Feb 14, 2026): d=24 AAM hybrid with all features on v6e-8 using tensor parallelism.

```bash
source ~/venv311/bin/activate && source ~/.tpu_env
cd ~/nanochat
export NANOCHAT_BASE_DIR=/home/dave/data

nohup python3 -u -m scripts.base_train \
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
    --core_metric_every=5000 --save_every=5000 --sample_every=5000 \
    > ~/train_d24_aam_64k_tp4.log 2>&1 &
```

**Run config**:

| Parameter | Value |
|-----------|-------|
| Model | d=24, model_dim=1536, 12 heads, head_dim=128 |
| Parameters | 877M (AAM hybrid: Attention-Attention-Mamba pattern) |
| Sequence length | 65,536 (64K) |
| Parallelism | 2-way data × 4-way tensor (SPMD 2D mesh) |
| Batch size | 524,288 tokens per optimizer step |
| Grad accumulation | 4 steps (131K tokens per fwd/bwd) |
| Training tokens | 26.2B (30:1 token:param ratio) |
| Iterations | 50,000 |
| Throughput | ~200K tok/sec, ~2.5s per step |
| ETA | ~36 hours |

**Training data**: `cpp_compilable_64k` — 393,782 documents, 3.4 GB (8 shards + val)
- Source: 93 C++ open-source projects processed by `tools/cpp_chunker` in compilable mode
- Each document is a near-compilable C++ unit: preamble → types (topological) → functions (bottom-up)
- Max 65,536 tokens per document, FIM at 50%
- GCS: `gs://nanochat-training-data-2026/data/cpp_compilable_64k/`

**All features enabled**:
- Mamba-3 hybrid (AAM pattern): QK-norm, bias, complex RoPE, trapezoidal discretization
- Engram branches at layers 0, 3, 6
- mHC (multi-Head Collaboration) branch mixing
- DSA (DeepSeek Sparse Attention) on layers 7-23
- MTP (Multi-Token Prediction) with lambda=0.3
- FIM (Fill-in-the-Middle) at 50% rate
- Gradient checkpointing
- XLA Pallas flash attention (window_pattern=L, no sliding window)

### Tensor Parallelism on TPU (SPMD 2D Mesh)

The `--tensor_parallel=N` flag enables Megatron-style tensor parallelism via XLA SPMD.
With 8 chips and `--tensor_parallel=4`, the mesh is `(dp=2, tp=4)` — 2-way data, 4-way model.

**How it works**: Weight matrices are sharded across the `'model'` mesh axis:
- Attention Q/K/V: column-parallel `('model', None)` — each chip gets `n_heads/tp` heads
- Attention c_proj: row-parallel `(None, 'model')` — XLA inserts all-reduce
- MLP c_fc: column-parallel `('model', None)` — each chip gets `4*n_embd/tp` columns
- MLP c_proj: row-parallel `(None, 'model')` — XLA inserts all-reduce
- Mamba in_proj/out_proj: same column/row-parallel pattern
- Embeddings, lm_head, layer norms: replicated (not sharded)

**Constraint: `num_heads` must be divisible by `tp_degree`.**

#### TP Dimension Compatibility Table

| Depth | model_dim | num_heads (head_dim=128) | TP=2 | TP=4 | TP=8 |
|-------|-----------|--------------------------|------|------|------|
| 12    | 768       | 6                        | OK   | NO   | NO   |
| 16    | 1024      | 8                        | OK   | OK   | OK   |
| 20    | 1280      | 10                       | OK   | NO   | NO   |
| **24**| **1536**  | **12**                   | **OK** | **OK** | **NO** |
| 32    | 2048      | 16                       | OK   | OK   | OK   |
| 48    | 3072      | 24                       | OK   | OK   | OK   |

**Why TP=8 fails for d=24**: 12 heads / 8 chips = 1.5 — not integer. The Q/K/V weight
sharding creates tensors with incompatible dimensions for the attention head reshape.

#### Memory & Throughput Scaling

| Config (d=24, 877M) | Per-chip HBM | Seq Len | Grad Accum | Status |
|---------------------|-------------|---------|------------|--------|
| TP=1, dp=8          | 45.5 GB     | 65536   | 1          | OOM (31.25 GB limit) |
| TP=4, dp=2          | ~12 GB      | 65536   | 4          | **WORKING** (~200K tok/s) |
| TP=1, dp=8          | ~6 GB       | 2048    | 32         | OK (sanity test only) |

#### Grad Accum Calculation

```
tokens_per_fwdbwd = device_batch_size × max_seq_len = 1 × 65536 = 65,536
dp_degree = num_devices / tp_degree = 8 / 4 = 2
world_tokens = tokens_per_fwdbwd × dp_degree = 65,536 × 2 = 131,072
grad_accum = total_batch_size / world_tokens = 524,288 / 131,072 = 4
```

For grad_accum=8: set `--total_batch_size=1048576` (1M tokens per step).

### Flash Attention Integration

The `--xla_flash_attn` flag calls `enable_xla_flash_attention()` in `nanochat/flash_attention.py`, which:
1. Imports `flash_attention` from `torch_xla.experimental.custom_kernel`
2. Internally, torch_xla uses JAX Pallas to compile a TPU flash attention kernel
3. We call it as a pure PyTorch function — no JAX code in our codebase
4. Does NOT support sliding window attention — use `--window_pattern=L`

### Babysitter for Spot TPU Recovery

Spot TPUs get preempted without warning. The babysitter script auto-recovers:

```bash
# Run on build3 (external machine), not on the TPU itself
ssh build3
tmux new-session -s babysit
# Start one per spot TPU:
bash scripts/tpu/babysit_tpu.sh v6e-engram
bash scripts/tpu/babysit_tpu.sh v6e-longctx
bash scripts/tpu/babysit_tpu.sh v6e-mhc
```

Recovery cycle: detect PREEMPTED → delete TPU → recreate → deploy code → setup env → download data → resume from GCS checkpoint.

### TPU Training Gotchas

- **Always `source ~/.tpu_env`** before training (sets `PJRT_DEVICE=TPU`)
- **`XLA_NO_SPECIAL_SCALARS=1`** prevents recompilation on every LR change
- **`@torch.no_grad()` not `@torch.inference_mode()`** — inference_mode crashes on XLA when slicing RoPE buffers
- **Checkpoint save order**: Save BEFORE eval/sample to prevent data loss on eval crash
- **First XLA compilation** takes 30-60+ minutes — this is normal
- **Spot instances** can be preempted — always use `NANOCHAT_GCS_CHECKPOINT_BUCKET` for cloud checkpoints
- **gsutil fails on composite objects** without crcmod — use `gcloud storage cp` instead
- **Avoid data-dependent Python branches** in model code — forces XLA host-device sync (37x slowdown)
- **Checkpoint resume on XLA**: load to CPU first, use `assign=False` in load_state_dict
- **mhc at bs=2 OOMs** on v6e-4 (31.25G HBM limit) — use bs=1 for mhc-only experiments

## C++ Data Pipeline (Rust cpp-chunker)

For full details see [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md).

### Tool Location and Build

```bash
# Source: tools/cpp_chunker/src/{main,chunker,deps,compilable,global_index,project_graph}.rs
cd tools/cpp_chunker && cargo build --release
# Binary: tools/cpp_chunker/target/release/cpp-chunker
```

### Processing Modes

| Mode | Flags | Use Case |
|------|-------|----------|
| Project-aware | `--project-dirs <dir>` | Process raw project dirs with dependency DAG |
| **Compilable** | `--project-dirs <dir> --compilable` | **Best quality** — types before functions, bottom-up |
| Cross-file | `--cross-file --inputs <jsonl>` | Two-pass JSONL mode with global function index |

### Quick Commands (build3: 35.242.211.80)

```bash
# Compilable 16K chunks (recommended for training)
./cpp-chunker --project-dirs ~/data/cpp_raw --output out.jsonl \
    --max-tokens 16384 --cross-depth 3 --compilable --max-file-bytes 500000

# Compilable 64K chunks (for long-context training)
./cpp-chunker --project-dirs ~/data/cpp_raw --output out.jsonl \
    --max-tokens 65536 --cross-depth 3 --compilable --max-file-bytes 500000
```

### Datasets on GCS (gs://nanochat-training-data-2026/data/)

| Dataset | Docs | Size | Token Limit | Description |
|---------|------|------|-------------|-------------|
| `cpp_crossfile_16k` | 8.1M | 23 GB | 16K | Tree-sitter cross-file (JSONL input) |
| `cpp_project_crossfile_16k` | 1.0M | 8.0 GB | 16K | Project-aware cross-file |
| `cpp_clang_crossfile_16k` | 794K | 883 MB | 16K | Clang semantic indexer |
| **`cpp_compilable_16k`** | **1.0M** | **7.25 GB** | **16K** | **Compilable: types→functions (bottom-up)** |
| **`cpp_compilable_64k`** | **394K** | **3.33 GB** | **64K** | **Compilable: types→functions (bottom-up)** |

### Training with Compilable Data

```bash
# Download from GCS
gsutil -m cp -r gs://nanochat-training-data-2026/data/cpp_compilable_16k/ /path/to/data/

# Train (GPU)
python -m scripts.base_train --data_dir=/path/to/data/cpp_compilable_16k \
    --streaming_data --depth=16 --max_seq_len=1024 --device_batch_size=8

# Train (TPU v6e, long context with 64K data)
python -m scripts.base_train --data_dir=/path/to/data/cpp_compilable_64k \
    --streaming_data --depth=16 --max_seq_len=16384 --device_batch_size=1 \
    --xla_flash_attn --no_compile
```

### What "Compilable" Means

Each training document is ordered as a near-compilable C++ unit:
1. **Preamble** — `#include`, `using`, forward declarations
2. **Type definitions** — structs/classes/enums in topological order (`Base` before `Derived`)
3. **Functions** — bottom-up (leaf callees first, callers last)

Source: `tools/cpp_chunker/src/compilable.rs` (280 lines, 3 tests)

### Raw Projects (build3)

93 C++ open-source projects at `~/data/cpp_raw/` (641K files, 29GB). Includes: boost, linux, llvm-project, opencv, tensorflow, protobuf, grpc, rocksdb, clickhouse, godot, qemu, freebsd-src, etc.

Clang indexer: `tools/clang_indexer/index_project.py` (Python, libclang-based semantic extraction)

---

This project uses **beads (bd)** for issue tracking with Jira synchronization.

## Issue Tracking with bd + Jira

This repository is configured to sync issues between local beads and Jira project NANO.

### Quick Reference

```bash
# Find available work
bd ready

# View issue details
bd show <id>

# Create a new issue (syncs to Jira)
bd create "Issue title" --body "Description"

# Claim work
bd update <id> --status in_progress

# Complete work
bd close <id>

# Sync with Jira
bd jira sync              # Bidirectional sync
bd jira sync --pull       # Import from Jira only
bd jira sync --push       # Export to Jira only
bd jira sync --dry-run    # Preview without changes

# Check Jira sync status
bd jira status

# Sync with git
bd sync
```

### Workflow

1. **Start of session**: Run `bd ready` to see available tasks
2. **Claim work**: `bd update <id> --status in_progress`
3. **Do the work**: Implement the feature/fix
4. **Complete**: `bd close <id>` when done
5. **Sync**: Run `bd jira sync` to push changes to Jira

### Jira Integration Details

- **Jira URL**: https://cppcode.atlassian.net
- **Project Key**: NANO
- Issues created locally sync to Jira with `bd jira sync --push`
- Issues from Jira import with `bd jira sync --pull`

### Session Completion (Landing the Plane)

When ending a work session, complete ALL steps:

1. **File issues** for remaining work: `bd create "Follow-up task"`
2. **Run quality gates** if code changed (tests, linters, builds)
3. **Update issue status**: Close finished work, update in-progress items
4. **Sync to Jira**: `bd jira sync`
5. **Push to remote**:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
6. **Verify**: All changes committed AND pushed

### Environment Variables (set in ~/.bashrc)

- `JIRA_ORG_ID` - Atlassian Organization ID for admin API
- `ATLASSIAN_ADMIN_API_KEY` - Admin API key for org management (api.atlassian.com)
- `JIRA_API_TOKEN` - Personal API token for Jira REST API (create at https://id.atlassian.com/manage-profile/security/api-tokens)
- `JIRA_USERNAME` - Jira username (email): dave@cppcode.online
- `JIRA_URL` - Jira instance URL: https://cppcode.atlassian.net
- `BD_JIRA_SCRIPT` - Path to jira2jsonl.py script: /home/dave/Downloads/jira2jsonl.py

### Admin API Usage

Use the admin API key for organization management:
```bash
# Get org info
curl -s -H "Authorization: Bearer $ATLASSIAN_ADMIN_API_KEY" \
  "https://api.atlassian.com/admin/v1/orgs/$JIRA_ORG_ID"

# List domains
curl -s -H "Authorization: Bearer $ATLASSIAN_ADMIN_API_KEY" \
  "https://api.atlassian.com/admin/v1/orgs/$JIRA_ORG_ID/domains"
```
