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

### Quick Start: v6e with 16K Flash Attention

The v6e TPU supports 16K context training using XLA Pallas flash attention. This requires a **Python 3.11 venv** with specific JAX versions.

**Critical package versions:**

| Package   | Version   | Why                                            |
| --------- | --------- | ---------------------------------------------- |
| Python    | 3.11      | jaxlib 0.7.x needs >= 3.11                     |
| jax       | **0.7.0** | Pallas flash attention kernels                 |
| jaxlib    | **0.7.0** | Must match jax version                         |
| libtpu    | 0.0.21    | PJRT API 0.75, compatible with torch_xla 2.9.0 |
| torch_xla | 2.9.0     | Latest stable release                          |

> **Do NOT use jax 0.7.1** — it generates Mosaic IR v8, but libtpu 0.0.21 only supports v7.

**Setup venv:**
```bash
python3.11 -m venv ~/venv311
source ~/venv311/bin/activate
pip install torch~=2.9.0 torch_xla[tpu]~=2.9.0 \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html
pip install jax==0.7.0 jaxlib==0.7.0
pip install libtpu==0.0.21
pip install wandb filelock pyarrow psutil regex tabulate tiktoken tokenizers rustbpe
```

**Launch 16K training:**
```bash
source ~/venv311/bin/activate && source ~/.tpu_env
cd ~/nanochat
export NANOCHAT_BASE_DIR=/home/dave/data
export WANDB_API_KEY=<key>
export XLA_NO_SPECIAL_SCALARS=1
export NANOCHAT_GCS_CHECKPOINT_BUCKET=gs://nanochat-training-data-2026/checkpoints/v6e-longctx

nohup python3 -u -m scripts.base_train \
    --depth=16 --num_iterations=50000 \
    --device_batch_size=1 --max_seq_len=16384 \
    --kernel=current --no_compile --xla_flash_attn \
    --data_dir=/home/dave/data/parquet --streaming_data \
    --run=d16_400M_v6e4_longctx_16k_flash_cppeval \
    --core_metric_every=5000 --save_every=5000 --sample_every=5000 \
    > ~/train_longctx_16k.log 2>&1 &
```

**Performance:** 470-630K tok/sec, 35-62% MFU on v6e-4 (4 chips, 32GB HBM each).

### Flash Attention Integration

The `--xla_flash_attn` flag calls `enable_xla_flash_attention()` in `nanochat/flash_attention.py`, which:
1. Imports `flash_attention` from `torch_xla.experimental.custom_kernel`
2. Uses it in `flash_attn_func()` for training (B, T, H, D) -> transpose to (B, H, T, D)
3. Does NOT support sliding window attention — use `--window_pattern=L` for long-context models

### TPU Training Gotchas

- **Always `source ~/.tpu_env`** before training (sets `PJRT_DEVICE=TPU`)
- **`XLA_NO_SPECIAL_SCALARS=1`** prevents recompilation on every LR change
- **`@torch.no_grad()` not `@torch.inference_mode()`** — inference_mode crashes on XLA when slicing RoPE buffers
- **Checkpoint save order**: Save BEFORE eval/sample to prevent data loss on eval crash
- **First XLA compilation** takes 30-60+ minutes — this is normal
- **Spot instances** can be preempted — always use `NANOCHAT_GCS_CHECKPOINT_BUCKET` for cloud checkpoints

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
