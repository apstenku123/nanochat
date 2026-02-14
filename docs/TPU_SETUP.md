# Google Cloud TPU Setup for nanochat

This document describes our Google Cloud TPU infrastructure for training nanochat C++ code generation models.

## Current TPU Instances

| Name | TPU Type | Zone | Chips | HBM/chip | Pricing | Purpose |
| --------------------- | ----------- | ----------------- | ----- | -------- | ------- | ------------------------------------------- |
| nanochat-v6e-longctx | v6e-4 | asia-northeast1-b | 4 | 32 GB | Spot | Baseline: 16K context |
| nanochat-v6e-engram | v6e-4 | asia-northeast1-b | 4 | 32 GB | Spot | Engram experiment: 16K context |
| nanochat-v6e-mhc | v6e-4 | asia-northeast1-b | 4 | 32 GB | Spot | mHC experiment: 16K, bs=1 |
| nanochat-v6e-mhc-engram | v6e-4 | europe-west4-a | 4 | 32 GB | Regular | mHC+Engram: 16K (best loss) |
| nanochat-v6e8-mtp | v6e-8 | europe-west4-a | 8 | 32 GB | Regular | Full pipeline: 64K, MTP+DSA+mHC+Engram+FIM |
| nanochat-tpu | v5litepod-8 | us-west4-a | 8 | 16 GB | Spot | Idle (previous gen) |
| nanochat-v6e-small | v6e-4 | asia-northeast1-b | 4 | 32 GB | Spot | Idle (old baseline) |

### Project Configuration

- **Project ID**: `alpine-aspect-459819-m4`
- **GCS Bucket**: `gs://nanochat-training-data-2026/`
- **W&B Project**: `misaacson-texas-state-technical-college/nanochat`

## Multi-Chip Training (SPMD)

nanochat uses **SPMD (Single Program Multiple Data)** for multi-chip training. A single process runs on the host, and XLA automatically shards tensors across all chips.

### How SPMD Works

1. `xr.use_spmd()` is called in `main()` BEFORE any XLA runtime init
2. A mesh is created: `Mesh(device_ids, (num_devices,), ('data',))`
3. `xs.mark_sharding(tensor, mesh, ('data', None))` shards the batch dimension across chips
4. Dataloader loads the full global batch (`device_batch_size * num_chips`), SPMD shards it
5. Explicit `xla_all_reduce_gradients()` is still needed (removing it causes 30+ min compilation)
6. `torch_xla.compile()` wraps the entire training step for optimal performance

### Key SPMD Optimization: `torch_xla.compile()`

The compiled training step includes forward + backward + allreduce + optimizer as ONE XLA graph with ONE `mark_step` call. This reduces mark_step calls from ~11 to 1 per training step.

**Critical**: Pre-fetch all micro-batches OUTSIDE the compiled region to avoid OOM (data ops in graph = 62GB vs 16GB HBM).

### Batch Size Math

With SPMD, gradient accumulation steps decrease proportionally:

```
# v5e with 8 chips:
grad_accum_steps = 524288 / (8 * 1024 * 8) = 8

# v6e with 4 chips, seq_len=2048:
grad_accum_steps = 524288 / (8 * 2048 * 4) = 8

# v6e longctx with 4 chips, seq_len=16384:
grad_accum_steps = 524288 / (4 * 16384 * 4) = 2
```

### Performance Results

| TPU | Config | Seq Len | Throughput | MFU |
| ------------- | -------------------- | ------- | ---------- | --- |
| v6e-4 (flash) | 4 chips + flash attn | 16384 | **~500-630K tok/s** | 35-62% |
| v6e-4 (SPMD) | 4 chips + compile | 2048 | ~320K tok/s | ~18% |
| v5e-8 (SPMD) | 8 chips + compile | 1024 | ~175K tok/s | ~20% |

## XLA Flash Attention

For long-context training (seq_len >= 4096), XLA Flash Attention via Pallas TPU kernels provides O(n) memory attention:

```bash
python -m scripts.base_train --xla_flash_attn --max_seq_len=16384 --device_batch_size=1
```

### Working Configuration (verified Feb 14, 2026)

XLA Flash Attention works with **Python 3.11** and the following package combination:

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.11 | **Required** — jaxlib 0.7.x needs Python >= 3.11 |
| torch | 2.9.1 | PyTorch stable |
| torch_xla | 2.9.0 | Latest stable XLA backend (Nov 2025); **2.10 does NOT exist** |
| libtpu | **0.0.23.1** | Max compatible with torch_xla 2.9.0 (range: 0.0.21–0.0.23.1) |
| jax | **0.9.0** | Pallas flash attention kernels (0.7.0–0.9.0 all work) |
| jaxlib | **0.9.0** | Must match jax version |

> **Critical**: `libtpu >= 0.0.24` breaks with `PJRT_ExecuteOptions size mismatch (expected 112, got 80)`.
> The PJRT API struct size changed between 0.0.23.1 and 0.0.24. torch_xla 2.9.0 only supports the 80-byte version.

> **Python 3.10 does NOT work**: jaxlib 0.7.x requires Python >= 3.11. The old jax 0.4.x versions lack the Pallas kernels needed for flash attention.

### libtpu Version Compatibility (tested Feb 14, 2026)

Exhaustive testing of every stable libtpu release with torch_xla 2.9.0 + flash attention:

| libtpu | PJRT API | Status | Notes |
|--------|----------|--------|-------|
| 0.0.21 | OK | Working | Default pin from `torch_xla[tpu]` |
| 0.0.21.1 | OK | Working | |
| 0.0.22 | OK | Working | |
| 0.0.23 | OK | Working | |
| **0.0.23.1** | **OK** | **Working** | **Newest compatible (Oct 8, 2025)** |
| 0.0.24 | v0.78 | BROKEN | PJRT_ExecuteOptions size mismatch |
| 0.0.27–0.0.35 | v0.80–v0.90 | BROKEN | Same PJRT API break |

**Note**: Basic TPU init (`xm.xla_device()`) appears to work with all versions due to XLA lazy execution, but actual computation via `xm.mark_step()` fails on 0.0.24+.

**JAX compatibility with libtpu 0.0.23.1** (tested Feb 14, 2026): All stable JAX versions from 0.7.0 through 0.9.0 pass flash attention tests. Use JAX 0.9.0 for the latest Pallas kernel improvements.

**Nightly path (dead end as of Feb 2026)**: torch_xla nightly wheels stopped publishing Oct 2025. The matching PyTorch nightlies have been purged from the index. Cannot use libtpu 0.0.24+ until torch_xla 2.10 is released.

### Performance Results (v6e-4, 16K seq_len)

| Metric | Value |
|--------|-------|
| Sequence length | 16,384 |
| device_batch_size | 1 |
| Throughput | **470-630K tok/sec** |
| MFU | **35-62%** |
| Memory per chip | Fits in 32 GB HBM |

This is **2-3x faster** than the 4K SDPA fallback (which achieved ~250K tok/sec) and enables 4x longer context.

### Python 3.11 Venv Setup (for flash attention)

```bash
# Install Python 3.11
sudo apt-get install python3.11 python3.11-venv python3.11-dev

# Create dedicated venv
python3.11 -m venv ~/venv311
source ~/venv311/bin/activate

# Install PyTorch + XLA
pip install torch~=2.9.0 torch_xla[tpu]~=2.9.0 \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html

# Install JAX 0.9.0 for Pallas flash attention (0.7.0–0.9.0 all work)
pip install jax==0.9.0 jaxlib==0.9.0 \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html

# Pin libtpu (jax may have pulled a newer version)
pip install libtpu==0.0.23.1 \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html

# Install training dependencies
pip install wandb filelock pyarrow psutil regex tabulate tiktoken tokenizers rustbpe

# Verify
python -c "import jax; print(f'JAX {jax.__version__}')"
python -c "import torch_xla; print(f'torch_xla {torch_xla.__version__}')"
```

### Testing Flash Attention

```python
import torch_xla
from torch_xla.experimental.custom_kernel import jax_import_guard
jax_import_guard()  # MUST call before any jax imports
from torch_xla.experimental.custom_kernel import flash_attention
import torch
import torch_xla.core.xla_model as xm

device = xm.xla_device()
q = torch.randn(1, 8, 128, 64, device=device, dtype=torch.bfloat16)
k = torch.randn(1, 8, 128, 64, device=device, dtype=torch.bfloat16)
v = torch.randn(1, 8, 128, 64, device=device, dtype=torch.bfloat16)
y = flash_attention(q, k, v, causal=True)
xm.mark_step()
print(f"Output shape: {y.shape}")  # (1, 8, 128, 64)
```

### Historical Note: PJRT Version Debugging

Earlier attempts with Python 3.10 + jax 0.4.x failed because those JAX versions lack the Pallas kernels. Python 3.13 + libtpu 0.0.34 had PJRT API 0.89, incompatible with torch_xla 2.9.0. The solution was Python 3.11 + jax 0.7.0+ which provides Pallas while remaining compatible with libtpu 0.0.23.1. Testing confirmed JAX 0.7.0 through 0.9.0 all work with libtpu 0.0.23.1.

### Window Patterns

- `--window_pattern=L`: All layers use local (sliding window) attention
- `--window_pattern=GL`: Alternating global/local attention layers

## Connecting to TPUs

### Direct SSH (Recommended)

gcloud SSH wrapper can be flaky (retries with exit code 255). Direct SSH is more reliable:

```bash
# Get the external IP
gcloud compute tpus tpu-vm describe <name> --zone=<zone> \
  --format="value(networkEndpoints[0].accessConfig.externalIp)"

# SSH directly
ssh -i ~/.ssh/google_compute_engine dave@<IP>
```

### First Connection After Create/Recreate

Must push SSH keys via gcloud first:
```bash
gcloud compute tpus tpu-vm ssh <name> --zone=<zone> --command="echo ok"
```

Note: After TPU stop/start or recreation, the host key changes:
```bash
ssh-keygen -f ~/.ssh/known_hosts -R <old-ip>
```

## Deploying Code to TPUs

**Never SCP the full nanochat directory** (345 GB with data). Use a code-only tarball:

```bash
# Create tarball (from nanochat project root)
tar czf /tmp/nanochat_code.tar.gz \
  --exclude='.git' --exclude='data' --exclude='__pycache__' \
  --exclude='.cache' --exclude='reports' --exclude='*.log' \
  --exclude='vertex_ai' --exclude='*.tar.gz' --exclude='.beads' \
  nanochat/ scripts/

# SCP to TPU
scp /tmp/nanochat_code.tar.gz dave@<IP>:/tmp/

# Extract on TPU
ssh dave@<IP> "cd /home/dave/nanochat && tar xzf /tmp/nanochat_code.tar.gz"
```

## Starting Training

### v5e (8 chips, seq_len=1024)

```bash
ssh dave@<v5e-IP>
source ~/.tpu_env
source /home/dave/venv/bin/activate  # if using venv
cd /home/dave/nanochat
export NANOCHAT_BASE_DIR=/home/dave/data
export NANOCHAT_GCS_CHECKPOINT_BUCKET=gs://nanochat-training-data-2026/checkpoints/v5e
export WANDB_API_KEY=<key>
export XLA_NO_SPECIAL_SCALARS=1

nohup python3 -u -m scripts.base_train \
    --depth=16 --num_iterations=50000 \
    --device_batch_size=8 --max_seq_len=1024 \
    --kernel=current --no_compile \
    --run=d16_400M_tpu_v5e_65k_spmd_8chip_cppeval \
    --core_metric_every=5000 --save_every=5000 --sample_every=5000 \
    > ~/train_v5e_cppeval.log 2>&1 &
```

### v6e (4 chips, seq_len=2048)

```bash
ssh dave@<v6e-IP>
source /home/dave/venv/bin/activate
source ~/.tpu_env
cd /home/dave/nanochat
export NANOCHAT_BASE_DIR=/home/dave/data
export NANOCHAT_GCS_CHECKPOINT_BUCKET=gs://nanochat-training-data-2026/checkpoints/v6e
export WANDB_API_KEY=<key>
export XLA_NO_SPECIAL_SCALARS=1

nohup python3 -u -m scripts.base_train \
    --depth=16 --num_iterations=50000 \
    --device_batch_size=8 --max_seq_len=2048 \
    --kernel=current --no_compile \
    --run=d16_400M_v6e4_65k_spmd_4chip_cppeval \
    --core_metric_every=5000 --save_every=5000 --sample_every=5000 \
    > ~/train_v6e_cppeval.log 2>&1 &
```

### v6e Long-Context with Flash Attention (4 chips, seq_len=16384, streaming data)

Uses XLA Pallas flash attention for O(n) memory, enabling 16K context on 32GB HBM chips.

> **Requires Python 3.11 venv** (`~/venv`) with jax 0.9.0. See "XLA Flash Attention" section above for setup.

```bash
ssh dave@<v6e-longctx-IP>
source /home/dave/venv/bin/activate  # Python 3.11 with jax 0.9.0
source ~/.tpu_env
cd /home/dave/nanochat
export NANOCHAT_BASE_DIR=/home/dave/data
export NANOCHAT_GCS_CHECKPOINT_BUCKET=gs://nanochat-training-data-2026/checkpoints/v6e-longctx
export WANDB_API_KEY=<key>
export XLA_NO_SPECIAL_SCALARS=1

nohup python3 -u -m scripts.base_train \
    --depth=16 --num_iterations=50000 \
    --device_batch_size=1 --max_seq_len=16384 \
    --kernel=current --no_compile \
    --xla_flash_attn \
    --data_dir=/home/dave/data/parquet --streaming_data \
    --run=d16_400M_v6e4_longctx_16k_flash_cppeval \
    --core_metric_every=5000 --save_every=5000 --sample_every=5000 \
    > ~/train_longctx_16k.log 2>&1 &
```

> **Note**: `device_batch_size=1` because 16K tokens per sequence uses ~4x more memory than 4K. Total batch size remains 524,288 tokens via gradient accumulation (8 steps with 4 chips).

### Key Training Flags for TPU

| Flag | Value | Reason |
| ----------------------- | --------------------- | ----------------------------------------------------------- |
| `--kernel=current` | PyTorch native | CCE requires CUDA/Triton (not available on TPU) |
| `--no_compile` | Disable torch.compile | torch.compile doesn't work on TPU/XLA |
| `--device_batch_size=8` | Per-chip batch | v5e: 16GB, v6e: 32GB (don't use 16/32 on v6e - OOM) |
| `--max_seq_len=1024` | v5e | Shorter sequences for 16GB chips |
| `--max_seq_len=2048` | v6e | Standard sequences for 32GB chips |
| `--xla_flash_attn` | Enable flash attn | Requires Python 3.11 + jax 0.9.0 + libtpu 0.0.23.1 (see above) |
| `--window_pattern=L` | Sliding window | For use with flash attention (currently blocked) |
| `--streaming_data` | Dynamic shard load | For streaming data pipeline (see DATA_PIPELINE.md) |
| `--data_dir=<path>` | Custom data dir | Override default parquet path |
| `XLA_NO_SPECIAL_SCALARS=1` | Env var | Prevents recompilation on every LR change |

## GCS Checkpoint Storage

Checkpoints auto-upload to GCS when `NANOCHAT_GCS_CHECKPOINT_BUCKET` is set:

```
gs://nanochat-training-data-2026/checkpoints/v5e/
gs://nanochat-training-data-2026/checkpoints/v6e/
gs://nanochat-training-data-2026/checkpoints/v6e-longctx/
```

**Critical for spot instances**: TPU VMs can be preempted at any time, GCS checkpoints survive.

## C++ Code Evaluation

Training includes periodic C++ code evaluation (replaces NLP CORE metric):

- 17 problems from `data/cpp_bench.jsonl` (7 easy, 6 medium, 4 hard)
- Metrics: `cpp_compile_rate`, `cpp_pass_rate` (logged to W&B)
- Takes ~3 min per eval (vs 6+ hours for NLP CORE metric)
- Triggered every `--core_metric_every` steps

## Cost Estimates

Using **spot/preemptible pricing**:

| TPU Type | On-Demand ($/hr) | Spot ($/hr) | Instance |
| ----------- | ---------------- | ----------- | -------------------- |
| v6e-4 | ~$5.10 | **~$1.70** | nanochat-v6e-small |
| v6e-4 | ~$5.10 | **~$1.70** | nanochat-v6e-longctx |
| v5litepod-8 | ~$6.12 | **~$2.04** | nanochat-tpu |

**Combined hourly cost (3 TPUs)**: ~$5.44/hr (spot pricing)

> Spot instances can be preempted with 30 seconds notice. Save checkpoints frequently!

## TPU Type Comparison

### v6e (Trillium) - 32 GB HBM/chip
- Latest generation TPU (2024)
- Higher memory bandwidth, supports seq_len=2048 (standard) or 16384 (with flash attention)
- With XLA Pallas flash attention: 470-630K tok/sec at 16K context, 35-62% MFU
- **v6e-4 is the max single-host size** (4 chips)

### v5e (v5litepod) - 16 GB HBM/chip
- Previous generation, still excellent
- Limited to seq_len=1024 (memory)
- **v5litepod-8 is the max single-host size** (8 chips)

## Creating New TPU VMs

### Enable TPU API (first time only)

```bash
gcloud services enable tpu.googleapis.com --project=alpine-aspect-459819-m4
```

### Create v5e TPU

```bash
gcloud compute tpus tpu-vm create <name> \
  --zone=us-west4-a \
  --accelerator-type=v5litepod-8 \
  --version=v2-alpha-tpuv5-lite \
  --spot \
  --project=alpine-aspect-459819-m4
```

### Create v6e TPU

```bash
# IMPORTANT: v6e requires v2-alpha-tpuv6e runtime (NOT v6e-ubuntu-2404 or tpu-ubuntu2204-base)
gcloud compute tpus tpu-vm create <name> \
  --zone=asia-northeast1-b \
  --accelerator-type=v6e-4 \
  --version=v2-alpha-tpuv6e \
  --spot \
  --project=alpine-aspect-459819-m4
```

### Available Zones

| Type | Zones |
| ---- | -------------------------------------------- |
| v6e | asia-northeast1-b, us-east5-b, us-central2-b |
| v5e | us-west4-a, us-central1-a, europe-west4-a |

## Python & Package Setup

### Why JAX Is Installed (But Not Used Directly)

**We never import JAX in our code.** JAX is installed because `torch_xla.experimental.custom_kernel.flash_attention` — the XLA flash attention implementation — is built on JAX's Pallas kernel language internally. When you call `--xla_flash_attn`, torch_xla imports JAX behind the scenes to compile the Pallas kernel. This is an implementation detail of torch_xla, not something we interact with.

The software stack:
```
nanochat (pure PyTorch code)
    → torch_xla 2.9.0 (XLA backend for PyTorch, uses PJRT runtime)
        → libtpu 0.0.23.1 (user-space TPU driver, shared library)
        → jax 0.9.0 (only used internally by torch_xla for Pallas flash attention kernel)
    → TPU hardware
```

We do NOT use: torchax, jax_import_guard, Pallas directly, or any JAX APIs.

### Version Matrix (verified Feb 2026)

All experiments use a single Python 3.11 venv (`~/venv`):

| Package    | Version     | Why this exact version                                         |
| ---------- | ----------- | -------------------------------------------------------------- |
| Python     | **3.11**    | jaxlib requires >= 3.11; 3.12 tested but no benefit yet       |
| torch      | **2.9.1**   | Stable release (Oct 2025); PyTorch 2.10 exists but untested   |
| torch_xla  | **2.9.0**   | Must match torch major version; latest stable for 2.9.x       |
| libtpu     | **0.0.23.1** | Max compatible with torch_xla 2.9.0 (range: 0.0.21–0.0.23.1) |
| jax        | **0.9.0**   | Required by torch_xla Pallas kernel (0.7.0–0.9.0 all work)   |
| jaxlib     | **0.9.0**   | Must match jax version exactly                                 |

**Version pinning is critical — do NOT upgrade independently:**
- `libtpu >= 0.0.24` has PJRT API v0.78+, incompatible with `torch_xla 2.9.0` → PJRT_ExecuteOptions size mismatch
- `jax` 0.7.0 through 0.9.0 all work with libtpu 0.0.23.1 (tested Feb 14, 2026)
- Install order matters: torch_xla first, then jax, then re-pin libtpu (jax may pull newer)

### Standard Setup (all v6e TPUs)

```bash
# Install Python 3.11
sudo apt-get update -qq && sudo apt-get install -y python3.11-venv python3.11-dev

# Create venv
python3.11 -m venv ~/venv && source ~/venv/bin/activate && pip install --upgrade pip

# PyTorch + XLA (auto-resolves compatible libtpu initially)
pip install 'torch~=2.9.0' 'torch_xla[tpu]~=2.9.0' \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html

# JAX 0.9.0 (needed internally by torch_xla's flash attention kernel)
pip install 'jax==0.9.0' 'jaxlib==0.9.0' \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html

# Re-pin libtpu (jax install may have pulled a newer incompatible version)
pip install 'libtpu==0.0.23.1' \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html

# Training dependencies
pip install pyarrow tokenizers wandb tqdm datasets transformers \
  psutil tabulate scipy regex tiktoken rustbpe
```

Automated: `bash scripts/tpu/setup_v6e.sh` runs all steps above.

### GPU Setup (Vertex AI / A100)

```bash
# Docker image for GPU training
us-central1-docker.pkg.dev/alpine-aspect-459819-m4/nanochat/gpu-trainer:latest

# CRITICAL: Build Docker with --platform linux/amd64 (local machine may be arm64)
docker buildx create --name multiarch --driver docker-container --use
docker buildx build --platform linux/amd64 -t <tag> --push .

# A100 80GB: device_batch_size=32, kernel=cce
# A100 40GB: device_batch_size=8, kernel=cce
# L4: Too slow for training
```

### Local (GB10/SM121)

```bash
# Best performance: Original FA3 + CCE
# See CLAUDE.md for full benchmark results
python -m scripts.base_train --depth=20 --kernel=cce
```

## Environment Variables

These must be set on the TPU before training:

| Variable | Value | Purpose |
| -------------------------------- | --------------------------------- | ----------------------------------------- |
| `PJRT_DEVICE` | `TPU` | **Required** - enables TPU/XLA backend |
| `NANOCHAT_BASE_DIR` | `/home/dave/data` | Location of tokenizer and training data |
| `WANDB_API_KEY` | `<key>` | Weights & Biases logging |
| `NANOCHAT_GCS_CHECKPOINT_BUCKET` | `gs://...checkpoints/<tpu>/` | Auto-upload checkpoints to GCS |
| `XLA_NO_SPECIAL_SCALARS` | `1` | Prevents recompilation per LR change |

> After TPU stop/start, env vars and wandb login are lost. Must re-set them.

### ~/.tpu_env Template

```bash
export PJRT_DEVICE=TPU
export NANOCHAT_BASE_DIR=/home/dave/data
export XLA_NO_SPECIAL_SCALARS=1
```

## Tokenizer Setup

The 65K C++ tokenizer must be available on the TPU:

```bash
mkdir -p $NANOCHAT_BASE_DIR/tokenizer_65k
gcloud storage cp -r gs://nanochat-training-data-2026/tokenizer_65k/ \
  $NANOCHAT_BASE_DIR/tokenizer_65k/

# Create symlink expected by training code
ln -sfn $NANOCHAT_BASE_DIR/tokenizer_65k $NANOCHAT_BASE_DIR/tokenizer
```

### Tokenizer Versions

- **32K tokenizer** (`~/.cache/nanochat/tokenizer/`): Original, 32768 vocab. Local use only.
- **65K tokenizer** (`gs://nanochat-training-data-2026/tokenizer_65k/`): 65536 vocab, expanded C++ vocabulary.
  - 20800 fixed tokens + 44736 BPE tokens
  - 17825 real library tokens (STL, stdlib, Boost, ACE, Qt, POSIX, CUDA, etc.)

## Training Data

### Standard Training Data (base_data_v3)

105 parquet shards for base training:

```bash
mkdir -p $NANOCHAT_BASE_DIR/parquet/base_data_v3/
gcloud storage cp -r gs://nanochat-training-data-2026/parquet/base_data_v3/ \
  $NANOCHAT_BASE_DIR/parquet/base_data_v3/
```

### Long-Context Data (cpp_chunked_16k)

Function-boundary-based C++ chunks with max 16384 tokens:

```bash
mkdir -p $NANOCHAT_BASE_DIR/parquet
gcloud storage cp -r gs://nanochat-training-data-2026/parquet/cpp_chunked_16k/ \
  $NANOCHAT_BASE_DIR/parquet/
```

See [DATA_PIPELINE.md](DATA_PIPELINE.md) for how this data is produced.

## Managing TPUs

```bash
# List all TPUs across zones
gcloud compute tpus tpu-vm list --zone=-

# Check TPU status
gcloud compute tpus tpu-vm describe <name> --zone=<zone> \
  --format="value(state,health)"

# Stop (stops billing, keeps configuration)
gcloud compute tpus tpu-vm stop <name> --zone=<zone>

# Start
gcloud compute tpus tpu-vm start <name> --zone=<zone>

# Delete (permanent)
gcloud compute tpus tpu-vm delete <name> --zone=<zone>
```

## Troubleshooting

### XLA Compilation Takes Forever

First-run XLA compilation takes **30-60+ minutes** (v6e can take 3.5M HLO instructions!). This is normal. Subsequent runs with the same model architecture reuse compiled graphs.

### TPU Goes UNHEALTHY

v5e TPUs sometimes go `UNHEALTHY_TENSORFLOW`. Fix: stop and start the TPU:

```bash
gcloud compute tpus tpu-vm stop <name> --zone=<zone>
gcloud compute tpus tpu-vm start <name> --zone=<zone>
```

If it stays unhealthy, delete and recreate.

### PJRT Crash / Device or Resource Busy

The PJRT runtime can die silently, leaving `/dev/vfio` devices locked. Fix: stop/start the TPU. `kill -9` of XLA processes corrupts PJRT state.

### v5e Silently Runs on CPU

Without `PJRT_DEVICE=TPU`, the script silently runs on CPU. Always `source ~/.tpu_env` before training.

### Preempted TPU Recovery

Preempted TPUs must be **deleted and recreated** (cannot `start` a preempted VM).

**Automated recovery** via babysitter (runs on build3, not on TPU):

```bash
ssh build3
tmux new-session -s babysit
# One babysitter per spot TPU:
bash ~/nanochat/scripts/tpu/babysit_tpu.sh v6e-engram   # window 1
bash ~/nanochat/scripts/tpu/babysit_tpu.sh v6e-longctx   # window 2
bash ~/nanochat/scripts/tpu/babysit_tpu.sh v6e-mhc       # window 3
```

Recovery cycle: detect PREEMPTED → delete → recreate (spot) → deploy code → setup env → download data → find latest GCS checkpoint → resume training.

**Manual recovery:**

```bash
gcloud compute tpus tpu-vm delete <name> --zone=<zone> --quiet
gcloud compute tpus tpu-vm create <name> \
  --zone=<zone> --accelerator-type=<type> --version=v2-alpha-tpuv6e --spot
# Then re-run full environment setup
```

### Disk Space (v6e)

v6e has a small root disk (~97 GB). Clean `/tmp/nanochat_checkpoints/` periodically.

### gsutil OpenSSL Error (v6e)

`gsutil` may fail on v6e with OpenSSL errors. Use `gcloud storage cp` instead.

### JAX/libtpu Version Mismatch

```
RuntimeError: Unexpected PJRT_Plugin_Attributes_Args size: expected 32, got 24
```

This means JAX and libtpu have incompatible PJRT API versions. Fix:
- Python 3.10 (no flash attn): pin `jax==0.4.38 jaxlib==0.4.38 libtpu==0.0.21`
- Python 3.11 (flash attn): pin `jax==0.9.0 jaxlib==0.9.0 libtpu==0.0.23.1`
- Any JAX version from 0.7.0 through 0.9.0 works with libtpu 0.0.23.1

### torch.Generator on XLA

`torch.Generator(device=xla_device)` crashes. Use a CPU generator and move tensors to XLA.

### GCS Checkpoint Naming Mismatch

Old checkpoints used `model_step_25000.pt`, new code expects `model_025000.pt`. Cannot resume across formats.
