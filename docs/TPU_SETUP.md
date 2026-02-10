# Google Cloud TPU Setup for nanochat

This document describes our Google Cloud TPU infrastructure for training nanochat C++ code generation models.

## Current TPU Instances

We run three TPU VMs with **spot/preemptible pricing** for cost efficiency:

| Name | TPU Type | Zone | Chips | HBM/chip | Purpose | Status |
| -------------------- | ----------- | ----------------- | ----- | -------- | -------------------- | ------- |
| nanochat-tpu | v5litepod-8 | us-west4-a | 8 | 16 GB | Base training 1024 | Running |
| nanochat-v6e-small | v6e-4 | asia-northeast1-b | 4 | 32 GB | Base training 2048 | Running |
| nanochat-v6e-longctx | v6e-4 | asia-northeast1-b | 4 | 32 GB | Long-context 16384 | Setup |

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

| TPU | Config | Throughput | Speedup |
| ------------- | -------------------- | ---------- | ------- |
| v6e-4 (SPMD) | 4 chips + compile | ~320K tok/s | ~6.8x single-chip |
| v5e-8 (SPMD) | 8 chips + compile | ~175K tok/s | ~4.3x single-chip |

## XLA Flash Attention

For long-context training (seq_len >= 4096), we use XLA Flash Attention via Pallas TPU kernels:

```bash
python -m scripts.base_train --xla_flash_attn --max_seq_len=16384 --window_pattern=L
```

### Requirements

- **JAX** must be installed (provides Pallas kernels): `pip install jax==0.4.38 jaxlib==0.4.38`
- **libtpu 0.0.21** (for Python 3.10) or **libtpu 0.0.35** (for Python 3.13)
- **CRITICAL**: JAX 0.6+ is incompatible with libtpu 0.0.21 (PJRT API version mismatch)

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

### v6e Long-Context (4 chips, seq_len=16384, XLA Flash Attention)

```bash
ssh dave@<v6e-longctx-IP>
source /home/dave/venv/bin/activate
source ~/.tpu_env
cd /home/dave/nanochat
export NANOCHAT_BASE_DIR=/home/dave/data
export NANOCHAT_GCS_CHECKPOINT_BUCKET=gs://nanochat-training-data-2026/checkpoints/v6e-longctx
export WANDB_API_KEY=<key>
export XLA_NO_SPECIAL_SCALARS=1

nohup python3 -u -m scripts.base_train \
    --depth=16 --num_iterations=50000 \
    --device_batch_size=4 --max_seq_len=16384 \
    --kernel=current --no_compile \
    --xla_flash_attn --window_pattern=L \
    --data_dir=/home/dave/data/parquet --streaming_data \
    --run=d16_400M_v6e4_longctx_16k_cppeval \
    --core_metric_every=5000 --save_every=5000 --sample_every=5000 \
    > ~/train_longctx.log 2>&1 &
```

### Key Training Flags for TPU

| Flag | Value | Reason |
| ----------------------- | --------------------- | ----------------------------------------------------------- |
| `--kernel=current` | PyTorch native | CCE requires CUDA/Triton (not available on TPU) |
| `--no_compile` | Disable torch.compile | torch.compile doesn't work on TPU/XLA |
| `--device_batch_size=8` | Per-chip batch | v5e: 16GB, v6e: 32GB (don't use 16/32 on v6e - OOM) |
| `--device_batch_size=4` | Long-context batch | 16384 seq_len needs smaller batches |
| `--max_seq_len=1024` | v5e | Shorter sequences for 16GB chips |
| `--max_seq_len=2048` | v6e | Standard sequences for 32GB chips |
| `--max_seq_len=16384` | v6e long-context | Long sequences with XLA flash attention |
| `--xla_flash_attn` | Enable flash attn | Uses Pallas TPU kernels (requires JAX) |
| `--window_pattern=L` | Sliding window | All layers use local attention for 16K context |
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
- Higher memory bandwidth, supports seq_len=2048 or 16384 (with flash attn)
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

### Platform Requirements

The TPU runtime ships with Python 3.10. For latest libtpu (0.0.35+), upgrade to Python 3.13:

| Package | Python 3.10 (current) | Python 3.13 (recommended) |
| ---------- | --------------------- | ------------------------- |
| torch | 2.9.1 | 2.9.1 |
| torch_xla | 2.9.0 | 2.9.0 |
| libtpu | 0.0.21 (max for 3.10) | **0.0.35** (latest) |
| jax | 0.4.38 | latest compatible |
| jaxlib | 0.4.38 | latest compatible |
| pyarrow | 23.0.0 | 23.0.0 |
| wandb | 0.24.2 | 0.24.2 |
| numpy | 2.2.6 | 2.2.6 |

### Python 3.13 Upgrade (Recommended for v6e)

```bash
# Install Python 3.13 via deadsnakes PPA
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.13 python3.13-venv python3.13-dev

# Create new venv
python3.13 -m venv ~/venv313
source ~/venv313/bin/activate

# Install packages (latest versions)
pip install torch==2.9.1 torch_xla==2.9.0 libtpu==0.0.35
pip install jax jaxlib  # latest compatible
pip install wandb pyarrow filelock psutil regex tabulate tiktoken tokenizers
pip install rustbpe  # for tokenizer
```

### Python 3.10 Setup (Legacy / v5e)

```bash
sudo apt-get install python3.10-venv
python3 -m venv ~/venv
source ~/venv/bin/activate

pip install torch~=2.9.0 torch_xla[tpu]~=2.9.0 \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html
pip install wandb filelock pyarrow psutil regex tabulate tiktoken tokenizers rustbpe
pip install jax==0.4.38 jaxlib==0.4.38  # for XLA flash attention
pip install libtpu==0.0.21  # must pin after jax install
```

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

Preempted TPUs must be **deleted and recreated** (cannot `start` a preempted VM):

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
- Python 3.10: pin `jax==0.4.38 jaxlib==0.4.38 libtpu==0.0.21`
- Python 3.13: use latest jax + `libtpu==0.0.35`

### torch.Generator on XLA

`torch.Generator(device=xla_device)` crashes. Use a CPU generator and move tensors to XLA.

### GCS Checkpoint Naming Mismatch

Old checkpoints used `model_step_25000.pt`, new code expects `model_025000.pt`. Cannot resume across formats.
