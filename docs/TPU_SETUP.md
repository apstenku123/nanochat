# Google Cloud TPU Setup for nanochat

This document describes our Google Cloud TPU infrastructure for training nanochat C++ code generation models.

## Current TPU Instances

We run two TPU VMs with **spot/preemptible pricing** for cost efficiency:

| Name               | TPU Type    | Zone              | Chips | HBM/chip | Status  |
| ------------------ | ----------- | ----------------- | ----- | -------- | ------- |
| nanochat-tpu       | v5litepod-8 | us-west4-a        | 8     | 16 GB    | Running |
| nanochat-v6e-small | v6e-4       | asia-northeast1-b | 4     | 32 GB    | Running |

### Project Configuration

- **Project ID**: `alpine-aspect-459819-m4`
- **GCS Bucket**: `gs://nanochat-training-data-2026/`
- **W&B Project**: `misaacson-texas-state-technical-college/nanochat`

## Multi-Chip Training

nanochat uses **all TPU chips** on single-host TPUs via `xmp.spawn()` data parallelism. Each chip runs an independent process with its own model copy; gradients are averaged across chips via `xm.all_reduce`.

- **v5litepod-8**: 8 processes, one per chip (Distributed world size: 8)
- **v6e-4**: 4 processes, one per chip (Distributed world size: 4)

This is handled automatically by `scripts/base_train.py` when `PJRT_DEVICE=TPU` is set. The script detects the number of chips from the accelerator type string (e.g. `v5litepod-8` -> 8 chips) and spawns processes accordingly.

### How It Works

1. `base_train.py` parses args and sets up kernel config at **module level** (no XLA touching)
2. At the bottom, `main()` calls `xmp.spawn(_mp_fn, nprocs=None)` which auto-detects all devices
3. Each spawned process calls `train()` which does `compute_init()`, gets assigned its own XLA device
4. Dataloader shards data by rank with stride `world_size` (in `dataloader.py`)
5. Gradients are averaged via `xla_all_reduce_gradients()` (in `common.py`)
6. Only rank 0 logs to WandB, saves model checkpoints

### Important: XLA Runtime Initialization

`xmp.spawn()` requires that the XLA runtime is **not initialized** before it's called. This means:

- **No `print0()`, `print_banner()`, or `get_dist_info()` at module level** - these call `xr.world_size()` which initializes the runtime
- **No `xm.xla_device()` at module level** - creates an XLA device
- **All XLA-touching code must be inside `train()`**

If you see `RuntimeError: Runtime is already initialized. Do not use the XLA device before calling xmp.spawn.`, check that no module-level code is touching XLA.

### Batch Size Math

With multi-chip, gradient accumulation steps decrease proportionally:

```
# v5e with 8 chips:
grad_accum_steps = 524288 / (8 * 1024 * 8) = 8

# v6e with 4 chips:
grad_accum_steps = 524288 / (8 * 2048 * 4) = 8
```

## Connecting to TPUs

### SSH via gcloud

```bash
# v5e TPU (us-west4-a)
gcloud compute tpus tpu-vm ssh nanochat-tpu --zone=us-west4-a

# v6e TPU (asia-northeast1-b)
gcloud compute tpus tpu-vm ssh nanochat-v6e-small --zone=asia-northeast1-b
```

### Direct SSH

gcloud SSH wrapper can be flaky (retries with exit code 255). Direct SSH to the external IP is more reliable:

```bash
# Get the external IP
gcloud compute tpus tpu-vm describe nanochat-tpu --zone=us-west4-a \
  --format="value(networkEndpoints[0].accessConfig.externalIp)"

# SSH directly
ssh -o StrictHostKeyChecking=no dave@<IP>
```

Note: After TPU stop/start or recreation, the host key changes. Remove stale entries:
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
export PJRT_DEVICE=TPU
export NANOCHAT_BASE_DIR=/home/dave/data
export WANDB_API_KEY=<key>

cd /home/dave/nanochat
nohup python3 -m scripts.base_train \
    --depth=16 \
    --num_iterations=50000 \
    --device_batch_size=8 \
    --max_seq_len=1024 \
    --kernel=current \
    --no_compile \
    --run=d16_400M_tpu_v5e_65k_v6_8chip \
    > ~/train_v5e_8chip.log 2>&1 &
```

### v6e (4 chips, seq_len=2048)

```bash
ssh dave@<v6e-IP>
source /home/dave/venv/bin/activate  # v6e needs venv
source ~/.tpu_env
export PJRT_DEVICE=TPU
export WANDB_API_KEY=<key>

cd /home/dave/nanochat
nohup python3 -m scripts.base_train \
    --depth=16 \
    --num_iterations=50000 \
    --device_batch_size=8 \
    --max_seq_len=2048 \
    --kernel=current \
    --no_compile \
    --run=d16_400M_v6e4_65k_v9_4chip \
    > ~/train_v6e_4chip.log 2>&1 &
```

### Key Training Flags for TPU

| Flag                    | Value                 | Reason                                                      |
| ----------------------- | --------------------- | ----------------------------------------------------------- |
| `--kernel=current`      | PyTorch native        | CCE requires CUDA/Triton (not available on TPU)             |
| `--no_compile`          | Disable torch.compile | torch.compile doesn't work on TPU/XLA                       |
| `--device_batch_size=8` | Per-chip batch        | v5e: 16GB HBM, v6e: 32GB HBM (don't use 16/32 on v6e - OOM) |
| `--max_seq_len=1024`    | v5e                   | Shorter sequences for 16GB chips                            |
| `--max_seq_len=2048`    | v6e                   | Longer sequences for 32GB chips                             |

## Cost Estimates

Using **spot/preemptible pricing**:

| TPU Type    | On-Demand ($/hr) | Spot ($/hr) | Our Config         |
| ----------- | ---------------- | ----------- | ------------------ |
| v6e-4       | ~$5.10           | **~$1.70**  | nanochat-v6e-small |
| v5litepod-8 | ~$6.12           | **~$2.04**  | nanochat-tpu       |

**Combined hourly cost**: ~$3.74/hr (spot pricing)

> Spot instances can be preempted with 30 seconds notice. Save checkpoints frequently!

## TPU Type Comparison

### v6e (Trillium) - 32 GB HBM/chip
- Latest generation TPU (2024)
- Higher memory bandwidth, supports seq_len=2048
- **v6e-4 is the max single-host size** (4 chips)

### v5e (v5litepod) - 16 GB HBM/chip
- Previous generation, still excellent
- Limited to seq_len=1024 (memory)
- **v5litepod-8 is the max single-host size** (8 chips)

> Sizes above v5litepod-8 and v6e-4 are **multi-host** (e.g. v5litepod-16 = 2 hosts x 8 chips), which requires a different training setup not currently supported.

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

### v6e Post-Creation Setup

The v6e runtime doesn't include Python venv by default:

```bash
sudo apt-get install python3.10-venv
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install torch~=2.9.0 torch_xla[tpu]~=2.9.0 \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html
pip install wandb filelock pyarrow
```

### Available Zones

| Type | Zones                                        |
| ---- | -------------------------------------------- |
| v6e  | asia-northeast1-b, us-east5-b, us-central2-b |
| v5e  | us-west4-a, us-central1-a, europe-west4-a    |

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

## Environment Variables

These must be set on the TPU before training:

| Variable            | Value                   | Purpose                                 |
| ------------------- | ----------------------- | --------------------------------------- |
| `PJRT_DEVICE`       | `TPU`                   | **Required** - enables TPU/XLA backend  |
| `NANOCHAT_BASE_DIR` | `/home/dave/data` (v5e) | Location of tokenizer and training data |
| `WANDB_API_KEY`     | `<key>`                 | Weights & Biases logging                |

> After TPU stop/start, env vars and wandb login are lost. Must re-set them.

### ~/.tpu_env Template

```bash
export PJRT_DEVICE=TPU
export NANOCHAT_BASE_DIR=/home/dave/data
export WANDB_API_KEY=<your-key>
```

## Tokenizer Setup

The 65K C++ tokenizer must be available on the TPU. Download from GCS:

```bash
# On the TPU
gcloud storage cp -r gs://nanochat-training-data-2026/tokenizer_65k/ \
  $NANOCHAT_BASE_DIR/tokenizer/
gcloud storage cp gs://nanochat-training-data-2026/tokenizer_65k/token_bytes.pt \
  $NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt
```

## Training Data

Parquet training data shards are on GCS:

```bash
# Download training data (105 shards)
mkdir -p $NANOCHAT_BASE_DIR/parquet/base_data_v3/
gcloud storage cp -r gs://nanochat-training-data-2026/parquet/base_data_v3/ \
  $NANOCHAT_BASE_DIR/parquet/base_data_v3/

# Download validation data
gcloud storage cp -r gs://nanochat-training-data-2026/parquet/val/ \
  $NANOCHAT_BASE_DIR/parquet/val/
```

## Troubleshooting

### XLA Compilation Takes Forever

First-run XLA compilation takes **30-60+ minutes** (v6e can take 3.5M HLO instructions!). This is normal. Subsequent runs with the same model architecture reuse compiled graphs.

### TPU Goes UNHEALTHY

v5e TPUs sometimes go `UNHEALTHY_TENSORFLOW`. Fix: stop and start the TPU:

```bash
gcloud compute tpus tpu-vm stop nanochat-tpu --zone=us-west4-a
gcloud compute tpus tpu-vm start nanochat-tpu --zone=us-west4-a
```

If it stays unhealthy, delete and recreate.

### PJRT Crash (Ports Refuse)

The v5e PJRT runtime can die silently during XLA compilation (ports 8466/8472 refuse connections). Fix: stop/start the TPU.

### v5e Silently Runs on CPU

Without `PJRT_DEVICE=TPU`, the script silently runs on CPU. Always `source ~/.tpu_env` before training.

### Disk Space (v6e)

v6e has a small root disk (~97 GB). Clean `/tmp/nanochat_checkpoints/` periodically.

### gsutil OpenSSL Error (v6e)

`gsutil` may fail on v6e with OpenSSL errors. Use `gcloud storage cp` instead.

### torch.Generator on XLA

`torch.Generator(device=xla_device)` crashes. Use a CPU generator and move tensors to XLA.

## PyTorch/XLA Software Versions

| Component | v5e         | v6e         |
| --------- | ----------- | ----------- |
| Python    | 3.10        | 3.10        |
| torch     | 2.9.1+cu128 | 2.9.1+cu128 |
| torch_xla | 2.9.0       | 2.9.0       |
| libtpu    | 0.0.21      | 0.0.21      |

### API Notes

- Use `torch_xla.runtime.world_size()` and `torch_xla.runtime.global_ordinal()` (not deprecated `xm.xrt_world_size()`/`xm.get_ordinal()`)
- `fused=True` in AdamW breaks on XLA - omit it
- `LIBTPU_INIT_ARGS=--xla_tpu_disable_full_embedding_pipelining=true` is NOT supported by libtpu 0.0.21
