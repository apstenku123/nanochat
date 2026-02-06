# Google Cloud TPU Setup for nanochat

This document describes our Google Cloud TPU infrastructure for training nanochat models.

## Current TPU Instances

We have two TPU VMs running with **spot/preemptible pricing** for cost efficiency:

| Name | TPU Type | Zone | External IP | Cores | Status |
|------|----------|------|-------------|-------|--------|
| nanochat-v6e | v6e-4 | us-east5-b | 34.152.124.135 | 4 | Running |
| nanochat-tpu | v5e-8 | us-west4-a | 34.125.174.87 | 8 | Running |

### Project Configuration

- **Project ID**: `alpine-aspect-459819-m4`
- **Service Account**: `google_test@leadexplorer.io`

## Connecting to TPUs

### SSH via gcloud

```bash
# Connect to v6e-4 TPU (us-east5-b)
gcloud compute tpus tpu-vm ssh nanochat-v6e --zone=us-east5-b

# Connect to v5e-8 TPU (us-west4-a)
gcloud compute tpus tpu-vm ssh nanochat-tpu --zone=us-west4-a
```

### Direct SSH (if configured)

```bash
ssh 34.152.124.135  # nanochat-v6e
ssh 34.125.174.87   # nanochat-tpu
```

## Cost Estimates

Using **spot/preemptible pricing** (significantly cheaper than on-demand):

| TPU Type | On-Demand ($/hr) | Spot ($/hr) | Our Config |
|----------|------------------|-------------|------------|
| v6e-4 | ~$5.10 | **~$1.70** | nanochat-v6e |
| v5e-8 | ~$6.12 | **~$2.04** | nanochat-tpu |

**Combined hourly cost**: ~$3.74/hr (spot pricing)

> **Note**: Spot instances can be preempted with 30 seconds notice. Save checkpoints frequently!

## TPU Type Comparison

### v6e (Trillium)
- Latest generation TPU (2024)
- Higher memory bandwidth
- Better for large models
- Best price/performance for training

### v5e (v5litepod)
- Previous generation, still excellent
- Good balance of cost and performance
- Widely available

## Available Zones

### v6e Available Zones
- `us-east5-b` (our current setup)
- `us-central2-b`
- `asia-northeast1-b`

### v5e (v5litepod) Available Zones
- `us-west4-a` (our current setup)
- `us-central1-a`
- `europe-west4-a`

## Creating New TPU VMs

### 1. Enable TPU API (first time only)

```bash
gcloud services enable tpu.googleapis.com --project=alpine-aspect-459819-m4
```

### 2. Create a v6e TPU

```bash
# Create v6e-4 with spot pricing
# IMPORTANT: v6e requires the v2-alpha-tpuv6e runtime, not tpu-ubuntu2204-base
gcloud compute tpus tpu-vm create <name> \
  --zone=us-east5-b \
  --accelerator-type=v6e-4 \
  --version=v2-alpha-tpuv6e \
  --spot \
  --project=alpine-aspect-459819-m4
```

### 3. Create a v5e TPU

```bash
# Create v5e-8 with spot pricing
gcloud compute tpus tpu-vm create <name> \
  --zone=us-west4-a \
  --accelerator-type=v5litepod-8 \
  --version=v2-alpha-tpuv5-lite \
  --spot \
  --project=alpine-aspect-459819-m4
```

> Runtime note: use `v2-alpha-tpuv5-lite` for v5e and `v2-alpha-tpuv6e` for v6e.

### Available Accelerator Types

| Type | Cores | Memory | Use Case |
|------|-------|--------|----------|
| v6e-4 | 4 | 32 GB | Small/medium models |
| v6e-8 | 8 | 64 GB | Medium models |
| v6e-16 | 16 | 128 GB | Large models |
| v5litepod-4 | 4 | 16 GB | Small models |
| v5litepod-8 | 8 | 32 GB | Small/medium models |
| v5litepod-16 | 16 | 64 GB | Medium models |

## Managing TPUs

### List TPUs

```bash
# List all TPUs in a zone
gcloud compute tpus tpu-vm list --zone=us-east5-b

# List TPUs across all zones
gcloud compute tpus tpu-vm list --zone=-
```

### Stop/Start TPUs

```bash
# Stop a TPU (stops billing, keeps configuration)
gcloud compute tpus tpu-vm stop nanochat-v6e --zone=us-east5-b

# Start a stopped TPU
gcloud compute tpus tpu-vm start nanochat-v6e --zone=us-east5-b
```

### Delete TPUs

```bash
# Delete a TPU (permanent, stops all billing)
gcloud compute tpus tpu-vm delete nanochat-v6e --zone=us-east5-b
```

## Troubleshooting

### TPU Not Found

```bash
# Check if TPU exists
gcloud compute tpus tpu-vm describe nanochat-v6e --zone=us-east5-b
```

### Preemption Recovery

If your spot instance is preempted:

1. Check status: `gcloud compute tpus tpu-vm describe <name> --zone=<zone>`
2. Recreate if needed with same command used to create
3. Restore from latest checkpoint

### SSH Connection Issues

```bash
# Refresh SSH keys
gcloud compute config-ssh

# Force key regeneration
gcloud compute tpus tpu-vm ssh nanochat-v6e --zone=us-east5-b --ssh-key-file=~/.ssh/google_compute_engine
```

## Best Practices

1. **Save checkpoints frequently** - Spot instances can be preempted
2. **Use GCS for data storage** - Persistent across TPU recreations
3. **Monitor costs** - Set up billing alerts in Cloud Console
4. **Use spot pricing** - 60-70% cheaper than on-demand
5. **Choose the right zone** - Availability varies by zone and time

## PyTorch/XLA API Notes

- Prefer `torch_xla.runtime.world_size()` and `torch_xla.runtime.global_ordinal()` over deprecated `xm.xrt_world_size()`/`xm.get_ordinal()`.
- Prefer calling `torch_xla.runtime.use_spmd()` in code for SPMD mode (the `XLA_USE_SPMD` env var still works for compatibility).
