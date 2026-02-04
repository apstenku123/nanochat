# TPU Quick Start Guide

Get up and running with nanochat on Google Cloud TPUs in minutes.

## Prerequisites

### 1. Install Google Cloud CLI

```bash
# Linux/macOS
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Verify installation
gcloud --version
```

### 2. Authenticate and Configure

```bash
# Login to Google Cloud
gcloud auth login

# Set the project
gcloud config set project alpine-aspect-459819-m4

# Set default zone (optional)
gcloud config set compute/zone us-east5-b
```

## Connect to Existing TPUs

We have two TPUs ready to use:

```bash
# Connect to v6e-4 TPU (recommended for development)
gcloud compute tpus tpu-vm ssh nanochat-v6e --zone=us-east5-b

# Connect to v5e-8 TPU (more cores)
gcloud compute tpus tpu-vm ssh nanochat-tpu --zone=us-west4-a
```

## Install PyTorch/XLA on TPU

Once connected to a TPU VM:

```bash
# Create virtual environment
python3 -m venv ~/.venv
source ~/.venv/bin/activate

# Install PyTorch with XLA support for TPU
pip install torch torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html

# Verify TPU is detected
python -c "import torch_xla.core.xla_model as xm; print(f'TPU: {xm.xla_device()}')"
```

### For v6e TPUs (Trillium)

```bash
# v6e requires the latest libtpu
pip install torch torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html --upgrade
```

## Clone and Setup nanochat

```bash
# Clone the repository
git clone https://github.com/your-org/nanochat.git
cd nanochat

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

## Basic Training Example

### Simple Training Script

```python
#!/usr/bin/env python3
"""Basic TPU training example for nanochat."""

import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

# Get TPU device
device = xm.xla_device()
print(f"Running on: {device}")

# Your model
from nanochat.gpt import GPT, GPTConfig

config = GPTConfig(
    vocab_size=50257,
    n_layer=12,
    n_head=12,
    n_embd=768,
)
model = GPT(config).to(device)

# Training loop with XLA
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for step in range(100):
    # Your training code here
    optimizer.zero_grad()

    # Forward pass
    # loss = model(input_ids, targets)

    # Backward pass
    # loss.backward()

    # XLA-specific: mark step for compilation
    xm.mark_step()

    # Optimizer step
    # xm.optimizer_step(optimizer)

print("Training complete!")
```

### Run Training

```bash
# Single TPU core
python train.py

# Multi-core TPU (distributed)
python -m torch_xla.distributed.xla_multiprocessing train.py
```

## File Transfer

### Copy files to TPU

```bash
# Copy a file
gcloud compute tpus tpu-vm scp ./local_file.py nanochat-v6e:~/remote_file.py --zone=us-east5-b

# Copy a directory
gcloud compute tpus tpu-vm scp --recurse ./nanochat nanochat-v6e:~/ --zone=us-east5-b
```

### Copy files from TPU

```bash
# Copy checkpoint from TPU
gcloud compute tpus tpu-vm scp nanochat-v6e:~/checkpoints/model.pt ./local_checkpoint.pt --zone=us-east5-b
```

## Using Google Cloud Storage

For large datasets and checkpoints, use GCS:

```bash
# Install gsutil (included with gcloud)
gsutil mb gs://nanochat-data  # Create bucket (one time)

# Upload data
gsutil cp -r ./data gs://nanochat-data/

# Download data on TPU
gsutil cp -r gs://nanochat-data/train.bin ~/data/
```

## Environment Variables for TPU

```bash
# Add to ~/.bashrc on TPU VM
export XLA_USE_BF16=1  # Use bfloat16 for better performance
export TPU_NUM_DEVICES=4  # For v6e-4

# For debugging
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump"
```

## Quick Reference

| Task | Command |
|------|---------|
| Connect to v6e | `gcloud compute tpus tpu-vm ssh nanochat-v6e --zone=us-east5-b` |
| Connect to v5e | `gcloud compute tpus tpu-vm ssh nanochat-tpu --zone=us-west4-a` |
| Copy to TPU | `gcloud compute tpus tpu-vm scp ./file tpu-name:~/path --zone=ZONE` |
| List TPUs | `gcloud compute tpus tpu-vm list --zone=-` |
| Check TPU status | `gcloud compute tpus tpu-vm describe nanochat-v6e --zone=us-east5-b` |

## Next Steps

1. Read [TPU_SETUP.md](./TPU_SETUP.md) for detailed infrastructure documentation
2. Check the training scripts in `scripts/` directory
3. Review model configurations in `nanochat/gpt.py`

## Troubleshooting

### "No TPU devices found"

```bash
# Verify TPU is accessible
ls /dev/accel*

# Check libtpu installation
python -c "import libtpu"
```

### Slow first iteration

This is normal - XLA compiles the graph on first run. Subsequent iterations are fast.

### Out of memory

- Reduce batch size
- Use gradient checkpointing
- Try a smaller model configuration
