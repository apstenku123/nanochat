#!/bin/bash
# Setup script for TPU v6e with verified compatible packages
# Based on research: https://cloud.google.com/tpu/docs/v6e-training
# and https://github.com/pytorch/xla/releases
#
# Usage: gcloud compute tpus tpu-vm ssh nanochat-v6e --zone=us-east5-b --command="bash -s" < scripts/tpu/setup_v6e.sh

set -e

echo "============================================"
echo "TPU v6e Setup - Verified Package Versions"
echo "============================================"

# Install python3-venv (required on v2-alpha-tpuv6e runtime)
echo "Ensuring python3-venv is available..."
sudo apt-get update -qq
sudo apt-get install -y python3.10-venv python3.11-venv 2>/dev/null || true

# Prefer Python 3.11 if available, fall back to 3.10
if command -v python3.11 &>/dev/null && python3.11 -m venv --help &>/dev/null; then
    PYTHON=python3.11
    echo "Using Python 3.11"
elif python3 -m venv --help &>/dev/null; then
    PYTHON=python3
    echo "Using $(python3 --version 2>&1)"
else
    echo "ERROR: No Python with venv support found"
    exit 1
fi

# Create venv if needed
if [ ! -d "$HOME/venv" ] || [ ! -f "$HOME/venv/bin/activate" ]; then
    rm -rf "$HOME/venv"
    echo "Creating virtual environment..."
    $PYTHON -m venv "$HOME/venv"
fi
source "$HOME/venv/bin/activate"

# Install verified compatible packages for v6e
# torch_xla[tpu] auto-resolves compatible libtpu
echo "=== Installing PyTorch + XLA + libtpu (verified combo) ==="
pip install --upgrade pip

# Use latest stable matched torch/torch_xla minor for TPU support.
# Keep both packages aligned (currently 2.9.x).
pip install 'torch~=2.9.0' 'torch_xla[tpu]~=2.9.0' \
    -f https://storage.googleapis.com/libtpu-releases/index.html \
    -f https://storage.googleapis.com/libtpu-wheels/index.html

echo "=== Installing nanochat dependencies ==="
pip install pyarrow tokenizers wandb tqdm datasets transformers \
    psutil tabulate scipy regex tiktoken rustbpe

echo "=== Installing JAX for XLA Flash Attention (Pallas TPU kernels) ==="
pip install 'jax==0.7.0' 'jaxlib==0.7.0' \
    -f https://storage.googleapis.com/libtpu-releases/index.html \
    -f https://storage.googleapis.com/libtpu-wheels/index.html
# Pin libtpu to match torch_xla requirements
pip install 'libtpu==0.0.21' \
    -f https://storage.googleapis.com/libtpu-releases/index.html \
    -f https://storage.googleapis.com/libtpu-wheels/index.html

echo "=== Verifying installation ==="
python3 -c "
import torch
import torch_xla
import torch_xla.core.xla_model as xm
print(f'torch: {torch.__version__}')
print(f'torch_xla: {torch_xla.__version__}')
try:
    import libtpu
    print(f'libtpu: {libtpu.__version__}')
except:
    print('libtpu: installed (no version attr)')
dev = xm.xla_device()
print(f'Device: {dev}')
print(f'Num devices: {torch_xla.device_count()}')
print('TPU verification: PASSED')
"

echo "=== Setting up environment variables ==="
cat > "$HOME/.tpu_env" << 'ENVEOF'
export PJRT_DEVICE=TPU
export NANOCHAT_BASE_DIR=/home/dave/data
export XLA_NO_SPECIAL_SCALARS=1
ENVEOF

echo "source \$HOME/.tpu_env" >> "$HOME/.bashrc" 2>/dev/null || true

echo "=== Setup complete ==="
echo "To start training, run:"
echo "  source ~/venv/bin/activate && source ~/.tpu_env"
echo "  cd ~/nanochat && python3 -u -m scripts.base_train --depth=16 ..."
