# Dockerfile for PyTorch XLA TPU Training on Vertex AI
# Supports TPU v5e and v6e
#
# Build (from project root):
#   docker build -t us-central1-docker.pkg.dev/alpine-aspect-459819-m4/nanochat/tpu-trainer:latest -f vertex_ai/docker/Dockerfile.tpu .
#
# Or use Cloud Build:
#   gcloud builds submit --tag us-central1-docker.pkg.dev/alpine-aspect-459819-m4/nanochat/tpu-trainer:latest .
#
# Push:
#   docker push us-central1-docker.pkg.dev/alpine-aspect-459819-m4/nanochat/tpu-trainer:latest

# Use Google's PyTorch XLA TPU base image (pre-built for TPU v5e/v6e)
# See: https://github.com/pytorch/xla
FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.6.0_3.10_tpuvm

# Set environment variables for TPU
ENV PJRT_DEVICE=TPU
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install additional dependencies for nanochat
RUN pip3 install --no-cache-dir \
    pyarrow>=14.0.0 \
    tokenizers>=0.15.0 \
    wandb>=0.16.0 \
    tqdm>=4.66.0 \
    google-cloud-storage>=2.14.0 \
    gcsfs>=2024.1.0

# Create working directory
WORKDIR /app

# Copy nanochat package (minimal files needed for TPU training)
COPY nanochat/__init__.py /app/nanochat/
COPY nanochat/gpt.py /app/nanochat/
COPY nanochat/common.py /app/nanochat/
COPY nanochat/muon.py /app/nanochat/
COPY nanochat/adamw.py /app/nanochat/
COPY nanochat/kernels.py /app/nanochat/
COPY nanochat/tokenizer.py /app/nanochat/
COPY nanochat/flash_attention.py /app/nanochat/

# Copy TPU trainer script
COPY vertex_ai/trainer/train_tpu.py /app/train_tpu.py

# Set Python path
ENV PYTHONPATH=/app

# Entry point
ENTRYPOINT ["python3", "/app/train_tpu.py"]
