#!/bin/bash
# TPU v5e/v6e Full Training Pipeline
# Run this script on a TPU VM

set -e

echo "=============================================="
echo "nanochat TPU v5e/v6e Training Pipeline"
echo "=============================================="
echo "Started at: $(date)"
echo ""

# Configuration
export PJRT_DEVICE=TPU
export XLA_USE_SPMD=1
export LIBTPU_INIT_ARGS="--xla_enable_async_collective_permute=true"
export NANOCHAT_BASE_DIR=/home/dave/.cache/nanochat
TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-$(curl -fs -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type 2>/dev/null || true)}"
export TPU_ACCELERATOR_TYPE

if [[ "$TPU_ACCELERATOR_TYPE" == *"v6"* ]]; then
    TPU_SPOT_RATE=1.70
else
    TPU_SPOT_RATE=2.04
fi

# Create directories
mkdir -p $NANOCHAT_BASE_DIR
mkdir -p ~/nanochat/logs

# Install Python dependencies
echo "Installing dependencies..."
cd ~/nanochat
pip3 install --quiet --upgrade pip
pip3 install --quiet 'torch~=2.9.0' 'torch_xla[tpu]~=2.9.0' \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html
pip3 install --quiet transformers datasets tokenizers tqdm google-cloud-storage

# Verify TPU is available
echo "Checking TPU availability..."
python3 -c "
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
device = xm.xla_device()
print(f'TPU Device: {device}')
print(f'World Size: {xr.world_size()}')
print(f'Accelerator Type: ${TPU_ACCELERATOR_TYPE:-unknown}')
" || {
    echo "ERROR: TPU not available. Running in CPU mode."
    unset PJRT_DEVICE
}

# Record start time
START_TIME=$(date +%s)

echo ""
echo "=============================================="
echo "Stage 1: Downloading data from GCS"
echo "=============================================="

# Download data if not present
mkdir -p data/tokenizer data/eval

# Download tokenizer
echo "Downloading tokenizer..."
gsutil -m cp -n gs://nanochat-training-data-2026/tokenizer/* data/tokenizer/ 2>/dev/null || true

# Download training data
echo "Downloading training data..."
gsutil -m cp -n gs://nanochat-training-data-2026/data/combined_sft.jsonl data/ 2>/dev/null || true
gsutil -m cp -n gs://nanochat-training-data-2026/data/gspo_prompts.jsonl data/ 2>/dev/null || true
gsutil -m cp -n gs://nanochat-training-data-2026/data/docstring_pairs_full.jsonl data/ 2>/dev/null || true

# Download eval data
echo "Downloading eval data..."
gsutil -m cp -n gs://nanochat-training-data-2026/eval/humaneval_cpp.jsonl data/eval/ 2>/dev/null || true

echo ""
echo "=============================================="
echo "Stage 2: Running Base Training"
echo "=============================================="

# Base training with reduced iterations for testing
python3 -m scripts.base_train \
    --depth=16 \
    --num_iterations=1000 \
    --device_batch_size=8 \
    --total_batch_size=65536 \
    --max_seq_len=1024 \
    --eval_every=250 \
    --save_every=500 \
    --core_metric_every=-1 \
    --sample_every=-1 \
    --kernel=current \
    --run=dummy \
    2>&1 | tee logs/base_train.log

BASE_CHECKPOINT=$NANOCHAT_BASE_DIR/base_checkpoints/d16

echo ""
echo "=============================================="
echo "Stage 3: Running SFT Training"
echo "=============================================="

if [ -d "$BASE_CHECKPOINT" ]; then
    python3 -m scripts.sft_train \
        --data data/combined_sft.jsonl \
        --checkpoint_path $BASE_CHECKPOINT \
        --epochs 2 \
        --batch_size 8 \
        --max_seq_len 1024 \
        --lr 2e-4 \
        --kernel current \
        2>&1 | tee logs/sft_train.log
else
    echo "ERROR: Base checkpoint not found at $BASE_CHECKPOINT"
    echo "Skipping SFT training"
fi

SFT_CHECKPOINT=$NANOCHAT_BASE_DIR/sft_checkpoints/d16

echo ""
echo "=============================================="
echo "Stage 4: Running GSPO Training"
echo "=============================================="

if [ -d "$SFT_CHECKPOINT" ]; then
    python3 -m scripts.gspo_train \
        --checkpoint_path $SFT_CHECKPOINT \
        --prompts data/gspo_prompts.jsonl \
        --num_iterations 100 \
        --group_size 4 \
        --prompts_per_step 2 \
        --lr 5e-5 \
        --save_every 50 \
        --kernel current \
        2>&1 | tee logs/gspo_train.log
else
    echo "ERROR: SFT checkpoint not found at $SFT_CHECKPOINT"
    echo "Skipping GSPO training"
fi

GSPO_CHECKPOINT=$NANOCHAT_BASE_DIR/gspo_checkpoints/d16

echo ""
echo "=============================================="
echo "Stage 5: Running Evaluation"
echo "=============================================="

if [ -d "$GSPO_CHECKPOINT" ]; then
    python3 -m scripts.cpp_eval \
        --model-tag d16 \
        --max-samples 50 \
        --num-samples 3 \
        --temperature 0.2 \
        2>&1 | tee logs/eval.log
else
    echo "ERROR: GSPO checkpoint not found at $GSPO_CHECKPOINT"
    echo "Skipping evaluation"
fi

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo ""
echo "=============================================="
echo "TRAINING COMPLETE"
echo "=============================================="
echo "Total time: ${HOURS}h ${MINUTES}m"
echo "Estimated cost: \$$(echo "scale=2; $ELAPSED / 3600 * $TPU_SPOT_RATE" | bc)"
echo ""
echo "Logs saved to: ~/nanochat/logs/"
echo "Checkpoints saved to: $NANOCHAT_BASE_DIR/"
echo ""
echo "Finished at: $(date)"

# Generate summary report
cat << EOF > logs/training_report.txt
============================================
NANOCHAT TPU TRAINING REPORT
============================================
Date: $(date)
Accelerator type: ${TPU_ACCELERATOR_TYPE:-unknown}
Total Time: ${HOURS}h ${MINUTES}m (${ELAPSED}s)
Estimated Cost: \$$(echo "scale=2; $ELAPSED / 3600 * $TPU_SPOT_RATE" | bc)

Training Configuration:
- Model depth: 16
- Max sequence length: 1024
- Base training iterations: 1000
- SFT epochs: 2
- GSPO iterations: 100

Checkpoint Locations:
- Base: $BASE_CHECKPOINT
- SFT: $SFT_CHECKPOINT
- GSPO: $GSPO_CHECKPOINT

Logs:
- base_train.log
- sft_train.log
- gspo_train.log
- eval.log
============================================
EOF

echo "Report saved to: logs/training_report.txt"
