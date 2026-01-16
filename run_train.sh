#!/bin/bash
# Run single-GPU training for d20 model with BF16 or NVFP4
# Checkpoints every 2000 steps, 20000 steps total

cd "$(dirname "$0")"
source .venv/bin/activate

PRECISION="${1:-nvfp4}"  # Default to nvfp4, can pass bf16 as argument
DEPTH="${2:-20}"
BATCH_SIZE=8
TOTAL_STEPS=20000
SAVE_EVERY=2000

echo "Starting training with:"
echo "  Precision: $PRECISION"
echo "  Depth: $DEPTH"
echo "  Device batch size: $BATCH_SIZE"
echo "  Total steps: $TOTAL_STEPS"
echo "  Save every: $SAVE_EVERY steps"
echo ""

# PYTHONUNBUFFERED=1 forces immediate output flushing
nohup env PYTHONUNBUFFERED=1 python -m scripts.base_train \
    --depth=$DEPTH \
    --precision=$PRECISION \
    --device_batch_size=$BATCH_SIZE \
    --total_batch_size=262144 \
    --num_iterations=$TOTAL_STEPS \
    --save_every=$SAVE_EVERY \
    --eval_every=500 \
    --core_metric_every=-1 \
    --sample_every=-1 \
    > report.log 2>&1 &

echo $! > run.pid
echo "Training started with PID: $(cat run.pid)"
echo "Logs: tail -f report.log"
