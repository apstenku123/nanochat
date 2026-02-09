#!/bin/bash
# Quick deploy nanochat to a TPU VM (create if needed, restore env, upload code)
# Run FROM YOUR LOCAL MACHINE (not on the TPU)
#
# Usage:
#   bash scripts/tpu/quick_deploy.sh v6e                    # Setup only
#   bash scripts/tpu/quick_deploy.sh v5e                    # Setup v5e only
#   bash scripts/tpu/quick_deploy.sh v6e --start-training   # Setup + start training
#   bash scripts/tpu/quick_deploy.sh v5e --start-training   # Setup + start training on v5e
#
# What this does:
#   1. Checks if the TPU exists and is READY
#   2. If PREEMPTED or NOT_FOUND, recreates the TPU (spot pricing)
#   3. Uploads nanochat code via scp
#   4. Tries restore from GCS backup (fast, ~2 min)
#   5. Falls back to full setup if no backup (~10 min)
#   6. Optionally starts training in background with nohup

set -e

TPU_TYPE="${1:-v6e}"
START_TRAINING="${2:-}"
NANOCHAT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PROJECT="alpine-aspect-459819-m4"

# TPU configuration based on type
case "$TPU_TYPE" in
    v5e)
        TPU_NAME="nanochat-tpu"
        ZONE="us-west4-a"
        ACCEL_TYPE="v5litepod-8"
        RUNTIME="v2-alpha-tpuv5-lite"
        SPOT_RATE="2.04"
        ;;
    v6e)
        TPU_NAME="nanochat-v6e"
        ZONE="us-east5-b"
        ACCEL_TYPE="v6e-4"
        RUNTIME="v2-alpha-tpuv6e"
        SPOT_RATE="1.70"
        ;;
    *)
        echo "Usage: $0 [v5e|v6e] [--start-training]"
        echo ""
        echo "  v5e  - nanochat-tpu in us-west4-a (v5litepod-8, \$2.04/hr spot)"
        echo "  v6e  - nanochat-v6e in us-east5-b (v6e-4, \$1.70/hr spot)"
        exit 1
        ;;
esac

echo "============================================"
echo "Quick Deploy: $TPU_NAME ($TPU_TYPE)"
echo "Zone: $ZONE"
echo "Accelerator: $ACCEL_TYPE"
echo "Runtime: $RUNTIME"
echo "Spot rate: \$$SPOT_RATE/hr"
echo "Local code: $NANOCHAT_DIR"
echo "============================================"
echo ""

DEPLOY_START=$(date +%s)

# -------------------------------------------------------------------
# Step 1: Ensure TPU exists and is READY
# -------------------------------------------------------------------
echo "=== Checking TPU state ==="
TPU_STATE=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" \
    --zone="$ZONE" --project="$PROJECT" \
    --format="value(state)" 2>/dev/null || echo "NOT_FOUND")
echo "Current state: $TPU_STATE"

if [ "$TPU_STATE" = "READY" ]; then
    echo "TPU is already READY"

elif [ "$TPU_STATE" = "PREEMPTED" ] || [ "$TPU_STATE" = "NOT_FOUND" ] || [ "$TPU_STATE" = "TERMINATED" ]; then
    # Delete if preempted/terminated (can't restart spot instances)
    if [ "$TPU_STATE" != "NOT_FOUND" ]; then
        echo "Deleting $TPU_STATE TPU..."
        gcloud compute tpus tpu-vm delete "$TPU_NAME" \
            --zone="$ZONE" --project="$PROJECT" --quiet
        echo "Deleted"
    fi

    # Create new TPU
    echo ""
    echo "Creating TPU: $TPU_NAME"
    echo "  Accelerator: $ACCEL_TYPE"
    echo "  Runtime: $RUNTIME"
    echo "  Pricing: spot (\$$SPOT_RATE/hr)"
    gcloud compute tpus tpu-vm create "$TPU_NAME" \
        --zone="$ZONE" \
        --project="$PROJECT" \
        --accelerator-type="$ACCEL_TYPE" \
        --runtime-version="$RUNTIME" \
        --spot

    # Wait for READY state
    echo "Waiting for TPU to become READY..."
    for i in $(seq 1 60); do
        STATE=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" \
            --zone="$ZONE" --project="$PROJECT" \
            --format="value(state)" 2>/dev/null || echo "CREATING")
        if [ "$STATE" = "READY" ]; then
            echo "TPU is READY (after ${i}0s)"
            break
        fi
        if [ "$i" -eq 60 ]; then
            echo "ERROR: TPU did not become READY within 10 minutes"
            echo "Last state: $STATE"
            exit 1
        fi
        echo "  State: $STATE (attempt $i/60)"
        sleep 10
    done

elif [ "$TPU_STATE" = "STOPPED" ]; then
    echo "Starting stopped TPU..."
    gcloud compute tpus tpu-vm start "$TPU_NAME" \
        --zone="$ZONE" --project="$PROJECT"
    echo "Waiting for TPU to become READY..."
    for i in $(seq 1 30); do
        STATE=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" \
            --zone="$ZONE" --project="$PROJECT" \
            --format="value(state)" 2>/dev/null || echo "STARTING")
        if [ "$STATE" = "READY" ]; then
            echo "TPU is READY"
            break
        fi
        sleep 10
    done

else
    echo "ERROR: Unexpected TPU state: $TPU_STATE"
    echo "Manually investigate with:"
    echo "  gcloud compute tpus tpu-vm describe $TPU_NAME --zone=$ZONE --project=$PROJECT"
    exit 1
fi

echo ""

# -------------------------------------------------------------------
# Step 2: Upload nanochat code
# -------------------------------------------------------------------
echo "=== Uploading nanochat code ==="

# Create a clean tarball excluding large/unnecessary files
TARBALL="/tmp/nanochat_deploy.tar.gz"
echo "Creating tarball (excluding .git, __pycache__, logs, .cache)..."
tar czf "$TARBALL" \
    -C "$(dirname "$NANOCHAT_DIR")" \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='.cache' \
    --exclude='*.log' \
    --exclude='reports' \
    --exclude='.venv' \
    --exclude='vertex_ai' \
    "$(basename "$NANOCHAT_DIR")"

TARBALL_SIZE=$(du -sh "$TARBALL" | cut -f1)
echo "Tarball size: $TARBALL_SIZE"

echo "Uploading to TPU..."
gcloud compute tpus tpu-vm scp "$TARBALL" \
    "$TPU_NAME":/tmp/nanochat_deploy.tar.gz \
    --zone="$ZONE" --project="$PROJECT"

echo "Extracting on TPU..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" --project="$PROJECT" \
    --command="rm -rf ~/nanochat && tar xzf /tmp/nanochat_deploy.tar.gz -C ~ && rm /tmp/nanochat_deploy.tar.gz && echo 'Code uploaded: $(ls ~/nanochat/ | wc -l) items'"

rm "$TARBALL"
echo "Code uploaded"
echo ""

# -------------------------------------------------------------------
# Step 3: Restore environment (try backup first, then full setup)
# -------------------------------------------------------------------
echo "=== Setting up environment ==="
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" --project="$PROJECT" \
    --command="
        set -e
        if [ -f ~/nanochat/scripts/tpu/restore_env.sh ]; then
            bash ~/nanochat/scripts/tpu/restore_env.sh
        elif [ -f ~/nanochat/scripts/tpu/setup_v6e.sh ]; then
            bash ~/nanochat/scripts/tpu/setup_v6e.sh
        elif [ -f ~/nanochat/scripts/tpu/setup_v5e.sh ]; then
            bash ~/nanochat/scripts/tpu/setup_v5e.sh
        else
            echo 'ERROR: No setup or restore script found'
            exit 1
        fi
    "

echo ""

# -------------------------------------------------------------------
# Step 4: Optionally start training
# -------------------------------------------------------------------
if [ "$START_TRAINING" = "--start-training" ]; then
    echo "=== Starting training ==="
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
        --zone="$ZONE" --project="$PROJECT" \
        --command="
            source ~/venv/bin/activate
            [ -f ~/.tpu_env ] && source ~/.tpu_env
            cd ~/nanochat
            export PJRT_DEVICE=TPU
            export WANDB_API_KEY=\${WANDB_API_KEY}
            export NANOCHAT_BASE_DIR=~/data
            export NANOCHAT_GCS_CHECKPOINT_BUCKET=gs://nanochat-training-data-2026/checkpoints
            nohup python3 -u -m scripts.base_train \
                --depth=16 \
                --num_iterations=50000 \
                --fim_rate=0.4 \
                --structured_fim_rate=0.2 \
                --kernel=current \
                --no_compile \
                --run=d16_400M_${TPU_TYPE}_65k_v3 \
                --device_batch_size=32 \
                --total_batch_size=524288 \
                --max_seq_len=2048 \
                --eval_every=1000 \
                --core_metric_every=5000 \
                --save_every=5000 \
                > ~/train_${TPU_TYPE}.log 2>&1 &
            TRAIN_PID=\$!
            echo ''
            echo 'Training started in background'
            echo \"  PID: \$TRAIN_PID\"
            echo \"  Log: ~/train_${TPU_TYPE}.log\"
            echo ''
            echo 'Monitor with:'
            echo \"  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --command=\\\"tail -f ~/train_${TPU_TYPE}.log\\\"\"
        "
fi

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------
DEPLOY_END=$(date +%s)
ELAPSED=$((DEPLOY_END - DEPLOY_START))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "============================================"
echo "Deploy complete in ${MINUTES}m ${SECONDS}s"
echo "============================================"
echo ""
echo "TPU: $TPU_NAME ($ACCEL_TYPE)"
echo "Zone: $ZONE"
echo "Cost: \$$SPOT_RATE/hr (spot)"
echo ""
echo "SSH into TPU:"
echo "  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT"
echo ""
echo "Start training manually:"
echo "  source ~/venv/bin/activate && source ~/.tpu_env"
echo "  cd ~/nanochat && python3 -u -m scripts.base_train --depth=16 ..."
echo ""
echo "Backup environment (before stopping TPU):"
echo "  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --command=\"bash ~/nanochat/scripts/tpu/backup_env.sh\""
