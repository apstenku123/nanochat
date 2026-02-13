#!/bin/bash
# Babysit a spot/preemptible TPU: auto-recreate on preemption, resume training from latest GCS checkpoint.
# Run FROM YOUR LOCAL MACHINE (not on the TPU). Runs indefinitely until killed.
#
# Usage:
#   bash scripts/tpu/babysit_tpu.sh v6e-small     # babysit the v6e-small instance
#   bash scripts/tpu/babysit_tpu.sh v6e-longctx    # babysit the v6e-longctx instance
#   bash scripts/tpu/babysit_tpu.sh v5e             # babysit the v5e instance
#
# Environment variables (optional):
#   WANDB_API_KEY        - WandB API key (required for training)
#   POLL_INTERVAL        - seconds between state checks (default: 60)
#   MAX_RECREATE_WAIT    - max seconds to wait for TPU creation (default: 600)

set -euo pipefail

PROFILE="${1:?Usage: $0 <profile>}"
NANOCHAT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PROJECT="alpine-aspect-459819-m4"
GCS_BUCKET="gs://nanochat-training-data-2026"
POLL_INTERVAL="${POLL_INTERVAL:-60}"
MAX_RECREATE_WAIT="${MAX_RECREATE_WAIT:-600}"

# ─── Profile configuration ───────────────────────────────────────────────────
# Each profile defines: TPU name, zone, accelerator, runtime, training args
case "$PROFILE" in
    v6e-small)
        TPU_NAME="nanochat-v6e-small"
        ZONE="asia-northeast1-b"
        ACCEL_TYPE="v6e-4"
        RUNTIME="v2-alpha-tpuv6e"
        SPOT_FLAG="--spot"
        VENV="venv311"
        GCS_CKPT_PATH="checkpoints/v6e/base_checkpoints/d16"
        TRAIN_ARGS=(
            --depth=16
            --num_iterations=50000
            --device_batch_size=8
            --max_seq_len=2048
            --kernel=current
            --no_compile
            --xla_flash_attn
            --data_dir=/home/dave/data/parquet
            --streaming_data
            --run=dummy
            --core_metric_every=5000
            --save_every=5000
            --sample_every=5000
        )
        LOG_FILE="train_v6e_flash.log"
        ;;
    v6e-longctx)
        TPU_NAME="nanochat-v6e-longctx"
        ZONE="asia-northeast1-b"
        ACCEL_TYPE="v6e-4"
        RUNTIME="v2-alpha-tpuv6e"
        SPOT_FLAG="--spot"
        VENV="venv311"
        GCS_CKPT_PATH="checkpoints/v6e-longctx/base_checkpoints/d16"
        TRAIN_ARGS=(
            --depth=16
            --num_iterations=50000
            --device_batch_size=2
            --max_seq_len=16384
            --kernel=current
            --no_compile
            --xla_flash_attn
            --data_dir=/home/dave/data/parquet
            --streaming_data
            --run=dummy
            --core_metric_every=2500
            --save_every=2500
            --sample_every=2500
        )
        LOG_FILE="train_v6e_longctx.log"
        ;;
    v5e)
        TPU_NAME="nanochat-tpu"
        ZONE="us-west4-a"
        ACCEL_TYPE="v5litepod-8"
        RUNTIME="v2-alpha-tpuv5-lite"
        SPOT_FLAG="--spot"
        VENV="venv"
        GCS_CKPT_PATH="checkpoints/v5e/base_checkpoints/d16"
        TRAIN_ARGS=(
            --depth=16
            --num_iterations=50000
            --device_batch_size=8
            --max_seq_len=1024
            --kernel=current
            --no_compile
            --data_dir=/home/dave/data/parquet
            --streaming_data
            --run=dummy
            --core_metric_every=5000
            --save_every=5000
            --sample_every=5000
        )
        LOG_FILE="train_v5e.log"
        ;;
    v6e-mhc)
        TPU_NAME="nanochat-v6e-mhc"
        ZONE="asia-northeast1-b"
        ACCEL_TYPE="v6e-4"
        RUNTIME="v2-alpha-tpuv6e"
        SPOT_FLAG="--spot"
        VENV="venv"
        GCS_CKPT_PATH="checkpoints/v6e-mhc/base_checkpoints/d16"
        TRAIN_ARGS=(
            --depth=16
            --num_iterations=50000
            --device_batch_size=1
            --max_seq_len=16384
            --kernel=current
            --no_compile
            --xla_flash_attn
            --data_dir=/home/dave/data/parquet
            --streaming_data
            --run=dummy
            --mhc
            --core_metric_every=2500
            --save_every=2500
            --sample_every=2500
        )
        LOG_FILE="train_v6e_mhc.log"
        ;;
    v6e-engram)
        TPU_NAME="nanochat-v6e-engram"
        ZONE="asia-northeast1-b"
        ACCEL_TYPE="v6e-4"
        RUNTIME="v2-alpha-tpuv6e"
        SPOT_FLAG="--spot"
        VENV="venv"
        GCS_CKPT_PATH="checkpoints/v6e-engram/base_checkpoints/d16"
        TRAIN_ARGS=(
            --depth=16
            --num_iterations=50000
            --device_batch_size=2
            --max_seq_len=16384
            --kernel=current
            --no_compile
            --xla_flash_attn
            --data_dir=/home/dave/data/parquet
            --streaming_data
            --run=dummy
            --engram
            --engram_layers=2,6
            --core_metric_every=2500
            --save_every=2500
            --sample_every=2500
        )
        LOG_FILE="train_v6e_engram.log"
        ;;
    v6e-mhc-engram)
        TPU_NAME="nanochat-v6e-mhc-engram"
        ZONE="europe-west4-a"
        ACCEL_TYPE="v6e-4"
        RUNTIME="v2-alpha-tpuv6e"
        SPOT_FLAG=""
        VENV="venv"
        GCS_CKPT_PATH="checkpoints/v6e-mhc-engram/base_checkpoints/d16"
        TRAIN_ARGS=(
            --depth=16
            --num_iterations=50000
            --device_batch_size=2
            --max_seq_len=16384
            --kernel=current
            --no_compile
            --xla_flash_attn
            --data_dir=/home/dave/data/parquet
            --streaming_data
            --run=dummy
            --mhc
            --engram
            --engram_layers=2,6
            --core_metric_every=2500
            --save_every=2500
            --sample_every=2500
        )
        LOG_FILE="train_v6e_mhc_engram.log"
        ;;
    *)
        echo "Unknown profile: $PROFILE"
        echo ""
        echo "Available profiles:"
        echo "  v6e-small      - nanochat-v6e-small (v6e-4, spot, seq_len=2048)"
        echo "  v6e-longctx    - nanochat-v6e-longctx (v6e-4, spot, seq_len=16384)"
        echo "  v5e            - nanochat-tpu (v5litepod-8, spot, seq_len=1024)"
        echo "  v6e-mhc        - nanochat-v6e-mhc (v6e-4, spot, seq_len=16384, mHC)"
        echo "  v6e-engram     - nanochat-v6e-engram (v6e-4, spot, seq_len=16384, Engram)"
        echo "  v6e-mhc-engram - nanochat-v6e-mhc-engram (v6e-4, regular, seq_len=16384, mHC+Engram)"
        exit 1
        ;;
esac

# ─── Helper functions ────────────────────────────────────────────────────────

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

get_tpu_state() {
    gcloud compute tpus tpu-vm describe "$TPU_NAME" \
        --zone="$ZONE" --project="$PROJECT" \
        --format="value(state)" 2>/dev/null || echo "NOT_FOUND"
}

find_latest_gcs_checkpoint() {
    # Find the highest step number from model checkpoint files in GCS
    local gcs_dir="${GCS_BUCKET}/${GCS_CKPT_PATH}"
    local latest_step=-1
    # List model files and extract step numbers
    while IFS= read -r line; do
        if [[ "$line" =~ model_([0-9]+)\.pt ]]; then
            step=${BASH_REMATCH[1]}
            # Remove leading zeros for arithmetic
            step=$((10#$step))
            if (( step > latest_step )); then
                latest_step=$step
            fi
        fi
    done < <(gsutil ls "$gcs_dir/model_*.pt" 2>/dev/null || true)
    echo "$latest_step"
}

delete_tpu() {
    log "Deleting TPU $TPU_NAME..."
    gcloud compute tpus tpu-vm delete "$TPU_NAME" \
        --zone="$ZONE" --project="$PROJECT" --quiet 2>/dev/null || true
    log "Deleted"
}

create_tpu() {
    local pricing="regular"
    [ -n "$SPOT_FLAG" ] && pricing="spot"
    log "Creating TPU: $TPU_NAME ($ACCEL_TYPE) in $ZONE with $pricing pricing"
    gcloud compute tpus tpu-vm create "$TPU_NAME" \
        --zone="$ZONE" \
        --project="$PROJECT" \
        --accelerator-type="$ACCEL_TYPE" \
        --version="$RUNTIME" \
        $SPOT_FLAG

    # Wait for READY
    local elapsed=0
    while (( elapsed < MAX_RECREATE_WAIT )); do
        local state
        state=$(get_tpu_state)
        if [ "$state" = "READY" ]; then
            log "TPU is READY (took ${elapsed}s)"
            return 0
        fi
        log "  Waiting for READY (state=$state, ${elapsed}s/${MAX_RECREATE_WAIT}s)"
        sleep 10
        elapsed=$((elapsed + 10))
    done
    log "ERROR: TPU did not become READY within ${MAX_RECREATE_WAIT}s"
    return 1
}

deploy_code() {
    log "Uploading nanochat code to TPU..."
    local tarball="/tmp/nanochat_deploy_${PROFILE}.tar.gz"
    tar czf "$tarball" \
        -C "$(dirname "$NANOCHAT_DIR")" \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='.cache' \
        --exclude='*.log' \
        --exclude='reports' \
        --exclude='.venv' \
        --exclude='vertex_ai' \
        --exclude='wandb' \
        "$(basename "$NANOCHAT_DIR")"

    gcloud compute tpus tpu-vm scp "$tarball" \
        "$TPU_NAME":/tmp/nanochat_deploy.tar.gz \
        --zone="$ZONE" --project="$PROJECT"

    gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
        --zone="$ZONE" --project="$PROJECT" \
        --command="rm -rf ~/nanochat && tar xzf /tmp/nanochat_deploy.tar.gz -C ~ && rm /tmp/nanochat_deploy.tar.gz"

    rm -f "$tarball"
    log "Code deployed"
}

setup_env() {
    log "Setting up environment on TPU..."
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
    log "Environment ready"
}

download_data() {
    log "Ensuring training data is available..."
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
        --zone="$ZONE" --project="$PROJECT" \
        --command="
            set -e
            mkdir -p ~/data/parquet ~/data/tokenizer
            # Download tokenizer if missing
            if [ ! -f ~/data/tokenizer/tokenizer.json ]; then
                echo 'Downloading tokenizer...'
                gsutil -m cp -r ${GCS_BUCKET}/tokenizer/* ~/data/tokenizer/
            fi
            # Download parquet data if missing
            PARQUET_COUNT=\$(ls ~/data/parquet/*.parquet 2>/dev/null | wc -l)
            if [ \"\$PARQUET_COUNT\" -lt 10 ]; then
                echo 'Downloading training data...'
                gsutil -m cp ${GCS_BUCKET}/parquet/base_data_v3/*.parquet ~/data/parquet/
            fi
            # Streaming dataloader needs _COMPLETE sentinel to know dataset is finished
            touch ~/data/parquet/_COMPLETE
            echo \"Data ready: \$(ls ~/data/parquet/*.parquet | wc -l) parquet files\"
        "
    log "Data ready"
}

start_training() {
    local resume_step="$1"
    local resume_args=""
    if (( resume_step > 0 )); then
        # Use auto-detect mode (-2) which will find the latest checkpoint from local or GCS
        resume_args="--resume_from_step=-2"
        log "Resuming training (auto-detect from GCS, latest known: step $resume_step)"
    else
        log "Starting training from scratch"
    fi

    # Build the full training command
    local train_cmd="source ~/${VENV}/bin/activate"
    train_cmd+=" && [ -f ~/.tpu_env ] && source ~/.tpu_env || true"
    train_cmd+=" && cd ~/nanochat"
    train_cmd+=" && export NANOCHAT_BASE_DIR=/home/dave/data"
    train_cmd+=" && export WANDB_API_KEY=${WANDB_API_KEY:-}"
    train_cmd+=" && export XLA_NO_SPECIAL_SCALARS=1"
    train_cmd+=" && export NANOCHAT_GCS_CHECKPOINT_BUCKET=${GCS_BUCKET}/${GCS_CKPT_PATH%/base_checkpoints/d16}"
    train_cmd+=" && nohup python3 -u -m scripts.base_train"
    for arg in "${TRAIN_ARGS[@]}"; do
        train_cmd+=" $arg"
    done
    if [ -n "$resume_args" ]; then
        train_cmd+=" $resume_args"
    fi
    train_cmd+=" > ~/${LOG_FILE} 2>&1 &"
    train_cmd+=" echo \"Training PID: \$!\""

    gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
        --zone="$ZONE" --project="$PROJECT" \
        --command="$train_cmd"

    log "Training launched (log: ~/${LOG_FILE})"
}

is_training_running() {
    local result
    result=$(gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
        --zone="$ZONE" --project="$PROJECT" \
        --command="pgrep -f 'python3.*scripts[.]base_train' > /dev/null 2>&1 && echo 'BABYSIT_RUNNING' || echo 'BABYSIT_STOPPED'" 2>/dev/null || echo "BABYSIT_SSH_FAILED")
    # Extract our sentinel value from potentially noisy SSH output
    if echo "$result" | grep -q "BABYSIT_RUNNING"; then
        echo "RUNNING"
    elif echo "$result" | grep -q "BABYSIT_STOPPED"; then
        echo "STOPPED"
    else
        echo "SSH_FAILED"
    fi
}

# ─── Main babysit loop ───────────────────────────────────────────────────────

log "============================================"
log "Babysitting: $TPU_NAME ($PROFILE)"
log "  Zone: $ZONE"
log "  Accelerator: $ACCEL_TYPE"
log "  GCS checkpoints: ${GCS_BUCKET}/${GCS_CKPT_PATH}"
log "  Poll interval: ${POLL_INTERVAL}s"
log "============================================"

CONSECUTIVE_FAILURES=0
MAX_CONSECUTIVE_FAILURES=5

while true; do
    STATE=$(get_tpu_state)

    case "$STATE" in
        READY)
            # Check if training is still running
            TRAIN_STATUS=$(is_training_running)
            if [ "$TRAIN_STATUS" = "RUNNING" ]; then
                # All good, training is running
                CONSECUTIVE_FAILURES=0
                # Silently poll (log every 10 minutes = every 10 polls at 60s)
                if (( SECONDS % 600 < POLL_INTERVAL )); then
                    log "OK: Training running on $TPU_NAME"
                fi
            else
                log "WARNING: TPU is READY but training is not running"
                # Find latest checkpoint and resume
                LATEST_STEP=$(find_latest_gcs_checkpoint)
                log "Latest GCS checkpoint: step $LATEST_STEP"
                start_training "$LATEST_STEP"
                CONSECUTIVE_FAILURES=0
            fi
            ;;

        PREEMPTED|TERMINATED)
            log "TPU is $STATE — initiating recovery"
            CONSECUTIVE_FAILURES=0

            # Step 1: Delete the dead TPU
            delete_tpu

            # Step 2: Recreate
            if ! create_tpu; then
                log "Failed to create TPU, will retry next cycle"
                CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
                sleep "$POLL_INTERVAL"
                continue
            fi

            # Step 3: Deploy code and setup
            deploy_code
            setup_env
            download_data

            # Step 4: Find latest checkpoint and resume
            LATEST_STEP=$(find_latest_gcs_checkpoint)
            log "Latest GCS checkpoint: step $LATEST_STEP"

            # Step 5: Start training
            start_training "$LATEST_STEP"
            log "Recovery complete — training resumed from step $LATEST_STEP"
            ;;

        NOT_FOUND)
            log "TPU not found — creating from scratch"

            if ! create_tpu; then
                log "Failed to create TPU, will retry next cycle"
                CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
                sleep "$POLL_INTERVAL"
                continue
            fi

            deploy_code
            setup_env
            download_data

            LATEST_STEP=$(find_latest_gcs_checkpoint)
            log "Latest GCS checkpoint: step $LATEST_STEP"
            start_training "$LATEST_STEP"
            ;;

        CREATING|STARTING)
            log "TPU is $STATE, waiting..."
            ;;

        *)
            log "Unexpected state: $STATE"
            CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
            ;;
    esac

    if (( CONSECUTIVE_FAILURES >= MAX_CONSECUTIVE_FAILURES )); then
        log "ERROR: $CONSECUTIVE_FAILURES consecutive failures, exiting"
        log "Investigate manually:"
        log "  gcloud compute tpus tpu-vm describe $TPU_NAME --zone=$ZONE"
        exit 1
    fi

    sleep "$POLL_INTERVAL"
done
