#!/bin/bash
# TPU v6e Training Runner Script
# Run this locally to orchestrate the full training on TPU VM
#
# Usage:
#   ./run_tpu_training.sh [--stage all|base|sft|gspo|eval] [--depth 16] [--base-iters 5000]
#
# Prerequisites:
#   - gcloud CLI configured
#   - TPU VM 'nanochat-v6e' created in us-east5-b
#   - GCS bucket gs://nanochat-training-data-2026 with data

set -e

# Default configuration
TPU_NAME="${TPU_NAME:-nanochat-v6e}"
TPU_ZONE="${TPU_ZONE:-us-east5-b}"
GCS_BUCKET="${GCS_BUCKET:-gs://nanochat-training-data-2026}"
STAGE="${STAGE:-all}"
DEPTH="${DEPTH:-16}"
BASE_ITERS="${BASE_ITERS:-5000}"
SFT_ITERS="${SFT_ITERS:-2000}"
GSPO_ITERS="${GSPO_ITERS:-500}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --depth)
            DEPTH="$2"
            shift 2
            ;;
        --base-iters)
            BASE_ITERS="$2"
            shift 2
            ;;
        --sft-iters)
            SFT_ITERS="$2"
            shift 2
            ;;
        --gspo-iters)
            GSPO_ITERS="$2"
            shift 2
            ;;
        --tpu-name)
            TPU_NAME="$2"
            shift 2
            ;;
        --zone)
            TPU_ZONE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPORTS_DIR="$PROJECT_DIR/reports/v6e"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$REPORTS_DIR/training_${TIMESTAMP}.log"

mkdir -p "$REPORTS_DIR"

echo "============================================================"
echo "nanochat TPU v6e Training Pipeline"
echo "============================================================"
echo "TPU Name: $TPU_NAME"
echo "TPU Zone: $TPU_ZONE"
echo "GCS Bucket: $GCS_BUCKET"
echo "Stage: $STAGE"
echo "Model Depth: $DEPTH"
echo "Base Iterations: $BASE_ITERS"
echo "SFT Iterations: $SFT_ITERS"
echo "GSPO Iterations: $GSPO_ITERS"
echo "Log File: $LOG_FILE"
echo "============================================================"

# Check TPU status
echo ""
echo "Checking TPU status..."
gcloud compute tpus tpu-vm list --zone=$TPU_ZONE --filter="name=$TPU_NAME" 2>&1 | tee -a "$LOG_FILE"

# Step 1: Copy code to TPU VM
echo ""
echo "Step 1: Copying code to TPU VM..."
echo "Syncing nanochat directory..."

# Create a minimal package to upload (exclude large data files)
TEMP_DIR=$(mktemp -d)
cp -r "$PROJECT_DIR/nanochat" "$TEMP_DIR/"
cp -r "$PROJECT_DIR/scripts" "$TEMP_DIR/"
cp "$PROJECT_DIR/pyproject.toml" "$TEMP_DIR/"

echo "Uploading to TPU VM..."
gcloud compute tpus tpu-vm scp --recurse "$TEMP_DIR/nanochat" ${TPU_NAME}:~/nanochat/ --zone=$TPU_ZONE 2>&1 | tee -a "$LOG_FILE"
gcloud compute tpus tpu-vm scp --recurse "$TEMP_DIR/scripts" ${TPU_NAME}:~/nanochat/ --zone=$TPU_ZONE 2>&1 | tee -a "$LOG_FILE"
gcloud compute tpus tpu-vm scp "$TEMP_DIR/pyproject.toml" ${TPU_NAME}:~/nanochat/ --zone=$TPU_ZONE 2>&1 | tee -a "$LOG_FILE"

rm -rf "$TEMP_DIR"

# Step 2: Install dependencies on TPU VM
echo ""
echo "Step 2: Installing dependencies on TPU VM..."
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$TPU_ZONE --command="
set -e
cd ~/nanochat

# Install system dependencies
sudo apt-get update && sudo apt-get install -y python3-pip g++ 2>/dev/null || true

# Install Python dependencies
pip3 install --upgrade pip
pip3 install torch torch_xla google-cloud-storage pyarrow transformers tokenizers tqdm filelock 2>&1 | tail -20
" 2>&1 | tee -a "$LOG_FILE"

# Step 3: Run training
echo ""
echo "Step 3: Starting training on TPU VM..."
echo "This will take several hours. Logs are being saved to: $LOG_FILE"
echo ""

START_TIME=$(date +%s)

gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$TPU_ZONE --command="
set -e
cd ~/nanochat

# Set TPU environment
export PJRT_DEVICE=TPU
export XLA_USE_SPMD=1

echo 'Starting training...'
python3 scripts/tpu/train_v6e.py \\
    --gcs_bucket=$GCS_BUCKET \\
    --stage=$STAGE \\
    --depth=$DEPTH \\
    --base_iterations=$BASE_ITERS \\
    --sft_iterations=$SFT_ITERS \\
    --gspo_iterations=$GSPO_ITERS \\
    --batch_size=16 \\
    --max_seq_len=1024 \\
    --save_every=500 \\
    --max_eval_samples=50 \\
    2>&1
" 2>&1 | tee -a "$LOG_FILE"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "============================================================"
echo "Training Complete!"
echo "============================================================"
echo "Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Log file: $LOG_FILE"

# Step 4: Download results
echo ""
echo "Step 4: Downloading results..."
gcloud compute tpus tpu-vm scp ${TPU_NAME}:/tmp/training_report.json "$REPORTS_DIR/training_report_${TIMESTAMP}.json" --zone=$TPU_ZONE 2>&1 || echo "Could not download report"

# Calculate cost estimate
TOTAL_HOURS=$(echo "scale=2; $DURATION / 3600" | bc)
SPOT_RATE="1.70"
ESTIMATED_COST=$(echo "scale=2; $TOTAL_HOURS * $SPOT_RATE" | bc)

echo ""
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
echo "Training Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s ($TOTAL_HOURS hours)"
echo "Estimated Cost (spot): \$${ESTIMATED_COST}"
echo "Log File: $LOG_FILE"
echo "Report: $REPORTS_DIR/training_report_${TIMESTAMP}.json"
echo "============================================================"

# Display final report if available
if [ -f "$REPORTS_DIR/training_report_${TIMESTAMP}.json" ]; then
    echo ""
    echo "Training Report:"
    cat "$REPORTS_DIR/training_report_${TIMESTAMP}.json"
fi
