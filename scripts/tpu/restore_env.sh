#!/bin/bash
# Restore TPU environment from GCS backup after preemption/crash
# Run ON the TPU via SSH:
#   gcloud compute tpus tpu-vm ssh nanochat-v6e --zone=us-east5-b --command="bash ~/nanochat/scripts/tpu/restore_env.sh"
#   gcloud compute tpus tpu-vm ssh nanochat-tpu --zone=us-west4-a --command="bash ~/nanochat/scripts/tpu/restore_env.sh"
#
# Restore priority:
#   1. Try restoring venv from GCS tarball (fastest, ~2 min)
#   2. Fall back to full setup script (slower, ~10 min)
#
# Also restores:
#   - Environment config (.tpu_env)
#   - Training data from GCS (if missing)
#   - Tokenizer files from GCS (if missing)

set -e

BACKUP_BUCKET="gs://nanochat-training-data-2026/tpu-backups"
TPU_NAME=${TPU_NAME:-$(hostname)}
RESTORE_START=$(date +%s)

echo "=== Restoring TPU environment for: $TPU_NAME ==="
echo "Timestamp: $(date)"
echo ""

# -------------------------------------------------------------------
# Step 1: Restore Python venv
# -------------------------------------------------------------------
VENV_BACKUP="${BACKUP_BUCKET}/${TPU_NAME}_latest_venv.tar.gz"
VENV_RESTORED=0

if gsutil ls "$VENV_BACKUP" &>/dev/null; then
    echo "Found venv backup at $VENV_BACKUP"
    BACKUP_SIZE=$(gsutil du -s "$VENV_BACKUP" | awk '{print $1}')
    echo "Backup size: $(numfmt --to=iec $BACKUP_SIZE 2>/dev/null || echo "${BACKUP_SIZE} bytes")"

    echo "Downloading..."
    gsutil -o GSUtil:parallel_composite_upload_threshold=150M \
        cp "$VENV_BACKUP" /tmp/venv_restore.tar.gz

    echo "Extracting..."
    # Remove old venv if it exists (could be corrupted from preemption)
    rm -rf "$HOME/venv" "$HOME/venv_requirements.txt"
    tar xzf /tmp/venv_restore.tar.gz -C "$HOME"
    rm /tmp/venv_restore.tar.gz

    # Activate and verify
    source "$HOME/venv/bin/activate"
    echo "Verifying restored venv..."
    if python3 -c "import torch; import torch_xla; import torch_xla.core.xla_model as xm; dev = xm.xla_device(); print(f'torch={torch.__version__}, xla={torch_xla.__version__}, device={dev}')" 2>/dev/null; then
        echo "Venv restored and verified successfully"
        VENV_RESTORED=1
    else
        echo "WARNING: Venv restored but verification failed, will attempt pip install"
        VENV_RESTORED=0
    fi
else
    echo "No venv backup found at $VENV_BACKUP"
fi

# Fall back to full setup if restore failed
if [ "$VENV_RESTORED" -eq 0 ]; then
    echo ""
    echo "=== Falling back to full environment setup ==="

    # Detect TPU type from hostname or metadata
    TPU_TYPE=$(curl -fs -H 'Metadata-Flavor: Google' \
        http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type 2>/dev/null || echo "unknown")
    echo "Detected accelerator type: $TPU_TYPE"

    if [[ "$TPU_TYPE" == *"v5"* ]] && [ -f "$HOME/nanochat/scripts/tpu/setup_v5e.sh" ]; then
        echo "Running v5e setup..."
        bash "$HOME/nanochat/scripts/tpu/setup_v5e.sh"
    elif [ -f "$HOME/nanochat/scripts/tpu/setup_v6e.sh" ]; then
        echo "Running v6e setup..."
        bash "$HOME/nanochat/scripts/tpu/setup_v6e.sh"
    else
        echo "ERROR: No setup script found at ~/nanochat/scripts/tpu/"
        echo "Upload nanochat code first, then retry."
        exit 1
    fi
    source "$HOME/venv/bin/activate"
fi

echo ""

# -------------------------------------------------------------------
# Step 2: Restore config files
# -------------------------------------------------------------------
CONFIG_BACKUP="${BACKUP_BUCKET}/${TPU_NAME}_latest_config.tar.gz"
if gsutil ls "$CONFIG_BACKUP" &>/dev/null; then
    echo "Restoring config files..."
    gsutil cp "$CONFIG_BACKUP" /tmp/config_restore.tar.gz
    tar xzf /tmp/config_restore.tar.gz -C "$HOME"
    rm /tmp/config_restore.tar.gz
    echo "Config restored"
else
    echo "No config backup found, using defaults"
fi

# Source environment
if [ -f "$HOME/.tpu_env" ]; then
    source "$HOME/.tpu_env"
    echo "Loaded .tpu_env"
fi

echo ""

# -------------------------------------------------------------------
# Step 3: Ensure training data exists
# -------------------------------------------------------------------
DATA_DIR="$HOME/data"
echo "=== Checking training data ==="

# Tokenizer
if [ ! -d "$DATA_DIR/cpp_tokenizer_65k" ] || [ -z "$(ls $DATA_DIR/cpp_tokenizer_65k/ 2>/dev/null)" ]; then
    echo "Downloading tokenizer from GCS..."
    mkdir -p "$DATA_DIR/cpp_tokenizer_65k"
    gsutil -m cp -r gs://nanochat-training-data-2026/tokenizer_65k/* "$DATA_DIR/cpp_tokenizer_65k/"
    echo "Tokenizer downloaded"
else
    echo "Tokenizer already present ($(ls $DATA_DIR/cpp_tokenizer_65k/ | wc -l) files)"
fi

# Training parquet files
if [ ! -d "$DATA_DIR/base_data" ] || [ -z "$(ls $DATA_DIR/base_data/*.parquet 2>/dev/null)" ]; then
    echo "Downloading training data from GCS..."
    mkdir -p "$DATA_DIR/base_data"
    gsutil -m cp gs://nanochat-training-data-2026/parquet/base_data_v3/*.parquet "$DATA_DIR/base_data/"
    echo "Training data downloaded"
else
    PARQUET_COUNT=$(ls $DATA_DIR/base_data/*.parquet 2>/dev/null | wc -l)
    echo "Training data already present ($PARQUET_COUNT parquet files)"
fi

echo ""

# -------------------------------------------------------------------
# Step 4: Summary
# -------------------------------------------------------------------
RESTORE_END=$(date +%s)
ELAPSED=$((RESTORE_END - RESTORE_START))

echo "=== Restore complete in ${ELAPSED}s ==="
echo ""
echo "Environment:"
python3 -c "
import torch
print(f'  torch: {torch.__version__}')
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    print(f'  torch_xla: {torch_xla.__version__}')
    print(f'  device: {xm.xla_device()}')
    print(f'  num_devices: {torch_xla.device_count()}')
except Exception as e:
    print(f'  torch_xla: error - {e}')
" 2>/dev/null || echo "  (verification skipped)"

echo ""
echo "Data files:"
ls -lh "$DATA_DIR/base_data/"*.parquet 2>/dev/null | tail -3 || echo "  No parquet files"
ls -lh "$DATA_DIR/cpp_tokenizer_65k/" 2>/dev/null | tail -3 || echo "  No tokenizer files"

echo ""
echo "To start training:"
echo "  source ~/venv/bin/activate && source ~/.tpu_env"
echo "  cd ~/nanochat && python3 -u -m scripts.base_train --depth=16 ..."
