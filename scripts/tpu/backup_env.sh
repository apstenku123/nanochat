#!/bin/bash
# Backup TPU environment to GCS for fast restoration after preemption
# Run ON the TPU via SSH:
#   gcloud compute tpus tpu-vm ssh nanochat-v6e --zone=us-east5-b --command="bash ~/nanochat/scripts/tpu/backup_env.sh"
#   gcloud compute tpus tpu-vm ssh nanochat-tpu --zone=us-west4-a --command="bash ~/nanochat/scripts/tpu/backup_env.sh"
#
# What gets backed up:
#   - Python venv (tarball + frozen requirements)
#   - Environment config files (.tpu_env)
#
# Training data is NOT backed up (already lives in GCS, restored on demand)
# Checkpoints are NOT backed up (already saved to GCS by training script)

set -e

BACKUP_BUCKET="gs://nanochat-training-data-2026/tpu-backups"
TPU_NAME=${TPU_NAME:-$(hostname)}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="${TPU_NAME}_${TIMESTAMP}"

echo "=== Backing up TPU environment: $BACKUP_NAME ==="
echo "Timestamp: $(date)"
echo "Hostname: $TPU_NAME"
echo ""

# Backup the venv (frozen pip list + tarball)
if [ -d "$HOME/venv" ]; then
    source "$HOME/venv/bin/activate"
    pip freeze > "$HOME/venv_requirements.txt"

    # Show what we're backing up
    VENV_SIZE=$(du -sh "$HOME/venv" | cut -f1)
    echo "Venv size: $VENV_SIZE"
    echo "Packages: $(wc -l < "$HOME/venv_requirements.txt") installed"

    echo "Creating venv tarball..."
    tar czf "/tmp/${BACKUP_NAME}_venv.tar.gz" -C "$HOME" venv venv_requirements.txt

    TARBALL_SIZE=$(du -sh "/tmp/${BACKUP_NAME}_venv.tar.gz" | cut -f1)
    echo "Tarball size: $TARBALL_SIZE"

    echo "Uploading to GCS..."
    gsutil -o GSUtil:parallel_composite_upload_threshold=150M \
        cp "/tmp/${BACKUP_NAME}_venv.tar.gz" "${BACKUP_BUCKET}/${BACKUP_NAME}_venv.tar.gz"
    # Also save as "latest" for quick restore
    gsutil cp "${BACKUP_BUCKET}/${BACKUP_NAME}_venv.tar.gz" "${BACKUP_BUCKET}/${TPU_NAME}_latest_venv.tar.gz"
    rm "/tmp/${BACKUP_NAME}_venv.tar.gz"
    echo "Venv backed up to ${BACKUP_BUCKET}/${BACKUP_NAME}_venv.tar.gz"
    echo "Latest alias: ${BACKUP_BUCKET}/${TPU_NAME}_latest_venv.tar.gz"
else
    echo "WARNING: No venv found at $HOME/venv, skipping venv backup"
fi

echo ""

# Backup env config files
CONFIG_FILES=""
[ -f "$HOME/.tpu_env" ] && CONFIG_FILES="$CONFIG_FILES .tpu_env"
[ -f "$HOME/.bashrc" ] && CONFIG_FILES="$CONFIG_FILES .bashrc"

if [ -n "$CONFIG_FILES" ]; then
    echo "Backing up config files:$CONFIG_FILES"
    tar czf "/tmp/${BACKUP_NAME}_config.tar.gz" -C "$HOME" $CONFIG_FILES
    gsutil cp "/tmp/${BACKUP_NAME}_config.tar.gz" "${BACKUP_BUCKET}/${BACKUP_NAME}_config.tar.gz"
    gsutil cp "${BACKUP_BUCKET}/${BACKUP_NAME}_config.tar.gz" "${BACKUP_BUCKET}/${TPU_NAME}_latest_config.tar.gz"
    rm "/tmp/${BACKUP_NAME}_config.tar.gz"
    echo "Config backed up"
else
    echo "No config files found to backup"
fi

echo ""

# List recent backups for this TPU
echo "=== Recent backups for $TPU_NAME ==="
gsutil ls -l "${BACKUP_BUCKET}/${TPU_NAME}_*" 2>/dev/null | tail -10 || echo "No previous backups found"

echo ""
echo "=== Backup complete ==="
echo "To restore on a fresh TPU:"
echo "  bash ~/nanochat/scripts/tpu/restore_env.sh"
