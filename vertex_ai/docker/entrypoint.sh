#!/bin/bash
# Entrypoint script for Vertex AI GPU training
# Downloads tokenizer and training data from GCS before starting training

set -ex

echo "=========================================="
echo "NANOCHAT ENTRYPOINT STARTING"
echo "=========================================="

BASE_DIR="${NANOCHAT_BASE_DIR:-/app/data}"

# ========================================
# DOWNLOAD TOKENIZER
# ========================================
TOKENIZER_DIR="$BASE_DIR/tokenizer"
echo "Creating tokenizer directory: $TOKENIZER_DIR"
mkdir -p "$TOKENIZER_DIR"

echo "Downloading 65K C++ tokenizer from GCS bucket: nanochat-training-data-2026"
gsutil -m cp "gs://nanochat-training-data-2026/tokenizer_65k/*" "$TOKENIZER_DIR/"

# Verify tokenizer files exist (need either json OR pkl)
if [ ! -f "$TOKENIZER_DIR/tokenizer.json" ] && [ ! -f "$TOKENIZER_DIR/tokenizer.pkl" ]; then
    echo "ERROR: Neither tokenizer.json nor tokenizer.pkl found in $TOKENIZER_DIR"
    ls -la "$TOKENIZER_DIR/" || true
    exit 1
fi

echo "Tokenizer files verified successfully:"
ls -la "$TOKENIZER_DIR/"

# ========================================
# DOWNLOAD TRAINING DATA (parquet files)
# ========================================
DATA_DIR="$BASE_DIR/base_data"
echo "Creating data directory: $DATA_DIR"
mkdir -p "$DATA_DIR"

echo "Downloading training data from GCS bucket: nanochat-training-data-2026"
echo "This may take several minutes (approx 11 GB)..."

# Download all parquet files from the training data
gsutil -m cp "gs://nanochat-training-data-2026/parquet/base_data_v3/*.parquet" "$DATA_DIR/"

# Count downloaded files
NUM_PARQUET=$(ls -1 "$DATA_DIR"/*.parquet 2>/dev/null | wc -l)
if [ "$NUM_PARQUET" -eq 0 ]; then
    echo "ERROR: No parquet files downloaded to $DATA_DIR"
    ls -la "$DATA_DIR/" || true
    exit 1
fi

echo "Training data downloaded successfully: $NUM_PARQUET parquet files"
ls -la "$DATA_DIR/" | head -20

echo "=========================================="
echo "STARTING TRAINING"
echo "=========================================="

# Run the training script with all arguments passed to the container
exec python3 -m scripts.base_train "$@"
