#!/bin/bash
# Setup GCS bucket structure for nanochat TPU training
#
# This script:
#   1. Uploads training data to GCS
#   2. Uploads tokenizer files
#   3. Creates checkpoint directories

set -e

PROJECT_ID="alpine-aspect-459819-m4"
BUCKET_NAME="nanochat-training-data-2026"
GCS_BUCKET="gs://${BUCKET_NAME}"

echo "=============================================="
echo "Setting up GCS for nanochat TPU training"
echo "=============================================="
echo "Bucket: ${GCS_BUCKET}"
echo "=============================================="

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

echo ""
echo "Step 1: Verify bucket exists..."
if gsutil ls -b ${GCS_BUCKET} &>/dev/null; then
    echo "Bucket exists: ${GCS_BUCKET}"
else
    echo "Creating bucket: ${GCS_BUCKET}"
    gsutil mb -p ${PROJECT_ID} -l us-central1 ${GCS_BUCKET}
fi

echo ""
echo "Step 2: Upload tokenizer files..."
TOKENIZER_DIR="data/tokenizer"
if [ -d "${TOKENIZER_DIR}" ]; then
    echo "Uploading from ${TOKENIZER_DIR}..."
    gsutil -m cp -r ${TOKENIZER_DIR}/* ${GCS_BUCKET}/tokenizer/
else
    echo "Warning: Local tokenizer directory not found at ${TOKENIZER_DIR}"
    echo "You may need to upload tokenizer files manually."
fi

echo ""
echo "Step 3: Upload training data..."
DATA_DIR="data"
if [ -f "${DATA_DIR}/cpp_clean_v4.jsonl" ]; then
    echo "Uploading cpp_clean_v4.jsonl..."
    gsutil cp ${DATA_DIR}/cpp_clean_v4.jsonl ${GCS_BUCKET}/data/
fi
if [ -f "${DATA_DIR}/docstring_pairs_full.jsonl" ]; then
    echo "Uploading docstring_pairs_full.jsonl..."
    gsutil cp ${DATA_DIR}/docstring_pairs_full.jsonl ${GCS_BUCKET}/data/
fi

echo ""
echo "Step 4: Create checkpoint directory placeholders..."
# Create empty placeholder to establish directory structure
echo "placeholder" | gsutil cp - ${GCS_BUCKET}/checkpoints/.keep

echo ""
echo "Step 5: Verify bucket contents..."
echo ""
echo "Bucket structure:"
gsutil ls -r ${GCS_BUCKET}/ | head -30

echo ""
echo "=============================================="
echo "GCS setup complete!"
echo "=============================================="
echo ""
echo "Bucket contents:"
echo "  ${GCS_BUCKET}/data/       - Training data"
echo "  ${GCS_BUCKET}/tokenizer/  - Tokenizer files"
echo "  ${GCS_BUCKET}/checkpoints/- Training checkpoints"
echo "  ${GCS_BUCKET}/eval/       - Evaluation data"
