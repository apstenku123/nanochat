#!/bin/bash
# Download CommitPack C++ subset from HuggingFace
# CommitPack: 4TB of git commits across 350 languages (OctoPack/BigCode)
# We download only the C++ subset
#
# Usage: bash scripts/data/download_commitpack.sh

set -euo pipefail

OUTPUT_DIR="$HOME/data/commitpack_cpp"
mkdir -p "$OUTPUT_DIR"

echo "=== Downloading CommitPack C++ from HuggingFace ==="

# Check if huggingface-cli is available
if ! command -v huggingface-cli &> /dev/null; then
    pip install --quiet huggingface_hub[cli]
fi

# Download C++ subset of CommitPack
# The dataset is organized by language: data/c++/
echo "Downloading C++ commits..."
huggingface-cli download bigcode/commitpack \
    --repo-type dataset \
    --include "data/c++/*" \
    --local-dir "$OUTPUT_DIR" \
    --local-dir-use-symlinks False

echo ""
echo "=== Done downloading ==="
ls -lh "$OUTPUT_DIR"/data/c++/ 2>/dev/null || ls -lh "$OUTPUT_DIR"
