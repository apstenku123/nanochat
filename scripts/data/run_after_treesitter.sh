#!/bin/bash
# Wait for tree-sitter pipeline to finish, then run clang pipeline
# Also uploads results to GCS when done
set -euo pipefail

TREESITTER_PID="${1:-473612}"
LOG_FILE="$HOME/clang_pipeline.log"

echo "=== Waiting for tree-sitter pipeline (PID $TREESITTER_PID) ==="
echo "Start time: $(date)"

# Wait for tree-sitter process to finish
while kill -0 "$TREESITTER_PID" 2>/dev/null; do
    # Print progress every 5 minutes
    RCHAR=$(cat /proc/$TREESITTER_PID/io 2>/dev/null | grep rchar | awk '{print $2}')
    RSS=$(cat /proc/$TREESITTER_PID/status 2>/dev/null | grep VmRSS | awk '{print $2}')
    echo "  $(date +%H:%M) - tree-sitter: read $(echo "scale=1; $RCHAR/1073741824" | bc)GB, RSS ${RSS}KB"
    sleep 300
done

echo "Tree-sitter pipeline finished at $(date)"
echo ""

# Check tree-sitter output
echo "=== Tree-sitter output ==="
ls -lh ~/data/cpp_crossfile_16k.jsonl 2>/dev/null
wc -l ~/data/cpp_crossfile_16k.jsonl 2>/dev/null
ls -lh ~/data/parquet/cpp_crossfile_16k/ 2>/dev/null

# Upload tree-sitter results to GCS
echo ""
echo "=== Uploading tree-sitter results to GCS ==="
gsutil -m cp -r ~/data/parquet/cpp_crossfile_16k/ gs://nanochat-training-data-2026/data/ 2>&1 || echo "GCS upload failed (tree-sitter)"

# Now run clang pipeline
echo ""
echo "=== Starting Clang Pipeline ==="
bash ~/nanochat/scripts/data/run_clang_pipeline.sh ~/data/cpp_raw ~/data 48

# Upload clang results to GCS
echo ""
echo "=== Uploading clang results to GCS ==="
gsutil -m cp -r ~/data/parquet/cpp_clang_crossfile_16k/ gs://nanochat-training-data-2026/data/ 2>&1 || echo "GCS upload failed (clang)"

echo ""
echo "=== All pipelines complete ==="
echo "End time: $(date)"
