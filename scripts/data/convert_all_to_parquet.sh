#!/bin/bash
# Convert all JSONL datasets to parquet shards for streaming training.
# Usage: bash convert_all_to_parquet.sh

DATA_DIR="/mnt/nvme/nanochat_data"
PARQUET_BASE="/mnt/nvme/nanochat_data/parquet"
SCRIPT="$HOME/nanochat/scripts/data/batch_jsonl_to_parquet.py"

mkdir -p "$PARQUET_BASE"

convert() {
    local input="$1"
    local name="$2"
    local rows="${3:-50000}"
    local output="$PARQUET_BASE/$name"

    if [ -f "$output/_COMPLETE" ]; then
        echo "SKIP (already done): $name"
        return
    fi

    if [ ! -f "$input" ]; then
        echo "SKIP (not found): $input"
        return
    fi

    echo ""
    echo "=== Converting: $name ==="
    python3 "$SCRIPT" --input "$input" --parquet_dir "$output" --rows_per_file "$rows"
}

echo "=== Converting all JSONL datasets to parquet ==="
echo "Output base: $PARQUET_BASE"
echo ""

# Tree-sitter compilable chunks (source code training)
convert "$DATA_DIR/cpp_compilable_16k_v2.jsonl" "cpp_compilable_16k_v2" 50000
convert "$DATA_DIR/cpp_compilable_64k_v2.jsonl" "cpp_compilable_64k_v2" 20000
convert "$DATA_DIR/gpu_npu_compilable_16k.jsonl" "gpu_npu_compilable_16k" 50000
convert "$DATA_DIR/gpu_npu_compilable_64k.jsonl" "gpu_npu_compilable_64k" 20000

# Commit chains (git history pre->post code)
convert "$DATA_DIR/commitpack_chains.jsonl" "commitpack_chains" 50000
convert "$DATA_DIR/git_history_chains.jsonl" "git_history_chains" 50000
convert "$DATA_DIR/gpu_npu_chains.jsonl" "gpu_npu_chains" 50000

# Microsoft source
convert "$DATA_DIR/ms_src_cpp.jsonl" "ms_src_cpp" 50000

# Clang semantic (will be converted after clang indexer finishes)
convert "$DATA_DIR/gpu_npu_clang_16k.jsonl" "gpu_npu_clang_16k" 50000

echo ""
echo "=== All conversions complete ==="
echo ""
echo "Parquet directories:"
for d in "$PARQUET_BASE"/*/; do
    name=$(basename "$d")
    if [ -f "$d/_COMPLETE" ]; then
        shards=$(ls "$d"/shard_*.parquet 2>/dev/null | wc -l)
        size=$(du -sh "$d" | cut -f1)
        echo "  $name: $shards shards, $size [COMPLETE]"
    else
        echo "  $name: [INCOMPLETE]"
    fi
done
