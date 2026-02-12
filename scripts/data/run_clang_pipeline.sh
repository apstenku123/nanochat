#!/bin/bash
# Run clang-based cross-file dependency indexer on all downloaded C++ repos
# Processes projects in parallel, generates 16K-token training documents
set -euo pipefail

DATA_DIR="${1:-/home/dave/data/cpp_raw}"
OUTPUT_DIR="${2:-/home/dave/data}"
WORKERS="${3:-48}"
MAX_TOKENS=16384
MAX_DEP_DEPTH=5

INDEXER="$HOME/nanochat/tools/clang_indexer/index_project.py"
PYTHON="$HOME/venv/bin/python3"
PARQUET_SCRIPT="$HOME/nanochat/scripts/data/stream_jsonl_to_parquet.py"

echo "=== Clang Cross-File Dependency Pipeline ==="
echo "Data dir: $DATA_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Workers per project: $WORKERS"
echo "Max tokens: $MAX_TOKENS"
echo "Start time: $(date)"

# Check prerequisites
if [ ! -f "$INDEXER" ]; then
    echo "ERROR: Indexer not found at $INDEXER"
    exit 1
fi
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found at $DATA_DIR"
    exit 1
fi

# Try to generate compile_commands.json for CMake projects
generate_compile_db() {
    local project_dir="$1"
    local project_name=$(basename "$project_dir")

    # Check if CMakeLists.txt exists
    if [ -f "$project_dir/CMakeLists.txt" ]; then
        local build_dir="$project_dir/build_cc"
        if [ ! -f "$project_dir/compile_commands.json" ]; then
            echo "  Generating compile_commands.json for $project_name..."
            mkdir -p "$build_dir"
            cmake -S "$project_dir" -B "$build_dir" \
                -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
                -DCMAKE_BUILD_TYPE=Release \
                > /dev/null 2>&1 && \
                cp "$build_dir/compile_commands.json" "$project_dir/" 2>/dev/null && \
                echo "  compile_commands.json generated for $project_name" || \
                echo "  WARN: cmake failed for $project_name, using fallback mode"
            rm -rf "$build_dir"
        fi
    fi
}

# List all project directories
PROJECTS=()
for d in "$DATA_DIR"/*/; do
    [ -d "$d" ] && PROJECTS+=("${d%/}")
done

echo "Found ${#PROJECTS[@]} projects"
echo ""

# Phase 1: Generate compile_commands.json where possible (parallel)
echo "=== Phase 1: Generating compile databases ==="
PIDS=()
for project in "${PROJECTS[@]}"; do
    generate_compile_db "$project" &
    PIDS+=($!)
    # Limit parallel cmake invocations
    if [ ${#PIDS[@]} -ge 8 ]; then
        wait "${PIDS[0]}" 2>/dev/null || true
        PIDS=("${PIDS[@]:1}")
    fi
done
wait 2>/dev/null || true
echo "Phase 1 complete: $(date)"

# Phase 2: Run clang indexer on all projects
echo ""
echo "=== Phase 2: Clang semantic indexing ==="
OUTPUT_JSONL="$OUTPUT_DIR/cpp_clang_crossfile_16k.jsonl"
rm -f "$OUTPUT_JSONL"

# Sequential project processing with high intra-project parallelism
# (avoids oversubscribing CPU with workers * parse_workers processes)
$PYTHON "$INDEXER" \
    --projects-dir "$DATA_DIR" \
    --output "$OUTPUT_JSONL" \
    --max-tokens "$MAX_TOKENS" \
    --max-dep-depth "$MAX_DEP_DEPTH" \
    --workers 1 \
    --parse-workers "$WORKERS"

echo ""
echo "Phase 2 complete: $(date)"
echo "Output: $OUTPUT_JSONL"
ls -lh "$OUTPUT_JSONL"
wc -l "$OUTPUT_JSONL"

# Phase 3: Convert to parquet
echo ""
echo "=== Phase 3: Converting to parquet ==="
$PYTHON "$PARQUET_SCRIPT" \
    --input "$OUTPUT_JSONL" \
    --output-dir "$OUTPUT_DIR/parquet/cpp_clang_crossfile_16k" \
    --batch-size 50000

echo ""
echo "=== Pipeline complete ==="
echo "End time: $(date)"
ls -lh "$OUTPUT_DIR/parquet/cpp_clang_crossfile_16k/"
