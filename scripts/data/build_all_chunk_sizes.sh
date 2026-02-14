#!/bin/bash
# Build all chunk sizes for training: 16K, 64K, 128K, 256K, 512K
# Processes: tree-sitter compilable, commit chains, clang semantic
# Naming convention: {method}_{tokens}k.jsonl → parquet/{method}_{tokens}k/
#
# Methods:
#   treesitter_compilable - tree-sitter bottom-up compilable units
#   git_commit_chains     - pre→post code from git history via commit mode
#   clang_semantic        - libclang cross-file dependency indexer

DATA="/mnt/nvme/nanochat_data"
CHUNKER="$HOME/cpp-chunker"
PARQUET_SCRIPT="$HOME/nanochat/scripts/data/batch_jsonl_to_parquet.py"

# Token sizes to generate
SIZES=(131072 262144 524288)  # 128K, 256K, 512K
SIZE_LABELS=("128k" "256k" "512k")

echo "============================================"
echo "Building all chunk sizes: ${SIZE_LABELS[*]}"
echo "Start: $(date)"
echo "============================================"
echo ""

###############################################################
# 1. Tree-sitter compilable chunks
###############################################################
echo "=== TREE-SITTER COMPILABLE CHUNKS ==="

for i in "${!SIZES[@]}"; do
    tokens="${SIZES[$i]}"
    label="${SIZE_LABELS[$i]}"
    output="$DATA/treesitter_compilable_${label}.jsonl"

    if [ -f "$output" ] && [ -s "$output" ]; then
        echo "SKIP (exists): treesitter_compilable_${label} ($(wc -l < "$output") docs)"
        continue
    fi

    echo ""
    echo "--- treesitter_compilable_${label} (max_tokens=$tokens) ---"

    # Process old repos (regular clones in ~/data/cpp_raw)
    tmp_old="$DATA/ts_old_${label}.jsonl"
    echo "  Processing original repos..."
    $CHUNKER --project-dirs ~/data/cpp_raw --output "$tmp_old" \
        --max-tokens "$tokens" --cross-depth 3 --compilable --max-file-bytes 500000

    # Process new GPU/NPU repos (working copies in gpu_npu_src)
    tmp_new="$DATA/ts_gpu_${label}.jsonl"
    echo "  Processing GPU/NPU repos..."
    $CHUNKER --project-dirs "$DATA/gpu_npu_src" --output "$tmp_new" \
        --max-tokens "$tokens" --cross-depth 3 --compilable --max-file-bytes 500000

    # Concatenate
    cat "$tmp_old" "$tmp_new" > "$output"
    rm -f "$tmp_old" "$tmp_new"

    docs=$(wc -l < "$output")
    size=$(du -sh "$output" | cut -f1)
    echo "  treesitter_compilable_${label}: $docs docs, $size"
done

###############################################################
# 2. Git commit chains (pre→post code)
###############################################################
echo ""
echo "=== GIT COMMIT CHAINS ==="

# We have 3 raw commit sources:
#   commitpack_raw_commits.jsonl (2M records, 122 GB)
#   raw_commits_all.jsonl (502K records, 38 GB)
#   gpu_npu_raw_commits_all.jsonl (445K records, 29 GB)

RAW_INPUTS=(
    "$DATA/commitpack_raw_commits.jsonl"
    "$DATA/raw_commits_all.jsonl"
    "$DATA/gpu_npu_raw_commits_all.jsonl"
)
RAW_LABELS=("commitpack" "git_history" "gpu_npu")

for i in "${!SIZES[@]}"; do
    tokens="${SIZES[$i]}"
    label="${SIZE_LABELS[$i]}"
    combined="$DATA/git_commit_chains_${label}.jsonl"

    if [ -f "$combined" ] && [ -s "$combined" ]; then
        echo "SKIP (exists): git_commit_chains_${label} ($(wc -l < "$combined") docs)"
        continue
    fi

    echo ""
    echo "--- git_commit_chains_${label} (max_tokens=$tokens) ---"

    parts=()
    for j in "${!RAW_INPUTS[@]}"; do
        raw="${RAW_INPUTS[$j]}"
        raw_label="${RAW_LABELS[$j]}"

        if [ ! -f "$raw" ]; then
            echo "  SKIP (not found): $raw_label"
            continue
        fi

        part="$DATA/chains_${raw_label}_${label}.jsonl"
        echo "  Processing $raw_label..."
        $CHUNKER --commit-mode --commit-format both \
            --inputs "$raw" --output "$part" \
            --max-tokens "$tokens" --max-file-bytes 500000
        parts+=("$part")
    done

    # Concatenate all parts
    cat "${parts[@]}" > "$combined"
    rm -f "${parts[@]}"

    docs=$(wc -l < "$combined")
    size=$(du -sh "$combined" | cut -f1)
    echo "  git_commit_chains_${label}: $docs docs, $size"
done

###############################################################
# 3. Also rename existing datasets with proper names
###############################################################
echo ""
echo "=== RENAMING EXISTING DATASETS ==="

# Create symlinks with proper names for existing 16K and 64K data
for src_dst in \
    "cpp_compilable_16k_v2.jsonl:treesitter_compilable_16k.jsonl" \
    "cpp_compilable_64k_v2.jsonl:treesitter_compilable_64k.jsonl" \
    "gpu_npu_compilable_16k.jsonl:treesitter_compilable_gpu_npu_16k.jsonl" \
    "gpu_npu_compilable_64k.jsonl:treesitter_compilable_gpu_npu_64k.jsonl" \
    "gpu_npu_clang_16k.jsonl:clang_semantic_16k.jsonl" \
    ; do
    src="${src_dst%%:*}"
    dst="${src_dst##*:}"
    if [ -f "$DATA/$src" ] && [ ! -f "$DATA/$dst" ]; then
        ln -s "$DATA/$src" "$DATA/$dst" 2>/dev/null || cp "$DATA/$src" "$DATA/$dst"
        echo "  Linked: $dst -> $src"
    fi
done

# Combine old+new compilable data for 16K and 64K proper names
for label in "16k" "64k"; do
    combined="$DATA/treesitter_compilable_${label}.jsonl"
    if [ ! -f "$combined" ]; then
        old="$DATA/cpp_compilable_${label}_v2.jsonl"
        new="$DATA/gpu_npu_compilable_${label}.jsonl"
        if [ -f "$old" ] && [ -f "$new" ]; then
            echo "  Combining: treesitter_compilable_${label} (old + gpu/npu)"
            cat "$old" "$new" > "$combined"
        elif [ -f "$old" ]; then
            ln -s "$old" "$combined" 2>/dev/null || cp "$old" "$combined"
        fi
    fi
done

# Combine chain data for 16K
combined_chains_16k="$DATA/git_commit_chains_16k.jsonl"
if [ ! -f "$combined_chains_16k" ]; then
    echo "  Combining: git_commit_chains_16k"
    cat "$DATA/commitpack_chains.jsonl" "$DATA/git_history_chains.jsonl" \
        "$DATA/gpu_npu_chains.jsonl" > "$combined_chains_16k" 2>/dev/null || true
fi

###############################################################
# 4. Convert ALL to parquet
###############################################################
echo ""
echo "=== CONVERTING ALL TO PARQUET ==="

PARQUET_BASE="$DATA/parquet"
mkdir -p "$PARQUET_BASE"

to_parquet() {
    local input="$1"
    local name="$2"
    local rows="${3:-50000}"
    local dir="$PARQUET_BASE/$name"

    if [ -f "$dir/_COMPLETE" ]; then
        echo "  SKIP parquet (done): $name"
        return
    fi
    if [ ! -f "$input" ] || [ ! -s "$input" ]; then
        echo "  SKIP parquet (no input): $name"
        return
    fi

    # Delete partial parquet dir
    rm -rf "$dir"

    echo "  Converting to parquet: $name"
    python3 "$PARQUET_SCRIPT" --input "$input" --parquet_dir "$dir" --rows_per_file "$rows"
}

# Tree-sitter compilable (all sizes)
for label in "16k" "64k" "128k" "256k" "512k"; do
    input="$DATA/treesitter_compilable_${label}.jsonl"
    rows=50000
    [ "$label" = "64k" ] && rows=20000
    [ "$label" = "128k" ] && rows=10000
    [ "$label" = "256k" ] && rows=5000
    [ "$label" = "512k" ] && rows=2000
    to_parquet "$input" "treesitter_compilable_${label}" "$rows"
done

# Git commit chains (all sizes)
for label in "16k" "128k" "256k" "512k"; do
    input="$DATA/git_commit_chains_${label}.jsonl"
    rows=50000
    [ "$label" = "128k" ] && rows=10000
    [ "$label" = "256k" ] && rows=5000
    [ "$label" = "512k" ] && rows=2000
    to_parquet "$input" "git_commit_chains_${label}" "$rows"
done

# Clang semantic
to_parquet "$DATA/clang_semantic_16k.jsonl" "clang_semantic_16k" 50000

# MS source
to_parquet "$DATA/ms_src_cpp.jsonl" "ms_src_cpp" 50000

echo ""
echo "============================================"
echo "ALL DONE: $(date)"
echo "============================================"
echo ""
echo "=== PARQUET INVENTORY ==="
for d in "$PARQUET_BASE"/*/; do
    name=$(basename "$d")
    if [ -f "$d/_COMPLETE" ]; then
        shards=$(ls "$d"/shard_*.parquet 2>/dev/null | wc -l)
        size=$(du -sh "$d" | cut -f1)
        total=$(grep -o 'total' "$d/_COMPLETE" | head -1)
        info=$(cat "$d/_COMPLETE")
        echo "  $name: $shards shards, $size — $info"
    else
        echo "  $name: INCOMPLETE"
    fi
done
