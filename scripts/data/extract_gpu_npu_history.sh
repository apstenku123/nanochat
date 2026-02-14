#!/bin/bash
# Extract git history from new GPU/NPU bare repos.
# These are bare repos so we use --git-dir instead of -C.
# Usage: bash extract_gpu_npu_history.sh

set -uo pipefail

BARE_DIR="$HOME/data/cpp_raw"
OUTPUT_DIR="/mnt/nvme/nanochat_data/gpu_npu_commits"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUTPUT_DIR"

# List of new GPU/NPU repos (bare clones) - only the ones with meaningful C++ code
NEW_REPOS=(
    cutlass cccl tensorrt-llm transformer-engine nccl tensorrt-oss
    faster-transformer apex cudnn-frontend cuda-library-samples
    thrust cub libcudacxx cuda-quantum matx
    nvidia-open-gpu-kernel
    composable-kernel hip miopen rccl rocblas rocfft hipblas
    rocrand rocprim rocsparse rocsolver clr rocr-runtime
    openxla-xla stablehlo iree pytorch-xla
    fbgemm executorch kineto glow cpuinfo tensorpipe qnnpack
    arm-compute-library arm-nn cmsis-nn cmsis5 cmsis-dsp
    onednn intel-compute-runtime openvino intel-llvm-dpcpp
    ipex intel-graphics-compiler onetbb
    triton flash-attention flashinfer vllm xnnpack onnx onnxruntime-genai
    samsung-one xiaomi-mace qualcomm-aimet
    mindspore apple-mlx gemma-cpp
    aosp-frameworks-native aosp-frameworks-av aosp-system-core
    aosp-hardware-interfaces aosp-frameworks-ml
    vulkan-validation-layers spirv-tools spirv-cross-khronos
)

echo "Extracting git history from ${#NEW_REPOS[@]} repos..."
echo "Output dir: $OUTPUT_DIR"
echo ""

total_records=0
processed=0
skipped=0

for repo in "${NEW_REPOS[@]}"; do
    bare="$BARE_DIR/$repo"
    output="$OUTPUT_DIR/${repo}_commits.jsonl"

    if [ ! -d "$bare" ]; then
        echo "SKIP (not found): $repo"
        skipped=$((skipped + 1))
        continue
    fi

    if [ -f "$output" ] && [ -s "$output" ]; then
        count=$(wc -l < "$output")
        echo "SKIP (exists, $count records): $repo"
        total_records=$((total_records + count))
        processed=$((processed + 1))
        continue
    fi

    echo "EXTRACT: $repo (max 30000 commits)"
    python3 ~/nanochat/scripts/data/extract_git_history.py \
        --repo "$bare" \
        --output "$output" \
        --max_commits 30000 2>&1 | tail -5

    if [ -f "$output" ]; then
        count=$(wc -l < "$output")
        total_records=$((total_records + count))
        echo "  -> $count records"
    fi
    processed=$((processed + 1))
done

# Concatenate all files
ALL_OUTPUT="/mnt/nvme/nanochat_data/gpu_npu_raw_commits_all.jsonl"
echo ""
echo "Concatenating all commit files..."
cat "$OUTPUT_DIR"/*_commits.jsonl > "$ALL_OUTPUT" 2>/dev/null || true

final_count=$(wc -l < "$ALL_OUTPUT" 2>/dev/null || echo 0)
final_size=$(du -sh "$ALL_OUTPUT" 2>/dev/null | cut -f1 || echo "0")

echo ""
echo "=== Summary ==="
echo "Repos processed: $processed"
echo "Repos skipped: $skipped"
echo "Total records: $total_records"
echo "Combined output: $ALL_OUTPUT ($final_size, $final_count records)"
