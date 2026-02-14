#!/bin/bash
# Process new GPU/NPU bare repos: convert to working copies for compilable chunking.
# Usage: bash process_gpu_npu_repos.sh

set -uo pipefail

BARE_DIR="$HOME/data/cpp_raw"
WORK_DIR="/mnt/nvme/nanochat_data/gpu_npu_src"
mkdir -p "$WORK_DIR"

# List of new GPU/NPU repos (bare clones)
NEW_REPOS=(
    cutlass cccl tensorrt-llm transformer-engine nccl tensorrt-oss
    faster-transformer apex cudnn-frontend cuda-library-samples
    thrust cub libcudacxx cuda-quantum matx deep-learning-examples
    nvidia-open-gpu-kernel
    composable-kernel hip miopen rccl rocblas rocfft hipblas
    rocrand rocprim rocsparse rocsolver rocdbgapi clr rocr-runtime rocm-llvm
    openxla-xla stablehlo iree pytorch-xla jax
    fbgemm executorch kineto glow cpuinfo tensorpipe qnnpack
    arm-compute-library arm-nn cmsis-nn cmsis5 cmsis-dsp
    onednn intel-compute-runtime openvino intel-llvm-dpcpp
    ipex intel-neural-compressor intel-graphics-compiler onetbb
    triton flash-attention flashinfer vllm xnnpack apache-tvm onnx onnxruntime-genai
    samsung-one xiaomi-mace qualcomm-aimet
    mindspore apple-mlx gemma-cpp
    aosp-frameworks-native aosp-frameworks-av aosp-system-core
    aosp-hardware-interfaces aosp-frameworks-ml
    vulkan-headers vulkan-loader vulkan-validation-layers
    spirv-tools spirv-cross-khronos glslang-khronos
)

echo "Converting ${#NEW_REPOS[@]} bare repos to working copies..."

converted=0
skipped=0
failed=0

for repo in "${NEW_REPOS[@]}"; do
    bare="$BARE_DIR/$repo"
    work="$WORK_DIR/$repo"

    if [ -d "$work" ]; then
        echo "SKIP (exists): $repo"
        skipped=$((skipped + 1))
        continue
    fi

    if [ ! -d "$bare" ]; then
        echo "SKIP (no bare): $repo"
        skipped=$((skipped + 1))
        continue
    fi

    echo "CONVERT: $repo"
    if git clone --local --shared "$bare" "$work" 2>/dev/null; then
        converted=$((converted + 1))
    else
        echo "FAILED: $repo"
        rm -rf "$work"
        failed=$((failed + 1))
    fi
done

echo ""
echo "=== Summary ==="
echo "Converted: $converted"
echo "Skipped: $skipped"
echo "Failed: $failed"
echo "Total in $WORK_DIR: $(ls -d $WORK_DIR/*/ 2>/dev/null | wc -l)"
echo "Disk usage: $(du -sh $WORK_DIR | cut -f1)"
