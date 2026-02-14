#!/bin/bash
# Download GPU/NPU/accelerator C++ repos for training corpus.
# Target: CUDA, ROCm, XLA/TPU, ARM, Intel, Samsung, Xiaomi, PyTorch ecosystem,
#         ML compilers, Android AOSP native, Vulkan compute.
#
# Usage: bash scripts/data/download_gpu_npu_repos.sh ~/data/cpp_raw

set -euo pipefail

DEST="${1:-$HOME/data/cpp_raw}"
mkdir -p "$DEST"

clone_or_skip() {
    local url="$1"
    local name="$2"
    local target="$DEST/$name"
    if [ -d "$target" ]; then
        echo "SKIP (exists): $name"
        return
    fi
    echo "CLONING: $name from $url"
    git clone --bare "$url" "$target" 2>&1 | tail -1 || {
        echo "FAILED: $name"
        rm -rf "$target"
    }
}

echo "=== NVIDIA CUDA Ecosystem ==="
clone_or_skip https://github.com/NVIDIA/cutlass cutlass
clone_or_skip https://github.com/NVIDIA/cccl cccl
clone_or_skip https://github.com/NVIDIA/TensorRT-LLM tensorrt-llm
clone_or_skip https://github.com/NVIDIA/TransformerEngine transformer-engine
clone_or_skip https://github.com/NVIDIA/nccl nccl
clone_or_skip https://github.com/NVIDIA/TensorRT tensorrt-oss
clone_or_skip https://github.com/NVIDIA/FasterTransformer faster-transformer
clone_or_skip https://github.com/NVIDIA/apex apex
clone_or_skip https://github.com/NVIDIA/cudnn-frontend cudnn-frontend
clone_or_skip https://github.com/NVIDIA/CUDALibrarySamples cuda-library-samples
clone_or_skip https://github.com/NVIDIA/thrust thrust
clone_or_skip https://github.com/NVIDIA/cub cub
clone_or_skip https://github.com/NVIDIA/libcudacxx libcudacxx
clone_or_skip https://github.com/NVIDIA/cuda-quantum cuda-quantum
clone_or_skip https://github.com/NVIDIA/MatX matx
clone_or_skip https://github.com/NVIDIA/NeMo-Guardrails nemo-guardrails
clone_or_skip https://github.com/NVIDIA/DeepLearningExamples deep-learning-examples
clone_or_skip https://github.com/NVIDIA/gpu-operator gpu-operator
clone_or_skip https://github.com/NVIDIA/open-gpu-kernel-modules nvidia-open-gpu-kernel

echo ""
echo "=== AMD ROCm Ecosystem ==="
clone_or_skip https://github.com/ROCm/composable_kernel composable-kernel
clone_or_skip https://github.com/ROCm/HIP hip
clone_or_skip https://github.com/ROCm/MIOpen miopen
clone_or_skip https://github.com/ROCm/rccl rccl
clone_or_skip https://github.com/ROCm/rocBLAS rocblas
clone_or_skip https://github.com/ROCm/rocFFT rocfft
clone_or_skip https://github.com/ROCm/hipBLAS hipblas
clone_or_skip https://github.com/ROCm/rocRAND rocrand
clone_or_skip https://github.com/ROCm/rocPRIM rocprim
clone_or_skip https://github.com/ROCm/rocSPARSE rocsparse
clone_or_skip https://github.com/ROCm/rocSOLVER rocsolver
clone_or_skip https://github.com/ROCm/ROCdbgapi rocdbgapi
clone_or_skip https://github.com/ROCm/rocm-cmake rocm-cmake
clone_or_skip https://github.com/ROCm/clr clr
clone_or_skip https://github.com/ROCm/ROCR-Runtime rocr-runtime
clone_or_skip https://github.com/ROCm/llvm-project rocm-llvm

echo ""
echo "=== Google XLA / TPU / StableHLO ==="
clone_or_skip https://github.com/openxla/xla openxla-xla
clone_or_skip https://github.com/openxla/stablehlo stablehlo
clone_or_skip https://github.com/openxla/iree iree
clone_or_skip https://github.com/pytorch/xla pytorch-xla
clone_or_skip https://github.com/jax-ml/jax jax

echo ""
echo "=== PyTorch Ecosystem (C++ heavy) ==="
clone_or_skip https://github.com/pytorch/FBGEMM fbgemm
clone_or_skip https://github.com/pytorch/executorch executorch
clone_or_skip https://github.com/pytorch/kineto kineto
clone_or_skip https://github.com/pytorch/glow glow
clone_or_skip https://github.com/pytorch/cpuinfo cpuinfo
clone_or_skip https://github.com/pytorch/tensorpipe tensorpipe
clone_or_skip https://github.com/pytorch/QNNPACK qnnpack

echo ""
echo "=== ARM GPU/NPU ==="
clone_or_skip https://github.com/ARM-software/ComputeLibrary arm-compute-library
clone_or_skip https://github.com/ARM-software/armnn arm-nn
clone_or_skip https://github.com/ARM-software/CMSIS-NN cmsis-nn
clone_or_skip https://github.com/ARM-software/ethos-u-vela ethos-u-vela
clone_or_skip https://github.com/ARM-software/CMSIS_5 cmsis5
clone_or_skip https://github.com/ARM-software/CMSIS-DSP cmsis-dsp

echo ""
echo "=== Intel GPU / OneAPI ==="
clone_or_skip https://github.com/oneapi-src/oneDNN onednn
clone_or_skip https://github.com/intel/compute-runtime intel-compute-runtime
clone_or_skip https://github.com/openvinotoolkit/openvino openvino
clone_or_skip https://github.com/intel/llvm intel-llvm-dpcpp
clone_or_skip https://github.com/intel/intel-extension-for-pytorch ipex
clone_or_skip https://github.com/intel/neural-compressor intel-neural-compressor
clone_or_skip https://github.com/intel/intel-graphics-compiler intel-graphics-compiler
clone_or_skip https://github.com/oneapi-src/oneTBB onetbb

echo ""
echo "=== ML Compilers & Inference ==="
clone_or_skip https://github.com/triton-lang/triton triton
clone_or_skip https://github.com/Dao-AILab/flash-attention flash-attention
clone_or_skip https://github.com/flashinfer-ai/flashinfer flashinfer
clone_or_skip https://github.com/vllm-project/vllm vllm
clone_or_skip https://github.com/google/XNNPACK xnnpack
clone_or_skip https://github.com/apache/tvm apache-tvm
clone_or_skip https://github.com/onnx/onnx onnx
clone_or_skip https://github.com/microsoft/onnxruntime-genai onnxruntime-genai

echo ""
echo "=== Samsung / Xiaomi / Qualcomm ==="
clone_or_skip https://github.com/Samsung/ONE samsung-one
clone_or_skip https://github.com/XiaoMi/mace xiaomi-mace
clone_or_skip https://github.com/quic/aimet qualcomm-aimet

echo ""
echo "=== Huawei / Apple / Other ==="
clone_or_skip https://github.com/mindspore-ai/mindspore mindspore
clone_or_skip https://github.com/ml-explore/mlx apple-mlx
clone_or_skip https://github.com/google/gemma.cpp gemma-cpp
clone_or_skip https://github.com/ggerganov/whisper.cpp whisper-cpp-latest
clone_or_skip https://github.com/Mozilla-Ocho/llamafile llamafile-latest

echo ""
echo "=== Android AOSP Native ==="
clone_or_skip https://android.googlesource.com/platform/frameworks/native aosp-frameworks-native
clone_or_skip https://android.googlesource.com/platform/frameworks/av aosp-frameworks-av
clone_or_skip https://android.googlesource.com/platform/system/core aosp-system-core
clone_or_skip https://android.googlesource.com/platform/hardware/interfaces aosp-hardware-interfaces
clone_or_skip https://android.googlesource.com/platform/frameworks/ml aosp-frameworks-ml

echo ""
echo "=== Vulkan Compute ==="
clone_or_skip https://github.com/KhronosGroup/Vulkan-Headers vulkan-headers
clone_or_skip https://github.com/KhronosGroup/Vulkan-Loader vulkan-loader
clone_or_skip https://github.com/KhronosGroup/Vulkan-ValidationLayers vulkan-validation-layers
clone_or_skip https://github.com/KhronosGroup/SPIRV-Tools spirv-tools
clone_or_skip https://github.com/KhronosGroup/SPIRV-Cross spirv-cross-khronos
clone_or_skip https://github.com/KhronosGroup/glslang glslang-khronos

echo ""
echo "=== Summary ==="
echo "Total repos in $DEST: $(ls -d $DEST/*/ 2>/dev/null | wc -l)"
echo "Disk usage: $(du -sh $DEST | cut -f1)"
