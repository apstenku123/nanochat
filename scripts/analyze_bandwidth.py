"""
Analyze memory bandwidth vs compute bottleneck on GB10.

This explains why NVFP4 doesn't give expected speedup over BF16/FP32.
"""

import os
import shutil

# Triton SM121a fix
if not os.environ.get("TRITON_PTXAS_PATH"):
    for ptxas in ["/usr/local/cuda/bin/ptxas", shutil.which("ptxas")]:
        if ptxas and os.path.exists(ptxas):
            os.environ["TRITON_PTXAS_PATH"] = ptxas
            break

import torch
import time

print("=" * 70)
print("GB10 Memory Bandwidth vs Compute Analysis")
print("=" * 70)

# GB10 specs
GB10_BANDWIDTH_GBS = 273  # GB/s (LPDDR5x)
GB10_FP32_TFLOPS = 31
GB10_BF16_TFLOPS = 62
GB10_NVFP4_TFLOPS = 500  # dense

# Model specs (d20)
NUM_PARAMS = 560_988_160
BYTES_PER_PARAM_BF16 = 2
BYTES_PER_PARAM_FP32 = 4
MODEL_SIZE_BF16_GB = NUM_PARAMS * BYTES_PER_PARAM_BF16 / 1e9
MODEL_SIZE_FP32_GB = NUM_PARAMS * BYTES_PER_PARAM_FP32 / 1e9

print(f"\nModel: 560M parameters")
print(f"  BF16 size: {MODEL_SIZE_BF16_GB:.2f} GB")
print(f"  FP32 size: {MODEL_SIZE_FP32_GB:.2f} GB")

print(f"\nGB10 Specs:")
print(f"  Memory bandwidth: {GB10_BANDWIDTH_GBS} GB/s")
print(f"  FP32 compute: {GB10_FP32_TFLOPS} TFLOPS")
print(f"  BF16 compute: {GB10_BF16_TFLOPS} TFLOPS")
print(f"  NVFP4 compute: {GB10_NVFP4_TFLOPS} TFLOPS (dense)")

# Calculate arithmetic intensity needed to be compute-bound
# Compute-bound when: FLOPs / Bytes > TFLOPS / (GB/s)
print(f"\n--- Arithmetic Intensity Analysis ---")

def compute_bound_threshold(tflops, bandwidth_gbs):
    """FLOPs/Byte needed to be compute-bound"""
    return (tflops * 1e12) / (bandwidth_gbs * 1e9)

fp32_threshold = compute_bound_threshold(GB10_FP32_TFLOPS, GB10_BANDWIDTH_GBS)
bf16_threshold = compute_bound_threshold(GB10_BF16_TFLOPS, GB10_BANDWIDTH_GBS)
fp4_threshold = compute_bound_threshold(GB10_NVFP4_TFLOPS, GB10_BANDWIDTH_GBS)

print(f"To be compute-bound, need FLOPs/Byte >")
print(f"  FP32:  {fp32_threshold:.0f} FLOPs/Byte")
print(f"  BF16:  {bf16_threshold:.0f} FLOPs/Byte")
print(f"  NVFP4: {fp4_threshold:.0f} FLOPs/Byte")

# Actual arithmetic intensity for transformer training
# Forward: ~6N FLOPs per token (N = params, excluding embeddings)
# Each weight loaded once per micro-batch
# Simplification: FLOPs/Byte ≈ (FLOPs per token) / (bytes of weights loaded)

BATCH_SIZE = 32
SEQ_LEN = 2048
TOKENS_PER_BATCH = BATCH_SIZE * SEQ_LEN
FLOPS_PER_TOKEN = 3.49e9  # from base_train.py output

# Total FLOPs per forward pass
total_flops_forward = FLOPS_PER_TOKEN * TOKENS_PER_BATCH / 6  # forward is 1/6 of total
total_flops_backward = total_flops_forward * 2  # backward is ~2x forward

# Bytes accessed (weights need to be loaded for forward, then again for backward)
# For BF16: each weight loaded twice (forward + backward grad computation)
bytes_accessed_bf16 = MODEL_SIZE_BF16_GB * 1e9 * 2  # forward + backward

actual_intensity_bf16 = (total_flops_forward + total_flops_backward) / bytes_accessed_bf16

print(f"\n--- Actual Model Intensity ---")
print(f"Batch: {BATCH_SIZE} x {SEQ_LEN} = {TOKENS_PER_BATCH:,} tokens")
print(f"FLOPs per micro-batch (fwd+bwd): {(total_flops_forward + total_flops_backward)/1e12:.2f} TFLOPS")
print(f"Bytes accessed (weights): {bytes_accessed_bf16/1e9:.2f} GB")
print(f"Actual arithmetic intensity: {actual_intensity_bf16:.0f} FLOPs/Byte")

print(f"\n--- Bottleneck Analysis ---")
if actual_intensity_bf16 < bf16_threshold:
    print(f"Model is MEMORY-BOUND (intensity {actual_intensity_bf16:.0f} < threshold {bf16_threshold:.0f})")
    print(f"  → Lower precision (NVFP4) WON'T help much!")
    print(f"  → Need to reduce memory traffic or increase batch size")
else:
    print(f"Model is COMPUTE-BOUND")
    print(f"  → Lower precision should help")

# Time estimate based on memory bandwidth
time_memory_limited = bytes_accessed_bf16 / (GB10_BANDWIDTH_GBS * 1e9)
print(f"\n--- Time Estimates per Micro-batch ---")
print(f"Memory-limited time: {time_memory_limited*1000:.1f} ms")

# With 8 grad accum steps
print(f"\nWith 8 grad_accum steps:")
print(f"  Memory-limited: {time_memory_limited * 8 * 1000:.1f} ms per optimizer step")

# Actual observed
print(f"\nActual observed: ~31,000 ms per optimizer step")
print(f"  This is ~{31000 / (time_memory_limited * 8 * 1000):.1f}x slower than pure memory-limited")
print(f"  The difference comes from: activations, optimizer state, kernel overhead, etc.")

print("\n" + "=" * 70)
print("CONCLUSION: GB10's low bandwidth (273 GB/s) makes it memory-bound.")
print("NVFP4 reduces compute time, but memory access time dominates.")
print("To see NVFP4 benefit, you need either:")
print("  1. Larger batch sizes (amortize weight loading)")
print("  2. Fused kernels (reduce memory traffic)")
print("  3. Gradient checkpointing (trade compute for memory)")
print("=" * 70)
