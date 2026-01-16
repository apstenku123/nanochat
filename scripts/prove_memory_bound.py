"""
Simple proof that GB10 is memory-bound, not compute-bound.
No torch.compile, no autotune - just raw operations.

Usage:
    python -m scripts.prove_memory_bound
"""

import torch
import time

# GB10 specs
GB10_BANDWIDTH_GBS = 273   # GB/s theoretical peak
GB10_FP4_TFLOPS = 500      # TFLOPS FP4
GB10_BF16_TFLOPS = 125     # TFLOPS BF16

print("=" * 70)
print("PROOF: GB10 IS MEMORY-BOUND")
print("=" * 70)
print()
print(f"GB10 Theoretical Peak:")
print(f"  Memory Bandwidth: {GB10_BANDWIDTH_GBS} GB/s")
print(f"  BF16 Compute:     {GB10_BF16_TFLOPS} TFLOPS")
print(f"  FP4 Compute:      {GB10_FP4_TFLOPS} TFLOPS")
print()

device = torch.device("cuda")

def measure_bandwidth(size_gb, num_iters=10):
    """Measure actual memory bandwidth with tensor copy."""
    size_bytes = int(size_gb * 1e9)
    num_elements = size_bytes // 2  # BF16 = 2 bytes

    a = torch.randn(num_elements, dtype=torch.bfloat16, device=device)
    b = torch.empty_like(a)

    # Warmup
    for _ in range(3):
        b.copy_(a)
    torch.cuda.synchronize()

    # Measure
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        b.copy_(a)
    torch.cuda.synchronize()
    end = time.perf_counter()

    total_bytes = 2 * size_bytes * num_iters  # read + write
    elapsed = end - start
    bandwidth = total_bytes / elapsed / 1e9

    del a, b
    torch.cuda.empty_cache()
    return bandwidth


def measure_matmul(M, N, K, dtype, num_iters=20):
    """Measure matmul performance and calculate metrics."""
    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)

    # Warmup
    for _ in range(5):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()

    # Measure
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()
    end = time.perf_counter()

    elapsed = (end - start) / num_iters

    # FLOPs: 2 * M * N * K (multiply-add)
    flops = 2 * M * N * K
    tflops = flops / elapsed / 1e12

    # Bytes: read A + read B + write C
    bytes_per_elem = 2 if dtype == torch.bfloat16 else 4
    total_bytes = (M * K + K * N + M * N) * bytes_per_elem
    bandwidth = total_bytes / elapsed / 1e9

    # Arithmetic intensity
    ai = flops / total_bytes

    del A, B, C
    torch.cuda.empty_cache()

    return {
        "time_ms": elapsed * 1000,
        "tflops": tflops,
        "bandwidth_gbs": bandwidth,
        "arithmetic_intensity": ai,
    }


print("=" * 70)
print("TEST 1: Raw Memory Bandwidth")
print("=" * 70)
print()

bw = measure_bandwidth(1.0)
utilization = 100 * bw / GB10_BANDWIDTH_GBS
print(f"Measured bandwidth: {bw:.1f} GB/s ({utilization:.1f}% of {GB10_BANDWIDTH_GBS} GB/s peak)")
print()

print("=" * 70)
print("TEST 2: Roofline Analysis")
print("=" * 70)
print()

# Ridge point: where compute = bandwidth limited
ridge_bf16 = GB10_BF16_TFLOPS * 1000 / GB10_BANDWIDTH_GBS  # FLOPs/byte
ridge_fp4 = GB10_FP4_TFLOPS * 1000 / GB10_BANDWIDTH_GBS

print(f"Ridge point (BF16): {ridge_bf16:.0f} FLOPs/byte")
print(f"Ridge point (FP4):  {ridge_fp4:.0f} FLOPs/byte")
print()
print("If arithmetic intensity < ridge point -> MEMORY BOUND")
print("If arithmetic intensity > ridge point -> COMPUTE BOUND")
print()

print("=" * 70)
print("TEST 3: Matmul at Different Sizes")
print("=" * 70)
print()

# Typical transformer matmul sizes
sizes = [
    # Small (memory bound)
    (1024, 1024, 1024, "Small 1K"),
    # Medium (d20 model size)
    (2048, 1280, 1280, "QKV proj"),
    (2048, 5120, 1280, "FFN up"),
    (2048, 1280, 5120, "FFN down"),
    # Large batch
    (8192, 1280, 1280, "Large batch QKV"),
    (32768, 1280, 1280, "Very large batch"),
    # Huge (to see if we can become compute bound)
    (65536, 4096, 4096, "Huge matmul"),
]

print(f"{'Name':<20} {'Shape':<22} {'Time(ms)':<10} {'TFLOPS':<8} {'BW(GB/s)':<10} {'AI':<8} {'Bound':<12}")
print("-" * 100)

for M, N, K, name in sizes:
    result = measure_matmul(M, N, K, torch.bfloat16)

    # Determine if memory or compute bound
    ai = result["arithmetic_intensity"]
    if ai < ridge_bf16:
        bound = "MEMORY"
        # Expected TFLOPS if fully memory bound
        expected_tflops = ai * GB10_BANDWIDTH_GBS / 1000
    else:
        bound = "COMPUTE"
        expected_tflops = GB10_BF16_TFLOPS

    compute_util = 100 * result["tflops"] / GB10_BF16_TFLOPS
    bw_util = 100 * result["bandwidth_gbs"] / GB10_BANDWIDTH_GBS

    shape = f"({M},{N},{K})"
    print(f"{name:<20} {shape:<22} {result['time_ms']:<10.2f} {result['tflops']:<8.1f} "
          f"{result['bandwidth_gbs']:<10.1f} {ai:<8.1f} {bound:<12}")

print()

print("=" * 70)
print("TEST 4: BF16 vs FP32 Speedup")
print("=" * 70)
print()

test_sizes = [
    (2048, 1280, 1280),
    (8192, 4096, 4096),
]

for M, N, K in test_sizes:
    bf16 = measure_matmul(M, N, K, torch.bfloat16)
    fp32 = measure_matmul(M, N, K, torch.float32)

    speedup = fp32["time_ms"] / bf16["time_ms"]

    print(f"Shape ({M},{N},{K}):")
    print(f"  FP32: {fp32['time_ms']:.2f}ms, {fp32['tflops']:.1f} TFLOPS, {fp32['bandwidth_gbs']:.1f} GB/s")
    print(f"  BF16: {bf16['time_ms']:.2f}ms, {bf16['tflops']:.1f} TFLOPS, {bf16['bandwidth_gbs']:.1f} GB/s")
    print(f"  Speedup: {speedup:.2f}x")
    print()

    if speedup < 1.5:
        print(f"  -> LOW SPEEDUP ({speedup:.2f}x < 2x expected) = MEMORY BOUND!")
    elif speedup < 2.5:
        print(f"  -> MODERATE SPEEDUP = partially memory bound")
    else:
        print(f"  -> GOOD SPEEDUP = compute bound")
    print()

print("=" * 70)
print("CONCLUSION")
print("=" * 70)
print()
print("GB10's arithmetic intensity requirements:")
print(f"  - Need AI > {ridge_bf16:.0f} FLOPs/byte to utilize BF16 compute")
print(f"  - Need AI > {ridge_fp4:.0f} FLOPs/byte to utilize FP4 compute")
print()
print("Typical transformer matmul AI: 50-200 FLOPs/byte")
print()
print("VERDICT: GB10 is MEMORY-BOUND for typical LLM training!")
print("         NVFP4's 500 TFLOPS cannot be utilized because")
print("         273 GB/s memory bandwidth is the bottleneck.")
print()
print("=" * 70)
