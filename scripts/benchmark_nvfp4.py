"""
Benchmark raw matmul performance across FP32, BF16, and NVFP4 on GB10.

This script tests the raw compute throughput of different precision modes
to verify if NVFP4 is actually faster than BF16 on this hardware.

Usage:
    python -m scripts.benchmark_nvfp4

GB10 Theoretical Peak Performance:
- FP32: 31 TFLOPS
- BF16: 62 TFLOPS
- NVFP4: 500 TFLOPS (dense), 1000 TFLOPS (sparse)
- Memory Bandwidth: 273 GB/s (LPDDR5x)

Key insight: GB10's low memory bandwidth means matmuls are often memory-bound.
NVFP4 only helps when the problem is compute-bound (high arithmetic intensity).
"""

import os
import shutil
import time
import gc
from dataclasses import dataclass
from typing import Optional, List, Tuple

# Triton SM121a fix
if not os.environ.get("TRITON_PTXAS_PATH"):
    for ptxas in ["/usr/local/cuda/bin/ptxas", shutil.which("ptxas")]:
        if ptxas and os.path.exists(ptxas):
            os.environ["TRITON_PTXAS_PATH"] = ptxas
            break

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "1"

import torch

# Patch is_big_gpu for GB10 (48 SMs < 68 SM threshold)
try:
    import torch._inductor.utils as inductor_utils
    inductor_utils.is_big_gpu = lambda index=0: True
except Exception:
    pass

# ==============================================================================
# GB10 Hardware Specs
# ==============================================================================
GB10_FP32_TFLOPS = 31.0
GB10_BF16_TFLOPS = 62.0
GB10_NVFP4_TFLOPS = 500.0  # dense
GB10_NVFP4_SPARSE_TFLOPS = 1000.0  # with sparsity
GB10_BANDWIDTH_GBS = 273.0  # LPDDR5x memory bandwidth

# ==============================================================================
# TransformerEngine Support
# ==============================================================================
TE_AVAILABLE = False
te = None
TE_NVFP4BlockScaling = None
TE_Format = None

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format as TE_Format
    try:
        from transformer_engine.common.recipe import NVFP4BlockScaling as TE_NVFP4BlockScaling
        TE_AVAILABLE = True
    except ImportError:
        print("WARNING: NVFP4BlockScaling not available in TransformerEngine")
except ImportError:
    print("WARNING: TransformerEngine not installed")


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    m: int
    n: int
    k: int
    time_ms: float
    tflops: float
    efficiency: float  # vs theoretical peak
    bytes_accessed: float  # GB
    arithmetic_intensity: float  # FLOPs/Byte
    memory_bound: bool


def get_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def reset_memory():
    """Reset memory tracking and clear cache."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()


def compute_arithmetic_intensity(m: int, n: int, k: int, bytes_per_element: float) -> Tuple[float, float]:
    """
    Compute arithmetic intensity for a matmul A @ B = C.

    A: (m, k) - bytes_per_element each
    B: (k, n) - bytes_per_element each
    C: (m, n) - output (often same as bytes_per_element)

    FLOPs = 2 * m * n * k (multiply + accumulate)
    Bytes = (m*k + k*n + m*n) * bytes_per_element (read A, B, write C)

    Returns: (intensity in FLOPs/Byte, total bytes in GB)
    """
    flops = 2 * m * n * k
    bytes_accessed = (m * k + k * n + m * n) * bytes_per_element
    intensity = flops / bytes_accessed
    return intensity, bytes_accessed / 1e9


def is_memory_bound(intensity: float, peak_tflops: float, bandwidth_gbs: float = GB10_BANDWIDTH_GBS) -> bool:
    """
    Determine if a workload is memory-bound.

    Memory-bound when: time_memory > time_compute
    i.e., bytes / bandwidth > flops / peak_compute
    i.e., flops / bytes < peak_compute / bandwidth
    """
    threshold = (peak_tflops * 1e12) / (bandwidth_gbs * 1e9)  # FLOPs/Byte threshold
    return intensity < threshold


def benchmark_matmul_fp32(m: int, n: int, k: int, num_warmup: int = 10, num_iters: int = 100) -> BenchmarkResult:
    """Benchmark FP32 matmul."""
    reset_memory()

    A = torch.randn(m, k, dtype=torch.float32, device="cuda")
    B = torch.randn(k, n, dtype=torch.float32, device="cuda")

    # Warmup
    for _ in range(num_warmup):
        C = torch.mm(A, B)
    torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        C = torch.mm(A, B)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    time_ms = (elapsed / num_iters) * 1000
    flops = 2 * m * n * k
    tflops = (flops / (time_ms / 1000)) / 1e12
    efficiency = (tflops / GB10_FP32_TFLOPS) * 100

    intensity, bytes_gb = compute_arithmetic_intensity(m, n, k, 4.0)
    mem_bound = is_memory_bound(intensity, GB10_FP32_TFLOPS)

    del A, B, C
    reset_memory()

    return BenchmarkResult(
        name="FP32",
        m=m, n=n, k=k,
        time_ms=time_ms,
        tflops=tflops,
        efficiency=efficiency,
        bytes_accessed=bytes_gb,
        arithmetic_intensity=intensity,
        memory_bound=mem_bound
    )


def benchmark_matmul_bf16(m: int, n: int, k: int, num_warmup: int = 10, num_iters: int = 100) -> BenchmarkResult:
    """Benchmark BF16 matmul."""
    reset_memory()

    A = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(k, n, dtype=torch.bfloat16, device="cuda")

    # Warmup
    for _ in range(num_warmup):
        C = torch.mm(A, B)
    torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        C = torch.mm(A, B)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    time_ms = (elapsed / num_iters) * 1000
    flops = 2 * m * n * k
    tflops = (flops / (time_ms / 1000)) / 1e12
    efficiency = (tflops / GB10_BF16_TFLOPS) * 100

    intensity, bytes_gb = compute_arithmetic_intensity(m, n, k, 2.0)
    mem_bound = is_memory_bound(intensity, GB10_BF16_TFLOPS)

    del A, B, C
    reset_memory()

    return BenchmarkResult(
        name="BF16",
        m=m, n=n, k=k,
        time_ms=time_ms,
        tflops=tflops,
        efficiency=efficiency,
        bytes_accessed=bytes_gb,
        arithmetic_intensity=intensity,
        memory_bound=mem_bound
    )


def benchmark_matmul_nvfp4(m: int, n: int, k: int, num_warmup: int = 10, num_iters: int = 100) -> Optional[BenchmarkResult]:
    """
    Benchmark NVFP4 matmul using TransformerEngine.

    Uses te.Linear with NVFP4BlockScaling recipe.
    """
    if not TE_AVAILABLE:
        return None

    reset_memory()

    try:
        # Create recipe with disable_rht=True (required for SM121/GB10)
        recipe = TE_NVFP4BlockScaling(
            fp4_format=getattr(TE_Format, "E2M1", None),
            disable_rht=True,
        )

        # Create TE Linear layer (in_features=k, out_features=n)
        # Input will be (m, k), output will be (m, n)
        linear = te.Linear(k, n, bias=False).cuda().to(torch.bfloat16)

        # Input tensor
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")

        # Warmup with autocast
        for _ in range(num_warmup):
            with te.autocast(enabled=True, recipe=recipe):
                y = linear(x)
        torch.cuda.synchronize()

        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iters):
            with te.autocast(enabled=True, recipe=recipe):
                y = linear(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        time_ms = (elapsed / num_iters) * 1000
        flops = 2 * m * n * k
        tflops = (flops / (time_ms / 1000)) / 1e12
        efficiency = (tflops / GB10_NVFP4_TFLOPS) * 100

        # FP4 is 0.5 bytes per element (4 bits)
        intensity, bytes_gb = compute_arithmetic_intensity(m, n, k, 0.5)
        mem_bound = is_memory_bound(intensity, GB10_NVFP4_TFLOPS)

        del linear, x, y
        reset_memory()

        return BenchmarkResult(
            name="NVFP4",
            m=m, n=n, k=k,
            time_ms=time_ms,
            tflops=tflops,
            efficiency=efficiency,
            bytes_accessed=bytes_gb,
            arithmetic_intensity=intensity,
            memory_bound=mem_bound
        )

    except Exception as e:
        print(f"  NVFP4 benchmark failed: {e}")
        return None


def print_header():
    """Print benchmark header."""
    print("=" * 90)
    print("GB10 NVFP4 vs BF16 vs FP32 Matmul Benchmark")
    print("=" * 90)
    print()
    print("Hardware Theoretical Peaks:")
    print(f"  FP32:  {GB10_FP32_TFLOPS:6.1f} TFLOPS")
    print(f"  BF16:  {GB10_BF16_TFLOPS:6.1f} TFLOPS")
    print(f"  NVFP4: {GB10_NVFP4_TFLOPS:6.1f} TFLOPS (dense)")
    print(f"  Memory Bandwidth: {GB10_BANDWIDTH_GBS} GB/s")
    print()

    # Calculate arithmetic intensity thresholds
    fp32_thresh = (GB10_FP32_TFLOPS * 1e12) / (GB10_BANDWIDTH_GBS * 1e9)
    bf16_thresh = (GB10_BF16_TFLOPS * 1e12) / (GB10_BANDWIDTH_GBS * 1e9)
    fp4_thresh = (GB10_NVFP4_TFLOPS * 1e12) / (GB10_BANDWIDTH_GBS * 1e9)

    print("Compute-bound thresholds (FLOPs/Byte):")
    print(f"  FP32:  > {fp32_thresh:6.0f} FLOPs/Byte")
    print(f"  BF16:  > {bf16_thresh:6.0f} FLOPs/Byte")
    print(f"  NVFP4: > {fp4_thresh:6.0f} FLOPs/Byte")
    print()


def print_results_table(results: List[BenchmarkResult], title: str):
    """Print results in a formatted table."""
    print(f"\n{title}")
    print("-" * 90)
    print(f"{'Precision':<8} {'M':<8} {'N':<8} {'K':<8} {'Time(ms)':<10} {'TFLOPS':<10} {'Eff%':<8} {'Bound':<10}")
    print("-" * 90)

    for r in results:
        bound_str = "Memory" if r.memory_bound else "Compute"
        print(f"{r.name:<8} {r.m:<8} {r.n:<8} {r.k:<8} {r.time_ms:<10.3f} {r.tflops:<10.2f} {r.efficiency:<8.1f} {bound_str:<10}")


def print_comparison(fp32: BenchmarkResult, bf16: BenchmarkResult, nvfp4: Optional[BenchmarkResult]):
    """Print speedup comparison."""
    print()
    print("Speedup vs FP32:")
    print(f"  BF16:  {fp32.time_ms / bf16.time_ms:.2f}x faster")
    if nvfp4:
        print(f"  NVFP4: {fp32.time_ms / nvfp4.time_ms:.2f}x faster")

    print()
    print("Speedup vs BF16:")
    if nvfp4:
        print(f"  NVFP4: {bf16.time_ms / nvfp4.time_ms:.2f}x faster")
    else:
        print("  NVFP4: N/A")


def run_benchmark_suite():
    """Run the full benchmark suite."""
    print_header()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    device_name = torch.cuda.get_device_name()
    print(f"Device: {device_name}")
    print(f"TransformerEngine available: {TE_AVAILABLE}")
    print()

    # Define matrix sizes to test
    # (m, n, k) - typical transformer sizes
    sizes = [
        # Small: attention projections
        (128, 1280, 1280),      # Small batch, hidden dim
        (512, 1280, 1280),      # Medium batch

        # Medium: MLP layers
        (128, 5120, 1280),      # Small batch, MLP up-projection
        (512, 5120, 1280),      # Medium batch
        (2048, 5120, 1280),     # Large batch

        # Large: vocabulary projection
        (128, 65536, 1280),     # Small batch, vocab projection
        (512, 65536, 1280),     # Medium batch

        # Very large (compute should dominate)
        (4096, 4096, 4096),     # Square large
        (8192, 8192, 4096),     # Very large
        (16384, 4096, 4096),    # Even larger batch
    ]

    all_results = []

    for m, n, k in sizes:
        print(f"\nBenchmarking: M={m}, N={n}, K={k}")
        print(f"  FLOPs: {2*m*n*k/1e9:.2f} GFLOPs")

        # Run benchmarks
        fp32_result = benchmark_matmul_fp32(m, n, k)
        print(f"  FP32:  {fp32_result.time_ms:.3f}ms, {fp32_result.tflops:.2f} TFLOPS ({fp32_result.efficiency:.1f}% eff)")

        bf16_result = benchmark_matmul_bf16(m, n, k)
        print(f"  BF16:  {bf16_result.time_ms:.3f}ms, {bf16_result.tflops:.2f} TFLOPS ({bf16_result.efficiency:.1f}% eff)")

        nvfp4_result = benchmark_matmul_nvfp4(m, n, k)
        if nvfp4_result:
            print(f"  NVFP4: {nvfp4_result.time_ms:.3f}ms, {nvfp4_result.tflops:.2f} TFLOPS ({nvfp4_result.efficiency:.1f}% eff)")
        else:
            print(f"  NVFP4: Skipped (not available)")

        all_results.append((fp32_result, bf16_result, nvfp4_result))

    # Print summary tables
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    # Group by size category
    small_sizes = all_results[:2]
    medium_sizes = all_results[2:5]
    large_sizes = all_results[5:7]
    very_large_sizes = all_results[7:]

    # Print comparison table
    print("\n--- Full Results ---")
    print("-" * 110)
    print(f"{'M':<7} {'N':<7} {'K':<7} | {'FP32 (ms)':<12} {'BF16 (ms)':<12} {'NVFP4 (ms)':<12} | {'BF16/FP32':<10} {'NVFP4/BF16':<10}")
    print("-" * 110)

    for fp32, bf16, nvfp4 in all_results:
        nvfp4_ms = f"{nvfp4.time_ms:.3f}" if nvfp4 else "N/A"
        bf16_speedup = fp32.time_ms / bf16.time_ms
        nvfp4_speedup = bf16.time_ms / nvfp4.time_ms if nvfp4 else 0
        nvfp4_speedup_str = f"{nvfp4_speedup:.2f}x" if nvfp4 else "N/A"

        print(f"{fp32.m:<7} {fp32.n:<7} {fp32.k:<7} | {fp32.time_ms:<12.3f} {bf16.time_ms:<12.3f} {nvfp4_ms:<12} | {bf16_speedup:<10.2f}x {nvfp4_speedup_str:<10}")

    # Print TFLOPS achieved table
    print("\n--- TFLOPS Achieved ---")
    print("-" * 110)
    print(f"{'M':<7} {'N':<7} {'K':<7} | {'FP32':<12} {'BF16':<12} {'NVFP4':<12} | {'FP32 Eff':<10} {'BF16 Eff':<10} {'NVFP4 Eff':<10}")
    print("-" * 110)

    for fp32, bf16, nvfp4 in all_results:
        nvfp4_tflops = f"{nvfp4.tflops:.2f}" if nvfp4 else "N/A"
        nvfp4_eff = f"{nvfp4.efficiency:.1f}%" if nvfp4 else "N/A"

        print(f"{fp32.m:<7} {fp32.n:<7} {fp32.k:<7} | {fp32.tflops:<12.2f} {bf16.tflops:<12.2f} {nvfp4_tflops:<12} | {fp32.efficiency:<10.1f}% {bf16.efficiency:<10.1f}% {nvfp4_eff:<10}")

    # Memory bandwidth analysis
    print("\n--- Memory Bandwidth Analysis ---")
    print("-" * 90)
    print(f"{'Size':<25} | {'Intensity':<15} | {'Memory Bound?':<15} | {'Bottleneck':<20}")
    print("-" * 90)

    for fp32, bf16, nvfp4 in all_results:
        size_str = f"{fp32.m}x{fp32.n}x{fp32.k}"

        # Calculate actual bandwidth achieved
        fp32_bw = fp32.bytes_accessed / (fp32.time_ms / 1000)  # GB/s
        bf16_bw = bf16.bytes_accessed / (bf16.time_ms / 1000)  # GB/s

        if fp32.memory_bound:
            bottleneck = f"Memory ({fp32_bw:.0f} GB/s)"
        else:
            bottleneck = f"Compute ({fp32.tflops:.1f} TFLOPS)"

        print(f"{size_str:<25} | {fp32.arithmetic_intensity:<15.0f} | {'Yes' if fp32.memory_bound else 'No':<15} | {bottleneck:<20}")

    # Conclusions
    print("\n" + "=" * 90)
    print("CONCLUSIONS")
    print("=" * 90)

    # Find where NVFP4 wins
    nvfp4_wins = [(fp32, bf16, nvfp4) for fp32, bf16, nvfp4 in all_results
                  if nvfp4 and bf16.time_ms / nvfp4.time_ms > 1.1]

    if nvfp4_wins:
        print("\nNVFP4 shows significant speedup (>1.1x vs BF16) for these sizes:")
        for fp32, bf16, nvfp4 in nvfp4_wins:
            speedup = bf16.time_ms / nvfp4.time_ms
            print(f"  {fp32.m}x{fp32.n}x{fp32.k}: {speedup:.2f}x faster than BF16")
    else:
        print("\nNVFP4 did NOT show significant speedup over BF16 for tested sizes.")
        print("This is likely because:")
        print("  1. GB10 is memory-bandwidth limited (273 GB/s)")
        print("  2. Tested sizes have low arithmetic intensity")
        print("  3. TransformerEngine overhead may dominate for small sizes")

    print("\nRecommendations:")
    print("  - Use larger batch sizes to increase arithmetic intensity")
    print("  - Consider fused kernels to reduce memory traffic")
    print("  - NVFP4 benefits are most visible for very large matrix sizes")
    print()


if __name__ == "__main__":
    run_benchmark_suite()
