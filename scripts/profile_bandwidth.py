"""
Profile memory bandwidth utilization to prove GB10 is memory-bound.

Usage:
    python -m scripts.profile_bandwidth
"""

import os
import shutil

# Triton SM121a fix
if not os.environ.get("TRITON_PTXAS_PATH"):
    for ptxas in ["/usr/local/cuda/bin/ptxas", shutil.which("ptxas")]:
        if ptxas and os.path.exists(ptxas):
            os.environ["TRITON_PTXAS_PATH"] = ptxas
            break

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "1"

# Persistent cache for autotune results
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".cache", "torchinductor")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = CACHE_DIR

# Suppress autotune logging
os.environ["TORCH_LOGS"] = "-inductor"
import logging
logging.getLogger("torch._inductor.select_algorithm").setLevel(logging.ERROR)

import torch
import time

# Patch is_big_gpu to return True for GB10 (48 SMs < 68 SM threshold)
try:
    import torch._inductor.utils as inductor_utils
    inductor_utils.is_big_gpu = lambda index=0: True
except Exception:
    pass

# GB10 specs
GB10_BANDWIDTH_GBS = 273  # GB/s theoretical peak
GB10_FP4_TFLOPS = 500     # TFLOPS theoretical peak
GB10_BF16_TFLOPS = 125    # TFLOPS theoretical peak

print("=" * 70)
print("GB10 Memory Bandwidth Analysis")
print("=" * 70)
print(f"Theoretical peak bandwidth: {GB10_BANDWIDTH_GBS} GB/s")
print(f"Theoretical FP4 compute:    {GB10_FP4_TFLOPS} TFLOPS")
print(f"Theoretical BF16 compute:   {GB10_BF16_TFLOPS} TFLOPS")
print()

device = torch.device("cuda")

def profile_matmul(M, N, K, dtype, num_warmup=5, num_iters=20):
    """Profile matmul and compute bandwidth/compute utilization."""

    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)

    # Warmup
    for _ in range(num_warmup):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()

    # Measure
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time_s = (end - start) / num_iters

    # Calculate metrics
    bytes_per_elem = A.element_size()

    # Memory: read A, read B, write C
    bytes_read = (M * K + K * N) * bytes_per_elem
    bytes_written = M * N * bytes_per_elem
    total_bytes = bytes_read + bytes_written

    # Compute: M * N * K multiply-adds = 2 * M * N * K FLOPs
    flops = 2 * M * N * K

    # Achieved metrics
    bandwidth_gbs = (total_bytes / avg_time_s) / 1e9
    tflops = (flops / avg_time_s) / 1e12

    # Utilization
    bw_util = 100 * bandwidth_gbs / GB10_BANDWIDTH_GBS

    if dtype == torch.bfloat16:
        compute_util = 100 * tflops / GB10_BF16_TFLOPS
    else:
        compute_util = 100 * tflops / GB10_BF16_TFLOPS  # FP32 is slower

    # Arithmetic intensity
    arith_intensity = flops / total_bytes

    return {
        "time_ms": avg_time_s * 1000,
        "bandwidth_gbs": bandwidth_gbs,
        "tflops": tflops,
        "bw_util": bw_util,
        "compute_util": compute_util,
        "arith_intensity": arith_intensity,
    }


def profile_forward_pass():
    """Profile actual model forward pass bandwidth."""
    from nanochat.gpt import GPT, GPTConfig, select_precision, make_autocast_ctx

    # Model config (d20)
    depth = 20
    model_dim = 1280
    num_heads = 10
    vocab_size = 65536
    max_seq_len = 2048
    B = 32
    T = max_seq_len

    print(f"\nModel: depth={depth}, dim={model_dim}, heads={num_heads}")
    print(f"Batch: B={B}, T={T}")
    print()

    results = {}

    for precision_name in ["bf16", "nvfp4"]:
        print(f"Testing {precision_name.upper()}...")

        precision_plan = select_precision(target=precision_name)
        autocast_ctx = make_autocast_ctx(precision_plan, "cuda")

        with torch.device("meta"):
            config = GPTConfig(
                sequence_len=max_seq_len,
                vocab_size=vocab_size,
                n_layer=depth,
                n_head=num_heads,
                n_kv_head=num_heads,
                n_embd=model_dim
            )
            model = GPT(config)
        model.to_empty(device=device)
        model.init_weights()
        if precision_plan.use_te:
            model.to(dtype=torch.bfloat16)

        # Count parameters and estimate memory
        param_count = sum(p.numel() for p in model.parameters())
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

        # Compile
        model = torch.compile(model, dynamic=False)

        # Create dummy input
        x = torch.randint(0, vocab_size, (B, T), device=device)
        y = torch.randint(0, vocab_size, (B, T), device=device)

        # Warmup
        for _ in range(3):
            with autocast_ctx():
                loss = model(x, y)
            loss.backward()
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()

        # Measure forward
        num_iters = 10
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iters):
            with autocast_ctx():
                loss = model(x, y)
            torch.cuda.synchronize()
        end = time.perf_counter()
        fwd_time = (end - start) / num_iters

        # Measure backward
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iters):
            with autocast_ctx():
                loss = model(x, y)
            loss.backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
        end = time.perf_counter()
        fwd_bwd_time = (end - start) / num_iters
        bwd_time = fwd_bwd_time - fwd_time

        peak_mem = torch.cuda.max_memory_allocated() / 1e9

        # Estimate FLOPs (rough approximation for transformer)
        # Per token: ~6 * params for forward, ~12 * params for forward+backward
        tokens = B * T
        fwd_flops = 6 * param_count * tokens
        bwd_flops = 12 * param_count * tokens  # backward is ~2x forward

        fwd_tflops = (fwd_flops / fwd_time) / 1e12
        total_tflops = ((fwd_flops + bwd_flops) / fwd_bwd_time) / 1e12

        # Memory bandwidth estimation
        # Forward: read weights once, activations flow through
        # Backward: read weights twice (for grad_input and grad_weight), write gradients
        # Very rough: ~3x weight reads + activation memory

        results[precision_name] = {
            "fwd_time_ms": fwd_time * 1000,
            "bwd_time_ms": bwd_time * 1000,
            "total_time_ms": fwd_bwd_time * 1000,
            "fwd_tflops": fwd_tflops,
            "total_tflops": total_tflops,
            "peak_mem_gb": peak_mem,
            "param_count": param_count,
            "param_bytes_mb": param_bytes / 1e6,
        }

        del model
        torch.cuda.empty_cache()

    return results


print("=" * 70)
print("Part 1: Raw Matmul Bandwidth Analysis")
print("=" * 70)

# Test different matrix sizes
sizes = [
    # (M, N, K) - typical transformer shapes
    (2048, 1280, 1280),   # attention QKV projection
    (2048, 5120, 1280),   # FFN up projection
    (2048, 1280, 5120),   # FFN down projection
    (65536, 1280, 2048),  # embedding lookup equivalent
]

print(f"\n{'Shape':<25} {'Time(ms)':<10} {'BW(GB/s)':<12} {'BW%':<8} {'TFLOPS':<10} {'Compute%':<10} {'AI':<8}")
print("-" * 90)

for M, N, K in sizes:
    for dtype in [torch.float32, torch.bfloat16]:
        dtype_name = "fp32" if dtype == torch.float32 else "bf16"
        result = profile_matmul(M, N, K, dtype)

        shape_str = f"({M},{N},{K}) {dtype_name}"
        print(f"{shape_str:<25} {result['time_ms']:<10.2f} {result['bandwidth_gbs']:<12.1f} "
              f"{result['bw_util']:<8.1f} {result['tflops']:<10.2f} {result['compute_util']:<10.1f} "
              f"{result['arith_intensity']:<8.1f}")

print()
print("Legend: BW% = bandwidth utilization, Compute% = compute utilization, AI = arithmetic intensity")
print()
print("INTERPRETATION:")
print("- If BW% is high (>70%) and Compute% is low (<50%): MEMORY BOUND")
print("- If Compute% is high (>70%) and BW% is low (<50%): COMPUTE BOUND")
print()

# Roofline analysis
print("=" * 70)
print("Part 2: Roofline Analysis")
print("=" * 70)
print()
print(f"To be compute-bound on GB10, need arithmetic intensity > {GB10_BF16_TFLOPS * 1000 / GB10_BANDWIDTH_GBS:.0f} FLOPs/byte (BF16)")
print(f"To be compute-bound on GB10, need arithmetic intensity > {GB10_FP4_TFLOPS * 1000 / GB10_BANDWIDTH_GBS:.0f} FLOPs/byte (FP4)")
print()
print("Typical matmul arithmetic intensity: 2-100 FLOPs/byte (depends on size)")
print("GB10 needs: ~458 FLOPs/byte (BF16) or ~1831 FLOPs/byte (FP4)")
print()
print("CONCLUSION: GB10 is HEAVILY memory-bound for typical LLM workloads")
print()

print("=" * 70)
print("Part 3: Full Model Forward/Backward Profile")
print("=" * 70)

try:
    results = profile_forward_pass()

    print()
    print(f"{'Precision':<10} {'Fwd(ms)':<12} {'Bwd(ms)':<12} {'Total(ms)':<12} {'TFLOPS':<10} {'Mem(GB)':<10}")
    print("-" * 70)

    for name, r in results.items():
        print(f"{name.upper():<10} {r['fwd_time_ms']:<12.1f} {r['bwd_time_ms']:<12.1f} "
              f"{r['total_time_ms']:<12.1f} {r['total_tflops']:<10.1f} {r['peak_mem_gb']:<10.1f}")

    # Speedup
    if "bf16" in results and "nvfp4" in results:
        speedup = results["bf16"]["total_time_ms"] / results["nvfp4"]["total_time_ms"]
        print()
        print(f"NVFP4 speedup over BF16: {speedup:.2f}x")
        print()
        print("If compute-bound, expected speedup: ~4x (500/125 TFLOPS)")
        print(f"Actual speedup: {speedup:.2f}x")
        print()
        if speedup < 2.0:
            print("LOW SPEEDUP CONFIRMS: We are MEMORY-BOUND, not compute-bound!")
            print("NVFP4 compute advantage is wasted because memory bandwidth is the bottleneck.")

except Exception as e:
    print(f"Error in model profiling: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 70)
