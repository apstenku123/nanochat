"""
Compare FP32 vs BF16 vs NVFP4 training speed directly.

Usage:
    python -m scripts.compare_precision
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

# Persistent cache
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".cache", "torchinductor")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = CACHE_DIR

# Suppress logs
os.environ["TORCH_LOGS"] = "-inductor"
import logging
logging.getLogger("torch._inductor.select_algorithm").setLevel(logging.ERROR)

import torch
import time
import gc

# Patch is_big_gpu
try:
    import torch._inductor.utils as inductor_utils
    inductor_utils.is_big_gpu = lambda index=0: True
except Exception:
    pass

torch.set_float32_matmul_precision('high')

device = torch.device("cuda")

print("=" * 70)
print("PRECISION COMPARISON: FP32 vs BF16 vs NVFP4")
print("=" * 70)
print()

# Model config (smaller for faster testing)
depth = 12
model_dim = 768
num_heads = 12
vocab_size = 65536
max_seq_len = 1024
B = 16
T = max_seq_len

print(f"Model: depth={depth}, dim={model_dim}, heads={num_heads}")
print(f"Batch: B={B}, T={T}")
print()

from nanochat.gpt import GPT, GPTConfig, select_precision, make_autocast_ctx

def benchmark_precision(precision_name: str, num_warmup=3, num_iters=5):
    """Benchmark a specific precision."""
    print(f"\n{'='*60}")
    print(f"Testing: {precision_name.upper()}")
    print(f"{'='*60}")

    # Select precision
    precision_plan = select_precision(target=precision_name)
    print(f"Precision plan: {precision_plan.name}")
    print(f"  use_te: {precision_plan.use_te}")
    print(f"  recipe: {precision_plan.recipe}")

    autocast_ctx = make_autocast_ctx(precision_plan, "cuda")

    # Create model
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

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

    # Set dtype based on precision
    if precision_name == "fp32":
        model.to(dtype=torch.float32)
    else:
        model.to(dtype=torch.bfloat16)

    # Print actual weight dtype
    sample_weight = next(model.parameters())
    print(f"  weight dtype: {sample_weight.dtype}")

    # Compile
    model = torch.compile(model, dynamic=False)

    # Create input
    x = torch.randint(0, vocab_size, (B, T), device=device)
    y = torch.randint(0, vocab_size, (B, T), device=device)

    # Warmup
    print(f"Warmup ({num_warmup} steps)...")
    for _ in range(num_warmup):
        with autocast_ctx():
            loss = model(x, y)
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    # Reset memory
    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    print(f"Benchmark ({num_iters} steps)...")
    times = []
    for i in range(num_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with autocast_ctx():
            loss = model(x, y)
        loss.backward()
        model.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

        tokens = B * T
        tok_sec = tokens / (end - start)
        print(f"  Step {i}: {(end-start)*1000:.1f}ms | {tok_sec:,.0f} tok/sec | loss={loss.item():.4f}")

    avg_time = sum(times) / len(times)
    avg_tok_sec = B * T / avg_time
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    print(f"\nResults for {precision_name.upper()}:")
    print(f"  Avg time:    {avg_time*1000:.1f} ms")
    print(f"  Avg tok/sec: {avg_tok_sec:,.0f}")
    print(f"  Peak memory: {peak_mem:.1f} GB")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "precision": precision_name,
        "avg_time_ms": avg_time * 1000,
        "avg_tok_sec": avg_tok_sec,
        "peak_mem_gb": peak_mem,
    }

# Run benchmarks
results = []

for precision in ["fp32", "bf16", "nvfp4"]:
    try:
        result = benchmark_precision(precision)
        results.append(result)
    except Exception as e:
        print(f"ERROR with {precision}: {e}")
        import traceback
        traceback.print_exc()

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Precision':<10} {'Time(ms)':<12} {'Tok/sec':<15} {'Memory(GB)':<12} {'Speedup':<10}")
print("-" * 70)

baseline = results[0]["avg_time_ms"] if results else 1

for r in results:
    speedup = baseline / r["avg_time_ms"]
    print(f"{r['precision'].upper():<10} {r['avg_time_ms']:<12.1f} {r['avg_tok_sec']:<15,.0f} {r['peak_mem_gb']:<12.1f} {speedup:<10.2f}x")

print("=" * 70)

if len(results) >= 2:
    bf16_speedup = results[0]["avg_time_ms"] / results[1]["avg_time_ms"]
    print(f"\nBF16 speedup over FP32: {bf16_speedup:.2f}x")

if len(results) >= 3:
    nvfp4_speedup = results[0]["avg_time_ms"] / results[2]["avg_time_ms"]
    nvfp4_vs_bf16 = results[1]["avg_time_ms"] / results[2]["avg_time_ms"]
    print(f"NVFP4 speedup over FP32: {nvfp4_speedup:.2f}x")
    print(f"NVFP4 speedup over BF16: {nvfp4_vs_bf16:.2f}x")

print()
