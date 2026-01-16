"""
Compare BF16 vs NVFP4 training speed - proper comparison.

Usage:
    python -m scripts.compare_bf16_nvfp4
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
print("BF16 vs NVFP4 TRAINING COMPARISON")
print("=" * 70)
print()

# Model config - same as d20 benchmark
depth = 20
model_dim = depth * 64  # 1280
num_heads = 10
vocab_size = 65536
max_seq_len = 2048
B = 32
T = max_seq_len
grad_accum_steps = 4

print(f"Model: depth={depth}, dim={model_dim}, heads={num_heads}")
print(f"Batch: B={B}, T={T}, grad_accum={grad_accum_steps}")
print(f"Tokens per step: {B * T * grad_accum_steps:,}")
print()

from nanochat.gpt import GPT, GPTConfig, select_precision, make_autocast_ctx

def benchmark_precision(precision_name: str, num_warmup=2, num_iters=5):
    """Benchmark a specific precision."""
    print(f"\n{'='*60}")
    print(f"Testing: {precision_name.upper()}")
    print(f"{'='*60}")

    # Select precision
    precision_plan = select_precision(target=precision_name)
    print(f"Precision: {precision_plan.name}")

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
    model.to(dtype=torch.bfloat16)

    # Compile
    model = torch.compile(model, dynamic=False)

    # Warmup
    print(f"Warmup ({num_warmup} steps)...")
    for _ in range(num_warmup):
        for _ in range(grad_accum_steps):
            x = torch.randint(0, vocab_size, (B, T), device=device)
            y = torch.randint(0, vocab_size, (B, T), device=device)
            with autocast_ctx():
                loss = model(x, y)
            (loss / grad_accum_steps).backward()
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

        for _ in range(grad_accum_steps):
            x = torch.randint(0, vocab_size, (B, T), device=device)
            y = torch.randint(0, vocab_size, (B, T), device=device)
            with autocast_ctx():
                loss = model(x, y)
            (loss / grad_accum_steps).backward()
        model.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

        tokens = B * T * grad_accum_steps
        tok_sec = tokens / (end - start)
        print(f"  Step {i}: {(end-start)*1000:.1f}ms | {tok_sec:,.0f} tok/sec")

    avg_time = sum(times) / len(times)
    min_time = min(times)
    avg_tok_sec = B * T * grad_accum_steps / avg_time
    peak_tok_sec = B * T * grad_accum_steps / min_time
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    print(f"\nResults for {precision_name.upper()}:")
    print(f"  Avg time:     {avg_time*1000:.1f} ms")
    print(f"  Min time:     {min_time*1000:.1f} ms")
    print(f"  Avg tok/sec:  {avg_tok_sec:,.0f}")
    print(f"  Peak tok/sec: {peak_tok_sec:,.0f}")
    print(f"  Peak memory:  {peak_mem:.1f} GB")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "precision": precision_name,
        "avg_time_ms": avg_time * 1000,
        "min_time_ms": min_time * 1000,
        "avg_tok_sec": avg_tok_sec,
        "peak_tok_sec": peak_tok_sec,
        "peak_mem_gb": peak_mem,
    }

# Run benchmarks
results = []

for precision in ["bf16", "nvfp4"]:
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
print(f"{'Precision':<10} {'Avg Time(ms)':<14} {'Avg Tok/sec':<15} {'Peak Tok/sec':<15} {'Mem(GB)':<10}")
print("-" * 70)

for r in results:
    print(f"{r['precision'].upper():<10} {r['avg_time_ms']:<14.1f} {r['avg_tok_sec']:<15,.0f} {r['peak_tok_sec']:<15,.0f} {r['peak_mem_gb']:<10.1f}")

if len(results) == 2:
    bf16, nvfp4 = results[0], results[1]
    speedup_avg = bf16["avg_time_ms"] / nvfp4["avg_time_ms"]
    speedup_peak = bf16["min_time_ms"] / nvfp4["min_time_ms"]

    print("=" * 70)
    print(f"\nNVFP4 vs BF16:")
    print(f"  Average speedup: {speedup_avg:.2f}x")
    print(f"  Peak speedup:    {speedup_peak:.2f}x")
    print(f"  Memory savings:  {100*(bf16['peak_mem_gb']-nvfp4['peak_mem_gb'])/bf16['peak_mem_gb']:.1f}%")

    if speedup_avg > 1.5:
        print("\n  NVFP4 is significantly faster - good!")
    elif speedup_avg > 1.1:
        print("\n  NVFP4 is somewhat faster (~15-50% speedup)")
        print("  This is expected for memory-bound workloads on GB10")
    else:
        print("\n  WARNING: NVFP4 speedup is minimal!")
        print("  Something may be wrong with NVFP4 configuration")

print()
