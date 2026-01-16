"""
Profile actual nanochat training: BF16 vs NVFP4
Breaks down time by operation to understand why NVFP4 isn't faster than FP32.

Usage:
    python -m scripts.profile_training
"""
import os
import shutil

if not os.environ.get("TRITON_PTXAS_PATH"):
    for ptxas in ["/usr/local/cuda/bin/ptxas", shutil.which("ptxas")]:
        if ptxas and os.path.exists(ptxas):
            os.environ["TRITON_PTXAS_PATH"] = ptxas
            break

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "1"

# Use persistent autotune cache
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".cache", "torchinductor")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = CACHE_DIR

# Suppress autotune noise
os.environ["TORCH_LOGS"] = "-inductor"
import logging
logging.getLogger("torch._inductor.select_algorithm").setLevel(logging.ERROR)

import torch
import time

# Fix is_big_gpu check for GB10
try:
    import torch._inductor.utils as inductor_utils
    inductor_utils.is_big_gpu = lambda index=0: True
except: pass

device = torch.device("cuda")

print("=" * 70)
print("NANOCHAT TRAINING PROFILING: BF16 vs NVFP4")
print("=" * 70)

# Model config - same as typical training
depth = 20
model_dim = 1024
num_heads = 16
vocab_size = 65536
max_seq_len = 1024
B = 8  # Batch size to not OOM
T = max_seq_len

print(f"Model: depth={depth}, dim={model_dim}, heads={num_heads}")
print(f"Batch: B={B}, T={T}")
print(f"Total tokens per step: {B * T:,}")
print()

from nanochat.gpt import GPT, GPTConfig, select_precision, make_autocast_ctx

def profile_precision(precision_name: str):
    print(f"\n{'='*60}")
    print(f"PROFILING: {precision_name.upper()}")
    print(f"{'='*60}")

    precision_plan = select_precision(target=precision_name)
    autocast_ctx = make_autocast_ctx(precision_plan, "cuda")
    print(f"Precision: {precision_plan.name}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Create model
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

    # Compile model
    model = torch.compile(model, dynamic=False)

    param_count = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Parameters: {param_count:,} ({param_bytes/1e6:.1f} MB)")

    # Warmup with compile
    print("Warmup (compiling)...")
    for i in range(3):
        x = torch.randint(0, vocab_size, (B, T), device=device)
        y = torch.randint(0, vocab_size, (B, T), device=device)
        with autocast_ctx():
            loss = model(x, y)
        loss.backward()
        model.zero_grad(set_to_none=True)
        print(f"  Warmup step {i+1}/3 done")
    torch.cuda.synchronize()

    # Benchmark
    print("\nBenchmarking (5 steps)...")
    times = []
    for i in range(5):
        x = torch.randint(0, vocab_size, (B, T), device=device)
        y = torch.randint(0, vocab_size, (B, T), device=device)

        torch.cuda.synchronize()
        start = time.perf_counter()

        with autocast_ctx():
            loss = model(x, y)
        loss.backward()
        model.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        end = time.perf_counter()

        times.append(end - start)
        tok_sec = B * T / (end - start)
        print(f"  Step {i}: {(end-start)*1000:.1f}ms | {tok_sec:,.0f} tok/sec")

    avg_time = sum(times) / len(times)
    avg_tok_sec = B * T / avg_time
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    # Profile with torch.profiler to get breakdown
    print("\nProfiling kernel breakdown...")
    x = torch.randint(0, vocab_size, (B, T), device=device)
    y = torch.randint(0, vocab_size, (B, T), device=device)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_flops=True,
    ) as prof:
        with autocast_ctx():
            loss = model(x, y)
        loss.backward()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

    # Categorize kernels
    events = prof.key_averages()

    # Categorize by operation type
    matmul_time = 0
    attention_time = 0
    norm_time = 0
    other_time = 0

    for e in events:
        # Handle both cuda_time_total and self_cuda_time_total
        t = getattr(e, 'cuda_time_total', getattr(e, 'self_cuda_time_total', 0)) / 1e6  # ms
        key = e.key.lower()
        if any(s in key for s in ['gemm', 'matmul', 'mm_', 'linear', 'cublas']):
            matmul_time += t
        elif any(s in key for s in ['flash', 'attention', 'sdpa', 'softmax']):
            attention_time += t
        elif any(s in key for s in ['norm', 'rms', 'layer_norm']):
            norm_time += t
        else:
            other_time += t

    total_cuda_time = sum(getattr(e, 'cuda_time_total', getattr(e, 'self_cuda_time_total', 0)) for e in events) / 1e6

    print(f"\nKernel breakdown (total CUDA time: {total_cuda_time:.1f}ms):")
    print(f"  Matmul/Linear:  {matmul_time:.1f}ms ({100*matmul_time/total_cuda_time:.1f}%)")
    print(f"  Attention:      {attention_time:.1f}ms ({100*attention_time/total_cuda_time:.1f}%)")
    print(f"  Norm:           {norm_time:.1f}ms ({100*norm_time/total_cuda_time:.1f}%)")
    print(f"  Other:          {other_time:.1f}ms ({100*other_time/total_cuda_time:.1f}%)")

    # Print top kernels
    print(f"\nTop 15 kernels:")
    def get_cuda_time(e):
        return getattr(e, 'cuda_time_total', getattr(e, 'self_cuda_time_total', 0))
    for e in sorted(events, key=get_cuda_time, reverse=True)[:15]:
        t = get_cuda_time(e) / 1e6
        if t > 0:
            pct = 100 * t / total_cuda_time if total_cuda_time > 0 else 0
            print(f"  {t:6.2f}ms ({pct:4.1f}%) {e.key[:60]}")

    del model
    torch.cuda.empty_cache()

    return {
        "precision": precision_name,
        "avg_time_ms": avg_time * 1000,
        "avg_tok_sec": avg_tok_sec,
        "peak_mem_gb": peak_mem,
        "matmul_time_ms": matmul_time,
        "attention_time_ms": attention_time,
        "norm_time_ms": norm_time,
        "other_time_ms": other_time,
        "total_cuda_time_ms": total_cuda_time,
    }

results = []

# Test BF16 first
try:
    results.append(profile_precision("bf16"))
except Exception as e:
    print(f"BF16 ERROR: {e}")
    import traceback
    traceback.print_exc()

# Then test NVFP4
try:
    results.append(profile_precision("nvfp4"))
except Exception as e:
    print(f"NVFP4 ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

for r in results:
    print(f"\n{r['precision'].upper()}:")
    print(f"  Avg time:      {r['avg_time_ms']:.1f} ms")
    print(f"  Tok/sec:       {r['avg_tok_sec']:,.0f}")
    print(f"  Peak memory:   {r['peak_mem_gb']:.2f} GB")
    print(f"  CUDA time:     {r['total_cuda_time_ms']:.1f} ms")
    print(f"    Matmul:      {r['matmul_time_ms']:.1f} ms ({100*r['matmul_time_ms']/r['total_cuda_time_ms']:.1f}%)")
    print(f"    Attention:   {r['attention_time_ms']:.1f} ms ({100*r['attention_time_ms']/r['total_cuda_time_ms']:.1f}%)")
    print(f"    Norm:        {r['norm_time_ms']:.1f} ms ({100*r['norm_time_ms']/r['total_cuda_time_ms']:.1f}%)")

if len(results) == 2:
    bf16, nvfp4 = results[0], results[1]

    print("\n" + "=" * 70)
    print("COMPARISON (NVFP4 vs BF16)")
    print("=" * 70)

    overall_speedup = bf16["avg_time_ms"] / nvfp4["avg_time_ms"]
    print(f"\nOverall: {overall_speedup:.2f}x speedup")

    matmul_speedup = bf16["matmul_time_ms"] / nvfp4["matmul_time_ms"] if nvfp4["matmul_time_ms"] > 0 else 0
    attn_speedup = bf16["attention_time_ms"] / nvfp4["attention_time_ms"] if nvfp4["attention_time_ms"] > 0 else 0

    print(f"\nBy operation:")
    print(f"  Matmul speedup:    {matmul_speedup:.2f}x")
    print(f"  Attention speedup: {attn_speedup:.2f}x")

    # Analysis
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print("=" * 70)

    bf16_matmul_pct = 100 * bf16["matmul_time_ms"] / bf16["total_cuda_time_ms"]
    nvfp4_matmul_pct = 100 * nvfp4["matmul_time_ms"] / nvfp4["total_cuda_time_ms"]

    print(f"\nMatmul fraction of total:")
    print(f"  BF16:  {bf16_matmul_pct:.1f}%")
    print(f"  NVFP4: {nvfp4_matmul_pct:.1f}%")

    if matmul_speedup > 2.0 and overall_speedup < 1.5:
        print("\n*** FINDING: Matmul is much faster but overall speedup is limited ***")
        print("This means non-matmul operations (attention, norms, etc.) dominate!")
        print("NVFP4 only accelerates matmul, not attention kernels.")

    if attn_speedup < 1.1:
        print(f"\n*** FINDING: Attention is NOT faster with NVFP4 ***")
        print("Flash Attention runs in BF16 regardless of precision setting.")
        print("NVFP4 only affects Linear layers (matmuls), not attention.")
