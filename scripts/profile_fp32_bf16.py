"""
Profile FP32 vs BF16 with torch.profiler to understand why they're same speed.
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

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".cache", "torchinductor")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = CACHE_DIR

os.environ["TORCH_LOGS"] = "-inductor"
import logging
logging.getLogger("torch._inductor.select_algorithm").setLevel(logging.ERROR)

import torch
import time

try:
    import torch._inductor.utils as inductor_utils
    inductor_utils.is_big_gpu = lambda index=0: True
except: pass

# Monkey-patch to allow FP32 rotary embeddings
import nanochat.gpt as gpt_module
original_forward = gpt_module.GPT.forward

def patched_forward(self, idx, targets=None, kv_cache=None):
    # Skip the bfloat16 assertion for rotary embeddings
    return original_forward.__wrapped__(self, idx, targets, kv_cache) if hasattr(original_forward, '__wrapped__') else original_forward(self, idx, targets, kv_cache)

device = torch.device("cuda")

print("=" * 70)
print("FP32 vs BF16 PROFILING")
print("=" * 70)

# Model config
depth = 12  # Smaller for faster testing
model_dim = 768
num_heads = 12
vocab_size = 65536
max_seq_len = 1024
B = 8
T = max_seq_len

print(f"Model: d{depth}, dim={model_dim}")
print(f"Batch: B={B}, T={T}")
print()

from nanochat.gpt import GPT, GPTConfig

def profile_dtype(dtype_name, dtype, num_warmup=3, num_iters=5):
    print(f"\n{'='*60}")
    print(f"Profiling: {dtype_name}")
    print(f"{'='*60}")

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
    model.to(dtype=dtype)

    # Also convert rotary embeddings
    model.cos = model.cos.to(dtype=torch.bfloat16)  # Keep rotary in bf16
    model.sin = model.sin.to(dtype=torch.bfloat16)

    print(f"Weight dtype: {next(model.parameters()).dtype}")
    print(f"Rotary dtype: {model.cos.dtype}")

    # Compile
    model = torch.compile(model, dynamic=False)

    # Warmup
    print(f"Warmup ({num_warmup} steps)...")
    for _ in range(num_warmup):
        x = torch.randint(0, vocab_size, (B, T), device=device)
        y = torch.randint(0, vocab_size, (B, T), device=device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(dtype == torch.bfloat16)):
            loss = model(x, y)
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    # Benchmark without profiler first
    print(f"Benchmark ({num_iters} steps)...")
    times = []
    for i in range(num_iters):
        x = torch.randint(0, vocab_size, (B, T), device=device)
        y = torch.randint(0, vocab_size, (B, T), device=device)

        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(dtype == torch.bfloat16)):
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

    # Profile one step
    print("\nProfiling with torch.profiler...")
    x = torch.randint(0, vocab_size, (B, T), device=device)
    y = torch.randint(0, vocab_size, (B, T), device=device)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(dtype == torch.bfloat16)):
            loss = model(x, y)
        loss.backward()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

    # Print top CUDA kernels
    print(f"\nTop 15 CUDA kernels for {dtype_name}:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    # Save trace
    trace_file = f"/tmp/trace_{dtype_name}.json"
    prof.export_chrome_trace(trace_file)
    print(f"Trace saved to: {trace_file}")

    del model
    torch.cuda.empty_cache()

    return {
        "dtype": dtype_name,
        "avg_time_ms": avg_time * 1000,
        "avg_tok_sec": avg_tok_sec,
        "peak_mem_gb": peak_mem,
    }

results = []

# Test FP32
results.append(profile_dtype("FP32", torch.float32))

# Test BF16
results.append(profile_dtype("BF16", torch.bfloat16))

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Dtype':<10} {'Time(ms)':<12} {'Tok/sec':<15} {'Memory(GB)':<12}")
print("-" * 50)

for r in results:
    print(f"{r['dtype']:<10} {r['avg_time_ms']:<12.1f} {r['avg_tok_sec']:<15,.0f} {r['peak_mem_gb']:<12.1f}")

if len(results) == 2:
    fp32, bf16 = results[0], results[1]
    speedup = fp32["avg_time_ms"] / bf16["avg_time_ms"]
    print("=" * 50)
    print(f"\nBF16 speedup over FP32: {speedup:.2f}x")
    if speedup < 1.2:
        print("\nWARNING: BF16 is NOT significantly faster than FP32!")
        print("This suggests something is wrong or workload is memory-bound.")
