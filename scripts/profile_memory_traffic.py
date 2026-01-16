"""
Profile actual memory traffic for BF16 vs NVFP4 using torch profiler memory stats.
"""
import os
import shutil

if not os.environ.get("TRITON_PTXAS_PATH"):
    for ptxas in ["/usr/local/cuda/bin/ptxas", shutil.which("ptxas")]:
        if ptxas and os.path.exists(ptxas):
            os.environ["TRITON_PTXAS_PATH"] = ptxas
            break

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import time

device = torch.device("cuda")

print("=" * 70)
print("MEMORY TRAFFIC ANALYSIS: BF16 vs NVFP4")
print("=" * 70)

# Smaller model for quick profiling
depth = 12
model_dim = 768
num_heads = 12
vocab_size = 65536
max_seq_len = 1024
B = 4
T = max_seq_len

print(f"Model: d{depth}, dim={model_dim}")
print(f"Batch: B={B}, T={T}")
print()

from nanochat.gpt import GPT, GPTConfig, select_precision, make_autocast_ctx

def profile_memory(precision_name: str):
    print(f"\n{'='*60}")
    print(f"Profiling: {precision_name.upper()}")
    print(f"{'='*60}")

    precision_plan = select_precision(target=precision_name)
    autocast_ctx = make_autocast_ctx(precision_plan, "cuda")
    print(f"Precision: {precision_plan.name}")

    torch.cuda.empty_cache()
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

    # Count parameters and their sizes
    param_count = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Parameters: {param_count:,} ({param_bytes/1e6:.1f} MB)")

    # Warmup without compile
    print("Warmup (no compile)...")
    x = torch.randint(0, vocab_size, (B, T), device=device)
    y = torch.randint(0, vocab_size, (B, T), device=device)

    for _ in range(2):
        with autocast_ctx():
            loss = model(x, y)
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    # Profile with detailed memory tracking
    print("Profiling memory...")
    torch.cuda.reset_peak_memory_stats()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
        with_flops=True,
    ) as prof:
        with autocast_ctx():
            loss = model(x, y)
        loss.backward()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

    # Get memory events
    events = prof.key_averages()

    # Calculate totals
    total_cuda_time = sum(e.cuda_time_total for e in events if e.cuda_time_total > 0)
    total_flops = sum(e.flops for e in events if e.flops > 0)

    # Memory allocated during forward/backward
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    print(f"\nTotal CUDA time: {total_cuda_time/1e6:.1f} ms")
    print(f"Total FLOPs: {total_flops/1e12:.2f} TFLOPS")
    print(f"Peak memory: {peak_mem:.2f} GB")

    # Estimate memory traffic from activation sizes
    # Forward: read weights + write activations
    # Backward: read weights + read activations + write gradients
    # Roughly 3x weight reads + activation traffic

    # Print top memory-consuming ops
    print(f"\nTop ops by CUDA time:")
    for e in sorted(events, key=lambda x: x.cuda_time_total, reverse=True)[:10]:
        if e.cuda_time_total > 0:
            flops_str = f"{e.flops/1e9:.1f}G" if e.flops > 0 else "N/A"
            print(f"  {e.key[:50]:<50} {e.cuda_time_total/1e3:.1f}ms  FLOPs:{flops_str}")

    # Benchmark speed
    print("\nBenchmarking speed...")
    times = []
    for i in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with autocast_ctx():
            loss = model(x, y)
        loss.backward()
        model.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)
        print(f"  Step {i}: {(end-start)*1000:.1f}ms | {B*T/(end-start):,.0f} tok/sec")

    avg_time = sum(times) / len(times)
    avg_tok_sec = B * T / avg_time

    del model
    torch.cuda.empty_cache()

    return {
        "precision": precision_name,
        "avg_time_ms": avg_time * 1000,
        "avg_tok_sec": avg_tok_sec,
        "peak_mem_gb": peak_mem,
        "total_flops": total_flops,
    }

results = []
for p in ["bf16", "nvfp4"]:
    try:
        results.append(profile_memory(p))
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

for r in results:
    print(f"\n{r['precision'].upper()}:")
    print(f"  Time: {r['avg_time_ms']:.1f} ms")
    print(f"  Tok/sec: {r['avg_tok_sec']:,.0f}")
    print(f"  Memory: {r['peak_mem_gb']:.2f} GB")

if len(results) == 2:
    bf16, nvfp4 = results[0], results[1]
    speedup = bf16["avg_time_ms"] / nvfp4["avg_time_ms"]
    print(f"\nNVFP4 speedup: {speedup:.2f}x")
