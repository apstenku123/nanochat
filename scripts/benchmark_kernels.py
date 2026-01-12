"""
Benchmark different kernel backends for nanochat training.

Usage:
    python -m scripts.benchmark_kernels

Compares:
- current: PyTorch native operations
- liger: Liger-Kernel optimized Triton kernels
- triton: Unsloth-style kernels (same as liger for now)
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

# GB10 SM count fix: Force max_autotune_gemm before torch import
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "1"

# Persistent cache for autotune results (survives reboot)
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".cache", "torchinductor")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = CACHE_DIR

# Suppress verbose autotune errors (some triton configs exceed GB10's 101KB shared memory)
os.environ["TORCH_LOGS"] = "-inductor"  # Disable inductor logs
import logging
logging.getLogger("torch._inductor.select_algorithm").setLevel(logging.ERROR)

import time
import warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

import torch
import gc

# Fix Liger-Kernel graph breaks: LigerFusedLinearCrossEntropy calls .item() internally
# which causes torch.compile graph breaks. This config enables capturing scalar outputs.
torch._dynamo.config.capture_scalar_outputs = True

# Patch is_big_gpu to return True for GB10 (48 SMs < 68 SM threshold)
try:
    import torch._inductor.utils as inductor_utils
    inductor_utils.is_big_gpu = lambda index=0: True
except Exception:
    pass

from nanochat.common import print_banner, autodetect_device_type, compute_init, compute_cleanup
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state
from nanochat.gpt import GPT, GPTConfig, select_precision, make_autocast_ctx
from nanochat import kernels

print_banner()

device_type = autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

# TF32 precision - use only the new API
if device_type == "cuda":
    torch.set_float32_matmul_precision('high')

# Model config (same as d20)
depth = 20
model_dim = depth * 64
num_heads = 10
vocab_size = 65536
max_seq_len = 2048

# Training config
B = 32  # device batch size
T = max_seq_len
grad_accum_steps = 8
num_warmup = 2
num_benchmark = 5

print(f"Model: depth={depth}, dim={model_dim}, heads={num_heads}, vocab={vocab_size}")
print(f"Training: B={B}, T={T}, grad_accum={grad_accum_steps}")
print(f"Benchmark: {num_warmup} warmup + {num_benchmark} timed steps")
print()

# Precision
precision_plan = select_precision(target="nvfp4")
print(f"Precision: {precision_plan.name}")
autocast_ctx = make_autocast_ctx(precision_plan, device_type)

def get_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0

def reset_memory():
    """Reset memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()

def benchmark_kernel(backend: str):
    """Benchmark a specific kernel backend."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {backend.upper()}")
    print(f"{'='*60}")

    # Set kernel backend
    kernels.set_kernel_backend(backend)
    reset_memory()

    # Create fresh model
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

    # Compile model
    model = torch.compile(model, dynamic=False)

    # Create dataloader
    loader = tokenizing_distributed_data_loader_with_state(
        B, T, split="train", device=device
    )

    # Warmup (includes compilation)
    print(f"Warmup ({num_warmup} steps)...")
    for i in range(num_warmup):
        for micro_step in range(grad_accum_steps):
            x, y, _ = next(loader)
            with autocast_ctx():
                loss = model(x, y)
            (loss / grad_accum_steps).backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    # Reset memory after warmup
    reset_memory()

    # Benchmark
    print(f"Benchmarking ({num_benchmark} steps)...")
    times = []
    for step in range(num_benchmark):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        for micro_step in range(grad_accum_steps):
            x, y, _ = next(loader)
            with autocast_ctx():
                loss = model(x, y)
            (loss / grad_accum_steps).backward()
        model.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

        tokens = B * T * grad_accum_steps
        tok_per_sec = tokens / (t1 - t0)
        print(f"  Step {step}: {(t1-t0)*1000:.1f}ms | {tok_per_sec:,.0f} tok/sec")

    # Results
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    tokens_per_step = B * T * grad_accum_steps
    avg_tok_sec = tokens_per_step / avg_time
    peak_memory = get_memory_mb()

    print(f"\nResults for {backend.upper()}:")
    print(f"  Avg time:     {avg_time*1000:.1f} ms/step")
    print(f"  Min time:     {min_time*1000:.1f} ms/step")
    print(f"  Max time:     {max_time*1000:.1f} ms/step")
    print(f"  Avg tok/sec:  {avg_tok_sec:,.0f}")
    print(f"  Peak memory:  {peak_memory:,.0f} MB")

    # Cleanup
    del model, loader
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "backend": backend,
        "avg_time_ms": avg_time * 1000,
        "min_time_ms": min_time * 1000,
        "max_time_ms": max_time * 1000,
        "avg_tok_sec": avg_tok_sec,
        "peak_memory_mb": peak_memory,
    }

# Run benchmarks
results = []

# Check if Liger is available
if not kernels.LIGER_AVAILABLE:
    print("WARNING: Liger-Kernel not available, skipping liger/triton benchmarks")
    backends = ["current"]
else:
    backends = ["current", "liger", "triton"]

for backend in backends:
    try:
        result = benchmark_kernel(backend)
        results.append(result)
    except Exception as e:
        print(f"ERROR benchmarking {backend}: {e}")
        import traceback
        traceback.print_exc()

# Summary
print("\n" + "="*70)
print("BENCHMARK SUMMARY")
print("="*70)
print(f"{'Backend':<10} {'Time (ms)':<12} {'Tok/sec':<15} {'Memory (MB)':<12} {'Speedup':<10}")
print("-"*70)

baseline_time = results[0]["avg_time_ms"] if results else 1

for r in results:
    speedup = baseline_time / r["avg_time_ms"]
    print(f"{r['backend']:<10} {r['avg_time_ms']:<12.1f} {r['avg_tok_sec']:<15,.0f} {r['peak_memory_mb']:<12,.0f} {speedup:<10.2f}x")

print("="*70)

# Memory savings
if len(results) >= 2:
    mem_current = results[0]["peak_memory_mb"]
    mem_liger = results[1]["peak_memory_mb"]
    mem_savings = (mem_current - mem_liger) / mem_current * 100
    print(f"\nMemory savings with Liger: {mem_savings:.1f}%")

compute_cleanup()
