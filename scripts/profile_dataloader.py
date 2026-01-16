"""
Profile the dataloader to find bottlenecks.

Usage:
    python -m scripts.profile_dataloader
"""

import time
import torch

from nanochat.common import print_banner, autodetect_device_type
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state

print_banner()

device_type = autodetect_device_type()
device = torch.device(device_type)

B = 32  # batch size
T = 2048  # sequence length
num_iterations = 20

print(f"Profiling dataloader: B={B}, T={T}, device={device}")
print(f"Running {num_iterations} iterations...\n")

# Create dataloader
loader = tokenizing_distributed_data_loader_with_state(
    B, T, split="train", device=device,
    tokenizer_threads=32, tokenizer_batch_size=512
)

# Warmup
print("Warmup...")
x, y, _ = next(loader)
torch.cuda.synchronize() if device_type == "cuda" else None

# Profile
times = []
print("Profiling...")
for i in range(num_iterations):
    torch.cuda.synchronize() if device_type == "cuda" else None
    t0 = time.perf_counter()

    x, y, _ = next(loader)

    torch.cuda.synchronize() if device_type == "cuda" else None
    t1 = time.perf_counter()

    dt = (t1 - t0) * 1000  # ms
    times.append(dt)
    tokens = B * T
    tok_per_sec = tokens / (t1 - t0)
    print(f"  iter {i:2d}: {dt:8.2f}ms | {tok_per_sec:,.0f} tok/sec")

print()
avg_time = sum(times) / len(times)
avg_tokens_per_sec = (B * T) / (avg_time / 1000)
print(f"Average: {avg_time:.2f}ms per batch | {avg_tokens_per_sec:,.0f} tok/sec")
print()

# Now compare with what a training step needs
# With grad_accum_steps=8, we need 8 batches per optimizer step
grad_accum = 8
total_batch_time = avg_time * grad_accum
print(f"With grad_accum={grad_accum}: {total_batch_time:.2f}ms for data loading per optimizer step")
print()

# Estimate GPU compute time (rough)
# At 6% MFU, if data loading is the bottleneck, GPU compute should be much faster
print("If MFU is ~6%, that suggests:")
print(f"  - Data loading takes: ~{total_batch_time:.0f}ms")
print(f"  - GPU compute takes:  ~{total_batch_time * 0.06:.0f}ms (estimated)")
print(f"  - Total step time:    ~{total_batch_time * 1.06:.0f}ms")
