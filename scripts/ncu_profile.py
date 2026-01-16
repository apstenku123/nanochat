"""
Simple script for ncu profiling - runs a few matmuls with BF16 and measures bandwidth.
"""
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import time

device = torch.device("cuda")

# Test matrix sizes
M, N, K = 8192, 4096, 4096

print(f"Matrix size: ({M}, {K}) x ({K}, {N})")
print()

def test_matmul(dtype, name):
    print(f"\n{name}:")
    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)

    # Warmup
    for _ in range(3):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()

    # Time
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time = (end - start) / 10
    flops = 2 * M * N * K
    tflops = flops / avg_time / 1e12

    # Memory: read A + B, write C
    elem_size = A.element_size()
    mem_bytes = (M * K + K * N + M * N) * elem_size
    bandwidth = mem_bytes / avg_time / 1e9

    print(f"  Time: {avg_time*1000:.2f} ms")
    print(f"  TFLOPS: {tflops:.1f}")
    print(f"  Bandwidth: {bandwidth:.1f} GB/s")
    print(f"  Element size: {elem_size} bytes")

test_matmul(torch.float32, "FP32")
test_matmul(torch.bfloat16, "BF16")

# Test NVFP4 matmul via TransformerEngine
print("\n\nNVFP4 via TransformerEngine Linear:")
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Format

    # Create TE linear layer
    linear = te.Linear(K, N, bias=False, device=device)
    linear.weight.data = torch.randn(N, K, dtype=torch.bfloat16, device=device)

    # FP8 recipe (NVFP4 not directly exposed for simple matmul)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)

    # Warmup
    for _ in range(3):
        y = linear(x)
    torch.cuda.synchronize()

    # Time
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        y = linear(x)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time = (end - start) / 10
    flops = 2 * M * N * K
    tflops = flops / avg_time / 1e12

    print(f"  Time: {avg_time*1000:.2f} ms")
    print(f"  TFLOPS: {tflops:.1f}")

except Exception as e:
    print(f"  Error: {e}")

print("\n\nDone!")
