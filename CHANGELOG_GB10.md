# GB10 / DGX Spark Changelog

This document tracks all changes and fixes made for running nanochat on NVIDIA GB10 (DGX Spark).

## Hardware Specs: NVIDIA GB10

- **GPU Architecture**: Blackwell (SM121a - desktop variant)
- **SMs**: 48 (vs 68 required by PyTorch's is_big_gpu check)
- **Memory**: 128 GB unified LPDDR5x
- **Memory Bandwidth**: 273 GB/s (vs 3,350 GB/s on H100)
- **Compute**:
  - FP32: 31 TFLOPS
  - BF16: ~62 TFLOPS
  - NVFP4: 500 TFLOPS dense, 1000 TFLOPS with 2:4 sparsity
- **CUDA Compute Capability**: 12.1 (sm_121a)

---

## Fixes Applied

### 1. Triton PTXAS SM121a Fix

**Problem**: Triton bundles ptxas 12.8 which doesn't support SM121a (GB10).
**Error**: `ptxas fatal: Value 'sm_121a' is not defined for option 'gpu-name'`

**Fix** (in `scripts/base_train.py`):
```python
if not os.environ.get("TRITON_PTXAS_PATH"):
    for ptxas in ["/usr/local/cuda/bin/ptxas", shutil.which("ptxas")]:
        if ptxas and os.path.exists(ptxas):
            os.environ["TRITON_PTXAS_PATH"] = ptxas
            break
```

**Requirement**: CUDA 13.0+ must be installed on system.

---

### 2. PyTorch is_big_gpu / Max Autotune Fix

**Problem**: PyTorch's `torch._inductor.utils.is_big_gpu()` requires 68 SMs minimum, but GB10 has 48 SMs.
**Warning**: `Not enough SMs to use max_autotune_gemm mode`

**Fix** (in `scripts/base_train.py`):
```python
# Before torch import - force max_autotune_gemm
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "1"

import torch

# After torch import - patch is_big_gpu to return True
import torch._inductor.utils as inductor_utils
inductor_utils.is_big_gpu = lambda index=0: True
```

**Result**: Autotune now works on GB10, testing multiple kernel configurations.

**Note**: Some Triton configs may fail with shared memory errors - this is expected and autotune handles it gracefully.

---

### 3. MFU Calculation Fix

**Problem**: MFU (Model FLOP Utilization) was calculated against H100's 989 TFLOPS, showing ~6% MFU on GB10.

**Fix** (in `scripts/base_train.py`):
```python
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
if "GB10" in gpu_name:
    # GB10: use NVFP4 peak if using TE precision, else BF16
    promised_flops = 500e12 if precision_plan.use_te else 62e12
else:
    # Default to H100 dense BF16
    promised_flops = 495e12
```

**Result**: MFU now shows ~11% (correct for GB10 with NVFP4).

---

### 4. Memory Allocator Config

**Fix** (in `scripts/base_train.py`):
```python
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
```

Helps with memory fragmentation on unified memory systems.

---

### 5. Liger-Kernel Graph Break Fix

**Problem**: Liger-Kernel's `LigerFusedLinearCrossEntropyFunction` calls `.item()` internally to count non-ignored tokens, causing torch.compile graph breaks and performance degradation.

**Error/Warning**:
```
Graph break from `Tensor.item()`, consider setting:
    torch._dynamo.config.capture_scalar_outputs = True
```

**Root Cause**: In `liger_kernel/ops/fused_linear_cross_entropy.py` line 75:
```python
total_n_non_ignore = target_mask.sum().item()
```

**Fix** (in `scripts/base_train.py` and `scripts/benchmark_kernels.py`):
```python
import torch
torch._dynamo.config.capture_scalar_outputs = True
```

**Alternative**: Environment variable `TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1`

**Result**: Eliminates graph breaks, enabling full torch.compile optimization with Liger kernels.

**Benchmark Results** (GB10, depth=20, B=32, T=2048, grad_accum=8):

| Backend | Before Fix | After Fix | Speedup | Memory |
|---------|-----------|-----------|---------|--------|
| current | 14,409 tok/sec | 14,455 tok/sec | 1.00x | 68,595 MB |
| liger | 12,095 tok/sec | 14,202 tok/sec | 0.98x | 65,939 MB |
| triton | 14,277 tok/sec | **16,950 tok/sec** | **1.17x** | 65,939 MB |

**Memory savings**: 3.9% (2.6 GB) with Liger/Triton backends.

**Reference**: [PyTorch torch.compile Troubleshooting](https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html)

---

## New Features Added

### 1. Kernel Backend Selection (`--kernel` parameter)

Added support for optimized Triton kernels via Liger-Kernel.

**Usage**:
```bash
python -m scripts.base_train --kernel=liger  # Use Liger-Kernel
python -m scripts.base_train --kernel=current  # PyTorch native (default)
python -m scripts.base_train --kernel=triton  # Same as liger
```

**Files added**:
- `nanochat/kernels.py` - Kernel backend abstraction

**Key optimization**: FusedLinearCrossEntropy avoids materializing the huge logits tensor (B×T×V), saving ~60% memory.

---

### 2. Tokenizer CLI Parameters

Added CLI parameters for tokenizer tuning:
```bash
--tokenizer_threads=32  # Number of threads for tokenization (default: 4)
--tokenizer_batch_size=512  # Batch size for tokenization (default: 128)
```

---

### 3. Training Iteration Calculation

**Note**: `--target_param_data_ratio` default is 8, but Chinchilla optimal is 20.

```bash
--target_param_data_ratio=20  # Chinchilla optimal, gives ~20,000 iterations
--target_param_data_ratio=8   # Default, gives ~8,560 iterations
```

---

## Performance Analysis

### Why NVFP4 Doesn't Give Expected Speedup

**Expected**: NVFP4 (500 TFLOPS) vs BF16 (62 TFLOPS) = ~8x speedup
**Actual**: Similar tok/sec for both

**Root Cause**: GB10 is **memory-bandwidth limited** (273 GB/s), not compute-limited.

**Profiling Results** (`scripts/profile_training.py`):
```
Data loading:  0.3% of time
Forward:      33% of time
Backward:     67% of time
```

The GPU is working at near full capacity relative to its memory bandwidth. Lower precision reduces compute time, but memory access time dominates.

**Analysis script**: `scripts/analyze_bandwidth.py`

---

## Files Modified

| File | Changes |
|------|---------|
| `scripts/base_train.py` | SM121a fix, is_big_gpu patch, MFU fix, kernel backend, tokenizer params |
| `nanochat/gpt.py` | Kernel backend integration, fused cross-entropy support |
| `nanochat/kernels.py` | New file - kernel backend abstraction |
| `nanochat/dataloader.py` | Binary dataloader for pre-tokenized data (optional) |

---

## Scripts Added

| Script | Purpose |
|--------|---------|
| `scripts/benchmark_kernels.py` | Benchmark current/liger/triton backends |
| `scripts/profile_training.py` | Profile training step breakdown |
| `scripts/profile_dataloader.py` | Profile dataloader performance |
| `scripts/analyze_bandwidth.py` | Analyze memory bandwidth vs compute |
| `scripts/pretokenize.py` | Pre-tokenize dataset to binary (optional) |

---

## Recommended Run Configuration

```bash
#!/bin/bash
# run_train.sh for GB10

cd "$(dirname "$0")"
source .venv/bin/activate

nohup env PYTHONUNBUFFERED=1 python -m scripts.base_train \
    --depth=20 \
    --precision=nvfp4 \
    --save_every=2000 \
    --target_param_data_ratio=20 \
    --kernel=triton \
    > report.log 2>&1 &

echo $! > run.pid
echo "Training started with PID: $(cat run.pid)"
```

---

## Known Issues

1. **PyTorch SM121a Support**: Official PyTorch doesn't list sm_121a in `torch.cuda.get_arch_list()`, but JIT compilation works via compute_120 fallback.

2. **NVFP4 on GB10**: FIXED! Requires `disable_rht=True` in recipe.

   **Problem**: TransformerEngine's Random Hadamard Transform (RHT) crashes on SM120/SM121 (Blackwell consumer GPUs).

   **Error**: `hadamard_transform_cast_fusion.cu` fails with `CUDA Error: invalid argument` when M > 32.

   **Fix** (already applied in `nanochat/gpt.py`):
   ```python
   recipe = NVFP4BlockScaling(
       disable_rht=True,  # Required for GB10/SM121
       fp4_format=Format.E2M1,
       override_linear_precision=(False, False, True),  # WGrad in BF16
   )
   ```

   **Benchmark Results** (with fix applied):
   | Size | BF16 | NVFP4 | Speedup |
   |------|------|-------|---------|
   | 4096³ | 87 TFLOPS | 117 TFLOPS | **1.34x** |
   | 8192×4096² | 96 TFLOPS | 118 TFLOPS | **1.23x** |
   | 16384×4096² | 98 TFLOPS | 133 TFLOPS | **1.36x** |

   **Reference**: [GitHub Issue #2372](https://github.com/NVIDIA/TransformerEngine/issues/2372)

3. **Memory Bandwidth**: GB10's 273 GB/s bandwidth limits training speed regardless of compute precision.

---

## References

- [nanochat GB10 Discussion](https://github.com/karpathy/nanochat/discussions/28)
- [PyTorch SM121 Forum](https://discuss.pytorch.org/t/dgx-spark-gb10-cuda-13-0-python-3-12-sm-121/223744)
- [GB10 Performance Metrics](https://forums.developer.nvidia.com/t/detailed-compute-performance-metrics-for-dgx-spark/351993)
- [Liger-Kernel GitHub](https://github.com/linkedin/Liger-Kernel)
