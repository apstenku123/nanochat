# Claude Code Instructions for nanochat

## GPU Performance Testing Rules

**ALWAYS use nsys profiler when testing GPU performance** to verify the correct kernels are being used:

```bash
nsys profile --stats=true -o /tmp/profile python script.py
```

Key things to verify in nsys output:
- Check which attention backend is used (nvte_flash_attn vs cunn_SoftMax)
- Verify NVFP4 kernels are active (cutlass3x_sm120_bstensorop, nvte_quantize)
- Look for unexpected kernel overhead (quantize/transpose should be <10%)

**Never trust throughput numbers alone** - always profile to confirm expected kernels are running.

## Required Environment Variables for Benchmarks

**ALWAYS set these before running benchmarks** to enable caching and avoid repeated autotune:

```bash
# Persistent inductor cache (avoids re-autotuning triton kernels)
export TORCHINDUCTOR_CACHE_DIR=/home/dave/Downloads/source/nanochat/.cache/torchinductor
export TORCHINDUCTOR_FX_GRAPH_CACHE=1

# GB10/SM121 fixes
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas  # SM121a needs CUDA 13+ ptxas
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1  # Force autotune on GB10 (48 SMs)
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Transformer Engine settings
export NVTE_NVFP4_DISABLE_RHT=1  # Required for torch.compile compatibility
export NVTE_FUSED_ATTN=1
export NVTE_FLASH_ATTN=1
```

**Why triton_mm autotune happens**: Liger's fused_linear_cross_entropy uses torch matmul for lm_head projection, which goes through inductor's triton backend. TE fused layers use cuBLAS/cuDNN. The autotune caches to TORCHINDUCTOR_CACHE_DIR when FX_GRAPH_CACHE=1.

**Quick benchmark command** (after env vars are set):
```bash
.venv/bin/python -m scripts.base_train --depth=20 --num_iterations=10 --warmup_steps=2
```

## Flash Attention 3 (Local Build)

We have a **local FA3 build** for GB10/SM121 (NOT from HuggingFace `kernels` module which lacks aarch64 support):

```python
# Import directly - available in venv via PyTorch build
from flash_attn import flash_attn_func, flash_attn_with_kvcache

# Location: /home/dave/PyTorch_cuda_13_1_main_4fd1b9b7/third_party/flash-attention/
# Binary: flash_attn_2_cuda.cpython-313-aarch64-linux-gnu.so
```

**DO NOT use** `from kernels import get_kernel` (HF Hub) - it lacks SM121/aarch64 builds.

## Performance Comparison (D20 Model, B=8, T=1024, GB10/SM121)

### Architecture Comparison (20 training steps)

| Architecture | Kernel Backend | Throughput | Peak Memory | Notes |
|--------------|----------------|------------|-------------|-------|
| **Original FA3** | CCE | **21,756 tok/s** | **8.59 GB** | **FASTEST - Use this!** |
| **Original FA3** | current (PyTorch) | 16,040 tok/s | 9.65 GB | Good baseline |
| **Original FA3** | Liger | 15,645 tok/s | 17.06 GB | Slower + 2x memory |
| TE + BF16 | compiled | 12,524 tok/s | 17.79 GB | TE overhead |
| TE + NVFP4 | compiled | 13,010 tok/s | 17.19 GB | 4% faster than TE BF16 |

### Kernel Backend Comparison (Original FA3 architecture)

| Backend | Throughput | Memory | Speedup vs Baseline |
|---------|------------|--------|---------------------|
| **CCE (Apple)** | 21,756 tok/s | 8.59 GB | **1.36x** |
| current (PyTorch) | 16,040 tok/s | 9.65 GB | 1.00x |
| Liger | 15,645 tok/s | 17.06 GB | 0.98x |

**Winner: Apple Cut Cross Entropy (CCE)** - 36% faster, 1 GB less memory than baseline!

### Top Kernels (CCE + FA3 profile)

| Kernel | Time % | Description |
|--------|--------|-------------|
| `_cce_backward_kernel` | 5.8% | CCE custom backward pass |
| `flash_bwd_dq_dk_dv_loop_seqk_parallel` | 4.8% | Flash Attention backward |
| `triton_tem_fused__fused_rms_norm_mm_t` | 4.7% | Fused RMSNorm + MatMul |
| `_cce_lse_forward_kernel` | 3.2% | CCE log-sum-exp forward |
| `flash_fwd_kernel` | 2.1% | Flash Attention forward |

### Key Findings

1. **Original FA3 beats TE** - Simple custom layers + flash_attn_func is 1.6x faster than te.TransformerLayer
2. **CCE is the best loss kernel** - Apple's Cut Cross Entropy saves memory without materializing logits
3. **Liger uses too much memory** - 17 GB vs 9 GB for PyTorch native on this model
4. **torch.compile matters more for FA3** - 2x speedup with compile vs only ~10% for TE

### Recommended Configuration

```python
# Best performance: Original FA3 + CCE
from nanochat import kernels
kernels.set_kernel_backend('cce')  # Apple Cut Cross Entropy

# In model forward:
# return kernels.fused_linear_cross_entropy(
#     hidden_states.to(torch.bfloat16),
#     lm_head.weight.to(torch.bfloat16),
#     targets
# )
```

### References

- [Apple Cut Cross Entropy](https://github.com/apple/ml-cross-entropy) - ICLR 2025 paper
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel) - LinkedIn's Triton kernels
- Flash Attention 3 - Local build at `/home/dave/PyTorch_cuda_13_1_main_4fd1b9b7/third_party/flash-attention/`

### TE Experiment Backup

The Transformer Engine refactor experiment (te.TransformerLayer) is archived at:
`/home/dave/Downloads/source/nanochat_te_experiment_backup/`

Contains:
- `BENCHMARK_REPORT.md` - Detailed benchmark results and analysis
- `gpt.py` - TE-refactored GPT model
- `kernels.py` - With CCE backend support
- Benchmark scripts used for testing

**Conclusion**: TE adds overhead for small models. Original FA3 + CCE is 1.7x faster.

---

This project uses **beads (bd)** for issue tracking with Jira synchronization.

## Issue Tracking with bd + Jira

This repository is configured to sync issues between local beads and Jira project NANO.

### Quick Reference

```bash
# Find available work
bd ready

# View issue details
bd show <id>

# Create a new issue (syncs to Jira)
bd create "Issue title" --body "Description"

# Claim work
bd update <id> --status in_progress

# Complete work
bd close <id>

# Sync with Jira
bd jira sync              # Bidirectional sync
bd jira sync --pull       # Import from Jira only
bd jira sync --push       # Export to Jira only
bd jira sync --dry-run    # Preview without changes

# Check Jira sync status
bd jira status

# Sync with git
bd sync
```

### Workflow

1. **Start of session**: Run `bd ready` to see available tasks
2. **Claim work**: `bd update <id> --status in_progress`
3. **Do the work**: Implement the feature/fix
4. **Complete**: `bd close <id>` when done
5. **Sync**: Run `bd jira sync` to push changes to Jira

### Jira Integration Details

- **Jira URL**: https://cppcode.atlassian.net
- **Project Key**: NANO
- Issues created locally sync to Jira with `bd jira sync --push`
- Issues from Jira import with `bd jira sync --pull`

### Session Completion (Landing the Plane)

When ending a work session, complete ALL steps:

1. **File issues** for remaining work: `bd create "Follow-up task"`
2. **Run quality gates** if code changed (tests, linters, builds)
3. **Update issue status**: Close finished work, update in-progress items
4. **Sync to Jira**: `bd jira sync`
5. **Push to remote**:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
6. **Verify**: All changes committed AND pushed

### Environment Variables (set in ~/.bashrc)

- `JIRA_ORG_ID` - Atlassian Organization ID for admin API
- `ATLASSIAN_ADMIN_API_KEY` - Admin API key for org management (api.atlassian.com)
- `JIRA_API_TOKEN` - Personal API token for Jira REST API (create at https://id.atlassian.com/manage-profile/security/api-tokens)
- `JIRA_USERNAME` - Jira username (email): dave@cppcode.online
- `JIRA_URL` - Jira instance URL: https://cppcode.atlassian.net
- `BD_JIRA_SCRIPT` - Path to jira2jsonl.py script: /home/dave/Downloads/jira2jsonl.py

### Admin API Usage

Use the admin API key for organization management:
```bash
# Get org info
curl -s -H "Authorization: Bearer $ATLASSIAN_ADMIN_API_KEY" \
  "https://api.atlassian.com/admin/v1/orgs/$JIRA_ORG_ID"

# List domains
curl -s -H "Authorization: Bearer $ATLASSIAN_ADMIN_API_KEY" \
  "https://api.atlassian.com/admin/v1/orgs/$JIRA_ORG_ID/domains"
```
