# Stepping Stones: Incremental Path to C++ Specialist LLM

**Status**: Draft v1
**Design doc**: `docs/design/07-stepping-stones.md`

---

## Philosophy

**Never skip validation.** Each milestone is a complete end-to-end cycle:
data → tokenizer → train → evaluate → iterate.

Scale up only after the previous milestone proves the concept works.

---

## Milestone 0: Proof of Concept (Local GB10, ~125M params)

**Goal**: Full pipeline working end-to-end at nanochat speedrun scale.
**Hardware**: 1x NVIDIA GB10 (DGX Spark)
**Time**: Hours (data prep) + ~1-2h (training)
**Cost**: $0 (local machine)

### What We Build

1. **Data (small)**:
   - Download `nvidia/Nemotron-Pretraining-Code-v1` C++ subset (~5-10GB)
   - Supplement with `bigcode/the-stack-dedup` `data/c++` (small sample)
   - Apply normalization (strip indentation) using bigcode-dataset filters
   - Dedup with `datasketch` (MinHash) — or skip for this small set
   - PII strip with `bigcode/pii-lib`
   - Target: **~2-5GB clean normalized C++ text**

2. **Tokenizer**:
   - Train C++ BPE (32K vocab) on the clean corpus using HuggingFace `tokenizers`
   - Custom pre-tokenization regex for C++ operators
   - Add special tokens (BOS, EOS, FIM_PREFIX/MIDDLE/SUFFIX)
   - Validate: tokenize sample files, compare tokens/byte vs tiktoken

3. **Pre-tokenize**:
   - Convert to nanochat binary format (.bin + .idx)
   - Apply FIM augmentation (50% rate)

4. **Model**:
   - Use nanochat's existing architecture (depth=12, ~125M params)
   - Only change: vocab_size=32000, use our C++ tokenizer
   - Keep everything else (RMSNorm, RoPE, CCE, Muon optimizer)

5. **Train**:
   - Single GPU training on GB10
   - `python -m scripts.base_train --depth=12 --num_iterations=5000`
   - Monitor loss convergence, tok/sec, memory

6. **Evaluate**:
   - Bits per byte on C++ validation set
   - Sample completions: does it write valid C++?
   - Manual inspection: does it understand `std::`, templates, etc.?
   - Tokenizer efficiency comparison (our BPE vs tiktoken on C++ files)

### Success Criteria
- [ ] Loss converges (decreasing BPB)
- [ ] Sampled completions are syntactically valid C++
- [ ] Our tokenizer is 1.5x+ more efficient than tiktoken on C++ code
- [ ] Full pipeline runs without crashes on GB10

### Deliverables
- `scripts/data/download_cpp_data.py` — download Nemotron + Stack C++ subsets
- `scripts/data/normalize_cpp.py` — normalization pipeline (using bigcode-dataset filters)
- `scripts/tok_train_cpp.py` — train C++ tokenizer
- `scripts/pretokenize_cpp.py` — pre-tokenize to binary with FIM
- Modified `nanochat/tokenizer.py` — CppTokenizer class
- Evaluation notebook showing tokenizer comparison

---

## Milestone 1: Speedrun Scale ($100, 8xH100, ~313M params)

**Goal**: Train a useful small C++ model at nanochat's "speedrun" scale.
**Hardware**: 8x H100 (rented)
**Time**: ~4 hours training
**Cost**: ~$100

### What Changes from M0

1. **Data (medium)**:
   - Full `nvidia/Nemotron-Pretraining-Code-v1` C++ subset
   - Full `bigcode/the-stack-dedup` C++ subset
   - Download Linux Kernel v6.10 + LLVM 19.x + Boost 1.86 sources
   - Full dedup pipeline (MinHash via datasketch)
   - Full quality filtering (bigcode-dataset heuristics)
   - Target: **~20-50GB clean C++, ~15-30B tokens**

2. **Model**:
   - Depth=20 (nanochat's standard $100 config, ~313M params)
   - vocab_size=32000 (our C++ tokenizer)
   - Chinchilla-optimal: ~6B tokens for 313M params

3. **Training**:
   - `torchrun --nproc_per_node=8 -m scripts.base_train --depth=20`
   - Standard nanochat hyperparameters
   - FIM objective (50/50)

4. **Evaluation**:
   - Validation BPB
   - Compilation rate (sample 100 completions, try `g++ -fsyntax-only`)
   - Compare BPB vs nanochat's English model at same scale
   - FIM completion quality (hide middle of function, check if model fills correctly)

### Success Criteria
- [ ] BPB competitive with or better than English nanochat at same param count
- [ ] >50% of sampled completions compile
- [ ] FIM completions are syntactically coherent
- [ ] Training completes in <4h on 8xH100

---

## Milestone 2: Quality Scale ($300-1000, 8xH100, ~700M-2B params)

**Goal**: Train a model that writes genuinely useful C++ code.
**Hardware**: 8x H100
**Time**: 12-40 hours
**Cost**: $300-$1000

### What Changes from M1

1. **Data (large)**:
   - All sources: Nemotron v1+v2, Stack dedup, Kernel, LLVM, GCC, Boost, CUDA, STL
   - Full dedup + filter pipeline
   - Add synthetic textbook data (20% of mix)
   - Add code exercises (10%)
   - Target: **~100-150GB clean C++, ~40-60B unique tokens, multi-epoch to ~100B**

2. **Model**:
   - Depth=26 (~700M) or Depth=32 (~2B)
   - Add reasoning special tokens (THOUGHT_START/END, QUERY_TOOL)
   - Keep GQA, SwiGLU, RMSNorm, CCE

3. **Training**:
   - Pretrain: NTP+FIM
   - Basic SFT: Small set of bug-fix pairs (1K-10K examples)
   - Evaluate reasoning quality

4. **Evaluation**:
   - >70% compilation rate
   - Basic bug-fix accuracy on synthetic benchmark
   - HumanEval-C++ (if available) or custom C++ coding benchmark

### Success Criteria
- [ ] >70% compilation rate on sampled completions
- [ ] Can fix simple null-pointer bugs given debug context
- [ ] Reasoning blocks (/* REASONING */) are coherent

---

## Milestone 3: Full 3B Model (8xB200, 200h)

**Goal**: Production-quality 3B C++ specialist.
**Hardware**: 8x B200 (Blackwell)
**Time**: 200 hours
**Cost**: ~$5,000-10,000

### What Changes from M2

1. **Data (full)**:
   - Full 300B token budget (multi-epoch)
   - Full synthetic data pipeline (textbooks, exercises, Q&A, bug-fix pairs)
   - Debug state training data

2. **Model**:
   - Full 3B config (hidden=2560, layers=32, heads=32, kv=8)
   - All special tokens
   - YaRN for long context (LCFT stages)

3. **Training**:
   - Stage 1: Pretrain (150h)
   - Stage 2: SFT with reasoning (20h)
   - Stage 3: RLVR with compiler feedback (30h)

4. **Agent Integration**:
   - Tool token detection
   - Basic agent loop

---

## Milestone 4: Agent + Indexing (Post-Training)

**Goal**: Full agent system with codebase indexing.
**Hardware**: Inference GPU (single H100/B200)

1. Latent code indexing (CLaRa-style)
2. Contrastive fine-tuning
3. FAISS vector DB integration
4. Full ReAct agent loop with 7 tools

---

## Milestone 5: MoE + Long Context (Optional)

**Goal**: Scale to 10-12B MoE with 1M context.

1. MoE architecture (16 experts, 2 shared)
2. YaRN scaling to 1M
3. MLA for KV-cache compression
4. Context parallelism for training

---

## Libraries We Use (Not Build)

Based on web research, these existing tools replace custom implementations:

| Task | Library | Notes |
|------|---------|-------|
| **Pipeline framework** | [DataTrove](https://github.com/huggingface/datatrove) | Composable data processing, proven on FineWeb |
| **Code-specific filters** | [bigcode-dataset](https://github.com/bigcode-project/bigcode-dataset) | StarCoder's exact filters |
| **Near-deduplication** | [datasketch](https://github.com/ekzhu/datasketch) + BigCode MinHash script | Battle-tested |
| **PII removal** | [bigcode/pii-lib](https://github.com/bigcode-project/pii-lib) + [StarPII](https://huggingface.co/bigcode/starpii) | Regex + NER model |
| **Secret detection** | [Yelp/detect-secrets](https://github.com/Yelp/detect-secrets) | Enterprise-grade |
| **License detection** | ScanCode Toolkit | Used by The Stack v2 |
| **Tokenizer training** | [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) | Industry standard Rust BPE |
| **FIM transform** | Custom (trivial, ~20 lines) | Split prefix/middle/suffix |
| **Pre-training data** | `nvidia/Nemotron-Pretraining-Code-v1+v2` | Already deduped/filtered C++ |
| **Code data (backup)** | `bigcode/the-stack-dedup` (`data/c++`) | Permissive-license C++ |
| **Code data (CC)** | `nvidia/Nemotron-CC-Code-v1` | 428B tokens, LLM quality-scored |

### What We Build Custom

| Component | Why Custom |
|-----------|-----------|
| C++ pre-tokenization regex | No existing regex handles all C++ operators correctly |
| C++ normalizer (indentation stripping) | Simple, 30 lines, specific to our needs |
| FIM data transform | Trivial, specific to our token format |
| Binary packaging | Must match nanochat's .bin/.idx format |
| Compilation rate evaluator | Unique to our pipeline |
| Agent loop | Unique to our architecture |
| Contrastive fine-tuning | Research code, no library exists |

---

## Data Sources Summary

### Pre-Processed (Use Directly)

| Dataset | HF Path | C++ Content | Size | Quality |
|---------|---------|-------------|------|---------|
| Nemotron Code v1 | `nvidia/Nemotron-Pretraining-Code-v1` | Yes | ~747B tokens | Highest (multi-stage filter + synthetic) |
| Nemotron Code v2 | `nvidia/Nemotron-Pretraining-Code-v2` | Yes (incl. code review, transpilation) | 897GB download | Highest |
| Nemotron CC Code v1 | `nvidia/Nemotron-CC-Code-v1` | Yes | 428B tokens | High (LLM quality-scored) |
| The Stack (dedup) | `bigcode/the-stack-dedup` | Yes (`data/c++`) | ~3TB total | Good |

### Raw Sources (Download + Process)

| Source | Why Include | Processing |
|--------|-------------|-----------|
| Linux Kernel | Canonical C, systems programming | Normalize, no dedup needed (unique) |
| LLVM/Clang | Modern C++, compiler internals | Normalize |
| GCC | C/C++ compiler, different style | Normalize |
| Boost | Advanced C++ templates, libraries | Normalize |
| CUDA toolkit | GPU programming patterns | Normalize |
| libstdc++ / libc++ | STL implementations | Normalize (use sparingly, very dense) |

### HuggingFace Token

Set `HF_TOKEN` environment variable for authenticated access to gated datasets (Nemotron collections).
