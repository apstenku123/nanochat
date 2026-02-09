# Data Pipeline Design

**Status**: Draft v2 — Updated with existing libraries and NVIDIA Nemotron datasets
**Related bd epic**: `cpp-data-pipeline`
**Design doc**: `docs/design/02-data-pipeline.md`

---

## Principle: Use Existing Libraries, Don't Reinvent

| Task                       | Library / Dataset                                                                                                 | Build Custom? |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------- |
| Pipeline framework         | [DataTrove](https://github.com/huggingface/datatrove) (HuggingFace)                                               | No            |
| Code quality filters       | [bigcode-dataset](https://github.com/bigcode-project/bigcode-dataset)                                             | No            |
| Near-deduplication         | [datasketch](https://github.com/ekzhu/datasketch) + BigCode MinHash script                                        | No            |
| PII removal                | [bigcode/pii-lib](https://github.com/bigcode-project/pii-lib) + [StarPII](https://huggingface.co/bigcode/starpii) | No            |
| Secret detection           | [Yelp/detect-secrets](https://github.com/Yelp/detect-secrets)                                                     | No            |
| License detection          | ScanCode Toolkit                                                                                                  | No            |
| Tokenizer training         | [HuggingFace tokenizers](https://github.com/huggingface/tokenizers)                                               | No            |
| Pre-processed C++ data     | `nvidia/Nemotron-Pretraining-Code-v1+v2`                                                                          | No            |
| C++ open-source code       | `bigcode/the-stack-dedup` (data/c++)                                                                              | No            |
| C++ pre-tokenization regex | Custom (~50 lines)                                                                                                | Yes           |
| Indentation stripping      | Custom (~30 lines)                                                                                                | Yes           |
| FIM data transform         | Custom (~20 lines)                                                                                                | Yes           |
| Binary .bin/.idx packaging | Custom (match nanochat format)                                                                                    | Yes           |
| Compilation rate evaluator | Custom                                                                                                            | Yes           |

---

## Data Budget

| Model      | Params | Chinchilla Optimal | Our Target  | Epochs on 80B unique |
| ---------- | ------ | ------------------ | ----------- | -------------------- |
| Dense 3B   | 3B     | 60B tokens         | 300B tokens | 3.75                 |
| MoE 12B/2B | 12B    | 240B tokens        | 300B tokens | 3.75                 |

Target: **300-400 billion tokens** total throughput.

---

## Source Code Acquisition

### Pre-Processed Datasets (Use Directly — Already Deduped/Filtered)

**HuggingFace Token**: Set `HF_TOKEN` env var (see `.env` or ask maintainer)

| Dataset                 | HF Path                               | C++                                | Size          | Quality                                  |
| ----------------------- | ------------------------------------- | ---------------------------------- | ------------- | ---------------------------------------- |
| **Nemotron Code v1**    | `nvidia/Nemotron-Pretraining-Code-v1` | Yes                                | ~747B tokens  | Highest (multi-stage filter + synthetic) |
| **Nemotron Code v2**    | `nvidia/Nemotron-Pretraining-Code-v2` | Yes (+ code review, transpilation) | 897GB         | Highest                                  |
| **Nemotron CC Code v1** | `nvidia/Nemotron-CC-Code-v1`          | Yes                                | 428B tokens   | High (LLM quality-scored)                |
| **The Stack (dedup)**   | `bigcode/the-stack-dedup`             | Yes (`data/c++`)                   | ~3TB total    | Good (MinHash deduped)                   |
| **CommitPack**          | `bigcode/commitpack`                  | Yes (C++ subset)                   | ~4TB total    | Good (git commit+diff pairs)             |
| **CommitPackFT**        | `bigcode/commitpackft`                | Yes (C++ subset)                   | ~2GB filtered | High (quality-filtered commits)          |

### Raw Sources (Supplement for Diversity)

| Source                               | Size (raw)         | Language | Download Method                        |
| ------------------------------------ | ------------------ | -------- | -------------------------------------- |
| **Linux Kernel**                     | ~1.2GB per version | C        | `git clone` by tag (v6.0, v6.6, v6.10) |
| **LLVM/Clang**                       | ~2GB               | C++      | `git clone` by release tag (17-19)     |
| **GCC**                              | ~1.5GB             | C, C++   | `git clone` (13-14)                    |
| **Boost**                            | ~800MB per version | C++      | GitHub releases (1.84-1.86)            |
| **CUDA Toolkit** (samples + headers) | ~500MB             | C, C++   | NVIDIA developer downloads (12.x)      |
| **STL implementations**              | ~200MB             | C++      | libstdc++ (GCC), libc++ (LLVM)         |
| **Chromium**                         | ~30GB              | C++      | `depot_tools` fetch                    |
| **Qt Framework**                     | ~2GB               | C++      | `git clone`                            |

### Download Scripts

Each source gets a version-pinned download script in `scripts/data/`:

```bash
scripts/data/
├── download_stack_v2.py       # HuggingFace streaming download
├── download_linux_kernel.sh   # git clone specific tags
├── download_llvm.sh           # git clone release branches
├── download_gcc.sh            # git clone or mirror
├── download_boost.sh          # GitHub release tarballs
├── download_cuda_samples.sh   # NVIDIA download
├── download_stl.sh            # libstdc++ and libc++
└── download_all.sh            # Master script
```

### Version Strategy

Pin to recent stable releases to get modern C++ (C++17/20/23):
- Linux Kernel: v6.0, v6.6, v6.10
- LLVM: 17.x, 18.x, 19.x
- GCC: 13.x, 14.x
- Boost: 1.84, 1.85, 1.86
- CUDA: 12.x headers and samples

---

## Deduplication

### Why Deduplication is Critical

GitHub contains massive duplication:
- Forked repositories (same code, different repos)
- Copy-pasted utility files (json.hpp, stb_image.h)
- Vendored dependencies (entire libraries copied into repos)

Without dedup: model memorizes popular libraries, loses generalization.

### Method: MinHash + LSH

```
Pipeline:
Raw files → Shingling (5-grams) → MinHash (128 hashes) → LSH (20 bands × 6 rows)
                                                            ↓
                                                    Cluster duplicates
                                                            ↓
                                                    Keep 1 per cluster
```

**Tools**: `datasketch` library or custom implementation.

**Parameters**:
- Jaccard threshold: 0.7 (files sharing >70% content are duplicates)
- Shingle size: 5 tokens (after normalization)
- MinHash permutations: 128

### Exact Dedup

Additionally, SHA-256 hash exact duplicates (identical files across repos).

### Expected Results

| Stage                      | Size        |
| -------------------------- | ----------- |
| Raw C/C++ from all sources | ~500GB text |
| After exact dedup          | ~350GB      |
| After MinHash near-dedup   | ~150-200GB  |
| After quality filtering    | ~120-150GB  |
| Unique tokens (estimated)  | ~60-80B     |

---

## Code Normalization ("The Flattener")

### Algorithm

```python
import re

def normalize_cpp(code: str) -> str:
    """Normalize C++ code for training.

    - Strip leading whitespace (indentation)
    - Preserve newlines
    - Compress multiple blank lines to one
    - Preserve comments
    - Preserve string literals
    """
    lines = code.split('\n')
    result = []
    prev_blank = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if not prev_blank:
                result.append('')
                prev_blank = True
            continue
        prev_blank = False
        result.append(stripped)

    return '\n'.join(result)
```

### Before/After Example

**Before** (raw source):
```cpp
void ProcessPacket(Packet* p) {
    // Validate input
    if (!p) {
        LOG_ERROR("Null packet");
        return;
    }

    if (p->size() > MAX_SIZE) {
        LOG_WARN("Oversized packet");
        return;
    }

    dispatch(p);
}
```

**After** (normalized):
```cpp
void ProcessPacket(Packet* p) {
// Validate input
if (!p) {
LOG_ERROR("Null packet");
return;
}

if (p->size() > MAX_SIZE) {
LOG_WARN("Oversized packet");
return;
}

dispatch(p);
}
```

---

## Quality Filtering

### Heuristic Filters

| Filter                    | Threshold                                         | Rationale                        |
| ------------------------- | ------------------------------------------------- | -------------------------------- |
| Max file size             | 1MB                                               | Auto-generated files, data dumps |
| Max line length           | 1000 chars                                        | Minified code, embedded data     |
| Min file size             | 100 bytes                                         | Empty/trivial files              |
| Min unique lines ratio    | >30%                                              | Repetitive boilerplate           |
| Max comment-to-code ratio | <80%                                              | Files that are mostly comments   |
| Has valid C++ extension   | `.c`, `.cc`, `.cpp`, `.cxx`, `.h`, `.hpp`, `.hxx` | Correct language                 |

### Content Filters

| Filter                 | Method                                  | Rationale                         |
| ---------------------- | --------------------------------------- | --------------------------------- |
| License headers        | Regex detect + strip first N lines      | Wastes context, no semantic value |
| Auto-generated markers | Detect `// Generated by`, `DO NOT EDIT` | Bad training signal               |
| Binary/hex dumps       | Entropy check (>4.5 bits/byte in ASCII) | Not useful code                   |
| Test data files        | Detect `.test.cpp` with only data       | Low code quality                  |
| Vendored deps          | Detect common vendored dirs             | Already in primary sources        |

### PII Redaction

Replace with placeholder tokens:
- Email addresses → `user@example.com`
- IP addresses → `127.0.0.1`
- API keys (high-entropy strings) → `API_KEY_REDACTED`
- File paths with usernames → `/home/user/`

---

## FIM Augmentation

### Fill-In-the-Middle Format

50% of training examples use FIM format:

```
Original:  AAAA BBBB CCCC
FIM Input: <FIM_PREFIX> AAAA <FIM_SUFFIX> CCCC <FIM_MIDDLE> BBBB <EOS>
```

### Split Strategy

For each file selected for FIM:
1. Choose random split point (uniform over character positions)
2. Split into prefix, middle, suffix
3. Middle length: 10-50% of file (random)
4. Concatenate in PSM order (Prefix-Suffix-Middle)

### Implementation

```python
import random

def create_fim_example(code: str, fim_rate: float = 0.5) -> str:
    if random.random() > fim_rate:
        return code  # Normal NTP

    # Random split point
    split1 = random.randint(0, len(code))
    split2 = random.randint(split1, len(code))

    prefix = code[:split1]
    middle = code[split1:split2]
    suffix = code[split2:]

    return f"<FIM_PREFIX>{prefix}<FIM_SUFFIX>{suffix}<FIM_MIDDLE>{middle}<EOS>"
```

---

## Data Mix

### Training Data Composition

| Source                          | Fraction | Tokens (est.) | Purpose                             |
| ------------------------------- | -------- | ------------- | ----------------------------------- |
| Clean GitHub C++                | 60%      | 180-240B      | Diverse real-world code             |
| Synthetic textbooks             | 20%      | 60-80B        | Concepts, best practices, reasoning |
| Code exercises (LeetCode-style) | 10%      | 30-40B        | Problem → Solution patterns         |
| Instruction Q&A                 | 10%      | 30-40B        | Question → Reasoning → Code format  |

### Synthetic Data Generation

Use a teacher model (Claude/GPT-4) to generate:

1. **Textbook examples**: "Explain RAII with code example" → `// QUESTION:` + `/* ANSWER */` + code
2. **Bug-fix pairs**: Inject bug into good code → generate debug trace → write fix
3. **Code review**: "What's wrong with this code?" → reasoning + fixed version

### Multi-Epoch Strategy

With ~80B unique tokens and 300B target:
- Epoch 1-2: Full corpus, standard order
- Epoch 3-4: Re-shuffled, different FIM splits
- Never repeat exact same FIM split of same file

---

## Binary Packaging

### Output Format

Pre-tokenized binary files compatible with nanochat's `binary_distributed_data_loader`:

```
File format:
[8 bytes: uint64 num_tokens] [num_tokens × uint16 token_ids]
```

### Pipeline

```
Raw C++ files
    → normalize_cpp()
    → quality_filter()
    → deduplicate()
    → create_fim_example() (50% chance)
    → tokenizer.encode()
    → write to .bin file
```

### Sharding

- Shard size: ~1GB each (~300M tokens per shard)
- Train shards: 900+ files
- Validation shard: Last shard (held out)
- Index file (`.idx`): Maps shard → token count for efficient seeking

---

## Validation Split

- **Size**: 1% of total data (~3B tokens)
- **Composition**: Same mix ratio as training data
- **Constraint**: No file appears in both train and validation
- **Purpose**: Monitor bits-per-byte during training
