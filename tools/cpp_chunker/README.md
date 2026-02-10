# C++ Chunker — Rust (tree-sitter) vs Python (brace-matching)

## Overview

Two implementations of syntax-aware C++ file chunking for training data:

| | **Rust (tree-sitter)** | **Python (brace-matching)** |
|---|---|---|
| Parser | tree-sitter AST | Regex + brace-depth tracking |
| Parallelism | rayon (N threads) | Single-threaded |
| Location | `tools/cpp_chunker/` | `scripts/data/cpp_chunker.py` |

## Performance Comparison (100K files from cpp_combined_10b_v3.jsonl)

| Metric | Rust | Python | Notes |
|--------|------|--------|-------|
| Wall time | **54s** | 218s | 4x faster (20 threads) |
| CPU time | 3m22s | 3m32s | Similar per-file cost |
| Rate | 1,867 files/sec | 462 files/sec | |
| Docs out | 713,380 | 691,972 | ~3% more from Rust |
| Dedup | 1,068,487 | 32,766 | Rust catches more exact dupes |

**At scale (5.2M files, 16K max_tokens):** Rust starts fast (~2000/sec) but slows on
larger files (~500/sec). Python maintains steady ~1000/sec. Both complete in 1-3 hours.

## Chunking Quality Differences

### Tree-sitter (Rust) — Proper AST
- Uses `function_definition`, `class_specifier`, `namespace_definition` AST nodes
- **Correctly handles**: templates, C++20 concepts, linkage specs, enum classes
- **Keeps class methods together** — never splits `push()` from `print()` in a class
- **Namespace wrapping** — functions inside `namespace utils { }` get wrapped
- **Handles parse errors** — tree-sitter's error recovery means partial/malformed files still chunk
- **Forward declarations** correctly classified as preamble

### Brace-matching (Python) — Simple but effective
- Scans for depth-0 `{` `}` pairs, classifies what's before the brace
- **Handles Allman and K&R** brace styles (fixed in v2)
- **Splits more aggressively** — produces smaller, more numerous documents
- **Can misclassify**: `while(x) {` as function, nested `extern "C"` blocks
- **No template awareness** — template functions treated as plain functions

### Token Distribution (1K records from ms_src.jsonl, max_tokens=1024)

| Percentile | Rust | Python |
|------------|------|--------|
| p10 | 40 | 30 |
| p50 | 1,010 | 192 |
| p90 | 1,470 | 950 |
| max | 2,046 | 135,519 |
| >4K tokens | 0 (0%) | 96 (1.9%) |

**Key difference**: Rust caps at ~2K tokens (oversized splitting), Python lets some monster
docs through (135K tokens). For training, Rust's distribution is more controlled.

## Streaming Pipeline

The full pipeline is designed to stream — no waiting for all data before parquet conversion:

```
┌──────────────┐     ┌─────────────────────┐     ┌──────────────────────┐
│ cpp-chunker  │────>│  growing JSONL file  │────>│ stream_jsonl_to_     │
│ (Rust/Python)│     │                      │     │ parquet.py           │
│ writes docs  │     │ cpp_chunked_16k.jsonl│     │ (writes shards as   │
│ as processed │     │                      │     │  data accumulates)   │
└──────────────┘     └─────────────────────┘     └──────────┬───────────┘
                                                            │
                                                   ┌────────▼────────┐
                                                   │  parquet shards │
                                                   │  shard_00000    │
                                                   │  shard_00001    │
                                                   │  ...            │
                                                   │  _COMPLETE      │
                                                   └─────────────────┘
```

### Usage

```bash
# Step 1: Build Rust chunker
cd tools/cpp_chunker && cargo build --release

# Step 2: Run chunker (streams JSONL as it processes)
nohup tools/cpp_chunker/target/release/cpp-chunker \
  --inputs data/cpp_combined_10b_v3.jsonl data/ms_src.jsonl \
  --output data/cpp_chunked_16k_rust.jsonl \
  --max-tokens 16384 \
  > /tmp/rust_chunk.log 2>&1 &

# Step 3: Run streaming parquet converter in parallel
nohup python -m scripts.data.stream_jsonl_to_parquet \
  --input data/cpp_chunked_16k_rust.jsonl \
  --parquet_dir data/cpp_chunked_16k_parquet \
  > /tmp/stream_pq.log 2>&1 &

# Step 4 (optional): Train with streaming data discovery
python -m scripts.base_train \
  --data_dir data/cpp_chunked_16k_parquet \
  --streaming_data \
  --max_seq_len 16384 \
  ...
```

The streaming pipeline means:
- Parquet shards appear while chunking is still running
- Training can start as soon as the first shard is written
- `_COMPLETE` sentinel signals all data is ready
