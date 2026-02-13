# C++ Data Processing Pipeline

This document describes how C++ source code is processed into training data for nanochat.

## Overview

```
Raw C++ Source (JSONL)
    ↓
chunk_cpp_data.py (function-boundary chunking)
    ↓
Intermediate JSONL (growing file)
    ↓  ← stream_jsonl_to_parquet.py (real-time)
Parquet Shards (50K docs each)
    ↓  ← stream_upload_parquet.sh
GCS Bucket
    ↓  ← stream_download_parquet.sh
TPU Local Storage
    ↓  ← dataloader.py (streaming mode)
Training
```

## Source Data

All raw C++ data is stored compressed in GCS:

| File | Size | Description | GCS Path |
| ---- | ---- | ----------- | -------- |
| `cpp_combined_10b_v3.jsonl.gz` | ~10 GB | 4.5M+ C++ source files | `gs://nanochat-training-data-2026/data/` |
| `ms_src.jsonl.gz` | ~0.7 GB | Microsoft open-source C++ code | `gs://nanochat-training-data-2026/data/` |

### Downloading Source Data

```bash
# Download and decompress
gcloud storage cp gs://nanochat-training-data-2026/data/cpp_combined_10b_v3.jsonl.gz data/
gcloud storage cp gs://nanochat-training-data-2026/data/ms_src.jsonl.gz data/
gunzip data/cpp_combined_10b_v3.jsonl.gz
gunzip data/ms_src.jsonl.gz
```

## Step 1: Function-Boundary Chunking

`scripts/data/chunk_cpp_data.py` splits C++ source files into training documents at function boundaries:

```bash
python -u -m scripts.data.chunk_cpp_data \
    --inputs data/cpp_combined_10b_v3.jsonl data/ms_src.jsonl \
    --output data/cpp_chunked_all.jsonl \
    --parquet_dir data/cpp_chunked_parquet \
    --max_tokens 16384
```

### Key Parameters

| Parameter | Description |
| --------- | ----------- |
| `--max_tokens` | Maximum tokens per document. **Must match `--max_seq_len` in training.** |
| `--inputs` | One or more JSONL files with `{"text": "..."}` format |
| `--output` | Intermediate JSONL output (can be very large) |
| `--parquet_dir` | Final parquet output directory |

### How Chunking Works

1. Reads each C++ file from the input JSONL
2. Tokenizes with the 65K C++ tokenizer
3. If file fits within `max_tokens`, keeps as single document
4. If too large, splits at function boundaries (top-level `{` / `}` braces)
5. Writes documents to output JSONL as `{"text": "..."}` lines

### Performance

- Processing speed: ~1000 files/sec
- 4.5M input files → ~7M+ output documents (with `--max_tokens 16384`)
- Total runtime: ~75 minutes on local machine

## Step 2: Streaming JSONL to Parquet

For large datasets, the chunking produces a growing JSONL file over 30+ minutes. `stream_jsonl_to_parquet.py` reads the growing JSONL and writes parquet shards in real-time:

```bash
python -u -m scripts.data.stream_jsonl_to_parquet \
    --input data/cpp_chunked_all.jsonl \
    --parquet_dir data/cpp_chunked_parquet \
    --rows_per_file 50000
```

### How It Works

1. Opens the JSONL file and reads lines with `readline()`
2. Accumulates documents in batches of 50,000
3. When a batch is full, shuffles and writes a parquet shard immediately
4. If no new data for 60 seconds (file stable), assumes writer is done
5. Writes final validation shard and `_COMPLETE` sentinel file

### Output

- Train shards: `shard_00000.parquet`, `shard_00001.parquet`, ...
- Validation shard: `val_shard.parquet` (1% of total docs)
- Sentinel: `_COMPLETE` (signals all data is written)

## Step 3: Upload to GCS

`scripts/stream_upload_parquet.sh` watches the local parquet directory and uploads new shards as they appear:

```bash
bash scripts/stream_upload_parquet.sh \
    data/cpp_chunked_parquet \
    gs://nanochat-training-data-2026/parquet/cpp_chunked_16k
```

The script runs in a loop, checking for new `.parquet` files every 10 seconds and uploading via `gcloud storage cp`.

## Step 4: Download to TPU

`scripts/stream_download_parquet.sh` runs on the TPU and downloads new shards from GCS:

```bash
bash scripts/stream_download_parquet.sh \
    gs://nanochat-training-data-2026/parquet/cpp_chunked_16k \
    /home/dave/data/parquet
```

## Step 5: Streaming Training

The dataloader supports **streaming mode** where it dynamically discovers new parquet shards:

```bash
python -m scripts.base_train \
    --data_dir=/home/dave/data/parquet \
    --streaming_data \
    --max_seq_len=16384 \
    ...
```

### Streaming Dataloader Behavior

1. Waits for first parquet shard to appear in `--data_dir`
2. Starts training on available shards
3. Between shard iterations, re-scans directory for new shards
4. When all known shards are consumed and `_COMPLETE` sentinel is absent, waits 30s and re-scans
5. Once `_COMPLETE` is found and all shards processed, wraps epoch normally

## Running the Full Pipeline

To run everything end-to-end in parallel:

```bash
# Terminal 1: Start chunking
python -u -m scripts.data.chunk_cpp_data \
    --inputs data/cpp_combined_10b_v3.jsonl data/ms_src.jsonl \
    --output data/cpp_chunked_all.jsonl \
    --parquet_dir data/cpp_chunked_parquet \
    --max_tokens 16384

# Terminal 2: Start streaming parquet writer (reads growing JSONL)
python -u -m scripts.data.stream_jsonl_to_parquet \
    --input data/cpp_chunked_all.jsonl \
    --parquet_dir data/cpp_chunked_parquet \
    --rows_per_file 50000

# Terminal 3: Start upload watcher
bash scripts/stream_upload_parquet.sh \
    data/cpp_chunked_parquet \
    gs://nanochat-training-data-2026/parquet/cpp_chunked_16k

# Terminal 4 (on TPU): Start download watcher
bash scripts/stream_download_parquet.sh \
    gs://nanochat-training-data-2026/parquet/cpp_chunked_16k \
    /home/dave/data/parquet

# Terminal 5 (on TPU): Start training in streaming mode
python -m scripts.base_train \
    --data_dir=/home/dave/data/parquet --streaming_data \
    --max_seq_len=16384 --xla_flash_attn --window_pattern=L \
    ...
```

## GCS Data Layout

```
gs://nanochat-training-data-2026/
├── data/                          # Raw source data + processed parquet
│   ├── cpp_combined_10b_v3.jsonl.gz   # 4.5M+ C++ files
│   ├── ms_src.jsonl.gz                # Microsoft C++ source
│   ├── cpp_clean_v4.jsonl             # Cleaned subset
│   ├── combined_sft.jsonl             # SFT training data
│   ├── diff_sft.jsonl                 # Diff-based SFT data
│   ├── docstring_pairs_full.jsonl     # Docstring extraction pairs
│   ├── gspo_prompts.jsonl             # GSPO prompts
│   ├── cpp_crossfile_16k/             # 8.1M docs, 23 GB (tree-sitter cross-file)
│   ├── cpp_project_crossfile_16k/     # 1.0M docs, 8 GB (project-aware)
│   ├── cpp_clang_crossfile_16k/       # 794K docs, 883 MB (clang semantic)
│   ├── cpp_compilable_16k/            # 1.0M docs, 7.25 GB (compilable, RECOMMENDED)
│   └── cpp_compilable_64k/            # 394K docs, 3.33 GB (compilable, long-context)
├── parquet/                       # Processed training data (legacy)
│   ├── base_data_v3/                  # 105 shards, standard training
│   └── cpp_chunked_16k/              # 140+ shards, 16K context
├── tokenizer_65k/                 # 65K C++ tokenizer
├── tokenizer/                     # 32K tokenizer (legacy)
├── checkpoints/                   # Model checkpoints
│   ├── v5e/
│   ├── v6e/
│   └── v6e-longctx/
├── eval/                          # Evaluation data
└── code/                          # Code snapshots
```

## Re-processing Data

To re-process with different parameters (e.g., different `max_tokens`):

```bash
# 1. Download raw source from GCS
gcloud storage cp gs://nanochat-training-data-2026/data/cpp_combined_10b_v3.jsonl.gz data/
gcloud storage cp gs://nanochat-training-data-2026/data/ms_src.jsonl.gz data/
gunzip data/*.gz

# 2. Re-chunk with new max_tokens
python -u -m scripts.data.chunk_cpp_data \
    --inputs data/cpp_combined_10b_v3.jsonl data/ms_src.jsonl \
    --output data/cpp_chunked_new.jsonl \
    --parquet_dir data/cpp_chunked_parquet_new \
    --max_tokens <NEW_VALUE>

# 3. Upload to GCS
gcloud storage cp -r data/cpp_chunked_parquet_new/ \
    gs://nanochat-training-data-2026/parquet/cpp_chunked_<NEW_VALUE>/
```

## Rust cpp-chunker (Advanced Pipeline)

The Rust-based `cpp-chunker` tool at `tools/cpp_chunker/` provides high-performance, dependency-aware C++ chunking with multiple modes. It processes raw C++ project directories directly (no JSONL intermediate).

### Source Code

```
tools/cpp_chunker/
├── src/
│   ├── main.rs           # CLI entry point, batch/project/cross-file modes
│   ├── chunker.rs         # Tree-sitter AST parsing (functions, classes, structs, enums)
│   ├── deps.rs            # Call graph analysis, dependency levels, cross-file resolution
│   ├── compilable.rs      # Compilable chunk mode (types before functions, bottom-up)
│   ├── global_index.rs    # Global function index for cross-file dependency resolution
│   └── project_graph.rs   # Project dependency DAG, topological ordering
└── Cargo.toml
```

### Building

```bash
cd tools/cpp_chunker
cargo build --release
# Binary: target/release/cpp-chunker
```

### Processing Modes

| Mode | Flag | Description |
|------|------|-------------|
| Basic | (default) | Tree-sitter chunking, function-boundary splitting |
| Dep-aware | `--dep-aware` | Bottom-up function ordering within each file |
| Cross-file | `--cross-file` | Two-pass: global function index + cross-file deps |
| **Project-aware** | `--project-dirs` | Reads raw project dirs, dependency DAG ordering |
| **Compilable** | `--project-dirs --compilable` | Near-compilable chunks: preamble → types → functions |

### Compilable Mode (Recommended)

The `--compilable` flag produces training documents structured as near-compilable C++ units:

1. **Preamble** — `#include` directives, `using` declarations, forward declarations
2. **Type definitions** — structs, classes, enums in topological dependency order (e.g., `Base` before `Derived`)
3. **Functions** — bottom-up call order (leaf callees first, root callers last)

This ensures the model sees every definition before its usage.

### Running on build3

**Prerequisites:** 93 C++ open-source projects in `~/data/cpp_raw/` on build3 (c4-highmem-48-lssd, IP 35.242.211.80).

```bash
# Generate 16K compilable chunks
./cpp-chunker \
    --project-dirs ~/data/cpp_raw \
    --output ~/data/cpp_compilable_16k.jsonl \
    --max-tokens 16384 \
    --cross-depth 3 \
    --compilable \
    --max-file-bytes 500000

# Generate 64K compilable chunks
./cpp-chunker \
    --project-dirs ~/data/cpp_raw \
    --output ~/data/cpp_compilable_64k.jsonl \
    --max-tokens 65536 \
    --cross-depth 3 \
    --compilable \
    --max-file-bytes 500000
```

### Parquet Conversion

After generating JSONL, convert to parquet for training:

```python
import json, os, random
import pyarrow as pa, pyarrow.parquet as pq

texts = [json.loads(l)["text"] for l in open("output.jsonl") if l.strip()]
random.Random(42).shuffle(texts)
val_count = max(1, int(len(texts) * 0.01))
train, val = texts[:-val_count], texts[-val_count:]

os.makedirs("parquet_dir", exist_ok=True)
for i in range(0, len(train), 50000):
    batch = train[i:i+50000]
    pq.write_table(pa.table({"text": batch}), f"parquet_dir/shard_{i//50000:05d}.parquet")
pq.write_table(pa.table({"text": val}), "parquet_dir/val_shard.parquet")
```

### All Datasets on GCS

All at `gs://nanochat-training-data-2026/data/`:

| Dataset | Docs | Parquet Size | Shards | Mode | Token Limit |
|---------|------|-------------|--------|------|-------------|
| `cpp_crossfile_16k` | 8,092,541 | 23 GB | 162+val | Cross-file (JSONL) | 16K |
| `cpp_project_crossfile_16k` | 1,015,901 | 8.0 GB | 21+val | Project-aware | 16K |
| `cpp_clang_crossfile_16k` | 794,028 | 883 MB | 16+val | Clang semantic | 16K |
| **`cpp_compilable_16k`** | **1,012,293** | **7.25 GB** | **21+val** | **Compilable** | **16K** |
| **`cpp_compilable_64k`** | **393,782** | **3.33 GB** | **8+val** | **Compilable** | **64K** |

### Training Commands

```bash
# 16K compilable (standard context)
python -m scripts.base_train \
    --data_dir=<local_path>/cpp_compilable_16k \
    --streaming_data \
    --max_seq_len=1024 --device_batch_size=8 \
    --depth=16 --num_iterations=50000

# 64K compilable (long context, TPU v6e)
python -m scripts.base_train \
    --data_dir=<local_path>/cpp_compilable_64k \
    --streaming_data \
    --max_seq_len=16384 --device_batch_size=1 \
    --depth=16 --xla_flash_attn --no_compile
```

### 93 Source Projects

The raw C++ projects (641K files, 29GB) were downloaded from GitHub and are stored at `~/data/cpp_raw/` on build3. Projects include: boost, llvm-project, linux, gcc-mirror, opencv, tensorflow, protobuf, grpc, rocksdb, clickhouse, godot, qemu, freebsd-src, and 80 more.

### Pipeline Performance

| Pipeline | Time | Rate | Notes |
|----------|------|------|-------|
| Project discovery + DAG | ~5s | — | 93 projects, symlink-safe |
| Compilable 16K generation | 237s | 1,225 files/sec | 290K source files |
| Compilable 64K generation | 200s | 1,447 files/sec | Fewer docs (more per file) |

## SFT Data

After base training completes, SFT training uses:

| Dataset | Size | Examples | Description |
| ------- | ---- | -------- | ----------- |
| `tool_call_sft_1.8M.jsonl` | 2.4 GB | 1.8M | Tool-call SFT (5 strategies) |
| `tool_call_sft.jsonl` | 264 MB | 180K | Original SFT dataset |

SFT training command:
```bash
python -m scripts.sft_train \
    --data data/tool_call_sft_1.8M.jsonl \
    --epochs 4 --batch_size 8 --lr 1e-4 \
    --kernel current --save_every 5000 --no_compile
```
