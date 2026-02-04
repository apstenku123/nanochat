#!/usr/bin/env python3
"""Download The Stack v2 dedup C++ data from HuggingFace as JSONL chunks.

Uses huggingface_hub to download parquet files directly (bypasses datasets gating issue),
then converts to JSONL with only the 'content' field.
"""

import json
import time
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, HfApi

OUTPUT_DIR = Path("/home/dave/Downloads/source/nanochat/data/stack_v2_cpp")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DOCS_PER_FILE = 100_000
TARGET_BYTES = 20 * 1024**3  # 20 GB

PARQUET_FILES = [
    "data/C++/train-00000-of-00004.parquet",
    "data/C++/train-00001-of-00004.parquet",
    "data/C++/train-00002-of-00004.parquet",
    "data/C++/train-00003-of-00004.parquet",
]

def main():
    token = HfApi().token
    print(f"Token available: {bool(token)}")

    total_bytes = 0
    total_docs = 0
    file_idx = 0
    buffer = []
    t0 = time.time()

    # Check for existing files to resume
    existing = sorted(OUTPUT_DIR.glob("chunk_*.jsonl"))
    if existing:
        print(f"Found {len(existing)} existing chunks, counting...")
        for ef in existing:
            with open(ef) as f:
                for line in f:
                    doc = json.loads(line)
                    total_bytes += len(doc["content"].encode("utf-8"))
                    total_docs += 1
        file_idx = len(existing)
        print(f"Resuming from {total_docs:,} docs, {total_bytes/1e9:.2f} GB")
        if total_bytes >= TARGET_BYTES:
            print("Target already reached!")
            return

    docs_to_skip = total_docs
    docs_seen = 0

    for pf in PARQUET_FILES:
        print(f"\nDownloading {pf}...")
        local_path = hf_hub_download(
            repo_id="bigcode/the-stack-v2-dedup",
            filename=pf,
            repo_type="dataset",
            token=token,
        )
        print(f"  Cached at: {local_path}")

        # Read parquet in batches to avoid loading all into memory
        parquet_file = pq.ParquetFile(local_path)
        for batch in parquet_file.iter_batches(batch_size=10_000, columns=["content"]):
            for content in batch.column("content"):
                content = content.as_py()
                docs_seen += 1

                if docs_to_skip > 0:
                    docs_to_skip -= 1
                    if docs_to_skip % 500_000 == 0 and docs_to_skip > 0:
                        print(f"  Skipping... {docs_to_skip:,} remaining")
                    continue

                if not content:
                    continue

                buffer.append({"content": content})
                total_bytes += len(content.encode("utf-8"))
                total_docs += 1

                if len(buffer) >= DOCS_PER_FILE:
                    outpath = OUTPUT_DIR / f"chunk_{file_idx:04d}.jsonl"
                    with open(outpath, "w") as f:
                        for doc in buffer:
                            f.write(json.dumps(doc) + "\n")
                    elapsed = time.time() - t0
                    rate = total_bytes / elapsed if elapsed > 0 else 0
                    print(
                        f"  Wrote {outpath.name}: {total_docs:,} docs, "
                        f"{total_bytes/1e9:.2f} GB, {rate/1e6:.1f} MB/s"
                    )
                    buffer = []
                    file_idx += 1

                if total_bytes >= TARGET_BYTES:
                    break
            if total_bytes >= TARGET_BYTES:
                break
        if total_bytes >= TARGET_BYTES:
            break

    # Write remaining buffer
    if buffer:
        outpath = OUTPUT_DIR / f"chunk_{file_idx:04d}.jsonl"
        with open(outpath, "w") as f:
            for doc in buffer:
                f.write(json.dumps(doc) + "\n")
        print(f"Wrote {outpath.name} (final): {len(buffer):,} docs")
        file_idx += 1

    elapsed = time.time() - t0
    print(f"\nDone! {total_docs:,} documents, {total_bytes/1e9:.2f} GB in {elapsed:.0f}s")
    print(f"Output: {file_idx} files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
