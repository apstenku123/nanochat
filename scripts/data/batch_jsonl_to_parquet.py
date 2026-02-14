"""Convert completed JSONL files to parquet shards for streaming training.

Reads {"text": "..."} JSONL, shuffles, splits into train shards + val shard,
writes parquet with _COMPLETE sentinel.

Usage:
    python3 scripts/data/batch_jsonl_to_parquet.py \
        --input /path/to/data.jsonl \
        --parquet_dir /path/to/parquet_output \
        --rows_per_file 50000
"""

import argparse
import json
import os
import random

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--parquet_dir", required=True, help="Output parquet directory")
    parser.add_argument("--rows_per_file", type=int, default=50000)
    parser.add_argument("--val_fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.parquet_dir, exist_ok=True)
    rng = random.Random(args.seed)

    # Read all docs
    print(f"Reading {args.input}...")
    docs = []
    errors = 0
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                text = json.loads(line)["text"]
                docs.append(text)
            except (json.JSONDecodeError, KeyError):
                errors += 1

    total_docs = len(docs)
    print(f"Read {total_docs:,} docs ({errors} errors)")

    # Shuffle
    rng.shuffle(docs)

    # Split val
    val_count = max(1, int(total_docs * args.val_fraction))
    val_docs = docs[:val_count]
    train_docs = docs[val_count:]

    # Write train shards
    shard_idx = 0
    for i in range(0, len(train_docs), args.rows_per_file):
        batch = train_docs[i : i + args.rows_per_file]
        table = pa.table({"text": batch})
        path = os.path.join(args.parquet_dir, f"shard_{shard_idx:05d}.parquet")
        pq.write_table(table, path, row_group_size=1024)
        print(f"  Written shard_{shard_idx:05d}.parquet ({len(batch):,} docs)")
        shard_idx += 1

    # Write val shard
    if val_docs:
        val_table = pa.table({"text": val_docs})
        val_path = os.path.join(args.parquet_dir, "val_shard.parquet")
        pq.write_table(val_table, val_path, row_group_size=1024)
        print(f"  Written val_shard.parquet ({len(val_docs):,} docs)")

    # Write _COMPLETE sentinel
    sentinel = os.path.join(args.parquet_dir, "_COMPLETE")
    with open(sentinel, "w") as sf:
        sf.write(
            f"{shard_idx} train shards, {val_count} val docs, {total_docs} total\n"
        )

    size_mb = sum(
        os.path.getsize(os.path.join(args.parquet_dir, f))
        for f in os.listdir(args.parquet_dir)
    ) / (1024**2)

    print(
        f"\nDONE: {shard_idx} train shards + 1 val shard, "
        f"{total_docs:,} total docs, {size_mb:.0f} MB"
    )


if __name__ == "__main__":
    main()
