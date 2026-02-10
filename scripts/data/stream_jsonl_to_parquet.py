"""Stream a growing JSONL file to parquet shards in real-time.

Reads from a JSONL file that is still being written to, and produces
parquet shards as documents accumulate. Watches for the JSONL writer
to finish (no new data for 60s + file not growing), then writes
the final validation shard and _COMPLETE sentinel.

Usage:
    python -m scripts.data.stream_jsonl_to_parquet \
        --input data/cpp_chunked_all.jsonl \
        --parquet_dir data/cpp_chunked_parquet \
        --rows_per_file 50000
"""
import argparse
import json
import os
import random
import time

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL (may still be growing)")
    parser.add_argument("--parquet_dir", required=True, help="Output parquet directory")
    parser.add_argument("--rows_per_file", type=int, default=50000)
    parser.add_argument("--val_fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.parquet_dir, exist_ok=True)
    rng = random.Random(args.seed)

    shard_idx = 0
    batch = []
    total_docs = 0
    stale_count = 0
    last_pos = 0

    print(f"[stream_pq] Watching {args.input} -> {args.parquet_dir}")
    print(f"[stream_pq] Rows per shard: {args.rows_per_file}")

    # Wait for input file to exist
    while not os.path.exists(args.input):
        print(f"[stream_pq] Waiting for {args.input}...")
        time.sleep(10)

    with open(args.input) as f:
        while True:
            line = f.readline()
            if not line:
                # No more data right now — check if writer is done
                current_size = os.path.getsize(args.input)
                if current_size == last_pos:
                    stale_count += 1
                else:
                    stale_count = 0
                    last_pos = current_size

                # If no new data for 60s (4 checks * 15s), assume writer is done
                if stale_count >= 4:
                    print(f"[stream_pq] Input file stable for 60s, finishing up")
                    break
                time.sleep(15)
                continue

            stale_count = 0
            line = line.strip()
            if not line:
                continue

            text = json.loads(line)["text"]
            batch.append(text)
            total_docs += 1

            if len(batch) >= args.rows_per_file:
                rng.shuffle(batch)
                table = pa.table({"text": batch})
                path = os.path.join(args.parquet_dir, f"shard_{shard_idx:05d}.parquet")
                pq.write_table(table, path, row_group_size=1024)
                print(f"[stream_pq] Written shard_{shard_idx:05d}.parquet ({len(batch)} docs, {total_docs:,} total)")
                shard_idx += 1
                batch = []

    # Flush remaining as train shard (keeping val_fraction for val)
    val_count = max(1, int(total_docs * args.val_fraction))
    if len(batch) > val_count:
        train_part = batch[:-val_count]
        val_part = batch[-val_count:]
    else:
        train_part = batch
        val_part = []

    if train_part:
        rng.shuffle(train_part)
        table = pa.table({"text": train_part})
        path = os.path.join(args.parquet_dir, f"shard_{shard_idx:05d}.parquet")
        pq.write_table(table, path, row_group_size=1024)
        print(f"[stream_pq] Written final train shard_{shard_idx:05d}.parquet ({len(train_part)} docs)")
        shard_idx += 1

    if val_part:
        val_table = pa.table({"text": val_part})
        val_path = os.path.join(args.parquet_dir, "val_shard.parquet")
        pq.write_table(val_table, val_path, row_group_size=1024)
        print(f"[stream_pq] Written val_shard.parquet ({len(val_part)} docs)")

    # Write _COMPLETE sentinel
    sentinel = os.path.join(args.parquet_dir, "_COMPLETE")
    with open(sentinel, "w") as sf:
        sf.write(f"{shard_idx} train shards, {len(val_part)} val docs, {total_docs} total\n")

    print(f"[stream_pq] DONE: {shard_idx} train shards, {total_docs:,} total docs, _COMPLETE written")


if __name__ == "__main__":
    main()
