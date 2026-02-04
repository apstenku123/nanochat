#!/usr/bin/env python3
"""Download The Stack v2 C++ subset from TempestTeam mirror to JSONL."""

import json
import time
from datasets import load_dataset

OUTPUT = "data/stack_v2_cpp.jsonl"
MAX_BYTES = 25_000_000_000  # 25GB target

print("Loading TempestTeam/dataset-the-stack-v2-dedup-sub C++ (streaming)...", flush=True)
ds = load_dataset(
    "TempestTeam/dataset-the-stack-v2-dedup-sub",
    "C++",
    split="train",
    streaming=True,
)

count = 0
skipped = 0
total_bytes = 0
t0 = time.time()

with open(OUTPUT, "w") as f:
    for row in ds:
        content = row.get("content", "")
        if not content or len(content) < 100:
            skipped += 1
            continue
        if len(content) > 1_000_000:  # Skip files > 1MB
            skipped += 1
            continue
        # Skip vendor/generated
        if row.get("is_vendor") or row.get("is_generated"):
            skipped += 1
            continue

        f.write(
            json.dumps(
                {
                    "text": content,
                    "path": row.get("path", ""),
                    "repo": row.get("repo_name", ""),
                }
            )
            + "\n"
        )
        total_bytes += len(content)
        count += 1

        if count % 100_000 == 0:
            elapsed = time.time() - t0
            rate = total_bytes / elapsed / 1e6
            print(
                f"  {count:,} files, {total_bytes/1e9:.2f} GB, "
                f"{rate:.1f} MB/s, skipped {skipped:,}",
                flush=True,
            )

        if total_bytes >= MAX_BYTES:
            break

elapsed = time.time() - t0
print(
    f"\nDone: {count:,} files, {total_bytes/1e9:.2f} GB, "
    f"{elapsed:.0f}s, skipped {skipped:,}",
    flush=True,
)
