"""Collect C/C++ files from extracted MS SDK/DDK into JSONL format.

Usage:
    python3 scripts/data/collect_ms_src.py \
        --input_dir /mnt/nvme/nanochat_data/ms_src_extracted \
        --output /mnt/nvme/nanochat_data/ms_src_cpp.jsonl
"""

import argparse
import json
import os
from pathlib import Path

CPP_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".inl",
    ".inc",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min_size", type=int, default=50)
    parser.add_argument("--max_size", type=int, default=500000)
    args = parser.parse_args()

    total = 0
    errors = 0

    with open(args.output, "w") as out:
        for root, _, files in os.walk(args.input_dir):
            for fname in files:
                ext = Path(fname).suffix.lower()
                if ext not in CPP_EXTENSIONS:
                    continue

                filepath = os.path.join(root, fname)
                try:
                    size = os.path.getsize(filepath)
                    if size < args.min_size or size > args.max_size:
                        continue

                    with open(filepath, "r", errors="replace") as f:
                        text = f.read()

                    if len(text) < args.min_size:
                        continue

                    out.write(json.dumps({"text": text}) + "\n")
                    total += 1

                    if total % 50000 == 0:
                        print(f"  Collected {total:,} files...")
                except Exception:
                    errors += 1

    size_gb = os.path.getsize(args.output) / (1024**3)
    print(f"Total: {total:,} C/C++ files ({errors} errors)")
    print(f"Output: {args.output} ({size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
