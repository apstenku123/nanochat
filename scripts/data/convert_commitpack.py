"""Convert CommitPack C++ JSONL files to nanochat training JSONL format.

Reads CommitPack's (old_contents, new_contents, subject, message) format
and produces our evolution training documents.

Usage:
    python3 scripts/data/convert_commitpack.py \
        --input_dir ~/data/commitpack_cpp/data/c++ \
        --output /mnt/nvme/nanochat_data/commitpack_cpp_training.jsonl \
        --max_tokens 16384
"""

import argparse
import glob
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="CommitPack C++ JSONL dir")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--max_tokens", type=int, default=16384)
    args = parser.parse_args()

    jsonl_files = sorted(
        glob.glob(os.path.join(args.input_dir, "**/*.jsonl"), recursive=True)
    )
    if not jsonl_files:
        jsonl_files = sorted(glob.glob(os.path.join(args.input_dir, "*.jsonl")))
    print(f"Found {len(jsonl_files)} JSONL files")

    total = 0
    skipped = 0
    errors = 0

    with open(args.output, "w") as out:
        for jf in jsonl_files:
            file_docs = 0
            with open(jf) as inp:
                for line_num, line in enumerate(inp):
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        errors += 1
                        continue

                    subject = str(row.get("subject", ""))
                    message = str(row.get("message", ""))
                    old_contents = str(row.get("old_contents", ""))
                    new_contents = str(row.get("new_contents", ""))
                    old_file = str(row.get("old_file", ""))
                    new_file = str(row.get("new_file", ""))
                    repos = str(row.get("repos", ""))

                    if len(old_contents) < 50 or len(new_contents) < 50:
                        skipped += 1
                        continue
                    if len(old_contents) > 200000 or len(new_contents) > 200000:
                        skipped += 1
                        continue

                    subj_lower = subject.lower()
                    if any(
                        s in subj_lower
                        for s in [
                            "merge branch",
                            "merge pull",
                            "bump version",
                            "auto-generated",
                        ]
                    ):
                        skipped += 1
                        continue

                    repo_name = repos.split(",")[0].strip() if repos else "unknown"
                    filepath = new_file or old_file

                    header = (
                        f"// Repository: {repo_name}\n"
                        f"// File: {filepath}\n"
                        f"// Change: {subject}"
                    )
                    if message and message != subject:
                        for msg_line in message.split("\n")[:3]:
                            header += f"\n// {msg_line}"

                    doc = (
                        f"{header}\n\n"
                        f"// === BEFORE ===\n{old_contents}\n\n"
                        f"// === AFTER ===\n{new_contents}"
                    )

                    tokens = len(doc) // 4
                    if tokens <= args.max_tokens:
                        out.write(json.dumps({"text": doc}) + "\n")
                        total += 1
                        file_docs += 1
                    else:
                        skipped += 1

                    if total % 100000 == 0 and total > 0:
                        print(f"    {total:,} docs so far...")

            print(f"  {os.path.basename(jf)}: {file_docs:,} docs")

    size_gb = os.path.getsize(args.output) / (1024**3)
    print(f"\nTotal: {total:,} docs ({skipped:,} skipped, {errors} errors)")
    print(f"Output: {args.output} ({size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
