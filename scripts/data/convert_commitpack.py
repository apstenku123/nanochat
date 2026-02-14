"""Convert CommitPack C++ JSONL files to raw commit records for cpp-chunker --commit-mode.

Reads CommitPack's (old_contents, new_contents, subject, message) format
and produces JSONL with {old_content, new_content, diff, subject, body, filepath, repo}
that the Rust cpp-chunker --commit-mode can process with tree-sitter.

Usage:
    python3 scripts/data/convert_commitpack.py \
        --input_dir ~/data/commitpack_cpp/data/c++ \
        --output /mnt/nvme/nanochat_data/commitpack_raw_commits.jsonl
"""

import argparse
import difflib
import glob
import json
import os
from pathlib import Path

# C/C++ file extensions
CPP_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".c++",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".h++",
    ".inl",
    ".inc",
    ".ipp",
    ".tcc",
    ".tpp",
}


def is_cpp_file(path: str) -> bool:
    return Path(path).suffix.lower() in CPP_EXTENSIONS


def generate_unified_diff(old_content: str, new_content: str, filepath: str) -> str:
    """Generate a unified diff from old and new content."""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    diff_lines = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{filepath}",
        tofile=f"b/{filepath}",
    )
    return "".join(diff_lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="CommitPack C++ JSONL dir")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    args = parser.parse_args()

    jsonl_files = sorted(
        glob.glob(os.path.join(args.input_dir, "**/*.jsonl"), recursive=True)
    )
    if not jsonl_files:
        jsonl_files = sorted(glob.glob(os.path.join(args.input_dir, "*.jsonl")))
    print(f"Found {len(jsonl_files)} JSONL files")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    total = 0
    skipped = 0
    errors = 0

    with open(args.output, "w") as out:
        for jf in jsonl_files:
            file_docs = 0
            with open(jf) as inp:
                for line in inp:
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

                    filepath = new_file or old_file
                    if not is_cpp_file(filepath):
                        skipped += 1
                        continue

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
                            "clang-format",
                            "fix whitespace",
                        ]
                    ):
                        skipped += 1
                        continue

                    # Generate unified diff
                    diff = generate_unified_diff(old_contents, new_contents, filepath)
                    if len(diff) < 50 or len(diff) > 50000:
                        skipped += 1
                        continue

                    repo_name = repos.split(",")[0].strip() if repos else "unknown"
                    body = message if message != subject else ""

                    record = {
                        "old_content": old_contents,
                        "new_content": new_contents,
                        "diff": diff,
                        "subject": subject,
                        "body": body,
                        "filepath": filepath,
                        "repo": repo_name,
                    }
                    out.write(json.dumps(record) + "\n")
                    total += 1
                    file_docs += 1

                    if total % 100000 == 0 and total > 0:
                        print(f"    {total:,} records so far...")

            print(f"  {os.path.basename(jf)}: {file_docs:,} records")

    size_gb = os.path.getsize(args.output) / (1024**3)
    print(f"\nTotal: {total:,} records ({skipped:,} skipped, {errors} errors)")
    print(f"Output: {args.output} ({size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
