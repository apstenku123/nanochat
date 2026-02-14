"""Concatenate compilable documents from the same project into larger training docs.

The tree-sitter compilable chunker produces per-file documents (~2-30K tokens).
For long-context training (128K, 256K, 512K), we need to combine multiple
related compilation units from the same project into larger documents.

Strategy:
  1. Group documents by project (extracted from preamble/filepath in text)
  2. Within each project, concatenate docs in order until reaching target token limit
  3. Add separator comments between concatenated units

Usage:
    python3 scripts/data/concat_compilable_docs.py \
        --input treesitter_compilable_16k.jsonl \
        --output treesitter_compilable_128k.jsonl \
        --max_tokens 131072
"""

import argparse
import json
import os
import re
import random
from collections import defaultdict


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def extract_project_name(text: str) -> str:
    """Extract project name from compilable document text."""
    # Look for patterns like "// Project: xyz" or "// File: xyz/..."
    # or filepath patterns in the preamble
    for line in text.split("\n")[:10]:
        m = re.match(r"// (?:Project|File): (\S+)", line)
        if m:
            path = m.group(1)
            # Extract top-level project directory
            parts = path.split("/")
            return parts[0] if parts else "unknown"

    # Try to find #include patterns to identify project
    includes = re.findall(r'#include\s*[<"]([^>"]+)[>"]', text[:2000])
    if includes:
        # Use the first include path's top directory
        for inc in includes:
            parts = inc.split("/")
            if len(parts) > 1:
                return parts[0]

    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True, help="Input JSONL file (small compilable docs)"
    )
    parser.add_argument(
        "--output", required=True, help="Output JSONL file (larger concatenated docs)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=131072,
        help="Target max tokens per output doc",
    )
    parser.add_argument(
        "--min_tokens",
        type=int,
        default=0,
        help="Min tokens per output doc (0 = auto, 25%% of max)",
    )
    parser.add_argument(
        "--separator",
        default="\n\n// ────────────────────────────────────────\n\n",
        help="Separator between concatenated docs",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.min_tokens == 0:
        args.min_tokens = args.max_tokens // 4

    rng = random.Random(args.seed)

    # Phase 1: Read all docs and group by project
    print(f"Reading {args.input}...")
    projects = defaultdict(list)
    total_input = 0

    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                text = json.loads(line)["text"]
            except (json.JSONDecodeError, KeyError):
                continue

            project = extract_project_name(text)
            projects[project].append(text)
            total_input += 1

    print(f"Read {total_input:,} docs from {len(projects):,} projects")

    # Phase 2: Concatenate docs within each project
    print(f"Concatenating to max {args.max_tokens:,} tokens per doc...")
    output_docs = []

    for project_name, docs in sorted(projects.items()):
        # Shuffle docs within project for variety
        rng.shuffle(docs)

        current_parts = []
        current_tokens = 0
        sep_tokens = estimate_tokens(args.separator)

        for doc in docs:
            doc_tokens = estimate_tokens(doc)

            # If adding this doc would exceed max, flush current
            if current_parts and (
                current_tokens + sep_tokens + doc_tokens > args.max_tokens
            ):
                combined = args.separator.join(current_parts)
                output_docs.append(combined)
                current_parts = []
                current_tokens = 0

            current_parts.append(doc)
            current_tokens += doc_tokens + (sep_tokens if len(current_parts) > 1 else 0)

        # Flush remaining
        if current_parts:
            combined = args.separator.join(current_parts)
            # Only include if meets minimum size (unless it's all we have from this project)
            if estimate_tokens(combined) >= args.min_tokens or len(docs) <= 3:
                output_docs.append(combined)

    # Phase 3: Write output
    print(f"Writing {len(output_docs):,} concatenated docs...")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    total_tokens = 0
    with open(args.output, "w") as f:
        for doc in output_docs:
            tokens = estimate_tokens(doc)
            total_tokens += tokens
            f.write(json.dumps({"text": doc}) + "\n")

    size_gb = os.path.getsize(args.output) / (1024**3)
    print("\nDONE:")
    print(f"  Input: {total_input:,} small docs")
    print(f"  Output: {len(output_docs):,} concatenated docs")
    print(f"  Size: {size_gb:.2f} GB")
    print(f"  Tokens: ~{total_tokens / 1e9:.1f}B")
    print(f"  Avg tokens/doc: {total_tokens // max(1, len(output_docs)):,}")


if __name__ == "__main__":
    main()
