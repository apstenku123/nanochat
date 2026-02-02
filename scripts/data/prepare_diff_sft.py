"""
Convert git diff JSONL into SFT training format for diff/patch generation.

Input: data/cpp_diffs.jsonl (from extract_diffs.py)
  {"repo", "commit_hash", "commit_msg", "diff", "files_changed", "insertions", "deletions"}

Output: data/diff_sft.jsonl (SFT format)
  {"instruction": "<commit_msg + file context>", "response": "<unified diff>"}

Uses two training formats mixed 50/50:
1. CarperAI-style: Given file context + commit msg → generate diff
2. FIM-style: Given before/after context → fill in the change

Special tokens used:
  <DIFF_START> (ID 15) - marks beginning of diff section
  <DIFF_END> (ID 16) - marks end of diff section
  Unified diff markers: diff, ---, +++, a/, b/ (IDs 1521-1526)
"""

import json
import sys
import os
import random
import re
from collections import Counter

def parse_diff_hunks(diff_text):
    """Parse unified diff into hunks with context."""
    hunks = []
    current_hunk = []
    current_header = None

    for line in diff_text.split('\n'):
        if line.startswith('@@'):
            if current_hunk and current_header:
                hunks.append((current_header, '\n'.join(current_hunk)))
            current_header = line
            current_hunk = []
        elif current_header is not None:
            current_hunk.append(line)

    if current_hunk and current_header:
        hunks.append((current_header, '\n'.join(current_hunk)))

    return hunks


def clean_commit_msg(msg):
    """Clean commit message: remove long descriptions, keep first line."""
    lines = msg.strip().split('\n')
    first_line = lines[0].strip()
    # Remove common prefixes like "commit abc123"
    first_line = re.sub(r'^(Signed-off-by|Reviewed-by|Acked-by|Tested-by):.*', '', first_line).strip()
    # Truncate to 200 chars
    if len(first_line) > 200:
        first_line = first_line[:200]
    return first_line


def format_carperai(sample):
    """Format as: instruction=commit_msg, response=diff."""
    msg = clean_commit_msg(sample['commit_msg'])
    files = ', '.join(sample['files_changed'][:5])  # Max 5 files listed

    instruction = f"Apply the following change to {files}:\n{msg}"
    response = sample['diff']

    # Truncate response if too long (keep it under 3000 chars for SFT)
    if len(response) > 3000:
        # Keep first 3000 chars, ending at a line boundary
        response = response[:3000].rsplit('\n', 1)[0]

    return {"instruction": instruction, "response": response}


def format_before_after(sample):
    """Format as: instruction=description+before_code, response=after_code.

    Extracts the before/after from the diff hunks.
    """
    msg = clean_commit_msg(sample['commit_msg'])
    diff = sample['diff']

    # Extract before lines (context + removed) and after lines (context + added)
    before_lines = []
    after_lines = []
    for line in diff.split('\n'):
        if line.startswith('---') or line.startswith('+++') or line.startswith('diff ') or line.startswith('index '):
            continue
        if line.startswith('@@'):
            before_lines.append(line)
            after_lines.append(line)
        elif line.startswith('-'):
            before_lines.append(line[1:])  # Remove the - prefix
        elif line.startswith('+'):
            after_lines.append(line[1:])  # Remove the + prefix
        elif line.startswith(' '):
            before_lines.append(line[1:])
            after_lines.append(line[1:])

    before_code = '\n'.join(before_lines[:100])  # Limit size
    after_code = '\n'.join(after_lines[:100])

    instruction = f"Fix the following C++ code:\n{msg}\n\nBefore:\n```cpp\n{before_code}\n```"
    response = f"```cpp\n{after_code}\n```"

    return {"instruction": instruction, "response": response}


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "data/cpp_diffs.jsonl"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/diff_sft.jsonl"

    random.seed(42)
    stats = Counter()
    kept = 0
    skipped = 0

    with open(input_path) as fin, open(output_path, 'w') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            # Quality filters
            diff = sample.get('diff', '')
            msg = sample.get('commit_msg', '')
            insertions = sample.get('insertions', 0)
            deletions = sample.get('deletions', 0)

            # Skip trivial or too-large diffs
            if insertions + deletions < 3:
                stats['too_small'] += 1
                skipped += 1
                continue
            if insertions + deletions > 300:
                stats['too_large'] += 1
                skipped += 1
                continue
            if len(diff) < 100:
                stats['short_diff'] += 1
                skipped += 1
                continue
            if len(diff) > 8000:
                stats['long_diff'] += 1
                skipped += 1
                continue

            # Skip empty/garbage commit messages
            first_line = msg.strip().split('\n')[0]
            if len(first_line) < 5:
                stats['short_msg'] += 1
                skipped += 1
                continue

            # 50/50 mix of formats
            if random.random() < 0.5:
                formatted = format_carperai(sample)
                stats['carperai'] += 1
            else:
                formatted = format_before_after(sample)
                stats['before_after'] += 1

            # Final length check
            total_len = len(formatted['instruction']) + len(formatted['response'])
            if total_len < 50 or total_len > 6000:
                stats['bad_length'] += 1
                skipped += 1
                continue

            fout.write(json.dumps(formatted) + '\n')
            kept += 1

    print(f"Processed: {kept + skipped} total, {kept} kept, {skipped} skipped")
    print(f"Stats: {dict(stats)}")
    print(f"Output: {output_path} ({kept} examples)")


if __name__ == "__main__":
    main()
