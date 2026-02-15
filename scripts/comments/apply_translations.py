#!/usr/bin/env python3
"""Apply translated comments back to C++ source files.

Usage:
    python apply_translations.py \
        --input comments_translated_final.jsonl \
        [--validate] [--dry-run] [--backup-dir /path/to/backup]

Reads translation records and replaces original comments in source files.
Uses a multi-strategy matching approach:
  1. Byte-exact match at recorded offset
  2. Proximity search (±512 bytes) for shifted offsets
  3. Full-file string search as last resort
Applies replacements from end→start within each file to avoid offset drift.
"""

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict


def find_comment_in_source(source_str, original_text, byte_start, byte_end):
    """Find the position of original_text in source, using multiple strategies.

    Returns (start_idx, end_idx) in the string, or None if not found.
    original_text includes comment markers (// or /* */).
    """
    src_len = len(source_str)

    # Strategy 1: byte-exact match
    # Note: byte offsets may differ from char offsets for non-ASCII files,
    # but we try it as chars first since most C++ is ASCII
    if byte_start < src_len and byte_end <= src_len:
        extracted = source_str[byte_start:byte_end]
        if extracted == original_text:
            return (byte_start, byte_end)

    # Strategy 2: proximity search (±512 chars around expected position)
    search_radius = 512
    search_start = max(0, byte_start - search_radius)
    search_end = min(src_len, byte_end + search_radius)
    search_region = source_str[search_start:search_end]

    pos = search_region.find(original_text)
    if pos >= 0:
        abs_start = search_start + pos
        abs_end = abs_start + len(original_text)
        return (abs_start, abs_end)

    # Strategy 3: full-file search (only if unique)
    pos = source_str.find(original_text)
    if pos >= 0:
        # Check uniqueness
        second = source_str.find(original_text, pos + 1)
        if second < 0:
            # Unique match
            return (pos, pos + len(original_text))
        else:
            # Ambiguous — pick the one closest to expected byte_start
            candidates = [pos]
            while second >= 0:
                candidates.append(second)
                second = source_str.find(original_text, second + 1)
            # Pick closest to byte_start
            best = min(candidates, key=lambda p: abs(p - byte_start))
            return (best, best + len(original_text))

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Apply translated comments to C++ files"
    )
    parser.add_argument("--input", required=True, help="Input JSONL with translations")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate that comments can be found, then exit (no modifications)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )
    parser.add_argument(
        "--backup-dir",
        help="Save original files here before modifying",
    )
    args = parser.parse_args()

    # Load translations
    records = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    print(f"Loaded {len(records):,} translation records")

    # Group by file_path
    by_file = defaultdict(list)
    for rec in records:
        by_file[rec["file_path"]].append(rec)

    print(f"Spanning {len(by_file):,} files")

    # Stats
    total_exact = 0
    total_proximity = 0
    total_fullsearch = 0
    total_notfound = 0
    total_applied = 0
    total_skipped = 0
    modified_files = 0

    for file_path, recs in sorted(by_file.items()):
        try:
            source_bytes = open(file_path, "rb").read()
        except OSError as e:
            print(f"  ERROR reading {file_path}: {e}")
            total_skipped += len(recs)
            continue

        try:
            source_str = source_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # Try latin-1 as fallback (preserves all bytes)
            source_str = source_bytes.decode("latin-1")

        if args.validate:
            # Validation mode: just check if we can find each comment
            for rec in recs:
                original = rec["original_text"]
                result = find_comment_in_source(
                    source_str, original, rec["byte_start"], rec["byte_end"]
                )
                if result is None:
                    print(f"  NOT FOUND: {rec['id']} original={original[:60]!r}")
                    total_notfound += 1
                else:
                    start, end = result
                    if start == rec["byte_start"] and end == rec["byte_end"]:
                        total_exact += 1
                    elif abs(start - rec["byte_start"]) <= 512:
                        total_proximity += 1
                    else:
                        total_fullsearch += 1
            continue

        # Apply mode: find, match, replace
        # Build list of (char_start, char_end, new_text) replacements
        replacements = []

        for rec in recs:
            translated = rec.get("translated_text", rec.get("translated", ""))

            # Skip if translated is a boolean (corrupt merge data)
            if isinstance(translated, bool) or not isinstance(translated, str):
                print(f"  WARNING: invalid translated field for {rec['id']}, skipping")
                total_skipped += 1
                continue

            # Unescape literal \n and \t from Gemini output
            if "\\n" in translated:
                translated = translated.replace("\\n", "\n")
            if "\\t" in translated:
                translated = translated.replace("\\t", "\t")

            # Skip empty translations
            if not translated.strip():
                print(f"  WARNING: empty translation for {rec['id']}, skipping")
                total_skipped += 1
                continue

            # Validate pure ASCII
            if not translated.isascii():
                print(f"  WARNING: non-ASCII in translation for {rec['id']}, skipping")
                total_skipped += 1
                continue

            original = rec["original_text"]

            # Find the comment in source
            pos = find_comment_in_source(
                source_str, original, rec["byte_start"], rec["byte_end"]
            )
            if pos is None:
                print(
                    f"  WARNING: cannot find comment for {rec['id']}: {original[:50]!r}"
                )
                total_skipped += 1
                continue

            char_start, char_end = pos

            # Reconstruct full comment with markers
            if rec["comment_type"] == "line":
                new_text = f"//{translated}"
            elif rec["comment_type"] == "block":
                new_text = f"/*{translated}*/"
            else:
                print(
                    f"  WARNING: unknown comment_type {rec['comment_type']!r} for {rec['id']}"
                )
                total_skipped += 1
                continue

            replacements.append((char_start, char_end, new_text))
            total_applied += 1

        if not replacements:
            continue

        # Sort by start position descending (apply from end to avoid drift)
        replacements.sort(key=lambda r: r[0], reverse=True)

        # Check for overlaps
        for i in range(len(replacements) - 1):
            if replacements[i + 1][1] > replacements[i][0]:
                print(
                    f"  WARNING: overlapping replacements in {file_path}, skipping overlap"
                )

        # Backup original file if requested
        if args.backup_dir and not args.dry_run:
            # Find relative path from cpp_raw
            parts = file_path.split("/")
            for i, p in enumerate(parts):
                if p == "cpp_raw":
                    rel = "/".join(parts[i + 1 :])
                    break
            else:
                rel = os.path.basename(file_path)
            backup_path = os.path.join(args.backup_dir, rel)
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            shutil.copy2(file_path, backup_path)

        # Apply replacements (end→start)
        result = source_str
        for char_start, char_end, new_text in replacements:
            result = result[:char_start] + new_text + result[char_end:]

        if args.dry_run:
            print(f"  [DRY RUN] Would modify {file_path} ({len(replacements)} replacements)")
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(result)

        modified_files += 1

    # Summary
    prefix = "[DRY RUN] " if args.dry_run else ""
    if args.validate:
        total_found = total_exact + total_proximity + total_fullsearch
        total_checked = total_found + total_notfound
        print(f"\nValidation results ({total_checked:,} records):")
        print(f"  Byte-exact match:  {total_exact:,}")
        print(f"  Proximity match:   {total_proximity:,}")
        print(f"  Full-file search:  {total_fullsearch:,}")
        print(f"  Not found:         {total_notfound:,}")
        print(f"  Match rate:        {100 * total_found / max(total_checked, 1):.1f}%")
        if total_notfound > 0:
            sys.exit(1)
    else:
        print(f"\n{prefix}Results:")
        print(f"  Applied:  {total_applied:,} translations across {modified_files:,} files")
        print(f"  Skipped:  {total_skipped:,}")


if __name__ == "__main__":
    main()
