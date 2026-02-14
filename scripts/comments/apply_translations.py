#!/usr/bin/env python3
"""Apply translated comments back to C++ source files.

Usage:
    python apply_translations.py \
        --input comments_translated.jsonl \
        --backup-dir /path/to/backup \
        --validate

Reads translation records and replaces original comments in-place.
Applies replacements from end→start within each file to avoid offset drift.
"""

import argparse
import json
import os
import sys
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(
        description="Apply translated comments to C++ files"
    )
    parser.add_argument("--input", required=True, help="Input JSONL with translations")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate byte offsets match before applying",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )
    parser.add_argument(
        "--backup-dir",
        help="Directory containing backup files for post-apply diffing",
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

    # Validate byte offsets if requested
    if args.validate:
        print("\nValidating byte offsets...")
        errors = 0
        for file_path, recs in by_file.items():
            try:
                source = open(file_path, "rb").read()
            except OSError as e:
                print(f"  ERROR reading {file_path}: {e}")
                errors += len(recs)
                continue

            for rec in recs:
                if rec["byte_end"] > len(source):
                    print(
                        f"  OFFSET ERROR: {rec['id']} byte_end {rec['byte_end']} > file size {len(source)}"
                    )
                    errors += 1
                    continue

                extracted = source[rec["byte_start"] : rec["byte_end"]]
                try:
                    extracted_str = extracted.decode("utf-8")
                except UnicodeDecodeError:
                    print(f"  UTF8 ERROR at {rec['id']}")
                    errors += 1
                    continue

                if extracted_str != rec["original_text"]:
                    print(
                        f"  MISMATCH: {rec['id']} expected {rec['original_text'][:50]!r} got {extracted_str[:50]!r}"
                    )
                    errors += 1

        if errors > 0:
            print(f"\nVALIDATION FAILED: {errors} errors")
            sys.exit(1)
        print(f"All {len(records):,} records validated byte-exact")

    # Apply translations
    modified_files = 0
    total_applied = 0
    total_skipped = 0

    for file_path, recs in sorted(by_file.items()):
        try:
            source = open(file_path, "rb").read()
        except OSError as e:
            print(f"  ERROR reading {file_path}: {e}")
            total_skipped += len(recs)
            continue

        # Sort by byte_start descending (apply from end to avoid offset drift)
        recs.sort(key=lambda r: r["byte_start"], reverse=True)

        result = bytearray(source)
        file_changed = False

        for rec in recs:
            translated = rec["translated"]

            # Validate pure ASCII
            if not translated.isascii():
                print(f"  WARNING: non-ASCII in translation for {rec['id']}, skipping")
                total_skipped += 1
                continue

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

            new_bytes = new_text.encode("ascii")

            # Verify original text still matches
            if rec["byte_end"] > len(result):
                print(f"  WARNING: offset out of bounds for {rec['id']}")
                total_skipped += 1
                continue

            current = bytes(result[rec["byte_start"] : rec["byte_end"]])
            try:
                current_str = current.decode("utf-8")
            except UnicodeDecodeError:
                print(f"  WARNING: UTF-8 decode error for {rec['id']}")
                total_skipped += 1
                continue

            if current_str != rec["original_text"]:
                print(
                    f"  WARNING: content mismatch for {rec['id']} (file already modified?)"
                )
                total_skipped += 1
                continue

            # Apply replacement
            result[rec["byte_start"] : rec["byte_end"]] = new_bytes
            file_changed = True
            total_applied += 1

        if file_changed:
            if args.dry_run:
                print(f"  [DRY RUN] Would modify {file_path}")
            else:
                with open(file_path, "wb") as f:
                    f.write(result)
            modified_files += 1

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Results:")
    print(f"  Applied: {total_applied:,} translations across {modified_files:,} files")
    print(f"  Skipped: {total_skipped:,}")

    # Post-apply validation: diff against backup
    if args.backup_dir and not args.dry_run:
        print(f"\nDiffing against backup at {args.backup_dir}...")
        src_root = None
        # Find common root
        for file_path in by_file:
            # Try to determine the root from file paths
            parts = file_path.split("/")
            for i, p in enumerate(parts):
                if p == "cpp_raw":
                    src_root = "/".join(parts[: i + 1])
                    break
            if src_root:
                break

        if src_root:
            code_changes = 0
            for file_path in sorted(by_file):
                rel = os.path.relpath(file_path, src_root)
                bak = os.path.join(args.backup_dir, rel)
                if not os.path.exists(bak):
                    continue

                with open(bak, "rb") as f:
                    bak_bytes = f.read()
                with open(file_path, "rb") as f:
                    cur_bytes = f.read()

                if bak_bytes != cur_bytes:
                    code_changes += 1

            print(f"  Files changed from backup: {code_changes:,}")
        else:
            print("  Could not determine source root for diff")


if __name__ == "__main__":
    main()
