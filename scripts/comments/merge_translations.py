#!/usr/bin/env python3
"""Merge multiple translation output files into a single canonical file.

Matches translated records to the current input file using:
1. Exact (file_path, byte_start) match
2. Proximity match (same file_path, byte_start within TOLERANCE bytes)

This handles the case where the input file was regenerated with slightly
different byte offsets after translations were produced.

Usage:
    python3 merge_translations.py \
        --input comments_to_translate.jsonl \
        --translations comments_translated.jsonl \
                       comments_translated_retry24.jsonl \
                       comments_translated_retry42.jsonl \
                       comments_translated_retry43.jsonl \
                       comments_translated_remaining.jsonl \
        --output comments_translated_merged.jsonl \
        --tolerance 10
"""
import argparse
import json
import sys
from bisect import bisect_left
from collections import defaultdict


def load_translations(paths):
    """Load all translation records, deduplicating by (file_path, byte_start).

    When duplicates exist, keep the record with higher confidence.
    """
    by_fp_bs = {}  # (file_path, byte_start) -> record
    total = 0
    dupes = 0

    for path in paths:
        count = 0
        try:
            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    total += 1
                    count += 1
                    key = (rec["file_path"], rec.get("byte_start"))
                    if key in by_fp_bs:
                        dupes += 1
                        existing = by_fp_bs[key]
                        if rec.get("confidence", 0) > existing.get("confidence", 0):
                            by_fp_bs[key] = rec
                    else:
                        by_fp_bs[key] = rec
        except FileNotFoundError:
            print("  WARNING: %s not found, skipping" % path, file=sys.stderr)
            continue
        print("  %s: %d records" % (path, count))

    print("  Total loaded: %d, unique: %d, duplicates resolved: %d" % (
        total, len(by_fp_bs), dupes))
    return by_fp_bs


def build_file_index(translations):
    """Build per-file sorted index for proximity matching."""
    by_file = defaultdict(list)  # file_path -> sorted list of (byte_start, record)
    for (fp, bs), rec in translations.items():
        if bs is not None:
            by_file[fp].append((bs, rec))
    for fp in by_file:
        by_file[fp].sort(key=lambda x: x[0])
    return by_file


def find_proximity_match(file_index, file_path, byte_start, tolerance):
    """Find the closest translated record within tolerance bytes."""
    if file_path not in file_index:
        return None

    entries = file_index[file_path]
    starts = [e[0] for e in entries]

    # Binary search for closest
    idx = bisect_left(starts, byte_start)

    best = None
    best_dist = tolerance + 1

    # Check idx-1, idx, idx+1
    for i in range(max(0, idx - 1), min(len(entries), idx + 2)):
        dist = abs(entries[i][0] - byte_start)
        if dist <= tolerance and dist < best_dist:
            best = entries[i][1]
            best_dist = dist

    return best


def merge(input_path, translation_paths, output_path, tolerance=10):
    """Merge translations against current input file."""
    print("Loading translations...")
    translations = load_translations(translation_paths)
    file_index = build_file_index(translations)

    print("\nMatching against input: %s" % input_path)
    exact = 0
    proximity = 0
    missing = 0
    total = 0
    non_ascii_content = 0
    low_confidence = 0

    matched_translation_keys = set()

    with open(input_path) as inf, open(output_path, "w") as outf:
        for line in inf:
            if not line.strip():
                continue
            rec = json.loads(line)
            total += 1
            fp = rec["file_path"]
            bs = rec["byte_start"]
            key = (fp, bs)

            # Try exact match first
            if key in translations:
                trans = translations[key]
                exact += 1
                matched_translation_keys.add(key)
            else:
                # Try proximity match
                trans = find_proximity_match(file_index, fp, bs, tolerance)
                if trans is not None:
                    proximity += 1
                    matched_translation_keys.add(
                        (trans["file_path"], trans.get("byte_start")))
                else:
                    missing += 1
                    continue

            # Emit merged record with the INPUT's IDs/offsets
            # but the TRANSLATION's translated text.
            # Handle multiple output formats:
            #   - Main Flash/retries: "translated" is a string (Gemini translation)
            #   - Pro retries: "translated" is bool True, "content" has translation
            #   - Parallel retries: "translated" may equal "content" (both are translation)
            raw_translated = trans.get("translated")
            if isinstance(raw_translated, str) and len(raw_translated) > 0:
                content = raw_translated
            else:
                content = trans.get("content", "")

            # Ensure ASCII
            if not content.isascii():
                content = content.encode("ascii", errors="replace").decode("ascii")
                non_ascii_content += 1

            confidence = trans.get("confidence", 0.0)
            if confidence < 0.5:
                low_confidence += 1

            merged = {
                "id": rec["id"],
                "file_path": fp,
                "byte_start": bs,
                "byte_end": rec["byte_end"],
                "comment_type": rec.get("comment_type", ""),
                # Use original_text (includes comment markers) for byte-exact
            # validation in apply_translations.py. Falls back to content
            # (inner text only) if original_text not available.
            "original_text": rec.get("original_text", rec.get("content", "")),
                "translated_text": content,
                "source_language": trans.get("source_language", "unknown"),
                "confidence": confidence,
            }
            outf.write(json.dumps(merged, ensure_ascii=False) + "\n")

    # Summary
    matched = exact + proximity
    print("\nResults:")
    print("  Input records:    %d" % total)
    print("  Exact match:      %d" % exact)
    print("  Proximity match:  %d (tolerance +-  %d bytes)" % (proximity, tolerance))
    print("  Missing:          %d" % missing)
    print("  Total matched:    %d (%.1f%%)" % (matched, matched / total * 100))
    print("\nQuality:")
    print("  Non-ASCII fixed:  %d" % non_ascii_content)
    print("  Low confidence:   %d (<0.5)" % low_confidence)
    print("\nOutput: %s (%d records)" % (output_path, matched))

    # Unused translations (in translation files but not matched to any input)
    unused = len(translations) - len(matched_translation_keys)
    if unused > 0:
        print("  Unused translations (no matching input): %d" % unused)

    return missing


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple translation files against current input")
    parser.add_argument("--input", required=True,
                        help="Current input JSONL (comments_to_translate.jsonl)")
    parser.add_argument("--translations", required=True, nargs="+",
                        help="Translation output JSONL files to merge")
    parser.add_argument("--output", required=True,
                        help="Merged output JSONL")
    parser.add_argument("--tolerance", type=int, default=10,
                        help="Byte proximity tolerance for matching (default: 10)")
    args = parser.parse_args()

    missing = merge(args.input, args.translations, args.output, args.tolerance)
    sys.exit(0 if missing == 0 else 1)


if __name__ == "__main__":
    main()
