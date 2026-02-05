#!/usr/bin/env python3
"""
Clean docstring_pairs_full.jsonl for SFT training.

Filters applied:
1. Remove body < 50 chars (trivial/useless)
2. Remove Java-style docstrings (@param, @return, @throws, ByteBuffer, etc.)
3. Remove code-as-docstring (high ;{} count, low English words)
4. Remove docstring < 10 chars
5. Cap body at 4000 chars, docstring at 2000 chars
6. Deduplicate by (signature, first 100 chars of body)

Usage:
    python -m scripts.data.clean_docstring_pairs \
        --input data/docstring_pairs_full.jsonl \
        --output data/docstring_pairs_clean.jsonl
"""

import argparse
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Java / JNI / C# interop indicators in docstrings
JAVA_INDICATORS = [
    "@param", "@return", "@throws", "@exception", "@see", "@since",
    "@Override", "@Deprecated", "@SuppressWarnings",
    "public void", "public static", "private void", "protected void",
    ".flip()", ".compact()", ".remaining()", ".position()",
    "ByteBuffer", "ByteArray", "ArrayList", "HashMap",
    "getString(", "getInt(", "getBoolean(",
    "JNIEnv", "jobject", "jstring", "jclass", "jmethodID",
    "CallObjectMethod", "CallVoidMethod", "CallIntMethod",
    "GetStringUTFChars", "ReleaseStringUTFChars",
    "FindClass", "GetMethodID", "GetFieldID",
]


def is_java_docstring(docstring: str) -> bool:
    """Check if docstring contains Java/JNI/C# interop patterns."""
    for indicator in JAVA_INDICATORS:
        if indicator in docstring:
            return True
    return False


def is_code_as_docstring(docstring: str) -> bool:
    """Check if the docstring is actually code, not documentation."""
    code_chars = docstring.count(";") + docstring.count("{") + docstring.count("}")
    code_chars += docstring.count("()") + docstring.count("->")

    # Count English-like words (3+ alpha chars)
    words = [w for w in docstring.split() if w.isalpha() and len(w) >= 3]
    text_words = len(words)

    # High code density + low text = code-as-docstring
    if code_chars > 5 and text_words < 10:
        return True
    # Very high code density
    if code_chars > 15 and text_words < 20:
        return True
    return False


def clean_docstring_pairs(input_path: str, output_path: str,
                          min_body: int = 50, max_body: int = 4000,
                          min_docstring: int = 10, max_docstring: int = 2000):
    """Clean docstring pairs with quality filters and deduplication."""

    stats = {
        "total": 0,
        "short_body": 0,
        "long_body": 0,
        "short_docstring": 0,
        "java_docstring": 0,
        "code_as_docstring": 0,
        "empty_signature": 0,
        "duplicate": 0,
        "kept": 0,
    }

    seen = set()

    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            stats["total"] += 1

            if stats["total"] % 500_000 == 0:
                logger.info(f"  Processed {stats['total']:,} lines...")

            line = line.strip()
            if not line:
                continue

            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            docstring = d.get("docstring", "")
            signature = d.get("signature", "")
            body = d.get("body", "")

            # Filter: empty signature
            if not signature.strip():
                stats["empty_signature"] += 1
                continue

            # Filter: body length
            body = body.strip()
            if len(body) < min_body:
                stats["short_body"] += 1
                continue
            if len(body) > max_body:
                stats["long_body"] += 1
                continue

            # Filter: docstring length
            docstring = docstring.strip()
            if len(docstring) < min_docstring:
                stats["short_docstring"] += 1
                continue

            # Filter: Java-style docstrings
            if is_java_docstring(docstring):
                stats["java_docstring"] += 1
                continue

            # Filter: code-as-docstring
            if is_code_as_docstring(docstring):
                stats["code_as_docstring"] += 1
                continue

            # Truncate long fields
            if len(docstring) > max_docstring:
                docstring = docstring[:max_docstring]
            if len(body) > max_body:
                body = body[:max_body]

            # Deduplicate by (signature_normalized, body_prefix)
            sig_norm = " ".join(signature.split())
            dedup_key = (sig_norm, body[:100])
            if dedup_key in seen:
                stats["duplicate"] += 1
                continue
            seen.add(dedup_key)

            # Write cleaned record
            out = {
                "docstring": docstring,
                "signature": signature,
                "body": body,
                "path": d.get("path", ""),
            }
            fout.write(json.dumps(out) + "\n")
            stats["kept"] += 1

    # Report
    logger.info(f"\nCleaning complete:")
    logger.info(f"  Total input:       {stats['total']:>10,}")
    logger.info(f"  Short body (<{min_body}): {stats['short_body']:>10,}")
    logger.info(f"  Long body (>{max_body}): {stats['long_body']:>10,}")
    logger.info(f"  Short docstring:   {stats['short_docstring']:>10,}")
    logger.info(f"  Java docstring:    {stats['java_docstring']:>10,}")
    logger.info(f"  Code-as-docstring: {stats['code_as_docstring']:>10,}")
    logger.info(f"  Empty signature:   {stats['empty_signature']:>10,}")
    logger.info(f"  Duplicate:         {stats['duplicate']:>10,}")
    logger.info(f"  Kept:              {stats['kept']:>10,}")
    kept_pct = 100.0 * stats["kept"] / max(stats["total"], 1)
    logger.info(f"  Keep rate:         {kept_pct:.1f}%")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Clean docstring pairs for SFT")
    parser.add_argument("--input", type=str,
                        default="data/docstring_pairs_full.jsonl",
                        help="Input JSONL path")
    parser.add_argument("--output", type=str,
                        default="data/docstring_pairs_clean.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--min-body", type=int, default=50)
    parser.add_argument("--max-body", type=int, default=4000)
    parser.add_argument("--min-docstring", type=int, default=10)
    parser.add_argument("--max-docstring", type=int, default=2000)
    args = parser.parse_args()

    clean_docstring_pairs(
        args.input, args.output,
        min_body=args.min_body, max_body=args.max_body,
        min_docstring=args.min_docstring, max_docstring=args.max_docstring,
    )


if __name__ == "__main__":
    main()
