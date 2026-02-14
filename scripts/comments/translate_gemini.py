#!/usr/bin/env python3
"""Translate non-English C++ comments to English using Gemini structured output.

Usage:
    python translate_gemini.py \
        --input comments_to_translate.jsonl \
        --output comments_translated.jsonl \
        --chunk-tokens 200000 \
        --review-first 2

Requires: pip install google-genai pydantic
Env: GEMINI_API_KEY must be set
"""

import argparse
import json
import os
import sys
import time
from typing import Optional

from pydantic import BaseModel, Field


class TranslatedComment(BaseModel):
    """A single translated comment."""

    id: str = Field(description="Unique comment ID from input")
    translated: str = Field(description="English translation (ASCII only, no Unicode)")
    source_language: str = Field(description="Detected source language ISO 639-1 code")
    confidence: float = Field(
        description="Translation confidence 0.0 to 1.0", ge=0.0, le=1.0
    )


class TranslationBatch(BaseModel):
    """Batch of translated comments."""

    comments: list[TranslatedComment]


SYSTEM_PROMPT = """You are a C++ code comment translator. You translate non-English comments to English.

Rules:
1. Translate ONLY the comment text content. Do NOT include comment markers (// or /* */).
2. Output MUST be pure ASCII - no Unicode characters at all.
3. Preserve code identifiers, function names, variable names, type names EXACTLY as-is.
4. Keep technical terminology in English.
5. Preserve whitespace structure (leading spaces, newlines in block comments).
6. If the comment is a mix of English and another language, translate only the non-English parts.
7. For author names with accents, transliterate to ASCII (e.g., Michał → Michal).
8. For comments that are just punctuation or very short, provide a reasonable ASCII equivalent.
"""


def estimate_tokens(text: str) -> int:
    """Conservative token estimate: ~3 chars per token for CJK, ~4 for mixed."""
    non_ascii = sum(1 for c in text if ord(c) > 127)
    ascii_chars = len(text) - non_ascii
    # CJK chars are roughly 1 token each, ASCII ~4 chars per token
    return non_ascii + ascii_chars // 4


def chunk_records(
    records: list[dict], max_tokens: int, max_records: int = 500
) -> list[list[dict]]:
    """Split records into chunks that fit within token and record limits.

    max_records caps each chunk to avoid exceeding the 64K output token limit
    (each translation record is ~100-200 output tokens in JSON).
    """
    chunks = []
    current_chunk = []
    current_tokens = 0

    for rec in records:
        tokens = estimate_tokens(rec["content"])
        tokens += 20  # overhead for prompt formatting

        if current_chunk and (
            current_tokens + tokens > max_tokens or len(current_chunk) >= max_records
        ):
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0

        current_chunk.append(rec)
        current_tokens += tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def build_prompt(records: list[dict]) -> str:
    """Build the translation prompt for a batch of comments."""
    lines = []
    for rec in records:
        lines.append(f"ID: {rec['id']}")
        lines.append(f"Content: {rec['content']}")
        lines.append("")

    return (
        f"Translate the following {len(records)} non-English C++ comment texts to English.\n"
        f"Return a JSON object with a 'comments' array.\n\n" + "\n".join(lines)
    )


def translate_batch(
    client,
    records: list[dict],
    model_name: str,
    max_retries: int = 3,
) -> Optional[list[TranslatedComment]]:
    """Translate a batch via Gemini with structured output and retry logic."""
    prompt = build_prompt(records)

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    "system_instruction": SYSTEM_PROMPT,
                    "response_mime_type": "application/json",
                    "response_schema": TranslationBatch,
                },
            )

            # Parse structured output
            if hasattr(response, "parsed") and response.parsed is not None:
                batch = response.parsed
            else:
                # Fallback: parse from text
                batch = TranslationBatch.model_validate_json(response.text)

            # Validate all translations are ASCII
            for t in batch.comments:
                if not t.translated.isascii():
                    print(
                        f"  WARNING: Non-ASCII in translation for {t.id}, cleaning..."
                    )
                    t.translated = t.translated.encode(
                        "ascii", errors="replace"
                    ).decode("ascii")

            return batch.comments

        except Exception as e:
            wait = 2 ** (attempt + 1)
            print(f"  Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  FAILED after {max_retries} attempts")
                return None


def main():
    parser = argparse.ArgumentParser(description="Translate C++ comments via Gemini")
    parser.add_argument(
        "--input", required=True, help="Input JSONL (translate records only)"
    )
    parser.add_argument(
        "--output", required=True, help="Output JSONL with translations"
    )
    parser.add_argument(
        "--chunk-tokens", type=int, default=200_000, help="Max input tokens per chunk"
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=500,
        help="Max records per chunk (output token limit)",
    )
    parser.add_argument(
        "--model", default="gemini-3-flash-preview", help="Gemini model name"
    )
    parser.add_argument(
        "--review-first",
        type=int,
        default=0,
        help="Translate only first N chunks, then stop for review",
    )
    parser.add_argument(
        "--delay", type=float, default=2.0, help="Seconds between API calls"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show chunks without calling API"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume: skip records already in output file",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: GEMINI_API_KEY not set")
        sys.exit(1)

    # Load records
    records = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Only process translate records
            if rec.get("classification") == "translate":
                records.append(rec)

    print(f"Loaded {len(records)} translate records")

    # Resume: filter out already-translated records
    if args.resume and os.path.exists(args.output):
        done_ids = set()
        with open(args.output) as f:
            for line in f:
                line = line.strip()
                if line:
                    done_ids.add(json.loads(line)["id"])
        before = len(records)
        records = [r for r in records if r["id"] not in done_ids]
        print(f"Resume: {before - len(records)} already done, {len(records)} remaining")

    if not records:
        print("No records to translate")
        sys.exit(0)

    # Chunk
    chunks = chunk_records(records, args.chunk_tokens, args.max_records)
    print(f"Split into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        total_tokens = sum(estimate_tokens(r["content"]) for r in chunk)
        print(f"  Chunk {i}: {len(chunk)} records, ~{total_tokens:,} tokens")

    if args.dry_run:
        print("\n[DRY RUN] Would translate these chunks. Exiting.")
        return

    # Initialize Gemini client
    from google import genai

    client = genai.Client(api_key=api_key)

    # Translate
    max_chunks = args.review_first if args.review_first > 0 else len(chunks)
    total_translated = 0
    total_failed = 0

    open_mode = "a" if args.resume else "w"
    with open(args.output, open_mode) as out_f:
        for chunk_idx, chunk in enumerate(chunks[:max_chunks]):
            print(
                f"\nTranslating chunk {chunk_idx + 1}/{max_chunks} ({len(chunk)} comments)..."
            )

            translations = translate_batch(client, chunk, args.model)

            if translations is None:
                # Save failed chunk
                failed_path = args.output.replace(
                    ".jsonl", f"_failed_{chunk_idx}.jsonl"
                )
                with open(failed_path, "w") as ff:
                    for rec in chunk:
                        ff.write(json.dumps(rec) + "\n")
                print(f"  Saved failed chunk to {failed_path}")
                total_failed += len(chunk)
                continue

            # Build lookup from translations
            trans_by_id = {t.id: t for t in translations}

            # Merge with original records
            for rec in chunk:
                t = trans_by_id.get(rec["id"])
                if t:
                    output_rec = {
                        "id": rec["id"],
                        "file_path": rec["file_path"],
                        "byte_start": rec["byte_start"],
                        "byte_end": rec["byte_end"],
                        "comment_type": rec["comment_type"],
                        "original_text": rec["original_text"],
                        "content": rec["content"],
                        "translated": t.translated,
                        "source_language": t.source_language,
                        "confidence": t.confidence,
                    }
                    out_f.write(json.dumps(output_rec) + "\n")
                    total_translated += 1
                else:
                    print(f"  WARNING: No translation for {rec['id']}")
                    total_failed += 1

            # Rate limit
            if chunk_idx < max_chunks - 1:
                time.sleep(args.delay)

    print(f"\nDone: {total_translated} translated, {total_failed} failed")
    print(f"Output: {args.output}")

    if args.review_first > 0 and args.review_first < len(chunks):
        print(
            f"\n*** REVIEW MODE: Only translated first {args.review_first} chunks ***"
        )
        print(
            f"*** Remaining {len(chunks) - args.review_first} chunks pending review ***"
        )
        print("*** Re-run without --review-first to translate all ***")


if __name__ == "__main__":
    main()
