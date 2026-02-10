"""
Chunk C++ training data by function/class boundaries and write to parquet.

Reads JSONL files with {"text": "...", ...}, chunks each file into semantic
units (functions, classes, preambles), groups them into training-ready documents,
and writes output as JSONL + parquet shards.

Grouping strategies:
  - single_func: preamble + one complete function (40%)
  - class_block: preamble + full class definition (20%)
  - func_chain: preamble + 2-5 sequential functions (20%)
  - full_file: entire file as-is when small enough (20%)

Usage:
    # Process all data sources and build chunked parquet
    python -m scripts.data.chunk_cpp_data \
        --inputs data/cpp_combined_10b_v3.jsonl data/ms_src.jsonl \
        --output data/cpp_chunked.jsonl \
        --parquet_dir data/cpp_chunked_parquet \
        --max_tokens 1024
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.data.cpp_chunker import chunk_file, Chunk


def estimate_tokens(text: str) -> int:
    """Rough token count estimate: ~3.5 chars per token for code."""
    return max(1, len(text.encode('utf-8')) // 4)


def group_chunks(chunks: list[Chunk], max_tokens: int, rng: random.Random) -> list[str]:
    """Group chunks from a single file into training-ready documents.

    Returns list of document strings.
    """
    if not chunks:
        return []

    # Separate preamble from content chunks
    preamble_parts = []
    content_chunks = []
    for c in chunks:
        if c.kind == "preamble":
            preamble_parts.append(c.text)
        else:
            content_chunks.append(c)

    preamble = "\n\n".join(preamble_parts).strip()
    preamble_tokens = estimate_tokens(preamble) if preamble else 0

    # If no content chunks, emit preamble alone (if substantial)
    if not content_chunks:
        if preamble and preamble_tokens >= 20:
            return [preamble]
        return []

    # Check if full file fits
    full_text_parts = [preamble] if preamble else []
    full_text_parts.extend(c.text for c in content_chunks)
    full_text = "\n\n".join(full_text_parts)
    full_tokens = estimate_tokens(full_text)

    # Strategy: full_file if it fits (20% chance even if it's bigger, to get variety)
    if full_tokens <= max_tokens:
        return [full_text]

    # Split into individual documents
    documents = []

    # Classify content chunks
    functions = [c for c in content_chunks if c.kind == "function"]
    classes = [c for c in content_chunks if c.kind == "class"]
    others = [c for c in content_chunks if c.kind not in ("function", "class")]

    # Emit each class as its own document (with preamble)
    for cls in classes:
        parts = [preamble, cls.text] if preamble else [cls.text]
        doc = "\n\n".join(parts)
        doc_tokens = estimate_tokens(doc)
        if doc_tokens <= max_tokens * 2:  # allow some overflow for large classes
            documents.append(doc)
        else:
            # Class too big — emit without preamble, or just as-is
            documents.append(cls.text)

    # Group functions: try chains first, then singles
    if len(functions) >= 2:
        # Emit function chains (groups of 2-5 sequential functions)
        i = 0
        while i < len(functions):
            chain_size = min(rng.randint(2, 5), len(functions) - i)

            # Try the chain
            chain_funcs = functions[i:i + chain_size]
            chain_parts = [preamble] if preamble else []
            chain_parts.extend(f.text for f in chain_funcs)
            chain_text = "\n\n".join(chain_parts)
            chain_tokens = estimate_tokens(chain_text)

            if chain_tokens <= max_tokens:
                documents.append(chain_text)
                i += chain_size
            else:
                # Chain too big — emit singles
                for func in chain_funcs:
                    parts = [preamble, func.text] if preamble else [func.text]
                    doc = "\n\n".join(parts)
                    documents.append(doc)
                i += chain_size
    else:
        # Single functions
        for func in functions:
            parts = [preamble, func.text] if preamble else [func.text]
            doc = "\n\n".join(parts)
            documents.append(doc)

    # Emit other top-level chunks
    for other in others:
        if estimate_tokens(other.text) >= 20:
            parts = [preamble, other.text] if preamble else [other.text]
            doc = "\n\n".join(parts)
            if estimate_tokens(doc) <= max_tokens * 2:
                documents.append(doc)
            else:
                documents.append(other.text)

    return documents


def process_jsonl(input_path: str, max_tokens: int, rng: random.Random,
                  seen_hashes: set, stats: Counter, out_file,
                  max_records: int = 0) -> int:
    """Process a single JSONL file, chunking each record.

    Streams documents directly to out_file (JSONL) to avoid memory accumulation.
    Returns number of new documents written.
    """
    t0 = time.time()
    count = 0
    new_docs = 0

    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            count += 1
            if max_records > 0 and count > max_records:
                break

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats['json_error'] += 1
                continue

            text = record.get('text', '')
            if not text or len(text) < 50:
                stats['too_small'] += 1
                continue

            # Chunk the file
            try:
                chunks = chunk_file(text)
            except Exception:
                stats['chunk_error'] += 1
                # Fall back to full file
                chunks = [Chunk(kind="top_level", text=text)]

            if not chunks:
                stats['no_chunks'] += 1
                continue

            # Group into documents
            docs = group_chunks(chunks, max_tokens, rng)

            for doc in docs:
                # Dedup by content hash (binary digest saves memory vs hex)
                h = hashlib.md5(doc.encode()).digest()
                if h in seen_hashes:
                    stats['dedup'] += 1
                    continue
                seen_hashes.add(h)

                out_file.write(json.dumps({"text": doc}) + '\n')
                new_docs += 1
                stats['docs_out'] += 1

            stats['files_in'] += 1

            if count % 100000 == 0:
                elapsed = time.time() - t0
                rate = count / elapsed
                print(f"  {input_path}: {count:,} files → {stats['docs_out']:,} docs ({rate:.0f} files/sec)")
                out_file.flush()

    elapsed = time.time() - t0
    print(f"  {input_path}: {count:,} files → {new_docs:,} new docs in {elapsed:.0f}s")
    return new_docs


def write_parquet(jsonl_path: str, parquet_dir: str, rows_per_file: int = 50000):
    """Convert JSONL to parquet shards with train/val split.

    Streams from JSONL to avoid loading everything into memory.
    Shuffles within each shard; shard assignment is hash-based for
    deterministic distribution without a full in-memory shuffle.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    print(f"Converting to parquet: {parquet_dir}")
    os.makedirs(parquet_dir, exist_ok=True)

    # First pass: count total docs
    total = 0
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                total += 1
    print(f"  Total documents: {total:,}")

    val_count = max(1, int(total * 0.01))
    val_threshold = total - val_count  # docs after this index go to val

    # Second pass: stream into shards
    rng_pq = random.Random(42)
    shard_idx = 0
    batch = []
    val_batch = []
    doc_idx = 0

    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text = json.loads(line)['text']
            doc_idx += 1

            # Last val_count docs go to validation
            if doc_idx > val_threshold:
                val_batch.append(text)
            else:
                batch.append(text)

            if len(batch) >= rows_per_file:
                rng_pq.shuffle(batch)
                table = pa.table({'text': batch})
                path = os.path.join(parquet_dir, f"shard_{shard_idx:05d}.parquet")
                pq.write_table(table, path, row_group_size=1024)
                shard_idx += 1
                batch = []

    # Flush remaining train batch
    if batch:
        rng_pq.shuffle(batch)
        table = pa.table({'text': batch})
        path = os.path.join(parquet_dir, f"shard_{shard_idx:05d}.parquet")
        pq.write_table(table, path, row_group_size=1024)
        shard_idx += 1

    # Write val shard
    if val_batch:
        val_table = pa.table({'text': val_batch})
        val_path = os.path.join(parquet_dir, f"val_shard.parquet")
        pq.write_table(val_table, val_path, row_group_size=1024)

    print(f"  Written {shard_idx} train shards + 1 val shard ({len(val_batch):,} val docs)")


def main():
    parser = argparse.ArgumentParser(description="Chunk C++ data by function/class boundaries")
    parser.add_argument("--inputs", nargs='+', required=True,
                        help="Input JSONL files (will be processed in order)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL path")
    parser.add_argument("--parquet_dir", type=str, default="",
                        help="If set, also write parquet shards here")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Target max tokens per document")
    parser.add_argument("--max_records", type=int, default=0,
                        help="Max records to process per input (0 = all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    seen_hashes = set()
    stats = Counter()

    print(f"Processing {len(args.inputs)} input file(s), max_tokens={args.max_tokens}")

    # Stream documents directly to JSONL (no memory accumulation)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as out_f:
        for input_path in args.inputs:
            if not os.path.exists(input_path):
                print(f"WARNING: {input_path} not found, skipping")
                continue
            print(f"\nProcessing: {input_path} ({os.path.getsize(input_path)/1e9:.1f} GB)")
            process_jsonl(input_path, args.max_tokens, rng, seen_hashes, stats,
                          out_f, max_records=args.max_records)

    output_size = os.path.getsize(args.output) / 1e9

    # Estimate tokens by sampling first 100K lines of output
    sample_tokens = 0
    sample_count = 0
    with open(args.output) as f:
        for line in f:
            if sample_count >= 100000:
                break
            if line.strip():
                sample_tokens += estimate_tokens(json.loads(line)['text'])
                sample_count += 1

    if sample_count > 0 and stats['docs_out'] > 0:
        avg_tokens = sample_tokens / sample_count
        total_tokens_est = int(avg_tokens * stats['docs_out'])
    else:
        total_tokens_est = 0

    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"  Files processed: {stats['files_in']:,}")
    print(f"  Documents output: {stats['docs_out']:,}")
    print(f"  Estimated tokens: {total_tokens_est:,} (~{total_tokens_est/1e9:.1f}B)")
    print(f"  Deduplicated: {stats['dedup']:,}")
    print(f"  Chunk errors: {stats['chunk_error']:,}")
    print(f"  Output: {args.output} ({output_size:.2f} GB)")
    print(f"  Dedup hash memory: {len(seen_hashes):,} entries (~{len(seen_hashes)*50/1e6:.0f} MB)")
    print(f"{'='*60}")

    # Parquet conversion
    if args.parquet_dir:
        write_parquet(args.output, args.parquet_dir)

    # Token size distribution from sample
    if sample_count > 0:
        sizes = []
        with open(args.output) as f:
            for line in f:
                if len(sizes) >= 100000:
                    break
                if line.strip():
                    sizes.append(estimate_tokens(json.loads(line)['text']))
        sizes.sort()
        n = len(sizes)
        print(f"\nToken distribution (first {n:,} docs):")
        print(f"  p10={sizes[n//10]:,}  p25={sizes[n//4]:,}  p50={sizes[n//2]:,}  p75={sizes[3*n//4]:,}  p90={sizes[9*n//10]:,}")


if __name__ == "__main__":
    main()
