"""
Pre-tokenize the dataset to binary files for fast training.

Usage:
    python -m scripts.pretokenize --num_workers=8

This reads parquet files, tokenizes them, and saves to .bin files that can be
memory-mapped during training for maximum throughput.
"""

import os
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial

import pyarrow.parquet as pq

from nanochat.common import get_base_dir, print0
from nanochat.dataset import list_parquet_files
from nanochat.tokenizer import get_tokenizer

# -----------------------------------------------------------------------------
# Configuration
base_dir = get_base_dir()
OUTPUT_DIR = os.path.join(base_dir, "tokenized_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
def tokenize_parquet_file(parquet_path, output_dir, tokenizer_batch_size=128):
    """
    Tokenize a single parquet file and save to binary.
    Returns the number of tokens processed.
    """
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()

    # Output filename: same name but .bin instead of .parquet
    basename = os.path.basename(parquet_path).replace('.parquet', '.bin')
    output_path = os.path.join(output_dir, basename)

    # Skip if already exists
    if os.path.exists(output_path):
        # Read header to get token count
        with open(output_path, 'rb') as f:
            header = np.frombuffer(f.read(8), dtype=np.uint64)
            num_tokens = int(header[0])
        print(f"Skipping {basename} (already exists, {num_tokens:,} tokens)")
        return num_tokens

    # Read all text from parquet file
    pf = pq.ParquetFile(parquet_path)
    all_tokens = []

    for rg_idx in range(pf.num_row_groups):
        rg = pf.read_row_group(rg_idx)
        texts = rg.column('text').to_pylist()

        # Tokenize in batches
        for i in range(0, len(texts), tokenizer_batch_size):
            batch = texts[i:i + tokenizer_batch_size]
            token_lists = tokenizer.encode(batch, prepend=bos_token, num_threads=4)
            for tokens in token_lists:
                all_tokens.extend(tokens)

    # Convert to numpy array (uint16 is enough for vocab_size < 65536)
    tokens_array = np.array(all_tokens, dtype=np.uint16)
    num_tokens = len(tokens_array)

    # Write to binary file with header
    # Header: 8 bytes for token count (uint64)
    # Data: tokens as uint16
    temp_path = output_path + '.tmp'
    with open(temp_path, 'wb') as f:
        # Write header
        header = np.array([num_tokens], dtype=np.uint64)
        f.write(header.tobytes())
        # Write tokens
        f.write(tokens_array.tobytes())

    # Atomic rename
    os.rename(temp_path, output_path)
    print(f"Tokenized {basename}: {num_tokens:,} tokens")
    return num_tokens


def process_file_wrapper(args):
    """Wrapper for multiprocessing."""
    parquet_path, output_dir, tokenizer_batch_size = args
    try:
        return tokenize_parquet_file(parquet_path, output_dir, tokenizer_batch_size)
    except Exception as e:
        print(f"Error processing {parquet_path}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize dataset to binary files")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--tokenizer_batch_size", type=int, default=128, help="Batch size for tokenization")
    args = parser.parse_args()

    # Get all parquet files
    parquet_paths = list_parquet_files()
    if not parquet_paths:
        print("No parquet files found. Run dataset download first.")
        return

    print(f"Found {len(parquet_paths)} parquet files")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Using {args.num_workers} workers")
    print()

    # Prepare arguments for each file
    work_items = [(p, OUTPUT_DIR, args.tokenizer_batch_size) for p in parquet_paths]

    # Process files in parallel
    if args.num_workers > 1:
        with Pool(processes=args.num_workers) as pool:
            token_counts = pool.map(process_file_wrapper, work_items)
    else:
        token_counts = [process_file_wrapper(item) for item in work_items]

    # Summary
    total_tokens = sum(token_counts)
    print()
    print(f"Done! Total tokens: {total_tokens:,}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Create a metadata file
    meta_path = os.path.join(OUTPUT_DIR, "meta.txt")
    with open(meta_path, 'w') as f:
        f.write(f"total_tokens: {total_tokens}\n")
        f.write(f"num_files: {len(parquet_paths)}\n")
        f.write(f"dtype: uint16\n")


if __name__ == "__main__":
    main()
