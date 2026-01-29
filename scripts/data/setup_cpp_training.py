"""
Prepare C++ data for nanochat training pipeline.

1. Convert cpp_clean.jsonl → parquet files (with 'text' column)
2. Install our C++ tokenizer as the nanochat tokenizer
3. Generate token_bytes.pt for BPB evaluation

Usage:
    python -m scripts.data.setup_cpp_training [--input data/cpp_clean.jsonl] [--tokenizer_dir data/cpp_tokenizer]
"""
import os
import sys
import json
import argparse
import random

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from nanochat.common import get_base_dir
from nanochat.cpp_tokenizer import CppTokenizer


def jsonl_to_parquets(input_path: str, output_dir: str, rows_per_file: int = 50_000,
                      val_fraction: float = 0.01):
    """Convert JSONL to parquet files with train/val split."""
    os.makedirs(output_dir, exist_ok=True)

    # Read all texts
    print(f"Reading {input_path}...")
    texts = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                texts.append(record['text'])

    print(f"Total documents: {len(texts):,}")

    # Shuffle for randomness
    random.seed(42)
    random.shuffle(texts)

    # Split train/val
    val_count = max(1, int(len(texts) * val_fraction))
    val_texts = texts[-val_count:]
    train_texts = texts[:-val_count]
    print(f"Train: {len(train_texts):,} docs, Val: {len(val_texts):,} docs")

    # Write train shards
    shard_idx = 0
    for start in range(0, len(train_texts), rows_per_file):
        batch = train_texts[start:start + rows_per_file]
        table = pa.table({'text': batch})
        path = os.path.join(output_dir, f"shard_{shard_idx:05d}.parquet")
        pq.write_table(table, path, row_group_size=1024)
        shard_idx += 1

    # Write val as the last shard (nanochat convention: last parquet = val)
    val_table = pa.table({'text': val_texts})
    val_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.parquet")
    pq.write_table(val_table, val_path, row_group_size=1024)

    print(f"Written {shard_idx} train shards + 1 val shard to {output_dir}")
    return shard_idx + 1


def install_tokenizer(tokenizer_dir: str, base_dir: str):
    """Copy our tokenizer to nanochat's expected location and generate token_bytes.pt."""
    target_dir = os.path.join(base_dir, "tokenizer")
    os.makedirs(target_dir, exist_ok=True)

    # Load our tokenizer
    tok = CppTokenizer(tokenizer_dir)
    print(f"Loaded C++ tokenizer: {tok.vocab_size} tokens")

    # Save as HuggingFace format (tokenizer.json) — it's already in this format
    src_path = os.path.join(tokenizer_dir, "tokenizer.json")
    dst_path = os.path.join(target_dir, "tokenizer.json")
    import shutil
    shutil.copy2(src_path, dst_path)
    print(f"Copied tokenizer.json to {dst_path}")

    # Also copy the fixed_vocab for reference
    fixed_src = os.path.join(tokenizer_dir, "fixed_vocab.json")
    if os.path.exists(fixed_src):
        shutil.copy2(fixed_src, os.path.join(target_dir, "fixed_vocab.json"))

    # Generate token_bytes.pt
    # For each token, count the number of UTF-8 bytes it represents.
    # Special tokens (angle bracket format) get 0 bytes.
    vocab = tok._vocab
    id_to_token = tok._id_to_token
    vocab_size = tok.vocab_size

    token_bytes = []
    for tid in range(vocab_size):
        token_str = id_to_token.get(tid, "")
        # Special/reserved tokens: 0 bytes
        if token_str.startswith("<") and token_str.endswith(">"):
            token_bytes.append(0)
        elif not token_str:
            token_bytes.append(0)
        else:
            token_bytes.append(len(token_str.encode("utf-8")))

    token_bytes_tensor = torch.tensor(token_bytes, dtype=torch.int32)
    token_bytes_path = os.path.join(target_dir, "token_bytes.pt")
    torch.save(token_bytes_tensor, token_bytes_path)
    print(f"Saved token_bytes.pt ({vocab_size} entries) to {token_bytes_path}")

    # Stats
    nonzero = token_bytes_tensor[token_bytes_tensor > 0].float()
    print(f"  Non-special tokens: {len(nonzero)}")
    print(f"  Bytes/token: mean={nonzero.mean():.2f}, min={nonzero.min()}, max={nonzero.max()}")

    # Generate a pickle version for RustBPETokenizer compatibility
    # Actually, the training uses get_tokenizer() which returns RustBPETokenizer.
    # We need to make get_tokenizer() return our CppTokenizer instead.
    # For now, let's create a wrapper pickle file.
    print("\nNOTE: You need to modify nanochat/tokenizer.py get_tokenizer() to use CppTokenizer")
    print("  or set NANOCHAT_TOKENIZER=cpp environment variable")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/cpp_clean.jsonl')
    parser.add_argument('--tokenizer_dir', default='data/cpp_tokenizer')
    parser.add_argument('--rows_per_file', type=int, default=50_000)
    parser.add_argument('--val_fraction', type=float, default=0.01)
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    input_path = os.path.join(project_root, args.input)
    tokenizer_dir = os.path.join(project_root, args.tokenizer_dir)

    base_dir = get_base_dir()
    data_dir = os.path.join(base_dir, "base_data")
    print(f"nanochat base dir: {base_dir}")
    print(f"Data output: {data_dir}")
    print()

    # Step 1: Convert JSONL → parquet
    print("=== Step 1: Convert JSONL to Parquet ===")
    jsonl_to_parquets(input_path, data_dir, args.rows_per_file, args.val_fraction)
    print()

    # Step 2: Install tokenizer
    print("=== Step 2: Install C++ Tokenizer ===")
    install_tokenizer(tokenizer_dir, base_dir)
    print()

    print("=== Setup Complete ===")
    print(f"Data: {data_dir}")
    print(f"Tokenizer: {os.path.join(base_dir, 'tokenizer')}")
    print()
    print("Next: run training with:")
    print("  python -m scripts.base_train --depth=4 --num_iterations=100 --device_batch_size=4")


if __name__ == '__main__':
    main()
