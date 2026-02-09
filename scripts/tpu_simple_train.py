#!/usr/bin/env python3
"""
Simple TPU training script for nanochat.
Designed for TPU v5e-8, avoids torch.compile overhead.
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

# TPU/XLA environment - must set before torch import
os.environ["PJRT_DEVICE"] = "TPU"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch

# Check TPU availability
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    IS_TPU = True
    print("TPU XLA imported successfully")
except ImportError:
    IS_TPU = False
    xm = None
    print("WARNING: torch_xla not available, running on CPU")

# Add nanochat to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer


def find_num_heads(model_dim: int, target_head_dim: int) -> int:
    """Find num_heads that divides model_dim evenly."""
    ideal = max(1, round(model_dim / target_head_dim))
    for offset in range(model_dim):
        for candidate in [ideal + offset, ideal - offset]:
            if candidate > 0 and model_dim % candidate == 0:
                return candidate
    return 1


def print0(*args, **kwargs):
    """Print only on master process."""
    if IS_TPU:
        xm.master_print(*args, **kwargs)
    else:
        print(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Simple TPU Training")
    parser.add_argument("--depth", type=int, default=12, help="Model depth")
    parser.add_argument("--aspect_ratio", type=int, default=64, help="Aspect ratio")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_iterations", type=int, default=100, help="Number of iterations")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="/tmp/nanochat_tpu", help="Output directory")
    args = parser.parse_args()

    print0("\n" + "="*60)
    print0("nanochat Simple TPU Training")
    print0("="*60)

    # Get device
    if IS_TPU:
        device = torch_xla.device()
        print0(f"Device: {device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print0(f"Device: {device}")

    # Load tokenizer
    print0("Loading tokenizer...")
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    print0(f"Vocab size: {vocab_size:,}")

    # Create model
    print0("Creating model...")
    num_layers = args.depth
    model_dim = args.depth * args.aspect_ratio
    num_heads = find_num_heads(model_dim, args.head_dim)

    config = GPTConfig(
        sequence_len=args.max_seq_len,
        vocab_size=vocab_size,
        n_layer=num_layers,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
    )

    model = GPT(config)
    model.to(device)
    model.init_weights()

    num_params = sum(p.numel() for p in model.parameters())
    print0(f"Parameters: {num_params:,}")
    print0(f"Layers: {num_layers}, Dim: {model_dim}, Heads: {num_heads}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )

    # Training loop
    print0("\nStarting training...")
    print0(f"Iterations: {args.num_iterations}")
    print0(f"Batch size: {args.batch_size}")
    print0(f"Sequence length: {args.max_seq_len}")

    model.train()
    start_time = time.time()
    total_tokens = 0

    for step in range(args.num_iterations):
        step_start = time.time()

        # Create random batch (replace with real data in production)
        x = torch.randint(0, vocab_size, (args.batch_size, args.max_seq_len), device=device)
        y = torch.randint(0, vocab_size, (args.batch_size, args.max_seq_len), device=device)

        # Forward pass
        loss = model(x, y)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # XLA sync
        if IS_TPU:
            xm.mark_step()

        tokens_per_step = args.batch_size * args.max_seq_len
        total_tokens += tokens_per_step
        step_time = time.time() - step_start

        if step % 10 == 0 or step == args.num_iterations - 1:
            elapsed = time.time() - start_time
            tok_per_sec = total_tokens / max(elapsed, 1e-6)
            print0(f"Step {step:05d}/{args.num_iterations} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Tok/s: {tok_per_sec:,.0f} | "
                  f"Time: {elapsed:.1f}s")

    total_time = time.time() - start_time

    # Save checkpoint
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, f"model_d{args.depth}.pt")

    if IS_TPU:
        xm.save(model.state_dict(), checkpoint_path)
    else:
        torch.save(model.state_dict(), checkpoint_path)

    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "total_time_seconds": total_time,
        "total_tokens": total_tokens,
        "tokens_per_second": total_tokens / total_time,
        "config": {
            "depth": args.depth,
            "model_dim": model_dim,
            "num_heads": num_heads,
            "vocab_size": vocab_size,
            "max_seq_len": args.max_seq_len,
            "batch_size": args.batch_size,
            "num_iterations": args.num_iterations,
        },
        "device": str(device),
        "is_tpu": IS_TPU,
    }

    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print0("\n" + "="*60)
    print0("Training Complete")
    print0("="*60)
    print0(f"Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
    print0(f"Total tokens: {total_tokens:,}")
    print0(f"Average throughput: {total_tokens/total_time:,.0f} tok/s")
    print0(f"Checkpoint saved to: {checkpoint_path}")
    print0("="*60)


if __name__ == "__main__":
    main()
