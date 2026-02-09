#!/usr/bin/env python3
"""
TPU Training Script for nanochat on Vertex AI

This script adapts the base_train.py for PyTorch/XLA on TPU v5e/v6e.
Key differences from GPU training:
- Uses torch_xla for TPU acceleration
- GCS-based data loading via gcsfs
- XLA-optimized training loop with mark_step

Usage:
    python train_tpu.py --gcs_bucket=gs://nanochat-training-data-2026 --model_depth=20
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime

# TPU/XLA environment setup - must happen before torch import
os.environ.setdefault("PJRT_DEVICE", "TPU")

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from google.cloud import storage
import pyarrow.parquet as pq
import gcsfs

# Import nanochat modules
sys.path.insert(0, "/app")
from nanochat.tokenizer import HuggingFaceTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="TPU Training for nanochat")

    # GCS Configuration
    parser.add_argument("--gcs_bucket", type=str, required=True,
                        help="GCS bucket for data and checkpoints (e.g., gs://nanochat-training-data-2026)")

    # Model Architecture
    parser.add_argument("--model_depth", type=int, default=20,
                        help="Depth of the Transformer model")
    parser.add_argument("--aspect_ratio", type=int, default=64,
                        help="model_dim = depth * aspect_ratio")
    parser.add_argument("--head_dim", type=int, default=128,
                        help="Target head dimension for attention")
    parser.add_argument("--max_seq_len", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--vocab_size", type=int, default=32768,
                        help="Vocabulary size")

    # Training Configuration
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Per-device batch size")
    parser.add_argument("--num_iterations", type=int, default=10000,
                        help="Number of training iterations")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="Weight decay")

    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="GCS path for checkpoints")
    parser.add_argument("--save_every", type=int, default=1000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--eval_every", type=int, default=500,
                        help="Evaluate every N steps")

    # Logging
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--run_name", type=str, default=None,
                        help="W&B run name")

    return parser.parse_args()


def download_from_gcs(gcs_path: str, local_path: str):
    """Download a file from GCS to local filesystem."""
    # Parse gs:// URL
    if gcs_path.startswith("gs://"):
        gcs_path = gcs_path[5:]
    bucket_name, blob_path = gcs_path.split("/", 1)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)
    print(f"Downloaded {gcs_path} to {local_path}")


def upload_to_gcs(local_path: str, gcs_path: str):
    """Upload a file from local filesystem to GCS."""
    if gcs_path.startswith("gs://"):
        gcs_path = gcs_path[5:]
    bucket_name, blob_path = gcs_path.split("/", 1)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} to gs://{gcs_path}")


def load_tokenizer_from_gcs(gcs_bucket: str) -> HuggingFaceTokenizer:
    """Load tokenizer from GCS."""
    local_tokenizer_dir = "/tmp/tokenizer"
    os.makedirs(local_tokenizer_dir, exist_ok=True)

    # Download tokenizer.json
    tokenizer_gcs_path = f"{gcs_bucket}/tokenizer/tokenizer.json"
    local_tokenizer_path = os.path.join(local_tokenizer_dir, "tokenizer.json")
    download_from_gcs(tokenizer_gcs_path, local_tokenizer_path)

    return HuggingFaceTokenizer.from_directory(local_tokenizer_dir)


# =============================================================================
# Simplified GPT Model for TPU (avoids Flash Attention dependency)
# =============================================================================
import torch.nn as nn
import torch.nn.functional as F


def norm(x):
    """RMS norm without learnable params."""
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    """Apply rotary embeddings."""
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    """Self-attention using PyTorch's SDPA (XLA-compatible)."""

    def __init__(self, n_embd, n_head, n_kv_head):
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = n_embd // n_head
        assert n_embd % n_head == 0

        self.c_q = nn.Linear(n_embd, n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x, cos_sin):
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply rotary embeddings
        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)  # QK norm

        # Transpose for SDPA: (B, T, H, D) -> (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Handle GQA (expand k, v heads to match q)
        if self.n_kv_head < self.n_head:
            repeat_factor = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        # PyTorch SDPA with causal mask
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Transpose back and project
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()  # ReGLU squared activation
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head):
        super().__init__()
        self.attn = CausalSelfAttention(n_embd, n_head, n_kv_head)
        self.mlp = MLP(n_embd)

    def forward(self, x, cos_sin):
        x = x + self.attn(norm(x), cos_sin)
        x = x + self.mlp(norm(x))
        return x


class GPTForTPU(nn.Module):
    """Simplified GPT model for TPU training."""

    def __init__(self, vocab_size, n_layer, n_embd, n_head, n_kv_head, sequence_len):
        super().__init__()
        self.sequence_len = sequence_len
        # Pad vocab to multiple of 64 for efficiency
        self.padded_vocab_size = ((vocab_size + 63) // 64) * 64

        self.wte = nn.Embedding(self.padded_vocab_size, n_embd)
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, n_kv_head) for _ in range(n_layer)
        ])
        self.lm_head = nn.Linear(n_embd, self.padded_vocab_size, bias=False)

        # Precompute rotary embeddings
        self.register_buffer("cos", None, persistent=False)
        self.register_buffer("sin", None, persistent=False)
        self._init_rope(sequence_len * 2, n_embd // n_head)

    def _init_rope(self, max_seq_len, head_dim):
        """Initialize rotary position embeddings."""
        theta = 10000.0
        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, freqs)
        self.cos = freqs.cos().unsqueeze(0).unsqueeze(2)  # (1, T, 1, D/2)
        self.sin = freqs.sin().unsqueeze(0).unsqueeze(2)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.sequence_len, f"Sequence length {T} exceeds max {self.sequence_len}"

        # Token embeddings
        x = self.wte(idx)
        x = norm(x)  # Norm after embedding

        # Rotary embeddings for this sequence
        cos = self.cos[:, :T, :, :].to(x.dtype).to(x.device)
        sin = self.sin[:, :T, :, :].to(x.dtype).to(x.device)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, (cos, sin))

        x = norm(x)

        if targets is not None:
            # Training: compute loss
            logits = self.lm_head(x)
            # Slice to actual vocab size for loss computation
            loss = F.cross_entropy(
                logits[:, :, :self.padded_vocab_size].view(-1, self.padded_vocab_size),
                targets.view(-1),
                ignore_index=-1
            )
            return loss
        else:
            # Inference: return logits
            logits = self.lm_head(x[:, -1:, :])
            return logits[:, :, :self.padded_vocab_size]

    def init_weights(self):
        """Initialize weights."""
        n_embd = self.wte.embedding_dim
        # Embedding
        nn.init.normal_(self.wte.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Blocks
        s = 3 ** 0.5 * n_embd ** -0.5
        for block in self.blocks:
            nn.init.uniform_(block.attn.c_q.weight, -s, s)
            nn.init.uniform_(block.attn.c_k.weight, -s, s)
            nn.init.uniform_(block.attn.c_v.weight, -s, s)
            nn.init.zeros_(block.attn.c_proj.weight)
            nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            nn.init.zeros_(block.mlp.c_proj.weight)


def create_model(args, device) -> GPTForTPU:
    """Create and initialize the model."""
    num_layers = args.model_depth
    model_dim = args.model_depth * args.aspect_ratio

    # Find optimal number of heads
    def find_num_heads(model_dim, target_head_dim):
        ideal = max(1, round(model_dim / target_head_dim))
        for offset in range(model_dim):
            for candidate in [ideal + offset, ideal - offset]:
                if candidate > 0 and model_dim % candidate == 0:
                    return candidate
        return 1

    num_heads = find_num_heads(model_dim, args.head_dim)

    model = GPTForTPU(
        vocab_size=args.vocab_size,
        n_layer=num_layers,
        n_embd=model_dim,
        n_head=num_heads,
        n_kv_head=num_heads,
        sequence_len=args.max_seq_len,
    )
    model.to(device)
    model.init_weights()

    return model


# =============================================================================
# Data Loading from GCS
# =============================================================================

class GCSParquetDataset:
    """Load training data from GCS parquet files."""

    def __init__(self, gcs_bucket: str, tokenizer, batch_size: int, seq_len: int):
        self.gcs_bucket = gcs_bucket.rstrip("/")
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.bos_token = tokenizer.encode("<|bos|>", add_bos=False)[0] if hasattr(tokenizer, 'encode') else 0

        # Initialize GCS filesystem
        self.fs = gcsfs.GCSFileSystem()

        # List parquet files
        bucket_path = gcs_bucket.replace("gs://", "")
        parquet_pattern = f"{bucket_path}/parquet/base_data_v3/*.parquet"
        self.parquet_files = sorted(self.fs.glob(parquet_pattern))
        print(f"Found {len(self.parquet_files)} parquet files")

        self.current_file_idx = 0
        self.current_row_idx = 0
        self.current_table = None
        self._load_next_file()

    def _load_next_file(self):
        """Load the next parquet file."""
        if self.current_file_idx >= len(self.parquet_files):
            self.current_file_idx = 0  # Loop back

        path = self.parquet_files[self.current_file_idx]
        print(f"Loading parquet file: {path}")
        with self.fs.open(path, 'rb') as f:
            self.current_table = pq.read_table(f)
        self.current_row_idx = 0
        self.current_file_idx += 1

    def get_batch(self, device):
        """Get a batch of tokenized data."""
        batch_tokens = []

        while len(batch_tokens) < self.batch_size:
            if self.current_row_idx >= len(self.current_table):
                self._load_next_file()

            # Get text from parquet
            row = self.current_table.slice(self.current_row_idx, 1)
            text = row.column('text')[0].as_py()
            self.current_row_idx += 1

            # Tokenize
            if hasattr(self.tokenizer, 'encode'):
                tokens = self.tokenizer.encode(text, add_bos=True)
            else:
                # Fallback for different tokenizer interface
                tokens = self.tokenizer.tokenizer.encode(text).ids
                tokens = [self.bos_token] + tokens

            # Truncate or pad to seq_len + 1 (for x, y shift)
            if len(tokens) > self.seq_len + 1:
                tokens = tokens[:self.seq_len + 1]
            elif len(tokens) < self.seq_len + 1:
                # Pad with zeros
                tokens = tokens + [0] * (self.seq_len + 1 - len(tokens))

            batch_tokens.append(tokens)

        # Convert to tensor
        batch = torch.tensor(batch_tokens, dtype=torch.long, device=device)
        x = batch[:, :-1]  # Input
        y = batch[:, 1:]   # Target
        return x, y


def create_dataloader(args, tokenizer, device):
    """Create dataloader from GCS parquet files."""
    try:
        dataset = GCSParquetDataset(
            gcs_bucket=args.gcs_bucket,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            seq_len=args.max_seq_len,
        )

        def generate_batch():
            while True:
                yield dataset.get_batch(device)

        return generate_batch()
    except Exception as e:
        print(f"Warning: Could not load GCS data: {e}")
        print("Falling back to dummy data...")
        return create_dummy_dataloader(args, device)


def create_dummy_dataloader(args, device):
    """Create a dummy dataloader for testing."""
    B, T = args.batch_size, args.max_seq_len

    def generate_batch():
        while True:
            x = torch.randint(0, args.vocab_size, (B, T), device=device)
            y = torch.randint(0, args.vocab_size, (B, T), device=device)
            yield x, y

    return generate_batch()


# =============================================================================
# Training Loop
# =============================================================================

def get_lr(step: int, warmup_steps: int, max_lr: float, total_steps: int) -> float:
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    elif step < total_steps:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return max_lr * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
    return max_lr * 0.1  # Final LR


def save_checkpoint(model, optimizer, step, args):
    """Save checkpoint to GCS."""
    if args.checkpoint_dir is None:
        return

    local_checkpoint_dir = "/tmp/checkpoint"
    os.makedirs(local_checkpoint_dir, exist_ok=True)

    # Save model state
    model_path = os.path.join(local_checkpoint_dir, f"model_step_{step}.pt")
    xm.save(model.state_dict(), model_path)

    # Save optimizer state
    optimizer_path = os.path.join(local_checkpoint_dir, f"optimizer_step_{step}.pt")
    xm.save(optimizer.state_dict(), optimizer_path)

    # Save metadata
    metadata = {
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "model_depth": args.model_depth,
        "max_seq_len": args.max_seq_len,
    }
    metadata_path = os.path.join(local_checkpoint_dir, f"metadata_step_{step}.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    # Upload to GCS
    gcs_checkpoint_dir = args.checkpoint_dir
    if not gcs_checkpoint_dir.startswith("gs://"):
        gcs_checkpoint_dir = f"gs://{gcs_checkpoint_dir}"

    for filename in [f"model_step_{step}.pt", f"optimizer_step_{step}.pt", f"metadata_step_{step}.json"]:
        local_path = os.path.join(local_checkpoint_dir, filename)
        gcs_path = f"{gcs_checkpoint_dir}/{filename}"
        upload_to_gcs(local_path, gcs_path)

    xm.master_print(f"Checkpoint saved at step {step}")


def train(args):
    """Main training function for a single TPU device."""
    # Get XLA device
    device = xm.xla_device()
    world_size = xm.xrt_world_size()
    rank = xm.get_ordinal()

    xm.master_print(f"Training on {world_size} TPU devices")
    xm.master_print(f"Device: {device}, Rank: {rank}")

    # Load tokenizer
    xm.master_print("Loading tokenizer from GCS...")
    try:
        tokenizer = load_tokenizer_from_gcs(args.gcs_bucket)
        xm.master_print(f"Tokenizer loaded, vocab size: {tokenizer.vocab_size()}")
    except Exception as e:
        xm.master_print(f"Warning: Could not load tokenizer: {e}")
        tokenizer = None

    # Create model
    xm.master_print("Creating model...")
    model = create_model(args, device)

    num_params = sum(p.numel() for p in model.parameters())
    xm.master_print(f"Model parameters: {num_params:,}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # Create dataloader
    xm.master_print("Creating dataloader...")
    if tokenizer is not None:
        train_loader = create_dataloader(args, tokenizer, device)
    else:
        train_loader = create_dummy_dataloader(args, device)

    # Training loop
    xm.master_print("Starting training...")
    model.train()

    total_tokens = 0
    start_time = time.time()
    losses = []

    for step in range(args.num_iterations):
        step_start = time.time()

        # Get batch
        x, y = next(train_loader)

        # Forward pass
        loss = model(x, y)

        # Backward pass
        loss.backward()

        # Update learning rate
        lr = get_lr(step, args.warmup_steps, args.learning_rate, args.num_iterations)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Optimizer step with XLA mark_step
        optimizer.step()
        optimizer.zero_grad()
        xm.mark_step()

        # Track loss (need to fetch from device)
        losses.append(loss.item())

        # Logging
        tokens_per_step = args.batch_size * args.max_seq_len * world_size
        total_tokens += tokens_per_step

        if step % 10 == 0:
            step_time = time.time() - step_start
            tokens_per_sec = tokens_per_step / step_time
            elapsed = time.time() - start_time
            avg_loss = sum(losses[-100:]) / len(losses[-100:])

            xm.master_print(
                f"Step {step:05d}/{args.num_iterations} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {lr:.2e} | "
                f"Tok/s: {tokens_per_sec:,.0f} | "
                f"Elapsed: {elapsed/60:.1f}m"
            )

        # Evaluation
        if args.eval_every > 0 and step > 0 and step % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                eval_x, eval_y = next(train_loader)
                eval_loss = model(eval_x, eval_y)
                xm.mark_step()
                xm.master_print(f"Step {step} | Eval Loss: {eval_loss.item():.4f}")
            model.train()

        # Checkpointing
        if args.save_every > 0 and step > 0 and step % args.save_every == 0:
            save_checkpoint(model, optimizer, step, args)

    # Final checkpoint
    save_checkpoint(model, optimizer, args.num_iterations, args)

    # Final stats
    total_time = time.time() - start_time
    xm.master_print(f"\nTraining complete!")
    xm.master_print(f"Total tokens: {total_tokens:,}")
    xm.master_print(f"Total time: {total_time/60:.1f} minutes")
    xm.master_print(f"Average throughput: {total_tokens/total_time:,.0f} tok/s")
    xm.master_print(f"Final loss: {sum(losses[-100:]) / len(losses[-100:]):.4f}")

    # Save final report to GCS
    if args.checkpoint_dir:
        report = {
            "total_tokens": total_tokens,
            "total_time_minutes": total_time / 60,
            "throughput_tokens_per_sec": total_tokens / total_time,
            "final_loss": sum(losses[-100:]) / len(losses[-100:]),
            "num_params": num_params,
            "model_depth": args.model_depth,
            "batch_size": args.batch_size,
            "max_seq_len": args.max_seq_len,
            "num_iterations": args.num_iterations,
        }
        local_report_path = "/tmp/training_report.json"
        with open(local_report_path, "w") as f:
            json.dump(report, f, indent=2)
        upload_to_gcs(local_report_path, f"{args.checkpoint_dir}/training_report.json")


def _mp_fn(index, args):
    """Multiprocessing entry point for distributed training."""
    train(args)


def main():
    args = parse_args()

    print("=" * 60)
    print("nanochat TPU Training")
    print("=" * 60)
    print(f"GCS Bucket: {args.gcs_bucket}")
    print(f"Model Depth: {args.model_depth}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Max Seq Len: {args.max_seq_len}")
    print(f"Iterations: {args.num_iterations}")
    print(f"Checkpoint Dir: {args.checkpoint_dir}")
    print("=" * 60)

    # Get TPU topology from environment
    tpu_chips = int(os.environ.get("TPU_CHIPS", "4"))
    tpu_topology = os.environ.get("TPU_TOPOLOGY", "2x2")
    print(f"TPU Chips: {tpu_chips}")
    print(f"TPU Topology: {tpu_topology}")

    # Run training
    # For multi-device TPU, use xmp.spawn
    # For single device, run directly
    if tpu_chips > 1:
        xmp.spawn(_mp_fn, args=(args,), nprocs=tpu_chips)
    else:
        train(args)


if __name__ == "__main__":
    main()
