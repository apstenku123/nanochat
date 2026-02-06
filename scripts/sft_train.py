"""
Supervised Fine-Tuning (SFT) training script for C++ code generation.

Train a pretrained model on instruction-response pairs.
Based on base_train.py but simplified for SFT:
- Loads SFT dataset (JSONL with instruction/response pairs)
- Applies loss masking (only train on response tokens)
- Uses lower learning rate (2e-4 default)
- Trains for a fixed number of epochs

Usage:
    .venv/bin/python -m scripts.sft_train --data data/sft_sample.jsonl --epochs 3
    .venv/bin/python -m scripts.sft_train --data data/sft_sample.jsonl --checkpoint_path /path/to/checkpoint --epochs 2
"""

import os
import shutil

# Triton SM121a auto-fix
if not os.environ.get("TRITON_PTXAS_PATH"):
    for ptxas in ["/usr/local/cuda/bin/ptxas", shutil.which("ptxas")]:
        if ptxas and os.path.exists(ptxas):
            os.environ["TRITON_PTXAS_PATH"] = ptxas
            break

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "1"

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".cache", "torchinductor")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = CACHE_DIR

import argparse
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

if os.environ.get("PJRT_DEVICE") == "TPU":
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.disable = True
else:
    torch._dynamo.config.capture_scalar_outputs = True

try:
    import torch._inductor.utils as inductor_utils
    inductor_utils.is_big_gpu = lambda index=0: True
except Exception:
    pass

from nanochat.gpt import GPT, GPTConfig, select_precision, make_autocast_ctx
from nanochat.sft_dataset import SFTDataset, sft_collate_fn
from nanochat.tool_sft_dataset import ToolCallSFTDataset, tool_sft_collate_fn
from nanochat.common import (
    compute_init,
    compute_cleanup,
    print0,
    autodetect_device_type,
    get_base_dir,
    xla_all_reduce_gradients,
)
from nanochat import kernels
from nanochat.tokenizer import get_tokenizer
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint, find_last_step

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="SFT training for C++ code generation")
# Data
parser.add_argument("--data", type=str, required=True, help="Path to JSONL SFT dataset")
parser.add_argument("--tokenizer", type=str, default="cpp", choices=["cpp", "default"], help="Tokenizer to use")
parser.add_argument("--max_seq_len", type=int, default=16384, help="Max sequence length")
# Model
parser.add_argument("--checkpoint_path", type=str, default="", help="Path to pretrained checkpoint directory (empty = train from scratch)")
parser.add_argument("--checkpoint_step", type=int, default=-1, help="Checkpoint step to load (-1 = latest)")
parser.add_argument("--depth", type=int, default=12, help="Model depth (used if training from scratch)")
parser.add_argument("--aspect_ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head_dim", type=int, default=128, help="Target head dimension")
# Training
parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
parser.add_argument("--warmup_ratio", type=float, default=0.1, help="LR warmup ratio")
# Output
parser.add_argument("--output_dir", type=str, default="", help="Output checkpoint directory (empty = auto)")
parser.add_argument("--save_every", type=int, default=-1, help="Save every N steps (-1 = only at end)")
parser.add_argument("--log_every", type=int, default=10, help="Log every N steps")
# Runtime
parser.add_argument("--device_type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--kernel", type=str, default="current", choices=["current", "liger", "cce", "triton"])
parser.add_argument("--compile", action="store_true", help="Use torch.compile")
parser.add_argument("--dataset_type", type=str, default="auto", choices=["auto", "instruction", "tool"],
                    help="Dataset type: 'instruction' for {instruction,response}, 'tool' for {text,source}, 'auto' detects from data")
args = parser.parse_args()

# Set kernel backend
kernels.set_kernel_backend(args.kernel)

# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0

if device_type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
xm = None
if device_type == "xla":
    import torch_xla.core.xla_model as xm

precision_plan = select_precision()
autocast_ctx = make_autocast_ctx(precision_plan, device_type)
if device_type == "cuda":
    synchronize = torch.cuda.synchronize
elif device_type == "xla":
    synchronize = xm.mark_step
else:
    synchronize = lambda: None

# Tokenizer
if args.tokenizer == "cpp":
    os.environ["NANOCHAT_CPP_TOKENIZER"] = "1"
tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# Dataset
print0(f"Loading SFT dataset from {args.data}")

# Auto-detect dataset type from first line
dataset_type = args.dataset_type
if dataset_type == "auto":
    import json as _json
    with open(args.data) as _f:
        first_line = None
        for _line in _f:
            _line = _line.strip()
            if _line:
                first_line = _json.loads(_line)
                break
    if first_line is None:
        raise ValueError(f"Dataset file is empty: {args.data}")
    if "text" in first_line and "source" in first_line:
        dataset_type = "tool"
    else:
        dataset_type = "instruction"
    print0(f"Auto-detected dataset type: {dataset_type}")

if dataset_type == "tool":
    dataset = ToolCallSFTDataset(args.data, tokenizer_name=args.tokenizer, max_len=args.max_seq_len)
    collate_fn = tool_sft_collate_fn
else:
    dataset = SFTDataset(args.data, tokenizer_name=args.tokenizer, max_len=args.max_seq_len)
    collate_fn = sft_collate_fn
print0(f"SFT dataset size: {len(dataset)} examples")

sampler = None
use_data_parallel = ddp_world_size > 1
if use_data_parallel:
    sampler = DistributedSampler(
        dataset,
        num_replicas=ddp_world_size,
        rank=ddp_rank,
        shuffle=True,
        drop_last=True,
    )

dataloader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=sampler is None,
    sampler=sampler,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=0,
)

# Model
def find_num_heads(model_dim, target_head_dim):
    ideal = max(1, round(model_dim / target_head_dim))
    for offset in range(model_dim):
        for candidate in [ideal + offset, ideal - offset]:
            if candidate > 0 and model_dim % candidate == 0:
                return candidate
    return 1

if args.checkpoint_path:
    # Load from pretrained checkpoint
    print0(f"Loading pretrained model from {args.checkpoint_path}")
    step_to_load = args.checkpoint_step if args.checkpoint_step >= 0 else find_last_step(args.checkpoint_path)
    model_state, _, meta = load_checkpoint(args.checkpoint_path, step_to_load, device)
    model_config_kwargs = meta["model_config"]
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)
    model.to(device)
    model.init_weights()
    model.load_state_dict(model_state, strict=True, assign=True)
    del model_state
    print0(f"Loaded checkpoint from step {meta.get('step', '?')}")
else:
    # Train from scratch
    num_layers = args.depth
    model_dim = args.depth * args.aspect_ratio
    num_heads = find_num_heads(model_dim, args.head_dim)
    model_config = GPTConfig(
        sequence_len=args.max_seq_len,
        vocab_size=vocab_size,
        n_layer=num_layers,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
    )
    model = GPT(model_config)
    model.to(device)
    model.init_weights()
    print0(f"Created new model: d{args.depth}, dim={model_dim}")

num_params = sum(p.numel() for p in model.parameters())
print0(f"Parameters: {num_params:,}")

orig_model = model
if args.compile and (ddp or device_type == "xla"):
    reason = "DDP" if ddp else "XLA/TPU"
    print0(f"WARNING: --compile is disabled under {reason} in scripts.sft_train")
if args.compile and not ddp and device_type != "xla":
    model = torch.compile(model, dynamic=False)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank] if device_type == "cuda" else None, broadcast_buffers=False)
    orig_model = model.module

# Optimizer (simple AdamW for SFT, no Muon)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay,
    betas=(0.9, 0.999),
    fused=device_type == "cuda",
)

# Output directory
base_dir = get_base_dir()
if args.output_dir:
    output_dir = args.output_dir
else:
    output_dir = os.path.join(base_dir, "sft_checkpoints", f"d{model_config.n_layer}")
os.makedirs(output_dir, exist_ok=True)

# Training loop
num_steps_per_epoch = len(dataloader)
total_steps = args.epochs * num_steps_per_epoch
warmup_steps = int(args.warmup_ratio * total_steps)
print0(f"Training for {args.epochs} epochs, {num_steps_per_epoch} steps/epoch, {total_steps} total steps")
print0(f"Warmup steps: {warmup_steps}")

def all_reduce_sums(loss_sum, token_sum):
    if ddp:
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_sum, op=dist.ReduceOp.SUM)
    elif device_type == "xla" and ddp_world_size > 1:
        xm.all_reduce(xm.REDUCE_SUM, [loss_sum, token_sum])
    return loss_sum, token_sum

global_step = 0
model.train()

for epoch in range(args.epochs):
    if sampler is not None:
        sampler.set_epoch(epoch)

    epoch_loss = 0.0
    epoch_tokens = 0
    t_epoch = time.time()

    for batch_idx, (input_ids, targets) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        train_tokens = (targets != -1).sum().item()
        if train_tokens == 0:
            continue

        # LR schedule: linear warmup then cosine decay
        if global_step < warmup_steps:
            lr = args.lr * (global_step + 1) / warmup_steps
        else:
            progress = (global_step - warmup_steps) / max(total_steps - warmup_steps, 1)
            lr = args.lr * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward
        with autocast_ctx():
            loss_sum = model(input_ids, targets, loss_reduction="sum")
            loss = loss_sum / train_tokens

        # Backward
        loss.backward()
        if device_type == "xla" and ddp_world_size > 1:
            xla_all_reduce_gradients(model, ddp_world_size)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        synchronize()

        # Bookkeeping
        epoch_loss += loss_sum.item()
        epoch_tokens += train_tokens
        global_step += 1

        if args.log_every > 0 and global_step % args.log_every == 0:
            loss_sum = torch.tensor(epoch_loss, device=device)
            token_sum = torch.tensor(float(epoch_tokens), device=device)
            loss_sum, token_sum = all_reduce_sums(loss_sum, token_sum)
            avg_loss = loss_sum.item() / max(token_sum.item(), 1)
            print0(f"  step {global_step:05d}/{total_steps} | epoch {epoch+1}/{args.epochs} | loss: {loss.item():.4f} | avg: {avg_loss:.4f} | lr: {lr:.2e}")

        if args.save_every > 0 and global_step % args.save_every == 0:
            save_checkpoint(
                output_dir, global_step,
                orig_model.state_dict(), [optimizer.state_dict()],
                {"step": global_step, "epoch": epoch, "model_config": vars(model_config)},
                rank=ddp_rank,
            )

    dt = time.time() - t_epoch
    loss_sum = torch.tensor(epoch_loss, device=device)
    token_sum = torch.tensor(float(epoch_tokens), device=device)
    loss_sum, token_sum = all_reduce_sums(loss_sum, token_sum)
    avg_loss = loss_sum.item() / max(token_sum.item(), 1)
    total_tokens = int(token_sum.item())
    print0(f"Epoch {epoch+1}/{args.epochs} done in {dt:.1f}s | avg loss: {avg_loss:.4f} | tokens: {total_tokens:,}")

# Save final checkpoint
save_checkpoint(
    output_dir, global_step,
    orig_model.state_dict(), [optimizer.state_dict()],
    {"step": global_step, "epoch": args.epochs, "model_config": vars(model_config)},
    rank=ddp_rank,
)
print0(f"Training complete. Checkpoint saved to {output_dir}")

compute_cleanup()
