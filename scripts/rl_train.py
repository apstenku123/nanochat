"""
GRPO Reinforcement Learning training script for C++ code generation.

Trains a model using Group Relative Policy Optimization (GRPO) with
g++ compilation and test execution as the reward signal.

Usage:
    .venv/bin/python -m scripts.rl_train --checkpoint_path /path/to/sft_checkpoint --prompts data/rl_prompts.jsonl
    .venv/bin/python -m scripts.rl_train --checkpoint_path /path/to/sft_checkpoint --prompts data/rl_prompts.jsonl --group_size 4 --num_iterations 100
"""

import os
import copy
import json
import shutil
import time

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

import torch

torch._dynamo.config.capture_scalar_outputs = True

try:
    import torch._inductor.utils as inductor_utils
    inductor_utils.is_big_gpu = lambda index=0: True
except Exception:
    pass

from nanochat.gpt import GPT, GPTConfig, select_precision, make_autocast_ctx
from nanochat.grpo import GRPOTrainer, GRPOConfig
from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type, get_base_dir
from nanochat import kernels
from nanochat.tokenizer import get_tokenizer
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint, find_last_step

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="GRPO RL training for C++ code generation")
# Data
parser.add_argument("--prompts", type=str, required=True, help="JSONL file with prompt pool: {instruction, test_code}")
parser.add_argument("--tokenizer", type=str, default="cpp", choices=["cpp", "default"])
parser.add_argument("--max_seq_len", type=int, default=1024, help="Max sequence length")
# Model
parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to SFT checkpoint directory")
parser.add_argument("--checkpoint_step", type=int, default=-1, help="Checkpoint step (-1 = latest)")
# GRPO
parser.add_argument("--group_size", type=int, default=8, help="Completions per prompt")
parser.add_argument("--kl_coeff", type=float, default=0.1, help="KL penalty coefficient")
parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clip epsilon")
parser.add_argument("--max_gen_len", type=int, default=512, help="Max generation length")
parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
# Training
parser.add_argument("--num_iterations", type=int, default=200, help="Number of GRPO steps")
parser.add_argument("--prompts_per_step", type=int, default=2, help="Prompts per GRPO step")
parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
# Output
parser.add_argument("--output_dir", type=str, default="", help="Output directory (empty = auto)")
parser.add_argument("--save_every", type=int, default=50, help="Save every N steps")
parser.add_argument("--log_every", type=int, default=1, help="Log every N steps")
# Runtime
parser.add_argument("--device_type", type=str, default="", help="cuda|cpu|mps")
parser.add_argument("--kernel", type=str, default="current", choices=["current", "liger", "cce", "triton"])
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

precision_plan = select_precision()
autocast_ctx = make_autocast_ctx(precision_plan, device_type)
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None

# Tokenizer
if args.tokenizer == "cpp":
    os.environ["NANOCHAT_CPP_TOKENIZER"] = "1"
tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# Load prompt pool
print0(f"Loading prompts from {args.prompts}")
prompt_pool = []
with open(args.prompts) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        prompt_pool.append(item)
print0(f"Prompt pool size: {len(prompt_pool)}")

# Tokenize prompts (instruction -> token IDs with BOS prefix)
bos_id = tokenizer.get_bos_token_id()
tokenized_prompts = []
test_harnesses = []
for item in prompt_pool:
    ids = [bos_id] + tokenizer.encode(item["instruction"] + "\n")
    tokenized_prompts.append(ids)
    test_harnesses.append(item.get("test_code", ""))

# Load policy model
print0(f"Loading model from {args.checkpoint_path}")
step_to_load = args.checkpoint_step if args.checkpoint_step >= 0 else find_last_step(args.checkpoint_path)
model_state, _, meta = load_checkpoint(args.checkpoint_path, step_to_load, device)
model_config_kwargs = meta["model_config"]
model_config = GPTConfig(**model_config_kwargs)

# Policy model
model = GPT(model_config)
model.to(device)
model.init_weights()
model.load_state_dict(model_state, strict=True, assign=True)
print0(f"Loaded checkpoint from step {meta.get('step', '?')}")

# Reference model (frozen copy)
ref_model = GPT(model_config)
ref_model.to(device)
ref_model.init_weights()
ref_model.load_state_dict(copy.deepcopy(dict(model.state_dict())), strict=True, assign=True)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False
print0("Created frozen reference model")

del model_state

num_params = sum(p.numel() for p in model.parameters())
print0(f"Parameters: {num_params:,}")

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay,
    betas=(0.9, 0.999),
    fused=device_type == "cuda",
)

# GRPO config
grpo_config = GRPOConfig(
    group_size=args.group_size,
    kl_coeff=args.kl_coeff,
    clip_eps=args.clip_eps,
    max_gen_len=args.max_gen_len,
    temperature=args.temperature,
)
trainer = GRPOTrainer(model, ref_model, tokenizer, grpo_config)

# Output directory
base_dir = get_base_dir()
if args.output_dir:
    output_dir = args.output_dir
else:
    output_dir = os.path.join(base_dir, "rl_checkpoints", f"d{model_config.n_layer}")
os.makedirs(output_dir, exist_ok=True)

# Training loop
print0(f"Starting GRPO training: {args.num_iterations} steps, {args.prompts_per_step} prompts/step, group_size={args.group_size}")
print0(f"LR: {args.lr}, KL coeff: {args.kl_coeff}, clip eps: {args.clip_eps}")

# Running statistics
ema_reward = 0.0
ema_compile = 0.0
ema_beta = 0.9

prompt_idx = 0
for step in range(1, args.num_iterations + 1):
    t0 = time.time()

    # Sample prompts (round-robin over the pool)
    step_prompts = []
    step_tests = []
    for _ in range(args.prompts_per_step):
        step_prompts.append(tokenized_prompts[prompt_idx % len(tokenized_prompts)])
        step_tests.append(test_harnesses[prompt_idx % len(test_harnesses)])
        prompt_idx += 1

    # GRPO step
    optimizer.zero_grad(set_to_none=True)
    metrics = trainer.step(step_prompts, step_tests)

    # Backward + optimize
    loss = metrics["loss"]
    if loss.requires_grad:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    synchronize()
    dt = time.time() - t0

    # EMA tracking
    ema_reward = ema_beta * ema_reward + (1 - ema_beta) * metrics["mean_reward"]
    ema_compile = ema_beta * ema_compile + (1 - ema_beta) * metrics["compile_rate"]
    debiased_reward = ema_reward / (1 - ema_beta ** step)
    debiased_compile = ema_compile / (1 - ema_beta ** step)

    if args.log_every > 0 and step % args.log_every == 0:
        print0(
            f"step {step:05d}/{args.num_iterations} | "
            f"loss: {loss.item():.4f} | "
            f"reward: {debiased_reward:.3f} | "
            f"compile: {debiased_compile:.1%} | "
            f"kl: {metrics['kl_div']:.4f} | "
            f"dt: {dt:.1f}s"
        )

    if args.save_every > 0 and step % args.save_every == 0:
        save_checkpoint(
            output_dir, step,
            model.state_dict(), [optimizer.state_dict()],
            {
                "step": step,
                "model_config": vars(model_config),
                "grpo_config": vars(grpo_config),
                "ema_reward": debiased_reward,
                "ema_compile_rate": debiased_compile,
            },
            rank=ddp_rank,
        )

# Final save
save_checkpoint(
    output_dir, step,
    model.state_dict(), [optimizer.state_dict()],
    {
        "step": step,
        "model_config": vars(model_config),
        "grpo_config": vars(grpo_config),
        "ema_reward": debiased_reward,
        "ema_compile_rate": debiased_compile,
    },
    rank=ddp_rank,
)
print0(f"GRPO training complete. Checkpoint saved to {output_dir}")
print0(f"Final reward: {debiased_reward:.3f}, compile rate: {debiased_compile:.1%}")

compute_cleanup()
