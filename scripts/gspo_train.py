"""
GSPO (Group Sequence Policy Optimization) training script for C++ code generation.

Uses sequence-level importance sampling instead of token-level (GRPO), providing:
- More stable training with lower variance gradients
- Better alignment with sequence-level rewards
- Natural stability for MoE models without routing replay

Usage:
    .venv/bin/python -m scripts.gspo_train --checkpoint_path /path/to/sft_checkpoint --prompts data/gspo_prompts.jsonl
    .venv/bin/python -m scripts.gspo_train --checkpoint_path /path/to/sft_checkpoint --prompts data/gspo_prompts.jsonl --use-gke-sandbox
"""

import os
import copy
import json
import shutil
import time
import asyncio

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
from nanochat.gspo import GSPOTrainer, GSPOTrainerAsync, GSPOConfig
from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type, get_base_dir
from nanochat import kernels
from nanochat.tokenizer import get_tokenizer
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint, find_last_step

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="GSPO RL training for C++ code generation")
# Data
parser.add_argument("--prompts", type=str, required=True, help="JSONL file with prompt pool")
parser.add_argument("--tokenizer", type=str, default="cpp", choices=["cpp", "default"])
parser.add_argument("--max_seq_len", type=int, default=1024, help="Max sequence length")
# Model
parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to SFT checkpoint directory")
parser.add_argument("--checkpoint_step", type=int, default=-1, help="Checkpoint step (-1 = latest)")
# GSPO hyperparameters (from Qwen3 paper)
parser.add_argument("--group_size", type=int, default=8, help="Completions per prompt")
parser.add_argument("--epsilon", type=float, default=3e-4, help="Left clipping range")
parser.add_argument("--epsilon_high", type=float, default=4e-4, help="Right clipping range")
parser.add_argument("--steps_per_generation", type=int, default=4, help="Minibatches per rollout")
parser.add_argument("--beta", type=float, default=0.0, help="KL penalty (0 = none)")
parser.add_argument("--max_gen_len", type=int, default=512, help="Max generation length")
parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
parser.add_argument("--importance_sampling_level", type=str, default="sequence",
                    choices=["sequence", "token"], help="GSPO (sequence) vs GRPO (token)")
# Training
parser.add_argument("--num_iterations", type=int, default=200, help="Number of GSPO steps")
parser.add_argument("--prompts_per_step", type=int, default=2, help="Prompts per GSPO step")
parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
# GKE Sandbox (optional)
parser.add_argument("--use-gke-sandbox", action="store_true", help="Use GKE sandbox for verification")
parser.add_argument("--gke-endpoint", type=str, default="", help="GKE execution controller endpoint")
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

# Tokenize prompts
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

# GSPO config
gspo_config = GSPOConfig(
    group_size=args.group_size,
    epsilon=args.epsilon,
    epsilon_high=args.epsilon_high,
    steps_per_generation=args.steps_per_generation,
    beta=args.beta,
    max_gen_len=args.max_gen_len,
    temperature=args.temperature,
    importance_sampling_level=args.importance_sampling_level,
)

# Create trainer (sync or async based on GKE usage)
if args.use_gke_sandbox:
    from nanochat.gke_sandbox import GKESandbox, GKESandboxConfig
    sandbox_config = GKESandboxConfig()
    if args.gke_endpoint:
        sandbox_config.endpoint = args.gke_endpoint
    sandbox = GKESandbox(sandbox_config)
    trainer = GSPOTrainerAsync(model, ref_model, tokenizer, sandbox, gspo_config)
    print0(f"Using GKE sandbox at {sandbox_config.endpoint}")
else:
    trainer = GSPOTrainer(model, ref_model, tokenizer, gspo_config)
    print0("Using local g++ verification")

# Output directory
base_dir = get_base_dir()
if args.output_dir:
    output_dir = args.output_dir
else:
    output_dir = os.path.join(base_dir, "gspo_checkpoints", f"d{model_config.n_layer}")
os.makedirs(output_dir, exist_ok=True)

# Training loop
print0(f"Starting GSPO training: {args.num_iterations} steps, {args.prompts_per_step} prompts/step")
print0(f"Group size: {args.group_size}, epsilon: {args.epsilon}, epsilon_high: {args.epsilon_high}")
print0(f"Importance sampling: {args.importance_sampling_level}-level")

# Running statistics
ema_reward = 0.0
ema_compile = 0.0
ema_beta = 0.9

prompt_idx = 0


async def run_training():
    global prompt_idx, ema_reward, ema_compile

    for step in range(1, args.num_iterations + 1):
        t0 = time.time()

        # Sample prompts (round-robin over the pool)
        step_prompts = []
        step_tests = []
        for _ in range(args.prompts_per_step):
            step_prompts.append(tokenized_prompts[prompt_idx % len(tokenized_prompts)])
            step_tests.append(test_harnesses[prompt_idx % len(test_harnesses)])
            prompt_idx += 1

        # GSPO step (multiple minibatches if steps_per_generation > 1)
        for _ in range(args.steps_per_generation):
            optimizer.zero_grad(set_to_none=True)

            if args.use_gke_sandbox:
                metrics = await trainer.step_async(step_prompts, step_tests)
            else:
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
                    "gspo_config": vars(gspo_config),
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
            "gspo_config": vars(gspo_config),
            "ema_reward": debiased_reward,
            "ema_compile_rate": debiased_compile,
        },
        rank=ddp_rank,
    )
    print0(f"GSPO training complete. Checkpoint saved to {output_dir}")
    print0(f"Final reward: {debiased_reward:.3f}, compile rate: {debiased_compile:.1%}")


# Run training
if args.use_gke_sandbox:
    asyncio.run(run_training())
else:
    # Sync version doesn't need async
    for step in range(1, args.num_iterations + 1):
        t0 = time.time()

        step_prompts = []
        step_tests = []
        for _ in range(args.prompts_per_step):
            step_prompts.append(tokenized_prompts[prompt_idx % len(tokenized_prompts)])
            step_tests.append(test_harnesses[prompt_idx % len(test_harnesses)])
            prompt_idx += 1

        for _ in range(args.steps_per_generation):
            optimizer.zero_grad(set_to_none=True)
            metrics = trainer.step(step_prompts, step_tests)

            loss = metrics["loss"]
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        synchronize()
        dt = time.time() - t0

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
                    "gspo_config": vars(gspo_config),
                    "ema_reward": debiased_reward,
                    "ema_compile_rate": debiased_compile,
                },
                rank=ddp_rank,
            )

    save_checkpoint(
        output_dir, step,
        model.state_dict(), [optimizer.state_dict()],
        {
            "step": step,
            "model_config": vars(model_config),
            "gspo_config": vars(gspo_config),
            "ema_reward": debiased_reward,
            "ema_compile_rate": debiased_compile,
        },
        rank=ddp_rank,
    )
    print0(f"GSPO training complete. Checkpoint saved to {output_dir}")
    print0(f"Final reward: {debiased_reward:.3f}, compile rate: {debiased_compile:.1%}")

compute_cleanup()
