"""
Train model. From root directory of the project, run as:

python -m scripts.base_train.py

or distributed as:

torchrun --nproc_per_node=8 -m scripts.base_train.py

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max_seq_len=512 --device_batch_size=1 --eval_tokens=512 --core_metric_every=-1 --total_batch_size=512 --num_iterations=20
"""

import os
import shutil

# Triton SM121a auto-fix: Triton bundles ptxas 12.8 which doesn't support SM121a (GB10)
# We need system ptxas from CUDA 13.0+ for GB10/DGX Spark
if not os.environ.get("TRITON_PTXAS_PATH"):
    for ptxas in ["/usr/local/cuda/bin/ptxas", shutil.which("ptxas")]:
        if ptxas and os.path.exists(ptxas):
            os.environ["TRITON_PTXAS_PATH"] = ptxas
            break

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# GB10 SM count fix: PyTorch's is_big_gpu() requires 68 SMs but GB10 has 48
# Force max_autotune_gemm to work on GB10 by setting env var before torch import
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "1"

# Persistent cache for autotune results (survives reboot, faster subsequent runs)
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".cache", "torchinductor")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = CACHE_DIR
import gc
import argparse
import time
from contextlib import nullcontext, contextmanager

import wandb
import torch

# Disable torch.compile entirely for XLA/TPU - the inductor backend doesn't support XLA devices
# Must be done before any torch.compile calls or dynamo configurations
if os.environ.get("PJRT_DEVICE") == "TPU":
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.disable = True
    # Note: --xla_tpu_disable_full_embedding_pipelining=true was tested
    # but is NOT supported by libtpu 0.0.21 (v2-alpha-tpuv6e runtime)
else:
    # Fix Liger-Kernel graph breaks: LigerFusedLinearCrossEntropy calls .item() internally
    # which causes torch.compile graph breaks. This config enables capturing scalar outputs.
    torch._dynamo.config.capture_scalar_outputs = True

# Patch is_big_gpu to return True for GB10 (48 SMs < 68 SM threshold)
# This eliminates the "Not enough SMs to use max_autotune_gemm mode" warning
try:
    import torch._inductor.utils as inductor_utils
    inductor_utils.is_big_gpu = lambda index=0: True
except Exception:
    pass

from nanochat.gpt import GPT, GPTConfig, select_precision, make_autocast_ctx
from nanochat.dataloader import tokenizing_distributed_data_loader, tokenizing_distributed_data_loader_with_state
from nanochat.common import (
    compute_init,
    compute_cleanup,
    print0,
    DummyWandb,
    print_banner,
    get_base_dir,
    autodetect_device_type,
    get_tpu_accelerator_type,
    xla_all_reduce_gradients,
    _is_tpu_requested,
)
from nanochat import kernels
from nanochat.tokenizer import get_tokenizer, get_token_bytes, verify_cpp_tokenizer
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from scripts.base_eval import evaluate_model

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device_type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Model architecture
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect_ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head_dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max_seq_len", type=int, default=2048, help="max context length (supports up to 16384 for long context)")
parser.add_argument("--window_pattern", type=str, default="SSSL", help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL')")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num_iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target_flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target_param_data_ratio", type=int, default=8, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# Optimization
parser.add_argument("--device_batch_size", type=int, default=32, help="per-device batch size")
parser.add_argument("--total_batch_size", type=int, default=524288, help="total batch size in tokens")
parser.add_argument("--embedding_lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding_lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight_decay", type=float, default=0.2, help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--matrix_lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--scalar_lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
parser.add_argument("--adam_beta1", type=float, default=0.8, help="Adam beta1 for embedding/unembedding")
parser.add_argument("--adam_beta2", type=float, default=0.95, help="Adam beta2 for embedding/unembedding")
parser.add_argument("--warmup_ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown_ratio", type=float, default=0.4, help="ratio of iterations for LR warmdown")
parser.add_argument("--final_lr_frac", type=float, default=0.0, help="final LR as fraction of initial LR")
parser.add_argument("--resume_from_step", type=int, default=-1, help="resume training from this step (-1 = disable)")
# Evaluation
parser.add_argument("--eval_every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval_tokens", type=int, default=20*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core_metric_every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core_metric_max_per_task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample_every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save_every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
# Dataloader
parser.add_argument("--tokenizer_threads", type=int, default=4, help="number of threads for tokenization")
parser.add_argument("--tokenizer_batch_size", type=int, default=128, help="batch size for tokenization")
parser.add_argument("--fim_rate", type=float, default=0.0, help="Fill-in-the-Middle rate (0.0=disabled, 0.5=50%% of docs get FIM). Requires NANOCHAT_CPP_TOKENIZER=1")
parser.add_argument("--structured_fim_rate", type=float, default=0.0, help="Structured FIM rate for docstring->code completion (0.0=disabled)")
parser.add_argument("--structured_fim_path", type=str, default="data/docstring_pairs_full.jsonl", help="Path to structured FIM pairs dataset")
# Output
parser.add_argument("--model_tag", type=str, default=None, help="override model tag for checkpoint directory name")
# Precision (NVFP4/FP8/BF16)
parser.add_argument("--precision", type=str, default="auto", help="precision: auto|nvfp4|fp8|bf16")
parser.add_argument("--nvfp4_disable_rht", type=bool, default=True, help="disable Random Hadamard Transform (required for SM121/GB10)")
parser.add_argument("--nvfp4_disable_sr", type=bool, default=True, help="disable Stochastic Rounding (required for SM121/GB10)")
# FP8 training with torchao (separate from TE-based --precision fp8)
parser.add_argument("--fp8", action="store_true", help="enable FP8 training with torchao (requires H100+ GPU)")
parser.add_argument("--fp8_recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe: tensorwise (faster) or rowwise (more accurate)")
# Kernel backend
parser.add_argument("--kernel", type=str, default="current", choices=["current", "liger", "cce", "triton"], help="kernel backend: current (PyTorch), liger (Liger-Kernel), cce (Apple Cut Cross Entropy), triton (Unsloth-style)")
# torch.compile
parser.add_argument("--no_compile", action="store_true", help="disable torch.compile (use for NVIDIA containers with triton issues)")
# XLA/TPU optimizations
parser.add_argument("--use_scan", action="store_true", help="use torch_xla scan_layers to reduce XLA compilation time (TPU only)")
args = parser.parse_args()
user_config = vars(args).copy()  # for logging

# If --no_compile is set, also disable compile in Muon optimizer via env var
# This must be done before importing muon.py
if args.no_compile:
    os.environ["NANOCHAT_NO_COMPILE"] = "1"
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.disable = True

# Set kernel backend
kernels.set_kernel_backend(args.kernel)
# -----------------------------------------------------------------------------


def train():
    """Main training function. Called once per process (per chip on TPU, per GPU on DDP)."""
    print_banner()

    # Compute init
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.

    # PyTorch performance optimizations
    if device_type == "cuda":
        torch.backends.cudnn.benchmark = True  # auto-tune cuDNN algorithms
        torch.set_float32_matmul_precision('high')  # TF32 on Ampere+, faster matmuls
        # Note: Don't use legacy allow_tf32 API - conflicts with set_float32_matmul_precision

    # Set up precision plan and autocast context factory
    precision_plan = select_precision(target=args.precision, disable_rht=args.nvfp4_disable_rht, disable_sr=args.nvfp4_disable_sr)
    print0(f"Precision: {precision_plan.name}")
    autocast_ctx = make_autocast_ctx(precision_plan, device_type)

    # NVFP4 on SM121 requires batch_size >= 2
    if "NVFP4" in precision_plan.name and args.device_batch_size < 2:
        print0(f"WARNING: NVFP4 requires device_batch_size >= 2, but got {args.device_batch_size}")
        print0("Automatically increasing device_batch_size to 2")
        args.device_batch_size = 2

    # Set synchronize and memory functions based on device type
    if device_type == "cuda":
        synchronize = torch.cuda.synchronize
        get_max_memory = torch.cuda.max_memory_allocated
    elif device_type == "xla":
        import torch_xla.core.xla_model as xm
        synchronize = xm.mark_step
        get_max_memory = lambda: 0  # XLA doesn't expose memory stats the same way
    else:
        synchronize = lambda: None
        get_max_memory = lambda: 0

    # wandb logging init
    use_dummy_wandb = args.run == "dummy" or not master_process
    wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)

    # Tokenizer will be useful for evaluation, also we need the vocab size
    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=device)
    vocab_size = tokenizer.get_vocab_size()
    print0(f"Vocab size: {vocab_size:,}")

    # Verify C++ tokenizer is active (pre-scan check)
    is_cpp_tokenizer = verify_cpp_tokenizer(tokenizer)
    if is_cpp_tokenizer:
        print0("C++ tokenizer verified: keywords are single tokens")

    # Model kwargs are derived from the desired depth of the model
    num_layers = args.depth
    model_dim = args.depth * args.aspect_ratio
    def find_num_heads(model_dim, target_head_dim):
        # Find num_heads that divides model_dim evenly, with head_dim closest to target.
        ideal = max(1, round(model_dim / target_head_dim))
        for offset in range(model_dim):
            for candidate in [ideal + offset, ideal - offset]:
                if candidate > 0 and model_dim % candidate == 0:
                    return candidate
        return 1
    num_heads = find_num_heads(model_dim, args.head_dim)
    num_kv_heads = num_heads # default is 1:1 GQA (Group Query Attention) ratio (i.e. GQA is disabled)
    print0(f"num_layers: {num_layers}")
    print0(f"model_dim: {model_dim}")
    print0(f"num_heads: {num_heads}")
    print0(f"num_kv_heads: {num_kv_heads}")

    # Optimizer / data / training length related hyperparameters
    # figure out the needed gradient accumulation to reach the desired total batch size
    tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
    world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
    assert args.total_batch_size % world_tokens_per_fwdbwd == 0
    grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
    print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
    print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
    print0(f"Total batch size {args.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

    # Batch size scaling for learning rates (hyperparameters were tuned at reference batch size 2^19)
    batch_lr_scale = 1.0
    reference_batch_size = 2**19
    batch_ratio = args.total_batch_size / reference_batch_size
    if batch_ratio != 1.0:
        # SGD: linear scaling with batch size is standard (not used in nanochat)
        # AdamW: sqrt scaling is standard
        # Muon: sqrt scaling is an assumption - not fully studied, but it's a second-order-ish optimizer
        batch_lr_scale = batch_ratio ** 0.5
        print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {args.total_batch_size:,} (reference: {reference_batch_size:,})")

    # Weight decay is tuned at d12 and its scaling seems to be \propto 1/channels^2 (or equivalently, \propto 1/depth^2 due to constant aspect ratio)
    weight_decay_scaled = args.weight_decay * (12 / args.depth)**2
    if args.depth != 12:
        print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")

    # -----------------------------------------------------------------------------
    # Initialize the Model

    # Create a new model with random weights (directly on CUDA, no meta device)
    model_config_kwargs = dict(sequence_len=args.max_seq_len, vocab_size=vocab_size, n_layer=num_layers, n_head=num_heads, n_kv_head=num_kv_heads, n_embd=model_dim, window_pattern=args.window_pattern)
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)
    model.to(device)  # Move model to GPU before init_weights (rotary embeddings need correct device)
    model.init_weights()

    # When using TE precision (NVFP4/FP8), convert model to bfloat16 for proper mixed precision
    if precision_plan.use_te:
        model.to(dtype=torch.bfloat16)
        print0("Converted model to bfloat16 for TE training")

    # If we are resuming, overwrite the model parameters with those of the checkpoint
    base_dir = get_base_dir()
    output_dirname = args.model_tag if args.model_tag else f"d{args.depth}" # e.g. d12
    checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
    resuming = args.resume_from_step != -1
    if resuming:
        print0(f"Resuming optimization from step {args.resume_from_step}")
        model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
        model.load_state_dict(model_data, strict=True, assign=True)
        del model_data # free up this memory after the copy

    # -----------------------------------------------------------------------------
    # FP8 training initialization with torchao (must be done before torch.compile)

    if args.fp8:
        if device_type != "cuda":
            print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
        else:
            from torchao.float8 import Float8LinearConfig, convert_to_float8_training
            import torch.nn as nn

            # Filter: only convert layers with dimensions divisible by 16 (FP8 hardware requirement)
            def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
                if not isinstance(mod, nn.Linear):
                    return False
                # FP8 requires both in_features and out_features divisible by 16
                if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                    return False
                return True

            fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
            convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
            num_fp8_layers = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
            num_skipped = sum(1 for m in model.modules() if isinstance(m, nn.Linear)) - num_fp8_layers
            print0(f"FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8_layers} layers, skipped {num_skipped} (dims not divisible by 16)")

    # Context manager to temporarily disable FP8 for BF16 evaluation
    @contextmanager
    def disable_fp8(model):
        """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation."""
        import torch.nn as nn

        # Find all Float8Linear modules and their locations
        fp8_locations = []  # list of (parent_module, attr_name, fp8_module)
        for name, module in model.named_modules():
            if 'Float8' in type(module).__name__:
                if '.' in name:
                    parent_name, attr_name = name.rsplit('.', 1)
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                    attr_name = name
                fp8_locations.append((parent, attr_name, module))

        if not fp8_locations:
            yield  # No FP8 modules, nothing to do
            return

        # Swap Float8Linear -> nn.Linear (shares the same weight tensor, no copy)
        for parent, attr_name, fp8_module in fp8_locations:
            linear = nn.Linear(
                fp8_module.in_features,
                fp8_module.out_features,
                bias=fp8_module.bias is not None,
                device=fp8_module.weight.device,
                dtype=fp8_module.weight.dtype,
            )
            linear.weight = fp8_module.weight  # share, don't copy
            if fp8_module.bias is not None:
                linear.bias = fp8_module.bias
            setattr(parent, attr_name, linear)

        try:
            yield
        finally:
            # Restore Float8Linear modules
            for parent, attr_name, fp8_module in fp8_locations:
                setattr(parent, attr_name, fp8_module)

    # -----------------------------------------------------------------------------
    # XLA scan_layers optimization: compile 1 transformer block and reuse for all layers
    # This reduces XLA compilation from ~60min to ~20min for 16-layer models by avoiding
    # a 3.5M instruction HLO graph.
    if args.use_scan and device_type == "xla":
        try:
            # torch_xla >= 2.10: scan_layers wraps nn.ModuleList directly
            from torch_xla.experimental.scan import scan_layers
            model.transformer.h = scan_layers(model.transformer.h)
            print0(f"Enabled XLA scan_layers for faster compilation ({num_layers} layers -> 1 compiled block)")
        except (ImportError, AttributeError):
            # torch_xla 2.9.x: scan_layers not available, lower-level scan() exists
            # but requires manual wrapper. Skip for now.
            print0("Warning: --use_scan requires torch_xla >= 2.10 (scan_layers not found in this version), ignoring")
    elif args.use_scan:
        print0("Warning: --use_scan is only effective on XLA/TPU devices, ignoring")

    # -----------------------------------------------------------------------------
    # Compile the model

    orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
    if device_type == "xla":
        # Skip torch.compile for TPU - XLA JIT compiler handles optimization
        # torch.compile with openxla backend can cause OOM during compilation
        print0("Using eager mode for TPU (XLA JIT handles optimization)")
        # model stays uncompiled, XLA will trace and compile lazily
    elif args.no_compile:
        print0("Using eager mode (--no_compile flag set)")
        # model stays uncompiled
    else:
        model = torch.compile(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe
    num_params = sum(p.numel() for p in model.parameters())
    num_scaling_params = orig_model.num_scaling_params()
    print0(f"Number of parameters: {num_params:,} (scaling: {num_scaling_params:,})")
    num_flops_per_token = model.estimate_flops()
    print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

    # Calculate number of iterations. Either it is given, or from target flops, or from target data:param ratio (in that order)
    assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
    if args.num_iterations > 0:
        num_iterations = args.num_iterations
        print0(f"Using user-provided number of iterations: {num_iterations:,}")
    elif args.target_flops > 0:
        # calculate the number of iterations from the target flops
        num_iterations = round(args.target_flops / (num_flops_per_token * args.total_batch_size))
        print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
    elif args.target_param_data_ratio > 0:
        # calculate the number of iterations from the target param data ratio (use scaling params per Kaplan et al.)
        target_tokens = args.target_param_data_ratio * num_scaling_params
        num_iterations = target_tokens // args.total_batch_size
        print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
    else:
        raise ValueError("No training horizon specified")
    total_tokens = args.total_batch_size * num_iterations
    print0(f"Total number of training tokens: {total_tokens:,}")
    print0(f"Tokens : Params ratio: {args.total_batch_size * num_iterations / num_scaling_params:.2f}") # Chinchilla is ~20
    print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

    # -----------------------------------------------------------------------------
    # Initialize the Optimizer (Muon for Linear layers, AdamW for embedding and lm_head)
    adam_betas = (args.adam_beta1, args.adam_beta2)
    optimizers = model.setup_optimizers(
        unembedding_lr=args.unembedding_lr * batch_lr_scale,
        embedding_lr=args.embedding_lr * batch_lr_scale,
        matrix_lr=args.matrix_lr * batch_lr_scale,
        weight_decay=weight_decay_scaled,
        adam_betas=adam_betas,
        scalar_lr=args.scalar_lr * batch_lr_scale,
    )
    adamw_optimizer, muon_optimizer = optimizers

    if resuming:
        for opt, dat in zip(optimizers, optimizer_data):
            opt.load_state_dict(dat)
        del optimizer_data # free up the memory

    # -----------------------------------------------------------------------------
    # Initialize the DataLoaders for train/val
    tokens_dir = os.path.join(base_dir, "tokenized_data")
    dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
    fim_rate = args.fim_rate
    structured_fim_rate = args.structured_fim_rate
    structured_fim_path = os.path.join(os.path.dirname(__file__), '..', args.structured_fim_path)
    if fim_rate > 0 or structured_fim_rate > 0:
        print0(f"FIM enabled: fim_rate={fim_rate}, structured_fim_rate={structured_fim_rate}")
    train_loader = tokenizing_distributed_data_loader_with_state(
        tokenizer, args.device_batch_size, args.max_seq_len, split="train",
        device=device, resume_state_dict=dataloader_resume_state_dict,
        fim_rate=fim_rate, structured_fim_rate=structured_fim_rate,
        structured_fim_path=structured_fim_path if structured_fim_rate > 0 else None
    )
    build_val_loader = lambda: tokenizing_distributed_data_loader(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)
    x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data

    # Pre-scan check: warn if data appears to be non-C++ (no #include, no semicolons)
    def check_cpp_data(x_batch, tokenizer, num_samples=4):
        """Check if the first batch contains C++ code markers."""
        cpp_markers = 0
        total_checked = min(num_samples, x_batch.size(0))
        for i in range(total_checked):
            # Decode a sample from the batch
            sample_ids = x_batch[i].tolist()
            sample_text = tokenizer.decode(sample_ids[:512])  # Check first 512 tokens
            # Look for C++ markers
            if '#include' in sample_text or '::' in sample_text:
                cpp_markers += 1
            elif sample_text.count(';') >= 3:  # At least 3 semicolons suggests C/C++
                cpp_markers += 1
        return cpp_markers, total_checked

    if is_cpp_tokenizer:
        cpp_markers, total_checked = check_cpp_data(x, tokenizer)
        if cpp_markers == 0:
            print0("=" * 60)
            print0("WARNING: Data does not appear to contain C++ code!")
            print0(f"Checked {total_checked} samples: 0 contained #include, ::, or multiple semicolons")
            print0("If training on non-C++ data, set NANOCHAT_CPP_TOKENIZER=0")
            print0("=" * 60)
        else:
            print0(f"Data check: {cpp_markers}/{total_checked} samples contain C++ markers")

    # -----------------------------------------------------------------------------
    # Set up hyperparameter schedulers

    # Learning rate scheduler
    def get_lr_multiplier(it):
        warmup_iters = round(args.warmup_ratio * num_iterations)
        warmdown_iters = round(args.warmdown_ratio * num_iterations)
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        elif it <= num_iterations - warmdown_iters:
            return 1.0
        else:
            progress = (num_iterations - it) / warmdown_iters
            return progress * 1.0 + (1 - progress) * args.final_lr_frac

    # Momentum scheduler for Muon optimizer
    def get_muon_momentum(it):
        frac = min(it / 300, 1)
        momentum = (1 - frac) * 0.85 + frac * 0.95
        return momentum

    # Weight decay scheduler for Muon optimizer (linear to zero over the course of training)
    def get_weight_decay(it):
        return weight_decay_scaled * (1 - it / num_iterations)

    # -----------------------------------------------------------------------------
    # Loop state (variables updated by the training loop)

    if not resuming:
        step = 0
        val_bpb = None # will be set if eval_every > 0
        min_val_bpb = float("inf")
        smooth_train_loss = 0 # EMA of training loss
        total_training_time = 0 # total wall-clock time of training
    else:
        step = meta_data["step"]
        loop_state = meta_data["loop_state"]
        val_bpb = meta_data["val_bpb"]
        min_val_bpb = loop_state["min_val_bpb"]
        smooth_train_loss = loop_state["smooth_train_loss"]
        total_training_time = loop_state["total_training_time"]

    # -----------------------------------------------------------------------------
    # Training loop
    while True:
        last_step = step == num_iterations # loop runs num_iterations+1 times so that we can eval/save at the end
        flops_so_far = num_flops_per_token * args.total_batch_size * step

        # once in a while: evaluate the val bpb (all ranks participate)
        if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
            model.eval()
            val_loader = build_val_loader()
            eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
            with disable_fp8(model), autocast_ctx():
                val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes, synchronize=synchronize)
            print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
            if val_bpb < min_val_bpb:
                min_val_bpb = val_bpb
            wandb_run.log({
                "step": step,
                "total_training_flops": flops_so_far,
                "total_training_time": total_training_time,
                "val/bpb": val_bpb,
            })
            model.train()

        # once in a while: estimate the CORE metric (all ranks participate)
        # use the original uncompiled model because the inputs keep changing shape
        # Disable FP8 for evaluation to use BF16 for more consistent/accurate results
        results = {}
        if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
            model.eval()
            with disable_fp8(orig_model), autocast_ctx():
                results = evaluate_model(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
            print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
            wandb_run.log({
                "step": step,
                "total_training_flops": flops_so_far,
                "core_metric": results["core_metric"],
                "centered_results": results["centered_results"],
            })
            model.train()

        # once in a while: sample from the model (only on master process)
        # use the original uncompiled model because the inputs keep changing shape
        if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
            model.eval()
            prompts = [
                "The capital of France is",
                "The chemical symbol of gold is",
                "If yesterday was Friday, then tomorrow will be",
                "The opposite of hot is",
                "The planets of the solar system are:",
                "My favorite color is",
                "If 5*x + 3 = 13, then x is",
            ]
            engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
            for prompt in prompts:
                tokens = tokenizer(prompt, prepend="<|bos|>")
                with disable_fp8(orig_model), autocast_ctx():
                    sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
                print0(tokenizer.decode(sample[0]))
            model.train()

        # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
        if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
            save_checkpoint(
                checkpoint_dir,
                step,
                orig_model.state_dict(), # model parameters
                [opt.state_dict() for opt in optimizers], # optimizer states
                { # metadata saved as json
                    "step": step,
                    "val_bpb": val_bpb, # loss at last step
                    "model_config": model_config_kwargs,
                    "user_config": user_config, # inputs to the training script
                    "device_batch_size": args.device_batch_size,
                    "max_seq_len": args.max_seq_len,
                    "dataloader_state_dict": dataloader_state_dict,
                    "loop_state": { # all loop state (other than step) so that we can resume training
                        "min_val_bpb": min_val_bpb,
                        "smooth_train_loss": smooth_train_loss,
                        "total_training_time": total_training_time,
                    },
                },
                rank=ddp_rank,
            )

        # termination conditions (TODO: possibly also add loss explosions etc.)
        if last_step:
            break

        # -------------------------------------------------------------------------
        # single training step
        # evaluate the gradient
        synchronize()
        t0 = time.time()
        for micro_step in range(grad_accum_steps):
            with autocast_ctx():
                loss = model(x, y)
            train_loss = loss.detach() # for logging
            loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
            loss.backward()
            if device_type == "xla":
                synchronize()  # XLA: break graph at each micro-step to avoid huge HLO compilation
            x, y, dataloader_state_dict = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
        # step the optimizers
        lrm = get_lr_multiplier(step)
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * lrm
        if device_type == "xla" and ddp_world_size > 1:
            xla_all_reduce_gradients(model, ddp_world_size)
        muon_momentum = get_muon_momentum(step)
        muon_weight_decay = get_weight_decay(step)
        for group in muon_optimizer.param_groups:
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
        for opt in optimizers:
            opt.step()
        if device_type == "xla":
            synchronize()  # XLA: break graph between optimizer step and zero_grad to limit HLO size
        model.zero_grad(set_to_none=True)
        synchronize()
        t1 = time.time()
        dt = t1 - t0
        # -------------------------------------------------------------------------

        # logging
        ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item() # EMA the training loss
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
        pct_done = 100 * step / num_iterations
        tok_per_sec = int(args.total_batch_size / dt)
        flops_per_sec = num_flops_per_token * args.total_batch_size / dt
        # Theoretical peak FLOPS for MFU calculation (dense, no 2:4 sparsity)
        # H100 SXM BF16: 989 TFLOPS sparse, ~495 TFLOPS dense
        # GB10 NVFP4: 1000 TFLOPS sparse, ~500 TFLOPS dense; BF16: ~62 TFLOPS
        # We use dense numbers since nanochat doesn't use 2:4 structured sparsity
        if device_type == "xla":
            # Detect TPU type from metadata when available.
            tpu_type = get_tpu_accelerator_type().lower()
            if "v5" in tpu_type:
                gpu_name = "TPU v5e"
                promised_flops = 197e12
            elif "v6" in tpu_type:
                gpu_name = "TPU v6e"
                promised_flops = 918e12
            else:
                gpu_name = "TPU"
                promised_flops = 197e12
        elif torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            if "GB10" in gpu_name:
                # GB10: use NVFP4 peak if using TE precision, else BF16
                promised_flops = 500e12 if precision_plan.use_te else 62e12
            else:
                # Default to H100 dense BF16
                promised_flops = 495e12
        else:
            gpu_name = "CPU"
            promised_flops = 1e12  # Placeholder for CPU
        promised_flops_total = promised_flops * max(ddp_world_size, 1)
        mfu = 100 * flops_per_sec / promised_flops_total # in %
        if step > 10:
            total_training_time += dt # only count the time after the first 10 steps
        # Calculate ETA based on average time per step (excluding first 10 steps)
        steps_done = step - 10
        if steps_done > 0:
            avg_time_per_step = total_training_time / steps_done
            remaining_steps = num_iterations - step
            eta_seconds = remaining_steps * avg_time_per_step
            eta_str = f" | eta: {eta_seconds/60:.1f}m"
        else:
            eta_str = ""
        print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | total time: {total_training_time/60:.2f}m{eta_str}")
        if step % 100 == 0:
            log_data = {
                "step": step,
                "total_training_flops": flops_so_far,
                "total_training_time": total_training_time,
                "train/loss": debiased_smooth_loss,
                "train/lrm": lrm,
                "train/dt": dt,
                "train/tok_per_sec": tok_per_sec,
                "train/mfu": mfu,
            }
            wandb_run.log(log_data)

        # state update
        first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
        step += 1

        # The garbage collector is sadly a little bit overactive and for some poorly understood reason,
        # it spends ~500ms scanning for cycles quite frequently, just to end up cleaning up very few tiny objects each time.
        # So we manually manage and help it out here (from upstream karpathy/nanochat)
        if first_step_of_run:
            gc.collect()  # manually collect a lot of garbage from setup
            gc.freeze()   # immediately freeze all currently surviving objects and exclude them from GC
            gc.disable()  # nuclear intervention: disable GC entirely except:
        elif step % 5000 == 0:  # every 5000 steps...
            gc.collect()  # manually collect, just to be safe for very, very long runs

    # print a few more stats
    print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
    print0(f"Total training time: {total_training_time/60:.2f}m")
    if val_bpb is not None:
        print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

    # Log to report
    from nanochat.report import get_report
    get_report().log(section="Base model training", data=[
        user_config, # CLI args
        { # stats about the training setup
            "Number of parameters": num_params,
            "Number of FLOPs per token": f"{num_flops_per_token:e}",
            "Calculated number of iterations": num_iterations,
            "Number of training tokens": total_tokens,
            "Tokens : Params ratio": args.total_batch_size * num_iterations / num_params,
            "DDP world size": ddp_world_size,
            "warmup_ratio": args.warmup_ratio,
            "warmdown_ratio": args.warmdown_ratio,
            "final_lr_frac": args.final_lr_frac,
        },
        { # stats about training outcomes
            "Minimum validation bpb": min_val_bpb if val_bpb is not None else None,
            "Final validation bpb": val_bpb,
            "CORE metric estimate": results.get("core_metric", None),
            "MFU %": f"{mfu:.2f}%",
            "Total training flops": f"{flops_so_far:e}",
            "Total training time": f"{total_training_time/60:.2f}m",
            "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
        }
    ])

    # cleanup
    wandb_run.finish() # wandb run finish
    compute_cleanup()


# =============================================================================
# Entry point: multi-chip TPU launch via xmp.spawn, or direct call for GPU/CPU
# =============================================================================

def _mp_fn(index):
    """Entry point for each TPU chip process spawned by xmp.spawn."""
    train()

def main():
    # PJRT uses a separate process per TPU chip. xmp.spawn auto-detects the
    # correct count (and respects TPU_PROCESS_BOUNDS / TPU_VISIBLE_CHIPS).
    if _is_tpu_requested():
        import torch_xla.distributed.xla_multiprocessing as xmp

        xmp.spawn(_mp_fn, nprocs=None)
    else:
        train()


if __name__ == "__main__":
    main()
