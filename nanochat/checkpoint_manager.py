"""
Utilities for saving and loading model/optim/state checkpoints.
"""
import os
import re
import glob
import json
import logging
import subprocess
import torch

from nanochat.common import get_base_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.common import setup_default_logging

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)

def _patch_missing_config_keys(model_config_kwargs):
    """Add default values for new config keys missing in old checkpoints."""
    # Old models were trained with full context (no sliding window)
    if "window_pattern" not in model_config_kwargs:
        model_config_kwargs["window_pattern"] = "L"
    # Optional Engram + mHC integration defaults to baseline-safe disabled behavior
    model_config_kwargs.setdefault("engram_enabled", False)
    model_config_kwargs.setdefault("engram_layers", "")
    model_config_kwargs.setdefault("engram_ngram_orders", "2,3,4")
    model_config_kwargs.setdefault("engram_bottleneck_dim", 0)
    model_config_kwargs.setdefault("engram_dropout", 0.0)
    model_config_kwargs.setdefault("mhc_enabled", False)
    model_config_kwargs.setdefault("mhc_num_branches", 0)
    model_config_kwargs.setdefault("mhc_sinkhorn_iters", 5)
    model_config_kwargs.setdefault("mhc_temperature", 1.0)
    model_config_kwargs.setdefault("mhc_epsilon", 1e-6)
    model_config_kwargs.setdefault("mhc_blend_alpha", 1.0)
    model_config_kwargs.setdefault("aux_loss_weight", 0.0)


def _build_gpt_config(model_config_kwargs):
    supported_keys = set(getattr(GPTConfig, "__dataclass_fields__", {}).keys())
    if not supported_keys:
        return GPTConfig(**model_config_kwargs)
    filtered_kwargs = {k: v for k, v in model_config_kwargs.items() if k in supported_keys}
    ignored_keys = sorted(set(model_config_kwargs.keys()) - set(filtered_kwargs.keys()))
    if ignored_keys:
        log0(f"Ignoring unsupported GPTConfig keys in this checkout: {', '.join(ignored_keys)}")
    return GPTConfig(**filtered_kwargs)

def _patch_missing_keys(model_data, model_config):
    """Add default values for new parameters that may be missing in old checkpoints."""
    n_layer = model_config.n_layer
    # resid_lambdas defaults to 1.0 (identity scaling)
    if "resid_lambdas" not in model_data:
        model_data["resid_lambdas"] = torch.ones(n_layer)
    # x0_lambdas defaults to 0.0 (disabled)
    if "x0_lambdas" not in model_data:
        model_data["x0_lambdas"] = torch.zeros(n_layer)

def _upload_to_gcs(local_path, gcs_bucket=None, max_retries=3):
    """Upload a file to GCS if bucket is configured via NANOCHAT_GCS_CHECKPOINT_BUCKET env var."""
    if gcs_bucket is None:
        gcs_bucket = os.environ.get("NANOCHAT_GCS_CHECKPOINT_BUCKET")
    if not gcs_bucket:
        return
    # Build GCS path mirroring local structure
    base_dir = get_base_dir()
    rel_path = os.path.relpath(local_path, base_dir)
    gcs_path = f"{gcs_bucket.rstrip('/')}/{rel_path}"
    file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
    # Use gsutil for large files (better multipart upload support), gcloud storage for small
    if file_size_mb > 100:
        cmd = ["gsutil", "-m", "-o", "GSUtil:parallel_composite_upload_threshold=50M", "cp", local_path, gcs_path]
        timeout = 600  # 10 min for large files
    else:
        cmd = ["gcloud", "storage", "cp", local_path, gcs_path]
        timeout = 300
    for attempt in range(1, max_retries + 1):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode == 0:
                logger.info(f"Uploaded to GCS: {gcs_path} ({file_size_mb:.1f} MB)")
                return
            else:
                logger.warning(f"GCS upload attempt {attempt}/{max_retries} failed for {local_path} "
                             f"(rc={result.returncode}): {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            logger.warning(f"GCS upload attempt {attempt}/{max_retries} timed out for {local_path} "
                         f"({file_size_mb:.1f} MB, timeout={timeout}s)")
        except Exception as e:
            logger.warning(f"GCS upload attempt {attempt}/{max_retries} error for {local_path}: {e}")
    logger.error(f"GCS upload FAILED after {max_retries} attempts: {local_path} -> {gcs_path}")

def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save the model state parameters
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(model_data, model_path)
        logger.info(f"Saved model parameters to: {model_path}")
        _upload_to_gcs(model_path)
        # Save the metadata dict as json
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        logger.info(f"Saved metadata to: {meta_path}")
        _upload_to_gcs(meta_path)
    # Note that optimizer state is sharded across ranks, so each rank must save its own.
    if optimizer_data is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        torch.save(optimizer_data, optimizer_path)
        logger.info(f"Saved optimizer state to: {optimizer_path}")
        _upload_to_gcs(optimizer_path)

def _download_from_gcs(local_path, gcs_bucket=None):
    """Download a file from GCS if not available locally."""
    if os.path.exists(local_path):
        return True
    if gcs_bucket is None:
        gcs_bucket = os.environ.get("NANOCHAT_GCS_CHECKPOINT_BUCKET")
    if not gcs_bucket:
        return False
    base_dir = get_base_dir()
    rel_path = os.path.relpath(local_path, base_dir)
    gcs_path = f"{gcs_bucket.rstrip('/')}/{rel_path}"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        logger.info(f"Downloading from GCS: {gcs_path}")
        result = subprocess.run(["gsutil", "-m", "cp", gcs_path, local_path],
                               capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            logger.info(f"Downloaded from GCS: {local_path}")
            return True
        else:
            logger.warning(f"GCS download failed (rc={result.returncode}): {result.stderr.strip()}")
            return False
    except Exception as e:
        logger.warning(f"GCS download error for {local_path}: {e}")
        return False

def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    # Always load to CPU first (XLA doesn't support direct torch.load).
    # Caller is responsible for moving to device (e.g. via load_state_dict with assign=False).
    load_device = "cpu" if str(device).startswith("xla") else device
    # Load the model state (download from GCS if not available locally)
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    _download_from_gcs(model_path)
    model_data = torch.load(model_path, map_location=load_device)
    # Load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        _download_from_gcs(optimizer_path)
        optimizer_data = torch.load(optimizer_path, map_location=load_device)
    # Load the metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    _download_from_gcs(meta_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data


def build_model(checkpoint_dir, step, device, phase):
    """
    A bunch of repetitive code to build a model from a given checkpoint.
    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    if device.type in {"cpu", "mps"}:
        # Convert bfloat16 tensors to float for CPU inference
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]
    _patch_missing_config_keys(model_config_kwargs)
    log0(f"Building model with config: {model_config_kwargs}")
    model_config = _build_gpt_config(model_config_kwargs)
    _patch_missing_keys(model_data, model_config)
    with torch.device("meta"):
        model = GPT(model_config)
    # Load the model state
    model.to_empty(device=device)
    model.init_weights() # note: this is dumb, but we need to init the rotary embeddings. TODO: fix model re-init
    model.load_state_dict(model_data, strict=True, assign=True)
    # Put the model in the right training phase / mode
    if phase == "eval":
        model.eval()
    else:
        model.train()
    # Load the Tokenizer
    tokenizer = get_tokenizer()
    # Sanity check: compatibility between model and tokenizer
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"]
    return model, tokenizer, meta_data


def find_largest_model(checkpoints_dir):
    # attempt to guess the model tag: take the biggest model available
    model_tags = [f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    # 1) normally all model tags are of the form d<number>, try that first:
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    # 2) if that failed, take the most recently updated model:
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    """Find the highest checkpoint step. Checks local files first, falls back to GCS."""
    # Check local files first
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if checkpoint_files:
        last_step = int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in checkpoint_files))
        return last_step
    # Fall back to GCS if configured
    gcs_bucket = os.environ.get("NANOCHAT_GCS_CHECKPOINT_BUCKET")
    if gcs_bucket:
        base_dir = get_base_dir()
        rel_path = os.path.relpath(checkpoint_dir, base_dir)
        gcs_dir = f"{gcs_bucket.rstrip('/')}/{rel_path}"
        try:
            result = subprocess.run(["gsutil", "ls", f"{gcs_dir}/model_*.pt"],
                                   capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                gcs_files = result.stdout.strip().split("\n")
                steps = []
                for f in gcs_files:
                    fname = f.rstrip("/").split("/")[-1]
                    step_str = fname.split("_")[-1].split(".")[0]
                    steps.append(int(step_str))
                if steps:
                    last_step = max(steps)
                    logger.info(f"Found latest checkpoint in GCS: step {last_step}")
                    return last_step
        except Exception as e:
            logger.warning(f"GCS checkpoint listing failed: {e}")
    raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

# -----------------------------------------------------------------------------
# convenience functions that take into account nanochat's directory structure

def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None):
    if model_tag is None:
        # guess the model tag by defaulting to the largest model
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    # build the model
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
    return model, tokenizer, meta_data

def load_model(source, *args, **kwargs):
    model_dir = {
        "base": "base_checkpoints",
        "mid": "mid_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, *args, **kwargs)
