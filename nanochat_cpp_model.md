Project Path: nanochat

Source Tree:

```txt
nanochat
├── nanochat
│   ├── __init__.py
│   ├── checkpoint_manager.py
│   ├── common.py
│   ├── cpp_tokenizer.py
│   ├── engine.py
│   ├── fim.py
│   ├── flash_attention.py
│   ├── gpt.py
│   ├── kernels.py
│   ├── meta_init.py
│   ├── mhc.py
│   ├── mtp.py
│   ├── sparse_attention.py
│   └── tokenizer.py
├── scripts
│   ├── base_train.py
│   └── chat_cli.py
└── tools
    └── clang_indexer
        └── index_project.py

```

`nanochat/nanochat/checkpoint_manager.py`:

```py
   1 | """
   2 | Utilities for saving and loading model/optim/state checkpoints.
   3 | """
   4 | import os
   5 | import re
   6 | import glob
   7 | import json
   8 | import logging
   9 | import subprocess
  10 | import torch
  11 | 
  12 | from nanochat.common import get_base_dir
  13 | from nanochat.gpt import GPT, GPTConfig
  14 | from nanochat.tokenizer import get_tokenizer
  15 | from nanochat.common import setup_default_logging
  16 | 
  17 | # Set up logging
  18 | setup_default_logging()
  19 | logger = logging.getLogger(__name__)
  20 | def log0(message):
  21 |     if int(os.environ.get('RANK', 0)) == 0:
  22 |         logger.info(message)
  23 | 
  24 | def _patch_missing_config_keys(model_config_kwargs):
  25 |     """Add default values for new config keys missing in old checkpoints."""
  26 |     # Old models were trained with full context (no sliding window)
  27 |     if "window_pattern" not in model_config_kwargs:
  28 |         model_config_kwargs["window_pattern"] = "L"
  29 |     # Optional Engram + mHC integration defaults to baseline-safe disabled behavior
  30 |     model_config_kwargs.setdefault("engram_enabled", False)
  31 |     model_config_kwargs.setdefault("engram_layers", "")
  32 |     model_config_kwargs.setdefault("engram_ngram_orders", "2,3,4")
  33 |     model_config_kwargs.setdefault("engram_bottleneck_dim", 0)
  34 |     model_config_kwargs.setdefault("engram_dropout", 0.0)
  35 |     model_config_kwargs.setdefault("mhc_enabled", False)
  36 |     model_config_kwargs.setdefault("mhc_num_branches", 0)
  37 |     model_config_kwargs.setdefault("mhc_sinkhorn_iters", 5)
  38 |     model_config_kwargs.setdefault("mhc_temperature", 1.0)
  39 |     model_config_kwargs.setdefault("mhc_epsilon", 1e-6)
  40 |     model_config_kwargs.setdefault("mhc_blend_alpha", 1.0)
  41 |     model_config_kwargs.setdefault("aux_loss_weight", 0.0)
  42 | 
  43 | 
  44 | def _build_gpt_config(model_config_kwargs):
  45 |     supported_keys = set(getattr(GPTConfig, "__dataclass_fields__", {}).keys())
  46 |     if not supported_keys:
  47 |         return GPTConfig(**model_config_kwargs)
  48 |     filtered_kwargs = {k: v for k, v in model_config_kwargs.items() if k in supported_keys}
  49 |     ignored_keys = sorted(set(model_config_kwargs.keys()) - set(filtered_kwargs.keys()))
  50 |     if ignored_keys:
  51 |         log0(f"Ignoring unsupported GPTConfig keys in this checkout: {', '.join(ignored_keys)}")
  52 |     return GPTConfig(**filtered_kwargs)
  53 | 
  54 | def _patch_missing_keys(model_data, model_config):
  55 |     """Add default values for new parameters that may be missing in old checkpoints."""
  56 |     n_layer = model_config.n_layer
  57 |     # resid_lambdas defaults to 1.0 (identity scaling)
  58 |     if "resid_lambdas" not in model_data:
  59 |         model_data["resid_lambdas"] = torch.ones(n_layer)
  60 |     # x0_lambdas defaults to 0.0 (disabled)
  61 |     if "x0_lambdas" not in model_data:
  62 |         model_data["x0_lambdas"] = torch.zeros(n_layer)
  63 | 
  64 | def _upload_to_gcs(local_path, gcs_bucket=None, max_retries=3):
  65 |     """Upload a file to GCS if bucket is configured via NANOCHAT_GCS_CHECKPOINT_BUCKET env var."""
  66 |     if gcs_bucket is None:
  67 |         gcs_bucket = os.environ.get("NANOCHAT_GCS_CHECKPOINT_BUCKET")
  68 |     if not gcs_bucket:
  69 |         return
  70 |     # Build GCS path mirroring local structure
  71 |     base_dir = get_base_dir()
  72 |     rel_path = os.path.relpath(local_path, base_dir)
  73 |     gcs_path = f"{gcs_bucket.rstrip('/')}/{rel_path}"
  74 |     file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
  75 |     # Use gsutil for large files (better multipart upload support), gcloud storage for small
  76 |     if file_size_mb > 100:
  77 |         cmd = ["gsutil", "-m", "-o", "GSUtil:parallel_composite_upload_threshold=50M", "cp", local_path, gcs_path]
  78 |         timeout = 600  # 10 min for large files
  79 |     else:
  80 |         cmd = ["gcloud", "storage", "cp", local_path, gcs_path]
  81 |         timeout = 300
  82 |     for attempt in range(1, max_retries + 1):
  83 |         try:
  84 |             result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
  85 |             if result.returncode == 0:
  86 |                 logger.info(f"Uploaded to GCS: {gcs_path} ({file_size_mb:.1f} MB)")
  87 |                 return
  88 |             else:
  89 |                 logger.warning(f"GCS upload attempt {attempt}/{max_retries} failed for {local_path} "
  90 |                              f"(rc={result.returncode}): {result.stderr.strip()}")
  91 |         except subprocess.TimeoutExpired:
  92 |             logger.warning(f"GCS upload attempt {attempt}/{max_retries} timed out for {local_path} "
  93 |                          f"({file_size_mb:.1f} MB, timeout={timeout}s)")
  94 |         except Exception as e:
  95 |             logger.warning(f"GCS upload attempt {attempt}/{max_retries} error for {local_path}: {e}")
  96 |     logger.error(f"GCS upload FAILED after {max_retries} attempts: {local_path} -> {gcs_path}")
  97 | 
  98 | def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
  99 |     if rank == 0:
 100 |         os.makedirs(checkpoint_dir, exist_ok=True)
 101 |         # Save the model state parameters
 102 |         model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
 103 |         torch.save(model_data, model_path)
 104 |         logger.info(f"Saved model parameters to: {model_path}")
 105 |         _upload_to_gcs(model_path)
 106 |         # Save the metadata dict as json
 107 |         meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
 108 |         with open(meta_path, "w", encoding="utf-8") as f:
 109 |             json.dump(meta_data, f, indent=2)
 110 |         logger.info(f"Saved metadata to: {meta_path}")
 111 |         _upload_to_gcs(meta_path)
 112 |     # Note that optimizer state is sharded across ranks, so each rank must save its own.
 113 |     if optimizer_data is not None:
 114 |         os.makedirs(checkpoint_dir, exist_ok=True)
 115 |         optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
 116 |         torch.save(optimizer_data, optimizer_path)
 117 |         logger.info(f"Saved optimizer state to: {optimizer_path}")
 118 |         _upload_to_gcs(optimizer_path)
 119 | 
 120 | def _download_from_gcs(local_path, gcs_bucket=None):
 121 |     """Download a file from GCS if not available locally."""
 122 |     if os.path.exists(local_path):
 123 |         return True
 124 |     if gcs_bucket is None:
 125 |         gcs_bucket = os.environ.get("NANOCHAT_GCS_CHECKPOINT_BUCKET")
 126 |     if not gcs_bucket:
 127 |         return False
 128 |     base_dir = get_base_dir()
 129 |     rel_path = os.path.relpath(local_path, base_dir)
 130 |     gcs_path = f"{gcs_bucket.rstrip('/')}/{rel_path}"
 131 |     os.makedirs(os.path.dirname(local_path), exist_ok=True)
 132 |     try:
 133 |         logger.info(f"Downloading from GCS: {gcs_path}")
 134 |         result = subprocess.run(["gsutil", "-m", "cp", gcs_path, local_path],
 135 |                                capture_output=True, text=True, timeout=600)
 136 |         if result.returncode == 0:
 137 |             logger.info(f"Downloaded from GCS: {local_path}")
 138 |             return True
 139 |         else:
 140 |             logger.warning(f"GCS download failed (rc={result.returncode}): {result.stderr.strip()}")
 141 |             return False
 142 |     except Exception as e:
 143 |         logger.warning(f"GCS download error for {local_path}: {e}")
 144 |         return False
 145 | 
 146 | def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
 147 |     # Always load to CPU first (XLA doesn't support direct torch.load).
 148 |     # Caller is responsible for moving to device (e.g. via load_state_dict with assign=False).
 149 |     load_device = "cpu" if str(device).startswith("xla") else device
 150 |     # Load the model state (download from GCS if not available locally)
 151 |     model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
 152 |     _download_from_gcs(model_path)
 153 |     model_data = torch.load(model_path, map_location=load_device)
 154 |     # Load the optimizer state if requested
 155 |     optimizer_data = None
 156 |     if load_optimizer:
 157 |         optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
 158 |         _download_from_gcs(optimizer_path)
 159 |         optimizer_data = torch.load(optimizer_path, map_location=load_device)
 160 |     # Load the metadata
 161 |     meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
 162 |     _download_from_gcs(meta_path)
 163 |     with open(meta_path, "r", encoding="utf-8") as f:
 164 |         meta_data = json.load(f)
 165 |     return model_data, optimizer_data, meta_data
 166 | 
 167 | 
 168 | def build_model(checkpoint_dir, step, device, phase):
 169 |     """
 170 |     A bunch of repetitive code to build a model from a given checkpoint.
 171 |     Returns:
 172 |     - base model - uncompiled, not wrapped in DDP
 173 |     - tokenizer
 174 |     - meta data saved during base model training
 175 |     """
 176 |     assert phase in ["train", "eval"], f"Invalid phase: {phase}"
 177 |     model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
 178 |     if device.type in {"cpu", "mps"}:
 179 |         # Convert bfloat16 tensors to float for CPU inference
 180 |         model_data = {
 181 |             k: v.float() if v.dtype == torch.bfloat16 else v
 182 |             for k, v in model_data.items()
 183 |         }
 184 |     # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
 185 |     model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
 186 |     model_config_kwargs = meta_data["model_config"]
 187 |     _patch_missing_config_keys(model_config_kwargs)
 188 |     log0(f"Building model with config: {model_config_kwargs}")
 189 |     model_config = _build_gpt_config(model_config_kwargs)
 190 |     _patch_missing_keys(model_data, model_config)
 191 |     with torch.device("meta"):
 192 |         model = GPT(model_config)
 193 |     # Load the model state
 194 |     model.to_empty(device=device)
 195 |     model.init_weights() # note: this is dumb, but we need to init the rotary embeddings. TODO: fix model re-init
 196 |     model.load_state_dict(model_data, strict=True, assign=True)
 197 |     # Put the model in the right training phase / mode
 198 |     if phase == "eval":
 199 |         model.eval()
 200 |     else:
 201 |         model.train()
 202 |     # Load the Tokenizer
 203 |     tokenizer = get_tokenizer()
 204 |     # Sanity check: compatibility between model and tokenizer
 205 |     assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"]
 206 |     return model, tokenizer, meta_data
 207 | 
 208 | 
 209 | def find_largest_model(checkpoints_dir):
 210 |     # attempt to guess the model tag: take the biggest model available
 211 |     model_tags = [f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))]
 212 |     if not model_tags:
 213 |         raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
 214 |     # 1) normally all model tags are of the form d<number>, try that first:
 215 |     candidates = []
 216 |     for model_tag in model_tags:
 217 |         match = re.match(r"d(\d+)", model_tag)
 218 |         if match:
 219 |             model_depth = int(match.group(1))
 220 |             candidates.append((model_depth, model_tag))
 221 |     if candidates:
 222 |         candidates.sort(key=lambda x: x[0], reverse=True)
 223 |         return candidates[0][1]
 224 |     # 2) if that failed, take the most recently updated model:
 225 |     model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
 226 |     return model_tags[0]
 227 | 
 228 | 
 229 | def find_last_step(checkpoint_dir):
 230 |     """Find the highest checkpoint step. Checks local files first, falls back to GCS."""
 231 |     # Check local files first
 232 |     checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
 233 |     if checkpoint_files:
 234 |         last_step = int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in checkpoint_files))
 235 |         return last_step
 236 |     # Fall back to GCS if configured
 237 |     gcs_bucket = os.environ.get("NANOCHAT_GCS_CHECKPOINT_BUCKET")
 238 |     if gcs_bucket:
 239 |         base_dir = get_base_dir()
 240 |         rel_path = os.path.relpath(checkpoint_dir, base_dir)
 241 |         gcs_dir = f"{gcs_bucket.rstrip('/')}/{rel_path}"
 242 |         try:
 243 |             result = subprocess.run(["gsutil", "ls", f"{gcs_dir}/model_*.pt"],
 244 |                                    capture_output=True, text=True, timeout=30)
 245 |             if result.returncode == 0 and result.stdout.strip():
 246 |                 gcs_files = result.stdout.strip().split("\n")
 247 |                 steps = []
 248 |                 for f in gcs_files:
 249 |                     fname = f.rstrip("/").split("/")[-1]
 250 |                     step_str = fname.split("_")[-1].split(".")[0]
 251 |                     steps.append(int(step_str))
 252 |                 if steps:
 253 |                     last_step = max(steps)
 254 |                     logger.info(f"Found latest checkpoint in GCS: step {last_step}")
 255 |                     return last_step
 256 |         except Exception as e:
 257 |             logger.warning(f"GCS checkpoint listing failed: {e}")
 258 |     raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
 259 | 
 260 | # -----------------------------------------------------------------------------
 261 | # convenience functions that take into account nanochat's directory structure
 262 | 
 263 | def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None):
 264 |     if model_tag is None:
 265 |         # guess the model tag by defaulting to the largest model
 266 |         model_tag = find_largest_model(checkpoints_dir)
 267 |         log0(f"No model tag provided, guessing model tag: {model_tag}")
 268 |     checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
 269 |     if step is None:
 270 |         # guess the step by defaulting to the last step
 271 |         step = find_last_step(checkpoint_dir)
 272 |     assert step is not None, f"No checkpoints found in {checkpoint_dir}"
 273 |     # build the model
 274 |     log0(f"Loading model from {checkpoint_dir} with step {step}")
 275 |     model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
 276 |     return model, tokenizer, meta_data
 277 | 
 278 | def load_model(source, *args, **kwargs):
 279 |     model_dir = {
 280 |         "base": "base_checkpoints",
 281 |         "mid": "mid_checkpoints",
 282 |         "sft": "chatsft_checkpoints",
 283 |         "rl": "chatrl_checkpoints",
 284 |     }[source]
 285 |     base_dir = get_base_dir()
 286 |     checkpoints_dir = os.path.join(base_dir, model_dir)
 287 |     return load_model_from_dir(checkpoints_dir, *args, **kwargs)

```

`nanochat/nanochat/common.py`:

```py
   1 | """
   2 | Common utilities for nanochat.
   3 | """
   4 | 
   5 | import os
   6 | import re
   7 | import logging
   8 | import urllib.request
   9 | import torch
  10 | import torch.distributed as dist
  11 | from filelock import FileLock
  12 | 
  13 | class ColoredFormatter(logging.Formatter):
  14 |     """Custom formatter that adds colors to log messages."""
  15 |     # ANSI color codes
  16 |     COLORS = {
  17 |         'DEBUG': '\033[36m',    # Cyan
  18 |         'INFO': '\033[32m',     # Green
  19 |         'WARNING': '\033[33m',  # Yellow
  20 |         'ERROR': '\033[31m',    # Red
  21 |         'CRITICAL': '\033[35m', # Magenta
  22 |     }
  23 |     RESET = '\033[0m'
  24 |     BOLD = '\033[1m'
  25 |     def format(self, record):
  26 |         # Add color to the level name
  27 |         levelname = record.levelname
  28 |         if levelname in self.COLORS:
  29 |             record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
  30 |         # Format the message
  31 |         message = super().format(record)
  32 |         # Add color to specific parts of the message
  33 |         if levelname == 'INFO':
  34 |             # Highlight numbers and percentages
  35 |             message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
  36 |             message = re.sub(r'(Shard \d+)', rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}', message)
  37 |         return message
  38 | 
  39 | def setup_default_logging():
  40 |     handler = logging.StreamHandler()
  41 |     handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
  42 |     logging.basicConfig(
  43 |         level=logging.INFO,
  44 |         handlers=[handler]
  45 |     )
  46 | 
  47 | setup_default_logging()
  48 | logger = logging.getLogger(__name__)
  49 | 
  50 | def get_base_dir():
  51 |     # co-locate nanochat intermediates with other cached data in ~/.cache (by default)
  52 |     if os.environ.get("NANOCHAT_BASE_DIR"):
  53 |         nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
  54 |     else:
  55 |         home_dir = os.path.expanduser("~")
  56 |         cache_dir = os.path.join(home_dir, ".cache")
  57 |         nanochat_dir = os.path.join(cache_dir, "nanochat")
  58 |     os.makedirs(nanochat_dir, exist_ok=True)
  59 |     return nanochat_dir
  60 | 
  61 | def download_file_with_lock(url, filename, postprocess_fn=None):
  62 |     """
  63 |     Downloads a file from a URL to a local path in the base directory.
  64 |     Uses a lock file to prevent concurrent downloads among multiple ranks.
  65 |     """
  66 |     base_dir = get_base_dir()
  67 |     file_path = os.path.join(base_dir, filename)
  68 |     lock_path = file_path + ".lock"
  69 | 
  70 |     if os.path.exists(file_path):
  71 |         return file_path
  72 | 
  73 |     with FileLock(lock_path):
  74 |         # Only a single rank can acquire this lock
  75 |         # All other ranks block until it is released
  76 | 
  77 |         # Recheck after acquiring lock
  78 |         if os.path.exists(file_path):
  79 |             return file_path
  80 | 
  81 |         # Download the content as bytes
  82 |         print(f"Downloading {url}...")
  83 |         with urllib.request.urlopen(url) as response:
  84 |             content = response.read() # bytes
  85 | 
  86 |         # Write to local file
  87 |         with open(file_path, 'wb') as f:
  88 |             f.write(content)
  89 |         print(f"Downloaded to {file_path}")
  90 | 
  91 |         # Run the postprocess function if provided
  92 |         if postprocess_fn is not None:
  93 |             postprocess_fn(file_path)
  94 | 
  95 |     return file_path
  96 | 
  97 | def _is_tpu_requested() -> bool:
  98 |     return os.environ.get("PJRT_DEVICE", "").upper() == "TPU"
  99 | 
 100 | 
 101 | def no_grad_or_inference_mode():
 102 |     """Return the appropriate autograd-disabling context for the current device.
 103 | 
 104 |     torch.inference_mode() is faster than torch.no_grad() on CUDA/CPU because it
 105 |     disables view tracking and version counter bumps in addition to gradient
 106 |     computation.  However, it is not reliably supported on non-CUDA backends:
 107 |     PyTorch Lightning hit the same issue on HPU and fell back to no_grad (see
 108 |     lightning changelog for 1.8.x), and XLA/TPU backends can raise errors or
 109 |     produce incorrect results under inference_mode because the XLA tracing
 110 |     layer doesn't fully implement the InferenceMode dispatch key.
 111 | 
 112 |     Use this helper wherever you would write @torch.inference_mode() but need
 113 |     the code to also run on TPU/XLA devices.
 114 |     """
 115 |     if _is_tpu_requested():
 116 |         return torch.no_grad()
 117 |     return torch.inference_mode()
 118 | 
 119 | 
 120 | def _get_xla_dist_info():
 121 |     """Best-effort rank/world-size info for TPU PJRT runtimes."""
 122 |     try:
 123 |         import torch_xla.runtime as xr
 124 |         world_size = int(xr.world_size())
 125 |         rank = int(xr.global_ordinal())
 126 |         local_rank = int(xr.local_ordinal()) if hasattr(xr, "local_ordinal") else rank
 127 |         return rank, local_rank, max(world_size, 1)
 128 |     except Exception:
 129 |         pass
 130 | 
 131 |     # Backward compatibility with older torch_xla APIs.
 132 |     try:
 133 |         import torch_xla.core.xla_model as xm
 134 |         world_size = int(xm.xrt_world_size())
 135 |         rank = int(xm.get_ordinal())
 136 |         local_rank = int(xm.get_local_ordinal()) if hasattr(xm, "get_local_ordinal") else rank
 137 |         return rank, local_rank, max(world_size, 1)
 138 |     except Exception:
 139 |         return 0, 0, 1
 140 | 
 141 | 
 142 | def print0(s="",**kwargs):
 143 |     _, rank, _, _ = get_dist_info()
 144 |     if rank == 0:
 145 |         print(s, **kwargs)
 146 | 
 147 | def print_banner():
 148 |     # Cool DOS Rebel font ASCII banner made with https://manytools.org/hacker-tools/ascii-banner/
 149 |     banner = """
 150 |                                                        █████                █████
 151 |                                                       ░░███                ░░███
 152 |      ████████    ██████   ████████    ██████   ██████  ░███████    ██████  ███████
 153 |     ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███░░░███░
 154 |      ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████  ░███
 155 |      ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███  ░███ ███
 156 |      ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░███████  ░░█████
 157 |     ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░   ░░░░░
 158 |     """
 159 |     print0(banner)
 160 | 
 161 | def is_ddp_requested() -> bool:
 162 |     """
 163 |     True if launched by torchrun (env present), even before init.
 164 |     Used to decide whether we *should* initialize a PG.
 165 |     """
 166 |     return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
 167 | 
 168 | def is_ddp_initialized() -> bool:
 169 |     """
 170 |     True if torch.distributed is available and the process group is initialized.
 171 |     Used at cleanup to avoid destroying a non-existent PG.
 172 |     """
 173 |     return dist.is_available() and dist.is_initialized()
 174 | 
 175 | def get_dist_info():
 176 |     if is_ddp_initialized():
 177 |         ddp_rank = dist.get_rank()
 178 |         ddp_local_rank = int(os.environ.get("LOCAL_RANK", ddp_rank))
 179 |         ddp_world_size = dist.get_world_size()
 180 |         return True, ddp_rank, ddp_local_rank, ddp_world_size
 181 | 
 182 |     if _is_tpu_requested():
 183 |         # TPU world size/rank is managed by PJRT runtime, not torch.distributed.
 184 |         ddp_rank, ddp_local_rank, ddp_world_size = _get_xla_dist_info()
 185 |         return False, ddp_rank, ddp_local_rank, ddp_world_size
 186 | 
 187 |     if is_ddp_requested():
 188 |         # We rely on torchrun's env to decide if we SHOULD init.
 189 |         # (Initialization itself happens in compute init.)
 190 |         assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
 191 |         ddp_rank = int(os.environ['RANK'])
 192 |         ddp_local_rank = int(os.environ['LOCAL_RANK'])
 193 |         ddp_world_size = int(os.environ['WORLD_SIZE'])
 194 |         return True, ddp_rank, ddp_local_rank, ddp_world_size
 195 |     else:
 196 |         return False, 0, 0, 1
 197 | 
 198 | def autodetect_device_type():
 199 |     # Check for TPU first (via PJRT_DEVICE env var or torch_xla availability)
 200 |     if os.environ.get("PJRT_DEVICE") == "TPU":
 201 |         try:
 202 |             import torch_xla.core.xla_model as xm
 203 |             device_type = "xla"
 204 |             print0(f"Autodetected device type: {device_type} (TPU via PJRT_DEVICE)")
 205 |             return device_type
 206 |         except ImportError:
 207 |             print0("Warning: PJRT_DEVICE=TPU but torch_xla not available, falling back...")
 208 | 
 209 |     # prefer to use CUDA if available, otherwise use MPS, otherwise fallback on CPU
 210 |     if torch.cuda.is_available():
 211 |         device_type = "cuda"
 212 |     elif torch.backends.mps.is_available():
 213 |         device_type = "mps"
 214 |     else:
 215 |         device_type = "cpu"
 216 |     print0(f"Autodetected device type: {device_type}")
 217 |     return device_type
 218 | 
 219 | 
 220 | def get_tpu_accelerator_type() -> str:
 221 |     """Return TPU accelerator type (e.g. v5litepod-8, v6e-4) when available."""
 222 |     for key in ("TPU_ACCELERATOR_TYPE", "ACCELERATOR_TYPE"):
 223 |         value = os.environ.get(key, "").strip()
 224 |         if value:
 225 |             return value
 226 | 
 227 |     metadata_url = "http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type"
 228 |     req = urllib.request.Request(metadata_url, headers={"Metadata-Flavor": "Google"})
 229 |     try:
 230 |         with urllib.request.urlopen(req, timeout=0.2) as response:
 231 |             value = response.read().decode("utf-8").strip()
 232 |             if value:
 233 |                 os.environ["TPU_ACCELERATOR_TYPE"] = value
 234 |                 return value
 235 |     except Exception:
 236 |         pass
 237 |     return ""
 238 | 
 239 | 
 240 | def get_tpu_num_chips() -> int:
 241 |     """Detect number of TPU chips on this host from accelerator type string.
 242 | 
 243 |     Parses e.g. 'v5litepod-8' -> 8, 'v6e-4' -> 4.  Falls back to 1.
 244 |     """
 245 |     accel = get_tpu_accelerator_type()
 246 |     if accel:
 247 |         try:
 248 |             return int(accel.rsplit('-', 1)[-1])
 249 |         except (ValueError, IndexError):
 250 |             pass
 251 |     return 1
 252 | 
 253 | 
 254 | def xla_all_reduce_gradients(model, world_size: int):
 255 |     """
 256 |     Average gradients across TPU workers for non-DDP XLA training loops.
 257 |     No-op when world_size <= 1 or torch_xla is unavailable.
 258 |     """
 259 |     if world_size <= 1:
 260 |         return
 261 |     try:
 262 |         import torch_xla.core.xla_model as xm
 263 |     except ImportError:
 264 |         return
 265 |     grads = [p.grad for p in model.parameters() if p.grad is not None]
 266 |     if grads:
 267 |         xm.all_reduce(xm.REDUCE_SUM, grads, scale=1.0 / world_size)
 268 | 
 269 | 
 270 | def compute_init(device_type="cuda"): # cuda|cpu|mps|xla
 271 |     """Basic initialization that we keep doing over and over, so make common."""
 272 | 
 273 |     assert device_type in ["cuda", "mps", "cpu", "xla"], "Invalid device type atm"
 274 |     if device_type == "cuda":
 275 |         assert torch.cuda.is_available(), "Your PyTorch installation is not configured for CUDA but device_type is 'cuda'"
 276 |     if device_type == "mps":
 277 |         assert torch.backends.mps.is_available(), "Your PyTorch installation is not configured for MPS but device_type is 'mps'"
 278 |     if device_type == "xla":
 279 |         try:
 280 |             import torch_xla.core.xla_model as xm
 281 |         except ImportError:
 282 |             raise RuntimeError("device_type is 'xla' but torch_xla is not installed")
 283 | 
 284 |     # Reproducibility
 285 |     # Note that we set the global seeds here, but most of the code uses explicit rng objects.
 286 |     # The only place where global rng might be used is nn.Module initialization of the model weights.
 287 |     torch.manual_seed(42)
 288 |     if device_type == "cuda":
 289 |         torch.cuda.manual_seed(42)
 290 |     # skipping full reproducibility for now, possibly investigate slowdown later
 291 |     # torch.use_deterministic_algorithms(True)
 292 | 
 293 |     # Precision (torch 2.9+ API)
 294 |     if device_type == "cuda":
 295 |         torch.backends.fp32_precision = "tf32"  # uses tf32 instead of fp32 for matmuls
 296 | 
 297 |     # Distributed setup: Distributed Data Parallel (DDP), optional, and requires CUDA
 298 |     ddp_requested = is_ddp_requested()
 299 |     if ddp_requested and device_type == "cuda":
 300 |         ddp_rank = int(os.environ["RANK"])
 301 |         ddp_local_rank = int(os.environ["LOCAL_RANK"])
 302 |         device = torch.device("cuda", ddp_local_rank)
 303 |         torch.cuda.set_device(device)  # make "cuda" default to this device
 304 |         dist.init_process_group(backend="nccl", device_id=device)
 305 |         dist.barrier()
 306 |     elif device_type == "xla":
 307 |         import torch_xla.core.xla_model as xm
 308 |         device = xm.xla_device()
 309 |         # Note: xr.use_spmd() is called in main() before compute_init() when
 310 |         # multi-chip TPU is detected. It must be called before xm.xla_device().
 311 |     else:
 312 |         device = torch.device(device_type) # mps|cpu
 313 | 
 314 |     is_distributed, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
 315 |     if ddp_rank == 0:
 316 |         logger.info(f"Distributed world size: {ddp_world_size}")
 317 | 
 318 |     return is_distributed, ddp_rank, ddp_local_rank, ddp_world_size, device
 319 | 
 320 | def compute_cleanup():
 321 |     """Companion function to compute_init, to clean things up before script exit"""
 322 |     if is_ddp_initialized():
 323 |         dist.destroy_process_group()
 324 | 
 325 | class DummyWandb:
 326 |     """Useful if we wish to not use wandb but have all the same signatures"""
 327 |     def __init__(self):
 328 |         pass
 329 |     def log(self, *args, **kwargs):
 330 |         pass
 331 |     def finish(self):
 332 |         pass

```

`nanochat/nanochat/cpp_tokenizer.py`:

```py
   1 | """
   2 | C++ Hybrid Tokenizer: fixed C++ vocab + learned BPE, BERT-style whitespace.
   3 | See docs/design/01-tokenizer.md for full design.
   4 | 
   5 | Encode works via HuggingFace tokenizers library.
   6 | Decode uses custom space reconstruction (BERT-style: insert spaces between word tokens).
   7 | """
   8 | import os
   9 | import json
  10 | from tokenizers import Tokenizer
  11 | 
  12 | 
  13 | # Single-char punctuation set
  14 | _PUNCT = set("{}()[]<>;:,.+-*/%&|^~!?=#@$_\\\"'")
  15 | # Multi-char operators
  16 | _MULTI_OPS = {"::", "->", ".*", "->*", "++", "--", "##", "==", "!=",
  17 |               "<=", ">=", "<=>", "&&", "||", "<<", ">>",
  18 |               "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=",
  19 |               "<<=", ">>=", "...", "//", "/*", "*/",
  20 |               "@@", "---", "+++", "a/", "b/"}
  21 | 
  22 | 
  23 | class CppTokenizer:
  24 |     """Hybrid C++ tokenizer: fixed vocab + learned BPE, BERT-style whitespace."""
  25 | 
  26 |     def __init__(self, tokenizer_path: str):
  27 |         if os.path.isdir(tokenizer_path):
  28 |             tokenizer_path = os.path.join(tokenizer_path, "tokenizer.json")
  29 |         self._tokenizer = Tokenizer.from_file(tokenizer_path)
  30 |         # Build reverse map for decoding
  31 |         self._vocab = self._tokenizer.get_vocab()
  32 |         self._id_to_token = {v: k for k, v in self._vocab.items()}
  33 |         # Build set of added/fixed-vocab tokens for BPE subword detection.
  34 |         # Added tokens are standalone words (keywords, STL names, etc.) that
  35 |         # should always get spaces around them. BPE-learned tokens may be
  36 |         # subword fragments that should attach to the previous token.
  37 |         added = self._tokenizer.get_added_tokens_decoder()
  38 |         self._added_token_ids = frozenset(added.keys())
  39 | 
  40 |     # --- nanochat-compatible API ---
  41 | 
  42 |     def get_bos_token_id(self):
  43 |         return self.bos_token_id
  44 | 
  45 |     def get_vocab_size(self):
  46 |         return self.vocab_size
  47 | 
  48 |     def get_special_tokens(self):
  49 |         # Return all angle-bracket tokens
  50 |         return [t for t in self._vocab if t.startswith("<") and t.endswith(">")]
  51 | 
  52 |     def encode_special(self, text):
  53 |         # Support both nanochat-style <|bos|> and our <BOS> format
  54 |         result = self._vocab.get(text, None)
  55 |         if result is None:
  56 |             # Map nanochat special tokens to our format
  57 |             mapping = {
  58 |                 "<|bos|>": "<BOS>", "<|eos|>": "<EOS>",
  59 |                 "<|pad|>": "<PAD>", "<|endoftext|>": "<BOS>",
  60 |                 # Tool-calling tokens (nanochat-style aliases)
  61 |                 "<|code_start|>": "<CODE_START>",
  62 |                 "<|code_end|>": "<CODE_END>",
  63 |                 "<|thought_start|>": "<THOUGHT_START>",
  64 |                 "<|thought_end|>": "<THOUGHT_END>",
  65 |                 "<|query_tool|>": "<QUERY_TOOL>",
  66 |                 "<|tool_result|>": "<TOOL_RESULT>",
  67 |             }
  68 |             mapped = mapping.get(text)
  69 |             if mapped:
  70 |                 result = self._vocab.get(mapped, None)
  71 |         return result
  72 | 
  73 |     def encode(self, text, prepend=None, append=None, num_threads=8):
  74 |         """Encode text or list of texts. Compatible with nanochat API."""
  75 |         if prepend is not None:
  76 |             prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
  77 |         if append is not None:
  78 |             append_id = append if isinstance(append, int) else self.encode_special(append)
  79 | 
  80 |         if isinstance(text, str):
  81 |             ids = self._tokenizer.encode(text).ids
  82 |             if prepend is not None:
  83 |                 ids.insert(0, prepend_id)
  84 |             if append is not None:
  85 |                 ids.append(append_id)
  86 |             return ids
  87 |         elif isinstance(text, list):
  88 |             results = [enc.ids for enc in self._tokenizer.encode_batch(text)]
  89 |             if prepend is not None:
  90 |                 for row in results:
  91 |                     row.insert(0, prepend_id)
  92 |             if append is not None:
  93 |                 for row in results:
  94 |                     row.append(append_id)
  95 |             return results
  96 |         else:
  97 |             raise ValueError(f"Invalid input type: {type(text)}")
  98 | 
  99 |     def __call__(self, *args, **kwargs):
 100 |         return self.encode(*args, **kwargs)
 101 | 
 102 |     def encode_batch(self, texts: list[str]) -> list[list[int]]:
 103 |         return [enc.ids for enc in self._tokenizer.encode_batch(texts)]
 104 | 
 105 |     def _is_bpe_suffix(self, token: str, token_id: int, prev_token: str = None, prev_id: int = None) -> bool:
 106 |         """Check if a token is a BPE suffix fragment (not a standalone word).
 107 | 
 108 |         Returns True for tokens like 'ype', 'nal', 'tion' that are never
 109 |         standalone C++ identifiers and always attach to the previous token.
 110 |         Conservative: only matches clear suffixes, not ambiguous short words
 111 |         like 'is', 'or', 'if' which could be standalone.
 112 | 
 113 |         When prev_token/prev_id are provided, uses context to make better
 114 |         decisions (e.g., 's' after 'char' is a suffix forming 'chars').
 115 |         """
 116 |         # Fixed-vocab (added) tokens are always standalone
 117 |         if token_id in self._added_token_ids:
 118 |             return False
 119 |         # Must be purely alphabetic
 120 |         if not token.isalpha():
 121 |             return False
 122 |         # Only lowercase tokens can be suffixes
 123 |         if not token.islower():
 124 |             return False
 125 |         # 1-2 char tokens: context-dependent suffix detection.
 126 |         # Common single-char variable names (n, x, i, c, b, f, etc.) must
 127 |         # NOT be treated as suffixes after type keywords (int n, char c).
 128 |         _COMMON_SINGLE_VARS = frozenset("abcdefghijklmnopqrstuvwxyz")
 129 |         _COMMON_TWO_CHAR_IDS = frozenset({
 130 |             # Common 2-char variable names and words used standalone in code
 131 |             "is", "it", "if", "in", "or", "on", "to", "do", "no",
 132 |             "at", "by", "up", "an", "be", "he", "me", "my", "of",
 133 |             "so", "we", "us", "id", "ok", "fn", "go",
 134 |         })
 135 |         if len(token) <= 2:
 136 |             if prev_id is None or prev_token is None or not prev_token[-1:].isalpha():
 137 |                 return False
 138 |             if prev_id in self._added_token_ids:
 139 |                 # After an added keyword: allow common suffix chars (s, d)
 140 |                 # that form plurals/past tense (chars, ints, used, signed)
 141 |                 if len(token) == 1 and token in "sd":
 142 |                     return True
 143 |                 # 2-char: allow only when NOT a common standalone identifier
 144 |                 if len(token) == 2 and token not in _COMMON_TWO_CHAR_IDS:
 145 |                     return True
 146 |                 return False
 147 |             # After a BPE fragment (not an added keyword): likely mid-word
 148 |             if prev_token.isalpha():
 149 |                 # But not if token looks like a common standalone variable/word
 150 |                 if len(token) == 1 and token in _COMMON_SINGLE_VARS:
 151 |                     # Single char after a non-keyword word is ambiguous.
 152 |                     # Heuristic: treat as suffix if prev is a known BPE
 153 |                     # fragment (short, uncommon as standalone word)
 154 |                     return len(prev_token) <= 4 and prev_token not in _COMMON_TWO_CHAR_IDS
 155 |                 if len(token) == 2 and token in _COMMON_TWO_CHAR_IDS:
 156 |                     return False
 157 |                 return True
 158 |             return False
 159 |         # 3+ char tokens: only treat as suffix if they look like one
 160 |         # (not a common standalone word)
 161 |         _COMMON_SHORT_WORDS = frozenset({
 162 |             # C++ keywords and common identifiers that happen to be short
 163 |             "add", "all", "and", "any", "arg", "bad", "bar", "big", "bit",
 164 |             "buf", "bus", "can", "cap", "car", "cmd", "col", "con", "cpu",
 165 |             "cur", "del", "dim", "dir", "dns", "doc", "dst", "dup", "end",
 166 |             "env", "err", "ext", "fan", "far", "fix", "fmt", "foo", "fun",
 167 |             "gap", "get", "got", "gpu", "has", "hex", "hit", "hot", "hub",
 168 |             "idx", "img", "inc", "ini", "key", "len", "lib", "log", "low",
 169 |             "map", "max", "mem", "mid", "min", "mix", "mod", "msg", "neg",
 170 |             "net", "new", "nil", "nop", "not", "now", "num", "obj", "odd",
 171 |             "old", "one", "opt", "ord", "out", "own", "pad", "par", "per",
 172 |             "pid", "pin", "pkg", "pop", "pos", "pre", "ptr", "put", "raw",
 173 |             "red", "ref", "reg", "rem", "rep", "req", "res", "ret", "rev",
 174 |             "rgb", "row", "run", "say", "sec", "set", "sim", "sin", "src",
 175 |             "std", "str", "sub", "sum", "syn", "sys", "tab", "tag", "tan",
 176 |             "tcp", "tmp", "top", "try", "tty", "two", "udp", "uid", "url",
 177 |             "use", "usr", "val", "var", "vec", "via", "vol", "was", "way",
 178 |             "web", "win", "xml", "xor", "yes", "zip",
 179 |             # Common English words (prevent false suffix detection in prose)
 180 |             "the", "for", "are", "but", "yet", "nor", "had", "his", "her",
 181 |             "its", "our", "who", "how", "may", "did", "let", "got", "ago",
 182 |             "few", "own", "too", "day", "see", "saw", "ran", "ask", "why",
 183 |             "also", "been", "does", "each", "goes", "just", "like", "made",
 184 |             "many", "much", "only", "same", "some", "such", "than", "them",
 185 |             "they", "very", "what", "when", "will", "your",
 186 |             # 4-char common words
 187 |             "auto", "back", "base", "bind", "body", "bool", "byte", "call",
 188 |             "case", "cast", "char", "code", "copy", "core", "ctrl", "data",
 189 |             "date", "dead", "deep", "diff", "disc", "done", "down", "drop",
 190 |             "dump", "edge", "edit", "else", "emit", "enum", "eval",
 191 |             "even", "exec", "exit", "expr", "face", "fail", "fast", "file",
 192 |             "fill", "find", "flag", "flat", "flip", "flow", "fold", "font",
 193 |             "fork", "form", "free", "from", "full", "func", "glob", "good",
 194 |             "goto", "grid", "grow", "half", "halt", "hand", "hash", "have",
 195 |             "head", "heap", "help", "here", "hide", "high", "hint", "hold",
 196 |             "home", "hook", "host", "http", "huge", "info", "init", "into",
 197 |             "item", "iter", "join", "jump", "keep", "kern", "kill",
 198 |             "kind", "last", "lazy", "leaf", "left", "less", "line",
 199 |             "link", "list", "live", "load", "lock", "long", "look", "loop",
 200 |             "lost", "main", "make", "mark", "mask", "math", "menu",
 201 |             "meta", "mode", "more", "most", "move", "must", "mute",
 202 |             "name", "near", "need", "next", "node", "none", "norm", "note",
 203 |             "null", "open", "over", "pack", "page", "pair", "part",
 204 |             "pass", "past", "path", "peek", "pick", "ping", "pipe", "plan",
 205 |             "play", "plot", "plus", "poll", "pool", "port", "post", "prev",
 206 |             "proc", "prog", "prop", "pull", "pure", "push", "quit", "rand",
 207 |             "rank", "rate", "read", "real", "rect", "redo", "rest", "ring",
 208 |             "root", "rule", "safe", "save", "scan", "seed", "seek",
 209 |             "self", "send", "show", "shut", "side", "sign", "size", "skip",
 210 |             "slot", "slow", "snap", "sock", "sort", "spec", "spin",
 211 |             "sqrt", "stat", "stay", "step", "stop", "swap", "sync",
 212 |             "tail", "take", "task", "temp", "term", "test", "text", "that",
 213 |             "then", "this", "tick", "time", "tiny", "todo", "tone", "tool",
 214 |             "tree", "trim", "true", "turn", "type", "uint", "undo", "unit",
 215 |             "unix", "used", "user", "view", "void", "wait", "walk",
 216 |             "want", "warn", "wide", "with", "word", "work", "wrap",
 217 |             "zero", "zone",
 218 |             # 5-char common words (prevent false suffix in natural language)
 219 |             "about", "after", "again", "array", "async", "await", "begin",
 220 |             "being", "below", "block", "break", "build", "bytes", "cache",
 221 |             "catch", "chain", "check", "child", "chunk", "class", "clean",
 222 |             "clear", "close", "color", "const", "count", "cover", "crash",
 223 |             "debug", "defer", "delta", "depth", "dirty", "empty", "error",
 224 |             "event", "every", "exact", "extra", "false", "fetch", "field",
 225 |             "final", "first", "fixed", "flags", "flush", "focus", "force",
 226 |             "found", "frame", "front", "given", "graph", "green", "group",
 227 |             "guard", "guess", "happy", "hash5", "heavy", "hence", "image",
 228 |             "index", "inner", "input", "inter", "items", "known", "label",
 229 |             "large", "later", "layer", "level", "light", "limit", "local",
 230 |             "lower", "magic", "major", "match", "merge", "minor", "model",
 231 |             "mouse", "multi", "mutex", "never", "newer", "nodes", "occur",
 232 |             "often", "older", "order", "other", "outer", "owned", "owner",
 233 |             "param", "parse", "patch", "pause", "phase", "pixel", "place",
 234 |             "plain", "point", "power", "press", "print", "prior", "probe",
 235 |             "proof", "proxy", "query", "queue", "quick", "quiet", "quota",
 236 |             "raise", "range", "rapid", "ratio", "ready", "realm", "refer",
 237 |             "reply", "reset", "retry", "right", "round", "route", "scale",
 238 |             "scene", "scope", "serve", "setup", "shape", "share", "sharp",
 239 |             "shift", "short", "since", "sleep", "slice", "small", "smart",
 240 |             "space", "spawn", "split", "stack", "stage", "start", "state",
 241 |             "still", "store", "strip", "super", "table", "taken", "their",
 242 |             "there", "thing", "think", "those", "throw", "timer", "times",
 243 |             "title", "token", "total", "trace", "track", "trait", "tries",
 244 |             "tuple", "under", "union", "until", "upper", "using", "utils",
 245 |             "valid", "value", "watch", "where", "which", "while", "white",
 246 |             "whole", "width", "world", "would", "write", "yield",
 247 |             # 6-char common words
 248 |             "accept", "access", "action", "active", "actual", "affect",
 249 |             "always", "amount", "append", "assert", "assign", "atomic",
 250 |             "attach", "before", "better", "binary", "branch", "bridge",
 251 |             "broken", "bucket", "buffer", "bundle", "called", "cancel",
 252 |             "change", "client", "closed", "column", "commit", "common",
 253 |             "config", "create", "cursor", "custom", "daemon", "decode",
 254 |             "define", "delete", "deploy", "design", "detail", "detect",
 255 |             "device", "digest", "direct", "double", "driver", "during",
 256 |             "enable", "encode", "enough", "ensure", "entity", "equals",
 257 |             "escape", "except", "export", "extend", "extern", "failed",
 258 |             "family", "figure", "filter", "finish", "follow", "format",
 259 |             "friend", "frozen", "future", "gather", "global", "google",
 260 |             "gotten", "handle", "header", "height", "helper", "hidden",
 261 |             "ignore", "import", "inline", "insert", "inside", "invoke",
 262 |             "island", "itself", "launch", "layout", "length", "likely",
 263 |             "linear", "listen", "little", "loader", "locked", "logger",
 264 |             "lookup", "manage", "manual", "mapper", "margin", "marker",
 265 |             "master", "matrix", "member", "memory", "method", "middle",
 266 |             "module", "moment", "mostly", "native", "nested", "normal",
 267 |             "notice", "notify", "number", "object", "obtain", "offset",
 268 |             "online", "opener", "option", "origin", "output", "packet",
 269 |             "parent", "parser", "passed", "prefer", "public", "random",
 270 |             "reader", "reason", "record", "reduce", "reload", "remove",
 271 |             "render", "repair", "repeat", "report", "result", "resume",
 272 |             "retain", "return", "revert", "review", "rewind", "runner",
 273 |             "sample", "schema", "scroll", "search", "secure", "select",
 274 |             "sender", "server", "signal", "signed", "simple", "single",
 275 |             "sizeof", "socket", "source", "splice", "status", "stderr",
 276 |             "stdout", "stored", "stream", "string", "struct", "submit",
 277 |             "suffix", "switch", "symbol", "syntax", "system", "target",
 278 |             "thread", "throws", "toggle", "traits", "update", "upload",
 279 |             "vector", "verify", "weight", "widget", "window", "worker",
 280 |             "writer",
 281 |         })
 282 |         if token in _COMMON_SHORT_WORDS:
 283 |             return False
 284 |         # Remaining 3-6 char lowercase BPE tokens are likely suffixes
 285 |         # (e.g., 'ype', 'nal', 'ern', 'tion', 'ment', 'ible', 'clude')
 286 |         if len(token) <= 6:
 287 |             return True
 288 |         return False
 289 | 
 290 |     def decode(self, ids: list[int]) -> str:
 291 |         """Decode token IDs to text with BERT-style space reconstruction.
 292 | 
 293 |         Heuristic rules for C++ spacing. Output is approximate —
 294 |         use clang-format for exact formatting.
 295 |         """
 296 |         tokens = [self._id_to_token.get(i, "<UNK>") for i in ids]
 297 |         if not tokens:
 298 |             return ""
 299 | 
 300 |         # C++ type keywords — pointer/reference operators attach to these
 301 |         # without spaces: char*, int&, bool*, float&, etc.
 302 |         _TYPE_KEYWORDS = frozenset({
 303 |             "void", "bool", "char", "short", "int", "long", "float", "double",
 304 |             "signed", "unsigned", "auto", "wchar_t", "char8_t", "char16_t",
 305 |             "char32_t", "size_t", "string", "vector", "map", "set", "list",
 306 |             "deque", "array", "pair", "tuple", "shared_ptr", "unique_ptr",
 307 |             "weak_ptr", "optional", "variant", "any",
 308 |         })
 309 | 
 310 |         # Track whether we are inside a multi-token identifier (joined via
 311 |         # underscores). After an underscore join, all subsequent alphabetic
 312 |         # BPE fragments should continue joining (e.g., end + _ + po + int
 313 |         # = end_point). This is distinct from pure BPE suffix joins which
 314 |         # should not propagate to unrelated standalone words.
 315 |         in_underscore_id = False  # are we inside an underscore identifier?
 316 |         # Track whether the previous join was a short BPE suffix fragment
 317 |         # (1-2 chars like 'po', 'st'). After such a suffix, the next
 318 |         # alphabetic token is likely a continuation of the same word
 319 |         # (e.g., check+po+int = checkpoint).
 320 |         prev_short_suffix = False
 321 | 
 322 |         def need_space(prev, curr, prev_id, curr_id):
 323 |             nonlocal in_underscore_id, prev_short_suffix
 324 |             # Save and reset suffix tracking
 325 |             was_short_suffix = prev_short_suffix
 326 |             prev_short_suffix = False
 327 | 
 328 |             # Whitespace tokens: never add extra space
 329 |             if curr in ("\n", "\n\n") or prev in ("\n", "\n\n"):
 330 |                 in_underscore_id = False
 331 |                 return False
 332 |             # Special tokens: no space
 333 |             if (curr.startswith("<") and curr.endswith(">") and len(curr) > 1):
 334 |                 in_underscore_id = False
 335 |                 return False
 336 |             if (prev.startswith("<") and prev.endswith(">") and len(prev) > 1):
 337 |                 in_underscore_id = False
 338 |                 return False
 339 |             # Underscore attaches to adjacent word/identifier tokens.
 340 |             # The pre-tokenizer splits on _, so it appears as a separate
 341 |             # token between identifier parts: is + _ + prime -> is_prime
 342 |             # But _ should NOT attach to operators: "x = _" stays spaced.
 343 |             if curr == "_" and (prev[-1:].isalnum() or prev == "_"):
 344 |                 in_underscore_id = True
 345 |                 return False
 346 |             if prev == "_" and (curr[0:1].isalnum() or curr == "_"):
 347 |                 in_underscore_id = True
 348 |                 return False
 349 |             # Inside an underscore identifier (e.g., after end_), BPE
 350 |             # fragments that are part of the identifier must join.
 351 |             # end + _ + po + int = end_point (po and int are fragments)
 352 |             if (in_underscore_id
 353 |                     and curr.isalpha()
 354 |                     and prev[-1:].isalpha()):
 355 |                 # Still inside the identifier: keep joining
 356 |                 return False
 357 |             # No longer inside an underscore identifier
 358 |             in_underscore_id = False
 359 |             # BPE suffix continuation: attach to previous token when it
 360 |             # looks like a word fragment (e.g., size_t + ype -> size_type,
 361 |             # char + s -> chars)
 362 |             is_suffix = (self._is_bpe_suffix(curr, curr_id, prev, prev_id)
 363 |                          and prev[-1:].isalpha())
 364 |             if is_suffix:
 365 |                 # Track short suffixes for the next iteration
 366 |                 if len(curr) <= 2:
 367 |                     prev_short_suffix = True
 368 |                 return False
 369 |             # After a short BPE suffix (like 'po'), the next alphabetic
 370 |             # token is likely a continuation of the same word, even if it's
 371 |             # an added keyword (e.g., check+po+int = checkpoint).
 372 |             if (was_short_suffix
 373 |                     and curr.isalpha()
 374 |                     and prev[-1:].isalpha()):
 375 |                 return False
 376 |             # Attach operators (no space either side): :: -> .* ->* ++ -- ##
 377 |             if curr in ("::", "->", ".*", "->*", "++", "--", "##"):
 378 |                 return False
 379 |             if prev in ("::", "->", ".*", "->*", "++", "--", "##"):
 380 |                 return False
 381 |             # . always attaches (member access)
 382 |             if curr == "." or prev == ".":
 383 |                 return False
 384 |             # No space after ( [ and no space before ) ] ; , .
 385 |             if prev in ("(", "["):
 386 |                 return False
 387 |             if curr in (")", "]", ";", ","):
 388 |                 return False
 389 |             # No space between word/keyword and (  e.g. main(, if(, for(
 390 |             if curr == "(":
 391 |                 return False
 392 |             # No space between word/keyword and [  e.g. argv[], data[]
 393 |             if curr == "[":
 394 |                 return False
 395 |             # Space before { (block opener)
 396 |             if curr == "{":
 397 |                 return True
 398 |             # Template angle brackets:
 399 |             # No space before < after known template types: vector<int>
 400 |             # No space after < in templates: <int>
 401 |             # No space before > in templates: int>
 402 |             # But comparison operators (i < n) should keep spaces.
 403 |             _TEMPLATE_TYPES = frozenset({
 404 |                 "vector", "map", "set", "list", "deque", "array",
 405 |                 "pair", "tuple", "queue", "stack", "multimap", "multiset",
 406 |                 "unordered_map", "unordered_set", "shared_ptr", "unique_ptr",
 407 |                 "weak_ptr", "optional", "variant", "function", "basic_string",
 408 |                 "span", "mdspan", "expected", "template",
 409 |             })
 410 |             if curr == "<" and prev in _TEMPLATE_TYPES:
 411 |                 return False
 412 |             if curr == ">" and prev_id in self._added_token_ids:
 413 |                 # > after a type keyword is likely closing template: int>
 414 |                 return False
 415 |             if prev == "<" and curr_id in self._added_token_ids:
 416 |                 # < before a type keyword is likely opening template: <int
 417 |                 return False
 418 |             # Pointer/reference operators attach to type keywords without space:
 419 |             # char* ptr, int& ref, const string& s
 420 |             # But keep space for binary usage: a * b, a & b
 421 |             if curr in ("*", "&") and prev in _TYPE_KEYWORDS:
 422 |                 return False
 423 |             if curr in ("*", "&") and prev == "const":
 424 |                 return False
 425 |             # Binary operators: always space around
 426 |             if curr in _MULTI_OPS or prev in _MULTI_OPS:
 427 |                 return True
 428 |             # Single-char operators: = + - * / % & | ^ ~ ! ? < > #
 429 |             if curr in "=+-*/%&|^~!?<>#" or prev in "=+-*/%&|^~!?<>#":
 430 |                 return True
 431 |             # Remaining: no space before : (label, access specifier)
 432 |             if curr == ":":
 433 |                 return False
 434 |             if prev == ":":
 435 |                 return True
 436 |             # Two word tokens: space
 437 |             return True
 438 | 
 439 |         parts = [tokens[0]]
 440 |         for i in range(1, len(tokens)):
 441 |             if need_space(tokens[i - 1], tokens[i], ids[i - 1], ids[i]):
 442 |                 parts.append(" ")
 443 |             parts.append(tokens[i])
 444 | 
 445 |         return "".join(parts)
 446 | 
 447 |     def id_to_token(self, id: int) -> str:
 448 |         return self._id_to_token.get(id, "<UNK>")
 449 | 
 450 |     @property
 451 |     def vocab_size(self) -> int:
 452 |         return self._tokenizer.get_vocab_size()
 453 | 
 454 |     @property
 455 |     def bos_token_id(self) -> int:
 456 |         return self._vocab.get("<BOS>", 2)
 457 | 
 458 |     @property
 459 |     def eos_token_id(self) -> int:
 460 |         return self._vocab.get("<EOS>", 3)
 461 | 
 462 |     @property
 463 |     def pad_token_id(self) -> int:
 464 |         return self._vocab.get("<PAD>", 0)
 465 | 
 466 |     # Tool-calling special token IDs
 467 |     @property
 468 |     def code_start_id(self) -> int:
 469 |         return self._vocab.get("<CODE_START>", 7)
 470 | 
 471 |     @property
 472 |     def code_end_id(self) -> int:
 473 |         return self._vocab.get("<CODE_END>", 8)
 474 | 
 475 |     @property
 476 |     def thought_start_id(self) -> int:
 477 |         return self._vocab.get("<THOUGHT_START>", 9)
 478 | 
 479 |     @property
 480 |     def thought_end_id(self) -> int:
 481 |         return self._vocab.get("<THOUGHT_END>", 10)
 482 | 
 483 |     @property
 484 |     def query_tool_id(self) -> int:
 485 |         return self._vocab.get("<QUERY_TOOL>", 11)
 486 | 
 487 |     @property
 488 |     def tool_result_id(self) -> int:
 489 |         return self._vocab.get("<TOOL_RESULT>", 19)

```

`nanochat/nanochat/engine.py`:

```py
   1 | """
   2 | Engine for efficient inference of our models.
   3 | 
   4 | Everything works around token sequences:
   5 | - The user can send token sequences to the engine
   6 | - The engine returns the next token
   7 | 
   8 | Notes:
   9 | - The engine knows nothing about tokenization, it's purely token id sequences.
  10 | 
  11 | The whole thing is made as efficient as possible.
  12 | """
  13 | 
  14 | import torch
  15 | import torch.nn.functional as F
  16 | import signal
  17 | import warnings
  18 | from contextlib import contextmanager
  19 | from collections import deque
  20 | from nanochat.common import compute_init, autodetect_device_type, no_grad_or_inference_mode
  21 | from nanochat.checkpoint_manager import load_model
  22 | from contextlib import nullcontext
  23 | 
  24 | # -----------------------------------------------------------------------------
  25 | # Calculator tool helpers
  26 | @contextmanager
  27 | def timeout(duration, formula):
  28 |     def timeout_handler(signum, frame):
  29 |         raise Exception(f"'{formula}': timed out after {duration} seconds")
  30 | 
  31 |     signal.signal(signal.SIGALRM, timeout_handler)
  32 |     signal.alarm(duration)
  33 |     yield
  34 |     signal.alarm(0)
  35 | 
  36 | def eval_with_timeout(formula, max_time=3):
  37 |     try:
  38 |         with timeout(max_time, formula):
  39 |             with warnings.catch_warnings():
  40 |                 warnings.simplefilter("ignore", SyntaxWarning)
  41 |                 return eval(formula, {"__builtins__": {}}, {})
  42 |     except Exception as e:
  43 |         signal.alarm(0)
  44 |         # print(f"Warning: Failed to eval {formula}, exception: {e}") # it's ok ignore wrong calculator usage
  45 |         return None
  46 | 
  47 | def use_calculator(expr):
  48 |     """
  49 |     Evaluate a Python expression safely.
  50 |     Supports both math expressions and string operations like .count()
  51 |     """
  52 |     # Remove commas from numbers
  53 |     expr = expr.replace(",", "")
  54 | 
  55 |     # Check if it's a pure math expression (old behavior)
  56 |     if all([x in "0123456789*+-/.() " for x in expr]):
  57 |         if "**" in expr:  # disallow power operator
  58 |             return None
  59 |         return eval_with_timeout(expr)
  60 | 
  61 |     # Check if it's a string operation we support
  62 |     # Allow: strings (single/double quotes), .count(), letters, numbers, spaces, parens
  63 |     allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
  64 |     if not all([x in allowed_chars for x in expr]):
  65 |         return None
  66 | 
  67 |     # Disallow dangerous patterns
  68 |     dangerous_patterns = ['__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
  69 |                          'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
  70 |                          'getattr', 'setattr', 'delattr', 'hasattr']
  71 |     expr_lower = expr.lower()
  72 |     if any(pattern in expr_lower for pattern in dangerous_patterns):
  73 |         return None
  74 | 
  75 |     # Only allow .count() method for now (can expand later)
  76 |     if '.count(' not in expr:
  77 |         return None
  78 | 
  79 |     # Evaluate with timeout
  80 |     return eval_with_timeout(expr)
  81 | 
  82 | # -----------------------------------------------------------------------------
  83 | class KVCache:
  84 |     """
  85 |     KV Cache designed for Flash Attention 3's flash_attn_with_kvcache API.
  86 | 
  87 |     Key differences from FA2-style cache:
  88 |     - Tensors are (B, T, H, D) not (B, H, T, D)
  89 |     - FA3 updates the cache in-place during flash_attn_with_kvcache
  90 |     - Position tracked per batch element via cache_seqlens tensor
  91 |     """
  92 | 
  93 |     def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers, device, dtype=torch.bfloat16):
  94 |         self.batch_size = batch_size
  95 |         self.max_seq_len = seq_len
  96 |         self.n_layers = num_layers
  97 |         self.n_heads = num_heads
  98 |         self.head_dim = head_dim
  99 |         # Pre-allocate cache tensors: (n_layers, B, T, H, D)
 100 |         self.k_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
 101 |         self.v_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
 102 |         # Current sequence length per batch element (FA3 needs int32)
 103 |         self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
 104 | 
 105 |     def reset(self):
 106 |         """Reset cache to empty state."""
 107 |         self.cache_seqlens.zero_()
 108 | 
 109 |     def get_pos(self):
 110 |         """Get current position (assumes all batch elements at same position)."""
 111 |         return self.cache_seqlens[0].item()
 112 | 
 113 |     def get_layer_cache(self, layer_idx):
 114 |         """Return (k_cache, v_cache) views for a specific layer."""
 115 |         return self.k_cache[layer_idx], self.v_cache[layer_idx]
 116 | 
 117 |     def advance(self, num_tokens):
 118 |         """Advance the cache position by num_tokens."""
 119 |         self.cache_seqlens += num_tokens
 120 | 
 121 |     def prefill(self, other):
 122 |         """
 123 |         Copy cached KV from another cache into this one.
 124 |         Used when we do batch=1 prefill and then want to generate multiple samples in parallel.
 125 |         """
 126 |         assert self.get_pos() == 0, "Cannot prefill a non-empty KV cache"
 127 |         assert self.n_layers == other.n_layers and self.n_heads == other.n_heads and self.head_dim == other.head_dim
 128 |         assert self.max_seq_len >= other.max_seq_len
 129 |         other_pos = other.get_pos()
 130 |         self.k_cache[:, :, :other_pos, :, :] = other.k_cache[:, :, :other_pos, :, :]
 131 |         self.v_cache[:, :, :other_pos, :, :] = other.v_cache[:, :, :other_pos, :, :]
 132 |         self.cache_seqlens.fill_(other_pos)
 133 | 
 134 | # -----------------------------------------------------------------------------
 135 | @no_grad_or_inference_mode()
 136 | def sample_next_token(logits, rng, temperature=1.0, top_k=None):
 137 |     """Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1)."""
 138 |     assert temperature >= 0.0, "temperature must be non-negative"
 139 |     if temperature == 0.0:
 140 |         return torch.argmax(logits, dim=-1, keepdim=True)
 141 |     orig_device = logits.device
 142 |     # torch.multinomial + Generator doesn't work on XLA; move to CPU for sampling
 143 |     is_xla = str(orig_device).startswith("xla")
 144 |     if is_xla:
 145 |         logits = logits.cpu()
 146 |     if top_k is not None and top_k > 0:
 147 |         k = min(top_k, logits.size(-1))
 148 |         vals, idx = torch.topk(logits, k, dim=-1)
 149 |         vals = vals / temperature
 150 |         probs = F.softmax(vals, dim=-1)
 151 |         choice = torch.multinomial(probs, num_samples=1, generator=rng)
 152 |         result = idx.gather(1, choice)
 153 |     else:
 154 |         logits = logits / temperature
 155 |         probs = F.softmax(logits, dim=-1)
 156 |         result = torch.multinomial(probs, num_samples=1, generator=rng)
 157 |     return result.to(orig_device) if is_xla else result
 158 | 
 159 | # -----------------------------------------------------------------------------
 160 | 
 161 | class RowState:
 162 |     # Per-row state tracking during generation
 163 |     def __init__(self, current_tokens=None):
 164 |         self.current_tokens = current_tokens or [] # Current token sequence for this row
 165 |         self.forced_tokens = deque() # Queue of tokens to force inject
 166 |         self.in_python_block = False # Whether we are inside a python block
 167 |         self.python_expr_tokens = [] # Tokens of the current python expression
 168 |         self.in_tool_block = False # Whether we are inside a <QUERY_TOOL>...<CODE_END> block
 169 |         self.tool_expr_tokens = [] # Tokens of the current tool call expression
 170 |         self.completed = False # Whether this row has completed generation
 171 | 
 172 | class Engine:
 173 | 
 174 |     def __init__(self, model, tokenizer, tool_runtime=None):
 175 |         self.model = model
 176 |         self.tokenizer = tokenizer # needed for tool use
 177 |         self.tool_runtime = tool_runtime # ToolRuntime instance for C++ tool calls
 178 | 
 179 |     @torch.no_grad()
 180 |     def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
 181 |         """Same as generate, but does single prefill and then clones the KV cache."""
 182 |         assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"
 183 |         device = self.model.get_device()
 184 |         # torch.Generator doesn't support XLA devices; use CPU generator instead
 185 |         rng_device = "cpu" if str(device).startswith("xla") else device
 186 |         rng = torch.Generator(device=rng_device)
 187 |         rng.manual_seed(seed)
 188 | 
 189 |         # Get the special tokens we need to coordinate the tool use state machine
 190 |         def get_special(s):
 191 |             try:
 192 |                 return self.tokenizer.encode_special(s)
 193 |             except Exception:
 194 |                 return None
 195 |         python_start = get_special("<|python_start|>")
 196 |         python_end = get_special("<|python_end|>")
 197 |         output_start = get_special("<|output_start|>")
 198 |         output_end = get_special("<|output_end|>")
 199 |         assistant_end = get_special("<|assistant_end|>") # if sampled, ends row
 200 |         bos = self.tokenizer.get_bos_token_id() # if sampled, ends row
 201 | 
 202 |         # C++ tool-call tokens (from C++ tokenizer)
 203 |         query_tool = get_special("<QUERY_TOOL>")   # ID 11
 204 |         tool_result = get_special("<TOOL_RESULT>")  # ID 19
 205 |         code_end = get_special("<CODE_END>")        # ID 8
 206 |         eos = get_special("<EOS>")                  # ID 3
 207 | 
 208 |         # 1) Run a batch 1 prefill of the prompt tokens
 209 |         m = self.model.config
 210 |         kv_model_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}
 211 |         kv_cache_prefill = KVCache(
 212 |             batch_size=1,
 213 |             seq_len=len(tokens),
 214 |             device=device,
 215 |             **kv_model_kwargs,
 216 |         )
 217 |         ids = torch.tensor([tokens], dtype=torch.long, device=device)
 218 |         logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
 219 |         logits = logits[:, -1, :].expand(num_samples, -1)  # (num_samples, vocab_size)
 220 | 
 221 |         # 2) Replicate the KV cache for each sample/row
 222 |         kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
 223 |         kv_cache_decode = KVCache(
 224 |             batch_size=num_samples,
 225 |             seq_len=kv_length_hint,
 226 |             device=device,
 227 |             **kv_model_kwargs,
 228 |         )
 229 |         kv_cache_decode.prefill(kv_cache_prefill)
 230 |         del kv_cache_prefill # no need to keep this memory around
 231 | 
 232 |         # 3) Initialize states for each sample
 233 |         row_states = [RowState(tokens.copy()) for _ in range(num_samples)]
 234 | 
 235 |         # 4) Main generation loop
 236 |         num_generated = 0
 237 |         while True:
 238 |             # Stop condition: we've reached max tokens
 239 |             if max_tokens is not None and num_generated >= max_tokens:
 240 |                 break
 241 |             # Stop condition: all rows are completed
 242 |             if all(state.completed for state in row_states):
 243 |                 break
 244 | 
 245 |             # Sample the next token for each row
 246 |             next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
 247 |             sampled_tokens = next_ids[:, 0].tolist()
 248 | 
 249 |             # Process each row: choose the next token, update state, optional tool use
 250 |             token_column = [] # contains the next token id along each row
 251 |             token_masks = [] # contains the mask (was it sampled (1) or forced (0)?) along each row
 252 |             for i, state in enumerate(row_states):
 253 |                 # Select the next token in this row
 254 |                 is_forced = len(state.forced_tokens) > 0 # are there tokens waiting to be forced in deque?
 255 |                 token_masks.append(0 if is_forced else 1) # mask is 0 if forced, 1 if sampled
 256 |                 next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
 257 |                 token_column.append(next_token)
 258 |                 # Update the state of this row to include the next token
 259 |                 state.current_tokens.append(next_token)
 260 |                 # On <|assistant_end|>, <|bos|>, or <EOS>, mark the row as completed
 261 |                 if next_token == assistant_end or next_token == bos:
 262 |                     state.completed = True
 263 |                 if eos is not None and next_token == eos:
 264 |                     state.completed = True
 265 |                 # Handle Python calculator tool logic
 266 |                 if next_token == python_start:
 267 |                     state.in_python_block = True
 268 |                     state.python_expr_tokens = []
 269 |                 elif next_token == python_end and state.in_python_block:
 270 |                     state.in_python_block = False
 271 |                     if state.python_expr_tokens:
 272 |                         expr = self.tokenizer.decode(state.python_expr_tokens)
 273 |                         result = use_calculator(expr)
 274 |                         if result is not None:
 275 |                             result_tokens = self.tokenizer.encode(str(result))
 276 |                             state.forced_tokens.append(output_start)
 277 |                             state.forced_tokens.extend(result_tokens)
 278 |                             state.forced_tokens.append(output_end)
 279 |                     state.python_expr_tokens = []
 280 |                 elif state.in_python_block:
 281 |                     state.python_expr_tokens.append(next_token)
 282 |                 # Handle C++ tool-call logic: <QUERY_TOOL> expr <CODE_END>
 283 |                 if query_tool is not None and next_token == query_tool:
 284 |                     state.in_tool_block = True
 285 |                     state.tool_expr_tokens = []
 286 |                 elif code_end is not None and next_token == code_end and state.in_tool_block:
 287 |                     state.in_tool_block = False
 288 |                     if state.tool_expr_tokens and self.tool_runtime is not None:
 289 |                         expr = self.tokenizer.decode(state.tool_expr_tokens)
 290 |                         result = self.tool_runtime.execute(expr)
 291 |                         if result is not None and tool_result is not None and code_end is not None:
 292 |                             result_tokens = self.tokenizer.encode(str(result))
 293 |                             state.forced_tokens.append(tool_result)
 294 |                             state.forced_tokens.extend(result_tokens)
 295 |                             state.forced_tokens.append(code_end)
 296 |                     state.tool_expr_tokens = []
 297 |                 elif state.in_tool_block:
 298 |                     state.tool_expr_tokens.append(next_token)
 299 | 
 300 |             # Yield the token column
 301 |             yield token_column, token_masks
 302 |             num_generated += 1
 303 | 
 304 |             # Prepare logits for next iteration
 305 |             ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
 306 |             logits = self.model.forward(ids, kv_cache=kv_cache_decode)[:, -1, :]  # (B, vocab_size)
 307 | 
 308 |     def generate_batch(self, tokens, num_samples=1, **kwargs):
 309 |         """
 310 |         Non-streaming batch generation that just returns the final token sequences.
 311 |         Returns a list of token sequences (list of lists of ints).
 312 |         Terminal tokens (assistant_end, bos, eos) are not included in the results.
 313 |         """
 314 |         assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
 315 |         bos = self.tokenizer.get_bos_token_id()
 316 |         eos = self.tokenizer.encode_special("<EOS>")
 317 |         results = [tokens.copy() for _ in range(num_samples)]
 318 |         masks = [[0] * len(tokens) for _ in range(num_samples)]
 319 |         completed = [False] * num_samples
 320 |         for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
 321 |             for i, (token, mask) in enumerate(zip(token_column, token_masks)):
 322 |                 if not completed[i]:
 323 |                     if token == assistant_end or token == bos or (eos is not None and token == eos):
 324 |                         completed[i] = True
 325 |                     else:
 326 |                         results[i].append(token)
 327 |                         masks[i].append(mask)
 328 |             # Stop if all rows are completed
 329 |             if all(completed):
 330 |                 break
 331 |         return results, masks
 332 | 
 333 | 
 334 | if __name__ == "__main__":
 335 |     """
 336 |     Quick inline test to make sure that the naive/slow model.generate function
 337 |     is equivalent to the faster Engine.generate function here.
 338 |     """
 339 |     import time
 340 |     # init compute
 341 |     ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
 342 |     device_type = autodetect_device_type()
 343 |     autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
 344 | 
 345 |     # load the model and tokenizer
 346 |     model, tokenizer, meta = load_model("base", device, phase="eval")
 347 |     bos_token_id = tokenizer.get_bos_token_id()
 348 |     # common hyperparameters
 349 |     kwargs = dict(max_tokens=64, temperature=0.0)
 350 |     # set the starting prompt
 351 |     prompt_tokens = tokenizer.encode("The chemical formula of water is", prepend=bos_token_id)
 352 |     # generate the reference sequence using the model.generate() function
 353 |     generated_tokens = []
 354 |     torch.cuda.synchronize()
 355 |     t0 = time.time()
 356 |     stream = model.generate(prompt_tokens, **kwargs)
 357 |     with autocast_ctx:
 358 |         for token in stream:
 359 |             generated_tokens.append(token)
 360 |             chunk = tokenizer.decode([token])
 361 |             print(chunk, end="", flush=True)
 362 |     print()
 363 |     torch.cuda.synchronize()
 364 |     t1 = time.time()
 365 |     print(f"Reference time: {t1 - t0:.2f}s")
 366 |     reference_ids = generated_tokens
 367 |     # generate tokens with Engine
 368 |     generated_tokens = []
 369 |     engine = Engine(model, tokenizer)
 370 |     stream = engine.generate(prompt_tokens, num_samples=1, **kwargs) # note: runs in fp32
 371 |     torch.cuda.synchronize()
 372 |     t0 = time.time()
 373 |     with autocast_ctx:
 374 |         for token_column, token_masks in stream:
 375 |             token = token_column[0] # only print out the first row
 376 |             generated_tokens.append(token)
 377 |             chunk = tokenizer.decode([token])
 378 |             print(chunk, end="", flush=True)
 379 |     print()
 380 |     torch.cuda.synchronize()
 381 |     t1 = time.time()
 382 |     print(f"Engine time: {t1 - t0:.2f}s")
 383 |     # compare the two sequences
 384 |     for i in range(len(reference_ids)):
 385 |         if reference_ids[i] != generated_tokens[i]:
 386 |             print(f"Mismatch at {i}: {reference_ids[i]} != {generated_tokens[i]}")
 387 |             break
 388 |     print(f"Match: {reference_ids == generated_tokens}")

```

`nanochat/nanochat/fim.py`:

```py
   1 | """
   2 | Fill-in-the-Middle (FIM) augmentation for training data.
   3 | 
   4 | Applies FIM transformation to token sequences, supporting both PSM and SPM formats.
   5 | See: "Efficient Training of Language Models to Fill in the Middle" (Bavarian et al., 2022)
   6 | 
   7 | FIM token IDs (from our C++ tokenizer):
   8 |   <FIM_PREFIX> = 4
   9 |   <FIM_MIDDLE> = 5
  10 |   <FIM_SUFFIX> = 6
  11 |   <EOS>        = 3  (used as EOT sentinel)
  12 | 
  13 | Two FIM modes:
  14 | 1. Random FIM: Random splits (original) - good for general code infilling
  15 | 2. Structured FIM: Docstring→body splits - good for comment→code completion
  16 | """
  17 | 
  18 | import os
  19 | import json
  20 | import random
  21 | from typing import Optional
  22 | 
  23 | 
  24 | # Default token IDs matching our C++ tokenizer (scripts/tok_train_cpp.py)
  25 | FIM_PREFIX_ID = 4   # <FIM_PREFIX>
  26 | FIM_MIDDLE_ID = 5   # <FIM_MIDDLE>
  27 | FIM_SUFFIX_ID = 6   # <FIM_SUFFIX>
  28 | EOT_ID = 3          # <EOS> used as end-of-turn
  29 | 
  30 | 
  31 | def apply_fim(
  32 |     token_ids: list[int],
  33 |     fim_rate: float = 0.5,
  34 |     spm_rate: float = 0.5,
  35 |     fim_prefix_id: int = FIM_PREFIX_ID,
  36 |     fim_middle_id: int = FIM_MIDDLE_ID,
  37 |     fim_suffix_id: int = FIM_SUFFIX_ID,
  38 |     eot_id: int = EOT_ID,
  39 |     rng: random.Random | None = None,
  40 | ) -> list[int]:
  41 |     """Apply FIM transformation to a single token sequence.
  42 | 
  43 |     With probability `fim_rate`, rearranges tokens into FIM format.
  44 |     Otherwise returns the original sequence unchanged.
  45 | 
  46 |     Two FIM formats are used (chosen with probability `spm_rate` for SPM):
  47 |       PSM: <FIM_PREFIX> prefix <FIM_SUFFIX> suffix <FIM_MIDDLE> middle <EOT>
  48 |       SPM: <FIM_PREFIX> <FIM_SUFFIX> suffix <FIM_MIDDLE> prefix middle <EOT>
  49 | 
  50 |     Args:
  51 |         token_ids: Original token sequence (without BOS/EOS -- just content tokens).
  52 |         fim_rate: Probability of applying FIM (default 0.5).
  53 |         spm_rate: Probability of using SPM format when FIM is applied (default 0.5).
  54 |         fim_prefix_id: Token ID for <FIM_PREFIX>.
  55 |         fim_middle_id: Token ID for <FIM_MIDDLE>.
  56 |         fim_suffix_id: Token ID for <FIM_SUFFIX>.
  57 |         eot_id: Token ID for end-of-turn sentinel.
  58 |         rng: Optional random.Random instance for reproducibility.
  59 | 
  60 |     Returns:
  61 |         Transformed token list (FIM format) or original token list.
  62 |     """
  63 |     if rng is None:
  64 |         rng = random
  65 | 
  66 |     if rng.random() > fim_rate:
  67 |         return token_ids  # no FIM -- standard next-token prediction
  68 | 
  69 |     n = len(token_ids)
  70 |     if n < 2:
  71 |         return token_ids  # too short to split meaningfully
  72 | 
  73 |     # Pick two random split points to divide into prefix | middle | suffix
  74 |     split_start = rng.randint(0, n)
  75 |     split_end = rng.randint(split_start, n)
  76 | 
  77 |     prefix = token_ids[:split_start]
  78 |     middle = token_ids[split_start:split_end]
  79 |     suffix = token_ids[split_end:]
  80 | 
  81 |     if rng.random() < spm_rate:
  82 |         # SPM: suffix-prefix-middle (suffix comes right after sentinel, then
  83 |         # prefix+middle are contiguous so the model learns to continue from prefix)
  84 |         return [fim_prefix_id, fim_suffix_id] + suffix + [fim_middle_id] + prefix + middle + [eot_id]
  85 |     else:
  86 |         # PSM: prefix-suffix-middle
  87 |         return [fim_prefix_id] + prefix + [fim_suffix_id] + suffix + [fim_middle_id] + middle + [eot_id]
  88 | 
  89 | 
  90 | def apply_fim_batch(
  91 |     token_lists: list[list[int]],
  92 |     fim_rate: float = 0.5,
  93 |     spm_rate: float = 0.5,
  94 |     fim_prefix_id: int = FIM_PREFIX_ID,
  95 |     fim_middle_id: int = FIM_MIDDLE_ID,
  96 |     fim_suffix_id: int = FIM_SUFFIX_ID,
  97 |     eot_id: int = EOT_ID,
  98 |     rng: random.Random | None = None,
  99 | ) -> list[list[int]]:
 100 |     """Apply FIM transformation to a batch of token sequences.
 101 | 
 102 |     Args:
 103 |         token_lists: List of token sequences.
 104 |         Other args: same as apply_fim.
 105 | 
 106 |     Returns:
 107 |         List of (possibly transformed) token sequences.
 108 |     """
 109 |     return [
 110 |         apply_fim(
 111 |             toks,
 112 |             fim_rate=fim_rate,
 113 |             spm_rate=spm_rate,
 114 |             fim_prefix_id=fim_prefix_id,
 115 |             fim_middle_id=fim_middle_id,
 116 |             fim_suffix_id=fim_suffix_id,
 117 |             eot_id=eot_id,
 118 |             rng=rng,
 119 |         )
 120 |         for toks in token_lists
 121 |     ]
 122 | 
 123 | 
 124 | # -----------------------------------------------------------------------------
 125 | # Structured FIM: Docstring → Function Body completion
 126 | # -----------------------------------------------------------------------------
 127 | 
 128 | class StructuredFIMDataset:
 129 |     """
 130 |     Provides structured FIM examples from pre-extracted docstring pairs.
 131 | 
 132 |     Format: <FIM_PREFIX> docstring + signature { <FIM_SUFFIX> } <FIM_MIDDLE> body <EOT>
 133 | 
 134 |     This teaches the model to complete function bodies given docstrings.
 135 |     """
 136 | 
 137 |     def __init__(self, pairs_path: str, tokenizer, max_examples: int = -1):
 138 |         """
 139 |         Args:
 140 |             pairs_path: Path to JSONL file with docstring pairs
 141 |             tokenizer: Tokenizer instance (must have encode method)
 142 |             max_examples: Max examples to load (-1 = all)
 143 |         """
 144 |         self.pairs = []
 145 |         self.tokenizer = tokenizer
 146 | 
 147 |         if not os.path.exists(pairs_path):
 148 |             print(f"Warning: Structured FIM dataset not found: {pairs_path}")
 149 |             return
 150 | 
 151 |         with open(pairs_path, 'r') as f:
 152 |             for i, line in enumerate(f):
 153 |                 if max_examples > 0 and i >= max_examples:
 154 |                     break
 155 |                 if line.strip():
 156 |                     self.pairs.append(json.loads(line))
 157 | 
 158 |         print(f"Loaded {len(self.pairs):,} structured FIM examples from {pairs_path}")
 159 | 
 160 |     def __len__(self):
 161 |         return len(self.pairs)
 162 | 
 163 |     def get_random_example(self, rng: random.Random = None) -> Optional[list[int]]:
 164 |         """
 165 |         Get a random structured FIM example as token IDs.
 166 | 
 167 |         Returns:
 168 |             Token list in FIM format, or None if no examples available.
 169 |         """
 170 |         if not self.pairs:
 171 |             return None
 172 | 
 173 |         if rng is None:
 174 |             rng = random
 175 | 
 176 |         pair = rng.choice(self.pairs)
 177 |         return self.pair_to_tokens(pair)
 178 | 
 179 |     def pair_to_tokens(
 180 |         self,
 181 |         pair: dict,
 182 |         fim_prefix_id: int = FIM_PREFIX_ID,
 183 |         fim_middle_id: int = FIM_MIDDLE_ID,
 184 |         fim_suffix_id: int = FIM_SUFFIX_ID,
 185 |         eot_id: int = EOT_ID,
 186 |     ) -> list[int]:
 187 |         """
 188 |         Convert a docstring pair to FIM token sequence.
 189 | 
 190 |         Format: <FIM_PREFIX> /* docstring */ signature { <FIM_SUFFIX> } <FIM_MIDDLE> body <EOT>
 191 |         """
 192 |         docstring = pair.get('docstring', '')
 193 |         signature = pair.get('signature', '')
 194 |         body = pair.get('body', '')
 195 | 
 196 |         # Build prefix: /* docstring */ signature {
 197 |         prefix_text = f"/*\n{docstring}\n*/\n{signature} {{"
 198 | 
 199 |         # Build suffix: just the closing brace
 200 |         suffix_text = "\n}"
 201 | 
 202 |         # Tokenize
 203 |         prefix_tokens = self.tokenizer.encode(prefix_text)
 204 |         suffix_tokens = self.tokenizer.encode(suffix_text)
 205 |         body_tokens = self.tokenizer.encode("\n" + body + "\n")
 206 | 
 207 |         # PSM format: <FIM_PREFIX> prefix <FIM_SUFFIX> suffix <FIM_MIDDLE> middle <EOT>
 208 |         return (
 209 |             [fim_prefix_id] +
 210 |             prefix_tokens +
 211 |             [fim_suffix_id] +
 212 |             suffix_tokens +
 213 |             [fim_middle_id] +
 214 |             body_tokens +
 215 |             [eot_id]
 216 |         )
 217 | 
 218 | 
 219 | def apply_fim_mixed(
 220 |     token_ids: list[int],
 221 |     structured_dataset: Optional[StructuredFIMDataset],
 222 |     fim_rate: float = 0.4,
 223 |     structured_rate: float = 0.2,
 224 |     spm_rate: float = 0.5,
 225 |     fim_prefix_id: int = FIM_PREFIX_ID,
 226 |     fim_middle_id: int = FIM_MIDDLE_ID,
 227 |     fim_suffix_id: int = FIM_SUFFIX_ID,
 228 |     eot_id: int = EOT_ID,
 229 |     rng: random.Random = None,
 230 | ) -> list[int]:
 231 |     """
 232 |     Apply mixed FIM: either random FIM, structured FIM, or no FIM.
 233 | 
 234 |     Probabilities:
 235 |     - structured_rate: Use structured FIM (docstring→body)
 236 |     - fim_rate: Use random FIM
 237 |     - (1 - structured_rate - fim_rate): No FIM (standard next-token)
 238 | 
 239 |     Args:
 240 |         token_ids: Original token sequence
 241 |         structured_dataset: StructuredFIMDataset instance (can be None)
 242 |         fim_rate: Probability of random FIM
 243 |         structured_rate: Probability of structured FIM
 244 |         Other args: same as apply_fim
 245 | 
 246 |     Returns:
 247 |         Token list (possibly transformed)
 248 |     """
 249 |     if rng is None:
 250 |         rng = random
 251 | 
 252 |     roll = rng.random()
 253 | 
 254 |     # Try structured FIM first
 255 |     if roll < structured_rate and structured_dataset is not None:
 256 |         example = structured_dataset.get_random_example(rng)
 257 |         if example is not None:
 258 |             return example
 259 |         # Fall through to random FIM if no structured examples
 260 | 
 261 |     # Random FIM
 262 |     if roll < structured_rate + fim_rate:
 263 |         return apply_fim(
 264 |             token_ids,
 265 |             fim_rate=1.0,  # Always apply since we already rolled
 266 |             spm_rate=spm_rate,
 267 |             fim_prefix_id=fim_prefix_id,
 268 |             fim_middle_id=fim_middle_id,
 269 |             fim_suffix_id=fim_suffix_id,
 270 |             eot_id=eot_id,
 271 |             rng=rng,
 272 |         )
 273 | 
 274 |     # No FIM
 275 |     return token_ids
 276 | 
 277 | 
 278 | def apply_fim_mixed_batch(
 279 |     token_lists: list[list[int]],
 280 |     structured_dataset: Optional[StructuredFIMDataset],
 281 |     fim_rate: float = 0.4,
 282 |     structured_rate: float = 0.2,
 283 |     spm_rate: float = 0.5,
 284 |     fim_prefix_id: int = FIM_PREFIX_ID,
 285 |     fim_middle_id: int = FIM_MIDDLE_ID,
 286 |     fim_suffix_id: int = FIM_SUFFIX_ID,
 287 |     eot_id: int = EOT_ID,
 288 |     rng: random.Random = None,
 289 | ) -> list[list[int]]:
 290 |     """Apply mixed FIM to a batch of token sequences."""
 291 |     return [
 292 |         apply_fim_mixed(
 293 |             toks,
 294 |             structured_dataset,
 295 |             fim_rate=fim_rate,
 296 |             structured_rate=structured_rate,
 297 |             spm_rate=spm_rate,
 298 |             fim_prefix_id=fim_prefix_id,
 299 |             fim_middle_id=fim_middle_id,
 300 |             fim_suffix_id=fim_suffix_id,
 301 |             eot_id=eot_id,
 302 |             rng=rng,
 303 |         )
 304 |         for toks in token_lists
 305 |     ]

```

`nanochat/nanochat/flash_attention.py`:

```py
   1 | """
   2 | Unified Flash Attention interface with automatic FA3/SDPA switching.
   3 | 
   4 | Exports `flash_attn` module that matches the FA3 API exactly, but falls back
   5 | to PyTorch SDPA on unsupported GPUs, MPS, and CPU.
   6 | 
   7 | Our local FA3 build supports SM121 (GB10/DGX Spark). For other GPUs, use SDPA.
   8 | 
   9 | Usage (drop-in replacement for FA3):
  10 |     from nanochat.flash_attention import flash_attn_func, flash_attn_with_kvcache
  11 | 
  12 |     # Training (no KV cache)
  13 |     y = flash_attn_func(q, k, v, causal=True, window_size=window_size)
  14 | 
  15 |     # Inference (with KV cache)
  16 |     y = flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
  17 | """
  18 | import torch
  19 | import torch.nn.functional as F
  20 | 
  21 | 
  22 | # =============================================================================
  23 | # Detection: Try to load local FA3 build
  24 | # =============================================================================
  25 | def _load_flash_attention():
  26 |     """Try to load Flash Attention (local build or SDPA fallback)."""
  27 |     if not torch.cuda.is_available():
  28 |         return None, "no_cuda"
  29 | 
  30 |     try:
  31 |         major, minor = torch.cuda.get_device_capability()
  32 |         sm = major * 10 + minor
  33 | 
  34 |         # Our local FA3 build supports:
  35 |         # - SM90 (Hopper: H100, H200)
  36 |         # - SM121 (GB10/DGX Spark) - custom build
  37 |         # Blackwell (SM100) and Ada (SM89) need SDPA fallback
  38 |         if sm in (90, 121):
  39 |             from flash_attn import flash_attn_func as fa3_func
  40 |             from flash_attn import flash_attn_with_kvcache as fa3_kvcache
  41 |             return (fa3_func, fa3_kvcache), f"fa3_sm{sm}"
  42 |         else:
  43 |             return None, f"sdpa_sm{sm}"
  44 | 
  45 |     except ImportError as e:
  46 |         return None, f"sdpa_import_error:{e}"
  47 |     except Exception as e:
  48 |         return None, f"sdpa_error:{e}"
  49 | 
  50 | 
  51 | _fa3_funcs, _backend_info = _load_flash_attention()
  52 | HAS_FA3 = _fa3_funcs is not None
  53 | 
  54 | # =============================================================================
  55 | # XLA Flash Attention (TPU Pallas kernels)
  56 | # =============================================================================
  57 | _xla_flash_attn = None
  58 | _xla_flash_enabled = False
  59 | _spmd_mesh = None
  60 | _spmd_partition_spec = None
  61 | 
  62 | # =============================================================================
  63 | # Chunked Attention (memory-efficient, pure PyTorch, works on XLA/TPU)
  64 | # =============================================================================
  65 | _chunked_attn_enabled = False
  66 | _chunked_attn_chunk_size = 1024
  67 | _chunked_attn_threshold = 2048  # only use for seq_len > this
  68 | 
  69 | def enable_chunked_attention(chunk_size=1024, threshold=2048):
  70 |     """Enable chunked attention for long sequences on XLA/TPU.
  71 | 
  72 |     This avoids materializing the full O(n^2) attention matrix by processing
  73 |     queries in chunks. Reduces peak attention memory from O(n^2) to O(n * chunk_size).
  74 |     Uses the online softmax trick (same as FlashAttention) for numerical stability.
  75 |     """
  76 |     global _chunked_attn_enabled, _chunked_attn_chunk_size, _chunked_attn_threshold, _backend_info
  77 |     _chunked_attn_enabled = True
  78 |     _chunked_attn_chunk_size = chunk_size
  79 |     _chunked_attn_threshold = threshold
  80 |     _backend_info = f"chunked_attn_c{chunk_size}"
  81 |     print(f"Chunked attention enabled: chunk_size={chunk_size}, threshold={threshold}")
  82 | 
  83 | 
  84 | def _chunked_attention(q, k, v, chunk_size=1024, window=-1):
  85 |     """Memory-efficient causal attention using chunked computation.
  86 | 
  87 |     Processes queries in chunks of `chunk_size`, only attending to valid keys
  88 |     (causal mask). Uses online softmax for numerical stability without
  89 |     materializing the full (T, T) attention matrix.
  90 | 
  91 |     Args:
  92 |         q, k, v: (B, H, T, D) tensors
  93 |         chunk_size: number of query tokens per chunk
  94 |         window: sliding window size (-1 for full causal)
  95 |     Returns:
  96 |         (B, H, T, D) output tensor
  97 |     """
  98 |     B, H, T, D = q.shape
  99 |     scale = D ** -0.5
 100 | 
 101 |     # Pad T to multiple of chunk_size if needed
 102 |     pad = (chunk_size - T % chunk_size) % chunk_size
 103 |     if pad > 0:
 104 |         q = F.pad(q, (0, 0, 0, pad))
 105 |         k = F.pad(k, (0, 0, 0, pad))
 106 |         v = F.pad(v, (0, 0, 0, pad))
 107 |         T_padded = T + pad
 108 |     else:
 109 |         T_padded = T
 110 | 
 111 |     n_chunks = T_padded // chunk_size
 112 |     outputs = []
 113 | 
 114 |     for i in range(n_chunks):
 115 |         q_start = i * chunk_size
 116 |         q_end = q_start + chunk_size
 117 |         q_chunk = q[:, :, q_start:q_end, :]  # (B, H, chunk, D)
 118 | 
 119 |         # Determine key range (causal: only up to q_end)
 120 |         if window > 0:
 121 |             k_start = max(0, q_end - window)
 122 |         else:
 123 |             k_start = 0
 124 |         k_end = q_end
 125 | 
 126 |         k_slice = k[:, :, k_start:k_end, :]  # (B, H, k_len, D)
 127 |         v_slice = v[:, :, k_start:k_end, :]
 128 | 
 129 |         # Compute attention scores: (B, H, chunk, k_len)
 130 |         attn = torch.matmul(q_chunk, k_slice.transpose(-2, -1)) * scale
 131 | 
 132 |         # Apply causal mask within this chunk
 133 |         k_len = k_end - k_start
 134 |         # Row i in q_chunk corresponds to global position q_start + i
 135 |         # Col j in k_slice corresponds to global position k_start + j
 136 |         # Causal: global_row >= global_col => q_start + i >= k_start + j
 137 |         row_offset = q_start - k_start
 138 |         row_idx = torch.arange(chunk_size, device=q.device).unsqueeze(1) + row_offset
 139 |         col_idx = torch.arange(k_len, device=q.device).unsqueeze(0)
 140 |         causal_mask = col_idx <= row_idx  # (chunk, k_len)
 141 |         attn = attn.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
 142 | 
 143 |         attn = F.softmax(attn, dim=-1)
 144 |         out_chunk = torch.matmul(attn, v_slice)  # (B, H, chunk, D)
 145 |         outputs.append(out_chunk)
 146 | 
 147 |     result = torch.cat(outputs, dim=2)  # (B, H, T_padded, D)
 148 |     if pad > 0:
 149 |         result = result[:, :, :T, :]
 150 |     return result
 151 | 
 152 | def _load_xla_flash_attention():
 153 |     """Try to load XLA flash attention for TPU."""
 154 |     try:
 155 |         from torch_xla.experimental.custom_kernel import flash_attention
 156 |         return flash_attention
 157 |     except ImportError:
 158 |         return None
 159 | 
 160 | def enable_xla_flash_attention():
 161 |     """Enable XLA flash attention for TPU. Call before training."""
 162 |     global _xla_flash_attn, _xla_flash_enabled, _backend_info
 163 |     _xla_flash_attn = _load_xla_flash_attention()
 164 |     if _xla_flash_attn is not None:
 165 |         _xla_flash_enabled = True
 166 |         _backend_info = "xla_flash_pallas"
 167 |         print(f"XLA Flash Attention enabled (Pallas TPU kernels)")
 168 |     else:
 169 |         print("WARNING: XLA flash attention requested but not available, using SDPA")
 170 | 
 171 | def set_spmd_mesh(mesh, tp_degree=1):
 172 |     """Set SPMD mesh for flash attention partition spec.
 173 | 
 174 |     When set, the Pallas flash attention kernel will use SPMD-aware sharding,
 175 |     avoiding costly all-gathers of the batch dimension across TPU chips.
 176 | 
 177 |     With tensor parallelism (tp_degree > 1), the 2D mesh has axes ('data', 'model')
 178 |     and attention heads are sharded across the 'model' axis.
 179 |     """
 180 |     global _spmd_mesh, _spmd_partition_spec
 181 |     _spmd_mesh = mesh
 182 |     # Input format is (B, H, T, D).
 183 |     # Use plain tuple: torch_xla's FA serializes via str() then deserializes
 184 |     # internally, and PartitionSpec isn't in its deserialization namespace.
 185 |     if tp_degree > 1:
 186 |         # 2D mesh: shard batch across 'data', heads across 'model'
 187 |         _spmd_partition_spec = ('data', 'model', None, None)
 188 |     else:
 189 |         # 1D mesh: shard batch across 'data' only
 190 |         _spmd_partition_spec = ('data', None, None, None)
 191 |     print(f"Flash attention SPMD partition: {_spmd_partition_spec}")
 192 | 
 193 | # Print which backend is being used
 194 | def get_backend_info():
 195 |     return _backend_info
 196 | 
 197 | # Override for testing: set to 'fa3', 'sdpa', or None (auto)
 198 | _override_impl = None
 199 | 
 200 | 
 201 | def _use_xla_flash(tensor_device=None):
 202 |     """Check if we should use XLA flash attention."""
 203 |     if not _xla_flash_enabled or _xla_flash_attn is None:
 204 |         return False
 205 |     if tensor_device is not None and not str(tensor_device).startswith("xla"):
 206 |         return False
 207 |     return True
 208 | 
 209 | 
 210 | def _use_fa3(tensor_device=None):
 211 |     """Determine whether to use FA3 based on availability, override, and input device."""
 212 |     if tensor_device is not None and tensor_device.type != "cuda":
 213 |         return False
 214 |     if _override_impl == 'fa3':
 215 |         assert HAS_FA3, f"Cannot override to FA3: not available ({_backend_info})"
 216 |         return True
 217 |     if _override_impl == 'sdpa':
 218 |         return False
 219 |     return HAS_FA3  # auto
 220 | 
 221 | 
 222 | # =============================================================================
 223 | # SDPA helpers
 224 | # =============================================================================
 225 | # Check if enable_gqa is supported (PyTorch 2.5+)
 226 | def _sdpa_supports_gqa():
 227 |     try:
 228 |         # Try calling with enable_gqa to see if it's supported
 229 |         # Use tiny tensors to minimize overhead
 230 |         q = torch.zeros(1, 1, 1, 1, device='cpu')
 231 |         F.scaled_dot_product_attention(q, q, q, enable_gqa=False)
 232 |         return True
 233 |     except TypeError:
 234 |         return False
 235 |     except Exception:
 236 |         # Any other error means we can't tell, assume not supported
 237 |         return False
 238 | 
 239 | _HAS_SDPA_GQA = _sdpa_supports_gqa()
 240 | 
 241 | 
 242 | def _sdpa_attention(q, k, v, window_size, enable_gqa):
 243 |     """
 244 |     SDPA attention with sliding window support.
 245 |     q, k, v are (B, H, T, D) format.
 246 |     """
 247 |     Tq = q.size(2)
 248 |     Tk = k.size(2)
 249 |     window = window_size[0]
 250 | 
 251 |     # Build kwargs dict - enable_gqa only supported in PyTorch 2.5+
 252 |     extra_kwargs = {}
 253 |     if _HAS_SDPA_GQA:
 254 |         extra_kwargs['enable_gqa'] = enable_gqa
 255 | 
 256 |     # Full context, same length
 257 |     if (window < 0 or window >= Tq) and Tq == Tk:
 258 |         return F.scaled_dot_product_attention(q, k, v, is_causal=True, **extra_kwargs)
 259 | 
 260 |     # Single token generation
 261 |     if Tq == 1:
 262 |         if window >= 0 and window < Tk:
 263 |             # window is "left" tokens we need to include (window + 1) keys total
 264 |             start = max(0, Tk - (window + 1))
 265 |             k = k[:, :, start:, :]
 266 |             v = v[:, :, start:, :]
 267 |         return F.scaled_dot_product_attention(q, k, v, is_causal=False, **extra_kwargs)
 268 | 
 269 |     # Need explicit mask for sliding window/chunk inference
 270 |     device = q.device
 271 |     # For chunk inference (Tq != Tk), is_causal is not aligned to cache position => build an explicit bool mask
 272 |     row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
 273 |     col_idx = torch.arange(Tk, device=device).unsqueeze(0)
 274 |     mask = col_idx <= row_idx
 275 | 
 276 |     # sliding window (left)
 277 |     if window >= 0 and window < Tk:
 278 |         mask = mask & ((row_idx - col_idx) <= window)
 279 | 
 280 |     return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, **extra_kwargs)
 281 | 
 282 | 
 283 | # =============================================================================
 284 | # Public API: Same interface as FA3
 285 | # =============================================================================
 286 | def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
 287 |     """
 288 |     Flash Attention for training (no KV cache).
 289 | 
 290 |     Args:
 291 |         q, k, v: Tensors of shape (B, T, H, D)
 292 |         causal: Whether to use causal masking
 293 |         window_size: (left, right) sliding window. -1 means unlimited.
 294 | 
 295 |     Returns:
 296 |         Output tensor of shape (B, T, H, D)
 297 |     """
 298 |     if _use_fa3(q.device):
 299 |         fa3_func, _ = _fa3_funcs
 300 |         return fa3_func(q, k, v, causal=causal, window_size=window_size)
 301 | 
 302 |     # XLA Flash Attention (TPU Pallas kernels) — O(n) memory
 303 |     # Note: does not support sliding window, use window_pattern=L for long context
 304 |     if _use_xla_flash(q.device):
 305 |         # XLA flash expects (B, H, T, D), our input is (B, T, H, D)
 306 |         q_t = q.transpose(1, 2)
 307 |         k_t = k.transpose(1, 2)
 308 |         v_t = v.transpose(1, 2)
 309 |         # Pass SPMD partition_spec so XLA shards batch across chips
 310 |         # instead of all-gathering the full batch onto each chip
 311 |         y = _xla_flash_attn(q_t, k_t, v_t, causal=causal,
 312 |                             partition_spec=_spmd_partition_spec,
 313 |                             mesh=_spmd_mesh)
 314 |         return y.transpose(1, 2)  # back to (B, T, H, D)
 315 | 
 316 |     # On XLA with long sequences, use chunked attention to avoid O(n^2) memory
 317 |     T = q.size(1)
 318 |     if _chunked_attn_enabled and T > _chunked_attn_threshold:
 319 |         q_t = q.transpose(1, 2)  # (B, H, T, D)
 320 |         k_t = k.transpose(1, 2)
 321 |         v_t = v.transpose(1, 2)
 322 |         y = _chunked_attention(q_t, k_t, v_t, chunk_size=_chunked_attn_chunk_size, window=window_size[0])
 323 |         return y.transpose(1, 2)  # back to (B, T, H, D)
 324 | 
 325 |     # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D)
 326 |     q = q.transpose(1, 2)
 327 |     k = k.transpose(1, 2)
 328 |     v = v.transpose(1, 2)
 329 |     enable_gqa = q.size(1) != k.size(1)
 330 |     y = _sdpa_attention(q, k, v, window_size, enable_gqa)
 331 |     return y.transpose(1, 2)  # back to (B, T, H, D)
 332 | 
 333 | 
 334 | def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
 335 |                             causal=False, window_size=(-1, -1)):
 336 |     """
 337 |     Flash Attention with KV cache for inference.
 338 | 
 339 |     FA3 updates k_cache/v_cache in-place. Our SDPA fallback does the same.
 340 | 
 341 |     Args:
 342 |         q: Queries, shape (B, T_new, H, D)
 343 |         k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
 344 |         k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
 345 |         cache_seqlens: Current position in cache, shape (B,) int32
 346 |         causal: Whether to use causal masking
 347 |         window_size: (left, right) sliding window. -1 means unlimited.
 348 | 
 349 |     Returns:
 350 |         Output tensor of shape (B, T_new, H, D)
 351 |     """
 352 |     if _use_fa3(q.device):
 353 |         _, fa3_kvcache = _fa3_funcs
 354 |         return fa3_kvcache(
 355 |             q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
 356 |             causal=causal, window_size=window_size
 357 |         )
 358 | 
 359 |     # SDPA fallback: manually manage KV cache
 360 |     B, T_new, H, D = q.shape
 361 |     pos = cache_seqlens[0].item()  # assume uniform position across batch
 362 | 
 363 |     # Insert new K, V into cache
 364 |     if k is not None and v is not None:
 365 |         k_cache[:, pos:pos + T_new] = k
 366 |         v_cache[:, pos:pos + T_new] = v
 367 | 
 368 |     # Build full K, V from cache up to current position
 369 |     T_total = pos + T_new
 370 |     k_full = k_cache[:, :T_total]
 371 |     v_full = v_cache[:, :T_total]
 372 | 
 373 |     # Transpose for SDPA: (B, T, H, D) -> (B, H, T, D)
 374 |     q_t = q.transpose(1, 2)
 375 |     k_t = k_full.transpose(1, 2)
 376 |     v_t = v_full.transpose(1, 2)
 377 | 
 378 |     enable_gqa = q_t.size(1) != k_t.size(1)
 379 |     y = _sdpa_attention(q_t, k_t, v_t, window_size, enable_gqa)
 380 |     return y.transpose(1, 2)  # back to (B, T, H, D)

```

`nanochat/nanochat/gpt.py`:

```py
   1 | """
   2 | GPT model (rewrite, a lot simpler)
   3 | Notable features:
   4 | - rotary embeddings (and no positional embeddings)
   5 | - QK norm
   6 | - untied weights for token embedding and lm_head
   7 | - relu^2 activation in MLP
   8 | - norm after token embedding
   9 | - no learnable params in rmsnorm
  10 | - no bias in linear layers
  11 | - Group-Query Attention (GQA) support for more efficient inference
  12 | - Flash Attention 3 integration (local build for GB10/SM121)
  13 | - Apple Cut Cross Entropy (CCE) for memory-efficient loss computation
  14 | """
  15 | 
  16 | from functools import partial
  17 | from dataclasses import dataclass
  18 | 
  19 | import torch
  20 | import torch.nn as nn
  21 | import torch.nn.functional as F
  22 | 
  23 | # PyTorch 2.9 checkpoint calls getattr(torch, device_type) which fails for 'xla'
  24 | # because torch_xla doesn't register as torch.xla. Fix: register it explicitly.
  25 | try:
  26 |     import torch_xla
  27 |     if not hasattr(torch, 'xla'):
  28 |         torch.xla = torch_xla
  29 | except ImportError:
  30 |     pass
  31 | 
  32 | from nanochat.common import get_dist_info, print0
  33 | from nanochat.muon import Muon, DistMuon
  34 | from nanochat.adamw import DistAdamW
  35 | 
  36 | # =============================================================================
  37 | # Flash Attention with automatic FA3/SDPA fallback
  38 | # =============================================================================
  39 | # Uses local FA3 build for SM90 (Hopper) and SM121 (GB10/DGX Spark)
  40 | # Falls back to PyTorch SDPA for other GPUs (Ada SM89, Blackwell SM100, etc.)
  41 | from nanochat.flash_attention import flash_attn_func, flash_attn_with_kvcache
  42 | 
  43 | # =============================================================================
  44 | # Kernel backend for loss computation (CCE recommended for best performance)
  45 | # =============================================================================
  46 | from nanochat import kernels
  47 | 
  48 | # =============================================================================
  49 | # Precision plan (BF16 default; TE/NVFP4/FP8 stubs for compatibility)
  50 | # =============================================================================
  51 | from contextlib import nullcontext
  52 | from typing import Any, Optional
  53 | 
  54 | from nanochat.engram import EngramBranch
  55 | from nanochat.mhc import ManifoldBranchMixer
  56 | from nanochat.sparse_attention import DeepSeekSparseAttention
  57 | 
  58 | @dataclass
  59 | class PrecisionPlan:
  60 |     name: str
  61 |     recipe: Optional[Any]
  62 |     use_te: bool
  63 | 
  64 | def select_precision(target: str = "auto", disable_rht: bool = True, disable_sr: bool = True) -> PrecisionPlan:
  65 |     """Select precision plan. Without TE, always returns BF16."""
  66 |     return PrecisionPlan("PyTorch BF16", None, False)
  67 | 
  68 | def make_autocast_ctx(plan: PrecisionPlan, device_type: str = "cuda"):
  69 |     """Create autocast context factory."""
  70 |     if device_type == "cuda":
  71 |         return lambda: torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
  72 |     return nullcontext
  73 | 
  74 | 
  75 | @dataclass
  76 | class GPTConfig:
  77 |     sequence_len: int = 1024
  78 |     vocab_size: int = 50304
  79 |     n_layer: int = 12
  80 |     n_head: int = 6 # number of query heads
  81 |     n_kv_head: int = 6 # number of key/value heads (GQA)
  82 |     n_embd: int = 768
  83 |     # Sliding window attention pattern string, tiled across layers. Final layer always L.
  84 |     # Characters: L=long (full context), S=short (half context)
  85 |     # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
  86 |     window_pattern: str = "L"
  87 |     # Optional Engram branch
  88 |     engram_enabled: bool = False
  89 |     engram_layers: str = ""
  90 |     engram_ngram_orders: str = "2,3,4"
  91 |     engram_bottleneck_dim: int = 0
  92 |     engram_dropout: float = 0.0
  93 |     # Optional mHC branch mixer
  94 |     mhc_enabled: bool = False
  95 |     mhc_num_branches: int = 0
  96 |     mhc_sinkhorn_iters: int = 5
  97 |     mhc_temperature: float = 1.0
  98 |     mhc_epsilon: float = 1e-6
  99 |     mhc_blend_alpha: float = 1.0
 100 |     # Optional Multi-Token Prediction (DeepSeek-V3 style)
 101 |     mtp_enabled: bool = False
 102 |     mtp_lambda: float = 0.3       # MTP loss weight (DeepSeek uses 0.3 early, 0.1 later)
 103 |     # Optional DeepSeek Sparse Attention (DSA)
 104 |     dsa_enabled: bool = False
 105 |     dsa_start_layer: int = 7      # first layer to use sparse attention (0-indexed)
 106 |     dsa_top_k_ratio: float = 0.5  # fraction of tokens to select per query
 107 |     dsa_local_window: int = 128   # local window always included
 108 |     dsa_indexer_heads: int = 16   # number of lightweight indexer heads
 109 |     # Gradient checkpointing (saves memory by recomputing activations during backward)
 110 |     gradient_checkpointing: bool = False
 111 |     dsa_indexer_dim: int = 32     # dimension per indexer head
 112 |     # Reserved for optional auxiliary objectives
 113 |     aux_loss_weight: float = 0.0
 114 | 
 115 | 
 116 | def _parse_csv_ints(value: str) -> list[int]:
 117 |     if not value:
 118 |         return []
 119 |     values = []
 120 |     for raw in value.split(","):
 121 |         raw = raw.strip()
 122 |         if not raw:
 123 |             continue
 124 |         values.append(int(raw))
 125 |     return values
 126 | 
 127 | 
 128 | def _parse_engram_layers(layer_spec: str, n_layer: int) -> set[int]:
 129 |     layers = set()
 130 |     for layer_idx in _parse_csv_ints(layer_spec):
 131 |         # Allow negative indices for convenience (-1 means final layer).
 132 |         if layer_idx < 0:
 133 |             layer_idx += n_layer
 134 |         assert 0 <= layer_idx < n_layer, f"Invalid engram layer index {layer_idx}, expected [0, {n_layer - 1}]"
 135 |         layers.add(layer_idx)
 136 |     return layers
 137 | 
 138 | 
 139 | def norm(x):
 140 |     # Purely functional rmsnorm with no learnable params
 141 |     return F.rms_norm(x, (x.size(-1),))
 142 | 
 143 | 
 144 | def apply_rotary_emb(x, cos, sin):
 145 |     assert x.ndim == 4  # multihead attention
 146 |     d = x.shape[3] // 2
 147 |     x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
 148 |     y1 = x1 * cos + x2 * sin # rotate pairs of dims
 149 |     y2 = x1 * (-sin) + x2 * cos
 150 |     return torch.cat([y1, y2], 3)
 151 | 
 152 | class CausalSelfAttention(nn.Module):
 153 |     def __init__(self, config, layer_idx):
 154 |         super().__init__()
 155 |         self.layer_idx = layer_idx
 156 |         self.n_head = config.n_head
 157 |         self.n_kv_head = config.n_kv_head
 158 |         self.n_embd = config.n_embd
 159 |         self.head_dim = self.n_embd // self.n_head
 160 |         assert self.n_embd % self.n_head == 0
 161 |         assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
 162 |         self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
 163 |         self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
 164 |         self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
 165 |         self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
 166 | 
 167 |     def forward(self, x, cos_sin, window_size, kv_cache):
 168 |         B, T, C = x.size()
 169 | 
 170 |         # Project the input to get queries, keys, and values
 171 |         # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
 172 |         q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
 173 |         k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
 174 |         v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
 175 | 
 176 |         # Apply Rotary Embeddings to queries and keys to get relative positional encoding
 177 |         cos, sin = cos_sin
 178 |         q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
 179 |         q, k = norm(q), norm(k) # QK norm
 180 | 
 181 |         # Attention with Flash Attention 3
 182 |         # FA3 handles GQA automatically when n_kv_heads < n_heads
 183 |         # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
 184 |         if kv_cache is None:
 185 |             # Training: causal attention with optional sliding window
 186 |             y = flash_attn_func(q, k, v, causal=True, window_size=window_size)
 187 |         else:
 188 |             # Inference: use flash_attn_with_kvcache which handles cache management
 189 |             k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
 190 |             y = flash_attn_with_kvcache(
 191 |                 q, k_cache, v_cache,
 192 |                 k=k, v=v,
 193 |                 cache_seqlens=kv_cache.cache_seqlens,
 194 |                 causal=True,
 195 |                 window_size=window_size,
 196 |             )
 197 |             # Advance position after last layer processes
 198 |             if self.layer_idx == kv_cache.n_layers - 1:
 199 |                 kv_cache.advance(T)
 200 | 
 201 |         # Re-assemble the heads and project back to residual stream
 202 |         y = y.contiguous().view(B, T, -1)
 203 |         y = self.c_proj(y)
 204 |         return y
 205 | 
 206 | 
 207 | class MLP(nn.Module):
 208 |     def __init__(self, config):
 209 |         super().__init__()
 210 |         self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
 211 |         self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
 212 | 
 213 |     def forward(self, x):
 214 |         x = self.c_fc(x)
 215 |         x = F.relu(x).square()
 216 |         x = self.c_proj(x)
 217 |         return x
 218 | 
 219 | 
 220 | class Block(nn.Module):
 221 |     def __init__(self, config, layer_idx, engram_layers):
 222 |         super().__init__()
 223 |         # Use DSA for layers >= dsa_start_layer when enabled
 224 |         use_dsa = bool(config.dsa_enabled) and layer_idx >= config.dsa_start_layer
 225 |         if use_dsa:
 226 |             self.attn = DeepSeekSparseAttention(
 227 |                 config, layer_idx,
 228 |                 dsa_top_k_ratio=config.dsa_top_k_ratio,
 229 |                 dsa_local_window=config.dsa_local_window,
 230 |                 dsa_indexer_heads=config.dsa_indexer_heads,
 231 |                 dsa_indexer_dim=config.dsa_indexer_dim,
 232 |             )
 233 |         else:
 234 |             self.attn = CausalSelfAttention(config, layer_idx)
 235 |         self.mlp = MLP(config)
 236 |         self.engram = None
 237 |         self.mhc = None
 238 |         self.use_engram = bool(config.engram_enabled) and layer_idx in engram_layers
 239 |         self.use_mhc = bool(config.mhc_enabled)
 240 |         if self.use_engram:
 241 |             self.engram = EngramBranch(
 242 |                 n_embd=config.n_embd,
 243 |                 ngram_orders=config.engram_ngram_orders,
 244 |                 bottleneck_dim=config.engram_bottleneck_dim,
 245 |                 dropout=config.engram_dropout,
 246 |             )
 247 |         if self.use_mhc:
 248 |             self.mhc = ManifoldBranchMixer(
 249 |                 n_embd=config.n_embd,
 250 |                 sinkhorn_iters=config.mhc_sinkhorn_iters,
 251 |                 temperature=config.mhc_temperature,
 252 |                 epsilon=config.mhc_epsilon,
 253 |                 blend_alpha=config.mhc_blend_alpha,
 254 |                 max_branches=config.mhc_num_branches,
 255 |             )
 256 | 
 257 |     def forward(self, x, cos_sin, window_size, kv_cache):
 258 |         x_attn = x + self.attn(norm(x), cos_sin, window_size, kv_cache)
 259 |         baseline_out = x_attn + self.mlp(norm(x_attn))
 260 | 
 261 |         engram_out = None
 262 |         if self.engram is not None:
 263 |             engram_out = baseline_out + self.engram(norm(x))
 264 | 
 265 |         if self.mhc is None:
 266 |             return engram_out if engram_out is not None else baseline_out
 267 | 
 268 |         branches = [baseline_out, x]
 269 |         if engram_out is not None:
 270 |             branches.append(engram_out)
 271 |         return self.mhc(branches)
 272 | 
 273 | 
 274 | class GPT(nn.Module):
 275 |     def __init__(self, config, pad_vocab_size_to=64):
 276 |         """
 277 |         NOTE a major footgun: this __init__ function runs in meta device context (!!)
 278 |         Therefore, any calculations inside here are shapes and dtypes only, no actual data.
 279 |         => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
 280 |         """
 281 |         super().__init__()
 282 |         self.config = config
 283 |         self.engram_layers = _parse_engram_layers(config.engram_layers, config.n_layer) if config.engram_enabled else set()
 284 |         # Compute per-layer window sizes for sliding window attention
 285 |         # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
 286 |         self.window_sizes = self._compute_window_sizes(config)
 287 |         # Pad vocab size to multiple of 64 for tensor core efficiency (significant speedup for lm_head matmul).
 288 |         # Trade-off: During training, the softmax denominator includes padding logits (initialized to ~0),
 289 |         # adding ~(padding_count) to the denominator. Impact is tiny (<0.01% for 47 extra tokens vs 50K vocab)
 290 |         # and consistent across training. Inference correctly slices to vocab_size.
 291 |         # Default vocab_size=50304 is already aligned, so padding only affects custom unaligned vocab sizes.
 292 |         # Set pad_vocab_size_to=1 to disable padding if exact loss values are critical.
 293 |         padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
 294 |         if padded_vocab_size != config.vocab_size:
 295 |             print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
 296 |         self.padded_vocab_size = padded_vocab_size
 297 |         self.transformer = nn.ModuleDict({
 298 |             "wte": nn.Embedding(padded_vocab_size, config.n_embd),
 299 |             "h": nn.ModuleList([Block(config, layer_idx, self.engram_layers) for layer_idx in range(config.n_layer)]),
 300 |         })
 301 |         self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
 302 |         # Optional MTP head (DeepSeek-V3 Multi-Token Prediction)
 303 |         self.mtp = None
 304 |         if config.mtp_enabled:
 305 |             from nanochat.mtp import MTPModule
 306 |             self.mtp = MTPModule(config)
 307 |             self.mtp_lambda = config.mtp_lambda
 308 |         # Per-layer learnable scalars (inspired by modded-nanogpt)
 309 |         # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
 310 |         # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
 311 |         # Separate parameters so they can have different optimizer treatment
 312 |         self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
 313 |         self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
 314 |         # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
 315 |         # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
 316 |         # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
 317 |         # In the future we can dynamically grow the cache, for now it's fine.
 318 |         self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
 319 |         head_dim = config.n_embd // config.n_head
 320 |         cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
 321 |         self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
 322 |         self.register_buffer("sin", sin, persistent=False)
 323 | 
 324 |     def init_weights(self):
 325 |         """
 326 |         Initialize the full model in this one function for maximum clarity.
 327 | 
 328 |         wte (embedding):     normal, std=1.0
 329 |         lm_head:             normal, std=0.001
 330 |         for each block:
 331 |             attn.c_q:        uniform, std=1/sqrt(n_embd)
 332 |             attn.c_k:        uniform, std=1/sqrt(n_embd)
 333 |             attn.c_v:        uniform, std=1/sqrt(n_embd)
 334 |             attn.c_proj:     zeros
 335 |             mlp.c_fc:        uniform, std=1/sqrt(n_embd)
 336 |             mlp.c_proj:      zeros
 337 |         """
 338 | 
 339 |         # Embedding and unembedding
 340 |         torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
 341 |         torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
 342 | 
 343 |         # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
 344 |         n_embd = self.config.n_embd
 345 |         s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
 346 |         for block in self.transformer.h:
 347 |             torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
 348 |             torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
 349 |             torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
 350 |             torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
 351 |             torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
 352 |             torch.nn.init.zeros_(block.mlp.c_proj.weight)
 353 |             if block.engram is not None:
 354 |                 torch.nn.init.uniform_(block.engram.in_proj.weight, -s, s)
 355 |                 for mix in block.engram.order_mix:
 356 |                     torch.nn.init.uniform_(mix.weight, -s, s)
 357 |                 torch.nn.init.zeros_(block.engram.out_proj.weight)
 358 |             if block.mhc is not None:
 359 |                 torch.nn.init.uniform_(block.mhc.score_proj.weight, -s, s)
 360 |                 torch.nn.init.zeros_(block.mhc.score_out.weight)
 361 |             # DSA: initialize indexer projections
 362 |             if isinstance(block.attn, DeepSeekSparseAttention):
 363 |                 torch.nn.init.uniform_(block.attn.indexer.q_proj.weight, -s, s)
 364 |                 torch.nn.init.uniform_(block.attn.indexer.k_proj.weight, -s, s)
 365 |                 # Small random init for w_proj so importance scores vary (enables gradient flow)
 366 |                 torch.nn.init.uniform_(block.attn.indexer.w_proj.weight, -s * 0.1, s * 0.1)
 367 | 
 368 |         # MTP module initialization
 369 |         if self.mtp is not None:
 370 |             torch.nn.init.zeros_(self.mtp.proj.weight)  # conservative: start near zero
 371 |             blk = self.mtp.block
 372 |             torch.nn.init.uniform_(blk.attn.c_q.weight, -s, s)
 373 |             torch.nn.init.uniform_(blk.attn.c_k.weight, -s, s)
 374 |             torch.nn.init.uniform_(blk.attn.c_v.weight, -s, s)
 375 |             torch.nn.init.zeros_(blk.attn.c_proj.weight)
 376 |             torch.nn.init.uniform_(blk.mlp.c_fc.weight, -s, s)
 377 |             torch.nn.init.zeros_(blk.mlp.c_proj.weight)
 378 | 
 379 |         # Per-layer scalars
 380 |         with torch.no_grad():
 381 |             self.resid_lambdas.fill_(1.0)   # 1.0 => typical residual connections at init
 382 |             self.x0_lambdas.fill_(0.0)      # 0.0 => skip connection to input is disabled at init
 383 | 
 384 |         # Rotary embeddings
 385 |         head_dim = self.config.n_embd // self.config.n_head
 386 |         cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
 387 |         self.cos, self.sin = cos, sin
 388 | 
 389 |         # Cast entire model to bf16 for optimal performance with CCE
 390 |         # This is important for achieving ~20k tok/s on GB10
 391 |         if self.transformer.wte.weight.device.type == "cuda":
 392 |             self.to(dtype=torch.bfloat16)
 393 | 
 394 |     def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
 395 |         # TODO: bump base theta more? e.g. 100K is more common more recently
 396 |         # autodetect the device from model embeddings
 397 |         if device is None:
 398 |             device = self.transformer.wte.weight.device
 399 |         # stride the channels
 400 |         channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
 401 |         inv_freq = 1.0 / (base ** (channel_range / head_dim))
 402 |         # stride the time steps
 403 |         t = torch.arange(seq_len, dtype=torch.float32, device=device)
 404 |         # calculate the rotation frequencies at each (time, channel) pair
 405 |         freqs = torch.outer(t, inv_freq)
 406 |         cos, sin = freqs.cos(), freqs.sin()
 407 |         cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
 408 |         cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
 409 |         return cos, sin
 410 | 
 411 |     def _compute_window_sizes(self, config):
 412 |         """
 413 |         Compute per-layer window sizes for sliding window attention.
 414 | 
 415 |         Returns list of (left, right) tuples for FA3's window_size parameter:
 416 |         - left: how many tokens before current position to attend to (-1 = unlimited)
 417 |         - right: how many tokens after current position to attend to (0 for causal)
 418 | 
 419 |         Pattern string is tiled across layers. Final layer always gets L (full context).
 420 |         Characters: L=long (full context), S=short (half context)
 421 |         """
 422 |         pattern = config.window_pattern.upper()
 423 |         assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
 424 |         # Map characters to window sizes
 425 |         long_window = config.sequence_len
 426 |         short_window = long_window // 2
 427 |         char_to_window = {
 428 |             "L": (long_window, 0),
 429 |             "S": (short_window, 0),
 430 |         }
 431 |         # Tile pattern across layers
 432 |         window_sizes = []
 433 |         for layer_idx in range(config.n_layer):
 434 |             char = pattern[layer_idx % len(pattern)]
 435 |             window_sizes.append(char_to_window[char])
 436 |         # Final layer always gets full context
 437 |         window_sizes[-1] = (long_window, 0)
 438 |         return window_sizes
 439 | 
 440 |     def get_device(self):
 441 |         return self.transformer.wte.weight.device
 442 | 
 443 |     def estimate_flops(self):
 444 |         """
 445 |         Return the estimated FLOPs per token for the model (forward + backward).
 446 |         Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
 447 |         Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
 448 |         On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
 449 |         With sliding windows, effective_seq_len varies per layer (capped by window size).
 450 |         Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
 451 |         This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
 452 |         - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
 453 |         - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
 454 |         """
 455 |         nparams = sum(p.numel() for p in self.parameters())
 456 |         # Exclude non-matmul params: embeddings and per-layer scalars
 457 |         nparams_exclude = self.transformer.wte.weight.numel() + self.resid_lambdas.numel() + self.x0_lambdas.numel()
 458 |         h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
 459 |         # Sum attention FLOPs per layer, accounting for sliding window and DSA
 460 |         attn_flops = 0
 461 |         for i, window_size in enumerate(self.window_sizes):
 462 |             window = window_size[0]  # (left, right) tuple, we use left
 463 |             effective_seq = t if window < 0 else min(window, t)
 464 |             # DSA layers attend to fewer tokens (top_k_ratio fraction)
 465 |             if self.config.dsa_enabled and i >= self.config.dsa_start_layer:
 466 |                 effective_seq = int(effective_seq * self.config.dsa_top_k_ratio)
 467 |             attn_flops += 12 * h * q * effective_seq
 468 |         num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
 469 |         return num_flops_per_token
 470 | 
 471 |     def num_scaling_params(self):
 472 |         """
 473 |         Return all of the parameters, same as Chinchilla paper.
 474 |         Kaplan et al. did not include embedding parameters and said that this led to cleaner scaling laws.
 475 |         But Kaplan et al. also had a bug in their results (as pointed out by Chinchilla).
 476 |         My own experiments in nanochat confirm the Chinchilla approach gives the much cleaner scaling law.
 477 |         Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper <- good).
 478 |         Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper <- bad)
 479 |         """
 480 |         nparams = sum(p.numel() for p in self.parameters())
 481 |         return nparams
 482 | 
 483 |     def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
 484 |         model_dim = self.config.n_embd
 485 |         ddp, rank, local_rank, world_size = get_dist_info()
 486 |         # Separate out all parameters into groups (matrix, embedding, lm_head, resid_lambdas, x0_lambdas, [mtp])
 487 |         matrix_params = list(self.transformer.h.parameters())
 488 |         if self.mtp is not None:
 489 |             matrix_params += list(self.mtp.parameters())
 490 |         # On XLA/TPU, DSA indexer params are unused (DSA falls back to full attention
 491 |         # since mask-based sparse attention is O(T^2) memory). Exclude them from Muon
 492 |         # which crashes on None grads. They'll be excluded from all optimizer groups.
 493 |         device_type = str(next(self.parameters()).device).split(':')[0]
 494 |         dsa_indexer_params = set()
 495 |         if device_type == 'xla' and self.config.dsa_enabled:
 496 |             for block in self.transformer.h:
 497 |                 if hasattr(block.attn, 'indexer'):
 498 |                     for p in block.attn.indexer.parameters():
 499 |                         dsa_indexer_params.add(id(p))
 500 |             n_excluded = len(dsa_indexer_params)
 501 |             matrix_params = [p for p in matrix_params if id(p) not in dsa_indexer_params]
 502 |             print0(f"DSA: excluded {n_excluded} indexer params from optimizer (unused on XLA)")
 503 |         embedding_params = list(self.transformer.wte.parameters())
 504 |         lm_head_params = list(self.lm_head.parameters())
 505 |         resid_params = [self.resid_lambdas]
 506 |         x0_params = [self.x0_lambdas]
 507 |         n_optim = len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(resid_params) + len(x0_params) + len(dsa_indexer_params)
 508 |         assert len(list(self.parameters())) == n_optim, f"Parameter count mismatch: {len(list(self.parameters()))} != {n_optim}"
 509 |         # Create the AdamW optimizer for the embedding, lm_head, and per-layer scalars
 510 |         # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
 511 |         dmodel_lr_scale = (model_dim / 768) ** -0.5
 512 |         print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
 513 |         adam_groups = [
 514 |             dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
 515 |             dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
 516 |             dict(params=resid_params, lr=scalar_lr * 0.01), # these are a lot more sensitive because they accumulate in the residual stream
 517 |             dict(params=x0_params, lr=scalar_lr),
 518 |         ]
 519 |         adamw_kwargs = dict(betas=adam_betas, eps=1e-10, weight_decay=0.0) # NOTE: weight decay is hardcoded to 0.0 for AdamW, only used in Muon
 520 |         # fused=True not supported on XLA/TPU devices
 521 |         use_fused = device_type != 'xla'  # fused only works on CUDA, CPU, MPS
 522 |         AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=use_fused)
 523 |         adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
 524 |         # Create the Muon optimizer for the linear layers
 525 |         muon_kwargs = dict(lr=matrix_lr, momentum=0.95, weight_decay=weight_decay)
 526 |         MuonFactory = DistMuon if ddp else Muon
 527 |         muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
 528 |         # Combine them the two optimizers into one list
 529 |         optimizers = [adamw_optimizer, muon_optimizer]
 530 |         for opt in optimizers:
 531 |             for group in opt.param_groups:
 532 |                 group["initial_lr"] = group["lr"]
 533 |         return optimizers
 534 | 
 535 |     def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
 536 |         B, T = idx.size()
 537 | 
 538 |         # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
 539 |         assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
 540 |         assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
 541 |         assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
 542 |         # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
 543 |         T0 = 0 if kv_cache is None else kv_cache.get_pos()
 544 |         cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length
 545 | 
 546 |         # Forward the trunk of the Transformer
 547 |         x = self.transformer.wte(idx)
 548 |         x = norm(x)
 549 |         x0 = x  # save initial normalized embedding for x0 residual
 550 |         for i, block in enumerate(self.transformer.h):
 551 |             x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
 552 |             if self.training and self.config.gradient_checkpointing:
 553 |                 if idx.device.type == 'xla':
 554 |                     # XLA-specific checkpoint that uses optimization barriers
 555 |                     # to force the compiler to respect checkpoint boundaries.
 556 |                     # PyTorch's checkpoint does NOT save memory on XLA.
 557 |                     from torch_xla.utils.checkpoint import checkpoint as xla_checkpoint
 558 |                     x = xla_checkpoint(block, x, cos_sin, self.window_sizes[i], kv_cache,
 559 |                                        preserve_rng_state=False)
 560 |                 else:
 561 |                     x = torch.utils.checkpoint.checkpoint(
 562 |                         block, x, cos_sin, self.window_sizes[i], kv_cache,
 563 |                         use_reentrant=False,
 564 |                         preserve_rng_state=False,
 565 |                     )
 566 |             else:
 567 |                 x = block(x, cos_sin, self.window_sizes[i], kv_cache)
 568 |         x = norm(x)
 569 | 
 570 |         # Softcap: smoothly cap the logits to the range [-softcap, softcap]
 571 |         softcap = 15
 572 | 
 573 |         if targets is not None:
 574 |             # Training: use fused linear + cross entropy (CCE recommended)
 575 |             # CCE avoids materializing the huge logits tensor (B*T*V), saving ~8GB for large vocabs.
 576 |             # Note: lm_head.weight may have padded_vocab_size rows (see __init__ comment for trade-off).
 577 |             main_loss = kernels.fused_linear_cross_entropy(
 578 |                 x.to(torch.bfloat16),
 579 |                 self.lm_head.weight.to(torch.bfloat16),
 580 |                 targets,
 581 |                 ignore_index=-1,
 582 |                 softcap=softcap,
 583 |                 reduction=loss_reduction,
 584 |             )
 585 |             # Multi-Token Prediction: predict token at position i+2
 586 |             if self.mtp is not None and loss_reduction == 'mean':
 587 |                 mtp_loss = self.mtp(
 588 |                     x, targets, self.transformer.wte,
 589 |                     self.lm_head.weight, cos_sin, softcap=softcap,
 590 |                 )
 591 |                 return main_loss + self.mtp_lambda * mtp_loss
 592 |             return main_loss
 593 |         else:
 594 |             # Inference: compute full logits
 595 |             logits = self.lm_head(x) # (B, T, padded_vocab_size)
 596 |             logits = logits[..., :self.config.vocab_size] # slice to remove padding
 597 |             logits = logits.float() # switch to fp32 for logit softcap
 598 |             logits = softcap * torch.tanh(logits / softcap) # squash the logits
 599 |             return logits
 600 | 
 601 |     @torch.inference_mode()
 602 |     def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
 603 |         """
 604 |         Naive autoregressive streaming inference.
 605 |         To make it super simple, let's assume:
 606 |         - batch size is 1
 607 |         - ids and the yielded tokens are simple Python lists and ints
 608 |         """
 609 |         assert isinstance(tokens, list)
 610 |         device = self.get_device()
 611 |         rng = None
 612 |         if temperature > 0:
 613 |             rng = torch.Generator(device=device)
 614 |             rng.manual_seed(seed)
 615 |         ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
 616 |         for _ in range(max_tokens):
 617 |             logits = self.forward(ids) # (B, T, vocab_size)
 618 |             logits = logits[:, -1, :] # (B, vocab_size)
 619 |             if top_k is not None and top_k > 0:
 620 |                 v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
 621 |                 logits[logits < v[:, [-1]]] = -float('Inf')
 622 |             if temperature > 0:
 623 |                 logits = logits / temperature
 624 |                 probs = F.softmax(logits, dim=-1)
 625 |                 next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
 626 |             else:
 627 |                 next_ids = torch.argmax(logits, dim=-1, keepdim=True)
 628 |             ids = torch.cat((ids, next_ids), dim=1)
 629 |             token = next_ids.item()
 630 |             yield token

```

`nanochat/nanochat/kernels.py`:

```py
   1 | """
   2 | Optimized Triton kernels for nanochat training.
   3 | 
   4 | Provides four kernel backends:
   5 | - current: PyTorch native operations
   6 | - liger: Liger-Kernel optimized Triton kernels
   7 | - cce: Apple Cut Cross Entropy (most memory efficient)
   8 | - triton: Custom Triton kernels (Unsloth-style)
   9 | 
  10 | The biggest wins come from:
  11 | 1. FusedLinearCrossEntropy: Fuses lm_head projection with cross entropy
  12 |    - Avoids materializing huge logits tensor (B*T*V floats)
  13 |    - Can save 50-60% memory and speed up training
  14 |    - CCE saves even more: 28GB -> 1GB on some models!
  15 | 2. Fused RMSNorm: Reduces memory bandwidth
  16 | 3. Optimized RoPE: Better memory access patterns
  17 | """
  18 | 
  19 | import torch
  20 | import torch.nn.functional as F
  21 | from typing import Optional, Literal
  22 | 
  23 | # =============================================================================
  24 | # Kernel backend selection
  25 | # =============================================================================
  26 | 
  27 | KERNEL_BACKEND: Literal["current", "liger", "cce", "triton"] = "current"
  28 | 
  29 | def set_kernel_backend(backend: str):
  30 |     """Set the kernel backend for training."""
  31 |     global KERNEL_BACKEND
  32 |     assert backend in ["current", "liger", "cce", "triton"], f"Unknown backend: {backend}"
  33 |     KERNEL_BACKEND = backend
  34 |     print(f"Kernel backend: {KERNEL_BACKEND}")
  35 | 
  36 | def get_kernel_backend() -> str:
  37 |     return KERNEL_BACKEND
  38 | 
  39 | # =============================================================================
  40 | # Liger-Kernel imports (optional)
  41 | # =============================================================================
  42 | 
  43 | LIGER_AVAILABLE = False
  44 | try:
  45 |     from liger_kernel.ops.rms_norm import LigerRMSNormFunction
  46 |     from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction
  47 |     from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction
  48 |     from liger_kernel.ops.rope import LigerRopeFunction
  49 |     LIGER_AVAILABLE = True
  50 | except ImportError:
  51 |     pass
  52 | 
  53 | # =============================================================================
  54 | # Apple Cut Cross Entropy imports (optional)
  55 | # =============================================================================
  56 | 
  57 | CCE_AVAILABLE = False
  58 | try:
  59 |     from cut_cross_entropy import linear_cross_entropy as cce_linear_cross_entropy
  60 |     CCE_AVAILABLE = True
  61 | except ImportError:
  62 |     pass
  63 | 
  64 | # =============================================================================
  65 | # RMS Norm implementations
  66 | # =============================================================================
  67 | 
  68 | def rms_norm_current(x: torch.Tensor) -> torch.Tensor:
  69 |     """PyTorch native RMS norm (no learnable params)."""
  70 |     return F.rms_norm(x, (x.size(-1),))
  71 | 
  72 | 
  73 | def rms_norm_liger(x: torch.Tensor) -> torch.Tensor:
  74 |     """Liger-Kernel RMS norm.
  75 | 
  76 |     Note: Liger's RMSNorm expects a weight tensor, but nanochat doesn't use one.
  77 |     We pass ones and set in_place=False to avoid modifying input.
  78 |     """
  79 |     if not LIGER_AVAILABLE:
  80 |         return rms_norm_current(x)
  81 | 
  82 |     # Create dummy weight tensor (ones)
  83 |     hidden_size = x.size(-1)
  84 |     dtype = x.dtype
  85 |     device = x.device
  86 | 
  87 |     # Use cached weight if possible
  88 |     if not hasattr(rms_norm_liger, '_weight_cache'):
  89 |         rms_norm_liger._weight_cache = {}
  90 | 
  91 |     cache_key = (hidden_size, dtype, device)
  92 |     if cache_key not in rms_norm_liger._weight_cache:
  93 |         rms_norm_liger._weight_cache[cache_key] = torch.ones(hidden_size, dtype=dtype, device=device)
  94 | 
  95 |     weight = rms_norm_liger._weight_cache[cache_key]
  96 | 
  97 |     return LigerRMSNormFunction.apply(x, weight, 1e-6, 0.0, "llama", False)
  98 | 
  99 | 
 100 | def rms_norm(x: torch.Tensor) -> torch.Tensor:
 101 |     """Dispatch to appropriate RMS norm based on backend."""
 102 |     if KERNEL_BACKEND == "liger" and LIGER_AVAILABLE:
 103 |         return rms_norm_liger(x)
 104 |     return rms_norm_current(x)
 105 | 
 106 | # =============================================================================
 107 | # Cross Entropy implementations
 108 | # =============================================================================
 109 | 
 110 | def cross_entropy_current(
 111 |     logits: torch.Tensor,
 112 |     targets: torch.Tensor,
 113 |     ignore_index: int = -1,
 114 |     reduction: str = 'mean',
 115 |     softcap: Optional[float] = None,
 116 | ) -> torch.Tensor:
 117 |     """PyTorch native cross entropy with optional softcap."""
 118 |     if softcap is not None:
 119 |         logits = softcap * torch.tanh(logits / softcap)
 120 |     return F.cross_entropy(
 121 |         logits.view(-1, logits.size(-1)),
 122 |         targets.view(-1),
 123 |         ignore_index=ignore_index,
 124 |         reduction=reduction
 125 |     )
 126 | 
 127 | 
 128 | def cross_entropy_liger(
 129 |     logits: torch.Tensor,
 130 |     targets: torch.Tensor,
 131 |     ignore_index: int = -1,
 132 |     reduction: str = 'mean',
 133 |     softcap: Optional[float] = None,
 134 | ) -> torch.Tensor:
 135 |     """Liger-Kernel cross entropy."""
 136 |     if not LIGER_AVAILABLE:
 137 |         return cross_entropy_current(logits, targets, ignore_index, reduction, softcap)
 138 | 
 139 |     return LigerCrossEntropyFunction.apply(
 140 |         logits.view(-1, logits.size(-1)),
 141 |         targets.view(-1),
 142 |         None,  # weight
 143 |         ignore_index,
 144 |         0.0,  # lse_square_scale
 145 |         0.0,  # label_smoothing
 146 |         reduction,
 147 |         softcap,
 148 |         False,  # return_z_loss
 149 |         False,  # return_token_accuracy
 150 |     )
 151 | 
 152 | 
 153 | def cross_entropy(
 154 |     logits: torch.Tensor,
 155 |     targets: torch.Tensor,
 156 |     ignore_index: int = -1,
 157 |     reduction: str = 'mean',
 158 |     softcap: Optional[float] = None,
 159 | ) -> torch.Tensor:
 160 |     """Dispatch to appropriate cross entropy based on backend."""
 161 |     if KERNEL_BACKEND == "liger" and LIGER_AVAILABLE:
 162 |         return cross_entropy_liger(logits, targets, ignore_index, reduction, softcap)
 163 |     return cross_entropy_current(logits, targets, ignore_index, reduction, softcap)
 164 | 
 165 | # =============================================================================
 166 | # Fused Linear + Cross Entropy (the big win!)
 167 | # =============================================================================
 168 | 
 169 | def fused_linear_cross_entropy_current(
 170 |     hidden_states: torch.Tensor,
 171 |     lm_head_weight: torch.Tensor,
 172 |     targets: torch.Tensor,
 173 |     ignore_index: int = -1,
 174 |     softcap: Optional[float] = None,
 175 |     reduction: str = 'mean',
 176 | ) -> torch.Tensor:
 177 |     """Standard (non-fused) linear + cross entropy.
 178 | 
 179 |     This materializes the full logits tensor which is huge: (B*T, vocab_size).
 180 |     For B=32, T=2048, V=65536, that's 4.3B floats = 17GB in fp32 or 8.5GB in bf16!
 181 |     """
 182 |     B, T = hidden_states.size(0), hidden_states.size(1) if hidden_states.dim() == 3 else 1
 183 | 
 184 |     # Linear projection: (B*T, hidden) @ (hidden, vocab) -> (B*T, vocab)
 185 |     logits = F.linear(hidden_states, lm_head_weight)
 186 | 
 187 |     # Softcap
 188 |     if softcap is not None:
 189 |         logits = softcap * torch.tanh(logits / softcap)
 190 | 
 191 |     # Cross entropy
 192 |     loss = F.cross_entropy(
 193 |         logits.view(-1, logits.size(-1)),
 194 |         targets.view(-1),
 195 |         ignore_index=ignore_index,
 196 |         reduction=reduction
 197 |     )
 198 | 
 199 |     # Reshape for reduction='none' to match expected (B, T) shape
 200 |     if reduction == 'none' and hidden_states.dim() == 3:
 201 |         loss = loss.view(B, T)
 202 | 
 203 |     return loss
 204 | 
 205 | 
 206 | def fused_linear_cross_entropy_liger(
 207 |     hidden_states: torch.Tensor,
 208 |     lm_head_weight: torch.Tensor,
 209 |     targets: torch.Tensor,
 210 |     ignore_index: int = -1,
 211 |     softcap: Optional[float] = None,
 212 |     reduction: str = 'mean',
 213 | ) -> torch.Tensor:
 214 |     """Liger-Kernel fused linear + cross entropy.
 215 | 
 216 |     This computes the cross entropy loss WITHOUT materializing the full logits tensor.
 217 |     It processes chunks and only keeps the loss, not the logits.
 218 | 
 219 |     Memory savings: ~60% for large vocab sizes!
 220 |     """
 221 |     if not LIGER_AVAILABLE:
 222 |         return fused_linear_cross_entropy_current(
 223 |             hidden_states, lm_head_weight, targets, ignore_index, softcap, reduction
 224 |         )
 225 | 
 226 |     B, T = hidden_states.size(0), hidden_states.size(1) if hidden_states.dim() == 3 else 1
 227 |     B_T = B * T
 228 |     hidden_flat = hidden_states.view(B_T, -1)
 229 |     targets_flat = targets.view(-1)
 230 | 
 231 |     # LigerFusedLinearCrossEntropyFunction.forward signature:
 232 |     # (ctx, _input, weight, target, bias=None, ce_weight=None, ignore_index=-100,
 233 |     #  lse_square_scale=0.0, label_smoothing=0.0, reduction='mean', softcap=None,
 234 |     #  return_z_loss=False, accum_dtype=None, use_token_scaling=False, return_token_accuracy=False)
 235 |     # Returns: (loss, z_loss, token_accuracy) tuple
 236 |     result = LigerFusedLinearCrossEntropyFunction.apply(
 237 |         hidden_flat,       # _input
 238 |         lm_head_weight,    # weight (linear layer weight)
 239 |         targets_flat,      # target
 240 |         None,              # bias
 241 |         None,              # ce_weight (class weights for CE, not linear weight)
 242 |         ignore_index,      # ignore_index
 243 |         0.0,               # lse_square_scale
 244 |         0.0,               # label_smoothing
 245 |         reduction,         # reduction
 246 |         softcap,           # softcap
 247 |         False,             # return_z_loss
 248 |     )
 249 |     # Extract just the loss from the tuple
 250 |     loss = result[0] if isinstance(result, tuple) else result
 251 | 
 252 |     # Reshape for reduction='none' to match expected (B, T) shape
 253 |     if reduction == 'none' and hidden_states.dim() == 3:
 254 |         loss = loss.view(B, T)
 255 | 
 256 |     return loss
 257 | 
 258 | 
 259 | def fused_linear_cross_entropy_cce(
 260 |     hidden_states: torch.Tensor,
 261 |     lm_head_weight: torch.Tensor,
 262 |     targets: torch.Tensor,
 263 |     ignore_index: int = -1,
 264 |     softcap: Optional[float] = None,
 265 |     reduction: str = 'mean',
 266 | ) -> torch.Tensor:
 267 |     """Apple Cut Cross Entropy - most memory efficient.
 268 | 
 269 |     CCE achieves dramatic memory savings by never materializing the full logits:
 270 |     - Forward: 24,000 MB -> 1.1 MB
 271 |     - Forward+Backward: 28,000 MB -> 1,164 MB
 272 | 
 273 |     Reference: https://github.com/apple/ml-cross-entropy
 274 |     Paper: "Cut Your Losses in Large-Vocabulary Language Models" (ICLR 2025)
 275 |     """
 276 |     if not CCE_AVAILABLE:
 277 |         return fused_linear_cross_entropy_current(
 278 |             hidden_states, lm_head_weight, targets, ignore_index, softcap, reduction
 279 |         )
 280 | 
 281 |     B, T = hidden_states.size(0), hidden_states.size(1) if hidden_states.dim() == 3 else 1
 282 |     B_T = B * T
 283 |     hidden_flat = hidden_states.view(B_T, -1)
 284 |     targets_flat = targets.view(-1)
 285 | 
 286 |     # CCE expects: e (embeddings), c (classifier), targets
 287 |     # e @ c.T = logits, but never materialized
 288 |     loss = cce_linear_cross_entropy(
 289 |         e=hidden_flat,
 290 |         c=lm_head_weight,
 291 |         targets=targets_flat,
 292 |         ignore_index=ignore_index,
 293 |         softcap=softcap,
 294 |         reduction=reduction,
 295 |     )
 296 | 
 297 |     # Reshape for reduction='none' to match expected (B, T) shape
 298 |     if reduction == 'none' and hidden_states.dim() == 3:
 299 |         loss = loss.view(B, T)
 300 | 
 301 |     return loss
 302 | 
 303 | 
 304 | def fused_linear_cross_entropy_chunked(
 305 |     hidden_states: torch.Tensor,
 306 |     lm_head_weight: torch.Tensor,
 307 |     targets: torch.Tensor,
 308 |     ignore_index: int = -1,
 309 |     softcap: Optional[float] = None,
 310 |     reduction: str = 'mean',
 311 |     chunk_size: int = 4096,
 312 | ) -> torch.Tensor:
 313 |     """Chunked linear + cross entropy that avoids materializing full logits.
 314 | 
 315 |     Instead of computing all (T, V) logits at once (4GB+ for 64K tokens),
 316 |     processes tokens in chunks of `chunk_size`. Peak logits memory is
 317 |     (chunk_size, V) instead of (T, V).
 318 | 
 319 |     Works on any device (CPU, CUDA, XLA/TPU).
 320 |     """
 321 |     if hidden_states.dim() == 3:
 322 |         B, T, D = hidden_states.shape
 323 |         h = hidden_states.reshape(B * T, D)
 324 |         t = targets.reshape(B * T)
 325 |     else:
 326 |         h = hidden_states
 327 |         t = targets.reshape(-1)
 328 |         B, T = 1, h.size(0)
 329 | 
 330 |     total_tokens = h.size(0)
 331 | 
 332 |     if reduction == 'none':
 333 |         # Collect per-token losses for eval (bpb computation)
 334 |         losses = []
 335 |         for start in range(0, total_tokens, chunk_size):
 336 |             end = min(start + chunk_size, total_tokens)
 337 |             logits_chunk = F.linear(h[start:end], lm_head_weight)
 338 |             if softcap is not None:
 339 |                 logits_chunk = softcap * torch.tanh(logits_chunk / softcap)
 340 |             chunk_loss = F.cross_entropy(
 341 |                 logits_chunk, t[start:end], ignore_index=ignore_index, reduction='none'
 342 |             )
 343 |             losses.append(chunk_loss)
 344 |         loss = torch.cat(losses, dim=0)
 345 |         if hidden_states.dim() == 3:
 346 |             loss = loss.view(B, T)
 347 |         return loss
 348 | 
 349 |     # reduction == 'mean': accumulate sum and count
 350 |     total_loss = torch.tensor(0.0, device=h.device, dtype=torch.float32)
 351 |     n_valid = 0
 352 |     for start in range(0, total_tokens, chunk_size):
 353 |         end = min(start + chunk_size, total_tokens)
 354 |         logits_chunk = F.linear(h[start:end], lm_head_weight)
 355 |         if softcap is not None:
 356 |             logits_chunk = softcap * torch.tanh(logits_chunk / softcap)
 357 |         chunk_loss = F.cross_entropy(
 358 |             logits_chunk, t[start:end], ignore_index=ignore_index, reduction='sum'
 359 |         )
 360 |         total_loss = total_loss + chunk_loss
 361 |         n_valid += (t[start:end] != ignore_index).sum()
 362 | 
 363 |     return total_loss / n_valid.clamp(min=1)
 364 | 
 365 | 
 366 | def fused_linear_cross_entropy(
 367 |     hidden_states: torch.Tensor,
 368 |     lm_head_weight: torch.Tensor,
 369 |     targets: torch.Tensor,
 370 |     ignore_index: int = -1,
 371 |     softcap: Optional[float] = None,
 372 |     reduction: str = 'mean',
 373 | ) -> torch.Tensor:
 374 |     """Dispatch to appropriate fused linear + cross entropy based on backend."""
 375 |     if KERNEL_BACKEND == "cce" and CCE_AVAILABLE:
 376 |         return fused_linear_cross_entropy_cce(
 377 |             hidden_states, lm_head_weight, targets, ignore_index, softcap, reduction
 378 |         )
 379 |     if KERNEL_BACKEND in ["liger", "triton"] and LIGER_AVAILABLE:
 380 |         return fused_linear_cross_entropy_liger(
 381 |             hidden_states, lm_head_weight, targets, ignore_index, softcap, reduction
 382 |         )
 383 |     # On XLA/TPU, use chunked to avoid materializing huge logits tensor.
 384 |     # chunk_size=4096 is optimal: fewer XLA graph nodes than smaller chunks.
 385 |     if hidden_states.device.type == 'xla':
 386 |         return fused_linear_cross_entropy_chunked(
 387 |             hidden_states, lm_head_weight, targets, ignore_index, softcap, reduction,
 388 |         )
 389 |     return fused_linear_cross_entropy_current(
 390 |         hidden_states, lm_head_weight, targets, ignore_index, softcap, reduction
 391 |     )
 392 | 
 393 | # =============================================================================
 394 | # Rotary Position Embeddings
 395 | # =============================================================================
 396 | 
 397 | def apply_rotary_emb_current(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
 398 |     """PyTorch native rotary embeddings."""
 399 |     assert x.ndim == 4  # (B, T, H, D)
 400 |     d = x.shape[3] // 2
 401 |     x1, x2 = x[..., :d], x[..., d:]
 402 |     y1 = x1 * cos + x2 * sin
 403 |     y2 = x1 * (-sin) + x2 * cos
 404 |     return torch.cat([y1, y2], 3)
 405 | 
 406 | 
 407 | def apply_rotary_emb_liger(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
 408 |     """Liger-Kernel rotary embeddings.
 409 | 
 410 |     Note: Liger expects different tensor shapes, so we need to adapt.
 411 |     """
 412 |     if not LIGER_AVAILABLE:
 413 |         return apply_rotary_emb_current(x, cos, sin)
 414 | 
 415 |     # Liger expects (B, H, T, D) but nanochat uses (B, T, H, D)
 416 |     # Also Liger's rope expects cos/sin of shape (1, T, 1, D/2) or similar
 417 |     # For now, fall back to current implementation as Liger's rope has different assumptions
 418 |     return apply_rotary_emb_current(x, cos, sin)
 419 | 
 420 | 
 421 | def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
 422 |     """Dispatch to appropriate rotary embeddings based on backend."""
 423 |     if KERNEL_BACKEND == "liger" and LIGER_AVAILABLE:
 424 |         return apply_rotary_emb_liger(x, cos, sin)
 425 |     return apply_rotary_emb_current(x, cos, sin)
 426 | 
 427 | # =============================================================================
 428 | # Info
 429 | # =============================================================================
 430 | 
 431 | def print_kernel_info():
 432 |     """Print information about available kernels."""
 433 |     print(f"Kernel backend: {KERNEL_BACKEND}")
 434 |     print(f"Liger-Kernel available: {LIGER_AVAILABLE}")
 435 |     print(f"CCE available: {CCE_AVAILABLE}")
 436 |     if LIGER_AVAILABLE:
 437 |         import liger_kernel
 438 |         print(f"Liger-Kernel version: {liger_kernel.__version__ if hasattr(liger_kernel, '__version__') else 'unknown'}")
 439 |     if CCE_AVAILABLE:
 440 |         import cut_cross_entropy
 441 |         print(f"CCE version: {cut_cross_entropy.__version__ if hasattr(cut_cross_entropy, '__version__') else 'unknown'}")

```

`nanochat/nanochat/meta_init.py`:

```py
   1 | """
   2 | Meta-device model initialization for memory-efficient model creation.
   3 | 
   4 | Uses PyTorch's meta device to avoid double memory allocation when creating
   5 | large models. Instead of allocating on CPU and then moving to GPU (2x memory),
   6 | this allocates tensor metadata only (no backing storage) and then materializes
   7 | directly on the target device.
   8 | 
   9 | Usage:
  10 |     from nanochat.meta_init import create_model_on_device
  11 |     from nanochat.gpt import GPT, GPTConfig
  12 | 
  13 |     config = GPTConfig(...)
  14 |     model = create_model_on_device(GPT, config, device)
  15 | 
  16 | This is the modern PyTorch idiom (torch.device("meta") + to_empty + init_weights)
  17 | and is especially useful for large models where double allocation would exceed
  18 | available memory.
  19 | """
  20 | 
  21 | import torch
  22 | 
  23 | 
  24 | def create_model_on_device(model_cls, config, device):
  25 |     """Create a model using the meta device pattern to avoid double memory allocation.
  26 | 
  27 |     1. Instantiate on the meta device (no actual tensor storage allocated)
  28 |     2. Materialize empty tensors directly on the target device
  29 |     3. Initialize weights in-place
  30 | 
  31 |     Args:
  32 |         model_cls: The model class (e.g. GPT) - must have an init_weights() method.
  33 |         config: The model config to pass to the constructor.
  34 |         device: Target device (e.g. torch.device("cuda:0")).
  35 | 
  36 |     Returns:
  37 |         Initialized model on the target device.
  38 |     """
  39 |     with torch.device("meta"):
  40 |         model = model_cls(config)
  41 |     # Materialize on target device without intermediate CPU allocation
  42 |     model = model.to_empty(device=device)
  43 |     model.init_weights()
  44 |     return model

```

`nanochat/nanochat/mhc.py`:

```py
   1 | import torch
   2 | import torch.nn as nn
   3 | 
   4 | from nanochat.common import print0
   5 | 
   6 | 
   7 | class ManifoldBranchMixer(nn.Module):
   8 |     """
   9 |     Constrained branch mixer with Sinkhorn-normalized weights.
  10 | 
  11 |     Input: list[(B, T, C)]
  12 |     Output: (B, T, C)
  13 |     """
  14 | 
  15 |     def __init__(
  16 |         self,
  17 |         n_embd,
  18 |         sinkhorn_iters=5,
  19 |         temperature=1.0,
  20 |         epsilon=1e-6,
  21 |         blend_alpha=1.0,
  22 |         max_branches=0,
  23 |     ):
  24 |         super().__init__()
  25 |         hidden = max(8, min(256, n_embd // 4))
  26 |         self.score_proj = nn.Linear(n_embd, hidden, bias=False)
  27 |         self.score_out = nn.Linear(hidden, 1, bias=False)
  28 |         self.sinkhorn_iters = int(sinkhorn_iters)
  29 |         self.temperature = float(temperature)
  30 |         self.epsilon = float(epsilon)
  31 |         self.blend_alpha = float(blend_alpha)
  32 |         self.max_branches = int(max_branches)
  33 |         self._warned_fallback = False
  34 | 
  35 |     def _sinkhorn(self, raw_matrix):
  36 |         eps = self.epsilon
  37 |         m = raw_matrix - raw_matrix.amax(dim=(-2, -1), keepdim=True)
  38 |         transport = torch.exp(m).clamp_min(eps)
  39 |         for _ in range(max(0, self.sinkhorn_iters)):
  40 |             transport = transport / (transport.sum(dim=-1, keepdim=True) + eps)
  41 |             transport = transport / (transport.sum(dim=-2, keepdim=True) + eps)
  42 |         transport = transport / (transport.sum(dim=-1, keepdim=True) + eps)
  43 |         return transport
  44 | 
  45 |     def forward(self, branches):
  46 |         if not branches:
  47 |             raise ValueError("ManifoldBranchMixer requires at least one branch")
  48 |         if len(branches) == 1:
  49 |             return branches[0]
  50 |         if self.max_branches > 0 and len(branches) > self.max_branches:
  51 |             raise ValueError(f"Too many branches: got {len(branches)}, max_branches={self.max_branches}")
  52 | 
  53 |         ref_shape = branches[0].shape
  54 |         for branch in branches[1:]:
  55 |             if branch.shape != ref_shape:
  56 |                 raise ValueError(f"Branch shape mismatch: expected {ref_shape}, got {branch.shape}")
  57 | 
  58 |         # (B, T, N, C)
  59 |         stacked = torch.stack(branches, dim=2)
  60 |         pooled = stacked.mean(dim=1)  # (B, N, C)
  61 |         logits = self.score_out(torch.tanh(self.score_proj(pooled))).squeeze(-1)  # (B, N)
  62 |         temperature = max(self.temperature, self.epsilon)
  63 |         raw_matrix = (logits.unsqueeze(-1) + logits.unsqueeze(-2)) / temperature
  64 |         transport = self._sinkhorn(raw_matrix.float())
  65 | 
  66 |         # Compute weights from transport matrix, using torch.where for NaN/Inf
  67 |         # fallback instead of data-dependent Python `if` (which forces host-device
  68 |         # sync on XLA/TPU, causing 37x slowdown).
  69 |         weights = transport.mean(dim=1)
  70 |         weights = weights / (weights.sum(dim=-1, keepdim=True) + self.epsilon)
  71 |         n_branches = stacked.size(2)
  72 |         uniform = torch.full_like(weights, 1.0 / n_branches)
  73 |         is_valid = torch.isfinite(weights).all(dim=-1, keepdim=True)
  74 |         weights = torch.where(is_valid, weights, uniform).to(dtype=stacked.dtype)
  75 | 
  76 |         mixed = (stacked * weights[:, None, :, None]).sum(dim=2)
  77 |         alpha = min(max(self.blend_alpha, 0.0), 1.0)
  78 |         if alpha == 1.0:
  79 |             return mixed
  80 |         return branches[0] + alpha * (mixed - branches[0])

```

`nanochat/nanochat/mtp.py`:

```py
   1 | """
   2 | Multi-Token Prediction (MTP) module, following DeepSeek-V3 design.
   3 | 
   4 | At each depth k, the MTP module:
   5 | 1. Concatenates RMSNorm'd hidden state with RMSNorm'd embedding of the next token
   6 | 2. Projects from 2*n_embd back to n_embd
   7 | 3. Passes through a dedicated transformer block
   8 | 4. Computes cross-entropy loss for predicting the token 2 positions ahead
   9 | 
  10 | During training, MTP loss is added to the main next-token loss:
  11 |     total_loss = main_loss + mtp_lambda * mtp_loss
  12 | 
  13 | At inference, MTP modules are ignored (main model works independently).
  14 | 
  15 | Reference: DeepSeek-V3 Technical Report (arXiv:2412.19437)
  16 | """
  17 | 
  18 | import torch
  19 | import torch.nn as nn
  20 | import torch.nn.functional as F
  21 | 
  22 | from nanochat import kernels
  23 | 
  24 | 
  25 | class MTPModule(nn.Module):
  26 |     """Single-depth Multi-Token Prediction head (D=1).
  27 | 
  28 |     Predicts token at position i+2 given:
  29 |     - hidden_states[i]: output of the main transformer at position i
  30 |     - next_token_ids[i]: ground-truth token at position i+1 (= main model target)
  31 |     """
  32 | 
  33 |     def __init__(self, config):
  34 |         super().__init__()
  35 |         n_embd = config.n_embd
  36 |         # Projection: [RMSNorm(hidden); RMSNorm(emb)] -> n_embd
  37 |         self.proj = nn.Linear(2 * n_embd, n_embd, bias=False)
  38 | 
  39 |         # Dedicated transformer block (imports Block locally to avoid circular imports)
  40 |         from nanochat.gpt import Block
  41 |         # MTP block: no engram, no mhc, no DSA -- just plain attn+mlp
  42 |         # We create a minimal config copy to ensure the block is plain
  43 |         from dataclasses import replace
  44 |         plain_config = replace(
  45 |             config,
  46 |             engram_enabled=False,
  47 |             mhc_enabled=False,
  48 |             dsa_enabled=False,
  49 |         )
  50 |         self.block = Block(plain_config, layer_idx=0, engram_layers=set())
  51 | 
  52 |     def forward(self, hidden_states, next_token_ids, wte, lm_head_weight,
  53 |                 cos_sin, softcap=15):
  54 |         """
  55 |         Args:
  56 |             hidden_states: (B, T, C) - final hidden states from main model (before lm_head)
  57 |             next_token_ids: (B, T) - ground-truth token at position i+1
  58 |             wte: nn.Embedding - shared token embedding from main model
  59 |             lm_head_weight: (V, C) - shared lm_head weight from main model
  60 |             cos_sin: tuple of (cos, sin) rotary embeddings
  61 |             softcap: logit softcap value
  62 | 
  63 |         Returns:
  64 |             mtp_loss: scalar cross-entropy loss for predicting position i+2
  65 |         """
  66 |         B, T, C = hidden_states.shape
  67 | 
  68 |         # We can only predict up to position T-1 (need ground truth at i+2)
  69 |         # hidden_states[:, :-1] -> predict token at positions 2..T
  70 |         T_mtp = T - 1
  71 | 
  72 |         # Pallas flash attention requires seq_len divisible by 1024 (block size).
  73 |         # T-1 may not satisfy this (e.g. 65536-1=65535). Truncate to nearest
  74 |         # multiple of 1024 so FA backward pass works. This drops at most 1023
  75 |         # tokens from the end of the MTP sequence, which is negligible.
  76 |         fa_block = 1024
  77 |         if T_mtp % fa_block != 0:
  78 |             T_mtp = (T_mtp // fa_block) * fa_block
  79 | 
  80 |         h = hidden_states[:, :T_mtp]  # (B, T_mtp, C)
  81 |         next_emb = wte(next_token_ids[:, :T_mtp])  # (B, T_mtp, C)
  82 | 
  83 |         # RMSNorm both inputs independently, then concatenate and project
  84 |         h_norm = F.rms_norm(h, (C,))
  85 |         e_norm = F.rms_norm(next_emb, (C,))
  86 |         combined = torch.cat([h_norm, e_norm], dim=-1)  # (B, T_mtp, 2C)
  87 |         h_mtp = self.proj(combined)  # (B, T_mtp, C)
  88 | 
  89 |         # Truncate cos/sin to match shortened sequence
  90 |         cos, sin = cos_sin
  91 |         cos_short = cos[:, :T_mtp]
  92 |         sin_short = sin[:, :T_mtp]
  93 | 
  94 |         # Pass through dedicated transformer block
  95 |         h_mtp = self.block(h_mtp, (cos_short, sin_short), window_size=(-1, 0), kv_cache=None)
  96 |         h_mtp = F.rms_norm(h_mtp, (C,))
  97 | 
  98 |         # MTP targets: tokens at positions 1..T_mtp = next_token_ids[:, 1:T_mtp+1]
  99 |         mtp_targets = next_token_ids[:, 1:T_mtp + 1].contiguous()  # (B, T_mtp)
 100 | 
 101 |         # Compute loss using shared lm_head
 102 |         mtp_loss = kernels.fused_linear_cross_entropy(
 103 |             h_mtp.to(torch.bfloat16),
 104 |             lm_head_weight.to(torch.bfloat16),
 105 |             mtp_targets,
 106 |             ignore_index=-1,
 107 |             softcap=softcap,
 108 |         )
 109 |         return mtp_loss

```

`nanochat/nanochat/sparse_attention.py`:

```py
   1 | """
   2 | DeepSeek Sparse Attention (DSA) with Lightning Indexer.
   3 | 
   4 | Based on the DSA mechanism from DeepSeek-V3.2 (arXiv:2512.02556):
   5 | 1. A lightweight "lightning indexer" scores importance of each key for each query
   6 | 2. Top-k keys are selected per query position
   7 | 3. Standard attention is computed only over selected keys + local window
   8 | 
   9 | In this implementation we use mask-based sparse attention via PyTorch SDPA
  10 | (since flash attention doesn't support arbitrary masks). For short sequences
  11 | or inference with KV cache, we fall back to full flash attention.
  12 | 
  13 | The indexer uses multi-head ReLU-gated scoring in low dimensions (d_I=32)
  14 | following the DSA paper. A local sliding window is always included to ensure
  15 | nearby context is never dropped.
  16 | 
  17 | Reference: DeepSeek-V3.2 Technical Report (arXiv:2512.02556)
  18 | Reference: NSA Paper (arXiv:2502.11089)
  19 | """
  20 | 
  21 | import torch
  22 | import torch.nn as nn
  23 | import torch.nn.functional as F
  24 | 
  25 | from nanochat.flash_attention import flash_attn_func, flash_attn_with_kvcache
  26 | 
  27 | 
  28 | def _norm(x):
  29 |     return F.rms_norm(x, (x.size(-1),))
  30 | 
  31 | 
  32 | def _apply_rotary_emb(x, cos, sin):
  33 |     d = x.shape[3] // 2
  34 |     x1, x2 = x[..., :d], x[..., d:]
  35 |     y1 = x1 * cos + x2 * sin
  36 |     y2 = x1 * (-sin) + x2 * cos
  37 |     return torch.cat([y1, y2], 3)
  38 | 
  39 | 
  40 | class LightningIndexer(nn.Module):
  41 |     """Lightweight multi-head importance scorer (DeepSeek-V3.2 style).
  42 | 
  43 |     Computes: I(t,s) = sum_j w(t,j) * ReLU(q_I(t,j) . k_I(s))
  44 |     where j indexes indexer heads, using low-dimensional projections.
  45 |     """
  46 | 
  47 |     def __init__(self, n_embd, n_indexer_heads=16, indexer_dim=32):
  48 |         super().__init__()
  49 |         self.n_indexer_heads = n_indexer_heads
  50 |         self.indexer_dim = indexer_dim
  51 |         # Low-dimensional Q projection: per indexer head
  52 |         self.q_proj = nn.Linear(n_embd, n_indexer_heads * indexer_dim, bias=False)
  53 |         # Shared K projection across indexer heads
  54 |         self.k_proj = nn.Linear(n_embd, indexer_dim, bias=False)
  55 |         # Per-head weight derived from query token
  56 |         self.w_proj = nn.Linear(n_embd, n_indexer_heads, bias=False)
  57 | 
  58 |     def forward(self, x):
  59 |         """Compute importance scores for all (query, key) pairs.
  60 | 
  61 |         Args:
  62 |             x: (B, T, C) input hidden states
  63 | 
  64 |         Returns:
  65 |             importance: (B, T, T) where importance[b, t, s] = score of key s for query t
  66 |         """
  67 |         B, T, C = x.shape
  68 |         H_I = self.n_indexer_heads
  69 |         D_I = self.indexer_dim
  70 | 
  71 |         q = self.q_proj(x).view(B, T, H_I, D_I)  # (B, T, H_I, D_I)
  72 |         k = self.k_proj(x)  # (B, T, D_I) -- shared across heads
  73 |         w = self.w_proj(x)  # (B, T, H_I) -- per-head weights
  74 | 
  75 |         # Per-head scores: (B, T_q, H_I, D_I) x (B, T_k, D_I)^T -> (B, T_q, H_I, T_k)
  76 |         scores = torch.einsum('bqhd,bkd->bqhk', q, k)
  77 |         scores = F.relu(scores)  # ReLU gating (key DSA design choice)
  78 | 
  79 |         # Weighted sum across indexer heads: (B, T, H_I) * (B, T_q, H_I, T_k) -> (B, T_q, T_k)
  80 |         importance = torch.einsum('bqh,bqhk->bqk', w, scores)
  81 | 
  82 |         return importance
  83 | 
  84 | 
  85 | class DeepSeekSparseAttention(nn.Module):
  86 |     """Sparse attention with lightning indexer for token selection.
  87 | 
  88 |     Replaces CausalSelfAttention on designated layers (typically layer 7+).
  89 |     Uses mask-based SDPA for sparse attention during training.
  90 |     Falls back to full flash attention for short sequences and inference.
  91 |     """
  92 | 
  93 |     def __init__(self, config, layer_idx, dsa_top_k_ratio=0.5,
  94 |                  dsa_local_window=128, dsa_indexer_heads=16, dsa_indexer_dim=32):
  95 |         super().__init__()
  96 |         self.layer_idx = layer_idx
  97 |         self.n_head = config.n_head
  98 |         self.n_kv_head = config.n_kv_head
  99 |         self.n_embd = config.n_embd
 100 |         self.head_dim = self.n_embd // self.n_head
 101 |         self.top_k_ratio = dsa_top_k_ratio
 102 |         self.local_window = dsa_local_window
 103 | 
 104 |         assert self.n_embd % self.n_head == 0
 105 |         assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
 106 | 
 107 |         # Standard Q/K/V projections (same as CausalSelfAttention)
 108 |         self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
 109 |         self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
 110 |         self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
 111 |         self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
 112 | 
 113 |         # Lightning indexer
 114 |         self.indexer = LightningIndexer(
 115 |             self.n_embd,
 116 |             n_indexer_heads=dsa_indexer_heads,
 117 |             indexer_dim=dsa_indexer_dim,
 118 |         )
 119 | 
 120 |     def _full_attention(self, x, cos_sin, window_size, kv_cache):
 121 |         """Standard full attention (fallback for short sequences and inference)."""
 122 |         B, T, C = x.size()
 123 |         q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
 124 |         k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
 125 |         v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
 126 | 
 127 |         cos, sin = cos_sin
 128 |         q, k = _apply_rotary_emb(q, cos, sin), _apply_rotary_emb(k, cos, sin)
 129 |         q, k = _norm(q), _norm(k)
 130 | 
 131 |         if kv_cache is None:
 132 |             y = flash_attn_func(q, k, v, causal=True, window_size=window_size)
 133 |         else:
 134 |             k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
 135 |             y = flash_attn_with_kvcache(
 136 |                 q, k_cache, v_cache,
 137 |                 k=k, v=v,
 138 |                 cache_seqlens=kv_cache.cache_seqlens,
 139 |                 causal=True,
 140 |                 window_size=window_size,
 141 |             )
 142 |             if self.layer_idx == kv_cache.n_layers - 1:
 143 |                 kv_cache.advance(T)
 144 | 
 145 |         y = y.contiguous().view(B, T, -1)
 146 |         y = self.c_proj(y)
 147 |         return y
 148 | 
 149 |     def forward(self, x, cos_sin, window_size, kv_cache):
 150 |         B, T, C = x.size()
 151 | 
 152 |         # Inference or KV cache: always use full attention
 153 |         if kv_cache is not None:
 154 |             return self._full_attention(x, cos_sin, window_size, kv_cache)
 155 | 
 156 |         # Compute top_k from ratio
 157 |         top_k = max(int(T * self.top_k_ratio), self.local_window)
 158 | 
 159 |         # Short sequences: full attention is cheaper than sparse overhead
 160 |         if T <= top_k + 32:
 161 |             return self._full_attention(x, cos_sin, window_size, kv_cache)
 162 | 
 163 |         # Long sequences (>4096): mask-based SDPA would allocate O(T^2) memory
 164 |         # which is prohibitive (e.g., T=65536 => 8GB per mask). Fall back to
 165 |         # flash attention with sliding window. On XLA/TPU, always use flash attention
 166 |         # since Pallas FA already has O(n) memory. The indexer params still train
 167 |         # via their own gradient path and can be activated with custom kernels on GPU.
 168 |         if T > 4096 or x.device.type == 'xla':
 169 |             return self._full_attention(x, cos_sin, window_size, kv_cache)
 170 | 
 171 |         # --- Sparse attention path ---
 172 | 
 173 |         # 1. Compute Q, K, V with RoPE and QK-norm
 174 |         q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
 175 |         k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
 176 |         v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
 177 | 
 178 |         cos, sin = cos_sin
 179 |         q, k = _apply_rotary_emb(q, cos, sin), _apply_rotary_emb(k, cos, sin)
 180 |         q, k = _norm(q), _norm(k)
 181 | 
 182 |         # 2. Compute importance scores via lightning indexer
 183 |         importance = self.indexer(x)  # (B, T, T)
 184 | 
 185 |         # 3. Build causal mask and apply to importance scores
 186 |         causal_mask = torch.ones(T, T, device=x.device, dtype=torch.bool).triu(diagonal=1)
 187 |         importance = importance.masked_fill(causal_mask.unsqueeze(0), -1e9)
 188 | 
 189 |         # 4. Boost local window tokens so they're always selected
 190 |         positions = torch.arange(T, device=x.device)
 191 |         dist = positions.unsqueeze(0) - positions.unsqueeze(1)  # (T, T), dist[i,j] = i-j
 192 |         in_local = (dist >= 0) & (dist < self.local_window)  # (T, T)
 193 |         importance = importance.masked_fill(in_local.unsqueeze(0), 1e9)
 194 | 
 195 |         # 5. Select top-k tokens per query position
 196 |         actual_k = min(top_k, T)
 197 |         _, top_indices = importance.topk(actual_k, dim=-1)  # (B, T, top_k)
 198 | 
 199 |         # 6. Build sparse mask with Straight-Through Estimator (STE)
 200 |         # This ensures gradients flow back through the indexer during training.
 201 |         # Forward: hard {0,1} mask; Backward: gradients via sigmoid(importance)
 202 |         sparse_mask_hard = torch.zeros(B, T, T, device=x.device, dtype=q.dtype)
 203 |         sparse_mask_hard.scatter_(2, top_indices, 1.0)
 204 |         # Enforce causality and self-attention
 205 |         causal_float = causal_mask.unsqueeze(0).to(q.dtype)
 206 |         sparse_mask_hard = sparse_mask_hard * (1.0 - causal_float)
 207 |         diag = torch.arange(T, device=x.device)
 208 |         sparse_mask_hard[:, diag, diag] = 1.0
 209 | 
 210 |         # STE: soft path for gradients, hard path for forward
 211 |         # Temperature controls gradient sharpness (lower = more focused gradients)
 212 |         soft_scores = torch.sigmoid(importance * 0.1)
 213 |         soft_scores = soft_scores * (1.0 - causal_float)
 214 |         soft_scores[:, diag, diag] = 1.0
 215 |         # STE trick: forward uses hard mask, backward uses soft scores
 216 |         gate = soft_scores + (sparse_mask_hard - soft_scores).detach()
 217 | 
 218 |         # 7. Convert gate to attention bias for SDPA
 219 |         # Where gate ≈ 0, set bias to -inf; where gate ≈ 1, set bias to 0
 220 |         attn_bias = torch.log(gate.clamp(min=1e-6)).unsqueeze(1)  # (B, 1, T, T)
 221 | 
 222 |         # 8. Run SDPA with sparse mask
 223 |         # SDPA expects (B, H, T, D) layout
 224 |         q_sdpa = q.transpose(1, 2)  # (B, H, T, D)
 225 | 
 226 |         # Handle GQA: repeat K,V heads to match Q heads
 227 |         if self.n_kv_head < self.n_head:
 228 |             repeat_factor = self.n_head // self.n_kv_head
 229 |             k_sdpa = k.transpose(1, 2).repeat_interleave(repeat_factor, dim=1)
 230 |             v_sdpa = v.transpose(1, 2).repeat_interleave(repeat_factor, dim=1)
 231 |         else:
 232 |             k_sdpa = k.transpose(1, 2)
 233 |             v_sdpa = v.transpose(1, 2)
 234 | 
 235 |         y = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, attn_mask=attn_bias)
 236 | 
 237 |         # 9. Reshape back and project
 238 |         y = y.transpose(1, 2).contiguous().view(B, T, -1)
 239 |         y = self.c_proj(y)
 240 |         return y

```

`nanochat/nanochat/tokenizer.py`:

```py
   1 | """
   2 | BPE Tokenizer in the style of GPT-4.
   3 | 
   4 | Two implementations are available:
   5 | 1) HuggingFace Tokenizer that can do both training and inference but is really confusing
   6 | 2) Our own RustBPE Tokenizer for training and tiktoken for efficient inference
   7 | """
   8 | 
   9 | import os
  10 | import copy
  11 | from functools import lru_cache
  12 | 
  13 | SPECIAL_TOKENS = [
  14 |     # every document begins with the Beginning of Sequence (BOS) token that delimits documents
  15 |     "<|bos|>",
  16 |     # tokens below are only used during finetuning to render Conversations into token ids
  17 |     "<|user_start|>", # user messages
  18 |     "<|user_end|>",
  19 |     "<|assistant_start|>", # assistant messages
  20 |     "<|assistant_end|>",
  21 |     "<|python_start|>", # assistant invokes python REPL tool
  22 |     "<|python_end|>",
  23 |     "<|output_start|>", # python REPL outputs back to assistant
  24 |     "<|output_end|>",
  25 | ]
  26 | 
  27 | # NOTE: this split pattern deviates from GPT-4 in that we use \p{N}{1,2} instead of \p{N}{1,3}
  28 | # I did this because I didn't want to "waste" too many tokens on numbers for smaller vocab sizes.
  29 | # I verified that 2 is the sweet spot for vocab size of 32K. 1 is a bit worse, 3 was worse still.
  30 | SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
  31 | 
  32 | # -----------------------------------------------------------------------------
  33 | # Generic GPT-4-style tokenizer based on HuggingFace Tokenizer
  34 | from tokenizers import Tokenizer as HFTokenizer
  35 | from tokenizers import pre_tokenizers, decoders, Regex
  36 | from tokenizers.models import BPE
  37 | from tokenizers.trainers import BpeTrainer
  38 | 
  39 | class HuggingFaceTokenizer:
  40 |     """Light wrapper around HuggingFace Tokenizer for some utilities"""
  41 | 
  42 |     def __init__(self, tokenizer):
  43 |         self.tokenizer = tokenizer
  44 | 
  45 |     @classmethod
  46 |     def from_pretrained(cls, hf_path):
  47 |         # init from a HuggingFace pretrained tokenizer (e.g. "gpt2")
  48 |         tokenizer = HFTokenizer.from_pretrained(hf_path)
  49 |         return cls(tokenizer)
  50 | 
  51 |     @classmethod
  52 |     def from_directory(cls, tokenizer_dir):
  53 |         # init from a local directory on disk (e.g. "out/tokenizer")
  54 |         tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
  55 |         tokenizer = HFTokenizer.from_file(tokenizer_path)
  56 |         return cls(tokenizer)
  57 | 
  58 |     @classmethod
  59 |     def train_from_iterator(cls, text_iterator, vocab_size):
  60 |         # train from an iterator of text
  61 |         # Configure the HuggingFace Tokenizer
  62 |         tokenizer = HFTokenizer(BPE(
  63 |             byte_fallback=True, # needed!
  64 |             unk_token=None,
  65 |             fuse_unk=False,
  66 |         ))
  67 |         # Normalizer: None
  68 |         tokenizer.normalizer = None
  69 |         # Pre-tokenizer: GPT-4 style
  70 |         # the regex pattern used by GPT-4 to split text into groups before BPE
  71 |         # NOTE: The pattern was changed from \p{N}{1,3} to \p{N}{1,2} because I suspect it is harmful to
  72 |         # very small models and smaller vocab sizes, because it is a little bit wasteful in the token space.
  73 |         # (but I haven't validated this! TODO)
  74 |         gpt4_split_regex = Regex(SPLIT_PATTERN) # huggingface demands that you wrap it in Regex!!
  75 |         tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
  76 |             pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
  77 |             pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
  78 |         ])
  79 |         # Decoder: ByteLevel (it pairs together with the ByteLevel pre-tokenizer)
  80 |         tokenizer.decoder = decoders.ByteLevel()
  81 |         # Post-processor: None
  82 |         tokenizer.post_processor = None
  83 |         # Trainer: BPE
  84 |         trainer = BpeTrainer(
  85 |             vocab_size=vocab_size,
  86 |             show_progress=True,
  87 |             min_frequency=0, # no minimum frequency
  88 |             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
  89 |             special_tokens=SPECIAL_TOKENS,
  90 |         )
  91 |         # Kick off the training
  92 |         tokenizer.train_from_iterator(text_iterator, trainer)
  93 |         return cls(tokenizer)
  94 | 
  95 |     def get_vocab_size(self):
  96 |         return self.tokenizer.get_vocab_size()
  97 | 
  98 |     def get_special_tokens(self):
  99 |         special_tokens_map = self.tokenizer.get_added_tokens_decoder()
 100 |         special_tokens = [w.content for w in special_tokens_map.values()]
 101 |         return special_tokens
 102 | 
 103 |     def id_to_token(self, id):
 104 |         return self.tokenizer.id_to_token(id)
 105 | 
 106 |     def _encode_one(self, text, prepend=None, append=None, num_threads=None):
 107 |         # encode a single string
 108 |         # prepend/append can be either a string of a special token or a token id directly.
 109 |         # num_threads is ignored (only used by the nanochat Tokenizer for parallel encoding)
 110 |         assert isinstance(text, str)
 111 |         ids = []
 112 |         if prepend is not None:
 113 |             prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
 114 |             ids.append(prepend_id)
 115 |         ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
 116 |         if append is not None:
 117 |             append_id = append if isinstance(append, int) else self.encode_special(append)
 118 |             ids.append(append_id)
 119 |         return ids
 120 | 
 121 |     def encode_special(self, text):
 122 |         # encode a single special token via exact match
 123 |         return self.tokenizer.token_to_id(text)
 124 | 
 125 |     def get_bos_token_id(self):
 126 |         # Different HuggingFace models use different BOS tokens and there is little consistency
 127 |         # 1) attempt to find a <|bos|> token
 128 |         bos = self.encode_special("<|bos|>")
 129 |         # 2) if that fails, attempt to find a <|endoftext|> token (e.g. GPT-2 models)
 130 |         if bos is None:
 131 |             bos = self.encode_special("<|endoftext|>")
 132 |         # 3) if these fail, it's better to crash than to silently return None
 133 |         assert bos is not None, "Failed to find BOS token in tokenizer"
 134 |         return bos
 135 | 
 136 |     def encode(self, text, *args, **kwargs):
 137 |         if isinstance(text, str):
 138 |             return self._encode_one(text, *args, **kwargs)
 139 |         elif isinstance(text, list):
 140 |             return [self._encode_one(t, *args, **kwargs) for t in text]
 141 |         else:
 142 |             raise ValueError(f"Invalid input type: {type(text)}")
 143 | 
 144 |     def __call__(self, *args, **kwargs):
 145 |         return self.encode(*args, **kwargs)
 146 | 
 147 |     def decode(self, ids):
 148 |         return self.tokenizer.decode(ids, skip_special_tokens=False)
 149 | 
 150 |     def save(self, tokenizer_dir):
 151 |         # save the tokenizer to disk
 152 |         os.makedirs(tokenizer_dir, exist_ok=True)
 153 |         tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
 154 |         self.tokenizer.save(tokenizer_path)
 155 |         print(f"Saved tokenizer to {tokenizer_path}")
 156 | 
 157 | # -----------------------------------------------------------------------------
 158 | # Tokenizer based on rustbpe + tiktoken combo
 159 | import pickle
 160 | import rustbpe
 161 | import tiktoken
 162 | 
 163 | class RustBPETokenizer:
 164 |     """Light wrapper around tiktoken (for efficient inference) but train with rustbpe"""
 165 | 
 166 |     def __init__(self, enc, bos_token):
 167 |         self.enc = enc
 168 |         self.bos_token_id = self.encode_special(bos_token)
 169 | 
 170 |     @classmethod
 171 |     def train_from_iterator(cls, text_iterator, vocab_size):
 172 |         # 1) train using rustbpe
 173 |         tokenizer = rustbpe.Tokenizer()
 174 |         # the special tokens are inserted later in __init__, we don't train them here
 175 |         vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
 176 |         assert vocab_size_no_special >= 256, f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
 177 |         tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
 178 |         # 2) construct the associated tiktoken encoding for inference
 179 |         pattern = tokenizer.get_pattern()
 180 |         mergeable_ranks_list = tokenizer.get_mergeable_ranks()
 181 |         mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
 182 |         tokens_offset = len(mergeable_ranks)
 183 |         special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
 184 |         enc = tiktoken.Encoding(
 185 |             name="rustbpe",
 186 |             pat_str=pattern,
 187 |             mergeable_ranks=mergeable_ranks, # dict[bytes, int] (token bytes -> merge priority rank)
 188 |             special_tokens=special_tokens, # dict[str, int] (special token name -> token id)
 189 |         )
 190 |         return cls(enc, "<|bos|>")
 191 | 
 192 |     @classmethod
 193 |     def from_directory(cls, tokenizer_dir):
 194 |         pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
 195 |         with open(pickle_path, "rb") as f:
 196 |             enc = pickle.load(f)
 197 |         return cls(enc, "<|bos|>")
 198 | 
 199 |     @classmethod
 200 |     def from_pretrained(cls, tiktoken_name):
 201 |         # https://github.com/openai/tiktoken/blob/eedc8563/tiktoken_ext/openai_public.py
 202 |         enc = tiktoken.get_encoding(tiktoken_name)
 203 |         # tiktoken calls the special document delimiter token "<|endoftext|>"
 204 |         # yes this is confusing because this token is almost always PREPENDED to the beginning of the document
 205 |         # it most often is used to signal the start of a new sequence to the LLM during inference etc.
 206 |         # so in nanoChat we always use "<|bos|>" short for "beginning of sequence", but historically it is often called "<|endoftext|>".
 207 |         return cls(enc, "<|endoftext|>")
 208 | 
 209 |     def get_vocab_size(self):
 210 |         return self.enc.n_vocab
 211 | 
 212 |     def get_special_tokens(self):
 213 |         return self.enc.special_tokens_set
 214 | 
 215 |     def id_to_token(self, id):
 216 |         return self.enc.decode([id])
 217 | 
 218 |     @lru_cache(maxsize=32)
 219 |     def encode_special(self, text):
 220 |         return self.enc.encode_single_token(text)
 221 | 
 222 |     def get_bos_token_id(self):
 223 |         return self.bos_token_id
 224 | 
 225 |     def encode(self, text, prepend=None, append=None, num_threads=8):
 226 |         # text can be either a string or a list of strings
 227 | 
 228 |         if prepend is not None:
 229 |             prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
 230 |         if append is not None:
 231 |             append_id = append if isinstance(append, int) else self.encode_special(append)
 232 | 
 233 |         if isinstance(text, str):
 234 |             ids = self.enc.encode_ordinary(text)
 235 |             if prepend is not None:
 236 |                 ids.insert(0, prepend_id) # TODO: slightly inefficient here? :( hmm
 237 |             if append is not None:
 238 |                 ids.append(append_id)
 239 |         elif isinstance(text, list):
 240 |             ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
 241 |             if prepend is not None:
 242 |                 for ids_row in ids:
 243 |                     ids_row.insert(0, prepend_id) # TODO: same
 244 |             if append is not None:
 245 |                 for ids_row in ids:
 246 |                     ids_row.append(append_id)
 247 |         else:
 248 |             raise ValueError(f"Invalid input type: {type(text)}")
 249 | 
 250 |         return ids
 251 | 
 252 |     def __call__(self, *args, **kwargs):
 253 |         return self.encode(*args, **kwargs)
 254 | 
 255 |     def decode(self, ids):
 256 |         return self.enc.decode(ids)
 257 | 
 258 |     def save(self, tokenizer_dir):
 259 |         # save the encoding object to disk
 260 |         os.makedirs(tokenizer_dir, exist_ok=True)
 261 |         pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
 262 |         with open(pickle_path, "wb") as f:
 263 |             pickle.dump(self.enc, f)
 264 |         print(f"Saved tokenizer encoding to {pickle_path}")
 265 | 
 266 |     def render_conversation(self, conversation, max_tokens=2048):
 267 |         """
 268 |         Tokenize a single Chat conversation (which we call a "doc" or "document" here).
 269 |         Returns:
 270 |         - ids: list[int] is a list of token ids of this rendered conversation
 271 |         - mask: list[int] of same length, mask = 1 for tokens that the Assistant is expected to train on.
 272 |         """
 273 |         # ids, masks that we will return and a helper function to help build them up.
 274 |         ids, mask = [], []
 275 |         def add_tokens(token_ids, mask_val):
 276 |             if isinstance(token_ids, int):
 277 |                 token_ids = [token_ids]
 278 |             ids.extend(token_ids)
 279 |             mask.extend([mask_val] * len(token_ids))
 280 | 
 281 |         # sometimes the first message is a system message...
 282 |         # => just merge it with the second (user) message
 283 |         if conversation["messages"][0]["role"] == "system":
 284 |             # some conversation surgery is necessary here for now...
 285 |             conversation = copy.deepcopy(conversation) # avoid mutating the original
 286 |             messages = conversation["messages"]
 287 |             assert messages[1]["role"] == "user", "System message must be followed by a user message"
 288 |             messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
 289 |             messages = messages[1:]
 290 |         else:
 291 |             messages = conversation["messages"]
 292 |         assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"
 293 | 
 294 |         # fetch all the special tokens we need
 295 |         bos = self.get_bos_token_id()
 296 |         user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
 297 |         assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")
 298 |         python_start, python_end = self.encode_special("<|python_start|>"), self.encode_special("<|python_end|>")
 299 |         output_start, output_end = self.encode_special("<|output_start|>"), self.encode_special("<|output_end|>")
 300 | 
 301 |         # now we can tokenize the conversation
 302 |         add_tokens(bos, 0)
 303 |         for i, message in enumerate(messages):
 304 | 
 305 |             # some sanity checking here around assumptions, to prevent footguns
 306 |             must_be_from = "user" if i % 2 == 0 else "assistant"
 307 |             assert message["role"] == must_be_from, f"Message {i} is from {message['role']} but should be from {must_be_from}"
 308 | 
 309 |             # content can be either a simple string or a list of parts (e.g. containing tool calls)
 310 |             content = message["content"]
 311 | 
 312 |             if message["role"] == "user":
 313 |                 assert isinstance(content, str), "User messages are simply expected to be strings"
 314 |                 value_ids = self.encode(content)
 315 |                 add_tokens(user_start, 0)
 316 |                 add_tokens(value_ids, 0)
 317 |                 add_tokens(user_end, 0)
 318 |             elif message["role"] == "assistant":
 319 |                 add_tokens(assistant_start, 0)
 320 |                 if isinstance(content, str):
 321 |                     # simple string => simply add the tokens
 322 |                     value_ids = self.encode(content)
 323 |                     add_tokens(value_ids, 1)
 324 |                 elif isinstance(content, list):
 325 |                     for part in content:
 326 |                         value_ids = self.encode(part["text"])
 327 |                         if part["type"] == "text":
 328 |                             # string part => simply add the tokens
 329 |                             add_tokens(value_ids, 1)
 330 |                         elif part["type"] == "python":
 331 |                             # python tool call => add the tokens inside <|python_start|> and <|python_end|>
 332 |                             add_tokens(python_start, 1)
 333 |                             add_tokens(value_ids, 1)
 334 |                             add_tokens(python_end, 1)
 335 |                         elif part["type"] == "python_output":
 336 |                             # python output => add the tokens inside <|output_start|> and <|output_end|>
 337 |                             # none of these tokens are supervised because the tokens come from Python at test time
 338 |                             add_tokens(output_start, 0)
 339 |                             add_tokens(value_ids, 0)
 340 |                             add_tokens(output_end, 0)
 341 |                         else:
 342 |                             raise ValueError(f"Unknown part type: {part['type']}")
 343 |                 else:
 344 |                     raise ValueError(f"Unknown content type: {type(content)}")
 345 |                 add_tokens(assistant_end, 1)
 346 | 
 347 |         # truncate to max_tokens tokens MAX (helps prevent OOMs)
 348 |         ids = ids[:max_tokens]
 349 |         mask = mask[:max_tokens]
 350 |         return ids, mask
 351 | 
 352 |     def visualize_tokenization(self, ids, mask, with_token_id=False):
 353 |         """Small helper function useful in debugging: visualize the tokenization of render_conversation"""
 354 |         RED = '\033[91m'
 355 |         GREEN = '\033[92m'
 356 |         RESET = '\033[0m'
 357 |         GRAY = '\033[90m'
 358 |         tokens = []
 359 |         for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
 360 |             token_str = self.decode([token_id])
 361 |             color = GREEN if mask_val == 1 else RED
 362 |             tokens.append(f"{color}{token_str}{RESET}")
 363 |             if with_token_id:
 364 |                 tokens.append(f"{GRAY}({token_id}){RESET}")
 365 |         return '|'.join(tokens)
 366 | 
 367 |     def render_for_completion(self, conversation):
 368 |         """
 369 |         Used during Reinforcement Learning. In that setting, we want to
 370 |         render the conversation priming the Assistant for a completion.
 371 |         Unlike the Chat SFT case, we don't need to return the mask.
 372 |         """
 373 |         # We have some surgery to do: we need to pop the last message (of the Assistant)
 374 |         conversation = copy.deepcopy(conversation) # avoid mutating the original
 375 |         messages = conversation["messages"]
 376 |         assert messages[-1]["role"] == "assistant", "Last message must be from the Assistant"
 377 |         messages.pop() # remove the last message (of the Assistant) inplace
 378 | 
 379 |         # Now tokenize the conversation
 380 |         ids, mask = self.render_conversation(conversation)
 381 | 
 382 |         # Finally, to prime the Assistant for a completion, append the Assistant start token
 383 |         assistant_start = self.encode_special("<|assistant_start|>")
 384 |         ids.append(assistant_start)
 385 |         return ids
 386 | 
 387 | # -----------------------------------------------------------------------------
 388 | # nanochat-specific convenience functions
 389 | 
 390 | # C++ keywords that should be single tokens with the C++ tokenizer
 391 | _CPP_KEYWORDS = ["class", "struct", "namespace", "template", "virtual", "override",
 392 |                  "public", "private", "protected", "const", "static", "inline",
 393 |                  "void", "int", "bool", "char", "float", "double", "auto", "return"]
 394 | 
 395 | 
 396 | def verify_cpp_tokenizer(tokenizer, sample_code=None):
 397 |     """
 398 |     Verify that the C++ tokenizer is correctly tokenizing C++ keywords as single tokens.
 399 | 
 400 |     Call this at training start to confirm the tokenizer is appropriate for your data.
 401 |     Returns True if C++ keywords are single tokens, False otherwise.
 402 |     Prints a warning if the C++ tokenizer appears to be splitting keywords.
 403 |     """
 404 |     if sample_code is None:
 405 |         # Use a representative C++ snippet
 406 |         sample_code = "class MyClass { public: virtual void process() const override; };"
 407 | 
 408 |     tokens = tokenizer.encode(sample_code)
 409 | 
 410 |     # Check if common C++ keywords are single tokens
 411 |     keywords_found = 0
 412 |     keywords_single = 0
 413 | 
 414 |     for keyword in _CPP_KEYWORDS[:10]:  # Check first 10 keywords
 415 |         keyword_tokens = tokenizer.encode(keyword)
 416 |         if len(keyword_tokens) == 1:
 417 |             keywords_single += 1
 418 |         keywords_found += 1
 419 | 
 420 |     is_cpp_optimized = keywords_single >= 8  # At least 80% should be single tokens
 421 | 
 422 |     if not is_cpp_optimized:
 423 |         print(f"WARNING: C++ tokenizer check: only {keywords_single}/{keywords_found} C++ keywords "
 424 |               f"are single tokens. This tokenizer may not be optimal for C++ code.")
 425 |         print("NOTE: NANOCHAT_CPP_TOKENIZER=1 is now the default.")
 426 |         print("      Set NANOCHAT_CPP_TOKENIZER=0 to use RustBPE tokenizer for non-C++ training.")
 427 | 
 428 |     return is_cpp_optimized
 429 | 
 430 | 
 431 | def get_tokenizer():
 432 |     """
 433 |     Get the default nanochat tokenizer.
 434 | 
 435 |     C++ tokenizer (tokenizer.json) is now the DEFAULT for C++ code training.
 436 |     Set NANOCHAT_CPP_TOKENIZER=0 to disable and use RustBPE tokenizer instead.
 437 |     """
 438 |     from nanochat.common import get_base_dir
 439 |     base_dir = get_base_dir()
 440 |     tokenizer_dir = os.path.join(base_dir, "tokenizer")
 441 | 
 442 |     # C++ tokenizer is now the DEFAULT
 443 |     # Set NANOCHAT_CPP_TOKENIZER=0 to disable
 444 |     use_cpp = os.environ.get("NANOCHAT_CPP_TOKENIZER", "1") != "0"
 445 | 
 446 |     cpp_tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
 447 |     if use_cpp and os.path.exists(cpp_tokenizer_path):
 448 |         from nanochat.cpp_tokenizer import CppTokenizer
 449 |         return CppTokenizer(tokenizer_dir)
 450 | 
 451 |     # Fallback to RustBPE tokenizer
 452 |     return RustBPETokenizer.from_directory(tokenizer_dir)
 453 | 
 454 | def get_token_bytes(device="cpu"):
 455 |     import torch
 456 |     from nanochat.common import get_base_dir
 457 |     base_dir = get_base_dir()
 458 |     tokenizer_dir = os.path.join(base_dir, "tokenizer")
 459 |     token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
 460 |     assert os.path.exists(token_bytes_path), f"Token bytes not found at {token_bytes_path}? It gets written by tok_train.py"
 461 |     with open(token_bytes_path, "rb") as f:
 462 |         # Load to CPU first, then move to target device
 463 |         # (XLA devices don't support direct torch.load map_location)
 464 |         token_bytes = torch.load(f, map_location="cpu")
 465 |     # Handle 2D token_bytes (raw byte values) vs 1D (byte counts)
 466 |     # Some tokenizer builds save raw bytes as (vocab_size, max_bytes) uint8
 467 |     # while loss_eval expects 1D byte counts (vocab_size,)
 468 |     if token_bytes.dim() == 2:
 469 |         # Convert 2D raw bytes to 1D byte counts: count non-zero entries per row
 470 |         token_bytes = (token_bytes > 0).sum(dim=1).to(torch.int32)
 471 |     if str(device) != "cpu":
 472 |         token_bytes = token_bytes.to(device)
 473 |     return token_bytes

```

`nanochat/scripts/base_train.py`:

```py
   1 | """
   2 | Train model. From root directory of the project, run as:
   3 | 
   4 | python -m scripts.base_train.py
   5 | 
   6 | or distributed as:
   7 | 
   8 | torchrun --nproc_per_node=8 -m scripts.base_train.py
   9 | 
  10 | If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
  11 | python -m scripts.base_train --depth=4 --max_seq_len=512 --device_batch_size=1 --eval_tokens=512 --core_metric_every=-1 --total_batch_size=512 --num_iterations=20
  12 | """
  13 | 
  14 | import os
  15 | import shutil
  16 | 
  17 | # Triton SM121a auto-fix: Triton bundles ptxas 12.8 which doesn't support SM121a (GB10)
  18 | # We need system ptxas from CUDA 13.0+ for GB10/DGX Spark
  19 | if not os.environ.get("TRITON_PTXAS_PATH"):
  20 |     for ptxas in ["/usr/local/cuda/bin/ptxas", shutil.which("ptxas")]:
  21 |         if ptxas and os.path.exists(ptxas):
  22 |             os.environ["TRITON_PTXAS_PATH"] = ptxas
  23 |             break
  24 | 
  25 | os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
  26 | 
  27 | # GB10 SM count fix: PyTorch's is_big_gpu() requires 68 SMs but GB10 has 48
  28 | # Force max_autotune_gemm to work on GB10 by setting env var before torch import
  29 | os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "1"
  30 | 
  31 | # Persistent cache for autotune results (survives reboot, faster subsequent runs)
  32 | CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".cache", "torchinductor")
  33 | os.makedirs(CACHE_DIR, exist_ok=True)
  34 | os.environ["TORCHINDUCTOR_CACHE_DIR"] = CACHE_DIR
  35 | import gc
  36 | import argparse
  37 | import time
  38 | from contextlib import nullcontext, contextmanager
  39 | 
  40 | import wandb
  41 | import torch
  42 | 
  43 | # Disable torch.compile entirely for XLA/TPU - the inductor backend doesn't support XLA devices
  44 | # Must be done before any torch.compile calls or dynamo configurations
  45 | if os.environ.get("PJRT_DEVICE") == "TPU":
  46 |     torch._dynamo.config.suppress_errors = True
  47 |     torch._dynamo.config.disable = True
  48 |     # Note: --xla_tpu_disable_full_embedding_pipelining=true was tested
  49 |     # but is NOT supported by libtpu 0.0.21 (v2-alpha-tpuv6e runtime)
  50 | else:
  51 |     # Fix Liger-Kernel graph breaks: LigerFusedLinearCrossEntropy calls .item() internally
  52 |     # which causes torch.compile graph breaks. This config enables capturing scalar outputs.
  53 |     torch._dynamo.config.capture_scalar_outputs = True
  54 | 
  55 | # Patch is_big_gpu to return True for GB10 (48 SMs < 68 SM threshold)
  56 | # This eliminates the "Not enough SMs to use max_autotune_gemm mode" warning
  57 | try:
  58 |     import torch._inductor.utils as inductor_utils
  59 |     inductor_utils.is_big_gpu = lambda index=0: True
  60 | except Exception:
  61 |     pass
  62 | 
  63 | from nanochat.gpt import GPT, GPTConfig, select_precision, make_autocast_ctx
  64 | from nanochat.dataloader import tokenizing_distributed_data_loader, tokenizing_distributed_data_loader_with_state
  65 | from nanochat.common import (
  66 |     compute_init,
  67 |     compute_cleanup,
  68 |     print0,
  69 |     DummyWandb,
  70 |     print_banner,
  71 |     get_base_dir,
  72 |     autodetect_device_type,
  73 |     get_tpu_accelerator_type,
  74 |     xla_all_reduce_gradients,
  75 |     _is_tpu_requested,
  76 | )
  77 | from nanochat import kernels
  78 | from nanochat.tokenizer import get_tokenizer, get_token_bytes, verify_cpp_tokenizer
  79 | from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint, find_last_step
  80 | from nanochat.loss_eval import evaluate_bpb
  81 | from nanochat.engine import Engine
  82 | from nanochat.cpp_eval import evaluate_cpp_model
  83 | 
  84 | # -----------------------------------------------------------------------------
  85 | # CLI arguments
  86 | parser = argparse.ArgumentParser(description="Pretrain base model")
  87 | # Logging
  88 | parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
  89 | # Runtime
  90 | parser.add_argument("--device_type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
  91 | # Model architecture
  92 | parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
  93 | parser.add_argument("--aspect_ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
  94 | parser.add_argument("--head_dim", type=int, default=128, help="target head dimension for attention")
  95 | parser.add_argument("--max_seq_len", type=int, default=2048, help="max context length (supports up to 16384 for long context)")
  96 | parser.add_argument("--window_pattern", type=str, default="SSSL", help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL')")
  97 | parser.add_argument("--engram", action="store_true", help="enable optional Engram branches")
  98 | parser.add_argument("--engram_layers", type=str, default="", help="comma-separated layer indices for Engram insertion (empty=disabled)")
  99 | parser.add_argument("--engram_ngram_orders", type=str, default="2,3,4", help="comma-separated n-gram orders for Engram branch")
 100 | parser.add_argument("--engram_bottleneck_dim", type=int, default=0, help="Engram bottleneck dimension (0=auto)")
 101 | parser.add_argument("--engram_dropout", type=float, default=0.0, help="dropout on Engram branch")
 102 | parser.add_argument("--mhc", action="store_true", help="enable optional mHC branch mixing")
 103 | parser.add_argument("--mhc_num_branches", type=int, default=0, help="mHC branch count (0=auto)")
 104 | parser.add_argument("--mhc_sinkhorn_iters", type=int, default=5, help="mHC Sinkhorn iterations")
 105 | parser.add_argument("--mhc_temperature", type=float, default=1.0, help="mHC transport temperature")
 106 | parser.add_argument("--mhc_epsilon", type=float, default=1e-6, help="mHC numerical epsilon")
 107 | parser.add_argument("--mhc_blend_alpha", type=float, default=1.0, help="global mHC blend strength")
 108 | parser.add_argument("--aux_loss_weight", type=float, default=0.0, help="auxiliary regularization loss weight")
 109 | # Multi-Token Prediction (DeepSeek-V3 style)
 110 | parser.add_argument("--mtp", action="store_true", help="enable Multi-Token Prediction (predicts token i+2)")
 111 | parser.add_argument("--mtp_lambda", type=float, default=0.3, help="MTP loss weight (DeepSeek uses 0.3 early, 0.1 later)")
 112 | # DeepSeek Sparse Attention (DSA)
 113 | parser.add_argument("--dsa", action="store_true", help="enable DeepSeek Sparse Attention from dsa_start_layer to last layer")
 114 | parser.add_argument("--dsa_start_layer", type=int, default=7, help="first layer to use sparse attention (0-indexed)")
 115 | parser.add_argument("--dsa_top_k_ratio", type=float, default=0.5, help="fraction of tokens to attend to in sparse layers")
 116 | parser.add_argument("--dsa_local_window", type=int, default=128, help="local window always included in sparse attention")
 117 | parser.add_argument("--dsa_indexer_heads", type=int, default=16, help="number of lightweight indexer heads for DSA")
 118 | parser.add_argument("--dsa_indexer_dim", type=int, default=32, help="dimension per indexer head for DSA")
 119 | # Memory optimization
 120 | parser.add_argument("--gradient_checkpointing", action="store_true", help="enable gradient checkpointing (saves memory, trades compute)")
 121 | parser.add_argument("--tensor_parallel", type=int, default=1, help="tensor parallelism degree (1=data-only, 2/4/8=split model across chips)")
 122 | # Training horizon (only one used, in order of precedence)
 123 | parser.add_argument("--num_iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
 124 | parser.add_argument("--target_flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
 125 | parser.add_argument("--target_param_data_ratio", type=int, default=8, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
 126 | # Optimization
 127 | parser.add_argument("--device_batch_size", type=int, default=32, help="per-device batch size")
 128 | parser.add_argument("--total_batch_size", type=int, default=524288, help="total batch size in tokens")
 129 | parser.add_argument("--embedding_lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
 130 | parser.add_argument("--unembedding_lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
 131 | parser.add_argument("--weight_decay", type=float, default=0.2, help="cautious weight decay for the Muon optimizer (for weights)")
 132 | parser.add_argument("--matrix_lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
 133 | parser.add_argument("--scalar_lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
 134 | parser.add_argument("--adam_beta1", type=float, default=0.8, help="Adam beta1 for embedding/unembedding")
 135 | parser.add_argument("--adam_beta2", type=float, default=0.95, help="Adam beta2 for embedding/unembedding")
 136 | parser.add_argument("--warmup_ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
 137 | parser.add_argument("--warmdown_ratio", type=float, default=0.4, help="ratio of iterations for LR warmdown")
 138 | parser.add_argument("--final_lr_frac", type=float, default=0.0, help="final LR as fraction of initial LR")
 139 | parser.add_argument("--resume_from_step", type=int, default=-1, help="resume training from this step (-1 = disable, -2 = auto-detect latest checkpoint)")
 140 | # Evaluation
 141 | parser.add_argument("--eval_every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
 142 | parser.add_argument("--eval_tokens", type=int, default=20*524288, help="number of tokens to evaluate val loss on")
 143 | parser.add_argument("--core_metric_every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
 144 | parser.add_argument("--core_metric_max_per_task", type=int, default=500, help="examples per task for CORE metric")
 145 | parser.add_argument("--sample_every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
 146 | parser.add_argument("--save_every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
 147 | # Dataloader
 148 | parser.add_argument("--tokenizer_threads", type=int, default=4, help="number of threads for tokenization")
 149 | parser.add_argument("--tokenizer_batch_size", type=int, default=128, help="batch size for tokenization")
 150 | parser.add_argument("--fim_rate", type=float, default=0.0, help="Fill-in-the-Middle rate (0.0=disabled, 0.5=50%% of docs get FIM). Requires NANOCHAT_CPP_TOKENIZER=1")
 151 | parser.add_argument("--structured_fim_rate", type=float, default=0.0, help="Structured FIM rate for docstring->code completion (0.0=disabled)")
 152 | parser.add_argument("--structured_fim_path", type=str, default="data/docstring_pairs_full.jsonl", help="Path to structured FIM pairs dataset")
 153 | # Output
 154 | parser.add_argument("--model_tag", type=str, default=None, help="override model tag for checkpoint directory name")
 155 | # Precision (NVFP4/FP8/BF16)
 156 | parser.add_argument("--precision", type=str, default="auto", help="precision: auto|nvfp4|fp8|bf16")
 157 | parser.add_argument("--nvfp4_disable_rht", type=bool, default=True, help="disable Random Hadamard Transform (required for SM121/GB10)")
 158 | parser.add_argument("--nvfp4_disable_sr", type=bool, default=True, help="disable Stochastic Rounding (required for SM121/GB10)")
 159 | # FP8 training with torchao (separate from TE-based --precision fp8)
 160 | parser.add_argument("--fp8", action="store_true", help="enable FP8 training with torchao (requires H100+ GPU)")
 161 | parser.add_argument("--fp8_recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe: tensorwise (faster) or rowwise (more accurate)")
 162 | # Kernel backend
 163 | parser.add_argument("--kernel", type=str, default="current", choices=["current", "liger", "cce", "triton"], help="kernel backend: current (PyTorch), liger (Liger-Kernel), cce (Apple Cut Cross Entropy), triton (Unsloth-style)")
 164 | # torch.compile
 165 | parser.add_argument("--no_compile", action="store_true", help="disable torch.compile (use for NVIDIA containers with triton issues)")
 166 | # XLA/TPU optimizations
 167 | parser.add_argument("--use_scan", action="store_true", help="use torch_xla scan_layers to reduce XLA compilation time (TPU only)")
 168 | parser.add_argument("--xla_flash_attn", action="store_true", help="use XLA Pallas flash attention for TPU (O(n) memory, enables long seq_len). Does not support sliding window - use --window_pattern=L")
 169 | parser.add_argument("--chunked_attn", action="store_true", help="use chunked attention for long sequences on XLA/TPU. Pure PyTorch, no JAX needed. Reduces O(n^2) to O(n*chunk)")
 170 | parser.add_argument("--attn_chunk_size", type=int, default=1024, help="chunk size for chunked attention (default: 1024)")
 171 | # Data directory
 172 | parser.add_argument("--data_dir", type=str, default="", help="Custom parquet data directory (default: base_data from NANOCHAT_BASE_DIR)")
 173 | parser.add_argument("--streaming_data", action="store_true", help="Streaming mode: dynamically discover new parquet shards as they arrive. Waits for _COMPLETE sentinel.")
 174 | args = parser.parse_args()
 175 | user_config = vars(args).copy()  # for logging
 176 | 
 177 | 
 178 | def _build_gpt_config(model_config_kwargs):
 179 |     supported_keys = set(getattr(GPTConfig, "__dataclass_fields__", {}).keys())
 180 |     if not supported_keys:
 181 |         return GPTConfig(**model_config_kwargs)
 182 |     filtered_kwargs = {k: v for k, v in model_config_kwargs.items() if k in supported_keys}
 183 |     dropped_keys = sorted(set(model_config_kwargs.keys()) - set(filtered_kwargs.keys()))
 184 |     if dropped_keys:
 185 |         print0(f"Ignoring unsupported GPTConfig keys in this checkout: {', '.join(dropped_keys)}")
 186 |     return GPTConfig(**filtered_kwargs)
 187 | 
 188 | # If --no_compile is set, also disable compile in Muon optimizer via env var
 189 | # This must be done before importing muon.py
 190 | if args.no_compile:
 191 |     os.environ["NANOCHAT_NO_COMPILE"] = "1"
 192 |     torch._dynamo.config.suppress_errors = True
 193 |     torch._dynamo.config.disable = True
 194 | 
 195 | # Set kernel backend
 196 | kernels.set_kernel_backend(args.kernel)
 197 | 
 198 | # Enable XLA flash attention if requested (before any model creation)
 199 | if args.xla_flash_attn:
 200 |     from nanochat.flash_attention import enable_xla_flash_attention
 201 |     enable_xla_flash_attention()
 202 | if args.chunked_attn:
 203 |     from nanochat.flash_attention import enable_chunked_attention
 204 |     enable_chunked_attention(chunk_size=args.attn_chunk_size, threshold=2048)
 205 | # -----------------------------------------------------------------------------
 206 | 
 207 | 
 208 | def _apply_tensor_parallel_sharding(model, mesh):
 209 |     """Apply Megatron-style tensor parallelism via SPMD weight sharding.
 210 | 
 211 |     Shards attention Q/K/V columns and MLP columns across the 'model' axis,
 212 |     and attention/MLP output projections as row-parallel.
 213 |     """
 214 |     import torch_xla.distributed.spmd as xs
 215 | 
 216 |     n_sharded = 0
 217 |     for name, param in model.named_parameters():
 218 |         # Attention Q/K/V projections: column-parallel (shard output dim)
 219 |         # Weight shape: [n_head*head_dim, n_embd] -> shard dim 0 across 'model'
 220 |         if any(k in name for k in ('c_q.weight', 'c_k.weight', 'c_v.weight')):
 221 |             xs.mark_sharding(param, mesh, ('model', None))
 222 |             n_sharded += 1
 223 |         # Attention output projection: row-parallel (shard input dim)
 224 |         # Weight shape: [n_embd, n_embd] -> shard dim 1 across 'model'
 225 |         elif 'attn.c_proj.weight' in name:
 226 |             xs.mark_sharding(param, mesh, (None, 'model'))
 227 |             n_sharded += 1
 228 |         # MLP first layer: column-parallel (shard output dim)
 229 |         # Weight shape: [4*n_embd, n_embd] -> shard dim 0 across 'model'
 230 |         elif 'mlp.c_fc.weight' in name:
 231 |             xs.mark_sharding(param, mesh, ('model', None))
 232 |             n_sharded += 1
 233 |         # MLP output projection: row-parallel (shard input dim)
 234 |         # Weight shape: [n_embd, 4*n_embd] -> shard dim 1 across 'model'
 235 |         elif 'mlp.c_proj.weight' in name:
 236 |             xs.mark_sharding(param, mesh, (None, 'model'))
 237 |             n_sharded += 1
 238 |         # MTP projection: replicated (it combines hidden states, not parallelizable easily)
 239 |         # Embedding and lm_head: replicated across model dim
 240 |         # Everything else: replicated (default)
 241 | 
 242 |     print0(f"Tensor parallelism: sharded {n_sharded} weight matrices across 'model' axis")
 243 | 
 244 | 
 245 | def train():
 246 |     """Main training function. Single process for all backends (GPU DDP, TPU SPMD, CPU)."""
 247 |     print_banner()
 248 | 
 249 |     # Compute init
 250 |     device_type = autodetect_device_type() if args.device_type == "" else args.device_type
 251 |     ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
 252 |     master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
 253 | 
 254 |     # ---- SPMD setup for multi-chip TPU data + tensor parallelism ----
 255 |     # SPMD uses a single process; XLA distributes data across chips via sharding
 256 |     # annotations. This avoids the ~2s/mark_step overhead of xmp.spawn.
 257 |     # With --tensor_parallel > 1, we use a 2D mesh: (data, model) axes.
 258 |     spmd_mesh = None
 259 |     tp_degree = args.tensor_parallel
 260 |     if device_type == "xla":
 261 |         import torch_xla.runtime as xr
 262 |         num_devices = xr.global_runtime_device_count()
 263 |         if num_devices > 1:
 264 |             import numpy as np
 265 |             import torch_xla.distributed.spmd as xs
 266 |             from torch_xla.distributed.spmd import Mesh
 267 | 
 268 |             device_ids = np.arange(num_devices)
 269 |             if tp_degree > 1:
 270 |                 assert num_devices % tp_degree == 0, f"num_devices={num_devices} not divisible by tp={tp_degree}"
 271 |                 dp_degree = num_devices // tp_degree
 272 |                 spmd_mesh = Mesh(device_ids.reshape(dp_degree, tp_degree),
 273 |                                  (dp_degree, tp_degree), ('data', 'model'))
 274 |                 ddp_world_size = dp_degree  # data-parallel world size
 275 |                 print0(f"SPMD 2D mesh: {dp_degree}-way data × {tp_degree}-way tensor parallelism, {num_devices} TPU devices")
 276 |             else:
 277 |                 spmd_mesh = Mesh(device_ids, (num_devices,), ('data',))
 278 |                 ddp_world_size = num_devices
 279 |                 print0(f"SPMD data parallelism: {num_devices} TPU devices, mesh=({num_devices},)")
 280 |             # Tell flash attention about SPMD mesh so it uses partition specs
 281 |             from nanochat.flash_attention import set_spmd_mesh
 282 |             set_spmd_mesh(spmd_mesh, tp_degree=tp_degree)
 283 | 
 284 |     # PyTorch performance optimizations
 285 |     if device_type == "cuda":
 286 |         torch.backends.cudnn.benchmark = True  # auto-tune cuDNN algorithms
 287 |         torch.set_float32_matmul_precision('high')  # TF32 on Ampere+, faster matmuls
 288 |         # Note: Don't use legacy allow_tf32 API - conflicts with set_float32_matmul_precision
 289 | 
 290 |     # Set up precision plan and autocast context factory
 291 |     precision_plan = select_precision(target=args.precision, disable_rht=args.nvfp4_disable_rht, disable_sr=args.nvfp4_disable_sr)
 292 |     print0(f"Precision: {precision_plan.name}")
 293 |     autocast_ctx = make_autocast_ctx(precision_plan, device_type)
 294 | 
 295 |     # NVFP4 on SM121 requires batch_size >= 2
 296 |     if "NVFP4" in precision_plan.name and args.device_batch_size < 2:
 297 |         print0(f"WARNING: NVFP4 requires device_batch_size >= 2, but got {args.device_batch_size}")
 298 |         print0("Automatically increasing device_batch_size to 2")
 299 |         args.device_batch_size = 2
 300 | 
 301 |     # Set synchronize and memory functions based on device type
 302 |     if device_type == "cuda":
 303 |         synchronize = torch.cuda.synchronize
 304 |         get_max_memory = torch.cuda.max_memory_allocated
 305 |     elif device_type == "xla":
 306 |         import torch_xla.core.xla_model as xm
 307 |         synchronize = xm.mark_step
 308 |         get_max_memory = lambda: 0  # XLA doesn't expose memory stats the same way
 309 |     else:
 310 |         synchronize = lambda: None
 311 |         get_max_memory = lambda: 0
 312 | 
 313 |     # wandb logging init
 314 |     use_dummy_wandb = args.run == "dummy" or not master_process
 315 |     wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)
 316 | 
 317 |     # Tokenizer will be useful for evaluation, also we need the vocab size
 318 |     tokenizer = get_tokenizer()
 319 |     token_bytes = get_token_bytes(device=device)
 320 |     vocab_size = tokenizer.get_vocab_size()
 321 |     print0(f"Vocab size: {vocab_size:,}")
 322 | 
 323 |     # Verify C++ tokenizer is active (pre-scan check)
 324 |     is_cpp_tokenizer = verify_cpp_tokenizer(tokenizer)
 325 |     if is_cpp_tokenizer:
 326 |         print0("C++ tokenizer verified: keywords are single tokens")
 327 | 
 328 |     # Model kwargs are derived from the desired depth of the model
 329 |     num_layers = args.depth
 330 |     model_dim = args.depth * args.aspect_ratio
 331 |     def find_num_heads(model_dim, target_head_dim):
 332 |         # Find num_heads that divides model_dim evenly, with head_dim closest to target.
 333 |         ideal = max(1, round(model_dim / target_head_dim))
 334 |         for offset in range(model_dim):
 335 |             for candidate in [ideal + offset, ideal - offset]:
 336 |                 if candidate > 0 and model_dim % candidate == 0:
 337 |                     return candidate
 338 |         return 1
 339 |     num_heads = find_num_heads(model_dim, args.head_dim)
 340 |     num_kv_heads = num_heads # default is 1:1 GQA (Group Query Attention) ratio (i.e. GQA is disabled)
 341 |     print0(f"num_layers: {num_layers}")
 342 |     print0(f"model_dim: {model_dim}")
 343 |     print0(f"num_heads: {num_heads}")
 344 |     print0(f"num_kv_heads: {num_kv_heads}")
 345 |     if args.dsa:
 346 |         assert args.dsa_start_layer < num_layers, f"dsa_start_layer ({args.dsa_start_layer}) must be < num_layers ({num_layers})"
 347 |         dsa_layers = num_layers - args.dsa_start_layer
 348 |         print0(f"DSA enabled: layers {args.dsa_start_layer}-{num_layers-1} ({dsa_layers} sparse layers, top_k_ratio={args.dsa_top_k_ratio})")
 349 |     if args.mtp:
 350 |         print0(f"MTP enabled: lambda={args.mtp_lambda}")
 351 | 
 352 |     # Optimizer / data / training length related hyperparameters
 353 |     # figure out the needed gradient accumulation to reach the desired total batch size
 354 |     tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
 355 |     world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
 356 |     assert args.total_batch_size % world_tokens_per_fwdbwd == 0
 357 |     grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
 358 |     print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
 359 |     print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
 360 |     print0(f"Total batch size {args.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")
 361 | 
 362 |     # Batch size scaling for learning rates (hyperparameters were tuned at reference batch size 2^19)
 363 |     batch_lr_scale = 1.0
 364 |     reference_batch_size = 2**19
 365 |     batch_ratio = args.total_batch_size / reference_batch_size
 366 |     if batch_ratio != 1.0:
 367 |         # SGD: linear scaling with batch size is standard (not used in nanochat)
 368 |         # AdamW: sqrt scaling is standard
 369 |         # Muon: sqrt scaling is an assumption - not fully studied, but it's a second-order-ish optimizer
 370 |         batch_lr_scale = batch_ratio ** 0.5
 371 |         print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {args.total_batch_size:,} (reference: {reference_batch_size:,})")
 372 | 
 373 |     # Weight decay is tuned at d12 and its scaling seems to be \propto 1/channels^2 (or equivalently, \propto 1/depth^2 due to constant aspect ratio)
 374 |     weight_decay_scaled = args.weight_decay * (12 / args.depth)**2
 375 |     if args.depth != 12:
 376 |         print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")
 377 | 
 378 |     # -----------------------------------------------------------------------------
 379 |     # Initialize the Model
 380 | 
 381 |     # Create a new model with random weights (directly on CUDA, no meta device)
 382 |     model_config_kwargs = dict(
 383 |         sequence_len=args.max_seq_len,
 384 |         vocab_size=vocab_size,
 385 |         n_layer=num_layers,
 386 |         n_head=num_heads,
 387 |         n_kv_head=num_kv_heads,
 388 |         n_embd=model_dim,
 389 |         window_pattern=args.window_pattern,
 390 |         engram_enabled=args.engram,
 391 |         engram_layers=args.engram_layers,
 392 |         engram_ngram_orders=args.engram_ngram_orders,
 393 |         engram_bottleneck_dim=args.engram_bottleneck_dim,
 394 |         engram_dropout=args.engram_dropout,
 395 |         mhc_enabled=args.mhc,
 396 |         mhc_num_branches=args.mhc_num_branches,
 397 |         mhc_sinkhorn_iters=args.mhc_sinkhorn_iters,
 398 |         mhc_temperature=args.mhc_temperature,
 399 |         mhc_epsilon=args.mhc_epsilon,
 400 |         mhc_blend_alpha=args.mhc_blend_alpha,
 401 |         mtp_enabled=args.mtp,
 402 |         mtp_lambda=args.mtp_lambda,
 403 |         dsa_enabled=args.dsa,
 404 |         dsa_start_layer=args.dsa_start_layer,
 405 |         dsa_top_k_ratio=args.dsa_top_k_ratio,
 406 |         dsa_local_window=args.dsa_local_window,
 407 |         dsa_indexer_heads=args.dsa_indexer_heads,
 408 |         dsa_indexer_dim=args.dsa_indexer_dim,
 409 |         aux_loss_weight=args.aux_loss_weight,
 410 |         gradient_checkpointing=args.gradient_checkpointing,
 411 |     )
 412 |     model_config = _build_gpt_config(model_config_kwargs)
 413 |     model = GPT(model_config)
 414 |     model.to(device)  # Move model to GPU before init_weights (rotary embeddings need correct device)
 415 |     model.init_weights()
 416 | 
 417 |     # ---- Tensor parallelism: shard model weights across 'model' mesh axis ----
 418 |     if spmd_mesh is not None and tp_degree > 1:
 419 |         _apply_tensor_parallel_sharding(model, spmd_mesh)
 420 | 
 421 |     # When using TE precision (NVFP4/FP8), convert model to bfloat16 for proper mixed precision
 422 |     if precision_plan.use_te:
 423 |         model.to(dtype=torch.bfloat16)
 424 |         print0("Converted model to bfloat16 for TE training")
 425 | 
 426 |     # If we are resuming, overwrite the model parameters with those of the checkpoint
 427 |     base_dir = get_base_dir()
 428 |     output_dirname = args.model_tag if args.model_tag else f"d{args.depth}" # e.g. d12
 429 |     checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
 430 |     # Auto-detect latest checkpoint when resume_from_step == -2
 431 |     if args.resume_from_step == -2:
 432 |         try:
 433 |             args.resume_from_step = find_last_step(checkpoint_dir)
 434 |             print0(f"Auto-detected latest checkpoint: step {args.resume_from_step}")
 435 |         except FileNotFoundError:
 436 |             print0("No checkpoints found, starting from scratch")
 437 |             args.resume_from_step = -1
 438 |     resuming = args.resume_from_step != -1
 439 |     if resuming:
 440 |         print0(f"Resuming optimization from step {args.resume_from_step}")
 441 |         model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
 442 |         # When adding MTP/DSA to an existing model, checkpoint won't have those keys.
 443 |         # Use strict=False and report which new params were initialized from scratch.
 444 |         # On XLA, checkpoint is loaded to CPU; use assign=False to copy into existing XLA params
 445 |         # (assign=True would replace XLA params with CPU tensors).
 446 |         use_assign = device_type != "xla"
 447 |         missing, unexpected = model.load_state_dict(model_data, strict=False, assign=use_assign)
 448 |         if missing:
 449 |             print0(f"New parameters initialized from scratch ({len(missing)} tensors): {', '.join(missing[:10])}{'...' if len(missing) > 10 else ''}")
 450 |         if unexpected:
 451 |             print0(f"WARNING: unexpected keys in checkpoint ({len(unexpected)} tensors): {', '.join(unexpected[:10])}")
 452 |         del model_data # free up this memory after the copy
 453 | 
 454 |     # -----------------------------------------------------------------------------
 455 |     # FP8 training initialization with torchao (must be done before torch.compile)
 456 | 
 457 |     if args.fp8:
 458 |         if device_type != "cuda":
 459 |             print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
 460 |         else:
 461 |             from torchao.float8 import Float8LinearConfig, convert_to_float8_training
 462 |             import torch.nn as nn
 463 | 
 464 |             # Filter: only convert layers with dimensions divisible by 16 (FP8 hardware requirement)
 465 |             def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
 466 |                 if not isinstance(mod, nn.Linear):
 467 |                     return False
 468 |                 # FP8 requires both in_features and out_features divisible by 16
 469 |                 if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
 470 |                     return False
 471 |                 return True
 472 | 
 473 |             fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
 474 |             convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
 475 |             num_fp8_layers = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
 476 |             num_skipped = sum(1 for m in model.modules() if isinstance(m, nn.Linear)) - num_fp8_layers
 477 |             print0(f"FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8_layers} layers, skipped {num_skipped} (dims not divisible by 16)")
 478 | 
 479 |     # Context manager to temporarily disable FP8 for BF16 evaluation
 480 |     @contextmanager
 481 |     def disable_fp8(model):
 482 |         """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation."""
 483 |         import torch.nn as nn
 484 | 
 485 |         # Find all Float8Linear modules and their locations
 486 |         fp8_locations = []  # list of (parent_module, attr_name, fp8_module)
 487 |         for name, module in model.named_modules():
 488 |             if 'Float8' in type(module).__name__:
 489 |                 if '.' in name:
 490 |                     parent_name, attr_name = name.rsplit('.', 1)
 491 |                     parent = model.get_submodule(parent_name)
 492 |                 else:
 493 |                     parent = model
 494 |                     attr_name = name
 495 |                 fp8_locations.append((parent, attr_name, module))
 496 | 
 497 |         if not fp8_locations:
 498 |             yield  # No FP8 modules, nothing to do
 499 |             return
 500 | 
 501 |         # Swap Float8Linear -> nn.Linear (shares the same weight tensor, no copy)
 502 |         for parent, attr_name, fp8_module in fp8_locations:
 503 |             linear = nn.Linear(
 504 |                 fp8_module.in_features,
 505 |                 fp8_module.out_features,
 506 |                 bias=fp8_module.bias is not None,
 507 |                 device=fp8_module.weight.device,
 508 |                 dtype=fp8_module.weight.dtype,
 509 |             )
 510 |             linear.weight = fp8_module.weight  # share, don't copy
 511 |             if fp8_module.bias is not None:
 512 |                 linear.bias = fp8_module.bias
 513 |             setattr(parent, attr_name, linear)
 514 | 
 515 |         try:
 516 |             yield
 517 |         finally:
 518 |             # Restore Float8Linear modules
 519 |             for parent, attr_name, fp8_module in fp8_locations:
 520 |                 setattr(parent, attr_name, fp8_module)
 521 | 
 522 |     # -----------------------------------------------------------------------------
 523 |     # XLA scan_layers optimization: compile 1 transformer block and reuse for all layers
 524 |     # This reduces XLA compilation from ~60min to ~20min for 16-layer models by avoiding
 525 |     # a 3.5M instruction HLO graph.
 526 |     if args.use_scan and device_type == "xla":
 527 |         try:
 528 |             # torch_xla >= 2.10: scan_layers wraps nn.ModuleList directly
 529 |             from torch_xla.experimental.scan import scan_layers
 530 |             model.transformer.h = scan_layers(model.transformer.h)
 531 |             print0(f"Enabled XLA scan_layers for faster compilation ({num_layers} layers -> 1 compiled block)")
 532 |         except (ImportError, AttributeError):
 533 |             # torch_xla 2.9.x: scan_layers not available, lower-level scan() exists
 534 |             # but requires manual wrapper. Skip for now.
 535 |             print0("Warning: --use_scan requires torch_xla >= 2.10 (scan_layers not found in this version), ignoring")
 536 |     elif args.use_scan:
 537 |         print0("Warning: --use_scan is only effective on XLA/TPU devices, ignoring")
 538 | 
 539 |     # -----------------------------------------------------------------------------
 540 |     # Compile the model
 541 | 
 542 |     orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
 543 |     if device_type == "xla":
 544 |         # Skip torch.compile for TPU - XLA JIT compiler handles optimization
 545 |         # torch.compile with openxla backend can cause OOM during compilation
 546 |         print0("Using eager mode for TPU (XLA JIT handles optimization)")
 547 |         # model stays uncompiled, XLA will trace and compile lazily
 548 |     elif args.no_compile:
 549 |         print0("Using eager mode (--no_compile flag set)")
 550 |         # model stays uncompiled
 551 |     else:
 552 |         model = torch.compile(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe
 553 |     num_params = sum(p.numel() for p in model.parameters())
 554 |     num_scaling_params = orig_model.num_scaling_params()
 555 |     print0(f"Number of parameters: {num_params:,} (scaling: {num_scaling_params:,})")
 556 |     num_flops_per_token = model.estimate_flops()
 557 |     print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")
 558 | 
 559 |     # Calculate number of iterations. Either it is given, or from target flops, or from target data:param ratio (in that order)
 560 |     assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
 561 |     if args.num_iterations > 0:
 562 |         num_iterations = args.num_iterations
 563 |         print0(f"Using user-provided number of iterations: {num_iterations:,}")
 564 |     elif args.target_flops > 0:
 565 |         # calculate the number of iterations from the target flops
 566 |         num_iterations = round(args.target_flops / (num_flops_per_token * args.total_batch_size))
 567 |         print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
 568 |     elif args.target_param_data_ratio > 0:
 569 |         # calculate the number of iterations from the target param data ratio (use scaling params per Kaplan et al.)
 570 |         target_tokens = args.target_param_data_ratio * num_scaling_params
 571 |         num_iterations = target_tokens // args.total_batch_size
 572 |         print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
 573 |     else:
 574 |         raise ValueError("No training horizon specified")
 575 |     total_tokens = args.total_batch_size * num_iterations
 576 |     print0(f"Total number of training tokens: {total_tokens:,}")
 577 |     print0(f"Tokens : Params ratio: {args.total_batch_size * num_iterations / num_scaling_params:.2f}") # Chinchilla is ~20
 578 |     print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")
 579 | 
 580 |     # -----------------------------------------------------------------------------
 581 |     # Initialize the Optimizer (Muon for Linear layers, AdamW for embedding and lm_head)
 582 |     adam_betas = (args.adam_beta1, args.adam_beta2)
 583 |     optimizers = model.setup_optimizers(
 584 |         unembedding_lr=args.unembedding_lr * batch_lr_scale,
 585 |         embedding_lr=args.embedding_lr * batch_lr_scale,
 586 |         matrix_lr=args.matrix_lr * batch_lr_scale,
 587 |         weight_decay=weight_decay_scaled,
 588 |         adam_betas=adam_betas,
 589 |         scalar_lr=args.scalar_lr * batch_lr_scale,
 590 |     )
 591 |     adamw_optimizer, muon_optimizer = optimizers
 592 | 
 593 |     if resuming:
 594 |         for opt, dat in zip(optimizers, optimizer_data):
 595 |             if device_type == "xla":
 596 |                 # On XLA, checkpoint was loaded to CPU. Move optimizer state tensors
 597 |                 # to XLA one parameter at a time to avoid doubling HBM usage.
 598 |                 for param_state in dat.get("state", {}).values():
 599 |                     for k, v in param_state.items():
 600 |                         if isinstance(v, torch.Tensor):
 601 |                             param_state[k] = v.to(device)
 602 |             opt.load_state_dict(dat)
 603 |         del optimizer_data # free up the memory
 604 | 
 605 |     # -----------------------------------------------------------------------------
 606 |     # Initialize the DataLoaders for train/val
 607 |     tokens_dir = os.path.join(base_dir, "tokenized_data")
 608 |     dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
 609 |     fim_rate = args.fim_rate
 610 |     structured_fim_rate = args.structured_fim_rate
 611 |     structured_fim_path = os.path.join(os.path.dirname(__file__), '..', args.structured_fim_path)
 612 |     if fim_rate > 0 or structured_fim_rate > 0:
 613 |         print0(f"FIM enabled: fim_rate={fim_rate}, structured_fim_rate={structured_fim_rate}")
 614 | 
 615 |     # SPMD: single process loads full batches, then shards across devices.
 616 |     # Without SPMD: each process loads per-chip batches (DDP sharding in dataloader).
 617 |     dataloader_B = args.device_batch_size
 618 |     if spmd_mesh is not None:
 619 |         dataloader_B = args.device_batch_size * ddp_world_size
 620 |         print0(f"SPMD dataloader batch size: {dataloader_B} (= {args.device_batch_size} per chip x {ddp_world_size} chips)")
 621 | 
 622 |     data_dir = args.data_dir if args.data_dir else None
 623 |     train_loader = tokenizing_distributed_data_loader_with_state(
 624 |         tokenizer, dataloader_B, args.max_seq_len, split="train",
 625 |         device=device, resume_state_dict=dataloader_resume_state_dict,
 626 |         fim_rate=fim_rate, structured_fim_rate=structured_fim_rate,
 627 |         structured_fim_path=structured_fim_path if structured_fim_rate > 0 else None,
 628 |         data_dir=data_dir, streaming=args.streaming_data
 629 |     )
 630 |     def build_val_loader():
 631 |         loader = tokenizing_distributed_data_loader(tokenizer, dataloader_B, args.max_seq_len, split="val", device=device,
 632 |                                                      data_dir=data_dir)
 633 |         if spmd_mesh is None:
 634 |             return loader
 635 |         # Wrap val loader to shard each batch for SPMD
 636 |         def _sharded():
 637 |             for x, y in loader:
 638 |                 yield shard_data(x, y)
 639 |         return _sharded()
 640 | 
 641 |     # SPMD helper: annotate tensors for data-parallel sharding across TPU chips
 642 |     def shard_data(x, y):
 643 |         """Mark batch dimension as sharded across SPMD mesh devices."""
 644 |         if spmd_mesh is not None:
 645 |             xs.mark_sharding(x, spmd_mesh, ('data', None))
 646 |             xs.mark_sharding(y, spmd_mesh, ('data', None))
 647 |         return x, y
 648 | 
 649 |     x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data
 650 |     x, y = shard_data(x, y)
 651 | 
 652 |     # Pre-scan check: warn if data appears to be non-C++ (no #include, no semicolons)
 653 |     def check_cpp_data(x_batch, tokenizer, num_samples=4):
 654 |         """Check if the first batch contains C++ code markers."""
 655 |         cpp_markers = 0
 656 |         total_checked = min(num_samples, x_batch.size(0))
 657 |         for i in range(total_checked):
 658 |             # Decode a sample from the batch
 659 |             sample_ids = x_batch[i].tolist()
 660 |             sample_text = tokenizer.decode(sample_ids[:512])  # Check first 512 tokens
 661 |             # Look for C++ markers
 662 |             if '#include' in sample_text or '::' in sample_text:
 663 |                 cpp_markers += 1
 664 |             elif sample_text.count(';') >= 3:  # At least 3 semicolons suggests C/C++
 665 |                 cpp_markers += 1
 666 |         return cpp_markers, total_checked
 667 | 
 668 |     if is_cpp_tokenizer:
 669 |         cpp_markers, total_checked = check_cpp_data(x, tokenizer)
 670 |         if cpp_markers == 0:
 671 |             print0("=" * 60)
 672 |             print0("WARNING: Data does not appear to contain C++ code!")
 673 |             print0(f"Checked {total_checked} samples: 0 contained #include, ::, or multiple semicolons")
 674 |             print0("If training on non-C++ data, set NANOCHAT_CPP_TOKENIZER=0")
 675 |             print0("=" * 60)
 676 |         else:
 677 |             print0(f"Data check: {cpp_markers}/{total_checked} samples contain C++ markers")
 678 | 
 679 |     # -----------------------------------------------------------------------------
 680 |     # Set up hyperparameter schedulers
 681 | 
 682 |     # Learning rate scheduler
 683 |     def get_lr_multiplier(it):
 684 |         warmup_iters = round(args.warmup_ratio * num_iterations)
 685 |         warmdown_iters = round(args.warmdown_ratio * num_iterations)
 686 |         if it < warmup_iters:
 687 |             return (it + 1) / warmup_iters
 688 |         elif it <= num_iterations - warmdown_iters:
 689 |             return 1.0
 690 |         else:
 691 |             progress = (num_iterations - it) / warmdown_iters
 692 |             return progress * 1.0 + (1 - progress) * args.final_lr_frac
 693 | 
 694 |     # Momentum scheduler for Muon optimizer
 695 |     def get_muon_momentum(it):
 696 |         frac = min(it / 300, 1)
 697 |         momentum = (1 - frac) * 0.85 + frac * 0.95
 698 |         return momentum
 699 | 
 700 |     # Weight decay scheduler for Muon optimizer (linear to zero over the course of training)
 701 |     def get_weight_decay(it):
 702 |         return weight_decay_scaled * (1 - it / num_iterations)
 703 | 
 704 |     # -----------------------------------------------------------------------------
 705 |     # Loop state (variables updated by the training loop)
 706 | 
 707 |     if not resuming:
 708 |         step = 0
 709 |         val_bpb = None # will be set if eval_every > 0
 710 |         min_val_bpb = float("inf")
 711 |         smooth_train_loss = 0 # EMA of training loss
 712 |         total_training_time = 0 # total wall-clock time of training
 713 |     else:
 714 |         step = meta_data["step"]
 715 |         loop_state = meta_data["loop_state"]
 716 |         val_bpb = meta_data["val_bpb"]
 717 |         min_val_bpb = loop_state["min_val_bpb"]
 718 |         smooth_train_loss = loop_state["smooth_train_loss"]
 719 |         total_training_time = loop_state["total_training_time"]
 720 | 
 721 |     # -----------------------------------------------------------------------------
 722 |     # Training loop
 723 |     while True:
 724 |         last_step = step == num_iterations # loop runs num_iterations+1 times so that we can eval/save at the end
 725 |         flops_so_far = num_flops_per_token * args.total_batch_size * step
 726 | 
 727 |         # once in a while: evaluate the val bpb (all ranks participate)
 728 |         if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
 729 |             model.eval()
 730 |             val_loader = build_val_loader()
 731 |             eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
 732 |             with disable_fp8(model), autocast_ctx():
 733 |                 val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes, synchronize=synchronize)
 734 |             print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
 735 |             if val_bpb < min_val_bpb:
 736 |                 min_val_bpb = val_bpb
 737 |             wandb_run.log({
 738 |                 "step": step,
 739 |                 "total_training_flops": flops_so_far,
 740 |                 "total_training_time": total_training_time,
 741 |                 "val/bpb": val_bpb,
 742 |             })
 743 |             model.train()
 744 | 
 745 |         # save checkpoint FIRST (before eval/sample to prevent data loss on eval crash)
 746 |         if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
 747 |             save_checkpoint(
 748 |                 checkpoint_dir,
 749 |                 step,
 750 |                 orig_model.state_dict(), # model parameters
 751 |                 [opt.state_dict() for opt in optimizers], # optimizer states
 752 |                 { # metadata saved as json
 753 |                     "step": step,
 754 |                     "val_bpb": val_bpb, # loss at last step
 755 |                     "model_config": model_config_kwargs,
 756 |                     "user_config": user_config, # inputs to the training script
 757 |                     "device_batch_size": args.device_batch_size,
 758 |                     "max_seq_len": args.max_seq_len,
 759 |                     "dataloader_state_dict": dataloader_state_dict,
 760 |                     "loop_state": { # all loop state (other than step) so that we can resume training
 761 |                         "min_val_bpb": min_val_bpb,
 762 |                         "smooth_train_loss": smooth_train_loss,
 763 |                         "total_training_time": total_training_time,
 764 |                     },
 765 |                 },
 766 |                 rank=ddp_rank,
 767 |             )
 768 | 
 769 |         # once in a while: evaluate C++ code generation quality (master process only)
 770 |         # use the original uncompiled model because the inputs keep changing shape
 771 |         # Disable FP8 for evaluation to use BF16 for more consistent/accurate results
 772 |         results = {}
 773 |         if args.core_metric_every > 0 and master_process and (last_step or (step > 0 and step % args.core_metric_every == 0)):
 774 |             model.eval()
 775 |             try:
 776 |                 with disable_fp8(orig_model), autocast_ctx():
 777 |                     results = evaluate_cpp_model(orig_model, tokenizer, device)
 778 |                 print0(f"Step {step:05d} | C++ compile: {results['cpp_compile_rate']:.1%}, pass: {results['cpp_pass_rate']:.1%}")
 779 |                 wandb_run.log({
 780 |                     "step": step,
 781 |                     "total_training_flops": flops_so_far,
 782 |                     "total_training_time": total_training_time,
 783 |                     "cpp_metric": results["cpp_metric"],
 784 |                     "cpp_compile_rate": results["cpp_compile_rate"],
 785 |                     "cpp_pass_rate": results["cpp_pass_rate"],
 786 |                 })
 787 |             except Exception as e:
 788 |                 print0(f"Step {step:05d} | C++ eval error: {e}")
 789 |             model.train()
 790 | 
 791 |         # once in a while: sample from the model (only on master process)
 792 |         # use the original uncompiled model because the inputs keep changing shape
 793 |         if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
 794 |             model.eval()
 795 |             prompts = [
 796 |                 "#include <vector>\n\n// Return the sum of all elements in a vector.\nint sum(const std::vector<int>& v) {",
 797 |                 "#include <string>\n\n// Convert a string to uppercase.\nstd::string to_upper(const std::string& s) {",
 798 |                 "#include <algorithm>\n#include <vector>\n\n// Remove duplicates from a sorted vector.\nstd::vector<int> remove_duplicates(std::vector<int> v) {",
 799 |                 "// Swap two integers without using a temporary variable.\nvoid swap(int& a, int& b) {",
 800 |                 "#include <cmath>\n\n// Calculate the distance between two 2D points.\ndouble distance(double x1, double y1, double x2, double y2) {",
 801 |             ]
 802 |             try:
 803 |                 engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
 804 |                 for prompt in prompts:
 805 |                     tokens = tokenizer(prompt, prepend="<|bos|>")
 806 |                     with disable_fp8(orig_model), autocast_ctx():
 807 |                         sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=128, temperature=0)
 808 |                     print0(tokenizer.decode(sample[0]))
 809 |             except Exception as e:
 810 |                 print0(f"Step {step:05d} | Sample generation error: {e}")
 811 |             model.train()
 812 | 
 813 |         # termination conditions (TODO: possibly also add loss explosions etc.)
 814 |         if last_step:
 815 |             break
 816 | 
 817 |         # -------------------------------------------------------------------------
 818 |         # single training step
 819 |         # evaluate the gradient
 820 |         synchronize()
 821 |         t0 = time.time()
 822 | 
 823 |         if spmd_mesh is not None:
 824 |             # SPMD compiled training: compile the entire forward+backward+optimizer
 825 |             # as one XLA graph. This reduces mark_step calls from ~11 to 1,
 826 |             # dramatically cutting IR→HLO lowering overhead on multi-chip TPUs
 827 |             # (e.g. v5e-8: 30K→170K tok/sec, v6e-4: 190K→320K tok/sec).
 828 |             import torch_xla
 829 | 
 830 |             # Pre-fetch all micro-batches (data loading must be outside compiled
 831 |             # region to avoid OOM - data creation ops in the graph bloat HLO 4x)
 832 |             all_x = [x]
 833 |             all_y = [y]
 834 |             for _ in range(grad_accum_steps - 1):
 835 |                 xi, yi, dataloader_state_dict = next(train_loader)
 836 |                 xi, yi = shard_data(xi, yi)
 837 |                 all_x.append(xi)
 838 |                 all_y.append(yi)
 839 | 
 840 |             # Set hyperparameters before tracing (Python-level, not in the graph)
 841 |             lrm = get_lr_multiplier(step)
 842 |             for opt in optimizers:
 843 |                 for group in opt.param_groups:
 844 |                     group["lr"] = group["initial_lr"] * lrm
 845 |             muon_momentum = get_muon_momentum(step)
 846 |             muon_weight_decay = get_weight_decay(step)
 847 |             for group in muon_optimizer.param_groups:
 848 |                 group["momentum"] = muon_momentum
 849 |                 group["weight_decay"] = muon_weight_decay
 850 | 
 851 |             # Compile entire training step as single graph (1 mark_step on exit)
 852 |             with torch_xla.compile():
 853 |                 for i in range(grad_accum_steps):
 854 |                     with autocast_ctx():
 855 |                         loss = model(all_x[i], all_y[i])
 856 |                     train_loss = loss.detach()
 857 |                     loss = loss / grad_accum_steps
 858 |                     loss.backward()
 859 |                 xla_all_reduce_gradients(model, ddp_world_size)
 860 |                 for opt in optimizers:
 861 |                     opt.step()
 862 |                 model.zero_grad(set_to_none=True)
 863 | 
 864 |             # Prefetch first batch for next training step
 865 |             x, y, dataloader_state_dict = next(train_loader)
 866 |             x, y = shard_data(x, y)
 867 |         else:
 868 |             # Original path: per-micro-step synchronization (GPU/single-chip TPU)
 869 |             for micro_step in range(grad_accum_steps):
 870 |                 with autocast_ctx():
 871 |                     loss = model(x, y)
 872 |                 train_loss = loss.detach() # for logging
 873 |                 loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
 874 |                 loss.backward()
 875 |                 if device_type == "xla":
 876 |                     synchronize()  # XLA: break graph at each micro-step to keep HLO compilation fast
 877 |                 x, y, dataloader_state_dict = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
 878 |                 x, y = shard_data(x, y)
 879 |             # step the optimizers
 880 |             lrm = get_lr_multiplier(step)
 881 |             for opt in optimizers:
 882 |                 for group in opt.param_groups:
 883 |                     group["lr"] = group["initial_lr"] * lrm
 884 |             if device_type == "xla" and ddp_world_size > 1:
 885 |                 xla_all_reduce_gradients(model, ddp_world_size)
 886 |             muon_momentum = get_muon_momentum(step)
 887 |             muon_weight_decay = get_weight_decay(step)
 888 |             for group in muon_optimizer.param_groups:
 889 |                 group["momentum"] = muon_momentum
 890 |                 group["weight_decay"] = muon_weight_decay
 891 |             for opt in optimizers:
 892 |                 opt.step()
 893 |             if device_type == "xla":
 894 |                 synchronize()  # XLA: break graph between optimizer step and zero_grad
 895 |             model.zero_grad(set_to_none=True)
 896 |             synchronize()
 897 | 
 898 |         t1 = time.time()
 899 |         dt = t1 - t0
 900 |         # -------------------------------------------------------------------------
 901 | 
 902 |         # logging
 903 |         ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
 904 |         smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item() # EMA the training loss
 905 |         debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
 906 |         pct_done = 100 * step / num_iterations
 907 |         tok_per_sec = int(args.total_batch_size / dt)
 908 |         flops_per_sec = num_flops_per_token * args.total_batch_size / dt
 909 |         # Theoretical peak FLOPS for MFU calculation (dense, no 2:4 sparsity)
 910 |         # H100 SXM BF16: 989 TFLOPS sparse, ~495 TFLOPS dense
 911 |         # GB10 NVFP4: 1000 TFLOPS sparse, ~500 TFLOPS dense; BF16: ~62 TFLOPS
 912 |         # We use dense numbers since nanochat doesn't use 2:4 structured sparsity
 913 |         if device_type == "xla":
 914 |             # Detect TPU type from metadata when available.
 915 |             tpu_type = get_tpu_accelerator_type().lower()
 916 |             if "v5" in tpu_type:
 917 |                 gpu_name = "TPU v5e"
 918 |                 promised_flops = 197e12
 919 |             elif "v6" in tpu_type:
 920 |                 gpu_name = "TPU v6e"
 921 |                 promised_flops = 918e12
 922 |             else:
 923 |                 gpu_name = "TPU"
 924 |                 promised_flops = 197e12
 925 |         elif torch.cuda.is_available():
 926 |             gpu_name = torch.cuda.get_device_name(0)
 927 |             if "GB10" in gpu_name:
 928 |                 # GB10: use NVFP4 peak if using TE precision, else BF16
 929 |                 promised_flops = 500e12 if precision_plan.use_te else 62e12
 930 |             else:
 931 |                 # Default to H100 dense BF16
 932 |                 promised_flops = 495e12
 933 |         else:
 934 |             gpu_name = "CPU"
 935 |             promised_flops = 1e12  # Placeholder for CPU
 936 |         promised_flops_total = promised_flops * max(ddp_world_size, 1)
 937 |         mfu = 100 * flops_per_sec / promised_flops_total # in %
 938 |         if step > 10:
 939 |             total_training_time += dt # only count the time after the first 10 steps
 940 |         # Calculate ETA based on average time per step (excluding first 10 steps)
 941 |         steps_done = step - 10
 942 |         if steps_done > 0:
 943 |             avg_time_per_step = total_training_time / steps_done
 944 |             remaining_steps = num_iterations - step
 945 |             eta_seconds = remaining_steps * avg_time_per_step
 946 |             eta_str = f" | eta: {eta_seconds/60:.1f}m"
 947 |         else:
 948 |             eta_str = ""
 949 |         print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | total time: {total_training_time/60:.2f}m{eta_str}")
 950 |         if step % 100 == 0:
 951 |             log_data = {
 952 |                 "step": step,
 953 |                 "total_training_flops": flops_so_far,
 954 |                 "total_training_time": total_training_time,
 955 |                 "train/loss": debiased_smooth_loss,
 956 |                 "train/lrm": lrm,
 957 |                 "train/dt": dt,
 958 |                 "train/tok_per_sec": tok_per_sec,
 959 |                 "train/mfu": mfu,
 960 |             }
 961 |             wandb_run.log(log_data)
 962 | 
 963 |         # state update
 964 |         first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
 965 |         step += 1
 966 | 
 967 |         # The garbage collector is sadly a little bit overactive and for some poorly understood reason,
 968 |         # it spends ~500ms scanning for cycles quite frequently, just to end up cleaning up very few tiny objects each time.
 969 |         # So we manually manage and help it out here (from upstream karpathy/nanochat)
 970 |         if first_step_of_run:
 971 |             gc.collect()  # manually collect a lot of garbage from setup
 972 |             gc.freeze()   # immediately freeze all currently surviving objects and exclude them from GC
 973 |             gc.disable()  # nuclear intervention: disable GC entirely except:
 974 |         elif step % 5000 == 0:  # every 5000 steps...
 975 |             gc.collect()  # manually collect, just to be safe for very, very long runs
 976 | 
 977 |     # print a few more stats
 978 |     print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
 979 |     print0(f"Total training time: {total_training_time/60:.2f}m")
 980 |     if val_bpb is not None:
 981 |         print0(f"Minimum validation bpb: {min_val_bpb:.6f}")
 982 | 
 983 |     # Log to report
 984 |     from nanochat.report import get_report
 985 |     get_report().log(section="Base model training", data=[
 986 |         user_config, # CLI args
 987 |         { # stats about the training setup
 988 |             "Number of parameters": num_params,
 989 |             "Number of FLOPs per token": f"{num_flops_per_token:e}",
 990 |             "Calculated number of iterations": num_iterations,
 991 |             "Number of training tokens": total_tokens,
 992 |             "Tokens : Params ratio": args.total_batch_size * num_iterations / num_params,
 993 |             "DDP world size": ddp_world_size,
 994 |             "warmup_ratio": args.warmup_ratio,
 995 |             "warmdown_ratio": args.warmdown_ratio,
 996 |             "final_lr_frac": args.final_lr_frac,
 997 |         },
 998 |         { # stats about training outcomes
 999 |             "Minimum validation bpb": min_val_bpb if val_bpb is not None else None,
1000 |             "Final validation bpb": val_bpb,
1001 |             "CORE metric estimate": results.get("core_metric", None),
1002 |             "MFU %": f"{mfu:.2f}%",
1003 |             "Total training flops": f"{flops_so_far:e}",
1004 |             "Total training time": f"{total_training_time/60:.2f}m",
1005 |             "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
1006 |         }
1007 |     ])
1008 | 
1009 |     # cleanup
1010 |     wandb_run.finish() # wandb run finish
1011 |     compute_cleanup()
1012 | 
1013 | 
1014 | # =============================================================================
1015 | # Entry point: SPMD for multi-chip TPU, or direct call for GPU/CPU
1016 | # =============================================================================
1017 | 
1018 | def main():
1019 |     if _is_tpu_requested():
1020 |         from nanochat.common import get_tpu_num_chips
1021 |         num_chips = get_tpu_num_chips()
1022 |         if num_chips > 1:
1023 |             # Enable SPMD BEFORE any XLA runtime init (xm.xla_device(),
1024 |             # xr.global_runtime_device_count(), etc.). SPMD uses a single
1025 |             # process for all TPU chips, eliminating the ~2s/mark_step
1026 |             # overhead of the multi-process xmp.spawn approach.
1027 |             import torch_xla.runtime as xr
1028 |             xr.use_spmd()
1029 |             print(f"SPMD enabled for {num_chips} TPU chips")
1030 |     train()
1031 | 
1032 | 
1033 | if __name__ == "__main__":
1034 |     main()

```

`nanochat/scripts/chat_cli.py`:

```py
   1 | """
   2 | New and upgraded chat mode because a lot of the code has changed since the last one.
   3 | 
   4 | Intended to be run single GPU only atm:
   5 | python -m scripts.chat_cli -i mid
   6 | """
   7 | import argparse
   8 | import torch
   9 | from nanochat.common import compute_init, autodetect_device_type
  10 | from contextlib import nullcontext
  11 | from nanochat.engine import Engine
  12 | from nanochat.checkpoint_manager import load_model
  13 | 
  14 | parser = argparse.ArgumentParser(description='Chat with the model')
  15 | parser.add_argument('-i', '--source', type=str, default="sft", help="Source of the model: sft|mid|rl")
  16 | parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
  17 | parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
  18 | parser.add_argument('-p', '--prompt', type=str, default='', help='Prompt the model, get a single response back')
  19 | parser.add_argument('-t', '--temperature', type=float, default=0.6, help='Temperature for generation')
  20 | parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling parameter')
  21 | parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
  22 | parser.add_argument('-d', '--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'])
  23 | args = parser.parse_args()
  24 | 
  25 | # Init the model and tokenizer
  26 | 
  27 | device_type = autodetect_device_type() if args.device_type == "" else args.device_type
  28 | ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
  29 | ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
  30 | autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
  31 | model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
  32 | 
  33 | # Special tokens for the chat state machine
  34 | bos = tokenizer.get_bos_token_id()
  35 | user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
  36 | assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")
  37 | 
  38 | # Create Engine for efficient generation
  39 | engine = Engine(model, tokenizer)
  40 | 
  41 | print("\nNanoChat Interactive Mode")
  42 | print("-" * 50)
  43 | print("Type 'quit' or 'exit' to end the conversation")
  44 | print("Type 'clear' to start a new conversation")
  45 | print("-" * 50)
  46 | 
  47 | conversation_tokens = [bos]
  48 | 
  49 | while True:
  50 | 
  51 |     if args.prompt:
  52 |         # Get the prompt from the launch command
  53 |         user_input = args.prompt
  54 |     else:
  55 |         # Get the prompt interactively from the console
  56 |         try:
  57 |             user_input = input("\nUser: ").strip()
  58 |         except (EOFError, KeyboardInterrupt):
  59 |             print("\nGoodbye!")
  60 |             break
  61 | 
  62 |     # Handle special commands
  63 |     if user_input.lower() in ['quit', 'exit']:
  64 |         print("Goodbye!")
  65 |         break
  66 | 
  67 |     if user_input.lower() == 'clear':
  68 |         conversation_tokens = [bos]
  69 |         print("Conversation cleared.")
  70 |         continue
  71 | 
  72 |     if not user_input:
  73 |         continue
  74 | 
  75 |     # Add User message to the conversation
  76 |     conversation_tokens.append(user_start)
  77 |     conversation_tokens.extend(tokenizer.encode(user_input))
  78 |     conversation_tokens.append(user_end)
  79 | 
  80 |     # Kick off the assistant
  81 |     conversation_tokens.append(assistant_start)
  82 |     generate_kwargs = {
  83 |         "num_samples": 1,
  84 |         "max_tokens": 256,
  85 |         "temperature": args.temperature,
  86 |         "top_k": args.top_k,
  87 |     }
  88 |     response_tokens = []
  89 |     print("\nAssistant: ", end="", flush=True)
  90 |     with autocast_ctx:
  91 |         for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
  92 |             token = token_column[0] # pop the batch dimension (num_samples=1)
  93 |             response_tokens.append(token)
  94 |             token_text = tokenizer.decode([token])
  95 |             print(token_text, end="", flush=True)
  96 |     print()
  97 |     # we have to ensure that the assistant end token is the last token
  98 |     # so even if generation ends due to max tokens, we have to append it to the end
  99 |     if response_tokens[-1] != assistant_end:
 100 |         response_tokens.append(assistant_end)
 101 |     conversation_tokens.extend(response_tokens)
 102 | 
 103 |     # In the prompt mode, we only want a single response and exit
 104 |     if args.prompt:
 105 |         break

```

`nanochat/tools/clang_indexer/index_project.py`:

```py
   1 | #!/usr/bin/env python3
   2 | """
   3 | Clang-based cross-file dependency indexer for C++ training data preparation.
   4 | 
   5 | Uses libclang to parse C++ translation units with full semantic analysis,
   6 | building a cross-file call graph and generating bottom-up training documents.
   7 | 
   8 | Architecture:
   9 |   1. Walk project directory, find all .cpp/.cc/.cxx/.c files
  10 |   2. Parse each with libclang (optionally using compile_commands.json)
  11 |   3. Extract functions, classes, and cross-file call references
  12 |   4. Build global call graph
  13 |   5. Topological sort: HAL/system → drivers → subsystems → API
  14 |   6. Generate 16K-token training documents with bottom-up dependency ordering
  15 | 
  16 | Usage:
  17 |   # With compile_commands.json (best quality):
  18 |   python index_project.py --project-dir /path/to/project --output chunks.jsonl
  19 | 
  20 |   # Without build system (fallback mode):
  21 |   python index_project.py --project-dir /path/to/project --output chunks.jsonl --no-compile-db
  22 | 
  23 |   # Process multiple projects in parallel:
  24 |   python index_project.py --projects-list projects.txt --output chunks.jsonl --workers 48
  25 | """
  26 | 
  27 | import argparse
  28 | import json
  29 | import os
  30 | import sys
  31 | import hashlib
  32 | 
  33 | # Increase recursion limit for deeply nested ASTs (gcc-mirror, llvm-project, boost)
  34 | sys.setrecursionlimit(50000)
  35 | from collections import defaultdict, deque
  36 | from concurrent.futures import ProcessPoolExecutor, as_completed
  37 | from pathlib import Path
  38 | from typing import Optional
  39 | 
  40 | try:
  41 |     from clang.cindex import (
  42 |         Index, TranslationUnit, CursorKind, Cursor,
  43 |         Config as ClangConfig
  44 |     )
  45 | except ImportError:
  46 |     print("ERROR: libclang Python bindings not found.", file=sys.stderr)
  47 |     print("Install with: pip install libclang", file=sys.stderr)
  48 |     print("Or: sudo apt install python3-clang", file=sys.stderr)
  49 |     sys.exit(1)
  50 | 
  51 | 
  52 | # C++ source file extensions
  53 | CPP_EXTENSIONS = {'.cpp', '.cc', '.cxx', '.c', '.c++', '.cp'}
  54 | HEADER_EXTENSIONS = {'.h', '.hpp', '.hxx', '.hh', '.h++', '.inl', '.inc'}
  55 | 
  56 | # System/stdlib function prefixes (skip for dependency tracking)
  57 | SYSTEM_PREFIXES = (
  58 |     'std::', 'boost::', '__builtin', '__', 'operator', 'printf', 'fprintf',
  59 |     'sprintf', 'snprintf', 'scanf', 'malloc', 'calloc', 'realloc', 'free',
  60 |     'memcpy', 'memmove', 'memset', 'memcmp', 'strlen', 'strcpy', 'strcat',
  61 |     'strcmp', 'fopen', 'fclose', 'fread', 'fwrite', 'exit', 'abort',
  62 |     'assert', 'pthread_', 'EXPECT_', 'ASSERT_', 'TEST',
  63 | )
  64 | 
  65 | 
  66 | class FunctionDef:
  67 |     """A function definition with its source location and call references."""
  68 |     __slots__ = ['name', 'qualified_name', 'file', 'line', 'text', 'callees',
  69 |                  'dep_level', 'is_definition']
  70 | 
  71 |     def __init__(self, name: str, qualified_name: str, file: str, line: int,
  72 |                  text: str, callees: list, is_definition: bool = True):
  73 |         self.name = name
  74 |         self.qualified_name = qualified_name
  75 |         self.file = file
  76 |         self.line = line
  77 |         self.text = text
  78 |         self.callees = callees  # list of qualified names called
  79 |         self.dep_level = 0
  80 |         self.is_definition = is_definition
  81 | 
  82 |     def to_dict(self) -> dict:
  83 |         """Serialize for multiprocessing IPC."""
  84 |         return {
  85 |             'name': self.name, 'qualified_name': self.qualified_name,
  86 |             'file': self.file, 'line': self.line, 'text': self.text,
  87 |             'callees': self.callees, 'is_definition': self.is_definition,
  88 |         }
  89 | 
  90 |     @classmethod
  91 |     def from_dict(cls, d: dict) -> 'FunctionDef':
  92 |         return cls(**d)
  93 | 
  94 | 
  95 | class ProjectIndex:
  96 |     """Cross-file function index for a single project."""
  97 | 
  98 |     def __init__(self):
  99 |         # qualified_name -> FunctionDef (definitions only)
 100 |         self.functions: dict[str, FunctionDef] = {}
 101 |         # file -> list of function qualified_names defined there
 102 |         self.file_functions: dict[str, list[str]] = defaultdict(list)
 103 |         # file -> preamble text (includes, typedefs, forward decls)
 104 |         self.file_preambles: dict[str, str] = {}
 105 |         # qualified_name -> list of qualified_names that call it
 106 |         self.callers: dict[str, list[str]] = defaultdict(list)
 107 | 
 108 |     def add_function(self, func: FunctionDef):
 109 |         """Add a function definition to the index."""
 110 |         key = func.qualified_name
 111 |         if key in self.functions and self.functions[key].is_definition:
 112 |             return  # don't overwrite definitions with declarations
 113 |         self.functions[key] = func
 114 |         if func.is_definition:
 115 |             self.file_functions[func.file].append(key)
 116 | 
 117 |     def build_reverse_edges(self):
 118 |         """Build caller -> callee reverse edges for dep level computation."""
 119 |         self.callers.clear()
 120 |         for qname, func in self.functions.items():
 121 |             for callee in func.callees:
 122 |                 if callee in self.functions:
 123 |                     self.callers[callee].append(qname)
 124 | 
 125 |     def compute_dep_levels(self):
 126 |         """Compute dependency levels via BFS from leaves."""
 127 |         # Find leaves: functions with no callees in the index
 128 |         in_degree = {}
 129 |         for qname, func in self.functions.items():
 130 |             local_callees = [c for c in func.callees if c in self.functions and c != qname]
 131 |             in_degree[qname] = len(local_callees)
 132 | 
 133 |         queue = deque()
 134 |         for qname, deg in in_degree.items():
 135 |             if deg == 0:
 136 |                 self.functions[qname].dep_level = 0
 137 |                 queue.append(qname)
 138 | 
 139 |         self.build_reverse_edges()
 140 | 
 141 |         while queue:
 142 |             qname = queue.popleft()
 143 |             level = self.functions[qname].dep_level
 144 |             for caller_name in self.callers.get(qname, []):
 145 |                 new_level = level + 1
 146 |                 if new_level > self.functions[caller_name].dep_level:
 147 |                     self.functions[caller_name].dep_level = new_level
 148 |                 in_degree[caller_name] -= 1
 149 |                 if in_degree[caller_name] == 0:
 150 |                     queue.append(caller_name)
 151 | 
 152 |         # Handle cycles
 153 |         max_level = max((f.dep_level for f in self.functions.values()), default=0)
 154 |         for qname, deg in in_degree.items():
 155 |             if deg > 0:
 156 |                 self.functions[qname].dep_level = max_level + 1
 157 | 
 158 |     def stats(self) -> dict:
 159 |         """Return index statistics."""
 160 |         return {
 161 |             'total_functions': len(self.functions),
 162 |             'total_files': len(self.file_functions),
 163 |             'definitions': sum(1 for f in self.functions.values() if f.is_definition),
 164 |             'max_dep_level': max((f.dep_level for f in self.functions.values()), default=0),
 165 |         }
 166 | 
 167 | 
 168 | def is_system_function(name: str) -> bool:
 169 |     """Check if a function name looks like a system/stdlib function."""
 170 |     return any(name.startswith(p) for p in SYSTEM_PREFIXES)
 171 | 
 172 | 
 173 | def get_function_text(cursor: Cursor, tu: TranslationUnit) -> str:
 174 |     """Extract the source text for a cursor's extent."""
 175 |     extent = cursor.extent
 176 |     start = extent.start
 177 |     end = extent.end
 178 | 
 179 |     try:
 180 |         filename = start.file.name if start.file else None
 181 |         if not filename or not os.path.exists(filename):
 182 |             return ""
 183 | 
 184 |         with open(filename, 'r', errors='replace') as f:
 185 |             content = f.read()
 186 | 
 187 |         # Convert offsets
 188 |         start_offset = start.offset
 189 |         end_offset = end.offset
 190 |         if start_offset < len(content) and end_offset <= len(content):
 191 |             return content[start_offset:end_offset]
 192 |     except Exception:
 193 |         pass
 194 |     return ""
 195 | 
 196 | 
 197 | def extract_callees(cursor: Cursor) -> list[str]:
 198 |     """Extract all function call references from a cursor's children."""
 199 |     callees = set()
 200 | 
 201 |     def walk(node: Cursor):
 202 |         if node.kind == CursorKind.CALL_EXPR:
 203 |             ref = node.referenced
 204 |             if ref and ref.spelling:
 205 |                 # Get fully qualified name
 206 |                 qname = get_qualified_name(ref)
 207 |                 if qname and not is_system_function(qname):
 208 |                     callees.add(qname)
 209 |         for child in node.get_children():
 210 |             walk(child)
 211 | 
 212 |     walk(cursor)
 213 |     return list(callees)
 214 | 
 215 | 
 216 | def get_qualified_name(cursor: Cursor) -> str:
 217 |     """Get the fully qualified name of a cursor (namespace::class::func)."""
 218 |     parts = []
 219 |     c = cursor
 220 |     while c and c.kind != CursorKind.TRANSLATION_UNIT:
 221 |         if c.spelling:
 222 |             parts.append(c.spelling)
 223 |         c = c.semantic_parent
 224 |     parts.reverse()
 225 |     return '::'.join(parts)
 226 | 
 227 | 
 228 | def extract_preamble(tu: TranslationUnit, filename: str) -> str:
 229 |     """Extract #include directives and forward declarations from a file."""
 230 |     preamble_parts = []
 231 |     for cursor in tu.cursor.get_children():
 232 |         if cursor.location.file and cursor.location.file.name != filename:
 233 |             continue
 234 |         if cursor.kind in (CursorKind.INCLUSION_DIRECTIVE,
 235 |                            CursorKind.USING_DIRECTIVE,
 236 |                            CursorKind.USING_DECLARATION,
 237 |                            CursorKind.TYPEDEF_DECL,
 238 |                            CursorKind.TYPE_ALIAS_DECL,
 239 |                            CursorKind.NAMESPACE_ALIAS):
 240 |             text = get_function_text(cursor, tu)
 241 |             if text:
 242 |                 preamble_parts.append(text)
 243 |     return '\n'.join(preamble_parts)
 244 | 
 245 | 
 246 | FUNCTION_KINDS = {
 247 |     CursorKind.FUNCTION_DECL,
 248 |     CursorKind.CXX_METHOD,
 249 |     CursorKind.FUNCTION_TEMPLATE,
 250 |     CursorKind.CONSTRUCTOR,
 251 |     CursorKind.DESTRUCTOR,
 252 |     CursorKind.CONVERSION_FUNCTION,
 253 | }
 254 | 
 255 | CONTAINER_KINDS = {
 256 |     CursorKind.NAMESPACE,
 257 |     CursorKind.CLASS_DECL,
 258 |     CursorKind.STRUCT_DECL,
 259 |     CursorKind.CLASS_TEMPLATE,
 260 |     CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION,
 261 | }
 262 | 
 263 | 
 264 | def parse_translation_unit(
 265 |     filepath: str,
 266 |     index: Index,
 267 |     compile_args: list[str],
 268 |     project_dir: str,
 269 | ) -> list[FunctionDef]:
 270 |     """Parse a single C++ file and extract function definitions with callees."""
 271 |     functions = []
 272 | 
 273 |     try:
 274 |         tu = index.parse(
 275 |             filepath,
 276 |             args=compile_args,
 277 |             options=(
 278 |                 TranslationUnit.PARSE_INCOMPLETE |
 279 |                 TranslationUnit.PARSE_PRECOMPILED_PREAMBLE
 280 |             ),
 281 |         )
 282 |     except Exception as e:
 283 |         print(f"  WARN: Failed to parse {filepath}: {e}", file=sys.stderr)
 284 |         return functions
 285 | 
 286 |     rel_path = os.path.relpath(filepath, project_dir)
 287 | 
 288 |     def visit(cursor):
 289 |         """Recursively visit cursors, descending into namespaces and classes."""
 290 |         if not cursor.location.file:
 291 |             return
 292 |         if cursor.location.file.name != filepath:
 293 |             return
 294 | 
 295 |         if cursor.kind in FUNCTION_KINDS and cursor.is_definition():
 296 |             text = get_function_text(cursor, tu)
 297 |             if text and len(text) >= 20:
 298 |                 callees = extract_callees(cursor)
 299 |                 qname = get_qualified_name(cursor)
 300 |                 functions.append(FunctionDef(
 301 |                     name=cursor.spelling,
 302 |                     qualified_name=qname,
 303 |                     file=rel_path,
 304 |                     line=cursor.location.line,
 305 |                     text=text,
 306 |                     callees=callees,
 307 |                     is_definition=True,
 308 |                 ))
 309 | 
 310 |         elif cursor.kind in CONTAINER_KINDS:
 311 |             # Recurse into namespaces, classes, structs
 312 |             for child in cursor.get_children():
 313 |                 visit(child)
 314 | 
 315 |     for cursor in tu.cursor.get_children():
 316 |         visit(cursor)
 317 | 
 318 |     return functions
 319 | 
 320 | 
 321 | def find_cpp_files(project_dir: str) -> list[str]:
 322 |     """Find all C/C++ source files in a directory."""
 323 |     files = []
 324 |     for root, _, filenames in os.walk(project_dir):
 325 |         # Skip common non-source directories
 326 |         skip_dirs = {'.git', 'build', 'cmake-build', '__pycache__', 'node_modules',
 327 |                      '.vs', '.vscode', 'third_party', 'external', 'deps', 'vendor'}
 328 |         if any(d in root.split(os.sep) for d in skip_dirs):
 329 |             continue
 330 |         for fname in filenames:
 331 |             ext = os.path.splitext(fname)[1].lower()
 332 |             if ext in CPP_EXTENSIONS:
 333 |                 filepath = os.path.join(root, fname)
 334 |                 # Skip very large files
 335 |                 try:
 336 |                     if os.path.getsize(filepath) > 500_000:
 337 |                         continue
 338 |                 except OSError:
 339 |                     continue
 340 |                 files.append(filepath)
 341 |     return files
 342 | 
 343 | 
 344 | def load_compile_commands(project_dir: str) -> Optional[dict]:
 345 |     """Load compile_commands.json if available."""
 346 |     cc_path = os.path.join(project_dir, 'compile_commands.json')
 347 |     if os.path.exists(cc_path):
 348 |         with open(cc_path) as f:
 349 |             commands = json.load(f)
 350 |         # Build file -> args map
 351 |         file_args = {}
 352 |         for entry in commands:
 353 |             filepath = entry.get('file', '')
 354 |             if not os.path.isabs(filepath):
 355 |                 filepath = os.path.join(entry.get('directory', ''), filepath)
 356 |             filepath = os.path.normpath(filepath)
 357 |             cmd = entry.get('command', '') or ' '.join(entry.get('arguments', []))
 358 |             # Extract compiler flags (skip compiler name and file)
 359 |             args = cmd.split()
 360 |             flags = []
 361 |             skip_next = False
 362 |             for arg in args[1:]:  # skip compiler
 363 |                 if skip_next:
 364 |                     skip_next = False
 365 |                     continue
 366 |                 if arg in ('-o', '-MF', '-MQ', '-MT'):
 367 |                     skip_next = True
 368 |                     continue
 369 |                 if arg.startswith('-o') or arg == filepath or arg.endswith('.o'):
 370 |                     continue
 371 |                 if arg in ('-c', '-S'):
 372 |                     continue
 373 |                 flags.append(arg)
 374 |             file_args[filepath] = flags
 375 |         return file_args
 376 |     return None
 377 | 
 378 | 
 379 | def get_default_compile_args(project_dir: str) -> list[str]:
 380 |     """Generate default compile args for projects without compile_commands.json."""
 381 |     include_dirs = set()
 382 |     # Find common include directories
 383 |     for candidate in ['include', 'src', 'lib', 'source', '.']:
 384 |         d = os.path.join(project_dir, candidate)
 385 |         if os.path.isdir(d):
 386 |             include_dirs.add(d)
 387 | 
 388 |     args = [
 389 |         '-std=c++17',
 390 |         '-fsyntax-only',
 391 |         '-Wno-everything',  # suppress all warnings for speed
 392 |         f'-I{project_dir}',
 393 |     ]
 394 |     for d in include_dirs:
 395 |         args.append(f'-I{d}')
 396 | 
 397 |     return args
 398 | 
 399 | 
 400 | def estimate_tokens(text: str) -> int:
 401 |     """Estimate token count (~4 bytes per token for code)."""
 402 |     return max(1, len(text) // 4)
 403 | 
 404 | 
 405 | def collect_transitive_deps(
 406 |     root_qname: str,
 407 |     index: ProjectIndex,
 408 |     max_depth: int = 5,
 409 | ) -> list[str]:
 410 |     """BFS to collect transitive dependencies of a function."""
 411 |     visited = {root_qname}
 412 |     queue = deque([(root_qname, 0)])
 413 |     deps = []
 414 | 
 415 |     while queue:
 416 |         qname, depth = queue.popleft()
 417 |         if depth >= max_depth:
 418 |             continue
 419 |         func = index.functions.get(qname)
 420 |         if not func:
 421 |             continue
 422 |         for callee in func.callees:
 423 |             if callee not in visited and callee in index.functions:
 424 |                 visited.add(callee)
 425 |                 deps.append(callee)
 426 |                 queue.append((callee, depth + 1))
 427 | 
 428 |     return deps
 429 | 
 430 | 
 431 | def build_training_documents(
 432 |     index: ProjectIndex,
 433 |     max_tokens: int = 16384,
 434 |     max_dep_depth: int = 5,
 435 | ) -> list[str]:
 436 |     """Build training documents with bottom-up dependency ordering."""
 437 |     documents = []
 438 |     seen_hashes = set()
 439 | 
 440 |     index.compute_dep_levels()
 441 | 
 442 |     for qname, func in index.functions.items():
 443 |         if not func.is_definition:
 444 |             continue
 445 | 
 446 |         # Collect transitive deps
 447 |         dep_qnames = collect_transitive_deps(qname, index, max_dep_depth)
 448 | 
 449 |         # Sort by dep_level (leaves/most foundational first)
 450 |         dep_funcs = []
 451 |         for dq in dep_qnames:
 452 |             df = index.functions.get(dq)
 453 |             if df and df.is_definition and df.text:
 454 |                 dep_funcs.append(df)
 455 |         dep_funcs.sort(key=lambda f: f.dep_level)
 456 | 
 457 |         # Build document
 458 |         parts = []
 459 | 
 460 |         # Add preamble from root function's file
 461 |         preamble = index.file_preambles.get(func.file, '')
 462 |         if preamble:
 463 |             parts.append(preamble)
 464 | 
 465 |         # Add deps (leaves first = most foundational)
 466 |         for df in dep_funcs:
 467 |             parts.append(df.text)
 468 | 
 469 |         # Add root function last
 470 |         parts.append(func.text)
 471 | 
 472 |         doc = '\n\n'.join(parts)
 473 |         tokens = estimate_tokens(doc)
 474 | 
 475 |         # Token budget management
 476 |         if tokens > max_tokens * 2 and dep_funcs:
 477 |             # Too big: trim deps from highest dep_level first
 478 |             while tokens > max_tokens * 2 and dep_funcs:
 479 |                 dep_funcs.pop()  # remove highest-level dep
 480 |                 parts = []
 481 |                 if preamble:
 482 |                     parts.append(preamble)
 483 |                 for df in dep_funcs:
 484 |                     parts.append(df.text)
 485 |                 parts.append(func.text)
 486 |                 doc = '\n\n'.join(parts)
 487 |                 tokens = estimate_tokens(doc)
 488 | 
 489 |         if tokens < 20:
 490 |             continue
 491 | 
 492 |         # Deduplicate
 493 |         doc_hash = hashlib.md5(doc.encode()).hexdigest()
 494 |         if doc_hash in seen_hashes:
 495 |             continue
 496 |         seen_hashes.add(doc_hash)
 497 |         documents.append(doc)
 498 | 
 499 |     return documents
 500 | 
 501 | 
 502 | def _parse_file_batch(args_tuple):
 503 |     """Worker function for parallel parsing. Each worker creates its own Index."""
 504 |     filepaths, compile_db, default_args, project_dir = args_tuple
 505 |     sys.setrecursionlimit(50000)  # Set in each worker process too
 506 |     clang_index = Index.create()
 507 |     results = []
 508 |     errors = 0
 509 |     for filepath in filepaths:
 510 |         if compile_db and filepath in compile_db:
 511 |             args = compile_db[filepath]
 512 |         else:
 513 |             args = default_args
 514 |         try:
 515 |             functions = parse_translation_unit(filepath, clang_index, args, project_dir)
 516 |             results.extend(f.to_dict() for f in functions)
 517 |         except (Exception, RecursionError):
 518 |             errors += 1
 519 |     return results, len(filepaths), errors
 520 | 
 521 | 
 522 | def process_project(
 523 |     project_dir: str,
 524 |     max_tokens: int = 16384,
 525 |     max_dep_depth: int = 5,
 526 |     parse_workers: int = 1,
 527 | ) -> list[str]:
 528 |     """Process a single project: parse all files, build index, generate docs."""
 529 |     project_dir = os.path.abspath(project_dir)
 530 |     project_name = os.path.basename(project_dir)
 531 | 
 532 |     print(f"\n--- Processing project: {project_name} ---", file=sys.stderr)
 533 | 
 534 |     # Find source files
 535 |     cpp_files = find_cpp_files(project_dir)
 536 |     print(f"  Found {len(cpp_files)} C/C++ source files", file=sys.stderr)
 537 | 
 538 |     if not cpp_files:
 539 |         return []
 540 | 
 541 |     # Load or generate compile commands
 542 |     compile_db = load_compile_commands(project_dir)
 543 |     default_args = get_default_compile_args(project_dir)
 544 | 
 545 |     # Parse all files and build index
 546 |     index_obj = ProjectIndex()
 547 | 
 548 |     # Use parallel parsing for large projects
 549 |     effective_workers = min(parse_workers, max(1, len(cpp_files) // 100))
 550 |     if effective_workers > 1 and len(cpp_files) > 200:
 551 |         print(f"  Using {effective_workers} parse workers", file=sys.stderr)
 552 |         chunk_size = max(50, len(cpp_files) // effective_workers)
 553 |         batches = []
 554 |         for i in range(0, len(cpp_files), chunk_size):
 555 |             batch = cpp_files[i:i + chunk_size]
 556 |             batches.append((batch, compile_db, default_args, project_dir))
 557 | 
 558 |         total_parsed = 0
 559 |         total_errors = 0
 560 |         with ProcessPoolExecutor(max_workers=effective_workers) as executor:
 561 |             for func_dicts, parsed_count, error_count in executor.map(_parse_file_batch, batches):
 562 |                 for d in func_dicts:
 563 |                     index_obj.add_function(FunctionDef.from_dict(d))
 564 |                 total_parsed += parsed_count
 565 |                 total_errors += error_count
 566 |                 print(f"  Parsed {total_parsed}/{len(cpp_files)} files, "
 567 |                       f"{len(index_obj.functions)} functions", file=sys.stderr)
 568 | 
 569 |         print(f"  Parsed {total_parsed} files ({total_errors} errors), "
 570 |               f"{len(index_obj.functions)} functions indexed", file=sys.stderr)
 571 |     else:
 572 |         # Sequential for small projects
 573 |         clang_index = Index.create()
 574 |         parsed = 0
 575 |         errors = 0
 576 |         for filepath in cpp_files:
 577 |             if compile_db and filepath in compile_db:
 578 |                 args = compile_db[filepath]
 579 |             else:
 580 |                 args = default_args
 581 |             try:
 582 |                 functions = parse_translation_unit(filepath, clang_index, args, project_dir)
 583 |                 for func in functions:
 584 |                     index_obj.add_function(func)
 585 |                 parsed += 1
 586 |             except Exception as e:
 587 |                 errors += 1
 588 |                 if errors <= 5:
 589 |                     print(f"  ERROR parsing {filepath}: {e}", file=sys.stderr)
 590 |             if parsed % 500 == 0 and parsed > 0:
 591 |                 print(f"  Parsed {parsed}/{len(cpp_files)} files, "
 592 |                       f"{len(index_obj.functions)} functions", file=sys.stderr)
 593 |         print(f"  Parsed {parsed} files ({errors} errors), "
 594 |               f"{len(index_obj.functions)} functions indexed", file=sys.stderr)
 595 | 
 596 |     # Build training documents
 597 |     documents = build_training_documents(index_obj, max_tokens, max_dep_depth)
 598 |     print(f"  Generated {len(documents)} training documents", file=sys.stderr)
 599 | 
 600 |     stats = index_obj.stats()
 601 |     print(f"  Index stats: {stats}", file=sys.stderr)
 602 | 
 603 |     return documents
 604 | 
 605 | 
 606 | def main():
 607 |     parser = argparse.ArgumentParser(
 608 |         description='Clang-based cross-file C++ dependency indexer')
 609 |     parser.add_argument('--project-dir', type=str,
 610 |                         help='Single project directory to process')
 611 |     parser.add_argument('--projects-list', type=str,
 612 |                         help='File listing project directories (one per line)')
 613 |     parser.add_argument('--projects-dir', type=str,
 614 |                         help='Directory containing multiple project subdirectories')
 615 |     parser.add_argument('--output', type=str, required=True,
 616 |                         help='Output JSONL file path')
 617 |     parser.add_argument('--max-tokens', type=int, default=16384,
 618 |                         help='Max tokens per training document (default: 16384)')
 619 |     parser.add_argument('--max-dep-depth', type=int, default=5,
 620 |                         help='Max dependency resolution depth (default: 5)')
 621 |     parser.add_argument('--workers', type=int, default=1,
 622 |                         help='Number of parallel workers for multi-project mode')
 623 |     parser.add_argument('--parse-workers', type=int, default=8,
 624 |                         help='Number of parallel parse workers within each project (default: 8)')
 625 |     parser.add_argument('--libclang-path', type=str, default=None,
 626 |                         help='Path to libclang.so (auto-detected if not set)')
 627 |     parser.add_argument('--append', action='store_true',
 628 |                         help='Append to output file instead of overwriting')
 629 | 
 630 |     args = parser.parse_args()
 631 | 
 632 |     # Set libclang path if specified
 633 |     if args.libclang_path:
 634 |         ClangConfig.set_library_file(args.libclang_path)
 635 | 
 636 |     # Collect project directories
 637 |     project_dirs = []
 638 |     if args.project_dir:
 639 |         project_dirs.append(args.project_dir)
 640 |     elif args.projects_list:
 641 |         with open(args.projects_list) as f:
 642 |             project_dirs = [line.strip() for line in f if line.strip()]
 643 |     elif args.projects_dir:
 644 |         for entry in sorted(os.listdir(args.projects_dir)):
 645 |             full = os.path.join(args.projects_dir, entry)
 646 |             if os.path.isdir(full):
 647 |                 project_dirs.append(full)
 648 |     else:
 649 |         parser.error("Provide --project-dir, --projects-list, or --projects-dir")
 650 | 
 651 |     print(f"Processing {len(project_dirs)} project(s)", file=sys.stderr)
 652 |     print(f"Output: {args.output}", file=sys.stderr)
 653 |     print(f"Max tokens: {args.max_tokens}", file=sys.stderr)
 654 | 
 655 |     total_docs = 0
 656 |     seen_hashes = set()
 657 | 
 658 |     append_mode = getattr(args, 'append', False)
 659 |     with open(args.output, 'a' if append_mode else 'w') as out:
 660 |         if args.workers > 1 and len(project_dirs) > 1:
 661 |             # Multi-project parallel mode
 662 |             with ProcessPoolExecutor(max_workers=args.workers) as executor:
 663 |                 futures = {
 664 |                     executor.submit(
 665 |                         process_project, pd, args.max_tokens, args.max_dep_depth,
 666 |                         args.parse_workers
 667 |                     ): pd
 668 |                     for pd in project_dirs
 669 |                 }
 670 |                 for future in as_completed(futures):
 671 |                     pd = futures[future]
 672 |                     try:
 673 |                         docs = future.result()
 674 |                         for doc in docs:
 675 |                             doc_hash = hashlib.md5(doc.encode()).hexdigest()
 676 |                             if doc_hash in seen_hashes:
 677 |                                 continue
 678 |                             seen_hashes.add(doc_hash)
 679 |                             json.dump({'text': doc}, out)
 680 |                             out.write('\n')
 681 |                             total_docs += 1
 682 |                     except Exception as e:
 683 |                         print(f"ERROR processing {pd}: {e}", file=sys.stderr)
 684 |         else:
 685 |             # Sequential mode
 686 |             for pd in project_dirs:
 687 |                 try:
 688 |                     docs = process_project(pd, args.max_tokens, args.max_dep_depth,
 689 |                                            args.parse_workers)
 690 |                     for doc in docs:
 691 |                         doc_hash = hashlib.md5(doc.encode()).hexdigest()
 692 |                         if doc_hash in seen_hashes:
 693 |                             continue
 694 |                         seen_hashes.add(doc_hash)
 695 |                         json.dump({'text': doc}, out)
 696 |                         out.write('\n')
 697 |                         total_docs += 1
 698 |                     out.flush()
 699 |                 except Exception as e:
 700 |                     print(f"ERROR processing {pd}: {e}", file=sys.stderr)
 701 |                     print(f"  Skipping project, continuing...", file=sys.stderr)
 702 | 
 703 |     print(f"\n{'='*60}", file=sys.stderr)
 704 |     print(f"Total documents: {total_docs}", file=sys.stderr)
 705 |     print(f"Output: {args.output}", file=sys.stderr)
 706 |     print(f"{'='*60}", file=sys.stderr)
 707 | 
 708 | 
 709 | if __name__ == '__main__':
 710 |     main()

```