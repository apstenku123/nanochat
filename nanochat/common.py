"""
Common utilities for nanochat.
"""

import os
import re
import logging
import urllib.request
import torch
import torch.distributed as dist
from filelock import FileLock

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    def format(self, record):
        # Add color to the level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        # Format the message
        message = super().format(record)
        # Add color to specific parts of the message
        if levelname == 'INFO':
            # Highlight numbers and percentages
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
            message = re.sub(r'(Shard \d+)', rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}', message)
        return message

def setup_default_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler]
    )

setup_default_logging()
logger = logging.getLogger(__name__)

def get_base_dir():
    # co-locate nanochat intermediates with other cached data in ~/.cache (by default)
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanochat")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir

def download_file_with_lock(url, filename, postprocess_fn=None):
    """
    Downloads a file from a URL to a local path in the base directory.
    Uses a lock file to prevent concurrent downloads among multiple ranks.
    """
    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, filename)
    lock_path = file_path + ".lock"

    if os.path.exists(file_path):
        return file_path

    with FileLock(lock_path):
        # Only a single rank can acquire this lock
        # All other ranks block until it is released

        # Recheck after acquiring lock
        if os.path.exists(file_path):
            return file_path

        # Download the content as bytes
        print(f"Downloading {url}...")
        with urllib.request.urlopen(url) as response:
            content = response.read() # bytes

        # Write to local file
        with open(file_path, 'wb') as f:
            f.write(content)
        print(f"Downloaded to {file_path}")

        # Run the postprocess function if provided
        if postprocess_fn is not None:
            postprocess_fn(file_path)

    return file_path

def _is_tpu_requested() -> bool:
    return os.environ.get("PJRT_DEVICE", "").upper() == "TPU"


def _get_xla_dist_info():
    """Best-effort rank/world-size info for TPU PJRT runtimes."""
    try:
        import torch_xla.runtime as xr
        world_size = int(xr.world_size())
        rank = int(xr.global_ordinal())
        local_rank = int(xr.local_ordinal()) if hasattr(xr, "local_ordinal") else rank
        return rank, local_rank, max(world_size, 1)
    except Exception:
        pass

    # Backward compatibility with older torch_xla APIs.
    try:
        import torch_xla.core.xla_model as xm
        world_size = int(xm.xrt_world_size())
        rank = int(xm.get_ordinal())
        local_rank = int(xm.get_local_ordinal()) if hasattr(xm, "get_local_ordinal") else rank
        return rank, local_rank, max(world_size, 1)
    except Exception:
        return 0, 0, 1


def print0(s="",**kwargs):
    _, rank, _, _ = get_dist_info()
    if rank == 0:
        print(s, **kwargs)

def print_banner():
    # Cool DOS Rebel font ASCII banner made with https://manytools.org/hacker-tools/ascii-banner/
    banner = """
                                                       █████                █████
                                                      ░░███                ░░███
     ████████    ██████   ████████    ██████   ██████  ░███████    ██████  ███████
    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███░░░███░
     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████  ░███
     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███  ░███ ███
     ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░███████  ░░█████
    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░   ░░░░░
    """
    print0(banner)

def is_ddp_requested() -> bool:
    """
    True if launched by torchrun (env present), even before init.
    Used to decide whether we *should* initialize a PG.
    """
    return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))

def is_ddp_initialized() -> bool:
    """
    True if torch.distributed is available and the process group is initialized.
    Used at cleanup to avoid destroying a non-existent PG.
    """
    return dist.is_available() and dist.is_initialized()

def get_dist_info():
    if is_ddp_initialized():
        ddp_rank = dist.get_rank()
        ddp_local_rank = int(os.environ.get("LOCAL_RANK", ddp_rank))
        ddp_world_size = dist.get_world_size()
        return True, ddp_rank, ddp_local_rank, ddp_world_size

    if _is_tpu_requested():
        # TPU world size/rank is managed by PJRT runtime, not torch.distributed.
        ddp_rank, ddp_local_rank, ddp_world_size = _get_xla_dist_info()
        return False, ddp_rank, ddp_local_rank, ddp_world_size

    if is_ddp_requested():
        # We rely on torchrun's env to decide if we SHOULD init.
        # (Initialization itself happens in compute init.)
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1

def autodetect_device_type():
    # Check for TPU first (via PJRT_DEVICE env var or torch_xla availability)
    if os.environ.get("PJRT_DEVICE") == "TPU":
        try:
            import torch_xla.core.xla_model as xm
            device_type = "xla"
            print0(f"Autodetected device type: {device_type} (TPU via PJRT_DEVICE)")
            return device_type
        except ImportError:
            print0("Warning: PJRT_DEVICE=TPU but torch_xla not available, falling back...")

    # prefer to use CUDA if available, otherwise use MPS, otherwise fallback on CPU
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    print0(f"Autodetected device type: {device_type}")
    return device_type


def get_tpu_accelerator_type() -> str:
    """Return TPU accelerator type (e.g. v5litepod-8, v6e-4) when available."""
    for key in ("TPU_ACCELERATOR_TYPE", "ACCELERATOR_TYPE"):
        value = os.environ.get(key, "").strip()
        if value:
            return value

    metadata_url = "http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type"
    req = urllib.request.Request(metadata_url, headers={"Metadata-Flavor": "Google"})
    try:
        with urllib.request.urlopen(req, timeout=0.2) as response:
            value = response.read().decode("utf-8").strip()
            if value:
                os.environ["TPU_ACCELERATOR_TYPE"] = value
                return value
    except Exception:
        pass
    return ""


def get_tpu_num_chips() -> int:
    """Detect number of TPU chips on this host from accelerator type string.

    Parses e.g. 'v5litepod-8' -> 8, 'v6e-4' -> 4.  Falls back to 1.
    """
    accel = get_tpu_accelerator_type()
    if accel:
        try:
            return int(accel.rsplit('-', 1)[-1])
        except (ValueError, IndexError):
            pass
    return 1


def xla_all_reduce_gradients(model, world_size: int):
    """
    Average gradients across TPU workers for non-DDP XLA training loops.
    No-op when world_size <= 1 or torch_xla is unavailable.
    """
    if world_size <= 1:
        return
    try:
        import torch_xla.core.xla_model as xm
    except ImportError:
        return
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if grads:
        xm.all_reduce(xm.REDUCE_SUM, grads, scale=1.0 / world_size)


def compute_init(device_type="cuda"): # cuda|cpu|mps|xla
    """Basic initialization that we keep doing over and over, so make common."""

    assert device_type in ["cuda", "mps", "cpu", "xla"], "Invalid device type atm"
    if device_type == "cuda":
        assert torch.cuda.is_available(), "Your PyTorch installation is not configured for CUDA but device_type is 'cuda'"
    if device_type == "mps":
        assert torch.backends.mps.is_available(), "Your PyTorch installation is not configured for MPS but device_type is 'mps'"
    if device_type == "xla":
        try:
            import torch_xla.core.xla_model as xm
        except ImportError:
            raise RuntimeError("device_type is 'xla' but torch_xla is not installed")

    # Reproducibility
    # Note that we set the global seeds here, but most of the code uses explicit rng objects.
    # The only place where global rng might be used is nn.Module initialization of the model weights.
    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed(42)
    # skipping full reproducibility for now, possibly investigate slowdown later
    # torch.use_deterministic_algorithms(True)

    # Precision (torch 2.9+ API)
    if device_type == "cuda":
        torch.backends.fp32_precision = "tf32"  # uses tf32 instead of fp32 for matmuls

    # Distributed setup: Distributed Data Parallel (DDP), optional, and requires CUDA
    ddp_requested = is_ddp_requested()
    if ddp_requested and device_type == "cuda":
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)  # make "cuda" default to this device
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    elif device_type == "xla":
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        # Newer torch_xla prefers runtime.use_spmd() over env-only configuration.
        if os.environ.get("XLA_USE_SPMD") == "1":
            try:
                import torch_xla.runtime as xr
                xr.use_spmd()
            except Exception:
                pass
    else:
        device = torch.device(device_type) # mps|cpu

    is_distributed, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if ddp_rank == 0:
        logger.info(f"Distributed world size: {ddp_world_size}")

    return is_distributed, ddp_rank, ddp_local_rank, ddp_world_size, device

def compute_cleanup():
    """Companion function to compute_init, to clean things up before script exit"""
    if is_ddp_initialized():
        dist.destroy_process_group()

class DummyWandb:
    """Useful if we wish to not use wandb but have all the same signatures"""
    def __init__(self):
        pass
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass
