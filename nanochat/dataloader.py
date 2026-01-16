from collections import deque
import os

import numpy as np
import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info, get_base_dir
from nanochat.dataset import list_parquet_files

def tokenizing_distributed_data_loader_with_state(tokenizer, B, T, split, tokenizer_threads=4, tokenizer_batch_size=128, device="cuda", resume_state_dict=None):
    """
    Stream pretraining text from parquet files, tokenize, yield training batches.

    This implementation became a bit more complex because we wish to support approximate resume training.
    Instead of turning this into a Class, we opt to return the state_dict with every batch,
    and then the caller can pass in a state_dict to resume training from a desired point.
    Note that this resumption is atm only *approximate* for simplicity.
    We won't repeat the same documents but we might skip a few.
    The state_dict that is returned can be later passed into this function via `resume_state_dict` to approximately resume.

    Perfect state resumption is possible but would be a lot more bloated, probably not worth it atm.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    # infinite iterator over document batches (list of text strings)
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    def document_batches():
        parquet_paths = list_parquet_files()
        assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
        parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
        resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
        resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
        first_pass = True
        pq_idx = resume_pq_idx # we kick off parquet files at the resume index (or by default just 0)
        while True: # iterate infinitely (multi-epoch)
            pq_idx = resume_pq_idx if first_pass else 0
            while pq_idx < len(parquet_paths): # iterate over all parquet files
                filepath = parquet_paths[pq_idx]
                pf = pq.ParquetFile(filepath)
                # Start from resume point if resuming on same file, otherwise from DDP rank
                # I know this state resumption is a little bit tricky and a little bit hacky... sigh.
                if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                    base_idx = resume_rg_idx // ddp_world_size # in units of ddp_world_size
                    base_idx += 1 # advance by 1 so that we definitely don't repeat data after resuming
                    rg_idx = base_idx * ddp_world_size + ddp_rank
                    if rg_idx >= pf.num_row_groups:
                        pq_idx += 1
                        continue
                    resume_rg_idx = None # set to None as we only want to do this a single time
                else:
                    rg_idx = ddp_rank
                while rg_idx < pf.num_row_groups:
                    rg = pf.read_row_group(rg_idx)
                    batch = rg.column('text').to_pylist() # each batch is a parquet group, e.g. 1024 rows
                    # the tokenizer encode might want to go in even smaller batches, e.g. 128 rows
                    for i in range(0, len(batch), tokenizer_batch_size):
                        yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx)
                    rg_idx += ddp_world_size # advance to the next row group (in DDP)
                pq_idx += 1 # advance to the next parquet file
            first_pass = False
    batches = document_batches()

    # Now emit batches of tokens.
    needed_tokens = B * T + 1 # +1 is because we also need the target at the last token
    bos_token = tokenizer.get_bos_token_id()
    # scratch buffer holds the tokens for one iteration
    token_buffer = deque() # we stream tokens on the right and pop from the left
    while True:
        # Accumulate enough tokens for one iteration before yielding.
        while len(token_buffer) < needed_tokens:
            doc_batch, (pq_idx, rg_idx) = next(batches)
            token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
            for tokens in token_lists:
                token_buffer.extend(tokens)
        # Move tokens from the deque into the scratch buffer
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        # CUDA supports memory pinning for asynchronous transfers between CPU and GPU
        use_cuda_optimizations = device == "cuda"
        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda_optimizations) # in PyTorch, long=int64
        # Create the inputs/targets as 1D tensors
        inputs_cpu = scratch[:-1]
        targets_cpu = scratch[1:]
        # Reshape to 2D and move to GPU async
        inputs = inputs_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        targets = targets_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx} # we need this in case we wish to approximately resume training
        yield inputs, targets, state_dict

def tokenizing_distributed_data_loader(*args, **kwargs):
    # helper function that only emits the inputs/targets and not the state_dict
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state(*args, **kwargs):
        yield inputs, targets


# =============================================================================
# Fast binary dataloader (for pre-tokenized data)
# =============================================================================

def list_bin_files(data_dir=None):
    """List all .bin files in the tokenized data directory."""
    if data_dir is None:
        data_dir = os.path.join(get_base_dir(), "tokenized_data")
    if not os.path.exists(data_dir):
        return []
    bin_files = sorted([
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if f.endswith('.bin')
    ])
    return bin_files


def load_bin_file(path):
    """Memory-map a binary token file. Returns (memmap, num_tokens)."""
    # Read header to get token count
    with open(path, 'rb') as f:
        header = np.frombuffer(f.read(8), dtype=np.uint64)
        num_tokens = int(header[0])
    # Memory-map the data portion (skip 8-byte header)
    data = np.memmap(path, dtype=np.uint16, mode='r', offset=8, shape=(num_tokens,))
    return data, num_tokens


def binary_distributed_data_loader_with_state(B, T, split, device="cuda", resume_state_dict=None):
    """
    Fast dataloader that reads pre-tokenized binary files.

    This is much faster than tokenizing_distributed_data_loader because:
    1. No tokenization overhead
    2. Memory-mapped files for fast random access
    3. Simple sequential reads with minimal Python overhead

    Args:
        B: batch size (number of sequences per batch)
        T: sequence length
        split: "train" or "val"
        device: target device
        resume_state_dict: optional dict with {"file_idx", "position"} to resume from

    Yields:
        (inputs, targets, state_dict) where inputs/targets are (B, T) tensors
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    # Get all binary files
    bin_files = list_bin_files()
    assert len(bin_files) > 0, "No tokenized .bin files found. Run: python -m scripts.pretokenize"

    # Split: last file is val, rest is train
    # Need at least 2 files for train/val split
    if len(bin_files) == 1:
        if split == "train":
            raise ValueError(
                f"Cannot create train split: only 1 .bin file found, but need at least 2 "
                f"(last file is used for validation). Either add more data shards or use split='val'."
            )
        # For val split with 1 file, use that file
        # bin_files stays as-is
    else:
        bin_files = bin_files[:-1] if split == "train" else bin_files[-1:]

    # Load and concatenate all files (memory-mapped, so this is cheap)
    all_data = []
    file_boundaries = [0]  # cumulative token counts
    for path in bin_files:
        data, num_tokens = load_bin_file(path)
        all_data.append(data)
        file_boundaries.append(file_boundaries[-1] + num_tokens)

    total_tokens = file_boundaries[-1]
    tokens_per_batch = B * T * ddp_world_size  # tokens consumed per iteration across all ranks
    needed_tokens = B * T + 1  # +1 for the target at last position

    # Helper to get tokens from position
    def get_tokens_at(global_pos, count):
        """Get `count` tokens starting at global_pos, wrapping around if needed."""
        tokens = []
        pos = global_pos % total_tokens
        remaining = count

        while remaining > 0:
            # Find which file this position is in
            file_idx = 0
            for i, boundary in enumerate(file_boundaries[1:], 1):
                if pos < boundary:
                    file_idx = i - 1
                    break

            # Get position within file
            file_start = file_boundaries[file_idx]
            file_end = file_boundaries[file_idx + 1]
            pos_in_file = pos - file_start

            # How many tokens can we get from this file?
            available = file_end - pos
            to_take = min(remaining, available)

            # Get the tokens
            tokens.extend(all_data[file_idx][pos_in_file:pos_in_file + to_take])
            remaining -= to_take
            pos = (pos + to_take) % total_tokens

        return tokens

    # Resume state or start fresh
    if resume_state_dict is not None:
        global_pos = resume_state_dict.get("global_pos", 0)
    else:
        global_pos = 0

    # Each rank gets a different slice of the data
    rank_offset = ddp_rank * (B * T)

    while True:
        # Calculate this rank's starting position
        pos = (global_pos + rank_offset) % total_tokens

        # Get tokens for this batch
        tokens = get_tokens_at(pos, needed_tokens)

        # Create tensors
        use_cuda = device == "cuda"
        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda)
        inputs = scratch[:-1].view(B, T).to(device=device, non_blocking=use_cuda)
        targets = scratch[1:].view(B, T).to(device=device, non_blocking=use_cuda)

        state_dict = {"global_pos": global_pos}
        yield inputs, targets, state_dict

        # Advance global position for next iteration
        global_pos = (global_pos + tokens_per_batch) % total_tokens


def binary_distributed_data_loader(*args, **kwargs):
    """Helper that only yields inputs/targets without state_dict."""
    for inputs, targets, state_dict in binary_distributed_data_loader_with_state(*args, **kwargs):
        yield inputs, targets
