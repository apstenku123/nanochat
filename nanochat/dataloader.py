"""
Distributed dataloaders for pretraining.

BOS-aligned bestfit with memory optimization:
   - Every row starts with BOS token
   - Documents packed using best-fit algorithm to minimize cropping
   - Pre-allocated buffers for minimal GC pressure and single HtoD transfer
   - 100% utilization (no padding), ~35% tokens cropped at T=2048
   - Optional FIM (Fill-in-Middle) augmentation support

Fallback to the original streaming approach if needed:
https://github.com/karpathy/nanochat/blob/3c3a3d7/nanochat/dataloader.py#L78-L117
"""

import os
import numpy as np
import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info, get_base_dir
from nanochat.dataset import list_parquet_files
from nanochat.fim import apply_fim_batch, apply_fim_mixed_batch, StructuredFIMDataset

# Global structured FIM dataset (loaded once)
_structured_fim_dataset = None

def get_structured_fim_dataset(tokenizer, pairs_path: str = None):
    """Get or create the structured FIM dataset singleton."""
    global _structured_fim_dataset
    if _structured_fim_dataset is None and pairs_path is not None:
        if os.path.exists(pairs_path):
            _structured_fim_dataset = StructuredFIMDataset(pairs_path, tokenizer)
    return _structured_fim_dataset


def _document_batches(split, resume_state_dict, tokenizer_batch_size):
    """
    Infinite iterator over document batches (list of text strings) from parquet files.

    Handles DDP sharding and approximate resume. Each yield is (text_batch, (pq_idx, rg_idx, epoch))
    where text_batch is a list of document strings, indices track position for resumption,
    and epoch counts how many times we've cycled through the dataset (starts at 1).
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    parquet_paths = list_parquet_files()
    assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    first_pass = True
    pq_idx = resume_pq_idx
    epoch = resume_epoch

    while True:  # iterate infinitely (multi-epoch)
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)
            # Start from resume point if resuming on same file, otherwise from DDP rank
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                base_idx = resume_rg_idx // ddp_world_size
                base_idx += 1  # advance by 1 so we don't repeat data after resuming
                rg_idx = base_idx * ddp_world_size + ddp_rank
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None  # only do this once
            else:
                rg_idx = ddp_rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx, epoch)
                rg_idx += ddp_world_size
            pq_idx += 1
        first_pass = False
        epoch += 1


def tokenizing_distributed_data_loader_with_state(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=1000,
    fim_rate=0.0, spm_rate=0.5, structured_fim_rate=0.0, structured_fim_path=None
):
    """
    BOS-aligned dataloader with Best-Fit Cropping and memory optimization.

    Memory optimizations:
    - Pre-allocated row_buffer, cpu_buffer, gpu_buffer (reused across iterations)
    - Views into buffers instead of creating new tensors each iteration
    - Single HtoD transfer via gpu_buffer.copy_(cpu_buffer)

    Algorithm for each row:
    1. From buffered docs, pick the LARGEST doc that fits entirely
    2. Repeat until no doc fits
    3. When nothing fits, crop a doc to fill remaining space exactly

    Key properties:
    - Every row starts with BOS
    - 100% utilization (no padding, every token is trained on)
    - Approximately 35% of all tokens are discarded due to cropping
    - Optional FIM augmentation support
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    row_capacity = T + 1
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    pq_idx, rg_idx, epoch = 0, 0, 1

    # FIM configuration
    use_fim = (fim_rate > 0 or structured_fim_rate > 0) and split == "train"
    structured_dataset = get_structured_fim_dataset(tokenizer, structured_fim_path) if structured_fim_rate > 0 else None

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)

        # Apply FIM augmentation (per-document, after tokenization)
        if use_fim:
            # FIM is applied to content tokens (skip the prepended BOS)
            content_lists = [toks[1:] for toks in token_lists]  # strip BOS

            if structured_fim_rate > 0 and structured_dataset is not None:
                # Mixed FIM: random + structured
                content_lists = apply_fim_mixed_batch(
                    content_lists,
                    structured_dataset,
                    fim_rate=fim_rate,
                    structured_rate=structured_fim_rate,
                    spm_rate=spm_rate
                )
            elif fim_rate > 0:
                # Random FIM only
                content_lists = apply_fim_batch(content_lists, fim_rate=fim_rate, spm_rate=spm_rate)

            token_lists = [[bos_token] + toks for toks in content_lists]  # re-add BOS

        for tokens in token_lists:
            doc_buffer.append(tokens)

    # Pre-allocate buffers once: layout is [inputs (B*T) | targets (B*T)]
    # This gives us contiguous views and a single HtoD transfer
    use_cuda = device == "cuda"
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)  # for building rows without creating Python lists
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)  # staging area (CPU)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)  # on-device buffer
    cpu_inputs = cpu_buffer[:B * T].view(B, T)  # a few views into these buffers just for convenience
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                # Ensure buffer has documents
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    doc_len = len(doc)
                    row_buffer[row_idx, pos:pos + doc_len] = torch.tensor(doc, dtype=torch.long)
                    pos += doc_len
                else:
                    # No doc fits - crop shortest in buffer to fill remaining and minimize waste
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        # Copy to pinned CPU buffer, then single HtoD transfer
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}

        # Single HtoD copy into persistent GPU buffer and yield
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, state_dict


def tokenizing_distributed_data_loader(*args, **kwargs):
    """Helper that omits state_dict from yields."""
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
    Fast dataloader that reads pre-tokenized binary files with memory optimization.

    This is much faster than tokenizing_distributed_data_loader because:
    1. No tokenization overhead
    2. Memory-mapped files for fast random access
    3. Pre-allocated buffers for minimal GC pressure
    4. Single HtoD transfer

    Args:
        B: batch size (number of sequences per batch)
        T: sequence length
        split: "train" or "val"
        device: target device
        resume_state_dict: optional dict with {"global_pos"} to resume from

    Yields:
        (inputs, targets, state_dict) where inputs/targets are (B, T) tensors
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    # Get all binary files
    bin_files = list_bin_files()
    assert len(bin_files) > 0, "No tokenized .bin files found. Run: python -m scripts.pretokenize"

    # Split: last file is val, rest is train
    if len(bin_files) == 1:
        if split == "train":
            raise ValueError(
                f"Cannot create train split: only 1 .bin file found, but need at least 2 "
                f"(last file is used for validation). Either add more data shards or use split='val'."
            )
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

    # Pre-allocate buffers for memory optimization
    use_cuda = device == "cuda"
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        # Calculate this rank's starting position
        pos = (global_pos + rank_offset) % total_tokens

        # Get tokens for this batch
        tokens = get_tokens_at(pos, needed_tokens)

        # Fill CPU buffer
        scratch = torch.tensor(tokens, dtype=torch.long)
        cpu_inputs.copy_(scratch[:-1].view(B, T))
        cpu_targets.copy_(scratch[1:].view(B, T))

        # Single HtoD transfer
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)

        state_dict = {"global_pos": global_pos}
        yield inputs, targets, state_dict

        # Advance global position for next iteration
        global_pos = (global_pos + tokens_per_batch) % total_tokens


def binary_distributed_data_loader(*args, **kwargs):
    """Helper that only yields inputs/targets without state_dict."""
    for inputs, targets, state_dict in binary_distributed_data_loader_with_state(*args, **kwargs):
        yield inputs, targets
