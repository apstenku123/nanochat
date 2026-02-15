"""
Tool-call SFT dataset with auto-masking from special tokens.

Loads JSONL files with {"text": "...", "source": "..."} format where text
contains embedded special tokens (<THOUGHT_START>, <QUERY_TOOL>, <TOOL_RESULT>,
<CODE_START>, <CODE_END>, etc.).

Loss masking is computed automatically from token IDs:
- Tokens between <TOOL_RESULT> and next <CODE_END> → mask=0 (not trained)
- Tokens before first <THOUGHT_START>/<CODE_START>/<QUERY_TOOL> → mask=0 (instruction)
- Everything else → mask=1 (trained: thoughts, tool calls, code output)
"""

import hashlib
import json
import os
import time
import torch
import numpy as np
from torch.utils.data import Dataset

from nanochat.tokenizer import get_tokenizer


# Default token IDs for C++ tokenizer. Used as fallback if lookup fails.
DEFAULT_SPECIAL_IDS = {
    "bos": 2,
    "eos": 3,
    "code_start": 7,
    "code_end": 8,
    "thought_start": 9,
    "thought_end": 10,
    "query_tool": 11,
    "tool_result": 19,
    "fim_prefix": 4,
    "fim_middle": 5,
    "fim_suffix": 6,
}

# Backwards-compatible constant exports (used by tests and older callers).
BOS_ID = DEFAULT_SPECIAL_IDS["bos"]
EOS_ID = DEFAULT_SPECIAL_IDS["eos"]
CODE_START_ID = DEFAULT_SPECIAL_IDS["code_start"]
CODE_END_ID = DEFAULT_SPECIAL_IDS["code_end"]
THOUGHT_START_ID = DEFAULT_SPECIAL_IDS["thought_start"]
THOUGHT_END_ID = DEFAULT_SPECIAL_IDS["thought_end"]
QUERY_TOOL_ID = DEFAULT_SPECIAL_IDS["query_tool"]
TOOL_RESULT_ID = DEFAULT_SPECIAL_IDS["tool_result"]
FIM_PREFIX_ID = DEFAULT_SPECIAL_IDS["fim_prefix"]
FIM_MIDDLE_ID = DEFAULT_SPECIAL_IDS["fim_middle"]
FIM_SUFFIX_ID = DEFAULT_SPECIAL_IDS["fim_suffix"]


def _resolve_special_ids(tokenizer) -> dict:
    """Resolve special token IDs from tokenizer with fallback defaults."""

    def get_id(token: str, default: int | None = None) -> int | None:
        tid = tokenizer.encode_special(token)
        return default if tid is None else tid

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        eos_id = get_id("<EOS>", DEFAULT_SPECIAL_IDS["eos"])

    return {
        "bos": tokenizer.get_bos_token_id(),
        "eos": eos_id,
        "code_start": get_id("<CODE_START>", DEFAULT_SPECIAL_IDS["code_start"]),
        "code_end": get_id("<CODE_END>", DEFAULT_SPECIAL_IDS["code_end"]),
        "thought_start": get_id(
            "<THOUGHT_START>", DEFAULT_SPECIAL_IDS["thought_start"]
        ),
        "thought_end": get_id("<THOUGHT_END>", DEFAULT_SPECIAL_IDS["thought_end"]),
        "query_tool": get_id("<QUERY_TOOL>", DEFAULT_SPECIAL_IDS["query_tool"]),
        "tool_result": get_id("<TOOL_RESULT>", DEFAULT_SPECIAL_IDS["tool_result"]),
        "fim_prefix": get_id("<FIM_PREFIX>", DEFAULT_SPECIAL_IDS["fim_prefix"]),
        "fim_middle": get_id("<FIM_MIDDLE>", DEFAULT_SPECIAL_IDS["fim_middle"]),
        "fim_suffix": get_id("<FIM_SUFFIX>", DEFAULT_SPECIAL_IDS["fim_suffix"]),
    }


def compute_loss_mask(
    token_ids: list[int], special_ids: dict | None = None
) -> list[int]:
    """Compute loss mask from token IDs.

    Rules:
    - Instruction prefix (before first response token): mask=0
    - <THOUGHT_START>...<THOUGHT_END>: mask=1 (train reasoning)
    - <QUERY_TOOL>...<CODE_END>: mask=1 (train tool calls)
    - <TOOL_RESULT>...<CODE_END>: mask=0 (injected at inference)
    - <CODE_START>...<CODE_END>: mask=1 (train code output)
    - FIM tokens: mask=1 (train infilling)
    - <BOS>, <EOS>: mask=0

    Returns list of 0/1 same length as token_ids.
    """
    ids = DEFAULT_SPECIAL_IDS if special_ids is None else special_ids

    bos_id = ids.get("bos")
    eos_id = ids.get("eos")
    code_start_id = ids.get("code_start")
    code_end_id = ids.get("code_end")
    thought_start_id = ids.get("thought_start")
    query_tool_id = ids.get("query_tool")
    tool_result_id = ids.get("tool_result")
    fim_prefix_id = ids.get("fim_prefix")
    fim_middle_id = ids.get("fim_middle")

    response_start_tokens = {
        tid
        for tid in (thought_start_id, code_start_id, query_tool_id, fim_prefix_id)
        if tid is not None
    }

    n = len(token_ids)
    mask = [0] * n

    # Find first response token (where model output begins)
    response_start = n  # default: nothing is response
    for i, tid in enumerate(token_ids):
        if tid in response_start_tokens:
            response_start = i
            break

    # Special case: FIM sequences — everything after <FIM_MIDDLE> is trained
    has_fim = (
        fim_prefix_id is not None
        and fim_middle_id is not None
        and fim_prefix_id in token_ids
    )
    if has_fim:
        # For FIM: train on middle content (the infill), stop at EOS/BOS
        in_middle = False
        for i, tid in enumerate(token_ids):
            if tid == fim_middle_id:
                in_middle = True
                mask[i] = 1  # train on the FIM_MIDDLE token itself
                continue
            if tid == eos_id or tid == bos_id:
                in_middle = False
                mask[i] = 0
                continue
            if in_middle:
                mask[i] = 1
        return mask

    # Normal sequence: process blocks
    in_tool_result = False

    for i in range(response_start, n):
        tid = token_ids[i]

        # Skip BOS/EOS
        if tid == bos_id or tid == eos_id:
            mask[i] = 0
            continue

        # Track tool result blocks (mask=0)
        if tool_result_id is not None and tid == tool_result_id:
            in_tool_result = True
            mask[i] = 0
            continue

        if in_tool_result:
            if code_end_id is not None and tid == code_end_id:
                in_tool_result = False
                mask[i] = 0  # the CODE_END closing a tool result is also masked
            elif tid in (thought_start_id, code_start_id, query_tool_id):
                in_tool_result = False  # recover from unclosed result block
                mask[i] = 1
            else:
                mask[i] = 0
            continue

        # Everything else in the response is trained
        mask[i] = 1

    return mask


class ToolCallSFTDataset(Dataset):
    """SFT dataset for tool-call training with auto-masking.

    Each sample is loaded from JSONL with {"text": "...", "source": "..."}.
    The text contains embedded special tokens that define the structure.
    Loss masking is computed from token IDs automatically.

    Args:
        jsonl_path: Path to JSONL file with "text" field.
        tokenizer_name: "cpp" or "default".
        max_len: Maximum sequence length (truncated if longer).
    """

    def __init__(
        self, jsonl_path: str, tokenizer_name: str = "cpp", max_len: int = 16384
    ):
        self.max_len = max_len

        # Load tokenizer
        if tokenizer_name == "cpp":
            os.environ["NANOCHAT_CPP_TOKENIZER"] = "1"
        self.tokenizer = get_tokenizer()
        self.special_ids = _resolve_special_ids(self.tokenizer)

        # Try loading from pre-tokenized cache
        cache_path = self._cache_path(jsonl_path, max_len)
        if self._load_cache(cache_path, jsonl_path):
            return

        # Load and tokenize all examples
        t0 = time.time()
        all_inputs = []
        all_targets = []
        skipped = 0
        total = 0
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total += 1
                item = json.loads(line)
                text = item["text"]

                result = self._tokenize(text)
                if result is not None:
                    all_inputs.append(result[0])
                    all_targets.append(result[1])
                else:
                    skipped += 1

                if total % 100000 == 0:
                    print(
                        f"  tokenized {total:,} examples ({len(all_inputs):,} kept)..."
                    )

        dt = time.time() - t0
        print(
            f"Tokenized {len(all_inputs):,} examples in {dt:.1f}s (skipped {skipped})"
        )

        # Pack into flat tensors for fast cache/load
        self._pack_examples(all_inputs, all_targets)

        # Save cache for next run
        self._save_cache(cache_path)

    @staticmethod
    def _cache_path(jsonl_path: str, max_len: int) -> str:
        """Derive cache path from JSONL path and max_len."""
        # Include file size in cache key for invalidation
        fsize = os.path.getsize(jsonl_path) if os.path.exists(jsonl_path) else 0
        key = f"{os.path.abspath(jsonl_path)}:{fsize}:{max_len}"
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        return jsonl_path + f".cache.{h}.pt"

    def _pack_examples(self, all_inputs: list, all_targets: list):
        """Pack variable-length examples into flat int32 tensors + offset index."""
        n = len(all_inputs)
        offsets = np.zeros(n + 1, dtype=np.int64)
        for i, inp in enumerate(all_inputs):
            offsets[i + 1] = offsets[i] + len(inp)
        total_tokens = int(offsets[-1])

        input_flat = np.zeros(total_tokens, dtype=np.int32)
        target_flat = np.zeros(total_tokens, dtype=np.int32)
        for i, (inp, tgt) in enumerate(zip(all_inputs, all_targets)):
            start = offsets[i]
            end = offsets[i + 1]
            input_flat[start:end] = inp
            target_flat[start:end] = tgt

        self.offsets = torch.from_numpy(offsets)
        self.input_flat = torch.from_numpy(input_flat)
        self.target_flat = torch.from_numpy(target_flat)
        self._num_examples = n
        print(f"Packed {n:,} examples, {total_tokens:,} total tokens ({total_tokens * 8 / 1e9:.2f} GB)")

    def _save_cache(self, cache_path: str):
        """Save pre-tokenized data to cache file."""
        t0 = time.time()
        torch.save(
            {
                "offsets": self.offsets,
                "input_flat": self.input_flat,
                "target_flat": self.target_flat,
            },
            cache_path,
        )
        dt = time.time() - t0
        sz = os.path.getsize(cache_path) / 1e9
        print(f"Saved cache to {cache_path} ({sz:.2f} GB, {dt:.1f}s)")

    def _load_cache(self, cache_path: str, jsonl_path: str) -> bool:
        """Load pre-tokenized data from cache. Returns True on success."""
        if not os.path.exists(cache_path):
            return False
        # Invalidate if JSONL is newer
        if os.path.getmtime(jsonl_path) > os.path.getmtime(cache_path):
            print("Cache stale (JSONL newer), re-tokenizing...")
            return False
        try:
            t0 = time.time()
            cache = torch.load(cache_path, weights_only=True)
            self.offsets = cache["offsets"]
            self.input_flat = cache["input_flat"]
            self.target_flat = cache["target_flat"]
            self._num_examples = len(self.offsets) - 1
            dt = time.time() - t0
            total_tokens = len(self.input_flat)
            print(
                f"Loaded {self._num_examples:,} examples ({total_tokens:,} tokens) from cache in {dt:.1f}s"
            )
            return True
        except Exception as e:
            print(f"Cache load failed ({e}), re-tokenizing...")
            return False

    def _tokenize(self, text: str):
        """Tokenize text and compute loss mask.

        The text already contains special tokens as literal strings
        (e.g., "<THOUGHT_START>") which the tokenizer maps to their IDs.

        Returns (input_ids, target_ids) or None if too short.
        """
        # Encode the full text — special tokens are in the vocab
        all_ids = self.tokenizer.encode(text)

        if len(all_ids) < 3:
            return None

        # Truncate to max_len + 1
        all_ids = all_ids[: self.max_len + 1]

        # Compute loss mask on the full sequence
        full_mask = compute_loss_mask(all_ids, self.special_ids)

        # Standard LM: input = all_ids[:-1], target = all_ids[1:]
        input_ids = all_ids[:-1]
        target_ids = all_ids[1:]

        # Target mask: use mask[1:] since target[i] predicts token at position i+1
        target_mask = full_mask[1:]

        # Apply mask: set target to -1 where mask=0
        for i in range(len(target_ids)):
            if target_mask[i] == 0:
                target_ids[i] = -1

        return input_ids, target_ids

    def __len__(self):
        return self._num_examples

    def __getitem__(self, idx):
        start = self.offsets[idx].item()
        end = self.offsets[idx + 1].item()
        return (
            self.input_flat[start:end].to(torch.long),
            self.target_flat[start:end].to(torch.long),
        )


def tool_sft_collate_fn(batch, pad_id=0, pad_to=None):
    """Collate variable-length tool SFT examples into a padded batch.

    Pads input_ids with pad_id and target_ids with -1 (ignored by loss).
    Returns (input_ids, targets) both of shape (B, pad_len).
    When pad_to is set, all sequences are padded to that fixed length
    (required for XLA/TPU to avoid recompilation on every unique length).
    """
    inputs, targets = zip(*batch)
    max_len = max(x.size(0) for x in inputs)
    pad_len = pad_to if pad_to is not None else max_len

    padded_inputs = torch.full((len(inputs), pad_len), pad_id, dtype=torch.long)
    padded_targets = torch.full((len(targets), pad_len), -1, dtype=torch.long)

    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        padded_inputs[i, : inp.size(0)] = inp
        padded_targets[i, : tgt.size(0)] = tgt

    return padded_inputs, padded_targets


if __name__ == "__main__":
    import sys

    # Quick test
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/tool_call_sft.jsonl"
    ds = ToolCallSFTDataset(data_path, tokenizer_name="cpp", max_len=1024)
    print(f"Dataset size: {len(ds)}")

    if len(ds) > 0:
        x, y = ds[0]
        print(f"Input shape: {x.shape}, Target shape: {y.shape}")
        masked = (y == -1).sum().item()
        trained = (y != -1).sum().item()
        print(f"Masked positions (target=-1): {masked}")
        print(f"Training positions (target!=-1): {trained}")
        print(f"Training ratio: {100 * trained / (masked + trained):.1f}%")

        # Show token breakdown
        tok = ds.tokenizer
        print("\nFirst 30 input tokens:")
        for i in range(min(30, len(x))):
            tid = x[i].item()
            tgt = y[i].item() if i < len(y) else -1
            token_str = (
                tok.id_to_token(tid) if hasattr(tok, "id_to_token") else str(tid)
            )
            mask_str = "TRAIN" if tgt != -1 else "mask"
            print(f"  [{i:3d}] id={tid:5d} target={tgt:5d} {mask_str:5s}  {token_str}")

        # Verify collation
        from torch.utils.data import DataLoader

        loader = DataLoader(ds, batch_size=4, collate_fn=tool_sft_collate_fn)
        bx, by = next(iter(loader))
        print(f"\nBatch input shape: {bx.shape}, Batch target shape: {by.shape}")
        print(
            f"Batch masked: {(by == -1).sum().item()}, training: {(by != -1).sum().item()}"
        )

        # Show source distribution
        print("\nChecking source distribution (first 1000)...")
        sources = {}
        import json as _json

        with open(data_path) as f:
            for i, line in enumerate(f):
                if i >= 1000:
                    break
                d = _json.loads(line)
                src = d.get("source", "?")
                sources[src] = sources.get(src, 0) + 1
        for src, cnt in sorted(sources.items()):
            print(f"  {src}: {cnt}")
