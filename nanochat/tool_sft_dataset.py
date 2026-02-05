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

import json
import os
import torch
from torch.utils.data import Dataset

from nanochat.tokenizer import get_tokenizer


# Special token IDs (C++ tokenizer)
BOS_ID = 2
EOS_ID = 3
CODE_START_ID = 7
CODE_END_ID = 8
THOUGHT_START_ID = 9
THOUGHT_END_ID = 10
QUERY_TOOL_ID = 11
TOOL_RESULT_ID = 19
FIM_PREFIX_ID = 4
FIM_MIDDLE_ID = 5
FIM_SUFFIX_ID = 6

# Tokens that signal the start of model-generated content
RESPONSE_START_TOKENS = {THOUGHT_START_ID, CODE_START_ID, QUERY_TOOL_ID, FIM_PREFIX_ID}


def compute_loss_mask(token_ids: list[int]) -> list[int]:
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
    n = len(token_ids)
    mask = [0] * n

    # Find first response token (where model output begins)
    response_start = n  # default: nothing is response
    for i, tid in enumerate(token_ids):
        if tid in RESPONSE_START_TOKENS:
            response_start = i
            break

    # Special case: FIM sequences — everything after <FIM_MIDDLE> is trained
    has_fim = FIM_PREFIX_ID in token_ids
    if has_fim:
        # For FIM: train on middle content (the infill)
        in_middle = False
        for i, tid in enumerate(token_ids):
            if tid == FIM_MIDDLE_ID:
                in_middle = True
                mask[i] = 1  # train on the FIM_MIDDLE token itself
                continue
            if in_middle and tid not in (BOS_ID, EOS_ID):
                mask[i] = 1
        return mask

    # Normal sequence: process blocks
    in_tool_result = False

    for i in range(response_start, n):
        tid = token_ids[i]

        # Skip BOS/EOS
        if tid in (BOS_ID, EOS_ID):
            mask[i] = 0
            continue

        # Track tool result blocks (mask=0)
        if tid == TOOL_RESULT_ID:
            in_tool_result = True
            mask[i] = 0
            continue

        if in_tool_result:
            if tid == CODE_END_ID:
                in_tool_result = False
                mask[i] = 0  # the CODE_END closing a tool result is also masked
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

    def __init__(self, jsonl_path: str, tokenizer_name: str = "cpp", max_len: int = 16384):
        self.max_len = max_len

        # Load tokenizer
        if tokenizer_name == "cpp":
            os.environ["NANOCHAT_CPP_TOKENIZER"] = "1"
        self.tokenizer = get_tokenizer()

        # Load and tokenize all examples
        self.examples = []
        skipped = 0
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                text = item["text"]

                result = self._tokenize(text)
                if result is not None:
                    self.examples.append(result)
                else:
                    skipped += 1

        if skipped > 0:
            print(f"ToolCallSFTDataset: skipped {skipped} examples (too short after tokenization)")

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
        all_ids = all_ids[:self.max_len + 1]

        # Compute loss mask on the full sequence
        full_mask = compute_loss_mask(all_ids)

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
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids, target_ids = self.examples[idx]
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )


def tool_sft_collate_fn(batch, pad_id=0):
    """Collate variable-length tool SFT examples into a padded batch.

    Pads input_ids with pad_id and target_ids with -1 (ignored by loss).
    Returns (input_ids, targets) both of shape (B, max_len_in_batch).
    """
    inputs, targets = zip(*batch)
    max_len = max(x.size(0) for x in inputs)

    padded_inputs = torch.full((len(inputs), max_len), pad_id, dtype=torch.long)
    padded_targets = torch.full((len(targets), max_len), -1, dtype=torch.long)

    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        padded_inputs[i, :inp.size(0)] = inp
        padded_targets[i, :tgt.size(0)] = tgt

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
        print(f"Training ratio: {100*trained/(masked+trained):.1f}%")

        # Show token breakdown
        tok = ds.tokenizer
        print(f"\nFirst 30 input tokens:")
        for i in range(min(30, len(x))):
            tid = x[i].item()
            tgt = y[i].item() if i < len(y) else -1
            token_str = tok.id_to_token(tid) if hasattr(tok, 'id_to_token') else str(tid)
            mask_str = "TRAIN" if tgt != -1 else "mask"
            print(f"  [{i:3d}] id={tid:5d} target={tgt:5d} {mask_str:5s}  {token_str}")

        # Verify collation
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, batch_size=4, collate_fn=tool_sft_collate_fn)
        bx, by = next(iter(loader))
        print(f"\nBatch input shape: {bx.shape}, Batch target shape: {by.shape}")
        print(f"Batch masked: {(by == -1).sum().item()}, training: {(by != -1).sum().item()}")

        # Show source distribution
        print(f"\nChecking source distribution (first 1000)...")
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
