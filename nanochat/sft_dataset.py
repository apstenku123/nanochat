"""
SFT dataset for instruction-response pairs.

Loads JSONL files with {"instruction": "...", "response": "..."} format.
Tokenizes with a simple template: <BOS> instruction \n response <EOS>
Supports loss masking on instruction tokens (only train on response).
"""

import json
import os
import torch
from torch.utils.data import Dataset

from nanochat.tokenizer import get_tokenizer


class SFTDataset(Dataset):
    """Supervised fine-tuning dataset for instruction-code pairs.

    Each sample is tokenized as:
        <BOS> instruction \n response <EOS>

    Targets are shifted by one (standard LM next-token prediction).
    Instruction tokens (and BOS) are masked with -1 in targets so that
    the loss is only computed on response tokens.

    Args:
        jsonl_path: Path to a JSONL file with "instruction" and "response" fields.
        tokenizer_name: "cpp" to use CppTokenizer, otherwise default BPE.
        max_len: Maximum sequence length (truncated if longer).
    """

    def __init__(
        self, jsonl_path: str, tokenizer_name: str = "default", max_len: int = 1024
    ):
        self.max_len = max_len

        # Load tokenizer
        if tokenizer_name == "cpp":
            os.environ["NANOCHAT_CPP_TOKENIZER"] = "1"
        self.tokenizer = get_tokenizer()

        # Get special token IDs
        self.bos_id = self.tokenizer.get_bos_token_id()
        # EOS: try cpp-style <EOS> first, then nanochat-style
        eos_id = None
        if hasattr(self.tokenizer, "eos_token_id"):
            eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            eos_id = self.tokenizer.encode_special("<|assistant_end|>")
        if eos_id is None:
            eos_id = self.bos_id  # fallback: use BOS as document delimiter
        self.eos_id = eos_id

        # Load and tokenize all examples
        self.examples = []
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                instruction = item["instruction"]
                response = item["response"]
                self.examples.append(self._tokenize(instruction, response))

    def _tokenize(self, instruction: str, response: str):
        """Tokenize an instruction-response pair with loss masking.

        Layout: [BOS] instruction_tokens [\n] response_tokens [EOS]
        The boundary between instruction and response determines the mask.
        """
        # Encode instruction and response separately
        instr_ids = self.tokenizer.encode(instruction)
        resp_ids = self.tokenizer.encode(response)
        # Newline separator
        sep_ids = self.tokenizer.encode("\n")

        # Combine: BOS + instruction + \n + response + EOS
        all_ids = [self.bos_id] + instr_ids + sep_ids + resp_ids + [self.eos_id]

        # Number of prompt tokens (BOS + instruction + separator) that we mask
        n_prompt = 1 + len(instr_ids) + len(sep_ids)

        # Truncate to max_len + 1 (we need +1 for the shifted target)
        all_ids = all_ids[: self.max_len + 1]

        # Input: all_ids[:-1], Target: all_ids[1:]
        input_ids = all_ids[:-1]
        target_ids = all_ids[1:]

        # Mask: instruction portion of the *target* should be -1
        # The target at position i corresponds to predicting token i+1.
        # We mask targets for positions 0..n_prompt-2 (those are still in the prompt).
        mask_len = min(n_prompt - 1, len(target_ids))
        for i in range(mask_len):
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


def sft_collate_fn(batch, pad_id=0, pad_to=None):
    """Collate variable-length SFT examples into a padded batch.

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

    # Quick test
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "sft_sample.jsonl"
    )
    ds = SFTDataset(data_path, tokenizer_name="cpp", max_len=1024)
    print(f"Dataset size: {len(ds)}")
    x, y = ds[0]
    print(f"Input shape: {x.shape}, Target shape: {y.shape}")
    print(f"Masked positions (target=-1): {(y == -1).sum().item()}")
    print(f"Training positions (target!=-1): {(y != -1).sum().item()}")
    print(f"First 20 input ids: {x[:20].tolist()}")
    print(f"First 20 target ids: {y[:20].tolist()}")

    # Verify collation
    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=4, collate_fn=sft_collate_fn)
    bx, by = next(iter(loader))
    print(f"\nBatch input shape: {bx.shape}, Batch target shape: {by.shape}")
    print(
        f"Batch masked: {(by == -1).sum().item()}, training: {(by != -1).sum().item()}"
    )
