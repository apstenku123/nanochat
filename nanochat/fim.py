"""
Fill-in-the-Middle (FIM) augmentation for training data.

Applies FIM transformation to token sequences, supporting both PSM and SPM formats.
See: "Efficient Training of Language Models to Fill in the Middle" (Bavarian et al., 2022)

FIM token IDs (from our C++ tokenizer):
  <FIM_PREFIX> = 4
  <FIM_MIDDLE> = 5
  <FIM_SUFFIX> = 6
  <EOS>        = 3  (used as EOT sentinel)

Two FIM modes:
1. Random FIM: Random splits (original) - good for general code infilling
2. Structured FIM: Docstring→body splits - good for comment→code completion
"""

import os
import json
import random
from typing import Optional


# Default token IDs matching our C++ tokenizer (scripts/tok_train_cpp.py)
FIM_PREFIX_ID = 4   # <FIM_PREFIX>
FIM_MIDDLE_ID = 5   # <FIM_MIDDLE>
FIM_SUFFIX_ID = 6   # <FIM_SUFFIX>
EOT_ID = 3          # <EOS> used as end-of-turn


def apply_fim(
    token_ids: list[int],
    fim_rate: float = 0.5,
    spm_rate: float = 0.5,
    fim_prefix_id: int = FIM_PREFIX_ID,
    fim_middle_id: int = FIM_MIDDLE_ID,
    fim_suffix_id: int = FIM_SUFFIX_ID,
    eot_id: int = EOT_ID,
    rng: random.Random | None = None,
) -> list[int]:
    """Apply FIM transformation to a single token sequence.

    With probability `fim_rate`, rearranges tokens into FIM format.
    Otherwise returns the original sequence unchanged.

    Two FIM formats are used (chosen with probability `spm_rate` for SPM):
      PSM: <FIM_PREFIX> prefix <FIM_SUFFIX> suffix <FIM_MIDDLE> middle <EOT>
      SPM: <FIM_PREFIX> <FIM_SUFFIX> suffix <FIM_MIDDLE> prefix middle <EOT>

    Args:
        token_ids: Original token sequence (without BOS/EOS -- just content tokens).
        fim_rate: Probability of applying FIM (default 0.5).
        spm_rate: Probability of using SPM format when FIM is applied (default 0.5).
        fim_prefix_id: Token ID for <FIM_PREFIX>.
        fim_middle_id: Token ID for <FIM_MIDDLE>.
        fim_suffix_id: Token ID for <FIM_SUFFIX>.
        eot_id: Token ID for end-of-turn sentinel.
        rng: Optional random.Random instance for reproducibility.

    Returns:
        Transformed token list (FIM format) or original token list.
    """
    if rng is None:
        rng = random

    if rng.random() > fim_rate:
        return token_ids  # no FIM -- standard next-token prediction

    n = len(token_ids)
    if n < 2:
        return token_ids  # too short to split meaningfully

    # Pick two random split points to divide into prefix | middle | suffix
    split_start = rng.randint(0, n)
    split_end = rng.randint(split_start, n)

    prefix = token_ids[:split_start]
    middle = token_ids[split_start:split_end]
    suffix = token_ids[split_end:]

    if rng.random() < spm_rate:
        # SPM: suffix-prefix-middle (suffix comes right after sentinel, then
        # prefix+middle are contiguous so the model learns to continue from prefix)
        return [fim_prefix_id, fim_suffix_id] + suffix + [fim_middle_id] + prefix + middle + [eot_id]
    else:
        # PSM: prefix-suffix-middle
        return [fim_prefix_id] + prefix + [fim_suffix_id] + suffix + [fim_middle_id] + middle + [eot_id]


def apply_fim_batch(
    token_lists: list[list[int]],
    fim_rate: float = 0.5,
    spm_rate: float = 0.5,
    fim_prefix_id: int = FIM_PREFIX_ID,
    fim_middle_id: int = FIM_MIDDLE_ID,
    fim_suffix_id: int = FIM_SUFFIX_ID,
    eot_id: int = EOT_ID,
    rng: random.Random | None = None,
) -> list[list[int]]:
    """Apply FIM transformation to a batch of token sequences.

    Args:
        token_lists: List of token sequences.
        Other args: same as apply_fim.

    Returns:
        List of (possibly transformed) token sequences.
    """
    return [
        apply_fim(
            toks,
            fim_rate=fim_rate,
            spm_rate=spm_rate,
            fim_prefix_id=fim_prefix_id,
            fim_middle_id=fim_middle_id,
            fim_suffix_id=fim_suffix_id,
            eot_id=eot_id,
            rng=rng,
        )
        for toks in token_lists
    ]


# -----------------------------------------------------------------------------
# Structured FIM: Docstring → Function Body completion
# -----------------------------------------------------------------------------

class StructuredFIMDataset:
    """
    Provides structured FIM examples from pre-extracted docstring pairs.

    Format: <FIM_PREFIX> docstring + signature { <FIM_SUFFIX> } <FIM_MIDDLE> body <EOT>

    This teaches the model to complete function bodies given docstrings.
    """

    def __init__(self, pairs_path: str, tokenizer, max_examples: int = -1):
        """
        Args:
            pairs_path: Path to JSONL file with docstring pairs
            tokenizer: Tokenizer instance (must have encode method)
            max_examples: Max examples to load (-1 = all)
        """
        self.pairs = []
        self.tokenizer = tokenizer

        if not os.path.exists(pairs_path):
            print(f"Warning: Structured FIM dataset not found: {pairs_path}")
            return

        with open(pairs_path, 'r') as f:
            for i, line in enumerate(f):
                if max_examples > 0 and i >= max_examples:
                    break
                if line.strip():
                    self.pairs.append(json.loads(line))

        print(f"Loaded {len(self.pairs):,} structured FIM examples from {pairs_path}")

    def __len__(self):
        return len(self.pairs)

    def get_random_example(self, rng: random.Random = None) -> Optional[list[int]]:
        """
        Get a random structured FIM example as token IDs.

        Returns:
            Token list in FIM format, or None if no examples available.
        """
        if not self.pairs:
            return None

        if rng is None:
            rng = random

        pair = rng.choice(self.pairs)
        return self.pair_to_tokens(pair)

    def pair_to_tokens(
        self,
        pair: dict,
        fim_prefix_id: int = FIM_PREFIX_ID,
        fim_middle_id: int = FIM_MIDDLE_ID,
        fim_suffix_id: int = FIM_SUFFIX_ID,
        eot_id: int = EOT_ID,
    ) -> list[int]:
        """
        Convert a docstring pair to FIM token sequence.

        Format: <FIM_PREFIX> /* docstring */ signature { <FIM_SUFFIX> } <FIM_MIDDLE> body <EOT>
        """
        docstring = pair.get('docstring', '')
        signature = pair.get('signature', '')
        body = pair.get('body', '')

        # Build prefix: /* docstring */ signature {
        prefix_text = f"/*\n{docstring}\n*/\n{signature} {{"

        # Build suffix: just the closing brace
        suffix_text = "\n}"

        # Tokenize
        prefix_tokens = self.tokenizer.encode(prefix_text)
        suffix_tokens = self.tokenizer.encode(suffix_text)
        body_tokens = self.tokenizer.encode("\n" + body + "\n")

        # PSM format: <FIM_PREFIX> prefix <FIM_SUFFIX> suffix <FIM_MIDDLE> middle <EOT>
        return (
            [fim_prefix_id] +
            prefix_tokens +
            [fim_suffix_id] +
            suffix_tokens +
            [fim_middle_id] +
            body_tokens +
            [eot_id]
        )


def apply_fim_mixed(
    token_ids: list[int],
    structured_dataset: Optional[StructuredFIMDataset],
    fim_rate: float = 0.4,
    structured_rate: float = 0.2,
    spm_rate: float = 0.5,
    fim_prefix_id: int = FIM_PREFIX_ID,
    fim_middle_id: int = FIM_MIDDLE_ID,
    fim_suffix_id: int = FIM_SUFFIX_ID,
    eot_id: int = EOT_ID,
    rng: random.Random = None,
) -> list[int]:
    """
    Apply mixed FIM: either random FIM, structured FIM, or no FIM.

    Probabilities:
    - structured_rate: Use structured FIM (docstring→body)
    - fim_rate: Use random FIM
    - (1 - structured_rate - fim_rate): No FIM (standard next-token)

    Args:
        token_ids: Original token sequence
        structured_dataset: StructuredFIMDataset instance (can be None)
        fim_rate: Probability of random FIM
        structured_rate: Probability of structured FIM
        Other args: same as apply_fim

    Returns:
        Token list (possibly transformed)
    """
    if rng is None:
        rng = random

    roll = rng.random()

    # Try structured FIM first
    if roll < structured_rate and structured_dataset is not None:
        example = structured_dataset.get_random_example(rng)
        if example is not None:
            return example
        # Fall through to random FIM if no structured examples

    # Random FIM
    if roll < structured_rate + fim_rate:
        return apply_fim(
            token_ids,
            fim_rate=1.0,  # Always apply since we already rolled
            spm_rate=spm_rate,
            fim_prefix_id=fim_prefix_id,
            fim_middle_id=fim_middle_id,
            fim_suffix_id=fim_suffix_id,
            eot_id=eot_id,
            rng=rng,
        )

    # No FIM
    return token_ids


def apply_fim_mixed_batch(
    token_lists: list[list[int]],
    structured_dataset: Optional[StructuredFIMDataset],
    fim_rate: float = 0.4,
    structured_rate: float = 0.2,
    spm_rate: float = 0.5,
    fim_prefix_id: int = FIM_PREFIX_ID,
    fim_middle_id: int = FIM_MIDDLE_ID,
    fim_suffix_id: int = FIM_SUFFIX_ID,
    eot_id: int = EOT_ID,
    rng: random.Random = None,
) -> list[list[int]]:
    """Apply mixed FIM to a batch of token sequences."""
    return [
        apply_fim_mixed(
            toks,
            structured_dataset,
            fim_rate=fim_rate,
            structured_rate=structured_rate,
            spm_rate=spm_rate,
            fim_prefix_id=fim_prefix_id,
            fim_middle_id=fim_middle_id,
            fim_suffix_id=fim_suffix_id,
            eot_id=eot_id,
            rng=rng,
        )
        for toks in token_lists
    ]
