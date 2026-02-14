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
FIM_PREFIX_ID = 4  # <FIM_PREFIX>
FIM_MIDDLE_ID = 5  # <FIM_MIDDLE>
FIM_SUFFIX_ID = 6  # <FIM_SUFFIX>
EOT_ID = 3  # <EOS> used as end-of-turn


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
        return (
            [fim_prefix_id, fim_suffix_id]
            + suffix
            + [fim_middle_id]
            + prefix
            + middle
            + [eot_id]
        )
    else:
        # PSM: prefix-suffix-middle
        return (
            [fim_prefix_id]
            + prefix
            + [fim_suffix_id]
            + suffix
            + [fim_middle_id]
            + middle
            + [eot_id]
        )


def apply_fim_function_level(
    token_ids: list[int],
    tokenizer,
    fim_rate: float = 0.5,
    spm_rate: float = 0.5,
    fim_prefix_id: int = FIM_PREFIX_ID,
    fim_middle_id: int = FIM_MIDDLE_ID,
    fim_suffix_id: int = FIM_SUFFIX_ID,
    eot_id: int = EOT_ID,
    rng: random.Random | None = None,
) -> list[int]:
    """Apply FIM at C++ function boundaries instead of random positions.

    Finds function bodies (matched { } blocks) in the token stream and uses
    them as the "middle" region. This teaches the model to complete entire
    function bodies given surrounding context.

    Falls back to random FIM if no suitable function boundary is found.
    """
    if rng is None:
        rng = random

    if rng.random() > fim_rate:
        return token_ids

    n = len(token_ids)
    if n < 10:
        return token_ids

    # Find { } block boundaries by scanning token strings
    # We look for matched brace pairs that likely represent function bodies
    open_brace_id = None
    close_brace_id = None
    try:
        open_brace_id = (
            tokenizer.encode("{")[0] if hasattr(tokenizer, "encode") else None
        )
        close_brace_id = (
            tokenizer.encode("}")[0] if hasattr(tokenizer, "encode") else None
        )
    except (IndexError, TypeError):
        pass

    if open_brace_id is None or close_brace_id is None:
        # Can't find brace tokens, fall back to random FIM
        return apply_fim(
            token_ids,
            1.0,
            spm_rate,
            fim_prefix_id,
            fim_middle_id,
            fim_suffix_id,
            eot_id,
            rng,
        )

    # Find all top-level { } blocks (depth tracking)
    # These are candidate function bodies
    blocks = []  # list of (open_idx, close_idx)
    depth = 0
    block_start = -1
    for i, tid in enumerate(token_ids):
        if tid == open_brace_id:
            if depth == 0:
                block_start = i
            depth += 1
        elif tid == close_brace_id:
            depth -= 1
            if depth == 0 and block_start >= 0:
                # Only keep blocks with reasonable size (4-500 tokens)
                body_len = i - block_start - 1
                if 4 <= body_len <= 500:
                    blocks.append((block_start, i))
                block_start = -1
            if depth < 0:
                depth = 0  # reset on malformed input

    if not blocks:
        # No suitable blocks found, fall back to random FIM
        return apply_fim(
            token_ids,
            1.0,
            spm_rate,
            fim_prefix_id,
            fim_middle_id,
            fim_suffix_id,
            eot_id,
            rng,
        )

    # Pick a random function body block
    open_idx, close_idx = rng.choice(blocks)

    # Split: prefix = everything before {, middle = body inside braces, suffix = } and after
    prefix = token_ids[: open_idx + 1]  # includes the opening {
    middle = token_ids[open_idx + 1 : close_idx]  # function body
    suffix = token_ids[close_idx:]  # includes the closing } and everything after

    if rng.random() < spm_rate:
        return (
            [fim_prefix_id, fim_suffix_id]
            + suffix
            + [fim_middle_id]
            + prefix
            + middle
            + [eot_id]
        )
    else:
        return (
            [fim_prefix_id]
            + prefix
            + [fim_suffix_id]
            + suffix
            + [fim_middle_id]
            + middle
            + [eot_id]
        )


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

        with open(pairs_path, "r") as f:
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
        docstring = pair.get("docstring", "")
        signature = pair.get("signature", "")
        body = pair.get("body", "")

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
            [fim_prefix_id]
            + prefix_tokens
            + [fim_suffix_id]
            + suffix_tokens
            + [fim_middle_id]
            + body_tokens
            + [eot_id]
        )


def apply_fim_mixed(
    token_ids: list[int],
    structured_dataset: Optional[StructuredFIMDataset],
    fim_rate: float = 0.4,
    structured_rate: float = 0.2,
    function_fim_rate: float = 0.0,
    spm_rate: float = 0.5,
    tokenizer=None,
    fim_prefix_id: int = FIM_PREFIX_ID,
    fim_middle_id: int = FIM_MIDDLE_ID,
    fim_suffix_id: int = FIM_SUFFIX_ID,
    eot_id: int = EOT_ID,
    rng: random.Random = None,
) -> list[int]:
    """
    Apply mixed FIM: function-level, random, structured, or no FIM.

    Probabilities (checked in order):
    - structured_rate: Use structured FIM (docstring→body)
    - function_fim_rate: Use function-level FIM (split at { } boundaries)
    - fim_rate: Use random FIM
    - remainder: No FIM (standard next-token)

    Args:
        token_ids: Original token sequence
        structured_dataset: StructuredFIMDataset instance (can be None)
        fim_rate: Probability of random FIM
        structured_rate: Probability of structured FIM
        function_fim_rate: Probability of function-level FIM (requires tokenizer)
        tokenizer: Tokenizer instance (needed for function-level FIM)
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
        # Fall through if no structured examples

    # Function-level FIM
    if roll < structured_rate + function_fim_rate and tokenizer is not None:
        return apply_fim_function_level(
            token_ids,
            tokenizer,
            fim_rate=1.0,
            spm_rate=spm_rate,
            fim_prefix_id=fim_prefix_id,
            fim_middle_id=fim_middle_id,
            fim_suffix_id=fim_suffix_id,
            eot_id=eot_id,
            rng=rng,
        )

    # Random FIM
    if roll < structured_rate + function_fim_rate + fim_rate:
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
    function_fim_rate: float = 0.0,
    spm_rate: float = 0.5,
    tokenizer=None,
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
            function_fim_rate=function_fim_rate,
            spm_rate=spm_rate,
            tokenizer=tokenizer,
            fim_prefix_id=fim_prefix_id,
            fim_middle_id=fim_middle_id,
            fim_suffix_id=fim_suffix_id,
            eot_id=eot_id,
            rng=rng,
        )
        for toks in token_lists
    ]


# ---------------------------------------------------------------------------
# Token healing for FIM inference
# ---------------------------------------------------------------------------


def heal_fim_tokens(
    prefix_ids: list[int],
    suffix_ids: list[int],
    tokenizer,
) -> tuple[list[int], list[int], int]:
    """Token healing: fix split-token artifacts at FIM boundaries.

    When a FIM split lands inside a BPE token, the last token of the prefix
    and first token of the suffix may be "wrong" — they're fragments that
    wouldn't appear in normal tokenization. Token healing backs off these
    boundary tokens and re-tokenizes the overlap region.

    Example:
        "std::vect" | "or<int>" split inside "vector"
        prefix ends with [..."vect"], suffix starts with ["or"...]
        After healing: prefix ends with [..."std", "::"] and the model
        regenerates "vector<int>..."

    Args:
        prefix_ids: Token IDs for the prefix (before the cursor)
        suffix_ids: Token IDs for the suffix (after the cursor)
        tokenizer: Tokenizer with encode() and decode() methods

    Returns:
        (healed_prefix, healed_suffix, n_rollback) where n_rollback is the
        number of tokens removed from the prefix end (the model should
        regenerate these during inference).
    """
    if not prefix_ids or not suffix_ids:
        return prefix_ids, suffix_ids, 0

    # Decode the boundary tokens
    last_prefix_text = tokenizer.decode([prefix_ids[-1]])
    first_suffix_text = tokenizer.decode([suffix_ids[0]])

    # Re-tokenize the combined boundary
    combined = last_prefix_text + first_suffix_text
    retokenized = tokenizer.encode(combined)

    # If re-tokenizing produces the same tokens, no healing needed
    if retokenized == [prefix_ids[-1], suffix_ids[0]]:
        return prefix_ids, suffix_ids, 0

    # Healing needed: the boundary was split inside a BPE token.
    # Back off the last prefix token so the model regenerates it.
    healed_prefix = prefix_ids[:-1]
    return healed_prefix, suffix_ids, 1


def prepare_fim_inference(
    prompt: str,
    suffix: str,
    tokenizer,
    heal: bool = True,
    fim_prefix_id: int = FIM_PREFIX_ID,
    fim_middle_id: int = FIM_MIDDLE_ID,
    fim_suffix_id: int = FIM_SUFFIX_ID,
) -> tuple[list[int], int]:
    """Prepare token sequence for FIM inference with optional token healing.

    Builds the SPM format: <FIM_PREFIX> <FIM_SUFFIX> suffix <FIM_MIDDLE> prefix
    and applies token healing to fix boundary artifacts.

    Args:
        prompt: The prefix text (code before cursor)
        suffix: The suffix text (code after cursor)
        tokenizer: Tokenizer instance
        heal: Whether to apply token healing

    Returns:
        (input_ids, n_rollback) where input_ids is the token sequence to feed
        to the model and n_rollback is how many prefix tokens were rolled back.
    """
    prefix_ids = tokenizer.encode(prompt)
    suffix_ids = tokenizer.encode(suffix)

    n_rollback = 0
    if heal and prefix_ids and suffix_ids:
        prefix_ids, suffix_ids, n_rollback = heal_fim_tokens(
            prefix_ids, suffix_ids, tokenizer
        )

    # Build SPM format: <FIM_PREFIX><FIM_SUFFIX> suffix <FIM_MIDDLE> prefix
    input_ids = (
        [fim_prefix_id, fim_suffix_id] + suffix_ids + [fim_middle_id] + prefix_ids
    )

    return input_ids, n_rollback
