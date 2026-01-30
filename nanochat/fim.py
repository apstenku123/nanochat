"""
Fill-in-the-Middle (FIM) augmentation for training data.

Applies FIM transformation to token sequences, supporting both PSM and SPM formats.
See: "Efficient Training of Language Models to Fill in the Middle" (Bavarian et al., 2022)

FIM token IDs (from our C++ tokenizer):
  <FIM_PREFIX> = 4
  <FIM_MIDDLE> = 5
  <FIM_SUFFIX> = 6
  <EOS>        = 3  (used as EOT sentinel)
"""

import random


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
