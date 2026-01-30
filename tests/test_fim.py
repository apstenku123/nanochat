"""Tests for Fill-in-the-Middle (FIM) augmentation."""

import random
from collections import Counter

from nanochat.fim import (
    apply_fim,
    apply_fim_batch,
    FIM_PREFIX_ID,
    FIM_MIDDLE_ID,
    FIM_SUFFIX_ID,
    EOT_ID,
)


def test_no_fim_leaves_tokens_unchanged():
    """With fim_rate=0, tokens are never modified."""
    rng = random.Random(42)
    original = list(range(100, 200))
    for _ in range(100):
        result = apply_fim(original, fim_rate=0.0, rng=rng)
        assert result == original


def test_always_fim_always_transforms():
    """With fim_rate=1, every sequence gets FIM tokens."""
    rng = random.Random(42)
    original = list(range(100, 200))
    for _ in range(50):
        result = apply_fim(original, fim_rate=1.0, rng=rng)
        assert FIM_PREFIX_ID in result
        assert FIM_MIDDLE_ID in result
        assert FIM_SUFFIX_ID in result
        assert result[-1] == EOT_ID


def test_fim_tokens_appear_exactly_once():
    """Each FIM sentinel appears exactly once in transformed output."""
    rng = random.Random(123)
    original = list(range(500, 600))
    for _ in range(100):
        result = apply_fim(original, fim_rate=1.0, rng=rng)
        assert result.count(FIM_PREFIX_ID) == 1
        assert result.count(FIM_MIDDLE_ID) == 1
        assert result.count(FIM_SUFFIX_ID) == 1
        assert result.count(EOT_ID) == 1


def test_content_preserved():
    """The concatenation of prefix+middle+suffix equals the original tokens."""
    rng = random.Random(99)
    original = list(range(1000, 1050))
    sentinel_ids = {FIM_PREFIX_ID, FIM_MIDDLE_ID, FIM_SUFFIX_ID, EOT_ID}

    for _ in range(200):
        result = apply_fim(original, fim_rate=1.0, rng=rng)
        # Strip all sentinel tokens to recover content
        content = [t for t in result if t not in sentinel_ids]
        assert sorted(content) == sorted(original), f"Content mismatch: {sorted(content)} != {sorted(original)}"


def test_psm_format():
    """PSM format: <PREFIX> prefix <SUFFIX> suffix <MIDDLE> middle <EOT>."""
    rng = random.Random(42)
    original = list(range(100, 110))  # 10 tokens

    # Force PSM by setting spm_rate=0
    result = apply_fim(original, fim_rate=1.0, spm_rate=0.0, rng=rng)

    # First token must be FIM_PREFIX
    assert result[0] == FIM_PREFIX_ID
    # Last token must be EOT
    assert result[-1] == EOT_ID
    # Order: PREFIX comes before SUFFIX which comes before MIDDLE
    pi = result.index(FIM_PREFIX_ID)
    si = result.index(FIM_SUFFIX_ID)
    mi = result.index(FIM_MIDDLE_ID)
    assert pi < si < mi


def test_spm_format():
    """SPM format: <PREFIX> <SUFFIX> suffix <MIDDLE> prefix middle <EOT>."""
    rng = random.Random(42)
    original = list(range(100, 110))

    # Force SPM by setting spm_rate=1
    result = apply_fim(original, fim_rate=1.0, spm_rate=1.0, rng=rng)

    # First two tokens must be FIM_PREFIX, FIM_SUFFIX (adjacent)
    assert result[0] == FIM_PREFIX_ID
    assert result[1] == FIM_SUFFIX_ID
    # Last token must be EOT
    assert result[-1] == EOT_ID


def test_fim_rate_approximately_correct():
    """About 50% of sequences should be transformed with fim_rate=0.5."""
    rng = random.Random(42)
    original = list(range(100, 150))
    n_trials = 2000
    n_fim = 0
    for _ in range(n_trials):
        result = apply_fim(original, fim_rate=0.5, rng=rng)
        if FIM_PREFIX_ID in result:
            n_fim += 1

    ratio = n_fim / n_trials
    assert 0.4 < ratio < 0.6, f"FIM rate {ratio:.3f} not near 0.5"


def test_spm_psm_split_approximately_even():
    """When both are applied, PSM and SPM should be ~50/50."""
    rng = random.Random(42)
    original = list(range(100, 150))
    n_trials = 2000
    n_spm = 0
    n_total = 0

    for _ in range(n_trials):
        result = apply_fim(original, fim_rate=1.0, spm_rate=0.5, rng=rng)
        n_total += 1
        # SPM has PREFIX and SUFFIX adjacent at positions 0,1
        if result[0] == FIM_PREFIX_ID and result[1] == FIM_SUFFIX_ID:
            n_spm += 1

    ratio = n_spm / n_total
    assert 0.4 < ratio < 0.6, f"SPM ratio {ratio:.3f} not near 0.5"


def test_short_sequence():
    """Sequences with 0 or 1 tokens are returned unchanged even with fim_rate=1."""
    rng = random.Random(42)
    assert apply_fim([], fim_rate=1.0, rng=rng) == []
    assert apply_fim([999], fim_rate=1.0, rng=rng) == [999]


def test_batch_function():
    """apply_fim_batch processes all sequences."""
    rng = random.Random(42)
    batch = [list(range(100, 120)) for _ in range(10)]
    results = apply_fim_batch(batch, fim_rate=1.0, rng=rng)
    assert len(results) == 10
    for r in results:
        assert FIM_PREFIX_ID in r


def test_deterministic_with_seed():
    """Same seed produces same results."""
    original = list(range(100, 200))
    r1 = apply_fim(original, fim_rate=1.0, rng=random.Random(42))
    r2 = apply_fim(original, fim_rate=1.0, rng=random.Random(42))
    assert r1 == r2


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test_fn in tests:
        test_fn()
        print(f"  PASS: {test_fn.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
