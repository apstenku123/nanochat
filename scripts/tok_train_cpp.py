"""
Train the C++ hybrid tokenizer: fixed C++ vocab + learned BPE, BERT-style whitespace.

Usage:
    python -m scripts.tok_train_cpp [--input data/cpp_clean.jsonl] [--vocab_size 65536] [--max_chars 5000000000]

See docs/design/01-tokenizer.md for full design.

Fixed vocabulary structure (4800 tokens, IDs 0-4799):
    - IDs 0-19: Special tokens (20)
    - IDs 20-199: C++ keywords (180)
    - IDs 200-399: Operators (200)
    - IDs 400-499: Preprocessor (100)
    - IDs 500-699: Punctuation (200)
    - IDs 700-3699: STL/Standard library functions (3000)
    - IDs 3700-4699: Numbers 0-999 (1000)
    - IDs 4700-4799: Reserved (100)
    - IDs 4800-65535: Learned BPE (60736)
"""
import os
import json
import time
import argparse

from tokenizers import Tokenizer, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Sequence, Split, WhitespaceSplit
from tokenizers.decoders import BPEDecoder


# =============================================================================
# Load Fixed Vocabulary from JSON
# =============================================================================

def load_fixed_vocab(vocab_path: str) -> list[str]:
    """Load fixed vocabulary from JSON file. Returns ordered list where index = token ID."""
    with open(vocab_path, 'r') as f:
        vocab_dict = json.load(f)
    # Sort by ID to get ordered list
    max_id = max(vocab_dict.values())
    fixed_tokens = [''] * (max_id + 1)
    for token, id_ in vocab_dict.items():
        fixed_tokens[id_] = token
    return fixed_tokens


def build_fixed_tokens(vocab_path: str = None) -> list[str]:
    """Build ordered list of all fixed tokens. Index = token ID."""
    if vocab_path is None:
        # Default path relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        vocab_path = os.path.join(script_dir, '..', 'data', 'cpp_tokenizer', 'fixed_vocab.json')
    return load_fixed_vocab(vocab_path)


# =============================================================================
# Pre-tokenizer: BERT-style whitespace + C++ operator isolation
# =============================================================================

CPP_PRE_TOKENIZER_PATTERN = r"""(?x)
    # Multi-char operators (longest match first)
    <=>|<<=|>>=|->\*|\.\*|
    ::|->|==|!=|<=|>=|&&|\|\||<<|>>|\+\+|--|
    \+=|-=|\*=|/=|%=|&=|\|=|\^=|
    \.\.\.|\#\#|//|/\*|\*/|
    # Diff markers
    \@\@|---|\+\+\+|
    # Newlines (structural whitespace token)
    \n\n|\n|
    # Single-char punctuation (each is its own token)
    [{}()\[\]<>;:,.\+\-\*/%&\|^~!?=\#@\$_\\\"'`]
"""

def build_pre_tokenizer():
    """BERT-style: isolate fixed patterns then split on whitespace."""
    return Sequence([
        Split(Regex(CPP_PRE_TOKENIZER_PATTERN), behavior="isolated"),
        WhitespaceSplit(),
    ])


# =============================================================================
# Text iterator from JSONL
# =============================================================================

def text_iterator(input_path: str, max_chars: int, doc_cap: int = 10_000):
    nchars = 0
    with open(input_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get('text', '')
            if len(text) > doc_cap:
                text = text[:doc_cap]
            nchars += len(text)
            yield text
            if nchars >= max_chars:
                return
    print(f"Read {nchars:,} chars from {input_path}")


# =============================================================================
# Main: train tokenizer
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train C++ hybrid BPE tokenizer')
    parser.add_argument('--input', default='data/cpp_clean.jsonl', help='Input JSONL file')
    parser.add_argument('--output_dir', default='data/cpp_tokenizer', help='Output directory')
    parser.add_argument('--vocab_size', type=int, default=65536, help='Total vocab size (default: 65536)')
    parser.add_argument('--max_chars', type=int, default=5_000_000_000, help='Max chars to train on')
    parser.add_argument('--doc_cap', type=int, default=10_000, help='Max chars per document')
    parser.add_argument('--min_frequency', type=int, default=2, help='Min BPE merge frequency')
    parser.add_argument('--fixed_vocab', default=None, help='Path to fixed_vocab.json (default: data/cpp_tokenizer/fixed_vocab.json)')
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    input_path = os.path.join(project_root, args.input)
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Resolve fixed vocab path
    if args.fixed_vocab:
        fixed_vocab_path = args.fixed_vocab
    else:
        fixed_vocab_path = os.path.join(output_dir, 'fixed_vocab.json')

    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Fixed vocab: {fixed_vocab_path}")
    print(f"Vocab size: {args.vocab_size}")
    print(f"Max chars: {args.max_chars:,}")

    # Build fixed tokens from JSON
    fixed_tokens = build_fixed_tokens(fixed_vocab_path)
    print(f"Fixed tokens: {len(fixed_tokens)}")
    print(f"Learned BPE slots: {args.vocab_size - len(fixed_tokens)}")

    # Initialize tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = build_pre_tokenizer()

    # Train BPE
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=fixed_tokens,  # These get IDs 0..1599 in order
        show_progress=True,
    )

    print("\n=== Training BPE ===")
    t0 = time.time()
    text_iter = text_iterator(input_path, args.max_chars, args.doc_cap)
    tokenizer.train_from_iterator(text_iter, trainer=trainer)
    t1 = time.time()
    print(f"Training time: {t1 - t0:.1f}s")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    # Save
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"Saved to {tokenizer_path}")

    # === Validation ===
    print("\n=== Validation ===")
    validate(tokenizer, fixed_tokens)


def validate(tokenizer, fixed_tokens):
    """Run validation checks for the tokenizer."""
    fixed_map = {tok: i for i, tok in enumerate(fixed_tokens) if tok}

    tests = [
        # (input, description, expected_properties)
        ("std::vector<int>", "STL template", {
            "max_tokens": 7,  # std :: vector < int > (6 meaningful)
        }),
        ("nullptr", "keyword as single token", {
            "exact_tokens": 1,
        }),
        ("return 42;", "keyword + number + punct", {
            "max_tokens": 5,
        }),
        ("cout << endl", "stream output", {
            "max_tokens": 5,
        }),
        ("if (!buf) return;", "null check pattern", {
            "max_tokens": 10,
        }),
        ("cudaMalloc(&ptr, size)", "CUDA API call", {
            "max_tokens": 10,
        }),
        ("std::unique_ptr<T>", "smart pointer", {
            "max_tokens": 7,
        }),
    ]

    all_passed = True
    for text, desc, props in tests:
        enc = tokenizer.encode(text)
        ids = enc.ids
        tokens = enc.tokens
        print(f"\n  {desc}: \"{text}\"")
        print(f"    Tokens ({len(ids)}): {tokens}")
        print(f"    IDs: {ids}")

        if "exact_tokens" in props and len(ids) != props["exact_tokens"]:
            print(f"    FAIL: expected {props['exact_tokens']} tokens, got {len(ids)}")
            all_passed = False
        if "max_tokens" in props and len(ids) > props["max_tokens"]:
            print(f"    FAIL: expected <= {props['max_tokens']} tokens, got {len(ids)}")
            all_passed = False

    # Check fixed token IDs
    print("\n  Fixed token ID checks:")
    checks = [
        # (token, category, expected_id_range_start, expected_id_range_end)
        ("::", "operator", 200, 400),
        ("->", "operator", 200, 400),
        ("nullptr", "keyword", 20, 200),
        ("return", "keyword", 20, 200),
        ("std", "STL name", 700, 3700),
        ("vector", "STL name", 700, 3700),
        ("cudaMalloc", "CUDA API", 700, 3700),
        ("42", "number", 3700, 4700),
        ("0", "number", 3700, 4700),
        ("999", "number", 3700, 4700),
    ]
    vocab = tokenizer.get_vocab()
    for tok, cat, range_start, range_end in checks:
        if tok in fixed_map:
            expected_id = fixed_map[tok]
            actual_id = vocab.get(tok, -1)
            in_range = range_start <= actual_id < range_end
            status = "OK" if actual_id == expected_id and in_range else f"MISMATCH (expected {expected_id}, got {actual_id})"
            print(f"    {tok!r:20s} ({cat:10s}) ID={actual_id:5d} {status}")
            if actual_id != expected_id:
                all_passed = False
        else:
            print(f"    {tok!r:20s} ({cat:10s}) NOT in fixed vocab!")
            # Not a failure if not in fixed vocab - might be learned

    # Decode test
    print("\n  Decode test:")
    test_code = "int main() {\nreturn 0;\n}"
    enc = tokenizer.encode(test_code)
    decoded = tokenizer.decode(enc.ids)
    print(f"    Input:   {test_code!r}")
    print(f"    Decoded: {decoded!r}")
    print(f"    Match: {decoded.strip() == test_code.strip()}")

    # Efficiency test
    print("\n  Efficiency comparison:")
    sample = """#include <iostream>
#include <vector>
#include <string>

int main() {
std::vector<std::string> names;
names.push_back("hello");
names.push_back("world");

for (const auto& name : names) {
std::cout << name << std::endl;
}

return 0;
}"""
    enc = tokenizer.encode(sample)
    bytes_count = len(sample.encode('utf-8'))
    tokens_count = len(enc.ids)
    print(f"    Sample: {bytes_count} bytes -> {tokens_count} tokens")
    print(f"    Tokens/byte: {tokens_count/bytes_count:.3f}")
    print(f"    Bytes/token: {bytes_count/tokens_count:.1f}")

    # Fixed vocab stats
    print("\n  Fixed vocab coverage:")
    print(f"    Total fixed tokens: {len(fixed_tokens)}")
    print(f"    Keywords (20-199): 180 slots")
    print(f"    Operators (200-399): 200 slots")
    print(f"    Preprocessor (400-499): 100 slots")
    print(f"    Punctuation (500-699): 200 slots")
    print(f"    STL/stdlib (700-3699): 3000 slots")
    print(f"    Numbers (3700-4699): 1000 slots")
    print(f"    Reserved (4700-4799): 100 slots")

    if all_passed:
        print("\n  ALL CHECKS PASSED")
    else:
        print("\n  SOME CHECKS FAILED")


if __name__ == '__main__':
    main()
