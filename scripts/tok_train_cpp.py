"""
Train the C++ hybrid tokenizer: fixed C++ vocab + learned BPE, BERT-style whitespace.

Usage:
    python -m scripts.tok_train_cpp [--input data/cpp_clean.jsonl] [--vocab_size 32768] [--max_chars 5000000000]

See docs/design/01-tokenizer.md for full design.
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
# Fixed Vocabulary (IDs 0 .. ~1599)
# =============================================================================

# --- Control / Special Tokens (ID 0-19) ---
SPECIAL_TOKENS = [
    "<PAD>",              # 0
    "<UNK>",              # 1
    "<BOS>",              # 2
    "<EOS>",              # 3
    "<FIM_PREFIX>",       # 4
    "<FIM_MIDDLE>",       # 5
    "<FIM_SUFFIX>",       # 6
    "<CODE_START>",       # 7
    "<CODE_END>",         # 8
    "<THOUGHT_START>",    # 9
    "<THOUGHT_END>",      # 10
    "<QUERY_TOOL>",       # 11
    "<INDEX>",            # 12
    "<DEBUG_CONTEXT>",    # 13
    "<FILE_SEP>",         # 14
    "<DIFF_START>",       # 15
    "<DIFF_END>",         # 16
    "<COMMENT_START>",    # 17
    "<COMMENT_END>",      # 18
    "<RESERVED_19>",      # 19
]

# --- C/C++ Keywords (ID 20-119) ---
CPP_KEYWORDS = [
    "auto", "const", "constexpr", "consteval", "constinit",
    "extern", "inline", "mutable", "register", "static",
    "thread_local", "volatile", "virtual", "explicit",
    "void", "bool", "char", "short", "int", "long",
    "float", "double", "signed", "unsigned", "wchar_t",
    "char8_t", "char16_t", "char32_t", "size_t",
    "struct", "class", "union", "enum", "typedef",
    "typename", "template", "concept", "requires",
    "namespace", "using",
    "if", "else", "switch", "case", "default",
    "for", "while", "do", "break", "continue",
    "return", "goto",
    "try", "catch", "throw", "noexcept",
    "new", "delete", "nullptr", "sizeof", "alignof", "alignas",
    "static_cast", "dynamic_cast", "const_cast", "reinterpret_cast",
    "public", "private", "protected", "friend",
    "true", "false", "this",
    "operator", "decltype", "typeid",
    "co_await", "co_return", "co_yield",
    "NULL", "restrict",
    "assert", "define", "include", "ifdef", "ifndef",
    "endif", "pragma", "elif", "undef",
]

# --- Multi-Character Operators (ID 120-179) ---
CPP_OPERATORS = [
    "::", "->", ".*", "->*",
    "==", "!=", "<=", ">=", "<=>",
    "&&", "||",
    "<<", ">>",
    "++", "--",
    "+=", "-=", "*=", "/=", "%=",
    "&=", "|=", "^=", "<<=", ">>=",
    "...",
    "##",
    "//", "/*", "*/",
]

# --- Preprocessor directives as tokens (ID 180-219) ---
PREPROCESSOR = [
    "#include", "#define", "#ifdef", "#ifndef", "#endif",
    "#pragma", "#if", "#else", "#elif", "#undef",
    "#error", "#warning", "#line", "#import",
]

# --- Single-Character Punctuation (ID 220-319) ---
SINGLE_CHAR_PUNCT = list("{}()[]<>;:,.+-*/%&|^~!?=#@$_\\\"'")

# --- STL / Common Library Names (ID 320-519) ---
STL_NAMES = [
    "std", "boost", "absl", "fmt",
    "cout", "cerr", "cin", "endl", "printf", "fprintf", "sprintf",
    "scanf", "puts", "getchar", "putchar",
    "vector", "map", "set", "list", "deque", "array",
    "unordered_map", "unordered_set", "stack", "queue",
    "priority_queue", "pair", "tuple",
    "string", "string_view", "wstring",
    "unique_ptr", "shared_ptr", "weak_ptr", "make_unique", "make_shared",
    "allocator", "malloc", "calloc", "realloc", "free",
    "memcpy", "memset", "memmove",
    "sort", "find", "count", "transform", "accumulate",
    "begin", "end", "size", "empty", "push_back", "emplace_back",
    "insert", "erase", "clear", "reserve", "resize",
    "front", "back", "data",
    "iterator", "const_iterator", "reverse_iterator",
    "optional", "variant", "any", "expected",
    "function", "bind", "move", "forward", "swap",
    "numeric_limits", "type_traits",
    "enable_if", "is_same", "decay",
    "initializer_list",
    "ifstream", "ofstream", "stringstream", "ostringstream",
    "iostream", "fstream", "sstream",
    "mutex", "lock_guard", "unique_lock", "shared_lock",
    "thread", "atomic", "condition_variable",
    "future", "promise", "async",
    "exception", "runtime_error", "logic_error",
    "invalid_argument", "out_of_range", "overflow_error",
    "error_code", "error_category",
    "strlen", "strcmp", "strncmp", "strcpy", "strcat",
    "atoi", "atof", "strtol", "strtod",
    "exit", "abort", "atexit",
    "open", "close", "read", "write", "ioctl",
    "socket", "listen", "accept", "connect",
    "send", "recv", "select", "poll", "epoll",
    "fork", "exec", "wait", "pipe", "signal",
    "pthread_create", "pthread_join", "pthread_mutex",
    "cudaMalloc", "cudaFree", "cudaMemcpy",
    "cudaDeviceSynchronize", "cudaGetLastError",
    "__global__", "__device__", "__host__", "__shared__",
    "blockIdx", "threadIdx", "blockDim", "gridDim",
]

# --- Numbers 0-999 (ID 520-1519) ---
NUMBERS = [str(n) for n in range(1000)]

# --- Diff Markers (ID 1520-1535) ---
DIFF_TOKENS = [
    "@@", "diff", "---", "+++", "index", "a/", "b/",
]
# Note: single +/- already in SINGLE_CHAR_PUNCT

# --- Structural Whitespace (ID 1536-1551) ---
WHITESPACE_TOKENS = [
    "\n",
    "\n\n",
]

# --- Reserved (ID 1552-1599) ---
RESERVED = [f"<RESERVED_{i}>" for i in range(1552, 1600)]

# =============================================================================
# Build the complete fixed token list (order = ID assignment)
# =============================================================================

def build_fixed_tokens():
    """Build ordered list of all fixed tokens. Index = token ID."""
    fixed = []
    fixed.extend(SPECIAL_TOKENS)                    # 0-19

    # Pad keywords to 100 slots (20-119)
    kw = CPP_KEYWORDS[:100]
    kw.extend([f"<KW_RESERVED_{i}>" for i in range(len(kw), 100)])
    fixed.extend(kw)

    # Pad operators to 60 slots (120-179)
    ops = CPP_OPERATORS[:60]
    ops.extend([f"<OP_RESERVED_{i}>" for i in range(len(ops), 60)])
    fixed.extend(ops)

    # Pad preprocessor to 40 slots (180-219)
    pp = PREPROCESSOR[:40]
    pp.extend([f"<PP_RESERVED_{i}>" for i in range(len(pp), 40)])
    fixed.extend(pp)

    # Pad single-char punct to 100 slots (220-319)
    sc = SINGLE_CHAR_PUNCT[:100]
    sc.extend([f"<SC_RESERVED_{i}>" for i in range(len(sc), 100)])
    fixed.extend(sc)

    # Pad STL names to 200 slots (320-519)
    stl = STL_NAMES[:200]
    stl.extend([f"<STL_RESERVED_{i}>" for i in range(len(stl), 200)])
    fixed.extend(stl)

    # Numbers 0-999 (520-1519)
    fixed.extend(NUMBERS)

    # Diff tokens padded to 16 (1520-1535)
    dt = DIFF_TOKENS[:16]
    dt.extend([f"<DIFF_RESERVED_{i}>" for i in range(len(dt), 16)])
    fixed.extend(dt)

    # Whitespace tokens padded to 16 (1536-1551)
    ws = WHITESPACE_TOKENS[:16]
    ws.extend([f"<WS_RESERVED_{i}>" for i in range(len(ws), 16)])
    fixed.extend(ws)

    # Reserved (1552-1599)
    fixed.extend(RESERVED)

    assert len(fixed) == 1600, f"Expected 1600 fixed tokens, got {len(fixed)}"
    return fixed


# =============================================================================
# Pre-tokenizer: BERT-style whitespace + C++ operator isolation
# =============================================================================

CPP_PRE_TOKENIZER_PATTERN = r"""(?x)
    # Multi-char operators (longest match first)
    <=>|<<=|>>=|->\*|\.\*|
    ::|->|==|!=|<=|>=|&&|\|\||<<|>>|\+\+|--|
    \+=|-=|\*=|/=|%=|&=|\|=|\^=|
    \.\.\.|\#\#|//|/\*|\*/|
    # Newlines (structural whitespace token)
    \n\n|\n|
    # Single-char punctuation (each is its own token)
    [{}()\[\]<>;:,.\+\-\*/%&\|^~!?=\#@\$_\\\"']
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
    parser.add_argument('--vocab_size', type=int, default=32768, help='Total vocab size')
    parser.add_argument('--max_chars', type=int, default=5_000_000_000, help='Max chars to train on')
    parser.add_argument('--doc_cap', type=int, default=10_000, help='Max chars per document')
    parser.add_argument('--min_frequency', type=int, default=2, help='Min BPE merge frequency')
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    input_path = os.path.join(project_root, args.input)
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Vocab size: {args.vocab_size}")
    print(f"Max chars: {args.max_chars:,}")

    # Build fixed tokens
    fixed_tokens = build_fixed_tokens()
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

    # Also save the fixed vocab mapping for reference
    fixed_vocab_path = os.path.join(output_dir, "fixed_vocab.json")
    with open(fixed_vocab_path, 'w') as f:
        json.dump({tok: i for i, tok in enumerate(fixed_tokens)}, f, indent=2)
    print(f"Saved fixed vocab to {fixed_vocab_path}")

    # === Validation ===
    print("\n=== Validation ===")
    validate(tokenizer, fixed_tokens)


def validate(tokenizer, fixed_tokens):
    """Run validation checks from the design doc."""
    fixed_map = {tok: i for i, tok in enumerate(fixed_tokens)}

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
        ("::", "operator"),
        ("->", "operator"),
        ("nullptr", "keyword"),
        ("return", "keyword"),
        ("std", "STL name"),
        ("vector", "STL name"),
        ("42", "number"),
        ("0", "number"),
        ("999", "number"),
        ("\n", "whitespace"),
    ]
    for tok, cat in checks:
        if tok in fixed_map:
            expected_id = fixed_map[tok]
            vocab = tokenizer.get_vocab()
            actual_id = vocab.get(tok, -1)
            status = "OK" if actual_id == expected_id else f"MISMATCH (expected {expected_id}, got {actual_id})"
            print(f"    {tok!r:20s} ({cat:10s}) ID={actual_id:5d} {status}")
            if actual_id != expected_id:
                all_passed = False
        else:
            print(f"    {tok!r:20s} ({cat:10s}) NOT in fixed vocab!")
            all_passed = False

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

    if all_passed:
        print("\n  ALL CHECKS PASSED")
    else:
        print("\n  SOME CHECKS FAILED")


if __name__ == '__main__':
    main()
