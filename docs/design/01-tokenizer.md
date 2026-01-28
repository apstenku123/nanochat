# C++ Hybrid Tokenizer Design: Fixed Vocabulary + Learned BPE

**Status**: Draft v2 — Hybrid design (BERT-style whitespace + fixed C++ vocab + learned BPE)
**Related bd feature**: `cpp-tokenizer`
**Design doc**: `docs/design/01-tokenizer.md`

---

## Core Design Principles

1. **Spaces are delimiters, not tokens** (BERT-style). Whitespace splits input into words but is never encoded. Reconstruction inserts spaces between tokens. Only `\n` (newline) is an explicit token — it carries structural meaning in C++.

2. **Fixed vocabulary for C++ primitives**. Keywords, operators, STL names, numbers 0-999 are assigned static IDs that BPE never touches. These are guaranteed single tokens at all times.

3. **Learned BPE for everything else**. Identifiers, comments (English), string literals, and unknown patterns are handled by BPE merges trained on C++ corpus.

4. **Git diff awareness**. Unified diff markers (`+`, `-`, `@@`) are fixed tokens for processing merge requests and commit diffs.

---

## Vocabulary Layout

### Target Size: 32,768 tokens

```
ID Range        Category                    Count     Notes
─────────────────────────────────────────────────────────────
0-19            Control/Special tokens      20        BOS, EOS, PAD, FIM, etc.
20-119          C/C++ keywords              ~100      static, const, template, etc.
120-179         Operators (multi-char)       ~60       ::, ->, &&, ||, ==, <<, etc.
180-219         Preprocessor directives      ~40       #include, #define, #ifdef, etc.
220-319         Single-char punctuation      ~100      { } ( ) [ ] ; , . etc.
320-519         STL / common names           ~200      std, vector, string, cout, etc.
520-1519        Numbers 0-999               1000      Fixed: "0", "1", ..., "999"
1520-1535       Diff markers                 16        +, -, @@, diff, ---, +++, etc.
1536-1551       Structural whitespace        16        \n, \n\n (and reserved)
1552-1599       Reserved                     48        Future use
─────────────────────────────────────────────────────────────
1600-32767      Learned BPE merges          31168     Identifiers, comments, strings
```

Total: 32,768 (power of 2, efficient for embeddings)

---

## Fixed Vocabulary: Detailed Lists

### Control / Special Tokens (ID 0-19)

```python
SPECIAL_TOKENS = [
    "<PAD>",              # 0  - Padding
    "<UNK>",              # 1  - Unknown (should never appear)
    "<BOS>",              # 2  - Beginning of sequence / document delimiter
    "<EOS>",              # 3  - End of sequence
    "<FIM_PREFIX>",       # 4  - Fill-in-the-middle: prefix
    "<FIM_MIDDLE>",       # 5  - Fill-in-the-middle: middle (target)
    "<FIM_SUFFIX>",       # 6  - Fill-in-the-middle: suffix
    "<CODE_START>",       # 7  - Start of code block
    "<CODE_END>",         # 8  - End of code block
    "<THOUGHT_START>",    # 9  - Start of reasoning block
    "<THOUGHT_END>",      # 10 - End of reasoning block
    "<QUERY_TOOL>",       # 11 - Agent tool request
    "<INDEX>",            # 12 - Embedding extraction point
    "<DEBUG_CONTEXT>",    # 13 - Debugger state follows
    "<FILE_SEP>",         # 14 - Separator between files in context
    "<DIFF_START>",       # 15 - Start of diff block
    "<DIFF_END>",         # 16 - End of diff block
    "<COMMENT_START>",    # 17 - Start of comment/reasoning section
    "<COMMENT_END>",      # 18 - End of comment/reasoning section
    "<RESERVED_19>",      # 19 - Reserved
]
```

### C/C++ Keywords (ID 20-119) — NEVER split by BPE

```python
CPP_KEYWORDS = [
    # Storage / type qualifiers
    "auto", "const", "constexpr", "consteval", "constinit",
    "extern", "inline", "mutable", "register", "static",
    "thread_local", "volatile", "virtual", "explicit",
    # Types
    "void", "bool", "char", "short", "int", "long",
    "float", "double", "signed", "unsigned", "wchar_t",
    "char8_t", "char16_t", "char32_t", "size_t",
    # Type modifiers
    "struct", "class", "union", "enum", "typedef",
    "typename", "template", "concept", "requires",
    "namespace", "using",
    # Control flow
    "if", "else", "switch", "case", "default",
    "for", "while", "do", "break", "continue",
    "return", "goto",
    # Exception
    "try", "catch", "throw", "noexcept",
    # Memory
    "new", "delete", "nullptr", "sizeof", "alignof", "alignas",
    # Cast
    "static_cast", "dynamic_cast", "const_cast", "reinterpret_cast",
    # Access
    "public", "private", "protected", "friend",
    # Logical / other
    "true", "false", "this",
    "operator", "decltype", "typeid",
    "co_await", "co_return", "co_yield",
    # C-specific
    "NULL", "typedef", "restrict",
    # Common macros treated as keywords
    "assert", "define", "include", "ifdef", "ifndef",
    "endif", "pragma", "elif", "undef",
]
```

### Multi-Character Operators (ID 120-179) — Atomic tokens

```python
CPP_OPERATORS = [
    # Scope / member access
    "::", "->", ".*", "->*",
    # Comparison
    "==", "!=", "<=", ">=", "<=>",
    # Logical
    "&&", "||",
    # Bitwise shift
    "<<", ">>",
    # Increment / decrement
    "++", "--",
    # Compound assignment
    "+=", "-=", "*=", "/=", "%=",
    "&=", "|=", "^=", "<<=", ">>=",
    # Variadic
    "...",
    # Preprocessor
    "##",
    # Comment markers (as tokens, not content)
    "//", "/*", "*/",
]
```

### Single-Character Punctuation (ID 220-319)

All ASCII punctuation as individual fixed tokens:
```
{ } ( ) [ ] < > ; : , . + - * / % & | ^ ~ ! ? = # @ $ _ \ " '
```

### STL / Common Library Names (ID 320-519) — Single tokens

```python
STL_NAMES = [
    # Namespace
    "std", "boost", "absl", "fmt",
    # I/O
    "cout", "cerr", "cin", "endl", "printf", "fprintf", "sprintf",
    "scanf", "puts", "getchar", "putchar",
    # Containers
    "vector", "map", "set", "list", "deque", "array",
    "unordered_map", "unordered_set", "stack", "queue",
    "priority_queue", "pair", "tuple",
    "string", "string_view", "wstring",
    # Smart pointers
    "unique_ptr", "shared_ptr", "weak_ptr", "make_unique", "make_shared",
    # Memory
    "allocator", "malloc", "calloc", "realloc", "free",
    "memcpy", "memset", "memmove",
    # Algorithms
    "sort", "find", "count", "transform", "accumulate",
    "begin", "end", "size", "empty", "push_back", "emplace_back",
    "insert", "erase", "clear", "reserve", "resize",
    "front", "back", "data",
    # Iterators
    "iterator", "const_iterator", "reverse_iterator",
    # Types / utilities
    "optional", "variant", "any", "expected",
    "function", "bind", "move", "forward", "swap",
    "numeric_limits", "type_traits",
    "enable_if", "is_same", "decay",
    "initializer_list",
    # Streams
    "ifstream", "ofstream", "stringstream", "ostringstream",
    "iostream", "fstream", "sstream",
    # Threading
    "mutex", "lock_guard", "unique_lock", "shared_lock",
    "thread", "atomic", "condition_variable",
    "future", "promise", "async",
    # Error
    "exception", "runtime_error", "logic_error",
    "invalid_argument", "out_of_range", "overflow_error",
    "error_code", "error_category",
    # Common C functions
    "strlen", "strcmp", "strncmp", "strcpy", "strcat",
    "atoi", "atof", "strtol", "strtod",
    "exit", "abort", "atexit",
    # POSIX common
    "open", "close", "read", "write", "ioctl",
    "socket", "bind", "listen", "accept", "connect",
    "send", "recv", "select", "poll", "epoll",
    "fork", "exec", "wait", "pipe", "signal",
    "pthread_create", "pthread_join", "pthread_mutex",
    # CUDA common
    "cudaMalloc", "cudaFree", "cudaMemcpy",
    "cudaDeviceSynchronize", "cudaGetLastError",
    "__global__", "__device__", "__host__", "__shared__",
    "blockIdx", "threadIdx", "blockDim", "gridDim",
]
```

### Numbers 0-999 (ID 520-1519) — Fixed tokens

Every integer from `"0"` to `"999"` is a dedicated token. This means:
- `42` → 1 token (ID 562)
- `1024` → 2 tokens: `"1024"` is not in fixed vocab, pre-tokenizer splits it as identifier, BPE handles it
- `0xFF` → 2 tokens: `"0"` (fixed) + `"xFF"` (BPE) — or handled by hex literal rule
- `3.14` → `"3"` + `"."` + `"14"` = 3 tokens

**Why 0-999**: Covers line numbers in debug traces, small constants, array sizes, loop bounds, error codes, HTTP status codes. Matches GPT-4's approach.

### Diff / Git Markers (ID 1520-1535)

```python
DIFF_TOKENS = [
    "+",        # Added line (in diff context)
    "-",        # Removed line (in diff context)
    "@@",       # Hunk header
    "diff",     # diff command marker
    "---",      # Old file header
    "+++",      # New file header
    "index",    # Index line in git diff
    "a/",       # Old file path prefix
    "b/",       # New file path prefix
]
```

Note: `+` and `-` also exist as single-char punctuation. In diff context, the pre-tokenizer recognizes line-initial `+`/`-` as diff markers.

### Structural Whitespace (ID 1536-1551)

```python
WHITESPACE_TOKENS = [
    "\n",       # Newline — the ONLY whitespace that is a token
    "\n\n",     # Double newline (paragraph/function separator)
]
```

**All other whitespace (spaces, tabs) are DELIMITERS ONLY.** They split input into words but are never encoded as tokens. During decoding, spaces are inserted between tokens (BERT-style reconstruction).

---

## Pre-Tokenization: BERT-Style Whitespace + C++ Awareness

### Algorithm

```python
from tokenizers.pre_tokenizers import Sequence, WhitespaceSplit, Split
from tokenizers import Regex

cpp_pre_tokenizer = Sequence([
    # Step 1: Isolate all fixed-vocabulary tokens
    # This prevents BPE from ever merging across them
    Split(Regex(
        r"""(?x)
        # Multi-char operators (longest match first)
        <=>|<<=|>>=|->\\*|\\.\\*|
        ::|->|==|!=|<=|>=|&&|\|\||<<|>>|\+\+|--|
        \+=|-=|\*=|/=|%=|&=|\|=|\^=|<<=|>>=|
        \\.\\.\\.|##|//|/\\*|\\*/|
        # Diff markers at line start
        ^\\+\\+\\+|^---|^@@|
        # Hex literals
        0[xX][0-9a-fA-F]+|
        # Numbers 0-999 (word boundary)
        \\b[0-9]{1,3}\\b|
        # Newlines (structural)
        \\n|
        # String/char literals
        \"(?:[^\"\\\\]|\\\\.)*\"|'(?:[^'\\\\]|\\\\.)'|
        # Preprocessor
        \\#\\s*(?:include|define|ifdef|ifndef|endif|pragma|if|else|elif|undef|error|warning)\\b
        """
    ), behavior="isolated"),

    # Step 2: Split remaining text on whitespace (BERT-style)
    # Spaces become delimiters, NOT tokens
    WhitespaceSplit(),
])
```

### How It Works — Example

Input: `std::vector<int> *ptr = nullptr;`

1. **Fixed token isolation**: `std` `::` `vector` `<` `int` `>` `*` `ptr` `=` `nullptr` `;`
2. **Whitespace split**: Spaces between tokens are consumed (not encoded)
3. **BPE on remaining**: `ptr` is not in fixed vocab → BPE encodes it (likely as single learned token)

Result: `[std] [::] [vector] [<] [int] [>] [*] [ptr] [=] [nullptr] [;]` = 11 tokens

Compare GPT-4: `std::vector<int> *ptr = nullptr;` → ~12-15 tokens (with space tokens wasted)

### Decoding (Reconstruction)

Since spaces are not tokens, we need a rule for reinserting them:

```python
def decode(self, ids: list[int]) -> str:
    tokens = [self.id_to_token(id) for id in ids]
    result = []
    for i, token in enumerate(tokens):
        if token == "\n" or token == "\n\n":
            result.append(token)
        elif i > 0 and not self._is_punctuation(tokens[i-1]) and not self._is_punctuation(token):
            # Insert space between two non-punctuation tokens
            result.append(" ")
            result.append(token)
        else:
            result.append(token)
    return "".join(result)
```

The rule: insert space between two "word" tokens, but NOT between punctuation/operators and their neighbors. Since we run `clang-format` on output anyway, minor spacing errors don't matter.

---

## BPE Training (Learned Portion)

### What BPE Learns (IDs 1600-32767)

The BPE portion handles:
- **Identifiers**: `ProcessPacket`, `buffer_size`, `HandleRequest` → subword merges
- **English in comments**: `"pointer"`, `"function"`, `"returns"` → subword merges
- **String literal content**: Whatever appears in `"..."` strings
- **Hex digits beyond 0-999**: `0xFFEE`, `1024`, `65536`
- **Unknown patterns**: Anything not in the fixed vocabulary

### Training Procedure

```python
from tokenizers import Tokenizer, models, trainers

# 1. Initialize BPE with fixed vocabulary
tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))

# 2. Set BERT-style pre-tokenizer (see above)
tokenizer.pre_tokenizer = cpp_pre_tokenizer

# 3. ALL fixed tokens passed as special_tokens
#    BPE trainer will never merge them, IDs are reserved
all_fixed = (SPECIAL_TOKENS + CPP_KEYWORDS + CPP_OPERATORS
             + SINGLE_CHAR_PUNCT + STL_NAMES
             + [str(n) for n in range(1000)]
             + DIFF_TOKENS + WHITESPACE_TOKENS)

trainer = trainers.BpeTrainer(
    vocab_size=32768,
    min_frequency=2,
    special_tokens=all_fixed,   # These get IDs 0..len(all_fixed)-1
    show_progress=True,
)

# 4. Train — BPE fills remaining IDs with learned merges
tokenizer.train(files=["normalized_cpp_corpus.txt"], trainer=trainer)

# 5. Save
tokenizer.save("cpp_tokenizer.json")
```

### Training Data for Tokenizer

- 10GB of normalized (stripped indentation) C++ code with comments
- Sources: Linux kernel, LLVM, Boost, Chromium, top GitHub C++ repos
- Include English comments (BPE will learn common English subwords)
- Include git diffs (CommitPack C++ subset) for diff pattern learning
- Exclude license headers

---

## Handling Git Diffs / Merge Requests

### Diff Tokenization Example

```diff
@@ -42,6 +42,8 @@
 void Process(char* buf) {
+    if (!buf) return;
     Header* h = reinterpret_cast<Header*>(buf);
```

Tokenizes as:
```
[@@] [-] [42] [,] [6] [+] [42] [,] [8] [@@] [\n]
[void] [Process] [(] [char] [*] [buf] [)] [{] [\n]
[<DIFF_ADD>] [if] [(] [!] [buf] [)] [return] [;] [\n]
[Header] [*] [h] [=] [reinterpret_cast] [<] [Header] [*] [>] [(] [buf] [)] [;] [\n]
```

The `+`/`-` at line start are recognized by the pre-tokenizer as diff markers, not arithmetic operators.

### Commit Message Format for Training

```
<DIFF_START>
// COMMIT: Fix null pointer dereference in packet handler
@@ -42,6 +42,8 @@
 void Process(char* buf) {
+if (!buf) return;
 Header* h = reinterpret_cast<Header*>(buf);
<DIFF_END>
```

---

## Integration with nanochat

```python
class CppTokenizer:
    """Hybrid C++ tokenizer: fixed vocab + learned BPE, BERT-style whitespace."""

    def __init__(self, tokenizer_path: str):
        self._tokenizer = Tokenizer.from_file(tokenizer_path)

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs. Spaces are delimiters, not tokens."""
        return self._tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text. Reinsert spaces between word tokens."""
        # Custom space-insertion logic (see Decoding section above)
        ...

    def encode_ordinary_batch(self, texts: list[str], num_threads: int = 8) -> list[list[int]]:
        """Batch encode without special tokens."""
        return [enc.ids for enc in self._tokenizer.encode_batch(texts)]

    @property
    def vocab_size(self) -> int:
        return 32768

    @property
    def bos_token_id(self) -> int:
        return 2  # <BOS>

    @property
    def eos_token_id(self) -> int:
        return 3  # <EOS>

    @property
    def pad_token_id(self) -> int:
        return 0  # <PAD>
```

---

## Efficiency Comparison (Expected)

For a typical C++ file (~100 lines, 3KB):

| Tokenizer           | Tokens   | Tokens/byte | Space tokens wasted |
| ------------------- | -------- | ----------- | ------------------- |
| GPT-4 (cl100k)      | ~900     | 0.30        | ~15%                |
| Llama 3 (128k)      | ~850     | 0.28        | ~12%                |
| **CppReason (32k)** | **~350** | **0.12**    | **0%**              |

Expected **2.5x improvement** — from both domain-specific vocab AND eliminating space tokens.

---

## Validation Checklist

After training, verify:

1. `std::vector<int>` → `[std] [::] [vector] [<] [int] [>]` = 6 tokens (all fixed)
2. `::` is always token ID 120 (fixed operator)
3. `->` is always token ID 121 (fixed operator)
4. `nullptr` is always a single fixed token
5. `42` → single token (ID 562, fixed number)
6. `1024` → BPE handles (not in 0-999 range)
7. `0x7ffee1234abc` → `[0] [x7ffee1234abc]` or similar BPE split
8. `\n` is a single token; spaces are NOT tokens
9. `// Check null pointer` → `[//] [Check] [null] [pointer]` = 4 tokens (no space tokens)
10. `printf("hello %d\n", x);` → `[printf] [(] ["hello] [%d\n"] [,] [x] [)] [;]` = ~8 tokens
11. Git diff `+if (!buf) return;` → `[+] [if] [(] [!] [buf] [)] [return] [;]`
12. `cout << endl` → `[cout] [<<] [endl]` = 3 tokens (all fixed)
