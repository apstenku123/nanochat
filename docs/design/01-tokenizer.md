# C++ Native BPE Tokenizer Design

**Status**: Draft v1
**Related bd feature**: `cpp-tokenizer`
**Design doc**: `docs/design/01-tokenizer.md`

---

## Overview

Standard LLM tokenizers (GPT-4 cl100k_base, Llama SentencePiece) waste 60-70% of vocabulary on natural language tokens that are irrelevant for C++ code. We train a custom BPE tokenizer exclusively on C/C++ source code to maximize token density and context utilization.

---

## Vocabulary Design

### Target Size: 32,000 tokens

Rationale:
- GPT-4: 100k tokens (covers all Unicode, 170+ languages)
- Llama 3: 128k tokens
- Our model: only needs C++ syntax + technical English in comments
- 32k is sufficient and produces denser embeddings

### Token Categories

| Category              | Examples                                         | Approximate Count |
| --------------------- | ------------------------------------------------ | ----------------- |
| C++ keywords          | `template`, `typename`, `constexpr`, `noexcept`  | ~90               |
| Operators (atomic)    | `::`, `->`, `&&`, `\|\|`, `<<`, `>>`, `==`, `!=` | ~40               |
| STL common tokens     | `std::vector`, `std::string`, `unique_ptr`       | ~200              |
| BPE merges (code)     | Common subwords from C++ identifiers             | ~28,000           |
| BPE merges (comments) | Technical English subwords                       | ~3,000            |
| Special tokens        | See below                                        | ~20               |
| Whitespace            | Space (0x20), newline (0x0A)                     | 2                 |
| Digits/hex            | `0`-`9`, `a`-`f`, `0x` prefix                    | ~20               |

### Special Tokens

```python
SPECIAL_TOKENS = [
    "<BOS>",              # Beginning of sequence
    "<EOS>",              # End of sequence
    "<PAD>",              # Padding
    "<UNK>",              # Unknown (should never appear)
    "<CODE_START>",       # Start of code block
    "<CODE_END>",         # End of code block
    "<THOUGHT_START>",    # Start of reasoning block
    "<THOUGHT_END>",      # End of reasoning block
    "<QUERY_TOOL>",       # Agent tool request
    "<INDEX>",            # Embedding extraction point
    "<FIM_PREFIX>",       # Fill-in-the-middle: prefix
    "<FIM_MIDDLE>",       # Fill-in-the-middle: middle (target)
    "<FIM_SUFFIX>",       # Fill-in-the-middle: suffix
    "<DEBUG_CONTEXT>",    # Debugger state follows
    "<FILE_SEP>",         # Separator between files in context
]
```

---

## Pre-Tokenization

Before BPE merges, text is split by a regex to prevent cross-boundary merges.

### C++ Aware Split Pattern

```python
CPP_SPLIT_PATTERN = r"""
    # C++ operators (must come before single-char fallback)
    ::|\->|\.\.\.|<<=|>>=|&&|\|\||==|!=|<=|>=|<<|>>|\+\+|--|
    \+=|-=|\*=|/=|%=|&=|\|=|\^=|
    # Hex literals (keep together)
    0[xX][0-9a-fA-F]+|
    # Numeric literals
    [0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?[fFlLuU]*|
    # Identifiers (camelCase and snake_case split naturally by BPE)
    [a-zA-Z_][a-zA-Z0-9_]*|
    # String/char literals (keep content together)
    \"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)'|
    # Preprocessor directives
    \#\s*(?:include|define|ifdef|ifndef|endif|pragma|if|else|elif|undef|error|warning)\b|
    # Single characters (braces, parens, semicolons, etc.)
    [{}()\[\];,?~!@$]|
    # Whitespace (separate)
    \n|\s+
"""
```

### Why Custom Pre-Tokenization Matters

**Bad** (generic BPE): `std::vector<int>` → `std`, `::`, `vector`, `<`, `int`, `>` (6 tokens)
**Good** (our BPE): After training, `std::vector` becomes 1 token, `<int>` stays as `<`, `int`, `>` (4 tokens)

The pre-tokenizer ensures `::` is always a single unit, preventing BPE from merging it with adjacent identifiers in wrong ways.

---

## Whitespace Strategy

Since we normalize code (strip indentation) before training:

- **Keep**: Single space (0x20) and newline (0x0A)
- **Remove from vocab**: Tab (0x09), multi-space sequences, indentation tokens
- **Rationale**: Code is passed through `clang-format` post-generation, so model doesn't need to learn formatting

---

## Hex Address Handling

Debugger output contains memory addresses like `0x7ffee1234abc`. These should tokenize efficiently:

- Pre-tokenizer keeps `0x[0-9a-f]+` as one unit
- BPE learns common address prefixes: `0x7fff`, `0x0000`
- Individual hex digits (`0`-`9`, `a`-`f`) are atomic tokens for address arithmetic

---

## Training Procedure

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.pre_tokenizers import Split
import re

# 1. Initialize BPE model
tokenizer = Tokenizer(models.BPE())

# 2. Set pre-tokenizer with C++ regex
tokenizer.pre_tokenizer = Split(
    pattern=CPP_SPLIT_PATTERN,
    behavior="isolated",
    invert=False
)

# 3. Configure trainer
trainer = trainers.BpeTrainer(
    vocab_size=32000,
    min_frequency=2,
    special_tokens=SPECIAL_TOKENS,
    show_progress=True,
    initial_alphabet=list(set(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        " \n{}()[]<>;:.,+-*/%&|^~!?=#@$_\\\"\'"
    ))
)

# 4. Train on normalized C++ corpus
tokenizer.train(files=["normalized_cpp_corpus.txt"], trainer=trainer)

# 5. Save
tokenizer.save("cpp_tokenizer.json")
```

### Training Data for Tokenizer

- 10GB of normalized (stripped indentation) C++ code
- Sources: Linux kernel, LLVM, Boost, Chromium, top GitHub C++ repos
- Include comments (model needs to read them)
- Exclude license headers

---

## Integration with nanochat

The tokenizer must implement the same interface as `nanochat/tokenizer.py`:

```python
class CppTokenizer:
    """C++ specialist tokenizer compatible with nanochat pipeline."""

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text."""

    def encode_ordinary_batch(self, texts: list[str], num_threads: int = 8) -> list[list[int]]:
        """Batch encode without special tokens."""

    @property
    def vocab_size(self) -> int:
        return 32000

    @property
    def bos_token_id(self) -> int: ...
    @property
    def eos_token_id(self) -> int: ...
```

---

## Efficiency Comparison (Expected)

For a typical C++ file (~100 lines, 3KB):

| Tokenizer           | Tokens   | Tokens/byte |
| ------------------- | -------- | ----------- |
| GPT-4 (cl100k)      | ~900     | 0.30        |
| Llama 3 (128k)      | ~850     | 0.28        |
| **CppReason (32k)** | **~400** | **0.13**    |

Expected **2-3x improvement** in tokens per byte for C++ code.

---

## Validation Checklist

After training, verify:

1. `std::vector<int>` tokenizes to <=4 tokens
2. `::` is always a single token
3. `->` is always a single token
4. `0x7ffee1234abc` tokenizes to <=4 tokens
5. Common keywords (`template`, `constexpr`, `noexcept`) are single tokens
6. `\n` is a single token
7. Multi-space sequences are NOT single tokens (they shouldn't appear in normalized input)
8. Comments tokenize efficiently: `// Check null pointer` → ~4 tokens
