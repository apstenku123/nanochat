# Next-Gen C++ Tokenizer Proposal: v2 (48K) and v3 (64K)

## Current State (v1: 32K)

**Deployed**: 32,768 tokens = 1,600 fixed + 31,168 learned BPE

| Range | Count | Category |
|-------|-------|----------|
| 0-19 | 20 | Special tokens |
| 20-199 | 180 | C++ keywords |
| 200-399 | 200 | Operators |
| 400-499 | 100 | Preprocessor |
| 500-699 | 200 | Punctuation |
| 700-1599 | 900 | STL/stdlib |
| 1600-32767 | 31,168 | Learned BPE |

**Problems with v1**:
1. No thinking tokens (`<THINK_START>`, `<THINK_END>`, etc.)
2. No ChaiScript/Ch scripting tokens (model thinks in C++, not Python)
3. No number pattern tokens (hex `0xFF`, floats `3.14`, scientific `1e10`)
4. Pure BPE for learned portion — doesn't leverage C++ morphological structure
5. 254 reserved/unused slots in fixed vocab wasted
6. Noisy comments/labels from non-native speakers poorly handled by pure BPE

---

## Proposed Architecture

### v2: 48K Tokenizer (49,152 tokens)

| Range | Count | Category | Change from v1 |
|-------|-------|----------|----------------|
| 0-63 | 64 | **Special tokens** | +44 (thinking, scripting, ChaiScript) |
| 64-319 | 256 | C++ keywords | +76 (C++23/26, concepts, modules) |
| 320-639 | 320 | Operators | +120 (compound, trigraph, digraph) |
| 640-799 | 160 | Preprocessor | +60 (modules, feature test macros) |
| 800-1199 | 400 | **Number patterns** | NEW (hex, float, scientific, binary) |
| 1200-1499 | 300 | Punctuation + whitespace | +100 (indent levels) |
| 1500-4499 | 3000 | STL/stdlib functions | +2100 (C++20/23 ranges, concepts) |
| 4500-4699 | 200 | **ChaiScript/Ch tokens** | NEW |
| 4700-4899 | 200 | **C++ morphemes** | NEW (common stems/suffixes) |
| 4900-5099 | 200 | Numbers 0-199 | Reduced from 1000 (rest via patterns) |
| 5100-5299 | 200 | **Common identifier stems** | NEW (morpheme-aware) |
| 5300-5499 | 200 | Reserved | Expansion room |
| 5500-49151 | 43,652 | **Morpheme-aware BPE** | +12,484 learned merges |

### v3: 64K Tokenizer (65,536 tokens)

Same fixed structure as v2 (5,500 fixed), plus:
- 60,036 morpheme-aware BPE tokens (vs 43,652 in v2)
- More aggressive morpheme merges, better compression ratio

---

## New Special Tokens (IDs 0-63)

### Thinking Tokens (the model thinks in C++)
```
ID  Token                Purpose
──  ─────                ───────
0   <PAD>                Padding
1   <UNK>                Unknown
2   <BOS>                Begin of sequence
3   <EOS>                End of sequence
4   <FIM_PREFIX>         FIM prefix marker
5   <FIM_MIDDLE>         FIM middle marker
6   <FIM_SUFFIX>         FIM suffix marker
7   <CODE_START>         Code block start
8   <CODE_END>           Code block end
9   <THINK_START>        Begin thinking (replaces THOUGHT_START)
10  <THINK_END>          End thinking
11  <QUERY_TOOL>         Tool query
12  <INDEX>              Codebase indexing
13  <DEBUG_CONTEXT>      Debug context injection
14  <FILE_SEP>           File separator
15  <DIFF_START>         Diff block start
16  <DIFF_END>           Diff block end
17  <COMMENT_START>      Comment block start
18  <COMMENT_END>        Comment block end
19  <TOOL_RESULT>        Tool result (moved from 19)
20  <THINK_CODE>         Thinking: analyzing code
21  <THINK_ERROR>        Thinking: reasoning about error
22  <THINK_FIX>          Thinking: proposing fix
23  <THINK_VERIFY>       Thinking: verifying solution
24  <THINK_PLAN>         Thinking: planning approach
25  <THINK_TRACE>        Thinking: tracing execution
26  <SCRIPT_START>       Begin ChaiScript/Ch block
27  <SCRIPT_END>         End ChaiScript/Ch block
28  <SCRIPT_RESULT>      Script execution result
29  <COMPILE_START>      Compilation attempt start
30  <COMPILE_END>        Compilation result
31  <COMPILE_OK>         Compilation success
32  <COMPILE_ERROR>      Compilation error
33  <TEST_START>         Test execution start
34  <TEST_END>           Test execution end
35  <TEST_PASS>          Test passed
36  <TEST_FAIL>          Test failed
37  <AST_NODE>           AST node reference
38  <SYMBOL_REF>         Symbol cross-reference
39  <TYPE_INFO>          Type information
40  <SCOPE_ENTER>        Scope entry
41  <SCOPE_EXIT>         Scope exit
42  <INCLUDE_CONTEXT>    Include file context
43  <TEMPLATE_INST>      Template instantiation context
44  <OVERLOAD_SET>       Overload resolution context
45-63 <RESERVED_N>       Future expansion (19 slots)
```

### Why ChaiScript/Ch Instead of Python REPL

The model is a C++ specialist that **thinks in C++**. Its scripting language should be C++-compatible:

**ChaiScript** (primary): Header-only C++ embedded scripting
- Syntax: `var x = 5; fun square(x) { x * x }` — feels like C++
- Shares keywords: `for`, `while`, `if`, `class`, `auto`, `var`, `return`
- Unique tokens needed: `def`, `fun`, `attr`, `bind`, `Dynamic_Object`, `method_missing`, `eval`, `use`, `:=` (reference assign)

**Ch** (secondary): C/C++ interpreter with computational arrays
- Full C/C++ syntax compatibility (Ch IS C with extensions)
- Unique tokens: `string_t`, `array double`, `foreach`, computational arrays
- Good for numerical reasoning during thinking

### ChaiScript/Ch Fixed Tokens (IDs 4500-4699)

```
# ChaiScript-specific keywords not already in C++
def, fun, attr, bind, var (already in C++20 contextually)
Dynamic_Object, method_missing, set_explicit, call_exists
eval, eval_file, use, import (ChaiScript), namespace (ChaiScript-style)

# ChaiScript built-ins
back, bob_back, collate, concat, drop_while, drop, dump_system
empty, even, filter, foldl, for_each, front, generate_range
get_arity, get_contained_functions, is_type, join, product
puts, reduce, retro, reverse, size, sum, take_while, take
to_string, zip_with, zip

# Ch extensions
string_t, generic_t, foreach
array double, array int, array float, array complex

# ChaiScript operators
:=  (reference assignment — already in operators range)

# Script interaction tokens
chai_eval, chai_define, ch_run, ch_eval
```

---

## Number Pattern Tokens (IDs 800-1199)

Instead of storing 1000 individual numbers (0-999), store **pattern templates**:

### Hex Patterns (800-849, 50 tokens)
```
0x0-0xF (16 single hex digits)
0x00-0xFF (common byte values: 0x00, 0xFF, 0x80, 0x7F, etc.)
0xDEAD, 0xBEEF, 0xCAFE, 0xBABE (magic numbers)
0x0000, 0xFFFF, 0x8000 (16-bit boundaries)
```

### Float Patterns (850-899, 50 tokens)
```
0.0, 1.0, 2.0, 0.5, 0.1, 0.01, 0.001
1.0f, 0.0f, 0.5f, 2.0f (float suffixes)
3.14, 2.71, 1.41, 1.73 (mathematical constants)
-1.0, -0.5, -1.0f
1e-6, 1e-3, 1e-9, 1e3, 1e6 (common engineering)
```

### Scientific Notation (900-929, 30 tokens)
```
1e0-1e9 (powers of 10)
1e-1 through 1e-9 (negative powers)
2e0-9e0 (single digit mantissa common)
```

### Binary Literals (930-949, 20 tokens)
```
0b0, 0b1, 0b00, 0b01, 0b10, 0b11
0b0000, 0b1111, 0b10000000, 0b11111111
```

### Common Integer Literals (950-1199, 250 tokens)
```
0-199 (most common small integers)
256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536
1000, 10000, 100000, 1000000
INT_MAX, INT_MIN, UINT_MAX, SIZE_MAX (named constants)
```

---

## Morpheme-Aware BPE

### The Problem with Pure BPE for Code

Pure BPE treats code as a flat byte stream. It doesn't know that:
- `get_buffer_size` = `get` + `_` + `buffer` + `_` + `size` (3 morphemes)
- `CheckpointManager` = `Checkpoint` + `Manager` (2 morphemes)
- `toString` = `to` + `String` (2 morphemes via camelCase)

BPE might merge `get_bu` as one token and `ffer_size` as another — terrible for the model's ability to understand naming patterns.

### Solution: Morpheme-Guided BPE (MorphBPE for Code)

Based on research from:
- **MorphPiece** (Google, 2024): Morphological tokenization for LLMs
- **BPE-knockout** (Gutierrez et al., 2023): Constraining BPE merges at morpheme boundaries
- **SubwordRegularization** (Kudo, 2018): Multiple segmentation sampling

Our approach: **Pre-segment identifiers at morpheme boundaries before BPE training**.

#### C++ Identifier Morpheme Rules

```python
def segment_cpp_identifier(ident: str) -> list[str]:
    """Split C++ identifier into morphemes."""
    parts = []
    # 1. Split on underscores: get_buffer_size -> [get, buffer, size]
    for chunk in ident.split('_'):
        if not chunk:
            continue
        # 2. Split camelCase: CheckpointManager -> [Checkpoint, Manager]
        # 3. Split UPPER runs: HTTPSConnection -> [HTTPS, Connection]
        morphemes = split_camel_case(chunk)
        parts.extend(morphemes)
    return parts
```

#### Common C++ Morphemes as Fixed Tokens (IDs 4700-4899)

**Prefixes** (30 tokens):
```
pre, post, un, re, de, dis, non, sub, super, over
under, inter, intra, multi, poly, mono, bi, tri, quad
semi, pseudo, meta, proto, anti, counter, co, ex, out, in, up
```

**Suffixes** (40 tokens):
```
able, ible, tion, sion, ment, ness, ful, less, ous, ive
ize, ify, ate, ent, ant, ary, ory, ery, ure, ance
ence, ity, ty, dom, ship, ward, wise, like, fold, most
er, or, ist, ian, ed, ing, ly, al, ic, ical
```

**C++ specific stems** (30 tokens):
```
alloc, dealloc, init, deinit, lock, unlock, push, pop
read, write, open, close, begin, end, start, stop
create, destroy, insert, remove, find, search, sort, swap
load, save, parse, format, encode, decode
```

**Common identifier components** (100 tokens):
```
buffer, cache, queue, stack, list, tree, node, edge, graph
handler, manager, factory, builder, adapter, wrapper, proxy
callback, listener, observer, visitor, iterator, generator
config, context, session, request, response, message, event
value, index, count, total, offset, length, capacity, limit
```

### BPE Training Modifications

The BPE trainer gets a pre-segmented input:

```python
# Before BPE training, pre-segment identifiers at morpheme boundaries
# using a special boundary marker that BPE cannot merge across

def preprocess_for_morphbpe(text: str) -> str:
    """Insert boundary markers at morpheme splits."""
    # Identifiers are already isolated by our pre-tokenizer
    # For each identifier token, split at morpheme boundaries
    # and insert \x00 (null byte) that BPE won't merge across
    tokens = pre_tokenize(text)
    result = []
    for tok in tokens:
        if is_identifier(tok):
            morphemes = segment_cpp_identifier(tok)
            result.append('\x00'.join(morphemes))
        else:
            result.append(tok)
    return ' '.join(result)
```

This ensures BPE learns merges **within** morphemes (good: `buff` + `er` -> `buffer`) but not **across** boundaries (bad: `get_bu` + `ffer` should stay as `get` + `buffer`).

---

## Phoneme-Aware Handling for Noisy Comments

### The Problem

C++ codebases contain comments and string literals written by non-native English speakers with:
- Misspellings: "retrun", "lenght", "recieve"
- Phonetic approximations: "teh" for "the", "wiht" for "with"
- Transliterations from other languages

### Solution: Phoneme-Aware BPE for Comments

Instead of pure character BPE for comment text, we add:

1. **Common misspelling normalization during training**:
   Pre-process training data to normalize the most common misspellings in comments, so the model learns canonical spellings.

2. **Phoneme-inspired subword units**:
   Add fixed tokens for common English phoneme clusters that appear in technical writing:

```
# Phoneme-inspired subword units (IDs 5100-5299)
# These capture sound patterns that persist across misspellings

# Common consonant clusters
str, scr, spr, spl, thr, chr, shr, sch  (already good as BPE)
# Common vowel patterns
tion, sion, ment, ness, able, ible, ance, ence
# Technical writing stems
param, config, init, alloc, dealloc, iter, struct
# Common comment words as single tokens
TODO, FIXME, HACK, NOTE, WARNING, DEPRECATED
IMPORTANT, WORKAROUND, TEMPORARY, CLEANUP
```

3. **Phonetic distance in BPE merge scoring**:
   During BPE training, bias merge scores for pairs that share phonetic similarity. This naturally groups "recieve"/"receive" under similar subword patterns.

---

## Vocab Size Comparison

| | v1 (32K) | v2 (48K) | v3 (64K) |
|---|---|---|---|
| **Fixed tokens** | 1,600 | 5,500 | 5,500 |
| **Learned BPE** | 31,168 | 43,652 | 60,036 |
| **Total** | 32,768 | 49,152 | 65,536 |
| **Thinking tokens** | 0 | 11 | 11 |
| **Script tokens** | 0 | 200 | 200 |
| **Number patterns** | 1000 ints | 400 patterns | 400 patterns |
| **Morpheme tokens** | 0 | 200 | 200 |
| **Identifier stems** | 0 | 200 | 200 |
| **Morpheme-aware BPE** | No | Yes | Yes |
| **Est. bytes/token** | ~5.3 | ~6.5 | ~7.2 |
| **Est. token reduction** | baseline | ~18% fewer | ~26% fewer |

---

## Implementation Plan

### Phase 1: Fixed Vocab Design (1 day)
1. Create `data/cpp_tokenizer_v2/fixed_vocab.json` with 5,500 tokens
2. Add ChaiScript/Ch tokens to the special token ranges
3. Add thinking tokens (THINK_CODE, THINK_ERROR, etc.)
4. Add number pattern tokens (hex, float, scientific, binary)
5. Add morpheme tokens (prefixes, suffixes, stems)

### Phase 2: Morpheme-Aware BPE Training (2 days)
1. Implement `segment_cpp_identifier()` morpheme splitter
2. Modify `tok_train_cpp.py` to use morpheme-guided pre-processing
3. Add boundary markers to prevent cross-morpheme merges
4. Train on full C++ corpus with morpheme constraints
5. Validate morpheme boundary preservation

### Phase 3: Integration & Testing (1 day)
1. Update `CppTokenizer` to handle new special tokens
2. Add thinking token support to `Engine.generate()`
3. Add ChaiScript/Ch script execution support
4. Benchmark compression ratio vs v1
5. Verify backward-compatible with existing model (can't change mid-training)

### Phase 4: Training (next experiment)
1. Start new training run with v2 tokenizer
2. A/B test v2 vs v1 on identical data

---

## Migration Strategy

- **Current runs**: Keep v1 (32K). Cannot change mid-training.
- **Next experiment**: Use v2 (48K) or v3 (64K).
- **Checkpoint conversion**: Not possible (embedding table size changes). Must train from scratch.
- The `get_token_bytes()` function already computes dynamically from tokenizer, so no version management needed.

---

## References

- [MorphPiece](https://arxiv.org/abs/2307.07262) — Google's morphological tokenization for multilingual LLMs
- [BPE-knockout](https://arxiv.org/abs/2306.07141) — Constraining BPE at morpheme boundaries
- [ChaiScript](https://chaiscript.com/) — Header-only C++ embedded scripting
- [Ch Language](https://www.softintegration.com/) — C/C++ interpreter with computational extensions
- [SubwordRegularization](https://arxiv.org/abs/1804.10959) — Multiple segmentation for robust tokenization
- [Charformer](https://arxiv.org/abs/2106.12672) — Gradient-based subword tokenization
