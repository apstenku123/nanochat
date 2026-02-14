# Vocabulary Frequency Analysis — Empirical Results (Full Corpus)

**Date**: 2026-02-14
**Corpus**: `/home/dave/data/cpp_raw/` — 1,435,084 C++ source files, 22.3 GB
**Source**: 333 C++ open-source projects (boost, linux, llvm, opencv, tensorflow, protobuf, gRPC, rocksdb, clickhouse, godot, qemu, freebsd, Qt, LLDB, Eigen, Kokkos, cutlass, folly, etc.)
**Tool**: `tools/vocab_analyzer` (Rust, rayon parallel, 48 cores, 27,919 files/sec)
**Results**: `build3:/home/dave/vocab_analysis_full.json` (full JSON), `build3:/home/dave/vocab_analysis_raw.json` (initial run)
**Runtime**: 51.4s on c4-highmem-48 (48 vCPUs)

## Executive Summary

Of 697 proposed fixed vocabulary tokens across 30 categories, **647 (93%)** were found in the full 1.43M-file corpus — up from 68% on the smaller 150K-doc sample. Total hits: **6.94 million**. Key findings:

1. **93% coverage** — virtually all proposed domain tokens appear in real C++ code
2. **Morphemes are king** — 88.3M morpheme hits dwarf the 6.9M domain token hits (12.7x ratio)
3. **Comments are 24.2% of the corpus** — significant, needs proper tokenization
4. **Unicode is minimal** — only 0.020% non-ASCII bytes, 3.4% of files have non-English comments
5. **4-space indent dominates** (31%) but tabs are close behind (26%) — both need tokens
6. **Multi-spaces are 7.5%** of all space occurrences — needs a few multi-space BPE tokens

---

## Content Analysis for BPE Design

### Byte Distribution (22.3 GB total)

| Content Type | Bytes | % of Total |
|-------------|-------|-----------|
| Code (approx) | 15.97 GB | 71.6% |
| Comments | 5.39 GB | **24.2%** |
| String literals | 0.94 GB | 4.2% |

**BPE Implication**: Comments are nearly a quarter of the training corpus. Comment tokenization quality directly affects model loss. Design BPE tokens for common comment patterns.

### Line Statistics (507M lines)

| Metric | Count | % |
|--------|-------|---|
| Total lines | 506,696,380 | — |
| Blank lines | 66,139,848 | 13.1% |
| Line comments (`//`) | 34,509,252 | 6.8% |
| Block comments (`/* */`) | 43,041,990 | 8.5% |
| Total comments | 77,551,242 | 15.3% |

### Whitespace Patterns (BPE-critical)

| Pattern | Count | BPE Design Action |
|---------|-------|-------------------|
| Single spaces | 871,590,859 | Standard BPE token |
| Multi-space runs (2+) | 70,875,313 | **Add 2-space, 4-space, 8-space tokens** |
| Single tabs | 5,733,081 | Standard BPE token |
| Multi-tab runs (2+) | 5,086,292 | Add 2-tab token |
| Mixed indent (space+tab) | 9,630,280 | Handle naturally |

**Multi-space ratio: 7.5%** of all space occurrences are multi-space runs. This justifies adding a few dedicated multi-space tokens (SP×2, SP×4, SP×8) to BPE vocabulary.

### Indentation Style

| Style | Lines | % of Indented |
|-------|-------|---------------|
| 4-space | 87,059,103 | **31%** (dominant) |
| 2-space | 86,746,923 | 31% |
| Tab | 74,576,886 | 26% |
| 8-space | 35,071,760 | 12% |

**BPE Implication**: All three major indent styles (2-space, 4-space, tab) are nearly equal. The tokenizer must handle all three efficiently. Dedicate BPE tokens for `"  "` (2-space), `"    "` (4-space), `"        "` (8-space), and `"\t"`.

### Unicode / Non-English Content

| Metric | Value | % |
|--------|-------|---|
| Non-ASCII bytes | 4,481,732 | **0.020%** of total |
| UTF-8 multi-byte chars | 6,594 | Negligible |
| Docs with non-ASCII comments | 48,157 | **3.4%** of files |

**BPE Implication**: Corpus is >99.98% ASCII. Non-English content is negligible (3.4% of files, mostly accented names like `Michał Górny` and Unicode math symbols `×`, `²`, `–`). **No need for unicode-to-English normalization or translation.** Standard UTF-8 BPE fallback is sufficient.

**Non-English comment samples** (from 48K files):
- Author names with accents: `Michał Górny`
- Japanese punctuation: `。`, `．`, `｡`
- Unicode math: `2^32 – 2`, `2'b01`
- Bullet points: `•`, `…`

---

## Identifier Analysis (848M identifiers)

### Naming Style Distribution

| Style | Count | % |
|-------|-------|---|
| other/mixed | 490,476,023 | 57.8% |
| snake_case | 144,841,425 | **17.1%** |
| SCREAMING_CASE | 79,673,971 | **9.4%** |
| PascalCase | 76,670,183 | **9.0%** |
| camelCase | 56,932,523 | **6.7%** |

- **Mean identifier length**: 7.5 chars
- **Mean unique identifiers per doc**: 110

**BPE Implication**: The morpheme splitter should handle all four major naming conventions. snake_case dominates named identifiers (17.1%), but PascalCase and camelCase together are 15.7%. SCREAMING_CASE (9.4%) confirms the importance of macro tokens.

### Top Namespaces (28.2M namespace-qualified references)

| Namespace | Refs | Notes |
|-----------|------|-------|
| `std` | 6,818,050 | Standard library dominates |
| `llvm` | 552,842 | LLVM project |
| `boost` | 517,791 | Boost libraries |
| `detail` | 495,313 | Implementation namespaces |
| `cutlass` | 443,931 | NVIDIA CUTLASS |
| `absl` | 322,751 | Google Abseil |
| `proto` | 311,843 | Protobuf |
| `Kokkos` | 166,879 | Performance portability |
| `testing` | 134,409 | Google Test |
| `mlir` | 112,912 | MLIR compiler |
| `ImGui` | 105,905 | Dear ImGui |
| `Qt` | 95,394 | Qt framework |
| `Eigen` | 51,321 | Linear algebra |
| `cuda` | 47,696 | CUDA runtime |
| `torch` | 25,732 | PyTorch C++ |
| `folly` | 25,979 | Facebook Folly |

**BPE Implication**: `std::` appears 6.8M times — the `std::` prefix should be a highly efficient BPE merge. Consider ensuring `std::` is learned early in BPE training.

---

## Domain Token Analysis (697 proposed → 647 found, 93% coverage)

### Category Summary

| Category | Proposed | Found | Coverage | Total Hits |
|----------|----------|-------|----------|-----------|
| proto_keywords | 18 | 18 | 100% | 1,932,142 |
| attributes | 20 | 20 | 100% | 763,121 |
| gtest | 36 | 36 | 100% | 2,047,563 |
| catch_boost_test | 16 | 16 | 100% | 439,453 |
| grpc | 32 | 32 | 100% | 365,260 |
| protobuf | 32 | 32 | 100% | 215,480 |
| cpp23_ranges | 10 | 10 | 100% | 206,104 |
| cpp23_types | 22 | 22 | 100% | 205,904 |
| sql_keywords | 58 | 58 | 100% | 149,427 |
| cuda_qualifiers | 18 | 18 | 100% | 147,730 |
| graphql | 16 | 13 | 81% | 143,845 |
| cpp20_concepts | 29 | 29 | 100% | 77,235 |
| cpp26_features | 21 | 18 | 86% | 66,288 |
| cuda_runtime | 47 | 46 | 98% | 54,543 |
| xla_ops | 15 | 13 | 87% | 46,748 |
| sqlite3_api | 30 | 30 | 100% | 32,756 |
| thrust_cub | 12 | 12 | 100% | 11,502 |
| cuda_atomics | 21 | 21 | 100% | 10,256 |
| mysql_api | 18 | 18 | 100% | 9,972 |
| mongodb_cpp | 11 | 11 | 100% | 5,444 |
| hip_runtime | 28 | 28 | 100% | 3,248 |
| cublas | 20 | 20 | 100% | 2,711 |
| cmake | 22 | 13 | 59% | 2,356 |
| cudnn | 16 | 16 | 100% | 1,949 |
| odbc_api | 20 | 20 | 100% | 1,184 |
| nccl | 17 | 17 | 100% | 930 |
| redis_commands | 26 | 25 | 96% | 987 |
| cpp_orms | 13 | 9 | 69% | 488 |
| mongodb_dollar | 39 | 12 | 31% | 194 |
| rocblas_miopen | 14 | 14 | 100% | 154 |
| **TOTAL** | **697** | **647** | **93%** | **6,944,974** |

### Key observation from full corpus vs. sample

The full 1.43M-file corpus (333 projects) shows dramatically higher coverage than the 150K-doc sample (93% vs 68%). Low-frequency domain tokens (CUDA, HIP, ROCm) that were absent from the sample now have significant counts because the full corpus includes specialized GPU libraries (CUTLASS: 444K namespace refs, CUDA: 148K qualifier hits).

---

## Morpheme Analysis (88.3M total hits)

Morpheme analysis shows the strongest signal for BPE design:

### Summary

| Category | Found/Proposed | Total Hits | Coverage |
|----------|---------------|-----------|---------|
| Common Components | 52/52 | **59,903,189** | 100% |
| C++ Stems | 30/30 | **22,949,545** | 100% |
| Prefixes | 24/24 | **4,185,596** | 100% |
| Suffixes | 22/23 | **1,254,255** | 96% |
| **TOTAL** | **128/129** | **88,292,585** | **99%** |

### Top C++ Stems (building blocks of identifiers)

| Stem | Hits | Example Identifiers |
|------|------|-------------------|
| `init` | 2,191,060 | initialize, initBuffer, deinit |
| `read` | 1,888,838 | readFile, readBuffer, readOnly |
| `write` | 1,795,113 | writeData, writeBuffer, writeBack |
| `create` | 1,758,587 | createHandler, createBuffer |
| `start` | 1,689,929 | startTimer, startProcess |
| `format` | 1,508,143 | formatString, formatOutput |
| `end` | 1,364,319 | endBlock, endTransaction |
| `lock` | 1,162,779 | lockMutex, unlock, readLock |
| `begin` | 1,057,396 | beginTransaction, beginBlock |
| `load` | 909,964 | loadConfig, loadData |
| `push` | 835,909 | pushBack, pushFront |
| `parse` | 802,442 | parseJSON, parseConfig |
| `find` | 685,086 | findFirst, findByName |
| `alloc` | 682,877 | allocBuffer, dealloc |
| `insert` | 582,370 | insertNode, insertRow |

### Top Common Components (the nouns of C++ identifiers)

| Component | Hits | Example Identifiers |
|-----------|------|-------------------|
| `value` | 6,188,458 | getValue, setValue, defaultValue |
| `index` | 3,319,836 | getIndex, indexBuffer |
| `offset` | 3,314,628 | byteOffset, fileOffset |
| `node` | 2,966,716 | treeNode, childNode |
| `ptr` | 2,840,443 | dataPtr, sharedPtr |
| `buffer` | 2,730,513 | readBuffer, ringBuffer |
| `count` | 2,722,740 | refCount, itemCount |
| `context` | 2,526,817 | renderContext, execContext |
| `list` | 2,507,437 | nodeList, fileList |
| `tree` | 1,822,536 | parseTree, binaryTree |

### Top Prefixes

| Prefix | Hits | Example |
|--------|------|---------|
| `proto` | 757,214 | prototype, protocol |
| `sub` | 592,008 | subClass, subTree |
| `non` | 385,989 | nonNull, nonEmpty |
| `multi` | 347,615 | multiThread, multiLine |
| `meta` | 272,370 | metadata, metaClass |
| `mono` | 265,752 | monomorphic, monoState |

**BPE Design Conclusion**: Morpheme-aware BPE is strongly validated. The tokenizer should learn these stems and components as BPE merges, ensuring `initialize` → `init` + `ialize` rather than `in` + `iti` + `ali` + `ze`.

---

## C++ Keyword Frequency (169M total)

The top 20 C++ keywords account for the vast majority of keyword occurrences:

| Keyword | Count | Doc% |
|---------|-------|------|
| `if` | 20,058,320 | 42.6% |
| `const` | 18,529,109 | 56.8% |
| `return` | 16,622,711 | 57.8% |
| `int` | 13,127,390 | 54.6% |
| `void` | 10,879,929 | 60.5% |
| `struct` | 8,718,210 | 36.6% |
| `static` | 6,684,701 | 33.2% |
| `case` | 5,103,770 | 12.0% |
| `char` | 4,820,932 | 32.6% |
| `typename` | 4,292,020 | 11.1% |
| `else` | 4,086,961 | 26.2% |
| `unsigned` | 3,986,714 | 20.4% |
| `bool` | 3,930,633 | 30.2% |
| `break` | 3,163,346 | 15.2% |
| `for` | 3,074,496 | 26.6% |
| `false` | 2,870,273 | 20.1% |
| `auto` | 2,850,877 | 13.5% |
| `true` | 2,271,015 | 20.0% |
| `template` | 2,081,518 | 14.9% |
| `nullptr` | 1,414,505 | 10.8% |

**BPE Implication**: All C++ keywords will naturally become whole BPE tokens due to extreme frequency. No fixed vocab slots needed for keywords — BPE learns them in the first few merges.

---

## BPE Tokenizer Design Recommendations

Based on the full 22.3 GB corpus analysis:

### 1. Whitespace Tokenization
- **Single space**: Standard BPE token (872M occurrences)
- **Multi-space tokens**: Add `SP×2`, `SP×4`, `SP×8` (71M multi-space runs = 7.5% of spaces)
- **Tab**: Standard BPE token (5.7M)
- **Newline**: Standard `\n` token (507M lines)
- **No need for**: multi-tab tokens (only 5M), mixed indent handling (BPE handles naturally)

### 2. Comment Tokenization
- Comments are **24.2% of corpus** — substantial
- `//` and `/*` `*/` should be efficient BPE tokens
- Comment content is >99.98% ASCII — no special unicode handling needed
- Common comment starters: `// `, `/* `, ` * ` (Doxygen)

### 3. Unicode Handling
- **Skip unicode-to-English normalization** — only 0.020% non-ASCII bytes
- **Skip comment translation** — only 3.4% of files have non-English comments
- Standard UTF-8 byte fallback is sufficient
- The non-English content is mostly author names and Unicode math symbols

### 4. Fixed Vocabulary (domain-specific tokens that BPE splits poorly)
- **~340 tokens** from Tier 1-2 analysis (compiler attrs, GTest macros, CUDA, SQL APIs)
- Focus on tokens with `__double_underscore__`, `SCREAMING_CASE`, or long compound names
- Skip generic English words — BPE learns them naturally

### 5. Morpheme-Aware BPE Training
- Seed BPE with morpheme stems as initial merges: `init`, `read`, `write`, `create`, etc.
- Common components (`value`, `index`, `offset`, `node`, `ptr`, `buffer`) should be early merges
- Prefixes (`sub`, `non`, `multi`, `meta`, `proto`) and suffixes (`or`, `ic`, `al`, `less`) as merge hints

### 6. Separator Tokens for C/C++
- **Command separator**: `;` followed by newline or space
- **Block delimiters**: `{`, `}` with surrounding whitespace patterns
- **Include guard**: `#ifndef`, `#define`, `#endif` (high frequency: >200K each as preprocessor directives)
- **Namespace**: `::` (28.2M namespace-qualified refs, `std::` alone is 6.8M)

---

## Reproducibility

### Rust Tool (full corpus, 48-core)
```bash
# Build
cd tools/vocab_analyzer && cargo build --release

# Full analysis (all modes, 22.3 GB, ~50s on 48 cores)
./target/release/vocab-analyzer /path/to/cpp_raw/ --all --content \
    -o results.json > report.txt 2>&1

# Quick test (100 files)
./target/release/vocab-analyzer /path/to/cpp_raw/ --all --content --max-files 100

# Category filter
./target/release/vocab-analyzer /path/to/cpp_raw/ -c cuda --top 20
```

### Python Tool (smaller samples)
```bash
.venv/bin/python3 -m scripts.data.analyze_vocab_frequency \
    /path/to/shard.parquet --morphemes -o results.json
```
