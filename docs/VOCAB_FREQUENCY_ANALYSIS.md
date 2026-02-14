# Vocabulary Frequency Analysis — Empirical Results

**Date**: 2026-02-14
**Corpus**: cpp_compilable_16k (3 shards, 150,000 documents, ~1.17 GB)
**Source**: 93 C++ open-source projects (boost, linux, llvm, opencv, tensorflow, protobuf, gRPC, rocksdb, clickhouse, godot, qemu, freebsd, etc.)
**Tool**: `scripts/data/analyze_vocab_frequency.py`
**Results**: `/tmp/vocab_frequency_results.json`

## Executive Summary

Of 697 proposed fixed vocabulary tokens across 30 categories, **473 (68%)** were found in the corpus. However, a critical finding: **209 of those 473 are generic English/C++ words** (e.g., `returns`, `map`, `message`, `query`, `just`) whose high counts reflect general usage, not domain-specific patterns. BPE will learn these efficiently — they don't need fixed vocab slots.

Only **~160 tokens** are both genuinely domain-specific AND appear frequently enough to justify fixed vocabulary allocation.

### The Generic Word Problem

Many proposed categories had tokens that are common C++ identifiers:

| Token | Claimed Category | Raw Count | Doc% | Reality |
|-------|-----------------|-----------|------|---------|
| `returns` | proto_keywords | 249,417 | 28.5% | Common C++ identifier |
| `map` | proto_keywords | 216,483 | 18.3% | `std::map` usage |
| `enum` | proto_keywords | 140,338 | 23.0% | C++ keyword |
| `just` | cpp26_features | 113,616 | 28.8% | Variable/comment word |
| `query` | graphql | 85,274 | 7.9% | General DB variable name |
| `Status` | grpc | 24,938 | 3.7% | Generic class name |
| `predicate` | cpp20_concepts | 21,519 | 4.0% | Common STL parameter name |

**Conclusion**: Tokens appearing in >2% of documents without special character patterns (underscores, `$`, SCREAMING_CASE) are general C++ and will be learned by BPE. Don't waste fixed vocab slots on them.

---

## Tier Classification

### TIER 1: MUST INCLUDE — High frequency, truly domain-specific

These tokens have special naming patterns (double underscores, API prefixes, SCREAMING_CASE macros) that BPE typically splits poorly.

#### Compiler Attributes & Intrinsics (20 tokens, 186K hits, 100% coverage)

| Token | Count | Docs | Doc% |
|-------|-------|------|------|
| `__func__` | 49,050 | 9,864 | 6.6% |
| `unlikely` | 38,370 | 10,270 | 6.8% |
| `likely` | 21,808 | 11,697 | 7.8% |
| `__attribute__` | 21,002 | 7,280 | 4.9% |
| `__LINE__` | 13,878 | 2,820 | 1.9% |
| `deprecated` | 10,292 | 4,780 | 3.2% |
| `fallthrough` | 7,721 | 3,526 | 2.4% |
| `__cplusplus` | 6,344 | 2,163 | 1.4% |
| `__FILE__` | 4,423 | 1,588 | 1.1% |
| `__asm__` | 3,457 | 874 | 0.6% |
| `nodiscard` | 2,824 | 429 | 0.3% |
| `maybe_unused` | 2,682 | 1,190 | 0.8% |
| `__FUNCTION__` | 2,571 | 647 | 0.4% |
| `__declspec` | 862 | 604 | 0.4% |
| `__stdcall` | 450 | 246 | 0.2% |
| `no_unique_address` | 382 | 181 | 0.1% |
| `__forceinline` | 351 | 245 | 0.2% |
| `__cdecl` | 259 | 139 | 0.1% |
| `__fastcall` | 28 | 27 | 0.0% |
| `carries_dependency` | 24 | 15 | 0.0% |

**Verdict: INCLUDE ALL 20.** BPE splits `__attribute__` into `__` + `attribute` + `__`. These need single tokens.

#### GTest Macros (27 tokens with MODERATE+, 96K hits, 100% coverage)

| Token | Count | Docs | Doc% |
|-------|-------|------|------|
| `ASSERT_FALSE` | 19,760 | 436 | 0.3% |
| `ASSERT_EQ` | 17,400 | 1,098 | 0.7% |
| `EXPECT_EQ` | 17,057 | 886 | 0.6% |
| `ASSERT_TRUE` | 12,964 | 881 | 0.6% |
| `TEST` | 8,727 | 1,851 | 1.2% |
| `EXPECT_TRUE` | 5,084 | 605 | 0.4% |
| `EXPECT_FALSE` | 3,683 | 410 | 0.3% |
| `TEST_F` | 3,593 | 892 | 0.6% |
| `EXPECT_THAT` | 1,882 | 157 | 0.1% |
| `ASSERT_NE` | 862 | 282 | 0.2% |
| `TEST_P` | 804 | 296 | 0.2% |
| `EXPECT_NE` | 570 | 184 | 0.1% |
| `EXPECT_CALL` | 471 | 38 | 0.0% |
| `EXPECT_DEATH` | 401 | 93 | 0.1% |
| `TYPED_TEST` | 356 | 75 | 0.1% |
| `ASSERT_THAT` | 306 | 52 | 0.0% |
| `ASSERT_GT` | 280 | 110 | 0.1% |
| `SetUp` | 279 | 227 | 0.2% |
| `EXPECT_GT` | 278 | 63 | 0.0% |
| `ASSERT_GE` | 210 | 89 | 0.1% |
| `ASSERT_LT` | 203 | 82 | 0.1% |
| `EXPECT_LT` | 151 | 42 | 0.0% |
| `TearDown` | 138 | 109 | 0.1% |
| `ASSERT_LE` | 128 | 60 | 0.0% |
| `ON_CALL` | 128 | 18 | 0.0% |
| `EXPECT_LE` | 122 | 41 | 0.0% |
| `EXPECT_GE` | 111 | 41 | 0.0% |

**Verdict: INCLUDE TOP 27.** `ASSERT_EQ` → BPE: `ASS`+`ERT`+`_`+`EQ`. These are critical for test code generation.

Also include 9 lower-frequency GTest tokens (TYPED_TEST_SUITE, MOCK_METHOD, EXPECT_NO_THROW, SetUpTestSuite, ASSERT_DEATH, TearDownTestSuite, ASSERT_NO_THROW, EXPECT_THROW, ASSERT_THROW) for completeness = **36 total**.

#### MySQL C API (18 tokens, 4.2K hits, 100% coverage)

All 18 tokens found. `mysql_errno` (843), `mysql_query` (466), `mysql_close` (413) are top.

**Verdict: INCLUDE ALL 18.** The `mysql_` prefix pattern is consistently split by BPE.

#### SQLite3 C API (29 tokens, 673 hits, 97% coverage)

All but `sqlite3_prepare_v3` found. Individual counts are low (1-157) but the `sqlite3_` prefix is a consistent BPE-unfriendly pattern.

**Verdict: INCLUDE TOP 20** (those with count >= 2). Drop 9 single-occurrence tokens and `sqlite3_prepare_v3`.

#### Protobuf API (21 tokens with MODERATE+, 19.6K hits)

| Token | Count | Note |
|-------|-------|------|
| `Message` | 10,118 | Generic but also protobuf-specific class |
| `Descriptor` | 5,729 | Domain-specific |
| `Arena` | 1,173 | Domain-specific |
| `DebugString` | 515 | Domain-specific |
| `FieldDescriptor` | 324 | Domain-specific |
| `MergeFrom` | 266 | Domain-specific |
| `FileDescriptor` | 253 | Domain-specific |
| `EnumDescriptor` | 212 | Domain-specific |
| `ServiceDescriptor` | 167 | Domain-specific |
| `OneofDescriptor` | 166 | Domain-specific |
| `DescriptorPool` | 134 | Domain-specific |

**Verdict: INCLUDE TOP 15** compound names (`FieldDescriptor`, `MergeFrom`, `DebugString`, etc.) that BPE splits. Skip single generic words (`Message`, `Arena`) that BPE handles fine.

---

### TIER 2: INCLUDE SELECTIVELY — Moderate frequency, domain-specific

#### SQL Keywords in String Literals (58 tokens, 31K hits, 100% coverage)

SQL keywords appear in `"SELECT ... FROM ..."` string contexts. Top keywords:
- `SELECT` (5,071), `FROM` (3,081), `WHERE` (2,088), `TABLE` (1,701), `CREATE` (1,257)
- All 58 found, but these are UPPERCASE words that BPE handles reasonably well
- Most are short single words (SELECT, FROM, WHERE) that BPE won't split

**Verdict: INCLUDE TOP 30** (those with HIGH+ status: SELECT through DROP). Skip 28 lower-frequency keywords. BPE handles short uppercase words well, but having fixed tokens for common SQL ensures consistent tokenization in string contexts.

#### CUDA Qualifiers (13 found, 801 hits)

| Token | Count | BPE Issue |
|-------|-------|-----------|
| `__restrict__` | 364 | Double-underscore pattern |
| `blockIdx` | 139 | camelCase compound |
| `__device__` | 62 | Double-underscore pattern |
| `blockDim` | 54 | camelCase compound |
| `__host__` | 39 | Double-underscore pattern |
| `__global__` | 32 | Double-underscore pattern |
| `dim3` | 30 | Short, unlikely to be split |
| `gridDim` | 23 | camelCase compound |
| `__shared__` | 21 | Double-underscore pattern |
| `warpSize` | 13 | camelCase compound |
| `threadIdx` | 12 | camelCase compound |
| `__constant__` | 10 | Double-underscore pattern |

**Verdict: INCLUDE TOP 12.** Low counts but these `__double_underscore__` patterns are consistently mangled by BPE. Essential for CUDA code generation.

#### CUDA Runtime (top 10 of 32 found, 322 total hits)

| Token | Count |
|-------|-------|
| `cudaStream_t` | 75 |
| `cudaDeviceProp` | 34 |
| `cudaMemcpyAsync` | 22 |
| `cudaSuccess` | 21 |
| `cudaStreamSynchronize` | 19 |
| `cudaSetDevice` | 17 |
| `cudaDeviceGetAttribute` | 12 |
| `cudaMemcpy` | 11 |
| `cudaMemcpyHostToDevice` | 10 |
| `cudaGetDeviceProperties` | 9 |

**Verdict: INCLUDE TOP 15** (count >= 5). The `cuda` prefix is well-known but long compound names like `cudaMemcpyHostToDevice` and `cudaStreamSynchronize` benefit from single tokens.

#### cuBLAS (top 6, 187 total hits)

`cublasOperation_t` (62), `cublasHandle_t` (47), `cublasStatus_t` (21), `cublasLtMatmul` (17), `cublasSetStream` (15), `cublasCreate` (6).

**Verdict: INCLUDE TOP 6.** Low frequency but consistent API patterns.

#### Redis Commands (top 12, 1.2K hits)

`hiredis` (293), `PUBLISH` (262), `XADD` (239), `XREADGROUP` (122), `XREAD` (61), `ZRANGE` (60), `SUBSCRIBE` (54), `UNSUBSCRIBE` (44), `ZADD` (30), `PSUBSCRIBE` (24), `SREM` (16), `HDEL` (12).

**Verdict: INCLUDE TOP 12.** Short uppercase commands are fine for BPE, but `hiredis`, `XREADGROUP`, `PSUBSCRIBE` benefit.

#### CMake Functions (top 10, 5.2K hits)

`install` (5,049) dominates — generic word, skip. Remaining: `CMAKE_BUILD_TYPE` (36), `CMAKE_INSTALL_PREFIX` (20), `find_path` (19), etc.

**Verdict: INCLUDE 8 CMAKE_* variables only.** Skip function names that BPE handles fine.

#### Catch/Boost.Test (top 5, 26.9K hits)

`CHECK` (10,813) — already C++ generic. `SECTION` (7,798), `THEN` (5,941), `WHEN` (1,035) — generic words. `REQUIRE` (786), `TEST_CASE` (511).

**Verdict: INCLUDE `TEST_CASE`, `BOOST_AUTO_TEST_CASE`, `BOOST_AUTO_TEST_SUITE`, `BOOST_CHECK`, `BOOST_REQUIRE`, `SCENARIO`, `GIVEN`** = 7 tokens with distinctive naming. Skip generic words.

---

### TIER 3: SKIP — Generic words that BPE handles well

These tokens appear in the proposal but are common English/C++ words. BPE will learn them as whole tokens because they appear so frequently. Fixed vocab slots are wasted on them.

| Category | Skip Count | Examples |
|----------|-----------|---------|
| proto_keywords | 18 | `returns`, `map`, `enum`, `message`, `required`, `stream`, `option`, `reserved`, `optional`, `service`, `syntax`, `package`, `import`, `extend` |
| cpp26_features | 8 | `just`, `transfer`, `schedule`, `scheduler`, `receiver`, `sender`, `substitute` |
| cpp23_types | 5 | `expected`, `unexpected`, `generator`, `unreachable`, `stacktrace` |
| cpp23_ranges | 5 | `chunk`, `stride`, `zip`, `adjacent`, `enumerate` |
| cpp20_concepts | 6 | `predicate`, `regular`, `integral`, `movable`, `copyable`, `destructible` |
| graphql | 7 | `query`, `Field`, `fragment`, `Argument`, `Document`, `mutation`, `subscription` |
| grpc | 8 | `Status`, `Channel`, `Server`, `Service`, `NOT_FOUND`, `UNIMPLEMENTED`, `ABORTED` |
| catch_boost_test | 4 | `CHECK`, `SECTION`, `THEN`, `WHEN` |

**Total: ~61 generic tokens to REMOVE from fixed vocab.**

These words appear in 1-30% of all documents and will naturally become whole BPE tokens during training. Dedicating fixed vocab slots to them wastes space that could go to more BPE merges.

---

### TIER 4: DROP ENTIRELY — Near-zero corpus presence

| Category | Proposed | Found | Total Hits | Verdict |
|----------|----------|-------|------------|---------|
| cuda_atomics | 21 | 2 | 7 | DROP ALL — atomicAdd (4), atomicMin (3) only |
| nccl | 17 | 2 | 11 | DROP ALL — ncclComm_t (9) and ncclUniqueId (2) only |
| rocblas_miopen | 14 | 1 | 10 | DROP ALL — hipblasStatus_t (10) only |
| xla_ops | 15 | 2 | 45 | DROP ALL — XlaOp (42), XlaBuilder (3) only |
| hip_runtime | 28 | 9 | 40 | DROP ALL — hipStream_t (17), hipError_t (13) top |
| mongodb_dollar | 39 | 11 | 395 | DROP ALL — $count (132), $match (111) top |
| cpp_orms | 13 | 5 | 121 | DROP ALL — rowset (61), got_data (34) top |
| odbc_api | 20 | 12 | 82 | DROP ALL — SQLBindParameter (18) top |
| cudnn | 16 | 8 | 46 | DROP ALL — cudnnHandle_t (27) top |
| thrust_cub | 12 | 3 | 238 | KEEP 2 — device_vector (111), host_vector (97) |

**Total: ~193 tokens to DROP.** These APIs are virtually absent from the 93-project corpus. Even the top tokens in each category appear in <0.02% of documents.

**Exception**: Keep `device_vector` (111 hits) and `host_vector` (97 hits) from thrust_cub.

---

## Morpheme Analysis Results

All morpheme categories show extremely high corpus presence:

| Category | Proposed | Found | Total Occurrences |
|----------|----------|-------|-------------------|
| Prefixes | 24 | 24 (100%) | 1,268,358 |
| Suffixes | 23 | 22 (96%) | 350,550 |
| C++ Stems | 30 | 30 (100%) | 6,806,956 |
| Common Components | 52 | 52 (100%) | 15,882,542 |

### Top Morpheme Components (appearing as identifier sub-parts)

**C++ Stems** (top 15):
`lock` (685K), `read` (670K), `write` (569K), `start` (511K), `init` (500K), `create` (424K), `format` (336K), `end` (277K), `parse` (249K), `find` (243K), `load` (243K), `open` (236K), `alloc` (198K), `remove` (187K), `push` (169K)

**Common Components** (top 15):
`value` (1.73M), `list` (1.0M), `param` (858K), `node` (855K), `index` (763K), `offset` (714K), `context` (712K), `buffer` (694K), `count` (594K), `event` (579K), `length` (480K), `config` (479K), `ptr` (451K), `tree` (443K), `next` (412K)

**Verdict: STRONG SUPPORT for morpheme-aware BPE.** These stems and components are the building blocks of C++ identifiers. A morpheme-aware BPE that preferentially learns `init` + `ialize`, `alloc` + `ate`, `de` + `init` would produce better tokenization than pure byte-level BPE.

---

## Revised Token Budget

Based on empirical data, the fixed vocabulary should be restructured:

### Current Proposal (697 domain-specific tokens across IDs 5300-6999)

| Range | Proposed Count | Category |
|-------|---------------|----------|
| 5300-5799 | 500 | GPU/Accelerator tokens |
| 5800-6299 | 500 | SQL domain tokens |
| 6300-6599 | 300 | Query/DB tokens |
| 6600-6799 | 200 | C++23/26 additions |
| 6800-6999 | 200 | Testing/build framework |
| **Total** | **1,700** | |

### Revised Recommendation (data-driven, ~340 tokens)

| Category | Count | Tokens |
|----------|-------|--------|
| Compiler attributes | 20 | `__func__`, `__attribute__`, `__LINE__`, etc. |
| GTest macros | 36 | `ASSERT_EQ`, `EXPECT_EQ`, `TEST_F`, etc. |
| MySQL C API | 18 | `mysql_errno`, `mysql_query`, etc. |
| SQLite3 C API | 20 | `sqlite3_free`, `sqlite3_exec`, etc. |
| SQL keywords (top) | 30 | `SELECT`, `FROM`, `WHERE`, etc. |
| CUDA qualifiers | 12 | `__device__`, `__global__`, `__shared__`, etc. |
| CUDA runtime (top) | 15 | `cudaStream_t`, `cudaMemcpy`, etc. |
| cuBLAS (top) | 6 | `cublasHandle_t`, `cublasOperation_t`, etc. |
| Protobuf API (compound) | 15 | `FieldDescriptor`, `MergeFrom`, `DebugString`, etc. |
| Redis (top) | 12 | `hiredis`, `XREADGROUP`, `PSUBSCRIBE`, etc. |
| Catch/Boost.Test | 7 | `TEST_CASE`, `BOOST_AUTO_TEST_CASE`, etc. |
| CMake variables | 8 | `CMAKE_BUILD_TYPE`, `CMAKE_INSTALL_PREFIX`, etc. |
| gRPC (compound only) | 5 | `ClientContext`, `ServerContext`, `CompletionQueue`, etc. |
| MongoDB C++ | 3 | `mongocxx`, `bsoncxx`, `aggregate` |
| Thrust | 2 | `device_vector`, `host_vector` |
| C++23/26 (compound only) | 15 | `source_location`, `flat_map`, `flat_set`, `move_only_function`, etc. |
| C++20 concepts (compound) | 10 | `forward_iterator`, `equality_comparable`, `input_range`, etc. |
| HIP (top 2) | 2 | `hipStream_t`, `hipError_t` |
| **TOTAL** | **~236** | |

**Savings: ~460 fewer fixed tokens** → more room for BPE merges (which learn the generic words better than fixed tokens anyway).

The remaining ~100 slots (to round to ~340) should be allocated to additional compound identifiers discovered during BPE training analysis or reserved for future expansion.

---

## Methodology Notes

### What was measured
- **Identifier extraction**: `[a-zA-Z_]\w*(?:::\w+)*` regex on full document text
- **$-operator extraction**: `\$[a-zA-Z_]\w*` for MongoDB operators
- **SQL keyword extraction**: Keywords searched inside string literals (`"..."`)
- **Morpheme analysis**: Identifiers split on `_` and camelCase boundaries, components counted

### Limitations
1. **No context disambiguation**: `Status` counted whether it's `grpc::Status` or `enum Status` or variable name
2. **String literal SQL**: Only SQL keywords in `"..."` strings counted; might miss raw identifiers used in SQL builders
3. **3 of 21 shards**: Analysis covers ~15% of cpp_compilable_16k. Results are representative but not exhaustive
4. **Morpheme Doc%=0**: The morpheme analysis counts component occurrences but doesn't track unique documents (by design — focuses on frequency)

### Reproducibility
```bash
# Run full analysis
.venv/bin/python3 -m scripts.data.analyze_vocab_frequency \
    /tmp/shard_00000.parquet /tmp/shard_00001.parquet /tmp/shard_00002.parquet \
    --morphemes -o /tmp/vocab_frequency_results.json

# View specific category
.venv/bin/python3 -m scripts.data.analyze_vocab_frequency \
    /tmp/shard_00000.parquet -c gtest

# JSON output for programmatic analysis
.venv/bin/python3 -m scripts.data.analyze_vocab_frequency \
    /tmp/shard_00000.parquet --json
```
