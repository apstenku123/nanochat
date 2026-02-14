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
4. Pure BPE for learned portion -- doesn't leverage C++ morphological structure
5. 254 reserved/unused slots in fixed vocab wasted
6. Noisy comments/labels from non-native speakers poorly handled by pure BPE
7. No GPU/accelerator domain tokens (CUDA, ROCm, XLA)
8. No database/SQL tokens (SQL frequently embedded in C++ strings)
9. Missing C++23/26 standard library additions

---

## Proposed Architecture

### v3: 64K Tokenizer (65,536 tokens) — PRIMARY TARGET

Based on vocabulary scaling law research (NeurIPS 2024): mainstream LLMs are under-vocabularied. For a 270M-877M parameter C++ model, 48K-64K is the optimal range. Industry trend: 32K (2023) -> 128K (2024) -> 262K (2025, Gemini 3).

| Range | Count | Category | Change from v1 |
|-------|-------|----------|----------------|
| 0-63 | 64 | **Special tokens** | +44 (thinking, scripting, compilation) |
| 64-319 | 256 | C++ keywords (incl. C++23/26) | +76 |
| 320-639 | 320 | Operators | +120 |
| 640-799 | 160 | Preprocessor + attributes | +60 |
| 800-1199 | 400 | **Number patterns** | NEW (hex, float, scientific, binary) |
| 1200-1499 | 300 | Punctuation + whitespace | +100 (indent levels) |
| 1500-4499 | 3000 | STL/stdlib + C++20/23/26 | +2100 |
| 4500-4699 | 200 | **ChaiScript/Ch tokens** | NEW |
| 4700-4899 | 200 | **C++ morphemes** | NEW (stems/suffixes) |
| 4900-5099 | 200 | Numbers 0-199 | Reduced from 1000 |
| 5100-5299 | 200 | **Common identifier stems** | NEW (morpheme-aware) |
| 5300-5799 | 500 | **GPU/Accelerator tokens** | NEW (CUDA, ROCm, XLA) |
| 5800-6299 | 500 | **SQL domain tokens** | NEW (all dialects + C APIs) |
| 6300-6599 | 300 | **Query/DB tokens** | NEW (OQL, GraphQL, Protobuf, Redis) |
| 6600-6799 | 200 | **C++23/26 additions** | NEW (contracts, reflection, execution) |
| 6800-6999 | 200 | **Testing/build framework** | NEW (GTest, CMake, Boost.Test) |
| 7000-7199 | 200 | Reserved | Expansion room |
| 7200-65535 | 58,336 | **Morpheme-aware BPE** | Learned merges |

### v2: 48K Tokenizer (49,152 tokens) — FALLBACK

Same fixed structure (7,200 fixed), with 41,952 morpheme-aware BPE tokens.

---

## New Special Tokens (IDs 0-63)

### Thinking Tokens (the model thinks in C++)
```
ID  Token                Purpose
--  -----                -------
0   <PAD>                Padding
1   <UNK>                Unknown
2   <BOS>                Begin of sequence
3   <EOS>                End of sequence
4   <FIM_PREFIX>         FIM prefix marker
5   <FIM_MIDDLE>         FIM middle marker
6   <FIM_SUFFIX>         FIM suffix marker
7   <CODE_START>         Code block start
8   <CODE_END>           Code block end
9   <THINK_START>        Begin thinking
10  <THINK_END>          End thinking
11  <QUERY_TOOL>         Tool query
12  <INDEX>              Codebase indexing
13  <DEBUG_CONTEXT>      Debug context injection
14  <FILE_SEP>           File separator
15  <DIFF_START>         Diff block start
16  <DIFF_END>           Diff block end
17  <COMMENT_START>      Comment block start
18  <COMMENT_END>        Comment block end
19  <TOOL_RESULT>        Tool result
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

---

## ChaiScript/Ch Tokens (IDs 4500-4699)

### Why ChaiScript/Ch Instead of Python REPL

The model is a C++ specialist that **thinks in C++**. Its scripting language should be C++-compatible:

**ChaiScript** (primary): Header-only C++ embedded scripting
- Syntax: `var x = 5; fun square(x) { x * x }` -- feels like C++
- Shares keywords: `for`, `while`, `if`, `class`, `auto`, `var`, `return`
- Unique tokens: `def`, `fun`, `attr`, `bind`, `Dynamic_Object`, `method_missing`, `:=` (reference assign)

**Ch** (secondary): C/C++ interpreter with computational arrays
- Full C/C++ syntax compatibility (Ch IS C with extensions)
- Unique tokens: `string_t`, `array double`, `foreach`, computational arrays

### ChaiScript Keywords and Built-ins
```
# ChaiScript-specific keywords
def, fun, attr, bind, var
Dynamic_Object, method_missing, set_explicit, call_exists
eval, eval_file, use

# ChaiScript built-in functions
back, bob_back, collate, concat, drop_while, drop, dump_system
empty, even, filter, foldl, for_each, front, generate_range
get_arity, get_contained_functions, is_type, join, product
puts, reduce, retro, reverse, size, sum, take_while, take
to_string, zip_with, zip

# Ch extensions
string_t, generic_t, foreach
array double, array int, array float, array complex

# ChaiScript operator
:=  (reference assignment)

# Script interaction
chai_eval, chai_define, ch_run, ch_eval
```

---

## Number Pattern Tokens (IDs 800-1199)

### Hex Patterns (800-849, 50 tokens)
```
0x, 0X (prefixes -- critical, appear in virtually every C++ file)
0x0-0xF (16 single hex digits)
0x00, 0xFF, 0x80, 0x7F, 0x0F, 0xF0 (common byte values)
0xDEAD, 0xBEEF, 0xCAFE, 0xBABE (magic numbers)
0x0000, 0xFFFF, 0x8000, 0x7FFF (16-bit boundaries)
0x00000000, 0xFFFFFFFF, 0xDEADBEEF, 0xCAFEBABE (32-bit)
0x100, 0x200, 0x400, 0x800, 0x1000, 0x2000, 0x4000 (powers of 2)
0x10, 0x20, 0x40, 0x1F, 0x3F, 0xFF00, 0x00FF (masks)
```

### Float Patterns (850-899, 50 tokens)
```
0.0, 1.0, 2.0, 0.5, 0.1, 0.01, 0.001
1.0f, 0.0f, 0.5f, 2.0f, -1.0f (float suffixes)
3.14, 3.14f, 2.71, 1.41, 1.73 (mathematical constants)
f, F, l, L (suffix tokens in numeric context)
f16, F16, f32, F32, f64, F64, f128, F128, bf16, BF16 (C++23 float suffixes)
e+, e-, E+, E- (scientific notation markers)
```

### Scientific Notation (900-929, 30 tokens)
```
1e0-1e9 (powers of 10)
1e-1 through 1e-9 (negative powers)
```

### Binary Literals (930-949, 20 tokens)
```
0b, 0B (prefixes)
0b0, 0b1, 0b00, 0b01, 0b10, 0b11
0b0000, 0b1111, 0b10000000, 0b11111111
```

### Integer Suffixes (950-969, 20 tokens)
```
u, U, l, L, ll, LL, ul, UL, ull, ULL
lu, LU, llu, LLU, z, Z, uz, UZ (C++23 size_t suffixes)
```

### Common Integer Literals (970-1199, 230 tokens)
```
0-199 (most common small integers)
256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536
1000, 10000, 100000, 1000000, 0x10000, 0x100000
INT_MAX, INT_MIN, UINT_MAX, SIZE_MAX
```

---

## GPU/Accelerator Domain Tokens (IDs 5300-5799)

### CUDA Tokens (5300-5599, ~300 high-frequency tokens)

Selected from ~1,900 unique CUDA tokens identified across 45 categories. Priority: tokens appearing >1000 times in our training corpus.

#### CUDA Qualifiers & Built-ins (~20)
```
__global__, __device__, __host__, __shared__, __constant__, __managed__
__launch_bounds__, __restrict__
threadIdx, blockIdx, blockDim, gridDim, warpSize
__syncthreads, __threadfence, __threadfence_block, __threadfence_system
dim3, <<<, >>>
```

#### CUDA Runtime API (~60 highest-frequency)
```
# Memory
cudaMalloc, cudaFree, cudaMemcpy, cudaMemcpyAsync, cudaMemset
cudaMallocManaged, cudaMallocHost, cudaFreeHost, cudaHostAlloc
cudaMallocPitch, cudaMalloc3D, cudaMemcpy2D, cudaMemcpy3D
cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost
cudaMemcpyDeviceToDevice, cudaMemcpyDefault
cudaFreeAsync, cudaMallocAsync

# Device
cudaGetDevice, cudaSetDevice, cudaGetDeviceCount, cudaGetDeviceProperties
cudaDeviceSynchronize, cudaDeviceReset, cudaDeviceGetAttribute

# Stream/Event
cudaStreamCreate, cudaStreamDestroy, cudaStreamSynchronize, cudaStreamWaitEvent
cudaEventCreate, cudaEventDestroy, cudaEventRecord, cudaEventSynchronize
cudaEventElapsedTime

# Launch
cudaLaunchKernel, cudaFuncGetAttributes, cudaFuncSetAttribute
cudaOccupancyMaxActiveBlocksPerMultiprocessor
cudaOccupancyMaxPotentialBlockSize

# Error
cudaGetLastError, cudaPeekAtLastError, cudaGetErrorString, cudaGetErrorName
cudaError_t, cudaSuccess

# Types
cudaStream_t, cudaEvent_t, cudaDeviceProp, cudaMemcpyKind
```

#### cuBLAS (~30 highest-frequency)
```
cublasCreate, cublasDestroy, cublasSetStream, cublasGetStream
cublasSgemm, cublasDgemm, cublasHgemm, cublasCgemm, cublasZgemm
cublasGemmEx, cublasGemmBatchedEx, cublasGemmStridedBatchedEx
cublasSgemv, cublasDgemv, cublasSaxpy, cublasDaxpy
cublasSscal, cublasDscal, cublasSnrm2, cublasDnrm2
cublasHandle_t, cublasStatus_t, cublasOperation_t
cublasLtCreate, cublasLtMatmul, cublasLtMatmulDescCreate
```

#### cuDNN (~25 highest-frequency)
```
cudnnCreate, cudnnDestroy, cudnnSetStream
cudnnCreateTensorDescriptor, cudnnSetTensor4dDescriptor
cudnnConvolutionForward, cudnnConvolutionBackwardData, cudnnConvolutionBackwardFilter
cudnnBatchNormalizationForwardTraining, cudnnBatchNormalizationBackward
cudnnSoftmaxForward, cudnnPoolingForward, cudnnActivationForward
cudnnHandle_t, cudnnTensorDescriptor_t, cudnnDataType_t
```

#### Thrust/CUB (~30)
```
# Thrust
thrust::device_vector, thrust::host_vector, thrust::device_ptr
thrust::sort, thrust::reduce, thrust::transform, thrust::copy, thrust::fill
thrust::for_each, thrust::count_if, thrust::find_if, thrust::unique
thrust::inclusive_scan, thrust::exclusive_scan, thrust::transform_reduce
thrust::raw_pointer_cast, thrust::make_zip_iterator

# CUB
cub::BlockReduce, cub::BlockScan, cub::WarpReduce
cub::DeviceReduce, cub::DeviceScan, cub::DeviceRadixSort, cub::DeviceSelect
```

#### CUTLASS (~15)
```
cutlass::gemm, cutlass::half_t, cutlass::bfloat16_t
cutlass::layout::ColumnMajor, cutlass::layout::RowMajor
cutlass::arch::Sm80, cutlass::arch::Sm90, cutlass::arch::Sm100
cutlass::gemm::device::Gemm, cutlass::gemm::device::GemmUniversal
cutlass::float_e4m3_t, cutlass::float_e5m2_t
```

#### NCCL (~20)
```
ncclGetUniqueId, ncclCommInitRank, ncclCommInitAll, ncclCommDestroy
ncclAllReduce, ncclBroadcast, ncclReduce, ncclAllGather, ncclReduceScatter
ncclSend, ncclRecv, ncclGroupStart, ncclGroupEnd
ncclComm_t, ncclUniqueId, ncclResult_t, ncclSuccess
ncclFloat16, ncclFloat32, ncclBfloat16, ncclSum
```

#### Atomics & Intrinsics (~20)
```
atomicAdd, atomicSub, atomicExch, atomicMin, atomicMax
atomicAnd, atomicOr, atomicXor, atomicCAS, atomicInc
__shfl_sync, __shfl_up_sync, __shfl_down_sync, __shfl_xor_sync
__ballot_sync, __all_sync, __any_sync
__half, __half2, __nv_bfloat16, __nv_bfloat162
```

#### CUDA Math (~20)
```
__float2half, __half2float, __float2bfloat16, __bfloat162float
__hadd, __hsub, __hmul, __hdiv, __hfma
rsqrtf, __expf, __logf, __powf, __sinf, __cosf
__saturatef, __fmaf_rn, __fdividef
```

#### Graph API (~15)
```
cudaGraphCreate, cudaGraphDestroy, cudaGraphLaunch
cudaGraphInstantiate, cudaGraphAddKernelNode
cudaGraphExecUpdate, cudaGraphNodeSetParams
cudaGraph_t, cudaGraphExec_t, cudaGraphNode_t
```

### ROCm/HIP Tokens (5600-5699, ~100 high-frequency)
```
# Core Runtime (mirrors CUDA)
hipMalloc, hipFree, hipMemcpy, hipMemcpyAsync, hipMemset
hipMallocManaged, hipMallocHost, hipFreeHost, hipHostAlloc
hipGetDevice, hipSetDevice, hipGetDeviceCount, hipDeviceSynchronize
hipStreamCreate, hipStreamDestroy, hipStreamSynchronize, hipStreamWaitEvent
hipEventCreate, hipEventDestroy, hipEventRecord, hipEventElapsedTime
hipLaunchKernelGGL, hipGetLastError, hipGetErrorString

# Types
hipError_t, hipSuccess, hipStream_t, hipEvent_t, hipDeviceProp_t
hipMemcpyHostToHost, hipMemcpyHostToDevice, hipMemcpyDeviceToHost
hipMemcpyDeviceToDevice, hipMemcpyDefault, hipMemcpyDeviceToDeviceNoCU

# hipBLAS
hipblasCreate, hipblasDestroy, hipblasSetStream
hipblasSgemm, hipblasDgemm, hipblasHgemm
hipblasGemmEx, hipblasGemmStridedBatchedEx
hipblasHandle_t, hipblasStatus_t, hipblasOperation_t

# Peer Access
hipDeviceCanAccessPeer, hipDeviceEnablePeerAccess, hipDeviceDisablePeerAccess
hipMemcpyPeer, hipMemcpyPeerAsync

# Occupancy
hipOccupancyMaxActiveBlocksPerMultiprocessor
hipOccupancyMaxPotentialBlockSize

# Qualifiers
__global__, __device__, __host__, __shared__  (shared with CUDA)
HIP_DYNAMIC_SHARED

# rocBLAS/MIOpen
rocblas_create_handle, rocblas_destroy_handle, rocblas_sgemm, rocblas_dgemm
miopenCreateTensorDescriptor, miopenConvolutionForward
```

### TPU/XLA Tokens (5700-5799, ~100 high-frequency)
```
# XLA HLO Operations (from MHLO dialect, ~57 ops)
mhlo.add, mhlo.subtract, mhlo.multiply, mhlo.divide
mhlo.dot, mhlo.dot_general, mhlo.convolution
mhlo.reduce, mhlo.reduce_window, mhlo.scatter, mhlo.gather
mhlo.broadcast_in_dim, mhlo.transpose, mhlo.reshape, mhlo.dynamic_slice
mhlo.concatenate, mhlo.slice, mhlo.pad, mhlo.select
mhlo.compare, mhlo.and, mhlo.or, mhlo.not
mhlo.convert, mhlo.bitcast_convert
mhlo.batch_norm_training, mhlo.batch_norm_inference
mhlo.all_reduce, mhlo.all_gather, mhlo.all_to_all
mhlo.collective_permute, mhlo.partition_id, mhlo.replica_id
mhlo.while, mhlo.conditional, mhlo.custom_call
mhlo.fft, mhlo.sort, mhlo.iota, mhlo.rng

# Pallas/Mosaic TPU Dialect
pallas.program_id, pallas.num_programs
pl.load, pl.store, pl.dot, pl.broadcast_to
BlockSpec, GridSpec, Pallas

# SparseCore API
sc.send, sc.recv, sc.collective_permute

# Trillium v6e Architecture
MXU, SparseCore, HBM, VMEM, CMEM, ICI
```

---

## SQL Domain Tokens (IDs 5800-6299)

SQL is frequently embedded in C++ strings via raw string literals, ORM libraries, and database client code. All major dialects are represented.

### ANSI SQL:2023 Core Keywords (~200 tokens)
```
# DML
SELECT, INSERT, UPDATE, DELETE, MERGE, UPSERT, REPLACE
FROM, WHERE, HAVING, GROUP BY, ORDER BY, LIMIT, OFFSET, FETCH
JOIN, INNER, LEFT, RIGHT, FULL, OUTER, CROSS, NATURAL
ON, USING, AS, DISTINCT, ALL, ANY, SOME, EXISTS, IN, BETWEEN
LIKE, ILIKE, SIMILAR, ESCAPE, IS, NULL, NOT, AND, OR

# DDL
CREATE, ALTER, DROP, TRUNCATE, RENAME, COMMENT
TABLE, VIEW, INDEX, SEQUENCE, SCHEMA, DATABASE, TABLESPACE
COLUMN, CONSTRAINT, PRIMARY, FOREIGN, KEY, UNIQUE, CHECK, DEFAULT
REFERENCES, CASCADE, RESTRICT, SET NULL, SET DEFAULT, NO ACTION

# Types
INTEGER, SMALLINT, BIGINT, DECIMAL, NUMERIC, REAL, FLOAT, DOUBLE
CHAR, VARCHAR, TEXT, CLOB, BLOB, BINARY, VARBINARY
DATE, TIME, TIMESTAMP, INTERVAL, BOOLEAN, UUID, JSON, JSONB, XML
ARRAY, ROW, MULTISET

# Aggregates
COUNT, SUM, AVG, MIN, MAX, ARRAY_AGG, STRING_AGG, LISTAGG
GROUP_CONCAT, GROUPING, CUBE, ROLLUP, GROUPING SETS

# Window Functions
OVER, PARTITION BY, ROWS, RANGE, GROUPS
UNBOUNDED, PRECEDING, FOLLOWING, CURRENT ROW
ROW_NUMBER, RANK, DENSE_RANK, NTILE, PERCENT_RANK, CUME_DIST
LAG, LEAD, FIRST_VALUE, LAST_VALUE, NTH_VALUE

# CTE / Subqueries
WITH, RECURSIVE, LATERAL, MATERIALIZED

# Control
CASE, WHEN, THEN, ELSE, END, COALESCE, NULLIF, GREATEST, LEAST
CAST, CONVERT, TRY_CAST

# Transaction
BEGIN, COMMIT, ROLLBACK, SAVEPOINT, RELEASE
TRANSACTION, ISOLATION, READ, WRITE, SERIALIZABLE
REPEATABLE READ, READ COMMITTED, READ UNCOMMITTED

# Misc
UNION, INTERSECT, EXCEPT, ALL
EXPLAIN, ANALYZE, VACUUM, REINDEX
GRANT, REVOKE, DENY, EXECUTE, TRIGGER, PROCEDURE, FUNCTION
RETURNS, LANGUAGE, VOLATILE, STABLE, IMMUTABLE, DETERMINISTIC
```

### Dialect-Specific Keywords (~100 tokens)

#### PostgreSQL Extensions
```
RETURNING, ILIKE, SIMILAR TO, DISTINCT ON
SERIAL, BIGSERIAL, SMALLSERIAL, BYTEA, CIDR, INET, MACADDR
NOTIFY, LISTEN, UNLISTEN, COPY, VACUUM, ANALYZE
DO, PERFORM, RAISE, EXCEPTION, NOTICE, DEBUG
pg_catalog, information_schema, pg_stat_activity
CONCURRENTLY, IF EXISTS, IF NOT EXISTS
GENERATED, ALWAYS, IDENTITY, OVERRIDING
```

#### MySQL Extensions
```
AUTO_INCREMENT, ENGINE, CHARSET, COLLATE, UNSIGNED
SHOW, DESCRIBE, LOAD DATA, INFILE, OUTFILE
DELIMITER, SIGNAL, RESIGNAL, HANDLER
ON DUPLICATE KEY UPDATE, IGNORE, REPLACE INTO
FULLTEXT, SPATIAL, PARTITION BY RANGE, PARTITION BY HASH
InnoDB, MyISAM, MEMORY
```

#### SQL Server (T-SQL) Extensions
```
TOP, NOLOCK, WITH (NOLOCK), IDENTITY, NEWID, NEWSEQUENTIALID
NVARCHAR, NCHAR, NTEXT, UNIQUEIDENTIFIER, HIERARCHYID
CROSS APPLY, OUTER APPLY, PIVOT, UNPIVOT
MERGE, MATCHED, NOT MATCHED, OUTPUT, INSERTED, DELETED
TRY, CATCH, THROW, RAISERROR, @@ERROR, @@ROWCOUNT
DECLARE, SET, PRINT, EXEC, sp_executesql
```

#### ClickHouse Extensions
```
ENGINE = MergeTree, ReplacingMergeTree, SummingMergeTree
AggregatingMergeTree, CollapsingMergeTree, VersionedCollapsingMergeTree
ORDER BY, PARTITION BY, SAMPLE BY, TTL
Tuple, Nested, LowCardinality, Nullable, Map
FINAL, PREWHERE, GLOBAL IN, GLOBAL JOIN
ATTACH, DETACH, OPTIMIZE, SYSTEM FLUSH LOGS
toDateTime, toDate, toUInt32, toString, arrayJoin
```

### SQL JSON Functions (~30 tokens)
```
JSON_EXTRACT, JSON_VALUE, JSON_QUERY, JSON_TABLE
JSON_OBJECT, JSON_ARRAY, JSON_ARRAYAGG, JSON_OBJECTAGG
JSON_EXISTS, JSON_SET, JSON_INSERT, JSON_REPLACE, JSON_REMOVE
JSON_CONTAINS, JSON_CONTAINS_PATH, JSON_LENGTH, JSON_KEYS
JSON_SEARCH, JSON_TYPE, JSON_VALID, JSON_PRETTY
IS JSON, JSON_SERIALIZE, JSON_SCALAR
OPENJSON, FOR JSON, JSON_MODIFY
->>, ->, #>>, #>
```

### SQL C API Functions (~70 tokens)

#### SQLite3 C API (highest-frequency ~30)
```
sqlite3_open, sqlite3_open_v2, sqlite3_close, sqlite3_close_v2
sqlite3_exec, sqlite3_prepare_v2, sqlite3_prepare_v3
sqlite3_step, sqlite3_finalize, sqlite3_reset
sqlite3_bind_int, sqlite3_bind_int64, sqlite3_bind_double
sqlite3_bind_text, sqlite3_bind_blob, sqlite3_bind_null
sqlite3_column_int, sqlite3_column_int64, sqlite3_column_double
sqlite3_column_text, sqlite3_column_blob, sqlite3_column_count
sqlite3_column_type, sqlite3_column_name
sqlite3_errmsg, sqlite3_errcode, sqlite3_last_insert_rowid
sqlite3_changes, sqlite3_free, sqlite3_malloc
```

#### MySQL C API (~20)
```
mysql_init, mysql_real_connect, mysql_close
mysql_query, mysql_real_query, mysql_store_result, mysql_use_result
mysql_fetch_row, mysql_fetch_field, mysql_num_rows, mysql_num_fields
mysql_free_result, mysql_affected_rows, mysql_errno, mysql_error
mysql_autocommit, mysql_commit, mysql_rollback
mysql_escape_string, mysql_real_escape_string
```

#### ODBC (~20)
```
SQLAllocHandle, SQLFreeHandle, SQLConnect, SQLDisconnect
SQLDriverConnect, SQLBrowseConnect
SQLPrepare, SQLExecute, SQLExecDirect
SQLFetch, SQLFetchScroll, SQLGetData
SQLBindCol, SQLBindParameter, SQLNumResultCols, SQLRowCount
SQLEndTran, SQLCloseCursor, SQLCancel
SQLGetDiagRec, SQLGetDiagField
```

---

## Query Language & ORM Tokens (IDs 6300-6599)

Comprehensive catalog derived from research across ODMG OQL, db4o, ObjectStore, Realm, LINQ-like C++ libraries, Protocol Buffers, gRPC, GraphQL, MongoDB, Redis, SOCI, ODB, and sqlpp11. Full research output: ~1,600+ unique tokens; 300 highest-frequency tokens selected for fixed vocab.

### Protocol Buffers + gRPC (~60 tokens)
```
# Proto file keywords (embedded in C++ as string literals & .proto files)
syntax, package, import, option, message, enum, service, rpc, returns
repeated, optional, required, oneof, map, reserved, extensions, extend, stream

# Core protobuf classes
google::protobuf::Message, google::protobuf::MessageLite
google::protobuf::Arena, google::protobuf::Descriptor
google::protobuf::FieldDescriptor, google::protobuf::Reflection
google::protobuf::RepeatedField, google::protobuf::RepeatedPtrField

# Protobuf I/O
google::protobuf::io::CodedInputStream, google::protobuf::io::CodedOutputStream

# Common Message methods
SerializeToString, ParseFromString, SerializeToArray, ParseFromArray
ByteSizeLong, IsInitialized, CopyFrom, MergeFrom, Clear
GetDescriptor, GetReflection, New, GetArena, SpaceUsedLong, DebugString

# gRPC core classes
grpc::Server, grpc::ServerBuilder, grpc::ServerContext
grpc::Channel, grpc::ClientContext, grpc::CompletionQueue
grpc::Status, grpc::Service, grpc::CallCredentials, grpc::ChannelCredentials

# gRPC status codes
OK, CANCELLED, UNKNOWN, INVALID_ARGUMENT, DEADLINE_EXCEEDED, NOT_FOUND
ALREADY_EXISTS, PERMISSION_DENIED, UNAUTHENTICATED, RESOURCE_EXHAUSTED
FAILED_PRECONDITION, ABORTED, OUT_OF_RANGE, UNIMPLEMENTED, INTERNAL
UNAVAILABLE, DATA_LOSS

# gRPC credential functions
CreateChannel, InsecureChannelCredentials, SslCredentials
```

### C++ LINQ-like Libraries (~35 tokens)
```
# cpplinq (operator>> chain)
from, from_array, from_range, from_copy
where, select, select_many, orderby, orderby_descending
thenby, thenby_descending, take, skip, take_while, skip_while
distinct, concat, join, zip_with, reverse, pairwise
count, sum, avg, min, max, aggregate
first, first_or_default, last, last_or_default
any, all, for_each, element_at
to_vector, to_list, to_map

# boolinq additions (beyond shared with cpplinq)
groupBy, selectMany, toStdSet, toStdDeque
leftJoin, rightJoin, crossJoin
bytes, unbytes, bits, unbits
```

### GraphQL C++ (~25 tokens)
```
# GraphQL language keywords
query, mutation, subscription, fragment, on
type, input, enum, interface, union, scalar, schema, directive
extend, implements, repeatable
__typename, __schema, __type

# Built-in scalars & directives
Int, Float, String, Boolean, ID
@skip, @include, @deprecated, @specifiedBy

# libgraphqlparser key AST types (42 classes total)
Document, OperationDefinition, VariableDefinition, SelectionSet
Field, Argument, FragmentSpread, InlineFragment, FragmentDefinition
NamedType, ListType, NonNullType, SchemaDefinition, DirectiveDefinition
```

### MongoDB C++ Driver (~40 tokens)
```
# mongocxx core classes
mongocxx::client, mongocxx::database, mongocxx::collection
mongocxx::cursor, mongocxx::pipeline, mongocxx::pool, mongocxx::instance

# mongocxx::collection methods
find, find_one, insert_one, insert_many, update_one, update_many
delete_one, delete_many, aggregate, count_documents, distinct
find_one_and_update, find_one_and_delete, find_one_and_replace

# bsoncxx builder
bsoncxx::builder::basic::document, bsoncxx::builder::basic::array
bsoncxx::document::view, bsoncxx::document::value

# Aggregation stages (top 15)
$match, $group, $project, $sort, $limit, $skip, $unwind
$lookup, $graphLookup, $merge, $out, $addFields, $facet, $set, $unset

# Query operators
$eq, $ne, $gt, $gte, $lt, $lte, $in, $nin
$and, $or, $not, $nor, $exists, $type, $regex, $elemMatch

# Update operators
$set, $unset, $inc, $mul, $push, $pull, $addToSet, $pop
$rename, $min, $max, $currentDate, $setOnInsert
```

### Redis C++ Client (~40 tokens)
```
# String commands
GET, SET, MGET, MSET, INCR, DECR, APPEND, STRLEN
SETEX, SETNX, PSETEX, GETSET, GETDEL, GETEX, INCRBY, INCRBYFLOAT

# Hash commands
HGET, HSET, HMGET, HMSET, HDEL, HGETALL, HKEYS, HVALS, HEXISTS, HINCRBY

# List commands
LPUSH, RPUSH, LPOP, RPOP, LRANGE, LLEN, LINDEX, LINSERT, BLPOP, BRPOP

# Set commands
SADD, SREM, SMEMBERS, SCARD, SISMEMBER, SINTER, SUNION, SDIFF, SPOP

# Sorted set commands
ZADD, ZREM, ZRANGE, ZRANK, ZSCORE, ZCARD, ZINCRBY, ZRANGEBYSCORE, ZCOUNT

# Pub/Sub & Streams
SUBSCRIBE, UNSUBSCRIBE, PUBLISH, PSUBSCRIBE
XADD, XREAD, XREADGROUP, XACK, XLEN, XRANGE, XGROUP

# Transaction & Key
MULTI, EXEC, DISCARD, WATCH, UNWATCH
DEL, EXISTS, EXPIRE, TTL, KEYS, SCAN, TYPE, RENAME

# RedisJSON & RediSearch (modules)
JSON.SET, JSON.GET, JSON.DEL, JSON.MGET
FT.CREATE, FT.SEARCH, FT.AGGREGATE, FT.INFO

# Libraries
hiredis, redis-plus-plus
```

### C++ Database ORMs & SQL Libraries (~40 tokens)
```
# sqlpp11 (type-safe SQL for C++)
select, from, where, group_by, having, order_by, limit, offset
join, inner_join, left_outer_join, cross_join, on, using_
insert_into, update, remove_from, set, columns, values
in, not_in, is_null, is_not_null, like, between, exists
dynamic_select, dynamic_where, dynamic_join, unconditionally

# SOCI (SQL for C++)
session, statement, transaction, rowset, row
use, into, indicator, procedure, once, prepare, execute, fetch, got_data

# ODB (C++ ORM)
database, transaction, query, result, view, session, schema_catalog
persist, load, update, erase, reload, find, begin, commit, rollback
```

### Common Query Pattern Identifiers (~20 tokens)
```
# Cross-library high-frequency query patterns
query, filter, predicate, criteria, expression, condition
cursor, iterator, result, resultset, batch, bulk
connection, pool, schema, index, constraint, binding
ascending, descending, pagination, limit, offset
```

### OQL Databases (ODMG/Realm) (~20 tokens)
```
# ODMG OQL operators
select, from, where, order by, group by, having, distinct
flatten, element, exists, for all, like, in, between
is_defined, is_undefined, count, sum, avg, min, max, abs
struct, list, set, bag, array, first, last

# Realm C++ SDK
realm::db, realm::results, realm::object, realm::query
sort_descriptor, ascending, descending
contains, begins_with, ends_with, like
```

### Expanded ODBC API (~20 tokens)
```
# Additional ODBC functions beyond SQL section
SQLAllocEnv, SQLAllocConnect, SQLAllocStmt
SQLDriverConnect, SQLBrowseConnect
SQLFetchScroll, SQLSetConnectAttr, SQLGetConnectAttr
SQLSetStmtAttr, SQLGetStmtAttr, SQLSetEnvAttr, SQLGetEnvAttr
SQLColumns, SQLTables, SQLStatistics, SQLPrimaryKeys, SQLForeignKeys
SQLGetTypeInfo, SQLBulkOperations, SQLSetPos
SQLDescribeParam, SQLNumParams, SQLCompleteAsync
```

---

## C++23/26 Additions (IDs 6600-6799)

### C++23 New Library Features (~60 tokens)
```
# Utility
std::expected, std::unexpected
std::move_only_function, std::bind_back
std::unreachable, std::to_underlying, std::byteswap
std::invoke_r, std::forward_like
std::out_ptr, std::inout_ptr
std::start_lifetime_as
std::print, std::println

# Ranges
std::ranges::to, std::ranges::zip, std::ranges::zip_transform
std::ranges::adjacent, std::ranges::chunk, std::ranges::slide
std::ranges::stride, std::ranges::cartesian_product
std::ranges::repeat, std::ranges::as_rvalue
std::views::enumerate, std::views::as_const

# Type Traits (C++23)
std::is_scoped_enum, std::is_implicit_lifetime
std::reference_constructs_from_temporary
std::reference_converts_from_temporary

# Containers
std::flat_map, std::flat_set, std::flat_multimap, std::flat_multiset
std::mdspan, std::generator
std::basic_string::contains, std::basic_string::starts_with, std::basic_string::ends_with

# Misc
std::stacktrace, std::source_location
std::unreachable_sentinel
static operator(), static operator[]
if consteval, auto(x), auto{x}
```

### C++26 New Language Features (~60 tokens)
```
# Contracts
contract_assert, pre, post
[[assert: expr]], [[pre: expr]], [[post r: expr]]

# Reflection (P2996)
^^, std::meta::info
std::meta::name_of, std::meta::type_of, std::meta::members_of
std::meta::is_public, std::meta::is_static, std::meta::is_virtual
define_class, substitute, reflect_value

# Annotations for Reflection (P3394)
[[=annotation]]

# std::execution (Senders/Receivers)
std::execution::scheduler, std::execution::sender, std::execution::receiver
std::execution::run_loop, std::execution::static_thread_pool
std::execution::just, std::execution::then, std::execution::let_value
std::execution::when_all, std::execution::transfer, std::execution::schedule
std::execution::start_detached, std::execution::sync_wait
std::execution::counting_scope

# Atomics
std::atomic::fetch_max, std::atomic::fetch_min

# constexpr extensions
constexpr std::shared_ptr, constexpr std::unique_ptr
std::is_within_lifetime

# Standard Library Hardening (P3471)
[[indeterminate]]

# Other
std::text_encoding
pattern matching (inspect/is/as -- in progress)
```

### C++20/23 Concepts (~30 tokens)
```
same_as, derived_from, convertible_to, integral, floating_point
signed_integral, unsigned_integral, destructible, constructible_from
common_reference_with, assignable_from, swappable
movable, copyable, semiregular, regular
equality_comparable, totally_ordered, three_way_comparable
invocable, predicate, relation, strict_weak_order
input_iterator, forward_iterator, bidirectional_iterator
random_access_iterator, contiguous_iterator
input_range, forward_range, bidirectional_range
random_access_range, contiguous_range, sized_range, view
```

---

## Testing/Build Framework Tokens (IDs 6800-6999)

### Google Test (~40 tokens)
```
TEST, TEST_F, TEST_P, TYPED_TEST, TYPED_TEST_SUITE
EXPECT_EQ, EXPECT_NE, EXPECT_LT, EXPECT_GT, EXPECT_LE, EXPECT_GE
EXPECT_TRUE, EXPECT_FALSE, EXPECT_THAT
ASSERT_EQ, ASSERT_NE, ASSERT_LT, ASSERT_GT, ASSERT_LE, ASSERT_GE
ASSERT_TRUE, ASSERT_FALSE, ASSERT_THAT
EXPECT_THROW, EXPECT_NO_THROW, EXPECT_DEATH
ASSERT_THROW, ASSERT_NO_THROW, ASSERT_DEATH
MOCK_METHOD, EXPECT_CALL, ON_CALL, INVOKE
testing::Return, testing::Eq, testing::_, testing::Matcher
SetUp, TearDown, SetUpTestSuite, TearDownTestSuite
```

### CMake (~40 tokens)
```
cmake_minimum_required, project, add_executable, add_library
target_link_libraries, target_include_directories
target_compile_definitions, target_compile_options
find_package, find_library, find_path, find_program
include_directories, link_directories, add_subdirectory
set, option, if, else, elseif, endif, foreach, endforeach
message, install, configure_file
CMAKE_CXX_STANDARD, CMAKE_CXX_FLAGS, CMAKE_BUILD_TYPE
CMAKE_INSTALL_PREFIX, CMAKE_SOURCE_DIR, CMAKE_BINARY_DIR
CMAKE_CURRENT_SOURCE_DIR, CMAKE_PREFIX_PATH, CMAKE_TOOLCHAIN_FILE
CMakeLists.txt, CMAKE_
```

### Boost.Test & Other (~20 tokens)
```
BOOST_AUTO_TEST_CASE, BOOST_CHECK, BOOST_REQUIRE, BOOST_TEST
BOOST_FIXTURE_TEST_SUITE, BOOST_AUTO_TEST_SUITE
BOOST_DATA_TEST_CASE, BOOST_CHECK_EQUAL, BOOST_REQUIRE_EQUAL
Catch::, REQUIRE, CHECK, SECTION, TEST_CASE, SCENARIO
GIVEN, WHEN, THEN, AND_GIVEN, AND_WHEN, AND_THEN
```

### Compiler Attributes (~20 tokens)
```
__attribute__, __declspec, __stdcall, __cdecl, __fastcall
__forceinline, __restrict, __asm__
[[nodiscard]], [[maybe_unused]], [[deprecated]], [[likely]], [[unlikely]]
[[no_unique_address]], [[carries_dependency]], [[fallthrough]]
__cplusplus, __FILE__, __LINE__, __func__, __FUNCTION__
```

---

## Morpheme-Aware BPE Design

### Research Foundation (2025-2026)

#### MorphBPE (arXiv:2502.00894)
**Core algorithm**: Modify BPE training so merges never cross morpheme boundaries. During inference, the tokenizer functions identically to standard BPE -- the constraint only applies during training.

Results on 300M/1B parameter LLMs:
- Hungarian: F1 0.13 -> 0.87 (+0.74)
- Arabic: F1 ~0.00 -> 0.66
- English: F1 ~0.00 -> 0.24
- Consistently reduces cross-entropy loss and accelerates convergence

#### LiteToken (arXiv:2602.04706, Feb 2026)
Identifies and removes "intermediate merge residues" -- tokens frequent during BPE training but rarely emitted during tokenization. These waste vocabulary capacity. LiteToken removes them, reducing fragmentation and improving robustness without fine-tuning.

**Key technique for us**: After training BPE, scan for merge residues and remove them, freeing ~3-5% of vocabulary for more useful tokens.

#### SuperBPE (COLM 2025, arXiv:2503.13423)
Two-pass BPE: first learns standard subwords, then lifts pre-tokenization constraint to learn cross-word "superword" tokens. At 200K vocab: **33% fewer tokens**, +4.0% average across 30 benchmarks.

#### BoundlessBPE (COLM 2025, arXiv:2504.00178)
Single-pass approach allowing "supermerges" across pretoken boundaries. **21% higher distribution uniformity**, **20% more bytes per token**. Relevant: explicitly handles snake_case and camelCase patterns.

#### TokDrift (arXiv:2510.14972, Oct 2025)
**Critical finding**: 8.29% of code samples produce different LLM outputs due to tokenization changes from whitespace. Morpheme-aware tokenization mitigates this by ensuring consistent identifier segmentation.

#### Additional Research
- **AG-BPE**: Attention-guided BPE using semantic-aware merge decisions
- **StochasTok** (arXiv:2506.01687, June 2025): Tokenizer-agnostic subword regularization, maintains original vocabulary
- **Broken Tokens** (arXiv:2506.19004): Instruction-tuned LLMs retain 93.4% performance with random tokenizations
- **GlitchMiner** (AAAI 2026): ~4.3% of vocabulary entries are "glitch tokens" causing erratic behavior
- **"Say Anything but This"** (arXiv:2601.14658, Jan 2026): Non-unique BPE encodings cause "phantom edits" in reasoning
- **BLT (Byte Latent Transformer)**: Tokenization-free model using entropy-based byte patching (ACL 2025)
- **EvaByte**: 6.5B byte-level model rivaling token-based LMs with 5x less data

### C++ Identifier Morpheme Rules

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

### Common C++ Morphemes as Fixed Tokens (IDs 4700-4899)

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
impl, ctx, ptr, buf, cfg, msg, req, res, cb, fn
arg, param, iter, prev, curr, next, tmp, src, dst, len
cnt, idx, pos, sz, cap, ctor, dtor, vtbl
```

### BPE Training Modifications

```python
def preprocess_for_morphbpe(text: str) -> str:
    """Insert boundary markers at morpheme splits."""
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

This ensures BPE learns merges **within** morphemes (`buff` + `er` -> `buffer`) but not **across** boundaries (`get_bu` + `ffer` stays as `get` + `buffer`).

### Post-Training: LiteToken Residue Removal

After BPE training, apply LiteToken technique:
1. Tokenize a large held-out corpus
2. Count actual emission frequency of each vocabulary entry
3. Remove tokens that are never/rarely emitted (intermediate merge residues)
4. Redistribute freed slots to highest-frequency multi-byte-pair tokens from the held-out corpus

Expected savings: ~3-5% of vocabulary (1,750-2,900 tokens freed for better use).

---

## Phoneme-Aware Handling for Noisy Comments

### The Problem

C++ codebases contain comments written by non-native English speakers with:
- Misspellings: "retrun", "lenght", "recieve"
- Phonetic approximations: "teh" for "the", "wiht" for "with"

### Solution

1. **Common misspelling normalization during training**: Pre-process training data to normalize top ~200 misspellings
2. **Phoneme-inspired subword units** (covered in morpheme tokens above)
3. **BPE-Dropout (p=0.1) during training**: Stochastic tokenization exposes model to multiple segmentations, naturally handling spelling variants
4. **StochasTok as alternative**: Tokenizer-agnostic, can be retrofitted onto existing models

---

## Vocab Size Comparison

| | v1 (32K) | v2 (48K) | v3 (64K) |
|---|---|---|---|
| **Fixed tokens** | 1,600 | 7,200 | 7,200 |
| **Learned BPE** | 31,168 | 41,952 | 58,336 |
| **Total** | 32,768 | 49,152 | 65,536 |
| **Thinking tokens** | 0 | 11 | 11 |
| **Script tokens** | 0 | 200 | 200 |
| **Number patterns** | 1000 ints | 400 patterns | 400 patterns |
| **Morpheme tokens** | 0 | 200 | 200 |
| **GPU/Accelerator** | 0 | 500 | 500 |
| **SQL domain** | 0 | 500 | 500 |
| **Query/DB tokens** | 0 | 300 | 300 |
| **C++23/26** | 0 | 200 | 200 |
| **Test/Build** | 0 | 200 | 200 |
| **Morpheme-aware BPE** | No | Yes | Yes |
| **LiteToken residue removal** | No | Yes | Yes |
| **Est. bytes/token** | ~5.3 | ~6.5 | ~7.2 |
| **Est. token reduction** | baseline | ~18% fewer | ~26% fewer |

---

## Fixed Token Priority Ranking

If budget is constrained, add in this order:

1. **Fixed-width integer types** (int32_t, uint8_t, etc.) -- 8 tokens, massive impact
2. **Hex prefix `0x`** -- 2 tokens, improves all hex literal tokenization
3. **CUDA core API** (cudaMalloc, cudaMemcpy, etc.) -- 60 tokens
4. **SQL core keywords** (SELECT, INSERT, JOIN, etc.) -- 50 tokens
5. **Google Test macros** (TEST, EXPECT_EQ, ASSERT_TRUE) -- 15 tokens
6. **C++23/26 concepts and type traits** -- 30 tokens
7. **Common hex constants** (0xFF, 0x00, etc.) -- 32 tokens
8. **Compiler attributes** (__attribute__, [[nodiscard]], etc.) -- 12 tokens
9. **SQLite3 C API** (sqlite3_open, sqlite3_exec, etc.) -- 30 tokens
10. Everything else

---

## Implementation Plan

### Phase 1: Fixed Vocab Design (1 day)
1. Create `data/cpp_tokenizer_v2/fixed_vocab.json` with 7,200 tokens
2. Add all domain tokens (GPU, SQL, query languages, C++23/26)
3. Add thinking tokens, ChaiScript/Ch tokens
4. Add number pattern tokens and morpheme tokens

### Phase 2: Morpheme-Aware BPE Training (2 days)
1. Implement `segment_cpp_identifier()` morpheme splitter
2. Modify `tok_train_cpp.py` for morpheme-guided pre-processing
3. Add boundary markers to prevent cross-morpheme merges
4. Train on full C++ corpus with morpheme constraints
5. Apply LiteToken residue removal post-training

### Phase 3: Integration & Testing (1 day)
1. Update `CppTokenizer` to handle new special tokens
2. Add thinking token support to `Engine.generate()`
3. Benchmark compression ratio vs v1
4. Verify all domain tokens round-trip correctly

### Phase 4: Training (next experiment)
1. Start new training run with v3 (64K) tokenizer
2. A/B test v3 vs v1 on identical data

---

## Migration Strategy

- **Current runs**: Keep v1 (32K). Cannot change mid-training.
- **Next experiment**: Use v3 (64K) as primary target.
- **Checkpoint conversion**: Not possible (embedding table size changes). Must train from scratch.
- The `get_token_bytes()` function already computes dynamically from tokenizer, so no version management needed.

---

## References

### Core Tokenization Papers
- [MorphBPE](https://arxiv.org/abs/2502.00894) -- Morpho-aware tokenizer (Feb 2025)
- [LiteToken](https://arxiv.org/abs/2602.04706) -- Merge residue removal (Feb 2026)
- [SuperBPE](https://arxiv.org/abs/2503.13423) -- Cross-word superword tokens (COLM 2025)
- [BoundlessBPE](https://arxiv.org/abs/2504.00178) -- Pre-tokenization boundary removal (COLM 2025)
- [TokDrift](https://arxiv.org/abs/2510.14972) -- Tokenization sensitivity in code LLMs (Oct 2025)
- [AG-BPE] -- Attention-guided BPE merge decisions
- [StochasTok](https://arxiv.org/abs/2506.01687) -- Tokenizer-agnostic subword regularization (June 2025)
- [Broken Tokens](https://arxiv.org/abs/2506.19004) -- Robustness to random tokenizations (June 2025)
- [GlitchMiner](https://arxiv.org/abs/2601.XXXX) -- Glitch token detection (AAAI 2026)
- ["Say Anything but This"](https://arxiv.org/abs/2601.14658) -- Non-unique BPE phantom edits (Jan 2026)
- [BLT (Byte Latent Transformer)](https://arxiv.org/abs/2412.09871) -- Tokenization-free (ACL 2025)
- [Vocabulary Scaling Laws](https://arxiv.org/abs/2407.XXXX) -- Optimal vocab size (NeurIPS 2024)

### Morphological Tokenization
- [MorphPiece](https://arxiv.org/abs/2307.07262) -- Morphological lookup table + BPE fallback
- [MorphTok](https://arxiv.org/abs/2504.10335) -- Morphologically grounded for Indic (ICML 2025)
- [OBPE](https://arxiv.org/abs/2602.04241) -- Overlap-based BPE for cross-lingual (2026)
- [BPE-knockout](https://arxiv.org/abs/2306.07141) -- Constraining BPE at morpheme boundaries
- [SubwordRegularization](https://arxiv.org/abs/1804.10959) -- Multiple segmentation for robust tokenization
- [BPE-Dropout](https://arxiv.org/abs/1910.13267) -- Stochastic BPE merges

### C++ Ecosystem
- [ChaiScript](https://chaiscript.com/) -- Header-only C++ embedded scripting
- [Ch Language](https://www.softintegration.com/) -- C/C++ interpreter
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [ROCm/HIP Documentation](https://rocm.docs.amd.com/projects/HIP/)
- [XLA/MHLO Dialect](https://github.com/openxla/stablehlo)

### Query Languages & Database APIs
- [Protocol Buffers C++ API](https://protobuf.dev/reference/cpp/api-docs/) -- Google protobuf C++ reference
- [gRPC C++ Reference](https://grpc.github.io/grpc/cpp/namespacegrpc.html) -- gRPC C++ namespace (105 classes)
- [libgraphqlparser](https://github.com/graphql/libgraphqlparser) -- Facebook's C++ GraphQL parser (42 AST types)
- [cppgraphqlgen](https://github.com/microsoft/cppgraphqlgen) -- Microsoft's C++ GraphQL service generator
- [MongoDB C++ Driver](https://www.mongodb.com/docs/languages/cpp/cpp-driver/current/) -- mongocxx/bsoncxx classes
- [Redis Commands Reference](https://redis.io/docs/latest/commands/) -- Complete Redis command set
- [redis-plus-plus](https://github.com/sewenew/redis-plus-plus) -- Modern C++ Redis client
- [hiredis](https://github.com/redis/hiredis) -- Minimalistic C Redis client
- [sqlpp11](https://github.com/rbock/sqlpp11) -- Type-safe SQL for C++
- [SOCI](https://soci.sourceforge.net/) -- SQL library for C++
- [ODB](https://www.codesynthesis.com/products/odb/) -- C++ ORM
- [cpplinq](https://github.com/mrange/cpplinq) -- LINQ query operators for C++ sequences
- [boolinq](https://github.com/k06a/boolinq) -- C++ header-only LINQ library
- [Realm C++ SDK](https://github.com/realm/realm-cpp) -- MongoDB Realm object database
- [SQLite C/C++ Interface](https://sqlite.org/cintro.html) -- SQLite3 C API (225+ functions)
- [ODBC API Reference](https://learn.microsoft.com/en-us/sql/odbc/reference/syntax/odbc-api-reference) -- Microsoft ODBC specification
