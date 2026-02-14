#!/usr/bin/env python3
"""Analyze proposed tokenizer v3 vocabulary frequency in C++ training corpus.

Validates proposed fixed vocabulary tokens from TOKENIZER_V2_PROPOSAL.md against
actual usage in the cpp_compilable training dataset. Reports per-category coverage
and per-token frequency to guide fixed vocab allocation.

Usage:
    # Analyze single shard (fast, ~2 min)
    python -m scripts.data.analyze_vocab_frequency /tmp/shard_00000.parquet

    # Analyze multiple shards
    python -m scripts.data.analyze_vocab_frequency /tmp/shard_0000*.parquet

    # Filter by category
    python -m scripts.data.analyze_vocab_frequency /tmp/shard_00000.parquet -c cuda

    # Show morpheme analysis
    python -m scripts.data.analyze_vocab_frequency /tmp/shard_00000.parquet --morphemes

    # JSON output for downstream processing
    python -m scripts.data.analyze_vocab_frequency /tmp/shard_00000.parquet --json -o results.json
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import pyarrow.parquet as pq

# ─── Proposed Vocabulary by Category ─────────────────────────────────────────
# Each category: (match_mode, [tokens])
# match_modes: "ident" = word boundary, "dollar" = $-prefixed, "sql_string" = in string literals

PROPOSED_VOCAB = {
    # ── CUDA ──
    "cuda_qualifiers": (
        "ident",
        [
            "__global__",
            "__device__",
            "__host__",
            "__shared__",
            "__constant__",
            "__managed__",
            "__launch_bounds__",
            "__restrict__",
            "threadIdx",
            "blockIdx",
            "blockDim",
            "gridDim",
            "warpSize",
            "__syncthreads",
            "__threadfence",
            "__threadfence_block",
            "__threadfence_system",
            "dim3",
        ],
    ),
    "cuda_runtime": (
        "ident",
        [
            "cudaMalloc",
            "cudaFree",
            "cudaMemcpy",
            "cudaMemcpyAsync",
            "cudaMemset",
            "cudaMallocManaged",
            "cudaMallocHost",
            "cudaFreeHost",
            "cudaHostAlloc",
            "cudaMallocPitch",
            "cudaMalloc3D",
            "cudaMemcpy2D",
            "cudaMemcpy3D",
            "cudaMemcpyHostToHost",
            "cudaMemcpyHostToDevice",
            "cudaMemcpyDeviceToHost",
            "cudaMemcpyDeviceToDevice",
            "cudaMemcpyDefault",
            "cudaGetDevice",
            "cudaSetDevice",
            "cudaGetDeviceCount",
            "cudaGetDeviceProperties",
            "cudaDeviceSynchronize",
            "cudaDeviceReset",
            "cudaDeviceGetAttribute",
            "cudaStreamCreate",
            "cudaStreamDestroy",
            "cudaStreamSynchronize",
            "cudaStreamWaitEvent",
            "cudaEventCreate",
            "cudaEventDestroy",
            "cudaEventRecord",
            "cudaEventSynchronize",
            "cudaEventElapsedTime",
            "cudaLaunchKernel",
            "cudaFuncGetAttributes",
            "cudaFuncSetAttribute",
            "cudaGetLastError",
            "cudaPeekAtLastError",
            "cudaGetErrorString",
            "cudaGetErrorName",
            "cudaError_t",
            "cudaSuccess",
            "cudaStream_t",
            "cudaEvent_t",
            "cudaDeviceProp",
            "cudaMemcpyKind",
        ],
    ),
    "cublas": (
        "ident",
        [
            "cublasCreate",
            "cublasDestroy",
            "cublasSetStream",
            "cublasGetStream",
            "cublasSgemm",
            "cublasDgemm",
            "cublasHgemm",
            "cublasGemmEx",
            "cublasGemmBatchedEx",
            "cublasGemmStridedBatchedEx",
            "cublasSgemv",
            "cublasDgemv",
            "cublasSaxpy",
            "cublasDaxpy",
            "cublasHandle_t",
            "cublasStatus_t",
            "cublasOperation_t",
            "cublasLtCreate",
            "cublasLtMatmul",
            "cublasLtMatmulDescCreate",
        ],
    ),
    "cudnn": (
        "ident",
        [
            "cudnnCreate",
            "cudnnDestroy",
            "cudnnSetStream",
            "cudnnCreateTensorDescriptor",
            "cudnnSetTensor4dDescriptor",
            "cudnnConvolutionForward",
            "cudnnConvolutionBackwardData",
            "cudnnConvolutionBackwardFilter",
            "cudnnBatchNormalizationForwardTraining",
            "cudnnBatchNormalizationBackward",
            "cudnnSoftmaxForward",
            "cudnnPoolingForward",
            "cudnnActivationForward",
            "cudnnHandle_t",
            "cudnnTensorDescriptor_t",
            "cudnnDataType_t",
        ],
    ),
    "thrust_cub": (
        "ident",
        [
            # Thrust (namespace-qualified in source)
            "device_vector",
            "host_vector",
            "device_ptr",
            "raw_pointer_cast",
            "make_zip_iterator",
            # CUB
            "BlockReduce",
            "BlockScan",
            "WarpReduce",
            "DeviceReduce",
            "DeviceScan",
            "DeviceRadixSort",
            "DeviceSelect",
        ],
    ),
    "cuda_atomics": (
        "ident",
        [
            "atomicAdd",
            "atomicSub",
            "atomicExch",
            "atomicMin",
            "atomicMax",
            "atomicAnd",
            "atomicOr",
            "atomicXor",
            "atomicCAS",
            "atomicInc",
            "__shfl_sync",
            "__shfl_up_sync",
            "__shfl_down_sync",
            "__shfl_xor_sync",
            "__ballot_sync",
            "__all_sync",
            "__any_sync",
            "__half",
            "__half2",
            "__nv_bfloat16",
            "__nv_bfloat162",
        ],
    ),
    "nccl": (
        "ident",
        [
            "ncclGetUniqueId",
            "ncclCommInitRank",
            "ncclCommInitAll",
            "ncclCommDestroy",
            "ncclAllReduce",
            "ncclBroadcast",
            "ncclReduce",
            "ncclAllGather",
            "ncclReduceScatter",
            "ncclSend",
            "ncclRecv",
            "ncclGroupStart",
            "ncclGroupEnd",
            "ncclComm_t",
            "ncclUniqueId",
            "ncclResult_t",
            "ncclSuccess",
        ],
    ),
    # ── ROCm/HIP ──
    "hip_runtime": (
        "ident",
        [
            "hipMalloc",
            "hipFree",
            "hipMemcpy",
            "hipMemcpyAsync",
            "hipMemset",
            "hipMallocManaged",
            "hipMallocHost",
            "hipFreeHost",
            "hipGetDevice",
            "hipSetDevice",
            "hipGetDeviceCount",
            "hipDeviceSynchronize",
            "hipStreamCreate",
            "hipStreamDestroy",
            "hipStreamSynchronize",
            "hipEventCreate",
            "hipEventDestroy",
            "hipEventRecord",
            "hipEventElapsedTime",
            "hipLaunchKernelGGL",
            "hipGetLastError",
            "hipGetErrorString",
            "hipError_t",
            "hipSuccess",
            "hipStream_t",
            "hipEvent_t",
            "hipDeviceProp_t",
            "HIP_DYNAMIC_SHARED",
        ],
    ),
    "rocblas_miopen": (
        "ident",
        [
            "rocblas_create_handle",
            "rocblas_destroy_handle",
            "rocblas_sgemm",
            "rocblas_dgemm",
            "rocblas_hgemm",
            "hipblasCreate",
            "hipblasDestroy",
            "hipblasSgemm",
            "hipblasDgemm",
            "hipblasGemmEx",
            "hipblasHandle_t",
            "hipblasStatus_t",
            "miopenCreateTensorDescriptor",
            "miopenConvolutionForward",
        ],
    ),
    # ── TPU/XLA ──
    "xla_ops": (
        "ident",
        [
            # These appear as string identifiers in XLA/MLIR code
            "HloOpcode",
            "HloInstruction",
            "HloModule",
            "HloComputation",
            "XlaBuilder",
            "XlaOp",
            "XlaComputation",
            "PjRtClient",
            "PjRtDevice",
            "PjRtBuffer",
            "StableHloOp",
            "StablehloDialect",
            "BlockSpec",
            "GridSpec",
            "Pallas",
        ],
    ),
    # ── Protocol Buffers ──
    "protobuf": (
        "ident",
        [
            "Message",
            "MessageLite",
            "Arena",
            "Descriptor",
            "FieldDescriptor",
            "EnumDescriptor",
            "ServiceDescriptor",
            "FileDescriptor",
            "OneofDescriptor",
            "DescriptorPool",
            "RepeatedField",
            "RepeatedPtrField",
            "CodedInputStream",
            "CodedOutputStream",
            "ZeroCopyInputStream",
            "ZeroCopyOutputStream",
            "SerializeToString",
            "ParseFromString",
            "SerializeToArray",
            "ParseFromArray",
            "ByteSizeLong",
            "IsInitialized",
            "CopyFrom",
            "MergeFrom",
            "GetDescriptor",
            "GetReflection",
            "DebugString",
            "ShortDebugString",
            "TextFormat",
            "UnknownFieldSet",
            "DynamicMessageFactory",
            "GOOGLE_PROTOBUF_VERIFY_VERSION",
        ],
    ),
    "proto_keywords": (
        "ident",
        [
            # Proto file keywords (appear in .proto files and string literals)
            "syntax",
            "package",
            "import",
            "option",
            "message",
            "enum",
            "service",
            "rpc",
            "returns",
            "repeated",
            "optional",
            "required",
            "oneof",
            "map",
            "reserved",
            "extensions",
            "extend",
            "stream",
        ],
    ),
    # ── gRPC ──
    "grpc": (
        "ident",
        [
            "Server",
            "ServerBuilder",
            "ServerContext",
            "Channel",
            "ClientContext",
            "CompletionQueue",
            "Status",
            "Service",
            "CallCredentials",
            "ChannelCredentials",
            "ServerAsyncReader",
            "ServerAsyncWriter",
            "ClientReader",
            "ClientWriter",
            "ClientReaderWriter",
            "CreateChannel",
            "InsecureChannelCredentials",
            "SslCredentials",
            # Status codes
            "CANCELLED",
            "INVALID_ARGUMENT",
            "DEADLINE_EXCEEDED",
            "NOT_FOUND",
            "ALREADY_EXISTS",
            "PERMISSION_DENIED",
            "UNAUTHENTICATED",
            "RESOURCE_EXHAUSTED",
            "FAILED_PRECONDITION",
            "ABORTED",
            "OUT_OF_RANGE",
            "UNIMPLEMENTED",
            "UNAVAILABLE",
            "DATA_LOSS",
        ],
    ),
    # ── GraphQL ──
    "graphql": (
        "ident",
        [
            "query",
            "mutation",
            "subscription",
            "fragment",
            "Document",
            "OperationDefinition",
            "SelectionSet",
            "Field",
            "Argument",
            "FragmentSpread",
            "InlineFragment",
            "NamedType",
            "ListType",
            "NonNullType",
            "SchemaDefinition",
            "DirectiveDefinition",
        ],
    ),
    # ── MongoDB ──
    "mongodb_dollar": (
        "dollar",
        [
            "$match",
            "$group",
            "$project",
            "$sort",
            "$limit",
            "$skip",
            "$unwind",
            "$lookup",
            "$graphLookup",
            "$merge",
            "$out",
            "$addFields",
            "$facet",
            "$set",
            "$unset",
            "$count",
            "$redact",
            "$eq",
            "$ne",
            "$gt",
            "$gte",
            "$lt",
            "$lte",
            "$in",
            "$nin",
            "$and",
            "$or",
            "$not",
            "$nor",
            "$exists",
            "$type",
            "$regex",
            "$elemMatch",
            "$inc",
            "$mul",
            "$push",
            "$pull",
            "$addToSet",
            "$pop",
        ],
    ),
    "mongodb_cpp": (
        "ident",
        [
            "mongocxx",
            "bsoncxx",
            "find_one",
            "insert_one",
            "insert_many",
            "update_one",
            "update_many",
            "delete_one",
            "delete_many",
            "aggregate",
            "count_documents",
        ],
    ),
    # ── Redis ──
    "redis_commands": (
        "ident",
        [
            # These appear as method calls or string constants
            "SUBSCRIBE",
            "UNSUBSCRIBE",
            "PUBLISH",
            "PSUBSCRIBE",
            "LPUSH",
            "RPUSH",
            "LPOP",
            "RPOP",
            "LRANGE",
            "SADD",
            "SREM",
            "SMEMBERS",
            "ZADD",
            "ZREM",
            "ZRANGE",
            "ZSCORE",
            "HGET",
            "HSET",
            "HMGET",
            "HMSET",
            "HDEL",
            "HGETALL",
            "XADD",
            "XREAD",
            "XREADGROUP",
            "hiredis",
        ],
    ),
    # ── SQL C API ──
    "sqlite3_api": (
        "ident",
        [
            "sqlite3_open",
            "sqlite3_open_v2",
            "sqlite3_close",
            "sqlite3_close_v2",
            "sqlite3_exec",
            "sqlite3_prepare_v2",
            "sqlite3_prepare_v3",
            "sqlite3_step",
            "sqlite3_finalize",
            "sqlite3_reset",
            "sqlite3_bind_int",
            "sqlite3_bind_int64",
            "sqlite3_bind_double",
            "sqlite3_bind_text",
            "sqlite3_bind_blob",
            "sqlite3_bind_null",
            "sqlite3_column_int",
            "sqlite3_column_int64",
            "sqlite3_column_double",
            "sqlite3_column_text",
            "sqlite3_column_blob",
            "sqlite3_column_count",
            "sqlite3_column_type",
            "sqlite3_column_name",
            "sqlite3_errmsg",
            "sqlite3_errcode",
            "sqlite3_last_insert_rowid",
            "sqlite3_changes",
            "sqlite3_free",
            "sqlite3_malloc",
        ],
    ),
    "mysql_api": (
        "ident",
        [
            "mysql_init",
            "mysql_real_connect",
            "mysql_close",
            "mysql_query",
            "mysql_real_query",
            "mysql_store_result",
            "mysql_use_result",
            "mysql_fetch_row",
            "mysql_fetch_field",
            "mysql_num_rows",
            "mysql_num_fields",
            "mysql_free_result",
            "mysql_affected_rows",
            "mysql_errno",
            "mysql_error",
            "mysql_autocommit",
            "mysql_commit",
            "mysql_rollback",
        ],
    ),
    "odbc_api": (
        "ident",
        [
            "SQLAllocHandle",
            "SQLFreeHandle",
            "SQLConnect",
            "SQLDisconnect",
            "SQLDriverConnect",
            "SQLPrepare",
            "SQLExecute",
            "SQLExecDirect",
            "SQLFetch",
            "SQLFetchScroll",
            "SQLGetData",
            "SQLBindCol",
            "SQLBindParameter",
            "SQLNumResultCols",
            "SQLRowCount",
            "SQLEndTran",
            "SQLCloseCursor",
            "SQLCancel",
            "SQLGetDiagRec",
            "SQLGetDiagField",
        ],
    ),
    # ── SQL Keywords (in string literals) ──
    "sql_keywords": (
        "sql_string",
        [
            "SELECT",
            "INSERT",
            "UPDATE",
            "DELETE",
            "CREATE",
            "ALTER",
            "DROP",
            "FROM",
            "WHERE",
            "JOIN",
            "INNER",
            "LEFT",
            "RIGHT",
            "OUTER",
            "CROSS",
            "GROUP",
            "ORDER",
            "HAVING",
            "LIMIT",
            "OFFSET",
            "UNION",
            "EXCEPT",
            "DISTINCT",
            "EXISTS",
            "BETWEEN",
            "LIKE",
            "TABLE",
            "INDEX",
            "VIEW",
            "TRIGGER",
            "PROCEDURE",
            "FUNCTION",
            "PRIMARY",
            "FOREIGN",
            "CONSTRAINT",
            "DEFAULT",
            "UNIQUE",
            "CHECK",
            "BEGIN",
            "COMMIT",
            "ROLLBACK",
            "TRANSACTION",
            "SAVEPOINT",
            "INTEGER",
            "VARCHAR",
            "TEXT",
            "BLOB",
            "REAL",
            "BOOLEAN",
            "COUNT",
            "SUM",
            "AVG",
            "MIN",
            "MAX",
            "PRAGMA",
            "VACUUM",
            "EXPLAIN",
            "ANALYZE",
        ],
    ),
    # ── C++ ORMs ──
    "cpp_orms": (
        "ident",
        [
            # sqlpp11
            "insert_into",
            "remove_from",
            "left_outer_join",
            "right_outer_join",
            "cross_join",
            "inner_join",
            "dynamic_select",
            "dynamic_where",
            # SOCI
            "rowset",
            "got_data",
            "backend_factory",
            # ODB
            "schema_catalog",
            "prepared_query",
        ],
    ),
    # ── C++23/26 ──
    "cpp23_types": (
        "ident",
        [
            "expected",
            "unexpected",
            "move_only_function",
            "bind_back",
            "unreachable",
            "to_underlying",
            "byteswap",
            "invoke_r",
            "forward_like",
            "out_ptr",
            "inout_ptr",
            "start_lifetime_as",
            "flat_map",
            "flat_set",
            "flat_multimap",
            "flat_multiset",
            "mdspan",
            "generator",
            "stacktrace",
            "source_location",
            "is_scoped_enum",
            "is_implicit_lifetime",
        ],
    ),
    "cpp23_ranges": (
        "ident",
        [
            "zip",
            "zip_transform",
            "adjacent",
            "chunk",
            "slide",
            "stride",
            "cartesian_product",
            "enumerate",
            "as_const",
            "as_rvalue",
        ],
    ),
    "cpp26_features": (
        "ident",
        [
            "contract_assert",
            "define_class",
            "substitute",
            "reflect_value",
            "scheduler",
            "sender",
            "receiver",
            "run_loop",
            "static_thread_pool",
            "just",
            "let_value",
            "when_all",
            "transfer",
            "schedule",
            "start_detached",
            "sync_wait",
            "counting_scope",
            "fetch_max",
            "fetch_min",
            "is_within_lifetime",
            "text_encoding",
        ],
    ),
    "cpp20_concepts": (
        "ident",
        [
            "same_as",
            "derived_from",
            "convertible_to",
            "integral",
            "floating_point",
            "signed_integral",
            "unsigned_integral",
            "destructible",
            "constructible_from",
            "movable",
            "copyable",
            "semiregular",
            "regular",
            "equality_comparable",
            "totally_ordered",
            "three_way_comparable",
            "invocable",
            "predicate",
            "input_iterator",
            "forward_iterator",
            "bidirectional_iterator",
            "random_access_iterator",
            "contiguous_iterator",
            "input_range",
            "forward_range",
            "bidirectional_range",
            "random_access_range",
            "contiguous_range",
            "sized_range",
        ],
    ),
    # ── Testing/Build Frameworks ──
    "gtest": (
        "ident",
        [
            "TEST",
            "TEST_F",
            "TEST_P",
            "TYPED_TEST",
            "TYPED_TEST_SUITE",
            "EXPECT_EQ",
            "EXPECT_NE",
            "EXPECT_LT",
            "EXPECT_GT",
            "EXPECT_LE",
            "EXPECT_GE",
            "EXPECT_TRUE",
            "EXPECT_FALSE",
            "EXPECT_THAT",
            "ASSERT_EQ",
            "ASSERT_NE",
            "ASSERT_LT",
            "ASSERT_GT",
            "ASSERT_LE",
            "ASSERT_GE",
            "ASSERT_TRUE",
            "ASSERT_FALSE",
            "ASSERT_THAT",
            "EXPECT_THROW",
            "EXPECT_NO_THROW",
            "EXPECT_DEATH",
            "ASSERT_THROW",
            "ASSERT_NO_THROW",
            "ASSERT_DEATH",
            "MOCK_METHOD",
            "EXPECT_CALL",
            "ON_CALL",
            "SetUp",
            "TearDown",
            "SetUpTestSuite",
            "TearDownTestSuite",
        ],
    ),
    "cmake": (
        "ident",
        [
            "cmake_minimum_required",
            "add_executable",
            "add_library",
            "target_link_libraries",
            "target_include_directories",
            "target_compile_definitions",
            "target_compile_options",
            "find_package",
            "find_library",
            "find_path",
            "find_program",
            "include_directories",
            "link_directories",
            "add_subdirectory",
            "configure_file",
            "install",
            "CMAKE_CXX_STANDARD",
            "CMAKE_CXX_FLAGS",
            "CMAKE_BUILD_TYPE",
            "CMAKE_INSTALL_PREFIX",
            "CMAKE_SOURCE_DIR",
            "CMAKE_BINARY_DIR",
        ],
    ),
    "catch_boost_test": (
        "ident",
        [
            "BOOST_AUTO_TEST_CASE",
            "BOOST_CHECK",
            "BOOST_REQUIRE",
            "BOOST_TEST",
            "BOOST_FIXTURE_TEST_SUITE",
            "BOOST_AUTO_TEST_SUITE",
            "BOOST_CHECK_EQUAL",
            "BOOST_REQUIRE_EQUAL",
            "REQUIRE",
            "CHECK",
            "SECTION",
            "TEST_CASE",
            "SCENARIO",
            "GIVEN",
            "WHEN",
            "THEN",
        ],
    ),
    # ── Compiler Attributes ──
    "attributes": (
        "ident",
        [
            "__attribute__",
            "__declspec",
            "__stdcall",
            "__cdecl",
            "__fastcall",
            "__forceinline",
            "__asm__",
            "nodiscard",
            "maybe_unused",
            "deprecated",
            "likely",
            "unlikely",
            "no_unique_address",
            "carries_dependency",
            "fallthrough",
            "__cplusplus",
            "__FILE__",
            "__LINE__",
            "__func__",
            "__FUNCTION__",
        ],
    ),
}

# ── Morpheme stems for substring analysis ──
MORPHEME_STEMS = {
    "prefixes": [
        "pre",
        "post",
        "un",
        "re",
        "de",
        "dis",
        "non",
        "sub",
        "super",
        "over",
        "under",
        "inter",
        "intra",
        "multi",
        "poly",
        "mono",
        "bi",
        "tri",
        "semi",
        "pseudo",
        "meta",
        "proto",
        "anti",
        "co",
    ],
    "suffixes": [
        "able",
        "ible",
        "tion",
        "sion",
        "ment",
        "ness",
        "ful",
        "less",
        "ize",
        "ify",
        "ate",
        "ent",
        "ant",
        "ary",
        "ory",
        "er",
        "or",
        "ist",
        "ed",
        "ing",
        "ly",
        "al",
        "ic",
    ],
    "cpp_stems": [
        "alloc",
        "dealloc",
        "init",
        "deinit",
        "lock",
        "unlock",
        "push",
        "pop",
        "read",
        "write",
        "open",
        "close",
        "begin",
        "end",
        "start",
        "stop",
        "create",
        "destroy",
        "insert",
        "remove",
        "find",
        "search",
        "sort",
        "swap",
        "load",
        "save",
        "parse",
        "format",
        "encode",
        "decode",
    ],
    "common_components": [
        "buffer",
        "cache",
        "queue",
        "stack",
        "list",
        "tree",
        "node",
        "graph",
        "handler",
        "manager",
        "factory",
        "builder",
        "adapter",
        "wrapper",
        "proxy",
        "callback",
        "listener",
        "observer",
        "visitor",
        "iterator",
        "generator",
        "config",
        "context",
        "session",
        "request",
        "response",
        "message",
        "event",
        "value",
        "index",
        "count",
        "total",
        "offset",
        "length",
        "capacity",
        "impl",
        "ctx",
        "ptr",
        "buf",
        "cfg",
        "msg",
        "req",
        "res",
        "arg",
        "param",
        "iter",
        "prev",
        "curr",
        "next",
        "tmp",
        "src",
        "dst",
    ],
}

# ─── Token Extraction ────────────────────────────────────────────────────────

# C++ identifiers including namespace-qualified (std::vector, google::protobuf::Message)
IDENT_RE = re.compile(r"[a-zA-Z_]\w*(?:::\w+)*")

# $-prefixed operators (MongoDB)
DOLLAR_RE = re.compile(r"\$[a-zA-Z_]\w*")

# String literal contents (for SQL keyword detection)
# Handles escaped quotes inside strings
STRING_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')


# Camel/snake case splitter for morpheme analysis
def split_identifier(ident: str) -> list[str]:
    """Split a C++ identifier into morpheme components."""
    parts = []
    for chunk in ident.split("_"):
        if not chunk:
            continue
        # Split camelCase: maxActiveBlocks -> [max, Active, Blocks]
        # Split UPPER runs: HTTPSConnection -> [HTTPS, Connection]
        tokens = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+", chunk)
        parts.extend(t.lower() for t in tokens if len(t) > 1)
    return parts


# ─── Document Processing ─────────────────────────────────────────────────────


def analyze_document(text: str) -> dict:
    """Analyze a single C++ document for proposed vocabulary tokens.

    Returns dict with:
        ident_counts: Counter of all identifiers
        dollar_counts: Counter of $-prefixed operators
        sql_keyword_counts: Counter of SQL keywords found in string literals
        morpheme_counts: Counter of identifier morpheme components
    """
    # Pass 1: Extract all C++ identifiers
    ident_counts = Counter(IDENT_RE.findall(text))

    # Pass 2: Extract $-prefixed operators
    dollar_counts = Counter(DOLLAR_RE.findall(text))

    # Pass 3: Extract SQL keywords from string literals
    sql_keyword_counts = Counter()
    for string_content in STRING_RE.findall(text):
        # SQL keywords are typically uppercase, 2+ chars
        for word in re.findall(r"\b[A-Z][A-Z_]{1,}\b", string_content):
            sql_keyword_counts[word] += 1

    # Pass 4: Morpheme analysis - split identifiers into components
    morpheme_counts = Counter()
    for ident in ident_counts:
        if len(ident) > 3:  # Skip trivially short identifiers
            freq = ident_counts[ident]
            for part in split_identifier(ident):
                morpheme_counts[part] += freq

    return {
        "ident_counts": ident_counts,
        "dollar_counts": dollar_counts,
        "sql_keyword_counts": sql_keyword_counts,
        "morpheme_counts": morpheme_counts,
    }


def process_row_group(args):
    """Process a single parquet row group. Designed for multiprocessing."""
    parquet_path, rg_idx = args
    pf = pq.ParquetFile(parquet_path)
    rg = pf.read_row_group(rg_idx)
    texts = rg.column("text").to_pylist()

    merged = {
        "ident_counts": Counter(),
        "dollar_counts": Counter(),
        "sql_keyword_counts": Counter(),
        "morpheme_counts": Counter(),
        "num_docs": len(texts),
        "doc_presence": defaultdict(int),  # token -> num docs containing it
    }

    for text in texts:
        result = analyze_document(text)
        for key in (
            "ident_counts",
            "dollar_counts",
            "sql_keyword_counts",
            "morpheme_counts",
        ):
            merged[key] += result[key]

        # Track document presence (for doc_freq calculation)
        seen = set()
        for key in ("ident_counts", "dollar_counts", "sql_keyword_counts"):
            seen.update(result[key].keys())
        for token in seen:
            merged["doc_presence"][token] += 1

    return merged


# ─── Results Aggregation ─────────────────────────────────────────────────────


def aggregate_results(results: list[dict]) -> dict:
    """Merge results from multiple row groups."""
    merged = {
        "ident_counts": Counter(),
        "dollar_counts": Counter(),
        "sql_keyword_counts": Counter(),
        "morpheme_counts": Counter(),
        "num_docs": 0,
        "doc_presence": Counter(),
    }
    for r in results:
        for key in (
            "ident_counts",
            "dollar_counts",
            "sql_keyword_counts",
            "morpheme_counts",
        ):
            merged[key] += r[key]
        merged["num_docs"] += r["num_docs"]
        merged["doc_presence"] += Counter(r["doc_presence"])
    return merged


def lookup_proposed_vocab(merged: dict) -> dict:
    """Look up each proposed vocab token in the merged results."""
    results = {}
    for category, (match_mode, tokens) in PROPOSED_VOCAB.items():
        category_results = []
        for token in tokens:
            if match_mode == "ident":
                count = merged["ident_counts"].get(token, 0)
                doc_count = merged["doc_presence"].get(token, 0)
            elif match_mode == "dollar":
                count = merged["dollar_counts"].get(token, 0)
                doc_count = merged["doc_presence"].get(token, 0)
            elif match_mode == "sql_string":
                count = merged["sql_keyword_counts"].get(token, 0)
                doc_count = merged["doc_presence"].get(token, 0)
            else:
                count = 0
                doc_count = 0

            category_results.append(
                {
                    "token": token,
                    "count": count,
                    "doc_count": doc_count,
                }
            )

        # Sort by count descending
        category_results.sort(key=lambda x: x["count"], reverse=True)
        results[category] = category_results

    return results


def lookup_morphemes(merged: dict) -> dict:
    """Look up morpheme stems in the morpheme analysis results."""
    results = {}
    for category, stems in MORPHEME_STEMS.items():
        category_results = []
        for stem in stems:
            count = merged["morpheme_counts"].get(stem, 0)
            category_results.append({"stem": stem, "count": count})
        category_results.sort(key=lambda x: x["count"], reverse=True)
        results[category] = category_results
    return results


# ─── Output Formatting ───────────────────────────────────────────────────────


def print_category_report(category: str, results: list[dict], num_docs: int):
    """Print a formatted report for one category."""
    found = sum(1 for r in results if r["count"] > 0)
    total = len(results)
    total_hits = sum(r["count"] for r in results)
    coverage = found / total * 100 if total > 0 else 0

    print(f"\n{'═' * 78}")
    print(
        f"  {category.upper().replace('_', ' ')}  ({found}/{total} found, {coverage:.0f}% coverage)"
    )
    print(f"  Total occurrences: {total_hits:,}")
    print(f"{'═' * 78}")

    if not results:
        print("  (no tokens defined)")
        return

    # Header
    "token" if "token" in results[0] else "stem"
    print(f"  {'Token':<45s} {'Count':>10s} {'Docs':>8s} {'Doc%':>7s}  Status")
    print(f"  {'─' * 45} {'─' * 10} {'─' * 8} {'─' * 7}  {'─' * 12}")

    for r in results:
        token = r.get("token", r.get("stem", "?"))
        count = r["count"]
        doc_count = r.get("doc_count", 0)
        doc_pct = doc_count / num_docs * 100 if num_docs > 0 else 0

        if count == 0:
            status = "NOT FOUND"
        elif count < 10:
            status = "RARE"
        elif count < 100:
            status = "LOW"
        elif count < 1000:
            status = "MODERATE"
        elif count < 10000:
            status = "HIGH"
        else:
            status = "VERY HIGH"

        print(
            f"  {token:<45s} {count:>10,d} {doc_count:>8,d} {doc_pct:>6.1f}%  {status}"
        )


def print_summary(vocab_results: dict, morpheme_results: dict, num_docs: int):
    """Print overall summary across all categories."""
    print(f"\n{'━' * 78}")
    print(f"  OVERALL SUMMARY  ({num_docs:,} documents analyzed)")
    print(f"{'━' * 78}")

    print(
        f"\n  {'Category':<30s} {'Proposed':>8s} {'Found':>8s} {'Coverage':>8s} {'Total Hits':>12s}"
    )
    print(f"  {'─' * 30} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 12}")

    grand_proposed = 0
    grand_found = 0
    grand_hits = 0

    for category, results in vocab_results.items():
        total = len(results)
        found = sum(1 for r in results if r["count"] > 0)
        hits = sum(r["count"] for r in results)
        coverage = found / total * 100 if total > 0 else 0

        grand_proposed += total
        grand_found += found
        grand_hits += hits

        print(
            f"  {category:<30s} {total:>8d} {found:>8d} {coverage:>7.0f}% {hits:>12,d}"
        )

    print(f"  {'─' * 30} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 12}")
    grand_coverage = grand_found / grand_proposed * 100 if grand_proposed > 0 else 0
    print(
        f"  {'TOTAL':<30s} {grand_proposed:>8d} {grand_found:>8d} {grand_coverage:>7.0f}% {grand_hits:>12,d}"
    )

    if morpheme_results:
        print("\n  MORPHEME ANALYSIS:")
        for category, results in morpheme_results.items():
            found = sum(1 for r in results if r["count"] > 0)
            total = len(results)
            hits = sum(r["count"] for r in results)
            print(
                f"  {category:<30s} {total:>8d} {found:>8d} {found / total * 100:>7.0f}% {hits:>12,d}"
            )


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Analyze proposed tokenizer v3 vocabulary frequency in C++ corpus"
    )
    parser.add_argument(
        "parquet_files", nargs="+", help="Parquet file(s) or glob pattern"
    )
    parser.add_argument(
        "-c",
        "--category",
        type=str,
        default=None,
        help="Only show specific category (partial match)",
    )
    parser.add_argument(
        "--morphemes", action="store_true", help="Include morpheme stem analysis"
    )
    parser.add_argument(
        "--top", type=int, default=None, help="Only show top N tokens per category"
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=4, help="Number of parallel workers"
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Write results to file instead of stdout",
    )
    parser.add_argument(
        "--max-row-groups",
        type=int,
        default=None,
        help="Limit row groups processed per file (for quick testing)",
    )
    args = parser.parse_args()

    # Expand glob patterns
    parquet_files = []
    for pattern in args.parquet_files:
        expanded = glob.glob(pattern)
        if expanded:
            parquet_files.extend(expanded)
        elif os.path.isdir(pattern):
            parquet_files.extend(glob.glob(os.path.join(pattern, "*.parquet")))
        else:
            print(f"Warning: {pattern} not found", file=sys.stderr)
    parquet_files = sorted(set(parquet_files))

    if not parquet_files:
        print("Error: No parquet files found", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing {len(parquet_files)} parquet file(s)...")
    for f in parquet_files:
        sz = os.path.getsize(f) / (1024 * 1024)
        num_rg = pq.ParquetFile(f).metadata.num_row_groups
        print(f"  {os.path.basename(f)}: {sz:.0f} MB, {num_rg} row groups")

    # Build work items
    work_items = []
    for pf_path in parquet_files:
        pf = pq.ParquetFile(pf_path)
        num_rg = pf.metadata.num_row_groups
        if args.max_row_groups:
            num_rg = min(num_rg, args.max_row_groups)
        for rg_idx in range(num_rg):
            work_items.append((pf_path, rg_idx))

    print(f"Processing {len(work_items)} row groups with {args.workers} workers...")
    t0 = time.time()

    # Process in parallel
    all_results = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_row_group, item): item for item in work_items
        }
        for future in as_completed(futures):
            result = future.result()
            all_results.append(result)
            done += 1
            if done % 10 == 0 or done == len(work_items):
                elapsed = time.time() - t0
                docs = sum(r["num_docs"] for r in all_results)
                print(
                    f"  [{done}/{len(work_items)}] {docs:,} docs, {elapsed:.1f}s",
                    file=sys.stderr,
                )

    elapsed = time.time() - t0

    # Aggregate
    merged = aggregate_results(all_results)
    num_docs = merged["num_docs"]
    print(
        f"\nCompleted: {num_docs:,} documents in {elapsed:.1f}s ({num_docs / elapsed:.0f} docs/s)"
    )

    # Lookup proposed vocab
    vocab_results = lookup_proposed_vocab(merged)
    morpheme_results = lookup_morphemes(merged) if args.morphemes else {}

    # Filter by category if requested
    if args.category:
        cat_filter = args.category.lower()
        vocab_results = {
            k: v for k, v in vocab_results.items() if cat_filter in k.lower()
        }
        morpheme_results = {
            k: v for k, v in morpheme_results.items() if cat_filter in k.lower()
        }

    # Truncate to top N if requested
    if args.top:
        vocab_results = {k: v[: args.top] for k, v in vocab_results.items()}
        morpheme_results = {k: v[: args.top] for k, v in morpheme_results.items()}

    if args.json:
        output = {
            "num_docs": num_docs,
            "elapsed_s": elapsed,
            "vocab": vocab_results,
            "morphemes": morpheme_results,
        }
        text = json.dumps(output, indent=2)
        if args.output:
            with open(args.output, "w") as f:
                f.write(text)
            print(f"Results written to {args.output}")
        else:
            print(text)
    else:
        # Print per-category reports
        for category, results in vocab_results.items():
            print_category_report(category, results, num_docs)

        if morpheme_results:
            print(f"\n{'═' * 78}")
            print("  MORPHEME STEM ANALYSIS")
            print(f"{'═' * 78}")
            for category, results in morpheme_results.items():
                print_category_report(category, results, num_docs)

        # Print summary
        print_summary(vocab_results, morpheme_results, num_docs)

        # Save JSON alongside for programmatic use
        if args.output:
            output = {
                "num_docs": num_docs,
                "elapsed_s": elapsed,
                "vocab": vocab_results,
                "morphemes": morpheme_results,
            }
            with open(args.output, "w") as f:
                json.dump(output, f, indent=2)
            print(f"\nJSON results saved to {args.output}")


if __name__ == "__main__":
    main()
