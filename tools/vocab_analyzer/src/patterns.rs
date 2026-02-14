/// Token pattern definitions for vocabulary frequency analysis.
///
/// Each category has a match mode and list of tokens:
/// - Ident: match as C++ identifiers (word boundary)
/// - Dollar: match $-prefixed operators
/// - SqlString: match uppercase keywords inside string literals

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatchMode {
    Ident,
    Dollar,
    SqlString,
}

pub struct Category {
    pub name: &'static str,
    pub mode: MatchMode,
    pub tokens: &'static [&'static str],
}

pub static CATEGORIES: &[Category] = &[
    // ── CUDA ──
    Category {
        name: "cuda_qualifiers",
        mode: MatchMode::Ident,
        tokens: &[
            "__global__", "__device__", "__host__", "__shared__", "__constant__",
            "__managed__", "__launch_bounds__", "__restrict__",
            "threadIdx", "blockIdx", "blockDim", "gridDim", "warpSize",
            "__syncthreads", "__threadfence", "__threadfence_block", "__threadfence_system",
            "dim3",
        ],
    },
    Category {
        name: "cuda_runtime",
        mode: MatchMode::Ident,
        tokens: &[
            "cudaMalloc", "cudaFree", "cudaMemcpy", "cudaMemcpyAsync", "cudaMemset",
            "cudaMallocManaged", "cudaMallocHost", "cudaFreeHost", "cudaHostAlloc",
            "cudaMallocPitch", "cudaMalloc3D", "cudaMemcpy2D", "cudaMemcpy3D",
            "cudaMemcpyHostToHost", "cudaMemcpyHostToDevice", "cudaMemcpyDeviceToHost",
            "cudaMemcpyDeviceToDevice", "cudaMemcpyDefault",
            "cudaGetDevice", "cudaSetDevice", "cudaGetDeviceCount", "cudaGetDeviceProperties",
            "cudaDeviceSynchronize", "cudaDeviceReset", "cudaDeviceGetAttribute",
            "cudaStreamCreate", "cudaStreamDestroy", "cudaStreamSynchronize", "cudaStreamWaitEvent",
            "cudaEventCreate", "cudaEventDestroy", "cudaEventRecord", "cudaEventSynchronize",
            "cudaEventElapsedTime", "cudaLaunchKernel",
            "cudaFuncGetAttributes", "cudaFuncSetAttribute",
            "cudaGetLastError", "cudaPeekAtLastError", "cudaGetErrorString", "cudaGetErrorName",
            "cudaError_t", "cudaSuccess", "cudaStream_t", "cudaEvent_t",
            "cudaDeviceProp", "cudaMemcpyKind",
        ],
    },
    Category {
        name: "cublas",
        mode: MatchMode::Ident,
        tokens: &[
            "cublasCreate", "cublasDestroy", "cublasSetStream", "cublasGetStream",
            "cublasSgemm", "cublasDgemm", "cublasHgemm",
            "cublasGemmEx", "cublasGemmBatchedEx", "cublasGemmStridedBatchedEx",
            "cublasSgemv", "cublasDgemv", "cublasSaxpy", "cublasDaxpy",
            "cublasHandle_t", "cublasStatus_t", "cublasOperation_t",
            "cublasLtCreate", "cublasLtMatmul", "cublasLtMatmulDescCreate",
        ],
    },
    Category {
        name: "cudnn",
        mode: MatchMode::Ident,
        tokens: &[
            "cudnnCreate", "cudnnDestroy", "cudnnSetStream",
            "cudnnCreateTensorDescriptor", "cudnnSetTensor4dDescriptor",
            "cudnnConvolutionForward", "cudnnConvolutionBackwardData", "cudnnConvolutionBackwardFilter",
            "cudnnBatchNormalizationForwardTraining", "cudnnBatchNormalizationBackward",
            "cudnnSoftmaxForward", "cudnnPoolingForward", "cudnnActivationForward",
            "cudnnHandle_t", "cudnnTensorDescriptor_t", "cudnnDataType_t",
        ],
    },
    Category {
        name: "thrust_cub",
        mode: MatchMode::Ident,
        tokens: &[
            "device_vector", "host_vector", "device_ptr",
            "raw_pointer_cast", "make_zip_iterator",
            "BlockReduce", "BlockScan", "WarpReduce",
            "DeviceReduce", "DeviceScan", "DeviceRadixSort", "DeviceSelect",
        ],
    },
    Category {
        name: "cuda_atomics",
        mode: MatchMode::Ident,
        tokens: &[
            "atomicAdd", "atomicSub", "atomicExch", "atomicMin", "atomicMax",
            "atomicAnd", "atomicOr", "atomicXor", "atomicCAS", "atomicInc",
            "__shfl_sync", "__shfl_up_sync", "__shfl_down_sync", "__shfl_xor_sync",
            "__ballot_sync", "__all_sync", "__any_sync",
            "__half", "__half2", "__nv_bfloat16", "__nv_bfloat162",
        ],
    },
    Category {
        name: "nccl",
        mode: MatchMode::Ident,
        tokens: &[
            "ncclGetUniqueId", "ncclCommInitRank", "ncclCommInitAll", "ncclCommDestroy",
            "ncclAllReduce", "ncclBroadcast", "ncclReduce", "ncclAllGather", "ncclReduceScatter",
            "ncclSend", "ncclRecv", "ncclGroupStart", "ncclGroupEnd",
            "ncclComm_t", "ncclUniqueId", "ncclResult_t", "ncclSuccess",
        ],
    },
    // ── ROCm/HIP ──
    Category {
        name: "hip_runtime",
        mode: MatchMode::Ident,
        tokens: &[
            "hipMalloc", "hipFree", "hipMemcpy", "hipMemcpyAsync", "hipMemset",
            "hipMallocManaged", "hipMallocHost", "hipFreeHost",
            "hipGetDevice", "hipSetDevice", "hipGetDeviceCount", "hipDeviceSynchronize",
            "hipStreamCreate", "hipStreamDestroy", "hipStreamSynchronize",
            "hipEventCreate", "hipEventDestroy", "hipEventRecord", "hipEventElapsedTime",
            "hipLaunchKernelGGL", "hipGetLastError", "hipGetErrorString",
            "hipError_t", "hipSuccess", "hipStream_t", "hipEvent_t", "hipDeviceProp_t",
            "HIP_DYNAMIC_SHARED",
        ],
    },
    Category {
        name: "rocblas_miopen",
        mode: MatchMode::Ident,
        tokens: &[
            "rocblas_create_handle", "rocblas_destroy_handle",
            "rocblas_sgemm", "rocblas_dgemm", "rocblas_hgemm",
            "hipblasCreate", "hipblasDestroy",
            "hipblasSgemm", "hipblasDgemm", "hipblasGemmEx",
            "hipblasHandle_t", "hipblasStatus_t",
            "miopenCreateTensorDescriptor", "miopenConvolutionForward",
        ],
    },
    // ── TPU/XLA ──
    Category {
        name: "xla_ops",
        mode: MatchMode::Ident,
        tokens: &[
            "HloOpcode", "HloInstruction", "HloModule", "HloComputation",
            "XlaBuilder", "XlaOp", "XlaComputation",
            "PjRtClient", "PjRtDevice", "PjRtBuffer",
            "StableHloOp", "StablehloDialect", "BlockSpec", "GridSpec", "Pallas",
        ],
    },
    // ── Protocol Buffers ──
    Category {
        name: "protobuf",
        mode: MatchMode::Ident,
        tokens: &[
            "Message", "MessageLite", "Arena", "Descriptor",
            "FieldDescriptor", "EnumDescriptor", "ServiceDescriptor", "FileDescriptor",
            "OneofDescriptor", "DescriptorPool",
            "RepeatedField", "RepeatedPtrField",
            "CodedInputStream", "CodedOutputStream",
            "ZeroCopyInputStream", "ZeroCopyOutputStream",
            "SerializeToString", "ParseFromString", "SerializeToArray", "ParseFromArray",
            "ByteSizeLong", "IsInitialized", "CopyFrom", "MergeFrom",
            "GetDescriptor", "GetReflection", "DebugString", "ShortDebugString",
            "TextFormat", "UnknownFieldSet", "DynamicMessageFactory",
            "GOOGLE_PROTOBUF_VERIFY_VERSION",
        ],
    },
    Category {
        name: "proto_keywords",
        mode: MatchMode::Ident,
        tokens: &[
            "syntax", "package", "import", "option", "message", "enum", "service",
            "rpc", "returns", "repeated", "optional", "required", "oneof",
            "map", "reserved", "extensions", "extend", "stream",
        ],
    },
    // ── gRPC ──
    Category {
        name: "grpc",
        mode: MatchMode::Ident,
        tokens: &[
            "Server", "ServerBuilder", "ServerContext", "Channel", "ClientContext",
            "CompletionQueue", "Status", "Service",
            "CallCredentials", "ChannelCredentials",
            "ServerAsyncReader", "ServerAsyncWriter",
            "ClientReader", "ClientWriter", "ClientReaderWriter",
            "CreateChannel", "InsecureChannelCredentials", "SslCredentials",
            "CANCELLED", "INVALID_ARGUMENT", "DEADLINE_EXCEEDED", "NOT_FOUND",
            "ALREADY_EXISTS", "PERMISSION_DENIED", "UNAUTHENTICATED", "RESOURCE_EXHAUSTED",
            "FAILED_PRECONDITION", "ABORTED", "OUT_OF_RANGE", "UNIMPLEMENTED",
            "UNAVAILABLE", "DATA_LOSS",
        ],
    },
    // ── GraphQL ──
    Category {
        name: "graphql",
        mode: MatchMode::Ident,
        tokens: &[
            "query", "mutation", "subscription", "fragment",
            "Document", "OperationDefinition", "SelectionSet", "Field", "Argument",
            "FragmentSpread", "InlineFragment", "NamedType", "ListType",
            "NonNullType", "SchemaDefinition", "DirectiveDefinition",
        ],
    },
    // ── MongoDB ──
    Category {
        name: "mongodb_dollar",
        mode: MatchMode::Dollar,
        tokens: &[
            "$match", "$group", "$project", "$sort", "$limit", "$skip",
            "$unwind", "$lookup", "$graphLookup", "$merge", "$out",
            "$addFields", "$facet", "$set", "$unset", "$count", "$redact",
            "$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin",
            "$and", "$or", "$not", "$nor", "$exists", "$type", "$regex",
            "$elemMatch", "$inc", "$mul", "$push", "$pull", "$addToSet", "$pop",
        ],
    },
    Category {
        name: "mongodb_cpp",
        mode: MatchMode::Ident,
        tokens: &[
            "mongocxx", "bsoncxx", "find_one", "insert_one", "insert_many",
            "update_one", "update_many", "delete_one", "delete_many",
            "aggregate", "count_documents",
        ],
    },
    // ── Redis ──
    Category {
        name: "redis_commands",
        mode: MatchMode::Ident,
        tokens: &[
            "SUBSCRIBE", "UNSUBSCRIBE", "PUBLISH", "PSUBSCRIBE",
            "LPUSH", "RPUSH", "LPOP", "RPOP", "LRANGE",
            "SADD", "SREM", "SMEMBERS",
            "ZADD", "ZREM", "ZRANGE", "ZSCORE",
            "HGET", "HSET", "HMGET", "HMSET", "HDEL", "HGETALL",
            "XADD", "XREAD", "XREADGROUP", "hiredis",
        ],
    },
    // ── SQL C APIs ──
    Category {
        name: "sqlite3_api",
        mode: MatchMode::Ident,
        tokens: &[
            "sqlite3_open", "sqlite3_open_v2", "sqlite3_close", "sqlite3_close_v2",
            "sqlite3_exec", "sqlite3_prepare_v2", "sqlite3_prepare_v3",
            "sqlite3_step", "sqlite3_finalize", "sqlite3_reset",
            "sqlite3_bind_int", "sqlite3_bind_int64", "sqlite3_bind_double",
            "sqlite3_bind_text", "sqlite3_bind_blob", "sqlite3_bind_null",
            "sqlite3_column_int", "sqlite3_column_int64", "sqlite3_column_double",
            "sqlite3_column_text", "sqlite3_column_blob",
            "sqlite3_column_count", "sqlite3_column_type", "sqlite3_column_name",
            "sqlite3_errmsg", "sqlite3_errcode",
            "sqlite3_last_insert_rowid", "sqlite3_changes",
            "sqlite3_free", "sqlite3_malloc",
        ],
    },
    Category {
        name: "mysql_api",
        mode: MatchMode::Ident,
        tokens: &[
            "mysql_init", "mysql_real_connect", "mysql_close",
            "mysql_query", "mysql_real_query",
            "mysql_store_result", "mysql_use_result",
            "mysql_fetch_row", "mysql_fetch_field",
            "mysql_num_rows", "mysql_num_fields", "mysql_free_result",
            "mysql_affected_rows", "mysql_errno", "mysql_error",
            "mysql_autocommit", "mysql_commit", "mysql_rollback",
        ],
    },
    Category {
        name: "odbc_api",
        mode: MatchMode::Ident,
        tokens: &[
            "SQLAllocHandle", "SQLFreeHandle", "SQLConnect", "SQLDisconnect",
            "SQLDriverConnect", "SQLPrepare", "SQLExecute", "SQLExecDirect",
            "SQLFetch", "SQLFetchScroll", "SQLGetData", "SQLBindCol",
            "SQLBindParameter", "SQLNumResultCols", "SQLRowCount", "SQLEndTran",
            "SQLCloseCursor", "SQLCancel", "SQLGetDiagRec", "SQLGetDiagField",
        ],
    },
    // ── SQL Keywords ──
    Category {
        name: "sql_keywords",
        mode: MatchMode::SqlString,
        tokens: &[
            "SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP",
            "FROM", "WHERE", "JOIN", "INNER", "LEFT", "RIGHT", "OUTER", "CROSS",
            "GROUP", "ORDER", "HAVING", "LIMIT", "OFFSET",
            "UNION", "EXCEPT", "DISTINCT", "EXISTS", "BETWEEN", "LIKE",
            "TABLE", "INDEX", "VIEW", "TRIGGER", "PROCEDURE", "FUNCTION",
            "PRIMARY", "FOREIGN", "CONSTRAINT", "DEFAULT", "UNIQUE", "CHECK",
            "BEGIN", "COMMIT", "ROLLBACK", "TRANSACTION", "SAVEPOINT",
            "INTEGER", "VARCHAR", "TEXT", "BLOB", "REAL", "BOOLEAN",
            "COUNT", "SUM", "AVG", "MIN", "MAX",
            "PRAGMA", "VACUUM", "EXPLAIN", "ANALYZE",
        ],
    },
    // ── C++ ORMs ──
    Category {
        name: "cpp_orms",
        mode: MatchMode::Ident,
        tokens: &[
            "insert_into", "remove_from", "left_outer_join", "right_outer_join",
            "cross_join", "inner_join", "dynamic_select", "dynamic_where",
            "rowset", "got_data", "backend_factory", "schema_catalog", "prepared_query",
        ],
    },
    // ── C++23/26 ──
    Category {
        name: "cpp23_types",
        mode: MatchMode::Ident,
        tokens: &[
            "expected", "unexpected", "move_only_function", "bind_back",
            "unreachable", "to_underlying", "byteswap", "invoke_r", "forward_like",
            "out_ptr", "inout_ptr", "start_lifetime_as",
            "flat_map", "flat_set", "flat_multimap", "flat_multiset",
            "mdspan", "generator", "stacktrace", "source_location",
            "is_scoped_enum", "is_implicit_lifetime",
        ],
    },
    Category {
        name: "cpp23_ranges",
        mode: MatchMode::Ident,
        tokens: &[
            "zip", "zip_transform", "adjacent", "chunk", "slide", "stride",
            "cartesian_product", "enumerate", "as_const", "as_rvalue",
        ],
    },
    Category {
        name: "cpp26_features",
        mode: MatchMode::Ident,
        tokens: &[
            "contract_assert", "define_class", "substitute", "reflect_value",
            "scheduler", "sender", "receiver", "run_loop", "static_thread_pool",
            "just", "let_value", "when_all", "transfer", "schedule",
            "start_detached", "sync_wait", "counting_scope",
            "fetch_max", "fetch_min", "is_within_lifetime", "text_encoding",
        ],
    },
    Category {
        name: "cpp20_concepts",
        mode: MatchMode::Ident,
        tokens: &[
            "same_as", "derived_from", "convertible_to", "integral", "floating_point",
            "signed_integral", "unsigned_integral", "destructible", "constructible_from",
            "movable", "copyable", "semiregular", "regular",
            "equality_comparable", "totally_ordered", "three_way_comparable",
            "invocable", "predicate",
            "input_iterator", "forward_iterator", "bidirectional_iterator",
            "random_access_iterator", "contiguous_iterator",
            "input_range", "forward_range", "bidirectional_range",
            "random_access_range", "contiguous_range", "sized_range",
        ],
    },
    // ── Testing/Build Frameworks ──
    Category {
        name: "gtest",
        mode: MatchMode::Ident,
        tokens: &[
            "TEST", "TEST_F", "TEST_P", "TYPED_TEST", "TYPED_TEST_SUITE",
            "EXPECT_EQ", "EXPECT_NE", "EXPECT_LT", "EXPECT_GT", "EXPECT_LE", "EXPECT_GE",
            "EXPECT_TRUE", "EXPECT_FALSE", "EXPECT_THAT",
            "ASSERT_EQ", "ASSERT_NE", "ASSERT_LT", "ASSERT_GT", "ASSERT_LE", "ASSERT_GE",
            "ASSERT_TRUE", "ASSERT_FALSE", "ASSERT_THAT",
            "EXPECT_THROW", "EXPECT_NO_THROW", "EXPECT_DEATH",
            "ASSERT_THROW", "ASSERT_NO_THROW", "ASSERT_DEATH",
            "MOCK_METHOD", "EXPECT_CALL", "ON_CALL",
            "SetUp", "TearDown", "SetUpTestSuite", "TearDownTestSuite",
        ],
    },
    Category {
        name: "cmake",
        mode: MatchMode::Ident,
        tokens: &[
            "cmake_minimum_required", "add_executable", "add_library",
            "target_link_libraries", "target_include_directories",
            "target_compile_definitions", "target_compile_options",
            "find_package", "find_library", "find_path", "find_program",
            "include_directories", "link_directories", "add_subdirectory",
            "configure_file", "install",
            "CMAKE_CXX_STANDARD", "CMAKE_CXX_FLAGS",
            "CMAKE_BUILD_TYPE", "CMAKE_INSTALL_PREFIX",
            "CMAKE_SOURCE_DIR", "CMAKE_BINARY_DIR",
        ],
    },
    Category {
        name: "catch_boost_test",
        mode: MatchMode::Ident,
        tokens: &[
            "BOOST_AUTO_TEST_CASE", "BOOST_CHECK", "BOOST_REQUIRE",
            "BOOST_TEST", "BOOST_FIXTURE_TEST_SUITE", "BOOST_AUTO_TEST_SUITE",
            "BOOST_CHECK_EQUAL", "BOOST_REQUIRE_EQUAL",
            "REQUIRE", "CHECK", "SECTION", "TEST_CASE",
            "SCENARIO", "GIVEN", "WHEN", "THEN",
        ],
    },
    // ── Compiler Attributes ──
    Category {
        name: "attributes",
        mode: MatchMode::Ident,
        tokens: &[
            "__attribute__", "__declspec", "__stdcall", "__cdecl", "__fastcall",
            "__forceinline", "__asm__",
            "nodiscard", "maybe_unused", "deprecated", "likely", "unlikely",
            "no_unique_address", "carries_dependency", "fallthrough",
            "__cplusplus", "__FILE__", "__LINE__", "__func__", "__FUNCTION__",
        ],
    },
];

// ── Morpheme Analysis ──

pub struct MorphemeCategory {
    pub name: &'static str,
    pub stems: &'static [&'static str],
}

pub static MORPHEMES: &[MorphemeCategory] = &[
    MorphemeCategory {
        name: "prefixes",
        stems: &[
            "pre", "post", "un", "re", "de", "dis", "non", "sub", "super",
            "over", "under", "inter", "intra", "multi", "poly", "mono",
            "bi", "tri", "semi", "pseudo", "meta", "proto", "anti", "co",
        ],
    },
    MorphemeCategory {
        name: "suffixes",
        stems: &[
            "able", "ible", "tion", "sion", "ment", "ness", "ful", "less",
            "ize", "ify", "ate", "ent", "ant", "ary", "ory",
            "er", "or", "ist", "ed", "ing", "ly", "al", "ic",
        ],
    },
    MorphemeCategory {
        name: "cpp_stems",
        stems: &[
            "alloc", "dealloc", "init", "deinit", "lock", "unlock",
            "push", "pop", "read", "write", "open", "close",
            "begin", "end", "start", "stop", "create", "destroy",
            "insert", "remove", "find", "search", "sort", "swap",
            "load", "save", "parse", "format", "encode", "decode",
        ],
    },
    MorphemeCategory {
        name: "common_components",
        stems: &[
            "buffer", "cache", "queue", "stack", "list", "tree", "node", "graph",
            "handler", "manager", "factory", "builder", "adapter", "wrapper",
            "proxy", "callback", "listener", "observer", "visitor", "iterator",
            "generator", "config", "context", "session", "request", "response",
            "message", "event", "value", "index", "count", "total", "offset",
            "length", "capacity", "impl", "ctx", "ptr", "buf", "cfg", "msg",
            "req", "res", "arg", "param", "iter", "prev", "curr", "next",
            "tmp", "src", "dst",
        ],
    },
];

// ── C/C++ Reserved Keywords ──
// Full list for keyword frequency analysis.

pub static CPP_KEYWORDS: &[&str] = &[
    // C89/C99
    "auto", "break", "case", "char", "const", "continue", "default", "do",
    "double", "else", "enum", "extern", "float", "for", "goto", "if",
    "inline", "int", "long", "register", "restrict", "return", "short",
    "signed", "sizeof", "static", "struct", "switch", "typedef", "union",
    "unsigned", "void", "volatile", "while",
    // C++
    "alignas", "alignof", "and", "and_eq", "asm", "bitand", "bitor",
    "bool", "catch", "char8_t", "char16_t", "char32_t", "class",
    "co_await", "co_return", "co_yield", "compl", "concept", "const_cast",
    "consteval", "constexpr", "constinit", "decltype", "delete",
    "dynamic_cast", "explicit", "export", "false", "friend",
    "mutable", "namespace", "new", "noexcept", "not", "not_eq",
    "nullptr", "operator", "or", "or_eq", "private", "protected", "public",
    "reinterpret_cast", "requires", "static_assert", "static_cast",
    "template", "this", "thread_local", "throw", "true", "try",
    "typeid", "typename", "using", "virtual", "wchar_t", "xor", "xor_eq",
    // C++20
    "char8_t", "concept", "consteval", "constinit", "co_await", "co_return",
    "co_yield", "requires",
    // C++23
    "if consteval",
    // Common preprocessor (counted as identifiers)
    "define", "include", "ifdef", "ifndef", "endif", "pragma", "elif",
];
