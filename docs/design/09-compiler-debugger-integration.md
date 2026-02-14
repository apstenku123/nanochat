# 09 — Compiler & Debugger Integration for CppReason

How the model sees errors, talks to debuggers, and understands platforms.

---

## The Problem

CppReason speaks C++. It reasons in C++ comments, calls tools via C++ function
expressions, and outputs C++ code. But today it has gaps:

1. **Compiler errors** are fed as raw text blobs — no structured understanding
2. **Debugger interaction** is sketched (`__AGENT_QUERY__("trace", ...)`) but not trained
3. **Platform awareness** is implicit in training data, not explicit in the model's representation
4. **Error classification** doesn't exist — the model can't distinguish "missing semicolon" from "use-after-free" from "ABI mismatch"

This document proposes concrete solutions for each.

---

## 1. How the Model Should See Compiler Errors

### Current State

`tool_runtime.py` runs `g++ -fsyntax-only` and returns raw stderr as a string.
`cpp_verifier.py` captures compiler output for RLVR rewards (0.0 = compile fail).
Neither parses or structures the error.

### Problem

Raw compiler output mixes diagnostics, notes, and fixit suggestions in a single
stream. The model wastes capacity parsing `file.cpp:42:7: error: ...` format
strings instead of understanding the error.

### Proposal: Structured Diagnostic Format

Present compiler errors as C++ struct literals that the model already understands:

```cpp
// <COMPILE_START>
struct diagnostic {
    const char* file = "vector_sort.cpp";
    int line = 42;
    int col = 7;
    enum severity { error, warning, note } level = error;
    const char* code = "-Werror=return-type";    // clang/gcc flag
    const char* message = "non-void function does not return a value in all control paths";
    const char* source_line = "    if (v.empty()) { /* missing return */ }";
    const char* caret =      "    ^~~~~~~~~~~~~~";
    // related diagnostics
    struct note {
        int line = 38;
        const char* message = "control reaches end of non-void function";
    } notes[1];
};
// <COMPILE_END>
```

### Why This Works

- The model already understands C++ struct syntax — zero new token types needed
- Field names (`line`, `col`, `severity`, `message`) are self-documenting
- `source_line` + `caret` give spatial context without needing the full file
- `notes[]` array preserves diagnostic chains (template instantiation backtraces)
- Severity enum lets the model distinguish errors from warnings from notes

### Implementation

Replace raw stderr injection in `tool_runtime.py::tool_compile()` with a
parser that converts gcc/clang diagnostics into this struct format:

```python
def parse_diagnostic(stderr: str) -> str:
    """Convert gcc/clang stderr to C++ struct literal."""
    # gcc format: file:line:col: severity: message
    # clang format: file:line:col: severity: message [-Wflag]
    # Both emit source line + caret on subsequent lines
    ...
```

Use clang's `-fdiagnostics-format=json` (available since clang 18) when
available — it outputs structured JSON that maps directly to the struct fields.
Fall back to regex parsing for gcc.

### Training Data

Generate structured diagnostic pairs from the existing `diff_sft` dataset:
1. Take the "before" code (has bug)
2. Compile with `g++ -fdiagnostics-format=json` / clang
3. Format as struct literal
4. Pair with the "after" code (the fix)

This creates `(diagnostic_struct, fix_diff)` training pairs. Approximately
19K pairs from existing `diff_sft.jsonl` where the "before" code has
compilation errors.

---

## 2. How the Model Should Talk to Debuggers

### Current State

`__AGENT_QUERY__("trace", "HandleRequest")` exists in the vision doc but is
not implemented, not trained, and not wired to any debugger.

### Research Landscape (2025-2026)

Three protocols matter:

| Protocol                           | What It Does                                 | Who Uses It                   |
| ---------------------------------- | -------------------------------------------- | ----------------------------- |
| **DAP** (Debug Adapter Protocol)   | Standard debugger interface (VS Code)        | GDB, LLDB, Chrome DevTools    |
| **MCP** (Model Context Protocol)   | LLM tool calling standard                    | Claude, ChatDBG, mcp-debugger |
| **LSP** (Language Server Protocol) | Code intelligence (definitions, diagnostics) | clangd, rust-analyzer         |

Key systems:
- **ChatDBG** (UMass, FSE 2025): LLM executes GDB/LLDB commands directly
- **InspectCoder** (Alibaba, Oct 2025): LLM sets breakpoints, inspects state, iterates
- **Debug-gym** (Azure AI, Sep 2025): Textual pdb environment for LLM agents
- **LLDB MCP** (llvm.org): LLDB has native MCP support now
- **mcp-debugger** (github.com/debugmcp): MCP server wrapping DAP

### Proposal: Debugger as C++ Tool Calls

Keep the C++-native principle. The model requests debugger actions as C++
function calls, the runtime translates them to DAP commands, and results come
back as C++ struct literals.

**New tool functions** (additions to the existing 6 tools):

```cpp
// Set a breakpoint — returns breakpoint ID
int bp = breakpoint("vector_sort.cpp", 42);

// Set conditional breakpoint
int bp2 = breakpoint("allocator.cpp", 118, "size > 4096");

// Continue execution to next breakpoint
struct stop_info {
    const char* reason;   // "breakpoint", "signal", "exit"
    const char* file;
    int line;
    int signal;           // 0 if breakpoint, 11 for SIGSEGV, etc.
} stop = continue_exec();

// Step one source line
stop_info s = step();

// Inspect local variables at current frame
struct frame_vars {
    const char* function = "std::vector<int>::push_back";
    const char* file = "stl_vector.h";
    int line = 1198;
    struct var {
        const char* name;
        const char* type;
        const char* value;
    } locals[] = {
        {"this", "std::vector<int>*", "0x7ffd2340"},
        {"__x", "const int&", "42"},
        {"_M_impl", "std::_Vector_base::_Vector_impl", "{_M_start=0x55a812, _M_finish=0x55a830, _M_end=0x55a840}"},
    };
} vars = inspect();

// Evaluate expression in current frame
const char* result = eval("this->size()");  // returns "7"

// Get backtrace
struct frame {
    int depth;
    const char* function;
    const char* file;
    int line;
} backtrace[] = backtrace_get(/*max_depth=*/10);

// Read memory (for low-level debugging)
struct mem_dump {
    uint64_t address;
    int size;
    const char* hex;     // "48 8b 45 f8 48 89 c7..."
    const char* ascii;   // "H.E.H...."
} mem = read_memory(0x7ffd2340, 64);
```

**The `<DEBUG_CONTEXT>` token (ID 13)** should wrap ALL debugger output:

```
<DEBUG_CONTEXT>
struct stop_info stop = { .reason = "signal", .signal = 11, ... };
struct frame_vars vars = { .function = "HashMap::insert", ... };
<CODE_END>
```

### Implementation: DAP Backend

The tool runtime wraps a DAP client (talks to `lldb-dap` or `gdb --interpreter=dap`):

```python
# tool_runtime.py additions
class DebugSession:
    def __init__(self, executable: str, args: list[str]):
        self.dap = DAPClient("lldb-dap")  # or gdb-dap
        self.dap.launch(executable, args)

    def tool_breakpoint(self, file: str, line: int, cond: str = None) -> str:
        bp = self.dap.set_breakpoint(file, line, condition=cond)
        return f"int bp = {bp.id};"

    def tool_continue(self) -> str:
        event = self.dap.continue_and_wait()
        return format_as_cpp_struct("stop_info", event)

    def tool_inspect(self) -> str:
        frame = self.dap.get_top_frame()
        variables = self.dap.get_variables(frame.id)
        return format_as_cpp_struct("frame_vars", frame, variables)
```

### Training Data for Debugger Interaction

Three sources:

1. **Synthetic debugging traces** from existing C++ corpus:
   - Take a compilable function
   - Inject a known bug (null deref, off-by-one, use-after-free)
   - Compile with `-g`, run under GDB/LLDB script
   - Capture breakpoint/inspect/backtrace sequence
   - Pair: (buggy code + debug trace) → fix

2. **Crash dump training** from fuzzing:
   - Run functions through AddressSanitizer/UBSan
   - Capture sanitizer reports (structured format)
   - Format as C++ struct literals
   - Model learns to interpret crash signatures

3. **Real debugging sessions** scraped from GDB tutorials, StackOverflow debugging Q&As:
   - Convert GDB session transcripts to C++ struct format
   - Pair with the fix that resolved the issue

**Estimated volume**: ~50K synthetic debugging traces, ~5K sanitizer reports,
~10K converted GDB sessions.

---

## 3. How the Model Should Understand Platforms

### Current State

Platform awareness is implicit — the model sees `#ifdef _WIN32` in training
data but doesn't know it's generating code for a specific target.

### Proposal: Platform Context Header

Every generation request starts with a platform descriptor as a C++ comment
block. This is the model's "system prompt":

```cpp
// <BOS>
// platform: x86_64-linux-gnu
// compiler: g++ 13.2
// standard: c++20
// mode: user          // user | kernel | firmware
// os: linux 6.1
// arch: x86_64        // x86_64 | aarch64 | riscv64
// features: sse4.2 avx2 avx512f
// libs: boost/1.83 protobuf/4.25 fmt/10.2
// sanitizers: address,undefined
// build: cmake 3.28 / ninja 1.11
// debug: true         // compiled with -g
```

### Why C++ Comments (Not Struct)

- Comments are zero-cost tokens — the model already ignores/generates them
- System-prompt-like role without needing a separate input format
- Can be prepended to any code context without changing semantics
- Matches the model's "everything is a .cpp file" principle

### Training

Add platform headers to training data:
1. Extract platform info from build systems in the C++ corpus (CMakeLists.txt,
   configure.ac, meson.build → compiler/standard/features)
2. Tag each training document with its detected platform context
3. Prepend as comment block during tokenization

The model learns to condition its output on platform context:
- `standard: c++17` → uses `std::optional`, not `std::expected`
- `mode: kernel` → no exceptions, no dynamic allocation, `__attribute__((section))`
- `arch: aarch64` → NEON intrinsics instead of AVX
- `compiler: msvc 19.38` → `__declspec(dllexport)` instead of `__attribute__((visibility))`

### Error Feedback Loop

When the compiler returns an error, the platform context is already part of the
conversation. The model can cross-reference:

```cpp
// platform: x86_64-linux-gnu, compiler: g++ 13.2, standard: c++20
// <COMPILE_START>
diagnostic d = { .line = 42, .message = "'std::expected' is not a member of 'std'" };
// <COMPILE_END>
/* REASONING: g++ 13.2 with -std=c++20 doesn't have std::expected.
   Need c++23 or use std::variant<T, Error> instead. */
```

---

## 4. How the Model Should Classify and Reason About Errors

### Error Taxonomy

The model should learn to categorize errors into actionable classes. This is
expressed through the existing `<THOUGHT_START>` reasoning format:

```cpp
<THOUGHT_START>
/* ERROR_CLASS: type_mismatch
 * SEVERITY: error
 * ROOT_CAUSE: implicit conversion from 'const char*' to 'std::string_view'
 *             requires C++17 but platform specifies C++14
 * FIX_STRATEGY: explicit constructor call
 * CONFIDENCE: 0.95
 */
<THOUGHT_END>
```

Error classes the model should learn (training labels for RLVR):

| Class              | Examples                                 | Fix Pattern                 |
| ------------------ | ---------------------------------------- | --------------------------- |
| `syntax`           | Missing semicolon, unbalanced braces     | Local edit                  |
| `type_mismatch`    | Wrong argument type, bad conversion      | Cast or change type         |
| `undefined_symbol` | Missing include, typo in name            | Add include or fix spelling |
| `linker`           | Undefined reference, multiple definition | Link library or fix ODR     |
| `lifetime`         | Use-after-free, dangling reference       | Restructure ownership       |
| `concurrency`      | Data race, deadlock                      | Add synchronization         |
| `logic`            | Off-by-one, wrong comparison             | Fix algorithm               |
| `platform`         | ABI mismatch, missing API                | Conditional compilation     |
| `resource`         | Memory leak, fd leak                     | RAII wrapper                |
| `ub`               | Signed overflow, null deref, alignment   | Sanitizer-guided fix        |

### Training for Error Classification

1. **GSPO pairs** from `cpp_verifier.py`: Already compiles code and gets pass/fail.
   Add error classification labels to the reward signal.
2. **Sanitizer-augmented training**: Run compiled code through ASan/UBSan/TSan.
   Each sanitizer report maps to an error class.
3. **Static analysis integration**: Run `clang-tidy` on training code.
   Each diagnostic maps to a class + fix.

---

## 5. Thinking Tokens for Structured Reasoning

The v2 tokenizer proposal defines thinking tokens. Here's how they map to
the compiler/debugger workflow:

| Token            | When Model Emits It         | What Follows                                        |
| ---------------- | --------------------------- | --------------------------------------------------- |
| `<THINK_ERROR>`  | After seeing a diagnostic   | Error classification reasoning                      |
| `<THINK_FIX>`    | After classification        | Fix strategy selection                              |
| `<THINK_VERIFY>` | After generating fix        | Mental compilation check                            |
| `<THINK_TRACE>`  | After seeing debugger state | Causal chain reasoning                              |
| `<THINK_PLAN>`   | Before multi-step debugging | Debugging plan (which breakpoints, what to inspect) |

**Example multi-turn debugging session:**

```
// Input context
<DEBUG_CONTEXT>
struct stop_info stop = { .reason = "signal", .signal = 11,
    .file = "hashmap.cpp", .line = 89 };
struct frame_vars vars = { .function = "HashMap::insert",
    .locals = {
        {"this", "HashMap*", "0x55a812a0"},
        {"key", "const std::string&", "\"hello\""},
        {"bucket", "Bucket*", "0x0"},  // <-- NULL!
    }};
<CODE_END>

// Model generates:
<THINK_TRACE>
/* Signal 11 (SIGSEGV) at hashmap.cpp:89 in HashMap::insert.
 * bucket is NULL (0x0). This means find_bucket(key) returned nullptr.
 * Need to check: is the hash table initialized? Is capacity > 0?
 * ERROR_CLASS: lifetime (null pointer dereference)
 */
<THINK_END>

<QUERY_TOOL> eval("this->capacity_") <CODE_END>
<TOOL_RESULT> "0" <CODE_END>

<THINK_FIX>
/* capacity_ is 0 — the hash table was never initialized.
 * Root cause: default constructor doesn't call init().
 * Fix: add init() call in constructor, or lazy-init in insert().
 */
<THINK_END>

<CODE_START>
// Fix: lazy initialization in insert
void HashMap::insert(const std::string& key, int value) {
    if (capacity_ == 0) { resize(16); }  // <-- added
    Bucket* bucket = find_bucket(key);
    // ... rest unchanged
}
<CODE_END>
```

---

## 6. ChaiScript / Ch Scripting Integration

The v2 tokenizer proposal mentions ChaiScript (C++17 header-only scripting
engine) and Ch (C/C++ interpreter). These serve a specific purpose:

**The model can write executable expressions that the runtime evaluates.**

Current tool calls like `compile("...")` pass string literals. With ChaiScript:

```cpp
// Model writes ChaiScript that the runtime evaluates directly
<SCRIPT_START>
auto result = run_test("test_insert_empty", [] {
    HashMap m;
    m.insert("key", 42);
    assert(m.size() == 1);
    assert(m.get("key") == 42);
});
if (!result.passed) {
    auto vars = inspect_at(result.crash_file, result.crash_line);
    print(vars);
}
<SCRIPT_END>
```

This is more expressive than function-call tool syntax because it supports
control flow, lambdas, and composition — all in C++ syntax the model
already understands.

**Implementation**: ChaiScript runtime embedded in `tool_runtime.py` via
`pybind11` or subprocess calling a ChaiScript REPL binary.

---

## 7. Implementation Roadmap

### Phase 1: Structured Compiler Diagnostics (can start now)

1. Add `parse_diagnostic()` to `tool_runtime.py` — regex parser for gcc/clang output
2. Add `-fdiagnostics-format=json` path for clang
3. Generate training pairs from `diff_sft.jsonl` bug→fix pairs
4. Format diagnostics as C++ struct literals in `<COMPILE_START>...<COMPILE_END>`
5. Add error classification labels to GSPO reward

**Effort**: ~2 days code, ~1 day data generation
**Dependencies**: None — uses existing infrastructure

### Phase 2: Platform Context Headers (can start now)

1. Add platform detection to data pipeline (`tools/cpp_chunker`)
2. Extract compiler/standard/features from CMakeLists.txt in training corpus
3. Prepend platform comment blocks during tokenization
4. No model changes needed — just training data

**Effort**: ~3 days data pipeline, ~0 model changes
**Dependencies**: Access to CMakeLists.txt from training corpus projects

### Phase 3: Debugger Tool Integration (needs Phase 1)

1. Implement `DebugSession` class in `tool_runtime.py` wrapping DAP client
2. Add 6 new tool functions (breakpoint, continue, step, inspect, eval, backtrace)
3. Format debugger responses as C++ struct literals
4. Generate synthetic debugging traces for SFT
5. Wire `<DEBUG_CONTEXT>` token handling in `engine.py`

**Effort**: ~5 days code, ~3 days data generation
**Dependencies**: `lldb-dap` or `gdb` with DAP support installed

### Phase 4: Thinking Tokens (needs v2 tokenizer)

1. Implement v2 tokenizer with `<THINK_ERROR>`, `<THINK_FIX>`, `<THINK_TRACE>` etc.
2. Add thinking token emission patterns to SFT data
3. Train model to use thinking tokens for structured reasoning
4. Add thinking token masking/reward in RLVR

**Effort**: ~2 days tokenizer, ~5 days data, ~ongoing training
**Dependencies**: v2 tokenizer migration

### Phase 5: ChaiScript Runtime (needs Phase 3)

1. Embed ChaiScript in tool runtime
2. Add `<SCRIPT_START>/<SCRIPT_END>` token handling
3. Train on scripted debugging sessions
4. Enable compositional tool use via scripting

**Effort**: ~3 days runtime, ~5 days training data
**Dependencies**: ChaiScript library, v2 tokenizer

---

## 8. What the Model Does NOT Need to Know

- How the agent infrastructure works (Python, gRPC, GKE)
- How to launch processes or manage files
- Network protocols or API schemas
- The training pipeline itself

The model's world is: **C++ code in, C++ code out.** Everything else is the
runtime's responsibility. The model doesn't know that `compile("...")` runs
`g++ -fsyntax-only` — it just knows that `compile()` returns a diagnostic
struct, and `inspect()` returns a frame_vars struct.

---

## 9. Key Design Decisions

### Why C++ Struct Literals (Not JSON, Not XML, Not Special Tokens)

1. **Zero new syntax** — the model already parses C++ perfectly
2. **Type information** — `int line = 42` is self-documenting, `{"line": 42}` is not
3. **Nested structures** — arrays, nested structs, enums all work naturally
4. **Training efficiency** — reuses existing C++ token patterns, no domain shift
5. **Composability** — model can generate struct literals in its own output

### Why DAP (Not Raw GDB/LLDB Commands)

1. **Language-agnostic** — same protocol works for GDB, LLDB, Chrome DevTools
2. **Structured I/O** — JSON messages, not text parsing
3. **Stateful** — session with breakpoints, frames, variables
4. **Industry standard** — VS Code, JetBrains, Emacs all speak DAP
5. **MCP bridge** — DAP maps cleanly to MCP tool schemas if needed later

### Why Platform Headers in Comments (Not Struct, Not Config File)

1. **Zero-cost** — comments are already in every training document
2. **Composable** — prepend to any context without changing semantics
3. **Conditional** — model learns to condition output on platform fields
4. **Familiar** — developers write platform comments in real code

---

## 10. Open Questions for Discussion

1. **Error struct granularity**: Should template instantiation backtraces
   (which can be 50+ lines in C++) be fully expanded or summarized?

2. **Debugger session scope**: Should the model manage a persistent debug
   session across multiple turns, or should each `inspect()` call be
   stateless (snapshot mode)?

3. **Platform detection accuracy**: CMakeLists.txt parsing only covers
   ~60% of the corpus. Should we also parse configure.ac, meson.build,
   Makefile? Or just default to "generic x86_64-linux, c++17, gcc latest"?

4. **ChaiScript vs CLING**: CLING (CERN's C++ interpreter based on Clang)
   is more powerful but much heavier. ChaiScript is lightweight but only
   covers a subset of C++. Which fits the model's needs better?

5. **Training data volume**: Is 50K synthetic debugging traces enough to
   teach debugging, or do we need 500K+? How much diversity in bug types
   is needed?

6. **Thinking token granularity**: Are 6 thinking tokens enough, or does
   the model need finer-grained reasoning markers (e.g., `<THINK_HYPOTHESIS>`,
   `<THINK_ELIMINATE>`, `<THINK_CONFIRM>`)?
