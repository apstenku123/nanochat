# CppReason: C++ Specialist Reasoning LLM — Vision & Architecture

**Status**: Draft v1
**Related bd epic**: `cpp-data-pipeline`, `cpp-tokenizer`, `cpp-model-arch`, `cpp-training`, `cpp-agent`

---

## Mission

Build a **specialized reasoning LLM trained from scratch** that operates exclusively in the domain of C/C++ code. The model reads C++ code and debugger state, reasons about it, generates fixes, and requests missing information from an agent runtime when needed.

This is not a general-purpose chatbot that happens to know C++. It is a **domain-native model** whose tokenizer, training data, architecture, and I/O protocol are all optimized for C++ code understanding and generation.

---

## Key Differentiators vs General-Purpose LLMs

| Aspect             | General LLM (GPT-4, Llama)                | CppReason                                          |
| ------------------ | ----------------------------------------- | -------------------------------------------------- |
| Tokenizer          | Trained on English + all languages + code | Trained exclusively on C/C++ code                  |
| Vocab waste        | ~70% tokens unused for C++ tasks          | ~0% waste — every token is C++ relevant            |
| Context efficiency | 1 line of code = 10-20 tokens             | 1 line of code = 3-8 tokens                        |
| Input format       | Natural language chat                     | Structured C++ (comments = tasks, structs = state) |
| Output format      | Natural language + code blocks            | C++ code + `/* REASONING */` comments + tool calls |
| Debug awareness    | None (bolted on via prompting)            | Native — trained on debugger state from day 1      |
| Size               | 7B-70B+ needed for good C++               | 3B sufficient (domain density)                     |

---

## Architecture Summary

### Model Variants

**Primary: Dense 3B**
- 32 layers, d_model=2560, 32 heads, 8 KV heads (GQA 4:1)
- SwiGLU activation, RMSNorm, RoPE (theta=1M)
- ~3 billion parameters
- Ideal for: first training run, fast inference in agent loop

**Alternative: MoE 10-12B (2B active)**
- 8-16 fine-grained experts + 2 shared experts (DeepSeek-style)
- ~2B active parameters per token
- Same inference speed as Dense 2B, knowledge capacity of 12B
- Ideal for: production deployment with more domain knowledge

### Hardware Target

- **Training**: 8x NVIDIA B200 (Blackwell), 200 hours
- **Budget**: ~300-400B tokens through the model
- **Inference**: Single GPU (3B fits in 8GB VRAM in INT4)

---

## I/O Protocol: "Everything is a .cpp File"

The model communicates entirely in C++ syntax. No chat templates, no markdown, no `User:`/`Assistant:` turns.

### Input Format

```cpp
// ==========================================
// SYSTEM CONTEXT (Debugger Snapshot)
// ==========================================
struct ThreadState {
    string current_func = "DataProcessor::Parse";
    int last_error = 0xC0000005; // Access Violation
    void* ptr_val = nullptr;
};

// ==========================================
// USER REQUEST
// ==========================================
// TASK: Fix the crash. The 'ptr_val' seems to be null but we access it.

// ==========================================
// SOURCE CODE
// ==========================================
void DataProcessor::Parse(char* buffer) {
    Header* h = reinterpret_cast<Header*>(buffer);
    int id = h->id; // <--- CRASH HERE
```

### Output Format (Reasoning + Fix)

```cpp
/* REASONING:
   1. Debug state shows buffer (ptr_val) is nullptr.
   2. Casting nullptr to Header* is UB.
   3. Dereferencing h->id causes Access Violation.
   4. FIX: Add null check before cast.
*/
    if (!buffer) {
        LogError("Buffer is null");
        return;
    }
    Header* h = reinterpret_cast<Header*>(buffer);
    int id = h->id;
}
```

### Output Format (Tool Request)

```cpp
/* REASONING:
   I need to refactor ComputeX, but I don't know what MyType::DoSomething does.
   I need the definition to ensure thread safety.
*/
__AGENT_QUERY__("definition", "MyType");
```

### Q&A Format (for SFT training on textbooks)

```cpp
// QUESTION: Explain how std::move works and why it is efficient.

/* ANSWER:
   std::move converts an lvalue into an xvalue (rvalue reference),
   allowing resources to be stolen instead of copied.
*/
class BigBuffer {
    char* data;
public:
    BigBuffer(BigBuffer&& other) noexcept : data(other.data) {
        other.data = nullptr;
    }
};
```

---

## Training Stages

### Stage 1: Pretraining (Core Knowledge)

- **Objective**: Next Token Prediction (50%) + Fill-In-the-Middle (50%)
- **Data**: ~60-80B unique C++ tokens × 4 epochs = 240-320B tokens
- **Duration**: ~150h on 8xB200
- **Result**: Model understands C++ syntax, STL, idioms, patterns

### Stage 2: SFT (Reasoning)

- **Objective**: Learn to generate `/* REASONING */` blocks before code
- **Data**: Synthetic pairs (Bug + Debug State → Reasoning → Fix)
- **Duration**: ~20h on 8xB200
- **Result**: Model can analyze debug state and explain fixes

### Stage 3: RLVR (Compiler Feedback)

- **Objective**: Maximize compilation success and test passage
- **Method**: GRPO with compiler as reward model
- **Duration**: ~30h on 8xB200
- **Result**: Model avoids hallucinations that don't compile

---

## Agent Integration

The model operates within a finite state machine:

1. **State A**: Agent feeds model current code + debug state
2. **Generate**: Model produces tokens
3. **Check**: If model emits `__AGENT_QUERY__`:
   - Stop generation
   - Agent resolves query (via clangd, ctags, GDB)
   - Inject result into context
   - Resume generation from State A (updated)
4. **If model emits `<CODE_END>`**: Extract generated code

### Latent Code Indexing (CLaRa-style)

The model doubles as its own embedding engine:
- At `<INDEX>` token, extract hidden state vector (d=2560)
- Store in FAISS vector database
- Use same model's reasoning state for retrieval queries
- Single model = generator + retriever (GRIT architecture)

---

## Success Criteria

1. **Compilation rate**: >95% of generated code compiles with g++/clang++
2. **Bug fix accuracy**: >70% on synthetic bug-fix benchmark
3. **Tokenizer efficiency**: 2-3x fewer tokens per C++ file than Llama tokenizer
4. **Inference speed**: <100ms per token on single B200/H100
5. **Context utilization**: Can process 32k+ token files without quality degradation
