# CppOnline: C++ Code-Native Agent Plan

## Vision

A language model that lives entirely in C++ token space. It receives C++ code, outputs C++ code, thinks in C++ comments, and makes tool calls as C++ function expressions. No JSON, no markdown, no natural language prompts — pure C++.

## Architecture

```
User (C++ code/comments) → Nanochat 400M (C++ model)
    ↓ generates C++ with tool calls
    ↓ <QUERY_TOOL> search("query") <CODE_END>
    ↓
Tool Runtime (Python/ChaiScript)
    ↓ executes tool, returns result
    ↓ <TOOL_RESULT> // result code <CODE_END>
    ↓
Nanochat continues generating → final <CODE_START> ... <CODE_END>
    ↓
Optional: Gemma 270M (tool router for ask() calls)
```

## Base Model

- **Checkpoint**: `d16_400M_fim_cce_10b` (step 50000)
- **Architecture**: 16 layers, 1280 dim, 10 heads, seq_len=1024
- **Parameters**: 398M
- **Pretrained on**: 10B C++ tokens with 50% FIM augmentation
- **Val BPB**: 0.80
- **Tokenizer**: C++ hybrid tokenizer, 32768 vocab

## Token Protocol

All tokens already exist in the vocabulary — no retraining needed.

| ID | Token | Role |
|----|-------|------|
| 2 | `<BOS>` | Document start |
| 3 | `<EOS>` | Document end |
| 7 | `<CODE_START>` | Begin final code output |
| 8 | `<CODE_END>` | End any block (code, tool call, tool result) |
| 9 | `<THOUGHT_START>` | Begin chain-of-thought (C++ comments) |
| 10 | `<THOUGHT_END>` | End chain-of-thought |
| 11 | `<QUERY_TOOL>` | Signal: tool call expression follows |
| 19 | `<TOOL_RESULT>` | Signal: tool output follows (injected by runtime) |

### Example Sequence

```
<BOS>
// Fix the memory leak in Buffer::resize
<THOUGHT_START>
// Need to check the current resize implementation
// The issue is likely missing deallocation of old buffer
<THOUGHT_END>
<QUERY_TOOL> read_file("src/buffer.cpp", 42, 55) <CODE_END>
<TOOL_RESULT>
void Buffer::resize(size_t new_size) {
    data_ = new char[new_size];  // old data_ leaked
    size_ = new_size;
}
<CODE_END>
<THOUGHT_START>
// Found it — delete[] old data_ before new allocation
<THOUGHT_END>
<CODE_START>
void Buffer::resize(size_t new_size) {
    delete[] data_;
    data_ = new char[new_size];
    size_ = new_size;
}
<CODE_END>
<EOS>
```

### Loss Masking During Training

| Block | Mask | Why |
|-------|------|-----|
| Instruction (before first special token) | 0 | Given context |
| `<THOUGHT_START>...thoughts...<THOUGHT_END>` | 1 | Train reasoning |
| `<QUERY_TOOL>...call...<CODE_END>` | 1 | Train tool calling |
| `<TOOL_RESULT>...result...<CODE_END>` | 0 | Injected at inference |
| `<CODE_START>...code...<CODE_END>` | 1 | Train code output |

## Available Tools

```cpp
string search(string query);                       // search codebase
string ask(string prompt);                          // query Gemma 270M
string read_file(string path);                      // read source file
string read_file(string path, int start, int end);  // read line range
string compile(string code);                        // try g++ syntax check
string test(string test_name);                      // run a test
```

Tool calls are C++ function call expressions — valid C++ syntax that ChaiScript can execute directly.

## SFT Training Data

### Data Sources

| Source | Size | Examples | Status |
|--------|------|----------|--------|
| `docstring_pairs_full.jsonl` | 2.9 GB | 3.3M | Cleaned → 1.8M |
| `diff_sft.jsonl` | 139 MB | 60k | Already clean (99.8%) |
| `gspo_prompts.jsonl` | 388 KB | 272 | HumanEval C++ |

### SFT Data Mix (tool_call_sft.jsonl)

| Strategy | Source | Tool Used | Target |
|----------|--------|-----------|--------|
| A: Docstring → search + code | docstring_pairs_clean | `search()` | 100k |
| B: Diff → compile + fix | diff_sft | `compile()` | 50k |
| C: HumanEval → ask + solve | gspo_prompts | `ask()` | 2k |
| D: No-tool direct code | docstring + diff + FIM | none | 50k |
| **Total** | | | **~200k** |

### Why the Mix?

- **Tool examples (75%)**: Model learns WHEN and HOW to call tools
- **No-tool examples (25%)**: Model learns it can also solve things directly
- **FIM examples**: Maintains infilling capability from pretraining
- **Variety**: search/compile/ask tools so model learns different tool types

## Data Cleaning Applied

### Docstring Pairs (3.3M → 1.8M, 54% kept)

| Filter | Removed | % |
|--------|---------|---|
| Short body (<50 chars) | 587k | 17.6% |
| Duplicate (sig + body prefix) | 556k | 16.6% |
| Java-style docstrings | 228k | 6.8% |
| Code-as-docstring | 126k | 3.8% |
| Long body (>4000 chars) | 29k | 0.9% |

### Diff SFT (60k → ~59.9k)

- Removed 138 trivial renames (0.2%)
- Rest already clean from prepare_diff_sft.py quality filters

## Training Configuration

```bash
.venv/bin/python -m scripts.sft_train \
    --data data/tool_call_sft.jsonl \
    --checkpoint_path ~/.cache/nanochat/base_checkpoints/d16_400M_fim_cce_10b \
    --epochs 2 \
    --batch_size 8 \
    --lr 1e-4 \
    --kernel cce \
    --compile
```

- **Optimizer**: AdamW (simpler than Muon for fine-tuning)
- **LR**: 1e-4 (lower than pretraining, fine-tuning a strong base)
- **Epochs**: 2 (avoid overfitting on 200k examples)
- **Kernel**: CCE (Apple Cut Cross Entropy — fastest, lowest memory)

## Runtime Engine

### State Machine (in engine.py)

The generation loop detects special tokens and dispatches tool calls:

1. Model generates `<QUERY_TOOL>` → enter tool accumulation mode
2. Accumulate tokens until `<CODE_END>` → decode expression
3. Parse C++ function call: `func_name(arg1, arg2, ...)`
4. Dispatch to Python tool backend
5. Inject `<TOOL_RESULT>` + result tokens + `<CODE_END>` as forced tokens
6. Model continues generating (sees result, produces next thought or code)

### Tool Backends (tool_runtime.py)

| Tool | Backend |
|------|---------|
| `search(q)` | `rg` (ripgrep) over configured codebase directory |
| `ask(prompt)` | HTTP to local Gemma 270M inference server |
| `read_file(path)` | Python file I/O with line range |
| `compile(code)` | `g++ -fsyntax-only -x c++ -` via subprocess |
| `test(name)` | Configured test command via subprocess |

### ChaiScript Integration (Future)

ChaiScript is a header-only C++17 scripting engine. Future plan:
- Embed ChaiScript to evaluate arbitrary C++ expressions from model output
- Enables conditional tool calls, string manipulation, loops
- For now: Python-side regex parser handles 95% of cases

## File Manifest

| File | Status | Description |
|------|--------|-------------|
| `scripts/data/clean_docstring_pairs.py` | DONE | Data cleaning |
| `scripts/data/generate_tool_sft.py` | DONE | SFT data synthesis |
| `data/docstring_pairs_clean.jsonl` | DONE | 1.8M clean records |
| `data/tool_call_sft.jsonl` | GENERATING | ~200k training examples |
| `data/cpp_tokenizer/tokenizer.json` | DONE | RESERVED_19 → TOOL_RESULT |
| `nanochat/cpp_tokenizer.py` | DONE | Tool token properties |
| `nanochat/tool_sft_dataset.py` | TODO | Dataset class with auto-masking |
| `nanochat/tool_runtime.py` | TODO | Tool execution backends |
| `nanochat/engine.py` | TODO | C++ tool-call state machine |
| `scripts/sft_train.py` | TODO | Support ToolCallSFTDataset |
| `scripts/agent_cli.py` | TODO | Interactive agent CLI |

## Evaluation Plan

1. **HumanEval-X C++ pass@1**: 272 problems, with and without tools
2. **Tool usage quality**: Parse success rate, result relevance
3. **Comparison**: Tool-augmented vs plain SFT vs base model

## Key Decisions Made

1. **Pure C++ token space**: No JSON, no markdown — everything is C++ code/comments
2. **Existing special tokens**: No tokenizer retraining needed (RESERVED_19 → TOOL_RESULT)
3. **Loss masking on tool results**: Don't train on tool outputs (they're injected at inference)
4. **Mixed data**: 75% tool examples + 25% no-tool to teach when NOT to use tools
5. **Python tool backends first**: ChaiScript is future optimization
6. **Base model**: 10B-token checkpoint (strongest available)
7. **CCE kernel**: Apple Cut Cross Entropy for fastest training
