# Agent Integration Design

**Status**: Draft v1
**Related bd epic**: `cpp-agent`
**Design doc**: `docs/design/05-agent-integration.md`

---

## Overview

The CppReason model operates as part of an agentic system. It is not a standalone text generator — it participates in a feedback loop with external tools (compiler, debugger, code search).

---

## Agent Architecture: ReAct Loop

```
┌──────────────────────────────────────────────┐
│                 AGENT RUNTIME                │
│                                              │
│  ┌─────────┐    ┌──────────┐    ┌────────┐   │
│  │  Model  │───▶│  Parser  │───▶│ Router │   │
│  │ (3B LLM)│    │          │    │        │   │
│  └─────────┘    └──────────┘    └────┬───┘   │
│       ▲                              │       │
│       │         ┌────────────────────┘       │
│       │         ▼                            │
│  ┌────┴────┐   ┌──────────────────────┐      │
│  │ Context │   │      TOOLS           │      │
│  │ Manager │◀──│ ┌──────┐ ┌────────┐  │      │
│  │         │   │ │clangd│ │  GDB   │  │      │
│  └─────────┘   │ │ctags │ │ LLDB   │  │      │
│                │ │FAISS │ │ g++    │  │      │
│                │ └──────┘ └────────┘  │      │
│                └──────────────────────┘      │
└──────────────────────────────────────────────┘
```

---

## Tool Protocol

### Tool Request Format

The model emits tool requests as C++ pseudo-function calls:

```cpp
__AGENT_QUERY__("definition", "MyType");
__AGENT_QUERY__("signature", "ProcessPacket");
__AGENT_QUERY__("trace", "HandleRequest");
__AGENT_QUERY__("memory", "0x7ffee1234");
__AGENT_QUERY__("file", "network/handler.h");
__AGENT_QUERY__("compile_check", "<code>");
__AGENT_QUERY__("test_run", "test_buffer_overflow");
```

### Tool Types

| Tool            | Request           | Response (injected into context)          |
| --------------- | ----------------- | ----------------------------------------- |
| `definition`    | Class/struct name | Full class definition                     |
| `signature`     | Function name     | Function declaration with params          |
| `trace`         | Function name     | Stack trace from recent execution         |
| `memory`        | Address           | Memory region dump (hex + interpretation) |
| `file`          | File path         | File contents                             |
| `compile_check` | Code string       | Compiler output (success or errors)       |
| `test_run`      | Test name         | Test output (pass/fail + stdout)          |
| `search`        | Query string      | Top-5 relevant code snippets (from FAISS) |

### Response Injection Format

Tool responses are injected as C++ comments/structs:

```cpp
// SYSTEM: Response to __AGENT_QUERY__("definition", "MyType"):
struct MyType {
    int value;
    std::string name;
    bool valid() const { return value >= 0; }
};
// END RESPONSE
```

---

## Generation Loop (Pseudocode)

```python
def agent_loop(model, prompt, tools, max_rounds=5):
    context = prompt

    for round in range(max_rounds):
        # Generate tokens
        output = model.generate(
            context,
            stop_tokens=["<EOS>", "<CODE_END>", "<QUERY_TOOL>"],
            max_tokens=4096
        )

        # Check if model requested a tool
        if output.stop_reason == "<QUERY_TOOL>":
            # Parse tool request
            tool_call = parse_agent_query(output.text)

            # Execute tool
            result = tools.execute(tool_call)

            # Inject result and continue
            context = context + output.text + format_tool_response(result)
            continue

        # Model finished generating
        return output.text

    return "MAX_ROUNDS_EXCEEDED"
```

---

## Latent Code Indexing (CLaRa-style)

### Concept

Use the model itself as the embedding engine. When processing code, extract the hidden state at the `<INDEX>` token as a dense vector representation.

### Offline Indexing ("Dreaming Phase")

```python
def index_codebase(model, codebase, vector_db):
    """Index every function in the codebase."""
    for func in codebase.iter_functions():
        # Prepare prompt
        prompt = f"<CODE_START>{func.body}<CODE_END><INDEX>"

        # Forward pass (no generation, just encoding)
        with torch.no_grad():
            hidden_states = model.forward_hidden(prompt)

        # Extract embedding from last layer, last token (<INDEX>)
        embedding = hidden_states[-1, -1, :]  # shape: (2560,)
        embedding = F.normalize(embedding, dim=0)

        # Store in vector DB
        vector_db.add(
            vector=embedding.cpu().numpy(),
            metadata={
                "file": func.file_path,
                "line": func.line_number,
                "name": func.name,
                "code": func.body,
            }
        )
```

### Online Retrieval

```python
def retrieve_relevant_code(model, thought_text, vector_db, top_k=5):
    """Find code relevant to the model's current reasoning."""
    # Encode the thought
    prompt = f"<THOUGHT_START>{thought_text}<THOUGHT_END><INDEX>"

    with torch.no_grad():
        hidden_states = model.forward_hidden(prompt)

    query_vector = hidden_states[-1, -1, :]
    query_vector = F.normalize(query_vector, dim=0)

    # Search FAISS index
    results = vector_db.search(query_vector.cpu().numpy(), top_k=top_k)

    return results
```

### Why This Works

Both the "thought about code" and "the code itself" are processed by the **same model** in the **same latent space**. The vector for "I need to find where packets are validated" will be geometrically close to the vector for `void HandlePacket(Packet* p) { if (!p->valid()) return; }` because the model learned these concepts together during pretraining.

---

## Contrastive Fine-Tuning

### Purpose

Align thought vectors with code vectors in the embedding space.

### Training Data

```jsonl
{"query": "// Find function that validates user authentication", "positive": "bool AuthManager::ValidateUser(const Credentials& c) { ... }", "negative": "void Logger::WriteEntry(const string& msg) { ... }"}
```

### Loss Function: InfoNCE

```python
def info_nce_loss(query_emb, positive_emb, negative_embs, temperature=0.05):
    """Contrastive loss for embedding alignment."""
    # Positive similarity
    pos_sim = F.cosine_similarity(query_emb, positive_emb, dim=-1) / temperature

    # Negative similarities (in-batch)
    neg_sims = F.cosine_similarity(
        query_emb.unsqueeze(1), negative_embs.unsqueeze(0), dim=-1
    ) / temperature

    # InfoNCE
    logits = torch.cat([pos_sim.unsqueeze(-1), neg_sims], dim=-1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long)

    return F.cross_entropy(logits, labels)
```

### Multi-Task Training

To preserve generation capability while learning embeddings:

```python
total_loss = generation_loss + lambda_contrastive * contrastive_loss
# lambda_contrastive = 1.0 (tunable)
```

---

## Vector Database

### FAISS Configuration

```python
import faiss

# Dimension matches model hidden size
d = 2560

# Use IVF with Product Quantization for large codebases
nlist = 4096  # Number of clusters
m = 64        # Number of sub-quantizers
bits = 8      # Bits per sub-quantizer

quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits)

# Train on sample of vectors
index.train(sample_vectors)

# Set search parameters
index.nprobe = 64  # Search 64 clusters (accuracy vs speed)
```

### Memory Requirements

| Codebase Size  | Functions | Raw Vectors | PQ Compressed |
| -------------- | --------- | ----------- | ------------- |
| 100K functions | 100K      | 1 GB        | 6.4 MB        |
| 1M functions   | 1M        | 10 GB       | 64 MB         |
| 10M functions  | 10M       | 100 GB      | 640 MB        |

PQ compression makes even 10M-function codebases searchable in RAM.

---

## Re-Indexing Strategy

When the model is updated (v1 → v2), all embeddings become invalid.

### Incremental Re-indexing

1. Track model version in vector DB metadata
2. On model update, mark all vectors as "stale"
3. Re-index in background (batch processing)
4. Old vectors still usable (approximate) until replaced
5. Priority: re-index most-accessed functions first

---

## End-to-End Example

**User request**: "There's a segfault in the packet processing module"

**Agent Loop**:

1. **Round 1**: Model generates reasoning:
   ```cpp
   /* REASONING: I need to find the packet processing code first. */
   __AGENT_QUERY__("search", "packet processing segfault");
   ```

2. **Tool executes**: FAISS returns top-5 relevant functions

3. **Round 2**: Model sees results, generates more reasoning:
   ```cpp
   /* REASONING: HandlePacket at network.cpp:42 looks relevant.
      I need to see the full function. */
   __AGENT_QUERY__("definition", "HandlePacket");
   ```

4. **Tool executes**: Returns full function body

5. **Round 3**: Model identifies bug and generates fix:
   ```cpp
   /* REASONING:
      HandlePacket dereferences p->header without null check.
      Debug trace shows p can be nullptr when connection drops.
      FIX: Add null guard at function entry.
   */
   <CODE_START>
   void HandlePacket(Packet* p) {
   if (!p) return;
   // ... rest of fixed code ...
   }
   <CODE_END>
   ```
