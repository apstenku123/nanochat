# C++ Code Model Training Plan v3 - GSPO + Agent Architecture

## Overview

Three-stage training pipeline for C++ code generation with:
- **GSPO** (Group Sequence Policy Optimization) replacing GRPO
- **GKE Agent Sandbox** for secure code execution
- **FunctionGemma** (270M) as tool-use bridge
- **Contrastive learning** with good/bad/similar solutions
- **CMake specialist** model for build system generation

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Agent Orchestrator                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    FunctionGemma (270M)                          │   │
│  │         Tool Router + Natural Language ↔ Function Calls          │   │
│  └───────────────────────┬───────────────────────────────────────────┘   │
│                          │                                              │
│        ┌────────────────┼─────────────────┬────────────────────┐       │
│        ▼                ▼                 ▼                    ▼       │
│  ┌──────────┐    ┌───────────┐    ┌───────────────┐    ┌───────────┐  │
│  │ CodeGen  │    │ CMakeGen  │    │ GKE Sandbox   │    │ Analysis  │  │
│  │ (NanoChat│    │ (Specialist│    │ (Code Exec)   │    │ (Review)  │  │
│  │ 400M)    │    │ 100M)     │    │               │    │           │  │
│  └──────────┘    └───────────┘    └───────────────┘    └───────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Roles

| Component | Model Size | Purpose |
|-----------|------------|---------|
| **NanoChat** | 400M | C++ code generation (FIM, completion) |
| **CMakeGen** | 100M | CMakeLists.txt generation specialist |
| **FunctionGemma** | 270M | Tool routing, NL↔function translation |
| **GKE Sandbox** | - | Secure code execution environment |

---

## Stage 1: Base Pretraining (Enhanced)

### Data Mix
| Component | Ratio | Description |
|-----------|-------|-------------|
| Raw C++ | 40% | Standard next-token prediction |
| Random FIM | 40% | Random infilling (current) |
| **Structured FIM** | **20%** | Docstring → function body |

### Structured FIM Format
```
<FIM_PREFIX>
/* Check if two numbers are closer than threshold */
bool has_close_elements(vector<float> numbers, float threshold) {
<FIM_SUFFIX>
}
<FIM_MIDDLE>
    for (int i = 0; i < numbers.size(); i++)
        for (int j = i+1; j < numbers.size(); j++)
            if (abs(numbers[i] - numbers[j]) < threshold)
                return true;
    return false;
<EOT>
```

### Command
```bash
nohup .venv/bin/python -m scripts.base_train \
    --depth=16 \
    --num_iterations=50000 \
    --fim_rate=0.4 \
    --structured_fim_rate=0.2 \
    --kernel=cce \
    > train_400M_v3.log 2>&1 &
```

---

## Stage 2: SFT (Supervised Fine-Tuning)

### Data Sources
| Dataset | Size | Description |
|---------|------|-------------|
| `diff_sft.jsonl` | 60k | PR/MR code repairs |
| `docstring_sft.jsonl` | TBD | Docstring → implementation |
| `pass_k_sft.jsonl` | TBD | Multiple solution variations |

### Multi-Solution SFT Data Format

For each problem, we collect **multiple correct solutions** and train on all:

```json
{
  "instruction": "Implement has_close_elements...",
  "solutions": [
    {"code": "// Solution A - nested loops", "score": 1.0},
    {"code": "// Solution B - sorted approach", "score": 1.0},
    {"code": "// Solution C - hash map", "score": 1.0}
  ]
}
```

### Contrastive Preference Data Format

For GSPO, we need **preference pairs** (chosen, rejected):

```json
{
  "instruction": "Implement has_close_elements...",
  "chosen": "bool has_close_elements(...) { /* correct O(n²) */ }",
  "rejected": "bool has_close_elements(...) { /* wrong: off-by-one */ }",
  "similarity": 0.85
}
```

**Key insight from research**: Pairs with **large reward differences** and **high code similarity** provide the best training signal (Anchored Preference Optimization, ICLR 2025).

### Similarity-Based Contrastive Learning

From [Contrastive Preference Learning](https://github.com/jhejna/cpl):
- Train on pairs where solutions are **semantically similar but differ in correctness**
- This teaches the model **subtle distinctions** rather than obvious patterns
- Example: Two solutions both compile, one passes 4/5 tests, other passes 5/5

```python
# Generate contrastive pairs
def create_preference_pair(solutions: list[dict]) -> dict:
    """Create (chosen, rejected) pairs from multiple solutions."""
    # Sort by score
    sorted_sols = sorted(solutions, key=lambda x: x['score'], reverse=True)

    # Find pairs with high similarity but different scores
    for i, chosen in enumerate(sorted_sols[:-1]):
        for rejected in sorted_sols[i+1:]:
            sim = compute_code_similarity(chosen['code'], rejected['code'])
            if sim > 0.7 and chosen['score'] - rejected['score'] > 0.2:
                return {
                    "chosen": chosen['code'],
                    "rejected": rejected['code'],
                    "similarity": sim,
                    "margin": chosen['score'] - rejected['score']
                }
    return None
```

### SFT Command
```bash
.venv/bin/python -m scripts.sft_train \
    --data data/combined_sft.jsonl \
    --checkpoint_path ~/.cache/nanochat/base_checkpoints/d16_400M_v3 \
    --epochs 3 \
    --lr 2e-4 \
    --kernel cce
```

---

## Stage 3: GSPO (Group Sequence Policy Optimization)

### Why GSPO over GRPO

| Aspect | GRPO | GSPO |
|--------|------|------|
| **Importance ratio** | Token-level | Sequence-level |
| **Variance** | High (per-token) | Low (per-sequence) |
| **MoE stability** | Requires routing replay | Naturally stable |
| **Training efficiency** | Baseline | 1.5-2x faster |
| **Precision tolerance** | Sensitive | Tolerant (simpler infra) |

**References**:
- [GSPO Paper](https://arxiv.org/abs/2507.18071) (July 2025)
- [Qwen GSPO Blog](https://qwenlm.github.io/blog/gspo/)
- [Swift GSPO Docs](https://swift.readthedocs.io/en/latest/Instruction/GRPO/AdvancedResearch/GSPO.html)

### GSPO Loss Function

```python
# GRPO (token-level)
log_ratio = per_token_logps - old_per_token_logps
importance_weights = torch.exp(log_ratio)  # Token-level

# GSPO (sequence-level) - KEY CHANGE
seq_log_ratio = (log_ratio * mask).sum(-1) / mask.sum(-1)  # Normalize by length
importance_weights = torch.exp(seq_log_ratio.unsqueeze(-1))  # Broadcast to sequence
```

### GSPO Hyperparameters (from Qwen3 paper)

```python
@dataclass
class GSPOConfig:
    group_size: int = 8
    epsilon: float = 3e-4        # Left clipping range
    epsilon_high: float = 4e-4   # Right clipping range
    steps_per_generation: int = 4  # Minibatches per rollout
    beta: float = 0.0            # KL regularization (0 = none)
    importance_sampling_level: str = "sequence"  # "sequence" | "token"
```

### Multi-Pass Solution Collection (pass@k)

For each prompt, generate **k=32** solutions and categorize:

```python
def collect_pass_k_solutions(prompt: str, k: int = 32) -> dict:
    """Generate k solutions and categorize by correctness."""
    solutions = []
    for i in range(k):
        code = model.generate(prompt, seed=i)
        result = gke_sandbox.execute(code, test_cases)
        solutions.append({
            "code": code,
            "compile_ok": result["compile_ok"],
            "tests_passed": result["tests_passed"],
            "tests_total": result["tests_total"],
            "score": compute_reward(result),
        })

    # Categorize
    correct = [s for s in solutions if s["score"] == 1.0]
    partial = [s for s in solutions if 0 < s["score"] < 1.0]
    wrong = [s for s in solutions if s["score"] == 0.0]

    return {
        "prompt": prompt,
        "correct": correct,
        "partial": partial,
        "wrong": wrong,
        "pass_at_k": len(correct) / k,
    }
```

### GSPO Training Data Format

```json
{
  "instruction": "Implement has_close_elements with signature...",
  "test_harness": "#include<assert.h>\nint main() { ... }",
  "reference_solutions": [
    {"code": "...", "score": 1.0},
    {"code": "...", "score": 0.8}
  ],
  "negative_samples": [
    {"code": "...", "score": 0.0, "error": "compile_error"},
    {"code": "...", "score": 0.3, "error": "wrong_answer"}
  ]
}
```

### GSPO Command
```bash
.venv/bin/python -m scripts.gspo_train \
    --checkpoint_path ~/.cache/nanochat/sft_checkpoints/d16_400M_sft \
    --prompts data/gspo_prompts.jsonl \
    --group_size 8 \
    --epsilon 3e-4 \
    --epsilon_high 4e-4 \
    --steps_per_generation 4 \
    --num_iterations 500 \
    --lr 5e-5
```

---

## GKE Agent Sandbox Integration

### Why GKE Agent Sandbox

From [Google's announcement](https://cloud.google.com/blog/products/containers-kubernetes/gke-and-kubernetes-at-kubecon-2025):
- **gVisor isolation**: Each execution gets isolated kernel
- **Sub-second latency**: Pre-warmed sandbox pools
- **Pod Snapshots**: Checkpoint/restore for fast restarts
- **Zero-trust**: Designed for untrusted LLM-generated code

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    GKE Autopilot Cluster                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  Pre-warmed Sandbox Pool                  │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │   │
│  │  │Sandbox 1│ │Sandbox 2│ │Sandbox 3│ │Sandbox N│        │   │
│  │  │ gVisor  │ │ gVisor  │ │ gVisor  │ │ gVisor  │        │   │
│  │  │ +g++    │ │ +g++    │ │ +g++    │ │ +g++    │        │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │               Execution Controller Service                │   │
│  │  POST /execute {code, test_cases, timeout}               │   │
│  │  → Returns {compile_ok, run_ok, tests_passed, stdout}    │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Setup Commands

```bash
# 1. Create GKE Autopilot cluster (gVisor enabled by default)
gcloud container clusters create-auto nanochat-sandbox \
    --region=us-central1 \
    --release-channel=rapid

# 2. Deploy sandbox pool
kubectl apply -f k8s/sandbox-pool.yaml

# 3. Deploy execution controller
kubectl apply -f k8s/execution-controller.yaml
```

### Sandbox Pod Spec

```yaml
# k8s/sandbox-pod-template.yaml
apiVersion: v1
kind: Pod
metadata:
  name: cpp-sandbox
  labels:
    app: cpp-sandbox
spec:
  runtimeClassName: gvisor  # gVisor isolation
  containers:
  - name: compiler
    image: gcc:14
    resources:
      limits:
        cpu: "1"
        memory: "512Mi"
        ephemeral-storage: "1Gi"
      requests:
        cpu: "500m"
        memory: "256Mi"
    securityContext:
      runAsNonRoot: true
      runAsUser: 1000
      allowPrivilegeEscalation: false
      capabilities:
        drop: ["ALL"]
    volumeMounts:
    - name: workspace
      mountPath: /workspace
  volumes:
  - name: workspace
    emptyDir:
      sizeLimit: 100Mi
  # Strict timeouts
  activeDeadlineSeconds: 30
```

### Python Client

```python
# nanochat/gke_sandbox.py
import httpx
from dataclasses import dataclass

@dataclass
class GKESandboxConfig:
    endpoint: str = "http://execution-controller.default.svc.cluster.local"
    timeout: float = 10.0
    max_retries: int = 3

class GKESandbox:
    """Remote C++ verification via GKE Agent Sandbox."""

    def __init__(self, config: GKESandboxConfig = None):
        self.config = config or GKESandboxConfig()
        self.client = httpx.AsyncClient(timeout=self.config.timeout)

    async def verify(self, code: str, test_code: str = "") -> dict:
        """Execute C++ code in isolated sandbox."""
        payload = {
            "code": code,
            "test_code": test_code,
            "std": "c++20",
            "timeout": self.config.timeout,
        }

        resp = await self.client.post(
            f"{self.config.endpoint}/execute",
            json=payload,
        )
        return resp.json()

    async def batch_verify(self, items: list[tuple[str, str]]) -> list[dict]:
        """Verify multiple code samples concurrently."""
        import asyncio
        tasks = [self.verify(code, test) for code, test in items]
        return await asyncio.gather(*tasks)
```

---

## FunctionGemma Integration

### Role in Agent System

[FunctionGemma](https://blog.google/technology/developers/functiongemma/) is Google's 270M parameter model for function calling. We use it as:

1. **Tool Router**: Maps user intent to the right tool (CodeGen, CMakeGen, Sandbox)
2. **NL↔Function Bridge**: Translates between natural language and structured calls
3. **Result Summarizer**: Converts tool outputs back to user-friendly text

### Function Definitions

```json
{
  "functions": [
    {
      "name": "generate_code",
      "description": "Generate C++ code from a specification",
      "parameters": {
        "type": "object",
        "properties": {
          "specification": {"type": "string", "description": "What the code should do"},
          "signature": {"type": "string", "description": "Function signature if known"},
          "num_samples": {"type": "integer", "default": 1}
        },
        "required": ["specification"]
      }
    },
    {
      "name": "generate_cmake",
      "description": "Generate CMakeLists.txt for a C++ project",
      "parameters": {
        "type": "object",
        "properties": {
          "project_name": {"type": "string"},
          "source_files": {"type": "array", "items": {"type": "string"}},
          "dependencies": {"type": "array", "items": {"type": "string"}},
          "cpp_standard": {"type": "string", "default": "20"}
        },
        "required": ["project_name", "source_files"]
      }
    },
    {
      "name": "execute_code",
      "description": "Compile and run C++ code in sandbox",
      "parameters": {
        "type": "object",
        "properties": {
          "code": {"type": "string"},
          "test_cases": {"type": "string"},
          "timeout": {"type": "number", "default": 10}
        },
        "required": ["code"]
      }
    },
    {
      "name": "analyze_code",
      "description": "Analyze C++ code for issues or improvements",
      "parameters": {
        "type": "object",
        "properties": {
          "code": {"type": "string"},
          "analysis_type": {"type": "string", "enum": ["bugs", "performance", "style"]}
        },
        "required": ["code", "analysis_type"]
      }
    }
  ]
}
```

### Agent Orchestration Flow

```python
# nanochat/agent.py
from functiongemma import FunctionGemma
from nanochat.gpt import GPT
from nanochat.cmake_model import CMakeModel
from nanochat.gke_sandbox import GKESandbox

class NanoChatAgent:
    """Multi-model agent orchestrated by FunctionGemma."""

    def __init__(self):
        self.router = FunctionGemma.from_pretrained("google/functiongemma-270m-it")
        self.code_model = GPT.from_checkpoint("~/.cache/nanochat/gspo_checkpoints/d16")
        self.cmake_model = CMakeModel.from_checkpoint("~/.cache/nanochat/cmake/d8")
        self.sandbox = GKESandbox()

    async def process(self, user_message: str) -> str:
        """Route user request through appropriate tools."""

        # 1. FunctionGemma decides which function to call
        function_call = self.router.generate_function_call(
            user_message,
            available_functions=self.FUNCTION_DEFS,
        )

        # 2. Execute the function
        if function_call.name == "generate_code":
            code = self.code_model.generate(function_call.args["specification"])
            result = {"code": code}

        elif function_call.name == "generate_cmake":
            cmake = self.cmake_model.generate(function_call.args)
            result = {"cmake": cmake}

        elif function_call.name == "execute_code":
            exec_result = await self.sandbox.verify(
                function_call.args["code"],
                function_call.args.get("test_cases", "")
            )
            result = exec_result

        elif function_call.name == "analyze_code":
            analysis = self.analyze(function_call.args["code"])
            result = {"analysis": analysis}

        # 3. FunctionGemma summarizes result for user
        response = self.router.generate_response(
            user_message=user_message,
            function_result=result,
        )

        return response
```

### Multi-Turn Iteration Loop

```python
async def iterative_solve(self, problem: str, max_iterations: int = 5) -> str:
    """Iteratively generate, test, and refine code."""

    history = []
    current_code = None

    for i in range(max_iterations):
        # Generate or refine
        if current_code is None:
            current_code = self.code_model.generate(problem)
        else:
            # Use previous errors to refine
            feedback = history[-1]["feedback"]
            current_code = self.code_model.generate(
                f"{problem}\n\nPrevious attempt had errors:\n{feedback}\n\nFixed code:"
            )

        # Test in sandbox
        result = await self.sandbox.verify(current_code, test_cases)

        history.append({
            "iteration": i,
            "code": current_code,
            "result": result,
            "feedback": result.get("stderr", ""),
        })

        # Success?
        if result["run_ok"] and result["tests_passed"] == result["tests_total"]:
            return current_code

    # Return best attempt
    best = max(history, key=lambda x: x["result"].get("tests_passed", 0))
    return best["code"]
```

---

## CMake Specialist Model

### Why a Separate Model

CMake has unique syntax and patterns that differ from C++:
- Domain-specific keywords (`target_link_libraries`, `find_package`)
- Dependency graph understanding
- Platform/compiler detection logic
- Best practice patterns (modern CMake style)

A small specialized model (100M params) can outperform a larger general model.

### Training Data Sources

1. **GitHub CMake files**: Extract from C++ projects
2. **vcpkg port files**: High-quality dependency management
3. **Conan recipes**: Alternative package manager
4. **BUILD-BENCH dataset**: OSS build configurations

### Data Format

```json
{
  "instruction": "Create CMakeLists.txt for project 'mylib' with sources: main.cpp, utils.cpp. Dependencies: fmt, spdlog",
  "response": "cmake_minimum_required(VERSION 3.20)\nproject(mylib)\n\nfind_package(fmt REQUIRED)\nfind_package(spdlog REQUIRED)\n\nadd_executable(mylib main.cpp utils.cpp)\ntarget_link_libraries(mylib PRIVATE fmt::fmt spdlog::spdlog)\n"
}
```

### Training Command

```bash
.venv/bin/python -m scripts.cmake_train \
    --data data/cmake_sft.jsonl \
    --depth=8 \
    --num_iterations=10000 \
    --lr 3e-4
```

---

## Data Preparation Pipeline

### 1. Extract Structured FIM Data

```bash
python -m scripts.data.extract_docstring_pairs \
    --input data/cpp_combined_10b_v3.jsonl \
    --output data/docstring_pairs.jsonl
```

### 2. Generate Multi-Solution Data (pass@k)

```bash
python -m scripts.data.generate_pass_k \
    --prompts data/humaneval_cpp.jsonl \
    --model ~/.cache/nanochat/base_checkpoints/d16 \
    --k 32 \
    --output data/pass_k_solutions.jsonl \
    --use-gke-sandbox  # For real execution
```

### 3. Create Contrastive Preference Pairs

```bash
python -m scripts.data.create_preference_pairs \
    --input data/pass_k_solutions.jsonl \
    --output data/preference_pairs.jsonl \
    --min-similarity 0.7 \
    --min-margin 0.2
```

### 4. Prepare GSPO Prompts

```bash
python -m scripts.data.prepare_gspo_prompts \
    --humaneval data/eval/humaneval_cpp.jsonl \
    --custom data/custom_problems.jsonl \
    --output data/gspo_prompts.jsonl
```

### 5. Extract CMake Training Data

```bash
python -m scripts.data.extract_cmake \
    --repos data/cpp_raw/ \
    --output data/cmake_sft.jsonl
```

---

## Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `nanochat/gspo.py` | GSPO trainer (replaces grpo.py) |
| `nanochat/gke_sandbox.py` | GKE sandbox client |
| `nanochat/agent.py` | FunctionGemma orchestration |
| `nanochat/cmake_model.py` | CMake specialist wrapper |
| `scripts/gspo_train.py` | GSPO training script |
| `scripts/cmake_train.py` | CMake model training |
| `scripts/data/generate_pass_k.py` | Multi-solution generation |
| `scripts/data/create_preference_pairs.py` | Contrastive pair creation |
| `scripts/data/extract_cmake.py` | CMake data extraction |
| `k8s/sandbox-pool.yaml` | GKE sandbox deployment |
| `k8s/execution-controller.yaml` | Execution service |

### Modified Files

| File | Changes |
|------|---------|
| `nanochat/fim.py` | Add `apply_fim_structured()` |
| `nanochat/dataloader.py` | Add `structured_fim_rate` |
| `scripts/base_train.py` | Add `--structured_fim_rate` CLI |
| `scripts/rl_train.py` | Update to use GSPO |

---

## Training Timeline

```
Week 1-2: Data Preparation
├── Extract structured FIM data
├── Generate pass@k solutions (local g++)
├── Create contrastive preference pairs
└── Extract CMake training data

Week 3: Infrastructure
├── Set up GKE Autopilot cluster
├── Deploy sandbox pool
├── Implement execution controller
└── Test GKE sandbox integration

Week 4-5: Base Training
├── Train with enhanced FIM mix
├── Evaluate with CodeBLEU
└── Checkpoint best model

Week 6-7: SFT
├── Train on combined SFT data
├── Include multi-solution data
├── Evaluate pass@1 improvement

Week 8-10: GSPO
├── Implement GSPO trainer
├── Run GSPO with GKE sandbox rewards
├── Iterate with pass@k collection
└── Final evaluation

Week 11-12: Agent Integration
├── Fine-tune FunctionGemma for our tools
├── Train CMake specialist
├── Build agent orchestration
└── End-to-end testing
```

---

## Evaluation Checkpoints

### After Stage 1 (Base)
```bash
python -m scripts.cpp_eval_fim --model-tag d16_400M_v3 --checkpoint <step>
# Target: CodeBLEU > 0.3, pass@1 > 0.05
```

### After Stage 2 (SFT)
```bash
python -m scripts.cpp_eval_fim --model-tag d16_400M_sft --checkpoint <step>
# Target: CodeBLEU > 0.4, pass@1 > 0.15
```

### After Stage 3 (GSPO)
```bash
python -m scripts.cpp_eval_fim --model-tag d16_400M_gspo --checkpoint <step>
# Target: CodeBLEU > 0.5, pass@1 > 0.25
```

### Agent System
```bash
python -m scripts.agent_eval --problems data/eval/agent_problems.jsonl
# Target: End-to-end success rate > 0.4
```

---

## References

### GSPO
- [GSPO Paper (arXiv:2507.18071)](https://arxiv.org/abs/2507.18071)
- [Qwen GSPO Blog](https://qwenlm.github.io/blog/gspo/)
- [Swift GSPO Implementation](https://swift.readthedocs.io/en/latest/Instruction/GRPO/AdvancedResearch/GSPO.html)
- [TRL GSPO Issue](https://github.com/huggingface/trl/issues/3778)

### Contrastive Learning
- [Contrastive Preference Learning (CPL)](https://github.com/jhejna/cpl)
- [Anchored Preference Optimization (APO)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00748/130712)
- [SeRA: Better DPO Training](https://www.amazon.science/blog/a-better-training-method-for-reinforcement-learning-with-human-feedback)

### Pass@k Training
- [Top Pass Paper](https://arxiv.org/html/2408.05715v1)
- [Pass@k Policy Optimization](https://www.emergentmind.com/topics/pass-k-training)

### GKE Agent Sandbox
- [GKE Agent Sandbox Docs](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/agent-sandbox)
- [Agent Sandbox Blog](https://cloud.google.com/blog/products/containers-kubernetes/agentic-ai-on-kubernetes-and-gke)
- [GKE Sandbox Concepts](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/sandbox-pods)

### FunctionGemma
- [FunctionGemma Blog](https://blog.google/technology/developers/functiongemma/)
- [HuggingFace Model](https://huggingface.co/google/functiongemma-270m-it)
- [Multi-Agent Router Example](https://dev.to/saikumaryava/beyond-mobile-actions-exploring-functiongemma-for-intelligent-multi-agent-orchestration-4jlf)

### CMake/Build Systems
- [BUILD-BENCH Paper](https://www.arxiv.org/pdf/2509.25248)
- [CXXCrafter](https://dl.acm.org/doi/pdf/10.1145/3729386)
