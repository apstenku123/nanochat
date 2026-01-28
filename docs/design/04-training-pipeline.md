# Training Pipeline Design

**Status**: Draft v1
**Related bd epic**: `cpp-training`
**Design doc**: `docs/design/04-training-pipeline.md`

---

## Training Stages Overview

```
Stage 1: Pretraining (150-170h)
  ├── Phase A: Short context (8k), NTP+FIM, bulk of data
  └── Phase B: Long context fine-tune (32k → 128k), YaRN
      ↓
Stage 2: SFT — Reasoning Fine-Tuning (20h)
  ├── Debug state → Reasoning → Fix
  └── Textbook Q&A in C++ format
      ↓
Stage 3: RLVR — Compiler Feedback RL (30h)
  ├── GRPO optimization
  └── Compiler + unit test rewards
```

Total: ~200 hours on 8x B200

---

## Stage 1: Pretraining

### Objective

**50% Next Token Prediction (NTP) + 50% Fill-In-the-Middle (FIM)**

NTP trains the model to write C++ code left-to-right.
FIM trains the model to insert code in the middle — critical for bug fixing and code editing.

### Hardware & Parallelism

| Setting         | Value                              |
| --------------- | ---------------------------------- |
| GPUs            | 8x NVIDIA B200                     |
| Parallelism     | FSDP (Fully Sharded Data Parallel) |
| Precision       | BF16                               |
| Flash Attention | v3                                 |
| Framework       | nanochat (adapted) or TorchTitan   |

### Hyperparameters

Based on nanochat's proven configuration, scaled for 3B:

```python
# Model
depth = 32
aspect_ratio = 80  # hidden_size = 32 * 80 = 2560
head_dim = 80      # 2560 / 32 heads = 80
max_seq_len = 8192

# Batch
device_batch_size = 16       # Per-GPU
total_batch_size = 2_097_152  # 2M tokens per step (8 GPUs × 16 × 8192 × grad_accum)
grad_accum_steps = 2          # 8 × 16 × 8192 × 2 = 2M

# Optimizers (from nanochat, scaled)
matrix_lr = 0.015        # Muon for weight matrices
embedding_lr = 0.2       # AdamW for embeddings
unembedding_lr = 0.003   # AdamW for lm_head
scalar_lr = 0.4          # AdamW for residual scalars
weight_decay = 0.1

# Schedule
warmup_ratio = 0.02      # 2% warmup
warmdown_ratio = 0.3     # 30% cooldown
final_lr_frac = 0.1      # Final LR = 10% of peak

# Training horizon
target_tokens = 300_000_000_000  # 300B tokens
num_iterations = target_tokens / total_batch_size  # ~143,000 steps
```

### Learning Rate Schedule

```
│  ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲
│ ╱                        ╲
│╱                          ╲
└──────────────────────────────
  2%          68%           30%
 warmup     constant     cooldown
```

### Evaluation During Training

| Metric               | Frequency        | Method                           |
| -------------------- | ---------------- | -------------------------------- |
| Training loss        | Every step       | EMA smoothed                     |
| Validation BPB       | Every 500 steps  | Hold-out C++ validation set      |
| Tokenizer efficiency | Once             | Compare tokens/byte vs baselines |
| Compilation rate     | Every 5000 steps | Sample 100 completions, try g++  |
| Throughput           | Every step       | tok/sec, MFU                     |

### Checkpointing

- Save every 5000 steps
- Keep last 3 checkpoints (disk space)
- Save dataloader state for exact resumption
- Save optimizer state per rank

### Phase B: Long Context Fine-Tuning (LCFT)

After main pretraining, extend context:

1. **32k context** (10h): Train on long files and multi-file contexts
2. **128k context** (10h): Train on repository-level contexts
3. Enable YaRN scaling during this phase
4. Reduce batch size to fit longer sequences in memory

---

## Stage 2: SFT (Supervised Fine-Tuning)

### Objective

Teach the model to generate `/* REASONING */` blocks and respond to structured inputs.

### Data Format

```jsonl
{
  "input": "// TASK: Fix null pointer dereference\nstruct DebugState {\n...\n};\nvoid func() {\n  int* p = nullptr;\n  *p = 5;\n}",
  "output": "/* REASONING:\n   Variable p is nullptr.\n   Dereferencing nullptr is undefined behavior.\n   Add null check.\n*/\nvoid func() {\n  int* p = nullptr;\n  if (!p) return;\n  *p = 5;\n}"
}
```

### SFT Dataset Sources

| Source            | Size         | Generation Method                                                |
| ----------------- | ------------ | ---------------------------------------------------------------- |
| Bug-fix pairs     | 5M examples  | Extract from GitHub commit diffs (message contains "fix", "bug") |
| Synthetic bugs    | 10M examples | Teacher model injects bugs into clean code                       |
| Debug trace + fix | 2M examples  | Run code under sanitizer, capture trace, pair with fix           |
| Textbook Q&A      | 3M examples  | Teacher model generates from Effective C++, algorithms           |
| Code review       | 2M examples  | Teacher model reviews code, suggests improvements                |

### Training Configuration

```python
# SFT-specific
learning_rate = 1e-5          # Much lower than pretrain
epochs = 3                    # Over SFT dataset
max_seq_len = 8192
mask_input = True             # Only compute loss on output tokens
```

### Masking Strategy

```
Input:  // TASK: Fix crash\nvoid func() {\n  int* p = nullptr;\n  *p = 5;\n}
Output: /* REASONING: ... */\nvoid func() { if (!p) return; *p = 5; }

Loss mask: [0 0 0 0 0 0 ... 0 | 1 1 1 1 1 1 1 ... 1]
           ^--- input (masked) ^--- output (trained)
```

---

## Stage 3: RLVR (Reinforcement Learning with Verifiable Rewards)

### Why RLVR for C++?

C++ has a perfect automated judge: **the compiler**.
Unlike natural language tasks where reward is subjective, C++ compilation is binary and deterministic.

### Algorithm: GRPO (Group Relative Policy Optimization)

From DeepSeek-R1 paper. Simpler than PPO, no value network needed.

```python
# Simplified GRPO loop
for batch in dataloader:
    prompt = batch["prompt"]  # Bug + debug state

    # Generate K completions per prompt
    completions = [model.generate(prompt) for _ in range(K)]

    # Score each completion
    rewards = [reward_function(c) for c in completions]

    # Compute group-relative advantage
    mean_reward = mean(rewards)
    std_reward = std(rewards)
    advantages = [(r - mean_reward) / std_reward for r in rewards]

    # Policy gradient update (only on positive advantage)
    loss = -sum(advantage * log_prob(completion) for ...)
    loss.backward()
    optimizer.step()
```

### Reward Function

```python
def reward_function(code: str, test_suite: list[str] = None) -> float:
    reward = 0.0

    # 1. Compilation check
    result = subprocess.run(
        ["g++", "-std=c++20", "-fsyntax-only", "-x", "c++", "-"],
        input=code.encode(), capture_output=True
    )
    if result.returncode == 0:
        reward += 1.0  # Compiles
    else:
        return -1.0  # Doesn't compile

    # 2. No warnings
    if not result.stderr:
        reward += 0.5

    # 3. Test execution (if tests provided)
    if test_suite:
        passed = run_tests(code, test_suite)
        reward += 5.0 * (passed / len(test_suite))

    # 4. Reasoning quality (length heuristic)
    if "/* REASONING:" in code:
        reward += 0.5  # Model explained its thinking

    return reward
```

### RLVR Configuration

```python
# GRPO settings
K = 8                    # Completions per prompt
temperature = 0.8        # Higher for diversity
learning_rate = 5e-7     # Very low for stability
kl_penalty = 0.01        # Prevent drift from SFT model
max_gen_length = 2048    # Max tokens to generate
```

---

## Compute Budget Breakdown

| Stage                           | Hours    | Tokens Processed      | GPU Utilization |
| ------------------------------- | -------- | --------------------- | --------------- |
| Pretrain Phase A (8k ctx)       | 140h     | 280B                  | ~50% MFU        |
| Pretrain Phase B (32k-128k ctx) | 20h      | 20B                   | ~40% MFU        |
| SFT                             | 15h      | 10B                   | ~45% MFU        |
| RLVR                            | 25h      | 5B (generation-heavy) | ~30% MFU        |
| **Total**                       | **200h** | **315B**              |                 |

---

## Monitoring & Experiment Tracking

### WandB Metrics

```python
wandb.log({
    "train/loss": loss,
    "train/tok_per_sec": throughput,
    "train/mfu": mfu,
    "val/bpb": validation_bpb,
    "val/compile_rate": compile_rate,  # New: % of samples that compile
    "val/fim_accuracy": fim_accuracy,  # New: FIM completion quality
    "lr/matrix": current_lr,
    "memory/peak_gb": peak_memory,
})
```

### Alerts

- Loss spike > 2x average → save checkpoint, investigate
- MFU drop below 30% → check GPU utilization
- Compile rate drop below 50% → check data quality

---

## Nanochat Adaptation Summary

### Files to Modify

| File                     | Changes                                                      |
| ------------------------ | ------------------------------------------------------------ |
| `scripts/base_train.py`  | Add FIM data pipeline, compile rate eval, context curriculum |
| `nanochat/gpt.py`        | Update config defaults, add YaRN, vocab size                 |
| `nanochat/dataloader.py` | Add C++ binary data loader with FIM support                  |
| `nanochat/tokenizer.py`  | Add CppTokenizer class                                       |
| `nanochat/kernels.py`    | No changes (CCE works as-is)                                 |

### New Files

| File                         | Purpose                                     |
| ---------------------------- | ------------------------------------------- |
| `scripts/cpp_sft.py`         | SFT training script                         |
| `scripts/cpp_rl.py`          | RLVR training script                        |
| `nanochat/cpp_normalizer.py` | Code normalization pipeline                 |
| `nanochat/cpp_evaluator.py`  | Compilation rate + test execution evaluator |
| `scripts/data/`              | Download and processing scripts             |
