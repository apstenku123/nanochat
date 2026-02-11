# Engram + mHC Optional Integration Design for C++ Training

Status: Draft v1 (implementation-targeted)
Date: 2026-02-11
Owner: nanochat-qu0

## 1. Objective

Add two optional model capabilities to nanochat's Transformer stack for C++-focused pretraining and fine-tuning:

1. Engram-style static n-gram branches (long-context and code pattern retention).
2. mHC-style constrained multi-branch mixing (stable branch fusion beyond scalar residuals).

These capabilities must be independently togglable at runtime/training time, allowing four supported modes:

1. Baseline (existing model): `engram=off`, `mhc=off`
2. mHC-only: `engram=off`, `mhc=on`
3. Engram-only: `engram=on`, `mhc=off`
4. Combined: `engram=on`, `mhc=on`

Primary requirement from product/training side: preserve existing behavior by default and enable low-risk ablation testing for C++ data/training pipeline.

## 2. Non-goals

This integration explicitly does not include:

1. Full reproduction of all research-scale optimizations from both papers.
2. TPU-specific custom kernels for new components in first implementation.
3. New distributed-parallel strategies tied to the new modules.
4. End-to-end long-context curriculum redesign.

The first implementation targets correctness, optionality, checkpoint compatibility, and local/CI testability.

## 3. Current Architecture Baseline (nanochat)

Relevant current model behavior (from `nanochat/gpt.py`):

1. Decoder-only Transformer with FA3/SDPA fallback attention.
2. Pre-norm residual block:
   - `x = x + Attn(norm(x))`
   - `x = x + MLP(norm(x))`
3. Additional per-layer scalar path before each block:
   - `x = resid_lambdas[i] * x + x0_lambdas[i] * x0`
4. GQA support through separate query and kv head counts.
5. KV cache managed externally in `nanochat/engine.py`.

Implications:

1. Engram can be inserted as an extra branch in each selected block.
2. mHC can replace scalar-only mixing with constrained branch mixing.
3. Baseline fallback can remain zero-cost if new paths are disabled.

## 4. High-Level Integration Strategy

### 4.1 Stage Plan

Implement in two feature stages within one code delivery:

1. Stage A: Engram branch module + optional per-layer insertion.
2. Stage B: mHC mixer + branch routing for baseline and Engram outputs.

### 4.2 Composition Model

Define a normalized branch interface at block level:

1. Main branch: existing residual stream path.
2. Optional Engram branch: structured n-gram feature branch.
3. Optional identity/input branch: x0/skip-style branch.

mHC mixer (if enabled) receives branch stack and produces mixed representation under constraints.

When mHC is disabled, combine branches with simple additive/scalar gating for minimal behavior change.

## 5. Configuration and Optionality Contract

## 5.1 GPTConfig Additions

Add optional fields to `GPTConfig` (all default to baseline-safe values):

1. Engram toggles
   - `engram_enabled: bool = False`
   - `engram_layers: str = ""` (comma-separated indices; empty disables insertion)
   - `engram_ngram_orders: str = "2,3,4"`
   - `engram_bottleneck_dim: int = 0` (0 means auto = `n_embd // 4`)
   - `engram_dropout: float = 0.0`
2. mHC toggles
   - `mhc_enabled: bool = False`
   - `mhc_num_branches: int = 0` (auto from enabled branches)
   - `mhc_sinkhorn_iters: int = 5`
   - `mhc_temperature: float = 1.0`
   - `mhc_epsilon: float = 1e-6`
   - `mhc_blend_alpha: float = 1.0` (global strength)
3. Safety/behavior flags
   - `aux_loss_weight: float = 0.0` (regularization placeholder; disabled by default)

### 5.2 CLI Additions (`scripts/base_train.py`)

Expose all toggles in training entrypoint so all four combinations can be run from command line:

1. `--engram` (bool flag)
2. `--engram_layers`
3. `--engram_ngram_orders`
4. `--engram_bottleneck_dim`
5. `--engram_dropout`
6. `--mhc` (bool flag)
7. `--mhc_sinkhorn_iters`
8. `--mhc_temperature`
9. `--mhc_blend_alpha`

Derived behavior:

1. `--engram` false forces Engram path disabled regardless of layer list.
2. `--mhc` false skips all mHC compute even if params are set.
3. Combined mode requires explicit `--engram --mhc`.

## 6. Engram Design (Nanochat Adaptation)

### 6.1 Intent

Introduce a static local-pattern branch that captures reusable code n-gram structure (operators, idioms, repeated motifs), especially useful in C++ where syntax tokens and idioms recur with strong local dependencies.

### 6.2 Module Placement

Per selected layer, after attention+MLP update calculation but before final residual writeback, compute Engram branch from normalized hidden state.

Proposed insertion in block logic:

1. Start with `x_in`.
2. Compute baseline block update (`attn_update`, `mlp_update`) as currently.
3. If Engram enabled for layer, compute `engram_update`.
4. Merge via:
   - direct addition/gates when mHC disabled.
   - branch mixer when mHC enabled.

### 6.3 Engram Branch Shape/API

New module file: `nanochat/engram.py`.

Main class:

1. `EngramBranch(nn.Module)`
   - input: `x` shape `(B, T, C)`
   - output: branch update shape `(B, T, C)`

Internal structure (implementation-friendly approximation):

1. Bottleneck projection `C -> Bn`.
2. For each n-gram order k:
   - causal shifted local aggregation over last k positions.
   - learned projection for order-specific channel mixing.
3. Concatenate/sum multi-order outputs.
4. Expand projection `Bn -> C`.
5. Optional dropout.

This preserves causal generation safety and avoids introducing future-token leakage.

### 6.4 Complexity and Memory

1. Additional compute linear in selected layers and n-gram orders.
2. No full quadratic memory term beyond existing attention.
3. Works with current FA3/SDPA path because it is orthogonal to attention kernel.

## 7. mHC Design (Nanochat Adaptation)

### 7.1 Intent

Replace/augment scalar residual mixing with constrained branch mixing that is better conditioned than unconstrained learned linear combination.

### 7.2 Mixer API

New module file: `nanochat/mhc.py`.

Main class:

1. `ManifoldBranchMixer(nn.Module)`
   - input branches: tensor/list of branch activations `[(B,T,C), ...]`
   - output: mixed `(B,T,C)`

### 7.3 Constraint Mechanism

Use Sinkhorn normalization to map raw mix logits to approximately doubly-stochastic transport/mixing map.

Implementation-safe approach:

1. Learn branch score logits from pooled token features.
2. Build branch-by-branch raw matrix `M`.
3. Run Sinkhorn iterations with epsilon stabilization.
4. Use normalized weights to mix branch stack.

Rationale:

1. Stable constrained mixing.
2. Lightweight implementation using PyTorch ops.
3. Deterministic enough for tests.

### 7.4 Backward Compatibility with Existing Scalars

Current scalars:

1. `resid_lambdas`
2. `x0_lambdas`

Compatibility strategy:

1. Keep parameters and baseline code path intact when mHC is off.
2. When mHC is on, do not remove parameters from state dict immediately.
3. mHC path supersedes scalar mixing contribution by default, but legacy scalars remain loadable.

This avoids checkpoint breakage and allows gradual migration.

## 8. Combined Mode (Engram + mHC)

### 8.1 Branch Set

In combined mode per layer, branch candidates are:

1. Main updated stream (attention+MLP result).
2. Optional x0/identity branch.
3. Engram branch (if layer enabled).

mHC receives active branches and returns mixed output.

### 8.2 Fail-safe Behavior

1. If branch count < 2, bypass mHC and return branch 0.
2. If numerical instability detected in Sinkhorn (`nan`/`inf`), fallback to average mixing in forward pass (guard rail).
3. Log once per process when fallback triggers.

## 9. Checkpoint and Serialization Strategy

## 9.1 Model Config Patching

Update `nanochat/checkpoint_manager.py` patch helpers to set defaults for all new config keys if absent in old checkpoints.

## 9.2 State Dict Patching

When loading old checkpoints:

1. Missing Engram/mHC params are initialized with defaults.
2. Existing checkpoints load strict after patch expansion.

When loading new checkpoints in old code is out-of-scope (forward incompatibility).

## 10. Training Pipeline and C++ Focus

### 10.1 C++ Training Alignment

No tokenizer/data format change is required for first pass.

Why this still helps C++ training:

1. Engram branch should improve repetitive syntax and local motif retention.
2. mHC may stabilize branch cooperation and allow richer composition of syntax/semantics paths.

### 10.2 Recommended Early Hyperparameters

For first ablation runs:

1. Engram
   - `engram_layers="2,8,14"` (for depth 16 adjust accordingly)
   - `engram_ngram_orders="2,3,4"`
   - `engram_bottleneck_dim=n_embd/4`
   - `engram_dropout=0.0`
2. mHC
   - `mhc_sinkhorn_iters=5`
   - `mhc_temperature=1.0`
   - `mhc_blend_alpha=1.0`

## 11. Test Matrix Requirement (All Combinations)

Mandatory small-test matrix that must pass:

1. Baseline: `engram=0`, `mhc=0`
2. mHC-only: `engram=0`, `mhc=1`
3. Engram-only: `engram=1`, `mhc=0`
4. Combined: `engram=1`, `mhc=1`

Each mode must validate:

1. Model construction.
2. Forward pass shape correctness.
3. Backward pass finite gradients on tiny synthetic batch.
4. Optional checkpoint save/load roundtrip for new config fields.

## 12. Unit and Smoke Test Plan

### 12.1 New Unit Tests

Add `tests/test_engram_mhc_integration.py` with:

1. Parametrized mode test over 4 combinations.
2. Tiny config (e.g. depth 2, dim 64, seq 16, small vocab).
3. Assert finite loss and gradients.
4. Assert logits shape `(B, T, vocab)` and no `nan`.

### 12.2 Existing Tests Impact

Run targeted fast suite:

1. `tests/test_engine.py`
2. `tests/test_flash_attention.py` (if runtime allows)
3. New integration test.

## 13. Rollout Plan

Phase 1 (low risk):

1. Land design doc.
2. Land feature-gated implementation with defaults off.
3. Ensure baseline parity tests pass.

Phase 2 (ablation):

Run short CPP-focused training smoke across all 4 modes with same seed and tiny budget; compare:

1. train loss trend
2. val bpb
3. compile-rate sample metric (if evaluator run)

Phase 3 (tuning):

Adjust Engram layers/order and mHC sinkhorn params based on stability/quality.

## 14. Risks and Mitigations

1. Risk: unstable Sinkhorn normalization at small temperatures.
   - Mitigation: epsilon clamps, finite checks, safe fallback path.
2. Risk: hidden performance regression when features disabled.
   - Mitigation: explicit no-op branches and short-circuit checks.
3. Risk: checkpoint incompatibility.
   - Mitigation: config and parameter patching in checkpoint loader.
4. Risk: complexity creep in block forward.
   - Mitigation: isolate new logic in dedicated modules/files.

## 15. Acceptance Criteria

1. Defaults preserve baseline behavior (existing runs unaffected).
2. All four mode combinations available by parameters.
3. Small test matrix passes locally.
4. Design and implementation each committed and pushed.
5. Beads issue updated with final status and session sync completed.

## 16. Execution Checklist

1. Add doc and commit/push.
2. Implement `nanochat/engram.py`.
3. Implement `nanochat/mhc.py`.
4. Wire `nanochat/gpt.py` config, block logic, optimizer grouping as needed.
5. Wire `scripts/base_train.py` CLI and config pass-through.
6. Wire `nanochat/checkpoint_manager.py` defaults/patching.
7. Add tests for 4-mode matrix.
8. Run small tests, fix issues.
9. Commit/push implementation.
10. `bd close` + `bd sync` + final git push verification.

## 17. Source References

1. Engram repository: https://github.com/deepseek-ai/Engram
2. Engram paper PDF: https://github.com/deepseek-ai/Engram/blob/main/Engram_paper.pdf
3. mHC repository: https://github.com/tokenbender/mHC-manifold-constrained-hyper-connections
4. mHC paper PDF: https://arxiv.org/pdf/2512.24880
5. Secondary mHC preprint reference: https://arxiv.org/pdf/2601.07372
