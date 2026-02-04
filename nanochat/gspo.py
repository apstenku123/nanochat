"""
Group Sequence Policy Optimization (GSPO) for C++ code generation.

GSPO is an improvement over GRPO that operates at the sequence level instead
of the token level, providing:
- Lower variance gradients
- More stable MoE training
- Better alignment with sequence-level rewards
- Simpler infrastructure (precision-tolerant)

Reference: https://arxiv.org/abs/2507.18071 (Qwen3 GSPO paper)
"""

import torch
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass

from nanochat.cpp_verifier import verify_cpp, compute_reward


@dataclass
class GSPOConfig:
    """Configuration for GSPO training.

    Key differences from GRPO:
    - epsilon/epsilon_high: asymmetric clipping (GSPO uses smaller values)
    - steps_per_generation: multiple gradient steps per rollout batch
    - importance_sampling_level: "sequence" (GSPO) vs "token" (GRPO)
    """
    group_size: int = 8              # Number of completions per prompt
    epsilon: float = 3e-4            # Left clipping range (much smaller than GRPO's 0.2)
    epsilon_high: float = 4e-4       # Right clipping range
    steps_per_generation: int = 4    # Gradient steps per rollout (minibatches)
    beta: float = 0.0                # KL regularization (0 = none in GSPO)
    max_gen_len: int = 512           # Maximum generation length
    temperature: float = 1.0         # Sampling temperature
    top_k: Optional[int] = None      # Top-k sampling (None = no filtering)
    importance_sampling_level: str = "sequence"  # "sequence" | "token"


class GSPOTrainer:
    """Group Sequence Policy Optimization trainer for code generation.

    Key algorithm difference from GRPO:
    - GRPO: importance ratio = exp(sum of per-token log ratios)
    - GSPO: importance ratio = exp(sequence-level log ratio / seq_len)

    This sequence-level formulation:
    1. Reduces variance by normalizing over sequence length
    2. Aligns with the sequence-level reward structure
    3. Stabilizes training for MoE models without routing replay

    Args:
        model: The policy model (trainable).
        ref_model: Frozen reference model for KL/importance computation.
        tokenizer: Tokenizer instance.
        config: GSPOConfig with hyperparameters.
    """

    def __init__(self, model, ref_model, tokenizer, config: Optional[GSPOConfig] = None):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config or GSPOConfig()
        self.device = model.get_device()

        # EOS token for stopping generation
        if hasattr(tokenizer, 'eos_token_id'):
            self.eos_id = tokenizer.eos_token_id
        else:
            self.eos_id = tokenizer.encode_special("<|assistant_end|>")
            if self.eos_id is None:
                self.eos_id = tokenizer.get_bos_token_id()

    @torch.inference_mode()
    def generate_group(self, prompt_ids: list[int]) -> list[list[int]]:
        """Generate group_size completions for a single prompt."""
        cfg = self.config
        completions = []
        for i in range(cfg.group_size):
            generated = list(prompt_ids)
            for token in self.model.generate(
                prompt_ids,
                max_tokens=cfg.max_gen_len,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                seed=42 + i,
            ):
                generated.append(token)
                if token == self.eos_id:
                    break
            completions.append(generated)
        return completions

    def compute_advantages(self, rewards: list[float]) -> list[float]:
        """Compute group-relative advantages: (r - mean) / std."""
        t = torch.tensor(rewards, dtype=torch.float32)
        mean = t.mean()
        std = t.std()
        if std < 1e-8:
            return [0.0] * len(rewards)
        advantages = ((t - mean) / std).tolist()
        return advantages

    def _compute_per_token_log_probs(
        self, model, sequences: list[list[int]], prompt_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-token log probabilities and mask for completions.

        Returns:
            log_probs: (B, max_completion_len) per-token log probs
            mask: (B, max_completion_len) validity mask
        """
        # Pad sequences to same length
        max_len = max(len(s) for s in sequences)
        batch_ids = torch.full(
            (len(sequences), max_len), 0, dtype=torch.long, device=self.device
        )
        for i, seq in enumerate(sequences):
            batch_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=self.device)

        # Forward pass
        logits = model(batch_ids)  # (B, T, V)
        log_probs_all = F.log_softmax(logits.float(), dim=-1)

        # Extract completion log-probs
        max_comp_len = max_len - prompt_len
        per_token_log_probs = torch.zeros(
            len(sequences), max_comp_len, device=self.device
        )
        mask = torch.zeros(len(sequences), max_comp_len, device=self.device)

        for i, seq in enumerate(sequences):
            seq_len = len(seq)
            comp_len = seq_len - prompt_len
            if comp_len <= 0:
                continue

            # Positions [prompt_len-1, seq_len-2] predict tokens [prompt_len, seq_len-1]
            start = prompt_len - 1
            end = seq_len - 1
            token_ids = batch_ids[i, prompt_len:seq_len]
            lp = log_probs_all[i, start:end, :]
            token_lp = lp.gather(1, token_ids.unsqueeze(1)).squeeze(1)

            per_token_log_probs[i, :comp_len] = token_lp
            mask[i, :comp_len] = 1.0

        return per_token_log_probs, mask

    def _compute_importance_weights(
        self,
        policy_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute importance sampling weights based on config.

        GSPO key insight: Use sequence-level ratios instead of token-level.

        Args:
            policy_log_probs: (B, T) per-token log probs from policy
            ref_log_probs: (B, T) per-token log probs from reference
            mask: (B, T) validity mask

        Returns:
            importance_weights: (B,) sequence-level weights for GSPO
        """
        log_ratio = policy_log_probs - ref_log_probs

        if self.config.importance_sampling_level == "sequence":
            # GSPO: sequence-level importance ratio with length normalization
            seq_lengths = mask.sum(-1).clamp(min=1)
            seq_log_ratio = (log_ratio * mask).sum(-1) / seq_lengths
            importance_weights = torch.exp(seq_log_ratio)
        else:
            # GRPO: token-level (sum of log ratios = product of ratios)
            importance_weights = torch.exp((log_ratio * mask).sum(-1))

        return importance_weights

    def _clip_importance_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """Apply asymmetric clipping to importance weights.

        GSPO uses much smaller clipping ranges than PPO/GRPO (3e-4 vs 0.2)
        because sequence-level ratios are already more stable.
        """
        cfg = self.config
        # Asymmetric clipping: [1 - epsilon, 1 + epsilon_high]
        return torch.clamp(weights, 1 - cfg.epsilon, 1 + cfg.epsilon_high)

    def step(self, prompts: list[list[int]], test_harnesses: list[str]) -> dict:
        """One GSPO training step over a batch of prompts.

        Algorithm:
        1. Generate group_size completions per prompt
        2. Verify and compute rewards
        3. Compute group-relative advantages
        4. Compute sequence-level importance weights
        5. Apply GSPO loss with asymmetric clipping

        Args:
            prompts: List of prompt token sequences.
            test_harnesses: List of test code strings (one per prompt).

        Returns:
            dict with metrics: loss, mean_reward, compile_rate, etc.
        """
        cfg = self.config
        all_losses = []
        all_rewards = []
        all_compile_ok = []
        all_kl = []

        self.model.eval()

        for prompt_ids, test_code in zip(prompts, test_harnesses):
            prompt_len = len(prompt_ids)

            # 1. Generate completions
            completions = self.generate_group(prompt_ids)

            # 2. Verify and compute rewards
            rewards = []
            for seq in completions:
                completion_ids = seq[prompt_len:]
                code = self.tokenizer.decode(completion_ids)
                result = verify_cpp(code, test_code=test_code)
                reward = compute_reward(result)
                rewards.append(reward)
                all_compile_ok.append(float(result["compile_ok"]))

            all_rewards.extend(rewards)

            # 3. Compute group-relative advantages
            advantages = self.compute_advantages(rewards)
            advantages_t = torch.tensor(
                advantages, dtype=torch.float32, device=self.device
            )

            # Skip if no gradient signal
            if advantages_t.abs().max() < 1e-8:
                continue

            # 4. Compute per-token log-probs
            self.model.train()
            policy_log_probs, mask = self._compute_per_token_log_probs(
                self.model, completions, prompt_len
            )

            with torch.no_grad():
                ref_log_probs, _ = self._compute_per_token_log_probs(
                    self.ref_model, completions, prompt_len
                )

            # 5. Compute sequence-level importance weights (GSPO key step)
            importance_weights = self._compute_importance_weights(
                policy_log_probs, ref_log_probs, mask
            )

            # 6. Compute KL for monitoring (optional regularization)
            log_ratio = policy_log_probs - ref_log_probs
            # Approximate KL: E[r - 1 - log(r)] where r = exp(log_ratio)
            with torch.no_grad():
                ratio = torch.exp(log_ratio)
                kl_per_token = (ratio - 1) - log_ratio
                kl = (kl_per_token * mask).sum() / mask.sum().clamp(min=1)
                all_kl.append(kl.item())

            # 7. GSPO loss with asymmetric clipping
            # Unlike PPO which clips both sides equally, GSPO uses different ranges
            clipped_weights = self._clip_importance_weights(importance_weights)

            # Surrogate loss: -E[min(r*A, clip(r)*A)]
            surr1 = importance_weights * advantages_t
            surr2 = clipped_weights * advantages_t
            policy_loss = -torch.min(surr1, surr2).mean()

            # Optional KL penalty (typically beta=0 in GSPO)
            if cfg.beta > 0:
                kl_loss = cfg.beta * kl
                loss = policy_loss + kl_loss
            else:
                loss = policy_loss

            all_losses.append(loss)
            self.model.eval()

        # Aggregate losses
        if len(all_losses) > 0:
            total_loss = torch.stack(all_losses).mean()
        else:
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        metrics = {
            "loss": total_loss,
            "mean_reward": sum(all_rewards) / max(len(all_rewards), 1),
            "compile_rate": sum(all_compile_ok) / max(len(all_compile_ok), 1),
            "kl_div": sum(all_kl) / max(len(all_kl), 1) if all_kl else 0.0,
            "num_completions": len(all_rewards),
        }
        return metrics


class GSPOTrainerAsync(GSPOTrainer):
    """GSPO trainer with async GKE sandbox verification.

    Use this when running verification in GKE Agent Sandbox instead of local g++.
    """

    def __init__(self, model, ref_model, tokenizer, sandbox, config=None):
        super().__init__(model, ref_model, tokenizer, config)
        self.sandbox = sandbox

    async def step_async(
        self, prompts: list[list[int]], test_harnesses: list[str]
    ) -> dict:
        """Async GSPO step with GKE sandbox verification."""
        cfg = self.config
        all_losses = []
        all_rewards = []
        all_compile_ok = []
        all_kl = []

        self.model.eval()

        for prompt_ids, test_code in zip(prompts, test_harnesses):
            prompt_len = len(prompt_ids)

            # 1. Generate completions
            completions = self.generate_group(prompt_ids)

            # 2. Decode completions
            codes = []
            for seq in completions:
                completion_ids = seq[prompt_len:]
                code = self.tokenizer.decode(completion_ids)
                codes.append(code)

            # 3. Batch verify in GKE sandbox (async)
            items = [(code, test_code) for code in codes]
            results = await self.sandbox.batch_verify(items)

            # 4. Compute rewards
            rewards = []
            for result in results:
                reward = compute_reward(result)
                rewards.append(reward)
                all_compile_ok.append(float(result.get("compile_ok", False)))

            all_rewards.extend(rewards)

            # Rest of the algorithm is same as sync version
            advantages = self.compute_advantages(rewards)
            advantages_t = torch.tensor(
                advantages, dtype=torch.float32, device=self.device
            )

            if advantages_t.abs().max() < 1e-8:
                continue

            self.model.train()
            policy_log_probs, mask = self._compute_per_token_log_probs(
                self.model, completions, prompt_len
            )

            with torch.no_grad():
                ref_log_probs, _ = self._compute_per_token_log_probs(
                    self.ref_model, completions, prompt_len
                )

            importance_weights = self._compute_importance_weights(
                policy_log_probs, ref_log_probs, mask
            )

            log_ratio = policy_log_probs - ref_log_probs
            with torch.no_grad():
                ratio = torch.exp(log_ratio)
                kl_per_token = (ratio - 1) - log_ratio
                kl = (kl_per_token * mask).sum() / mask.sum().clamp(min=1)
                all_kl.append(kl.item())

            clipped_weights = self._clip_importance_weights(importance_weights)
            surr1 = importance_weights * advantages_t
            surr2 = clipped_weights * advantages_t
            policy_loss = -torch.min(surr1, surr2).mean()

            if cfg.beta > 0:
                loss = policy_loss + cfg.beta * kl
            else:
                loss = policy_loss

            all_losses.append(loss)
            self.model.eval()

        if len(all_losses) > 0:
            total_loss = torch.stack(all_losses).mean()
        else:
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        return {
            "loss": total_loss,
            "mean_reward": sum(all_rewards) / max(len(all_rewards), 1),
            "compile_rate": sum(all_compile_ok) / max(len(all_compile_ok), 1),
            "kl_div": sum(all_kl) / max(len(all_kl), 1) if all_kl else 0.0,
            "num_completions": len(all_rewards),
        }
