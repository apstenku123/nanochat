"""
Group Relative Policy Optimization (GRPO) for C++ code generation.

Implements the GRPO algorithm from DeepSeek-R1:
1. Generate a group of completions for each prompt
2. Verify each completion (compile + test)
3. Compute group-relative advantages
4. Update policy with clipped surrogate objective + KL penalty

Reference: https://arxiv.org/abs/2402.03300 (DeepSeekMath)
"""

import torch
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass, field

from nanochat.cpp_verifier import verify_cpp, compute_reward


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    group_size: int = 8        # Number of completions per prompt
    kl_coeff: float = 0.1     # KL divergence penalty coefficient
    clip_eps: float = 0.2     # PPO clipping epsilon
    max_gen_len: int = 512    # Maximum generation length
    temperature: float = 1.0  # Sampling temperature for generation
    top_k: Optional[int] = None  # Top-k sampling (None = no filtering)


class GRPOTrainer:
    """Group Relative Policy Optimization trainer for code generation.

    The training loop:
    1. For each prompt, generate `group_size` completions using the policy model
    2. Verify each completion with g++ (compile + optional test execution)
    3. Compute group-relative advantages: (reward - mean) / std
    4. Compute the GRPO loss with PPO-style clipping and KL penalty

    Args:
        model: The policy model (trainable).
        ref_model: Frozen reference model for KL computation.
        tokenizer: Tokenizer instance (CppTokenizer or RustBPETokenizer).
        config: GRPOConfig with hyperparameters.
    """

    def __init__(self, model, ref_model, tokenizer, config: Optional[GRPOConfig] = None):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config or GRPOConfig()
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
        """Generate group_size completions for a single prompt.

        Uses the model's built-in generate method with temperature sampling.
        Each completion is generated independently.

        Args:
            prompt_ids: Token IDs for the prompt.

        Returns:
            List of group_size token sequences (prompt + completion).
        """
        cfg = self.config
        completions = []
        for i in range(cfg.group_size):
            # Use model.generate (yields tokens one at a time)
            generated = list(prompt_ids)
            for token in self.model.generate(
                prompt_ids,
                max_tokens=cfg.max_gen_len,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                seed=42 + i,  # different seed per sample
            ):
                generated.append(token)
                if token == self.eos_id:
                    break
            completions.append(generated)
        return completions

    def compute_advantages(self, rewards: list[float]) -> list[float]:
        """Compute group-relative advantages: (r - mean) / std.

        If all rewards are identical, advantages are zero (no gradient signal).

        Args:
            rewards: List of scalar rewards for each completion in the group.

        Returns:
            List of advantage values, same length as rewards.
        """
        t = torch.tensor(rewards, dtype=torch.float32)
        mean = t.mean()
        std = t.std()
        if std < 1e-8:
            return [0.0] * len(rewards)
        advantages = ((t - mean) / std).tolist()
        return advantages

    def _compute_log_probs(self, model, sequences: list[list[int]], prompt_len: int) -> torch.Tensor:
        """Compute per-token log probabilities for the completion portion.

        Args:
            model: The model to use (policy or reference).
            sequences: List of token sequences (each is prompt + completion).
            prompt_len: Length of the prompt portion.

        Returns:
            Tensor of shape (group_size,) with summed log-probs over completion tokens.
        """
        # Pad sequences to same length
        max_len = max(len(s) for s in sequences)
        batch_ids = torch.full(
            (len(sequences), max_len), 0, dtype=torch.long, device=self.device
        )
        for i, seq in enumerate(sequences):
            batch_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=self.device)

        # Forward pass (no targets => returns logits)
        logits = model(batch_ids)  # (B, T, V)

        # Compute per-token log-probs
        log_probs = F.log_softmax(logits.float(), dim=-1)

        # Gather log-probs for actual next tokens
        # For position t, the model predicts token at t+1
        # We want log_probs for positions [prompt_len-1, ..., seq_len-2] predicting tokens [prompt_len, ..., seq_len-1]
        total_log_probs = []
        for i, seq in enumerate(sequences):
            seq_len = len(seq)
            if seq_len <= prompt_len:
                total_log_probs.append(torch.tensor(0.0, device=self.device))
                continue
            # Gather completion token log-probs
            start = prompt_len - 1  # logit position predicting first completion token
            end = seq_len - 1       # logit position predicting last token
            token_ids = batch_ids[i, prompt_len:seq_len]  # actual completion tokens
            lp = log_probs[i, start:end, :]  # (completion_len, V)
            token_lp = lp.gather(1, token_ids.unsqueeze(1)).squeeze(1)  # (completion_len,)
            total_log_probs.append(token_lp.sum())

        return torch.stack(total_log_probs)

    def step(self, prompts: list[list[int]], test_harnesses: list[str]) -> dict:
        """One GRPO training step over a batch of prompts.

        For each prompt:
        1. Generate group_size completions
        2. Decode and verify each completion
        3. Compute rewards and group-relative advantages
        4. Compute GRPO loss

        Args:
            prompts: List of prompt token sequences.
            test_harnesses: List of test code strings (one per prompt).
                           Empty string means compile-only verification.

        Returns:
            dict with metrics: loss, mean_reward, compile_rate, kl_div.
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
                # Decode the completion (after prompt)
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

            # 4. Compute log-probs under policy and reference
            self.model.train()
            policy_log_probs = self._compute_log_probs(self.model, completions, prompt_len)

            with torch.no_grad():
                ref_log_probs = self._compute_log_probs(self.ref_model, completions, prompt_len)

            # 5. Compute ratios and KL
            log_ratio = policy_log_probs - ref_log_probs
            ratio = torch.exp(log_ratio)
            kl = (ratio - 1) - log_ratio  # approximate KL
            all_kl.append(kl.mean().item())

            # 6. Clipped surrogate loss (PPO-style)
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * advantages_t
            policy_loss = -torch.min(surr1, surr2).mean()

            # 7. KL penalty
            kl_loss = cfg.kl_coeff * kl.mean()

            loss = policy_loss + kl_loss
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
