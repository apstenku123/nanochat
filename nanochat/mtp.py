"""
Multi-Token Prediction (MTP) module, following DeepSeek-V3 design.

At each depth k, the MTP module:
1. Concatenates RMSNorm'd hidden state with RMSNorm'd embedding of the next token
2. Projects from 2*n_embd back to n_embd
3. Passes through a dedicated transformer block
4. Computes cross-entropy loss for predicting the token 2 positions ahead

During training, MTP loss is added to the main next-token loss:
    total_loss = main_loss + mtp_lambda * mtp_loss

At inference, MTP modules are ignored (main model works independently).

Reference: DeepSeek-V3 Technical Report (arXiv:2412.19437)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat import kernels


class MTPModule(nn.Module):
    """Single-depth Multi-Token Prediction head (D=1).

    Predicts token at position i+2 given:
    - hidden_states[i]: output of the main transformer at position i
    - next_token_ids[i]: ground-truth token at position i+1 (= main model target)
    """

    def __init__(self, config):
        super().__init__()
        n_embd = config.n_embd
        # Projection: [RMSNorm(hidden); RMSNorm(emb)] -> n_embd
        self.proj = nn.Linear(2 * n_embd, n_embd, bias=False)

        # Dedicated transformer block (imports Block locally to avoid circular imports)
        from nanochat.gpt import Block
        # MTP block: no engram, no mhc, no DSA -- just plain attn+mlp
        # We create a minimal config copy to ensure the block is plain
        from dataclasses import replace
        plain_config = replace(
            config,
            engram_enabled=False,
            mhc_enabled=False,
            dsa_enabled=False,
        )
        self.block = Block(plain_config, layer_idx=0, engram_layers=set())

    def forward(self, hidden_states, next_token_ids, wte, lm_head_weight,
                cos_sin, softcap=15):
        """
        Args:
            hidden_states: (B, T, C) - final hidden states from main model (before lm_head)
            next_token_ids: (B, T) - ground-truth token at position i+1
            wte: nn.Embedding - shared token embedding from main model
            lm_head_weight: (V, C) - shared lm_head weight from main model
            cos_sin: tuple of (cos, sin) rotary embeddings
            softcap: logit softcap value

        Returns:
            mtp_loss: scalar cross-entropy loss for predicting position i+2
        """
        B, T, C = hidden_states.shape

        # We can only predict up to position T-1 (need ground truth at i+2)
        # hidden_states[:, :-1] -> predict token at positions 2..T
        T_mtp = T - 1

        # Pallas flash attention requires seq_len divisible by 1024 (block size).
        # T-1 may not satisfy this (e.g. 65536-1=65535). Truncate to nearest
        # multiple of 1024 so FA backward pass works. This drops at most 1023
        # tokens from the end of the MTP sequence, which is negligible.
        fa_block = 1024
        if T_mtp % fa_block != 0:
            T_mtp = (T_mtp // fa_block) * fa_block

        h = hidden_states[:, :T_mtp]  # (B, T_mtp, C)
        next_emb = wte(next_token_ids[:, :T_mtp])  # (B, T_mtp, C)

        # RMSNorm both inputs independently, then concatenate and project
        h_norm = F.rms_norm(h, (C,))
        e_norm = F.rms_norm(next_emb, (C,))
        combined = torch.cat([h_norm, e_norm], dim=-1)  # (B, T_mtp, 2C)
        h_mtp = self.proj(combined)  # (B, T_mtp, C)

        # Truncate cos/sin to match shortened sequence
        cos, sin = cos_sin
        cos_short = cos[:, :T_mtp]
        sin_short = sin[:, :T_mtp]

        # Pass through dedicated transformer block
        h_mtp = self.block(h_mtp, (cos_short, sin_short), window_size=(-1, 0), kv_cache=None)
        h_mtp = F.rms_norm(h_mtp, (C,))

        # MTP targets: tokens at positions 1..T_mtp = next_token_ids[:, 1:T_mtp+1]
        mtp_targets = next_token_ids[:, 1:T_mtp + 1].contiguous()  # (B, T_mtp)

        # Compute loss using shared lm_head
        mtp_loss = kernels.fused_linear_cross_entropy(
            h_mtp.to(torch.bfloat16),
            lm_head_weight.to(torch.bfloat16),
            mtp_targets,
            ignore_index=-1,
            softcap=softcap,
        )
        return mtp_loss
