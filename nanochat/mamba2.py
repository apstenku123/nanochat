"""
Mamba-2 Token Mixer with incremental Mamba-3 Upgrades.
Drop-in replacement for CausalSelfAttention in nanochat.

Phase 1 (Mamba-2): SSD chunked scan with Triton fast path on CUDA.
Phase 2 (Mamba-3 incremental): QK-norm on B/C, learnable B/C bias, complex RoPE.
Phase 3 (Mamba-3 trapezoidal): Trapezoidal discretization with lambda gate.

Three compute paths:
  1. CUDA training/prefill: mamba_chunk_scan_combined Triton kernel (fast)
  2. TPU/XLA training: torch_xla.experimental.scan (no Python loops)
  3. Fallback: _ssd_scan_ref chunked reference (matmul-based, ~8 iterations)
  4. Decode (L=1): _ssd_step_ref O(1) recurrence
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
except ImportError:
    mamba_chunk_scan_combined = None

try:
    from torch_xla.experimental.scan import scan as xla_scan
    _HAVE_XLA_SCAN = True
except ImportError:
    xla_scan = None
    _HAVE_XLA_SCAN = False


class Mamba2Layer(nn.Module):
    """
    Drop-in replacement for CausalSelfAttention.
    Signature: forward(x, cos_sin, window_size, kv_cache) -> (B, T, C)
    Ignores cos_sin and window_size (attention-only args).
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.d_model = config.n_embd
        self.d_state = getattr(config, "mamba_d_state", 64)
        self.d_conv = getattr(config, "mamba_d_conv", 4)
        self.expand = getattr(config, "mamba_expand", 2)
        self.headdim = getattr(config, "mamba_headdim", 128)
        self.ngroups = getattr(config, "mamba_ngroups", 1)
        self.chunk_size = getattr(config, "mamba_chunk_size", 256)

        self.d_inner = self.expand * self.d_model
        assert self.d_inner % self.headdim == 0, f"d_inner={self.d_inner} must be divisible by headdim={self.headdim}"
        self.nheads = self.d_inner // self.headdim
        assert self.nheads % self.ngroups == 0, f"nheads={self.nheads} must be divisible by ngroups={self.ngroups}"
        self.heads_per_group = self.nheads // self.ngroups

        # Phase 2 & 3 toggles
        self.mamba3_qknorm = getattr(config, 'mamba3_qknorm', False)
        self.mamba3_bias = getattr(config, 'mamba3_bias', False)
        self.mamba3_complex_rope = getattr(config, 'mamba3_complex_rope', False)
        self.mamba3_trapezoidal = getattr(config, 'mamba3_trapezoidal', False)

        # Input projection: [z, xBC, dt, (lambda)]
        # z: d_inner, xBC: d_inner + 2*ngroups*d_state, dt: nheads, lambda: nheads (Phase 3 only)
        self.conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        d_lambda = self.nheads if self.mamba3_trapezoidal else 0
        d_in_proj = self.d_inner + self.conv_dim + self.nheads + d_lambda
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False)

        # Causal depthwise conv1d (left at PyTorch kaiming_uniform default, fan_in=d_conv=4)
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim, out_channels=self.conv_dim, kernel_size=self.d_conv,
            groups=self.conv_dim, padding=self.d_conv - 1, bias=True
        )

        # Phase 2: learnable B/C bias (grouped)
        if self.mamba3_bias:
            self.B_bias = nn.Parameter(torch.zeros(self.ngroups, self.d_state))
            self.C_bias = nn.Parameter(torch.zeros(self.ngroups, self.d_state))

        # Per-head SSM parameters
        self.A_log = nn.Parameter(torch.empty(self.nheads))
        self.dt_bias = nn.Parameter(torch.empty(self.nheads))
        self.D = nn.Parameter(torch.ones(self.nheads))

        # Phase 2: complex RoPE frequencies
        if self.mamba3_complex_rope:
            assert self.d_state % 2 == 0, f"Complex RoPE requires even d_state, got {self.d_state}"
            rope_theta = float(getattr(config, 'rope_theta', 10000.0))
            inv_freq = 1.0 / (rope_theta ** (torch.arange(0, self.d_state, 2).float() / self.d_state))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

        # Constructor init (overridden by GPT.init_weights for meta-init safety)
        self._init_mamba_params()

    def _init_mamba_params(self):
        """Official Mamba2Simple init: A_log ~ log(U(1,16)), dt via inverse softplus."""
        A = torch.empty(self.nheads).uniform_(1, 16)
        self.A_log.data.copy_(torch.log(A))
        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=0.001)
        self.dt_bias.data.copy_(dt + torch.log(-torch.expm1(-dt)))

    # -------------------------------------------------------------------------
    # Phase 2: Complex RoPE on B/C
    # -------------------------------------------------------------------------
    def _apply_complex_rope(self, tensor, dt_soft, inference_params=None):
        """
        Rotate B or C using cumulative dt-based angles (Mamba-3 Proposition 4).
        tensor: (B, L, G, N) where N = d_state
        dt_soft: (B, L, H) after softplus
        """
        B_sz, L, G, N = tensor.shape
        # Average dt across heads within each group
        dt_avg = dt_soft.view(B_sz, L, G, self.heads_per_group).mean(dim=-1)  # (B, L, G)

        if inference_params is not None and L == 1:
            # Decode: accumulate angle from previous steps
            key = f"rope_angle_{self.layer_idx}"
            rope_angle = inference_params.key_value_memory_dict.setdefault(
                key, torch.zeros(B_sz, G, device=tensor.device, dtype=tensor.dtype)
            )
            rope_angle = rope_angle + dt_avg.squeeze(1)
            inference_params.key_value_memory_dict[key] = rope_angle
            angles = rope_angle.unsqueeze(1).unsqueeze(-1) * self.inv_freq.view(1, 1, 1, N // 2)
        else:
            # Prefill/training: cumulative sum over sequence
            cumsum_dt = torch.cumsum(dt_avg, dim=1)  # (B, L, G)
            angles = cumsum_dt.unsqueeze(-1) * self.inv_freq.view(1, 1, 1, N // 2)
            # Store final angle for decode continuation
            if inference_params is not None:
                key = f"rope_angle_{self.layer_idx}"
                inference_params.key_value_memory_dict[key] = cumsum_dt[:, -1]

        x1, x2 = tensor[..., :N // 2], tensor[..., N // 2:]
        cos, sin = torch.cos(angles), torch.sin(angles)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    # -------------------------------------------------------------------------
    # Forward dispatch
    # -------------------------------------------------------------------------
    def forward(self, x, cos_sin=None, window_size=None, kv_cache=None):
        B_sz, L, _ = x.shape
        inference_params = getattr(kv_cache, 'mamba_params', None) if kv_cache is not None else None

        # O(1) decode step
        if inference_params is not None and L == 1:
            return self._ssd_step_ref(x, inference_params)

        # --- Projection ---
        proj = self.in_proj(x)  # (B, L, d_in_proj)
        idx = 0
        z = proj[..., idx:idx + self.d_inner]; idx += self.d_inner
        xBC_raw = proj[..., idx:idx + self.conv_dim]; idx += self.conv_dim
        dt_raw = proj[..., idx:idx + self.nheads]; idx += self.nheads
        lam = None
        if self.mamba3_trapezoidal:
            lam = torch.sigmoid(proj[..., idx:idx + self.nheads])  # (B, L, H) in [0,1]

        # --- Depthwise causal conv ---
        xBC = xBC_raw.transpose(1, 2)  # (B, conv_dim, L)
        if inference_params is not None:
            states = inference_params.key_value_memory_dict.setdefault(self.layer_idx, {})
            if "conv_state" not in states:
                states["conv_state"] = torch.zeros(B_sz, self.conv_dim, self.d_conv, device=x.device, dtype=x.dtype)
            conv_state = states["conv_state"]
            if L >= self.d_conv:
                conv_state.copy_(xBC[:, :, -self.d_conv:])
            else:
                conv_state.copy_(torch.roll(conv_state, shifts=-L, dims=-1))
                conv_state[:, :, -L:] = xBC

        xBC = self.conv1d(xBC)[..., :L].transpose(1, 2)  # (B, L, conv_dim)
        xBC = F.silu(xBC)

        # --- Split x, B, C ---
        x_ssm = xBC[..., :self.d_inner].view(B_sz, L, self.nheads, self.headdim)
        B_ssm = xBC[..., self.d_inner:self.d_inner + self.ngroups * self.d_state].view(B_sz, L, self.ngroups, self.d_state)
        C_ssm = xBC[..., -self.ngroups * self.d_state:].view(B_sz, L, self.ngroups, self.d_state)

        # --- Phase 2: QK-norm + bias ---
        if self.mamba3_qknorm:
            B_ssm = F.rms_norm(B_ssm, (self.d_state,))
            C_ssm = F.rms_norm(C_ssm, (self.d_state,))
        if self.mamba3_bias:
            B_ssm = B_ssm + self.B_bias.view(1, 1, self.ngroups, self.d_state)
            C_ssm = C_ssm + self.C_bias.view(1, 1, self.ngroups, self.d_state)

        dt_soft = F.softplus(dt_raw + self.dt_bias)  # (B, L, H)

        # --- Phase 2: Complex RoPE ---
        if self.mamba3_complex_rope:
            B_ssm = self._apply_complex_rope(B_ssm, dt_soft, inference_params)
            C_ssm = self._apply_complex_rope(C_ssm, dt_soft, inference_params)

        A = -torch.exp(self.A_log)  # (H,) negative

        # --- Choose scan backend ---
        # xla_scan backward with SPMD requires torchax which pulls incompatible
        # jax versions.  Fall back to chunked reference scan on TPU for now.
        # The reference scan loops over nchunks (= L/256 ≈ 8) not L, so XLA
        # unrolls it into a manageable HLO graph.
        use_xla = False  # disabled: SPMD backward needs torchax (jax 0.9+)
        use_triton = (mamba_chunk_scan_combined is not None and x.device.type == "cuda"
                      and not self.mamba3_trapezoidal)

        if use_xla:
            y, final_state = self._ssd_scan_xla(x_ssm, dt_soft, A, B_ssm, C_ssm, self.D)
        elif use_triton:
            if inference_params is not None:
                y, final_states = mamba_chunk_scan_combined(
                    x_ssm, dt_soft, A, B_ssm, C_ssm,
                    chunk_size=self.chunk_size, D=self.D, return_final_states=True,
                )
                # Triton returns (B,H,headdim,d_state) -> transpose to our (B,H,d_state,headdim)
                final_state = final_states.transpose(-1, -2).to(torch.float32)
            else:
                y = mamba_chunk_scan_combined(
                    x_ssm, dt_soft, A, B_ssm, C_ssm,
                    chunk_size=self.chunk_size, D=self.D,
                )
                final_state = None
        else:
            y, final_state = self._ssd_scan_ref(x_ssm, dt_soft, A, B_ssm, C_ssm, self.D)

        # Store state for inference
        if inference_params is not None and final_state is not None:
            states = inference_params.key_value_memory_dict.setdefault(self.layer_idx, {})
            states["ssm_state"] = final_state.to(torch.float32)

        # --- Output gate + projection ---
        y = y.reshape(B_sz, L, self.d_inner)
        y = F.rms_norm(y, (self.d_inner,)) * F.silu(z)
        return self.out_proj(y)

    # -------------------------------------------------------------------------
    # O(1) Autoregressive Decode Step
    # -------------------------------------------------------------------------
    def _ssd_step_ref(self, x, inference_params):
        """Constant-time decode: x is (B, 1, d_model)."""
        B_sz = x.shape[0]

        # Project
        proj = self.in_proj(x.squeeze(1))  # (B, d_in_proj)
        idx = 0
        z = proj[..., idx:idx + self.d_inner]; idx += self.d_inner
        xBC_raw = proj[..., idx:idx + self.conv_dim]; idx += self.conv_dim
        dt_raw = proj[..., idx:idx + self.nheads]; idx += self.nheads
        lam = None
        if self.mamba3_trapezoidal:
            lam = torch.sigmoid(proj[..., idx:idx + self.nheads])  # (B, H)

        # Get/create state dict
        states = inference_params.key_value_memory_dict.setdefault(self.layer_idx, {})
        if "conv_state" not in states:
            states["conv_state"] = torch.zeros(B_sz, self.conv_dim, self.d_conv, device=x.device, dtype=x.dtype)
        if "ssm_state" not in states:
            states["ssm_state"] = torch.zeros(B_sz, self.nheads, self.d_state, self.headdim,
                                              device=x.device, dtype=torch.float32)
        conv_state = states["conv_state"]
        ssm_state = states["ssm_state"]

        # Defensive fp32 upcast (handles bf16 state from prefill)
        if ssm_state.dtype != torch.float32:
            ssm_state = ssm_state.float()
            states["ssm_state"] = ssm_state

        # Conv shift register
        conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
        conv_state[:, :, -1] = xBC_raw

        xBC_conv = torch.sum(conv_state * self.conv1d.weight.squeeze(1), dim=-1)
        if self.conv1d.bias is not None:
            xBC_conv = xBC_conv + self.conv1d.bias
        xBC_conv = F.silu(xBC_conv)

        # Split
        x_ssm = xBC_conv[..., :self.d_inner].view(B_sz, self.nheads, self.headdim)
        B_ssm = xBC_conv[..., self.d_inner:self.d_inner + self.ngroups * self.d_state].view(B_sz, self.ngroups, self.d_state)
        C_ssm = xBC_conv[..., -self.ngroups * self.d_state:].view(B_sz, self.ngroups, self.d_state)

        # Phase 2: QK-norm + bias
        if self.mamba3_qknorm:
            B_ssm = F.rms_norm(B_ssm, (self.d_state,))
            C_ssm = F.rms_norm(C_ssm, (self.d_state,))
        if self.mamba3_bias:
            B_ssm = B_ssm + self.B_bias.view(1, self.ngroups, self.d_state)
            C_ssm = C_ssm + self.C_bias.view(1, self.ngroups, self.d_state)

        dt_soft = F.softplus(dt_raw + self.dt_bias)  # (B, H)
        A = -torch.exp(self.A_log)

        # Phase 2: Complex RoPE (unsqueeze to L=1 for API compatibility)
        if self.mamba3_complex_rope:
            B_ssm = self._apply_complex_rope(B_ssm.unsqueeze(1), dt_soft.unsqueeze(1), inference_params).squeeze(1)
            C_ssm = self._apply_complex_rope(C_ssm.unsqueeze(1), dt_soft.unsqueeze(1), inference_params).squeeze(1)

        # SSM recurrence
        dA = torch.exp(dt_soft * A)  # (B, H)

        if self.mamba3_trapezoidal:
            assert lam is not None
            # Bug #20 fix: store prev_B at group-level (B,ngroups,d_state), not head-level
            prev_B = states.setdefault("prev_B", torch.zeros(B_sz, self.ngroups, self.d_state,
                                                              device=x.device, dtype=torch.float32))
            prev_x = states.setdefault("prev_x", torch.zeros(B_sz, self.nheads, self.headdim,
                                                              device=x.device, dtype=x.dtype))

            # Expand B to head-level for math
            prev_B_h = prev_B.repeat_interleave(self.heads_per_group, dim=1)  # (B, H, N)
            B_ssm_h = B_ssm.repeat_interleave(self.heads_per_group, dim=1)
            C_ssm_h = C_ssm.repeat_interleave(self.heads_per_group, dim=1)

            beta = (1.0 - lam) * dt_soft * dA   # (B, H)
            gamma = lam * dt_soft                 # (B, H)

            dBx_prev = (beta.unsqueeze(-1) * prev_B_h).unsqueeze(-1) * prev_x.to(torch.float32).unsqueeze(-2)
            dBx_curr = (gamma.unsqueeze(-1) * B_ssm_h.float()).unsqueeze(-1) * x_ssm.to(torch.float32).unsqueeze(-2)

            ssm_state.copy_(ssm_state * dA.view(B_sz, self.nheads, 1, 1).float() + dBx_prev + dBx_curr)

            # Cache at group-level for memory efficiency
            states["prev_B"].copy_(B_ssm.float())
            states["prev_x"].copy_(x_ssm)
        else:
            # Standard Mamba-2 recurrence
            B_ssm_h = B_ssm.repeat_interleave(self.heads_per_group, dim=1)
            C_ssm_h = C_ssm.repeat_interleave(self.heads_per_group, dim=1)

            dBx = (dt_soft.unsqueeze(-1) * B_ssm_h.float()).unsqueeze(-1) * x_ssm.to(torch.float32).unsqueeze(-2)
            ssm_state.copy_(ssm_state * dA.view(B_sz, self.nheads, 1, 1).float() + dBx)

        # Output
        y = (C_ssm_h.float().unsqueeze(-1) * ssm_state).sum(dim=-2).to(x.dtype) + self.D.view(1, self.nheads, 1) * x_ssm
        y = y.view(B_sz, 1, self.d_inner)
        y = F.rms_norm(y, (self.d_inner,)) * F.silu(z.unsqueeze(1))
        return self.out_proj(y)

    # -------------------------------------------------------------------------
    # XLA Scan (TPU-optimized, no Python loops)
    # -------------------------------------------------------------------------
    def _ssd_scan_xla(self, x_ssm, dt, A, B_ssm, C_ssm, D):
        """Native XLA scan for TPU: O(T) with no Python loop overhead."""
        B_sz, T, H, D_head = x_ssm.shape
        G, N = self.ngroups, self.d_state
        Hpg = self.heads_per_group

        # Reshape to grouped form to avoid repeat_interleave inside scan body
        x_g = x_ssm.view(B_sz, T, G, Hpg, D_head)
        dt_g = dt.view(B_sz, T, G, Hpg)
        A_g = A.view(G, Hpg)
        D_g = D.view(G, Hpg)

        state0 = torch.zeros(B_sz, G, Hpg, N, D_head, device=x_ssm.device, dtype=torch.float32)

        # Broadcast A_g and D_g to time-major tensors so they're traced as scan
        # inputs rather than captured closures (functorch requires FakeTensors).
        A_g_t = A_g.unsqueeze(0).expand(T, -1, -1).contiguous().float()   # (T, G, Hpg)
        D_g_t = D_g.unsqueeze(0).expand(T, -1, -1).contiguous().float()   # (T, G, Hpg)

        # Time-major for scan: leading dim is T
        xs = (
            x_g.transpose(0, 1).float(),
            dt_g.transpose(0, 1).float(),
            B_ssm.transpose(0, 1).float(),
            C_ssm.transpose(0, 1).float(),
            A_g_t,
            D_g_t,
        )

        def body(carry, inp):
            x_t, dt_t, B_t, C_t, A_t, D_t = inp
            alpha = torch.exp(dt_t * A_t.unsqueeze(0))
            dtB = dt_t.unsqueeze(-1) * B_t.unsqueeze(-2)       # (B,G,Hpg,N)
            dBx = dtB.unsqueeze(-1) * x_t.unsqueeze(-2)         # (B,G,Hpg,N,D)
            new_carry = carry * alpha.unsqueeze(-1).unsqueeze(-1) + dBx
            y_t = (C_t.unsqueeze(-2).unsqueeze(-1) * new_carry).sum(dim=-2)  # (B,G,Hpg,D)
            y_t = y_t + D_t.unsqueeze(0).unsqueeze(-1) * x_t
            return new_carry, y_t

        stateT, ys = xla_scan(body, state0, xs)
        y = ys.transpose(0, 1).reshape(B_sz, T, H, D_head).to(x_ssm.dtype)
        final_state = stateT.reshape(B_sz, H, N, D_head).to(torch.float32)
        return y, final_state

    # -------------------------------------------------------------------------
    # Chunked Reference Scan (CPU/CUDA fallback, ~8 iterations not T)
    # -------------------------------------------------------------------------
    def _ssd_scan_ref(self, x, dt, A, B, C, D):
        """
        Chunked SSD scan: cross-chunk + within-chunk (the "dual" of SSD).
        Loops over nchunks = L/256 ≈ 8, NOT over L.
        """
        B_sz, L, H, D_head = x.shape
        _, _, G, N = B.shape
        cs = self.chunk_size

        # Pad to chunk_size multiple
        pad = (cs - L % cs) % cs
        if pad > 0:
            # F.pad: 6 values for 4D = (dim3_L, dim3_R, dim2_L, dim2_R, dim1_L, dim1_R)
            x = F.pad(x, (0, 0, 0, 0, 0, pad))
            dt = F.pad(dt, (0, 0, 0, pad))
            B = F.pad(B, (0, 0, 0, 0, 0, pad))
            C = F.pad(C, (0, 0, 0, 0, 0, pad))

        nchunks = x.shape[1] // cs
        x_c = x.view(B_sz, nchunks, cs, H, D_head)
        dt_c = dt.view(B_sz, nchunks, cs, H)
        B_c = B.view(B_sz, nchunks, cs, G, N)
        C_c = C.view(B_sz, nchunks, cs, G, N)

        # Expand groups to heads
        B_h = B_c.repeat_interleave(self.heads_per_group, dim=3)
        C_h = C_c.repeat_interleave(self.heads_per_group, dim=3)

        # Cumulative decay within each chunk
        dA_c = dt_c * A.view(1, 1, 1, H)
        dA_cumsum = torch.cumsum(dA_c, dim=2)

        # Cross-chunk states
        decay_to_end = torch.exp(dA_cumsum[:, :, -1:] - dA_cumsum)
        x_dt = x_c * dt_c.unsqueeze(-1)
        chunk_states = torch.einsum('bclhn,bclhd->bchnd', B_h * decay_to_end.unsqueeze(-1), x_dt)

        chunk_decay = torch.exp(dA_cumsum[:, :, -1])

        # FP32 state accumulation for numerical stability
        running_state = torch.zeros(B_sz, H, N, D_head, device=x.device, dtype=torch.float32)
        chunk_decay_f32 = chunk_decay.to(torch.float32)
        chunk_states_f32 = chunk_states.to(torch.float32)

        all_prev_states = []
        for c in range(nchunks):
            all_prev_states.append(running_state)
            # running_state = running_state * ... + ... creates new tensor (safe without .clone())
            running_state = running_state * chunk_decay_f32[:, c].view(B_sz, H, 1, 1) + chunk_states_f32[:, c]

        prev_states = torch.stack(all_prev_states, dim=1).to(x.dtype)

        # y_cross: contribution from previous chunks
        cross_decay = torch.exp(dA_cumsum).unsqueeze(-1)
        y_cross = torch.einsum('bclhn,bchnd->bclhd', C_h, prev_states) * cross_decay

        # y_local: within-chunk attention-like quadratic (the SSD "dual")
        CB = torch.einsum('bclhn,bcshn->bclsh', C_h, B_h)
        diff = dA_cumsum.unsqueeze(3) - dA_cumsum.unsqueeze(2)
        causal_mask = torch.tril(torch.ones(cs, cs, device=x.device, dtype=torch.bool))
        decay_mat = torch.exp(diff.masked_fill(~causal_mask.view(1, 1, cs, cs, 1), float('-inf')))
        attn = CB * decay_mat
        y_local = torch.einsum('bclsh,bcshd->bclhd', attn, x_dt)

        y = y_local + y_cross + x_c * D.view(1, 1, 1, H, 1)
        y = y.reshape(B_sz, nchunks * cs, H, D_head)

        if pad > 0:
            y = y[:, :L]

        # Return fp32 running_state (preserves precision for decode)
        return y, running_state
