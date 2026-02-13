"""
Computational proof: Is the Idea G dual-scan trapezoidal decomposition mathematically exact?

The trapezoidal SSM recurrence is:
  h_t = exp(dt_t * A) * h_{t-1} + gamma_t * dt_t * B_t * x_t + beta_t * dt_t * exp(dt_t * A) * B_{t-1} * x_{t-1}

where gamma_t = lam_t, beta_t = (1 - lam_t).

The claim: this can be decomposed into two standard SSD scans:
  Scan 1 (current term):  h^(1)_t = exp(dt_t*A) * h^(1)_{t-1} + dt_t * B_t * (lam_t * x_t)
  Scan 2 (previous term): h^(2)_t = exp(dt_t*A) * h^(2)_{t-1} + dt_t * B_t' * ((1-lam_t)*exp(dt_t*A)*x_{t-1})

Wait — scan 2 uses B_prev (shifted B), not B_t. And the standard scan formula is:
  h_t = exp(dt_t*A) * h_{t-1} + dt_t * B_t * x_t

So for scan 2, with x_prev_mod = (1-lam_t)*exp(dt_t*A)*x_{t-1} and B_prev_t = B_{t-1}:
  h^(2)_t = exp(dt_t*A) * h^(2)_{t-1} + dt_t * B_{t-1} * [(1-lam_t)*exp(dt_t*A)*x_{t-1}]

The trapezoidal previous term is:
  beta_t * dt_t * exp(dt_t*A) * B_{t-1} * x_{t-1} = (1-lam_t) * dt_t * exp(dt_t*A) * B_{t-1} * x_{t-1}

These match! The key insight the alternative proposal makes is correct:
The scan applies exp(dt_t*A) to h_{t-1}, NOT to the new input dt_t*B_t*x_t.
So embedding exp(dt_t*A) into x_prev_mod doesn't cause double-decay.

But wait — does the cross-chunk/cross-step propagation cause issues? Let's verify numerically.
"""
import torch
import torch.nn.functional as F


def test_idea_g_single_step_exact():
    """Verify single-step trapezoidal matches dual-scan at step level."""
    torch.manual_seed(42)
    B, H, N, D = 1, 4, 8, 16
    A = -torch.rand(H)  # negative

    # Two consecutive steps
    dt = torch.rand(2, H) * 0.1
    lam = torch.sigmoid(torch.randn(2, H))
    x = torch.randn(2, H, D)
    B_mat = torch.randn(2, H, N)  # already expanded to heads

    # Ground truth: sequential trapezoidal recurrence
    h = torch.zeros(H, N, D, dtype=torch.float64)

    # step 0: no previous term (x_{-1} = 0)
    dA_0 = torch.exp(dt[0].double() * A.double())
    gamma_0 = lam[0].double() * dt[0].double()
    h = dA_0.view(H, 1, 1) * h + (gamma_0.unsqueeze(-1) * B_mat[0].double()).unsqueeze(-1) * x[0].double().unsqueeze(-2)
    y_true_0 = h.clone()

    # step 1: has previous term from step 0
    dA_1 = torch.exp(dt[1].double() * A.double())
    gamma_1 = lam[1].double() * dt[1].double()
    beta_1 = (1 - lam[1].double()) * dt[1].double() * dA_1

    h = dA_1.view(H, 1, 1) * h + \
        (gamma_1.unsqueeze(-1) * B_mat[1].double()).unsqueeze(-1) * x[1].double().unsqueeze(-2) + \
        (beta_1.unsqueeze(-1) * B_mat[0].double()).unsqueeze(-1) * x[0].double().unsqueeze(-2)
    y_true_1 = h.clone()

    # Dual-scan approach
    # Scan 1 (current): x_curr_mod = lam * x
    x_curr = torch.zeros(2, H, D, dtype=torch.float64)
    x_curr[0] = lam[0].double().unsqueeze(-1) * x[0].double()
    x_curr[1] = lam[1].double().unsqueeze(-1) * x[1].double()

    # Scan 2 (previous): x_prev_mod = (1-lam) * exp(dt*A) * x_{t-1}
    x_prev = torch.zeros(2, H, D, dtype=torch.float64)
    # step 0: x_{-1} = 0
    x_prev[0] = 0
    # step 1: x_{0}
    dA_1_vec = torch.exp(dt[1].double() * A.double())
    x_prev[1] = (1 - lam[1].double()).unsqueeze(-1) * dA_1_vec.unsqueeze(-1) * x[0].double()

    B_shifted = torch.zeros_like(B_mat.double())
    B_shifted[1] = B_mat[0].double()

    # Run standard Euler scans
    h1 = torch.zeros(H, N, D, dtype=torch.float64)
    h2 = torch.zeros(H, N, D, dtype=torch.float64)

    # Scan 1 step 0
    dA_0 = torch.exp(dt[0].double() * A.double())
    h1 = dA_0.view(H, 1, 1) * h1 + (dt[0].double().unsqueeze(-1) * B_mat[0].double()).unsqueeze(-1) * x_curr[0].unsqueeze(-2)

    # Scan 2 step 0
    h2 = dA_0.view(H, 1, 1) * h2 + (dt[0].double().unsqueeze(-1) * B_shifted[0]).unsqueeze(-1) * x_prev[0].unsqueeze(-2)

    y_dual_0 = h1 + h2

    # Scan 1 step 1
    dA_1 = torch.exp(dt[1].double() * A.double())
    h1 = dA_1.view(H, 1, 1) * h1 + (dt[1].double().unsqueeze(-1) * B_mat[1].double()).unsqueeze(-1) * x_curr[1].unsqueeze(-2)

    # Scan 2 step 1
    h2 = dA_1.view(H, 1, 1) * h2 + (dt[1].double().unsqueeze(-1) * B_shifted[1]).unsqueeze(-1) * x_prev[1].unsqueeze(-2)

    y_dual_1 = h1 + h2

    # Compare
    err_0 = (y_true_0 - y_dual_0).abs().max().item()
    err_1 = (y_true_1 - y_dual_1).abs().max().item()
    print(f"Step 0 max error: {err_0:.2e}")
    print(f"Step 1 max error: {err_1:.2e}")
    assert err_0 < 1e-12, f"Step 0 mismatch: {err_0}"
    assert err_1 < 1e-12, f"Step 1 mismatch: {err_1}"


def test_idea_g_multi_step_exact():
    """Verify over 32 steps that dual-scan matches sequential trapezoidal exactly."""
    torch.manual_seed(123)
    H, N, D = 4, 8, 16
    T = 32
    A = -torch.rand(H).double() * 0.5

    dt = torch.rand(T, H).double() * 0.1
    lam = torch.sigmoid(torch.randn(T, H)).double()
    x = torch.randn(T, H, D).double()
    B_mat = torch.randn(T, H, N).double()

    # Ground truth: sequential trapezoidal
    h_true = torch.zeros(H, N, D, dtype=torch.float64)
    states_true = []
    for t in range(T):
        dA_t = torch.exp(dt[t] * A)
        gamma_t = lam[t] * dt[t]
        beta_t = (1 - lam[t]) * dt[t] * dA_t

        new_h = dA_t.view(H, 1, 1) * h_true
        # current term
        new_h = new_h + (gamma_t.unsqueeze(-1) * B_mat[t]).unsqueeze(-1) * x[t].unsqueeze(-2)
        # previous term
        if t > 0:
            new_h = new_h + (beta_t.unsqueeze(-1) * B_mat[t - 1]).unsqueeze(-1) * x[t - 1].unsqueeze(-2)
        h_true = new_h
        states_true.append(h_true.clone())

    # Dual-scan approach
    # Prepare modified inputs
    x_curr = lam.unsqueeze(-1) * x  # (T, H, D)

    dA_all = torch.exp(dt * A.unsqueeze(0))  # (T, H)
    x_prev_mod = torch.zeros_like(x)
    B_shifted = torch.zeros_like(B_mat)
    for t in range(1, T):
        x_prev_mod[t] = (1 - lam[t]).unsqueeze(-1) * dA_all[t].unsqueeze(-1) * x[t - 1]
        B_shifted[t] = B_mat[t - 1]

    # Run two standard Euler scans
    h1 = torch.zeros(H, N, D, dtype=torch.float64)
    h2 = torch.zeros(H, N, D, dtype=torch.float64)
    states_dual = []
    for t in range(T):
        dA_t = torch.exp(dt[t] * A)
        h1 = dA_t.view(H, 1, 1) * h1 + (dt[t].unsqueeze(-1) * B_mat[t]).unsqueeze(-1) * x_curr[t].unsqueeze(-2)
        h2 = dA_t.view(H, 1, 1) * h2 + (dt[t].unsqueeze(-1) * B_shifted[t]).unsqueeze(-1) * x_prev_mod[t].unsqueeze(-2)
        states_dual.append(h1 + h2)

    # Compare all steps
    max_err = 0.0
    for t in range(T):
        err = (states_true[t] - states_dual[t]).abs().max().item()
        max_err = max(max_err, err)
    print(f"Max error over {T} steps: {max_err:.2e}")
    assert max_err < 1e-10, f"Multi-step mismatch: {max_err}"


def test_idea_g_with_chunked_scan():
    """Verify dual-scan works correctly through the chunked SSD scan implementation.

    This tests the actual code path: construct x_curr_mod and x_prev_mod,
    run them through _ssd_scan_ref twice, sum outputs.
    Compare against a naive sequential trapezoidal loop.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from dataclasses import dataclass

    @dataclass
    class TinyConfig:
        n_embd: int = 64
        mamba_d_state: int = 8
        mamba_d_conv: int = 4
        mamba_expand: int = 2
        mamba_headdim: int = 16
        mamba_ngroups: int = 1
        mamba_chunk_size: int = 8
        mamba3_qknorm: bool = False
        mamba3_bias: bool = False
        mamba3_complex_rope: bool = False
        mamba3_trapezoidal: bool = True
        rope_theta: float = 10000.0

    from nanochat.mamba2 import Mamba2Layer

    cfg = TinyConfig()
    layer = Mamba2Layer(cfg, layer_idx=0).double()

    B_sz, T = 1, 24
    H = layer.nheads
    D_head = layer.headdim
    G = layer.ngroups
    N = layer.d_state

    torch.manual_seed(777)
    x_ssm = torch.randn(B_sz, T, H, D_head, dtype=torch.float64)
    dt_soft = torch.rand(B_sz, T, H, dtype=torch.float64) * 0.1 + 0.01
    A = -torch.rand(H, dtype=torch.float64) * 0.5
    B_ssm = torch.randn(B_sz, T, G, N, dtype=torch.float64)
    C_ssm = torch.randn(B_sz, T, G, N, dtype=torch.float64)
    D_param = torch.randn(H, dtype=torch.float64)
    lam = torch.sigmoid(torch.randn(B_sz, T, H, dtype=torch.float64))

    # Ground truth: naive sequential trapezoidal
    hpg = H // G
    B_exp = B_ssm.repeat_interleave(hpg, dim=2)  # (B, T, H, N)
    C_exp = C_ssm.repeat_interleave(hpg, dim=2)

    h = torch.zeros(B_sz, H, N, D_head, dtype=torch.float64)
    y_true = torch.zeros(B_sz, T, H, D_head, dtype=torch.float64)
    for t in range(T):
        dA_t = torch.exp(dt_soft[:, t] * A.unsqueeze(0))  # (B, H)
        gamma_t = lam[:, t] * dt_soft[:, t]
        beta_t = (1 - lam[:, t]) * dt_soft[:, t] * dA_t

        h = dA_t.unsqueeze(-1).unsqueeze(-1) * h
        # current
        h = h + (gamma_t.unsqueeze(-1) * B_exp[:, t]).unsqueeze(-1) * x_ssm[:, t].unsqueeze(-2)
        # previous
        if t > 0:
            h = h + (beta_t.unsqueeze(-1) * B_exp[:, t - 1]).unsqueeze(-1) * x_ssm[:, t - 1].unsqueeze(-2)

        y_true[:, t] = (C_exp[:, t].unsqueeze(-1) * h).sum(dim=-2) + D_param.view(1, H, 1) * x_ssm[:, t]

    # Dual-scan approach through _ssd_scan_ref
    x_curr_mod = x_ssm * lam.unsqueeze(-1)
    dA_all = torch.exp(dt_soft * A.view(1, 1, H))
    x_prev_mod = torch.zeros_like(x_ssm)
    x_prev_mod[:, 1:] = (1 - lam[:, 1:]).unsqueeze(-1) * dA_all[:, 1:].unsqueeze(-1) * x_ssm[:, :-1]

    B_prev = torch.zeros_like(B_ssm)
    B_prev[:, 1:] = B_ssm[:, :-1]

    y_curr, s_curr = layer._ssd_scan_ref(x_curr_mod, dt_soft, A, B_ssm, C_ssm, D=None)
    y_prev, s_prev = layer._ssd_scan_ref(x_prev_mod, dt_soft, A, B_prev, C_ssm, D=None)

    y_dual = y_curr + y_prev + x_ssm * D_param.view(1, 1, H, 1)

    err = (y_true - y_dual).abs().max().item()
    print(f"Chunked dual-scan vs naive sequential: max error = {err:.2e}")
    assert err < 1e-6, f"Chunked dual-scan mismatch: {err}"

    # Also verify final state
    s_dual = s_curr + s_prev
    print(f"Final state dual-scan norm: {s_dual.norm():.4f}")
    assert torch.isfinite(s_dual).all()


if __name__ == "__main__":
    test_idea_g_single_step_exact()
    test_idea_g_multi_step_exact()
    test_idea_g_with_chunked_scan()
    print("\nAll Idea G math tests passed!")
