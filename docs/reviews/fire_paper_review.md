# Review: FIRE (arXiv 2602.08040v1)

**Frobenius-Isometry Reinitialization for Continual Learning in Deep Neural Networks**
February 2026

## What the paper does

FIRE reinitializes weight matrices at phase boundaries during continual learning by projecting them to the nearest orthogonal matrix. The mechanism is Newton-Schulz iteration, the same iteration Muon uses, but applied to a different problem. Muon orthogonalizes the update; FIRE orthogonalizes the weights themselves.

The math is clean. You frame it as constrained optimization: minimize Squared Frobenius Error between old and new weights (stability) subject to Deviation from Isometry = 0 (plasticity). The solution is the polar factor of the weight matrix. Newton-Schulz gets you there iteratively.

The paper proves four theorems connecting DfI (Deviation from Isometry) to things you actually care about: output covariance discrepancy (Theorem 1), Hessian spectral norm (Theorem 2), effective rank (Theorem 3), and dormant neuron count (Theorem 4). This is the strongest part of the paper. Most continual learning work hand-waves about why their method helps plasticity. FIRE gives you a single scalar (DfI) that controls four separate failure modes, and proves it.

## What works well

The constrained optimization framing is the right way to think about this. Polar decomposition is the mathematically natural answer to "closest orthogonal matrix," and the connection to Newton-Schulz is well-established in numerical linear algebra. The paper doesn't overcomplicate it.

Practical overhead is low. The authors report <1% wall-clock increase, 55MB GPU memory, and the only hyperparameter is the number of NS iterations. That's unusual for a continual learning method. Most of them bolt on replay buffers, regularization terms, or auxiliary networks that cost real compute.

The comparison with Muon is useful and honest. Both use NS iteration but they solve different problems. The paper is clear about this distinction, which I appreciate because it would have been easy to either oversell the connection or ignore it.

## What we found when we actually implemented it

We integrated FIRE into our training pipeline and hit several issues the paper doesn't discuss.

### The "5 iterations" claim is too optimistic

The paper says 5 NS iterations are sufficient. This is true for trained weight matrices, which have reasonably clustered singular values. It is not true for randomly initialized matrices. We tested convergence on random 64x64 matrices and needed 15 iterations before the DfI dropped below 1e-6. The paper should be explicit about this: 5 iterations is a property of trained networks, not of the algorithm.

### Frobenius norm initialization diverges on ill-conditioned matrices

The paper normalizes the input as X = W / ||W||_F before running Newton-Schulz. This works when singular values are in a narrow range. When they span a wide range (which happens with certain initialization schemes and after long training runs), the iteration diverges or converges to garbage.

We switched to spectral norm initialization: X = W / ||W||_2. This guarantees the largest singular value is 1, which keeps the iteration stable. The paper should discuss this. It's a known issue in numerical linear algebra and it matters in practice.

### No discussion of initialization scheme compatibility

The sqrt(d_out/d_in) scaling factor after projection is correct for standard initialization. The paper doesn't discuss what happens when you use uniform init, kaiming init, or residual stream scaling (e.g., resid_lambdas in our codebase). We had to work out the interaction ourselves. The scaling factor needs to match whatever initialization scheme the model was originally trained with, and the paper assumes this is obvious. It isn't.

### Optimizer state goes stale after FIRE

This is the biggest practical gap in the paper. After you project weights to the orthogonal manifold, the Adam optimizer's momentum and variance buffers still point at the old parameter landscape. They're garbage. The loss spikes, and it takes many steps for Adam to re-estimate reasonable statistics.

The paper doesn't mention this at all. We had to add selective optimizer state reset: zero out momentum and variance only for parameters that FIRE actually modified. This should be in the paper. Anyone implementing FIRE with Adam (which is most people) will hit this.

### Limited experimental scope

The experiments use ResNet-18 and GPT-0.1B. No experiments with Mamba, SwiGLU activations, grouped-query attention, or mixture-of-experts. These architectures have different weight matrix structure and conditioning. We don't know if the convergence properties or the theoretical bounds transfer.

### No principled layer targeting

The paper applies FIRE to all Linear layers for LLM experiments but only Q/K projections for ViT experiments. There's no justification for this difference. Which layers benefit from FIRE and which don't? Is it harmful to project the output projection? The MLP layers? We ended up using topology-aware targeting (checking `block.is_mamba` to skip Mamba blocks) but had to figure out the right targeting through experimentation.

## Production issues

Three things bit us during integration.

Newton-Schulz with a Python for loop breaks `torch.compile(fullgraph=True)`. You either unroll the loop, write it as a scan, or accept a graph break. The paper doesn't mention compilation at all.

`torch.linalg.matrix_norm(ord=2)` doesn't support bf16. You need an fp32 upcast for the spectral norm computation, then cast back. Minor, but it costs you an hour if you don't know about it going in.

If you're already using Muon (which also runs NS iteration, but on the gradient), there's no guidance on how FIRE interacts with it. Do they conflict? Compose well? We still don't have a good answer, and the paper doesn't address it.

## What we changed for our implementation

- Spectral norm initialization instead of Frobenius norm
- 15 default iterations instead of 5
- Pure orthogonal output from `newton_schulz` (scaling moved to the caller, so the function does one thing)
- Selective optimizer state reset for FIRE'd parameters only
- Topology-aware layer targeting via `block.is_mamba`
- Compiled ReDo surgery kernels for GPU kernel fusion

## Bottom line

The theoretical contribution is solid. Four theorems tying DfI to practical failure modes is real work, and the constrained optimization framing is the right abstraction. The implementation is simple enough that you can get it running in a day.

But the paper undersells the practical difficulties. The convergence claim is too strong, the normalization choice matters more than presented, optimizer state interaction is a real problem that's completely absent, and the experimental scope is narrow. If you're implementing this in a production training pipeline, expect to spend time on the issues above.

Worth reading. Worth implementing. Just don't trust the defaults.
