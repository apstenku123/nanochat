#!/usr/bin/env python3
"""
Mamba backend throughput/memory benchmark.

Usage:
    python scripts/bench_mamba_backends.py
    python scripts/bench_mamba_backends.py --n_layer 24 --n_embd 1536 --batch 8
"""
import argparse
import time
import torch
from dataclasses import replace
from nanochat.gpt import GPT, GPTConfig


def run_case(name, config, device, B=4, T=2048, warmup=5, iters=20):
    model = GPT(config)
    model.init_weights()
    model = model.to(device).bfloat16()
    model.train()

    x = torch.randint(0, config.vocab_size, (B, T), device=device)
    y = torch.randint(0, config.vocab_size, (B, T), device=device)

    # Use project's actual optimizer setup (Muon + AdamW split)
    try:
        optimizer_list = model.setup_optimizers(
            matrix_lr=0.01, embedding_lr=0.2, unembedding_lr=0.004,
            scalar_lr=0.5,
        )
    except Exception:
        # Fallback if Muon not available
        optimizer_list = [torch.optim.AdamW(model.parameters(), lr=1e-3)]

    for _ in range(warmup):
        loss = model(x, targets=y)
        loss.backward()
        for opt in optimizer_list:
            opt.step()
            opt.zero_grad()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for _ in range(iters):
        loss = model(x, targets=y)
        loss.backward()
        for opt in optimizer_list:
            opt.step()
            opt.zero_grad()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    ms = elapsed * 1000 / iters
    tps = B * T * iters / elapsed
    mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"  [{name:45s}] {ms:7.1f} ms/step | {tps:>10,.0f} tok/s | {mem:>7.0f} MB")

    del model, x, y
    for opt in optimizer_list:
        del opt
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Mamba backend benchmark")
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--n_embd", type=int, default=768)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required for benchmarking"
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Config: {args.n_layer}L x {args.n_embd}d, seq={args.seq_len}, batch={args.batch}\n")

    base = GPTConfig(
        n_layer=args.n_layer, n_embd=args.n_embd,
        sequence_len=args.seq_len, vocab_size=50304,
    )

    run_case("Attention only (baseline)", base, "cuda",
             B=args.batch, T=args.seq_len, warmup=args.warmup, iters=args.iters)

    run_case("AAM hybrid (Euler)", replace(base, mamba_enabled=True, mamba_pattern="AAM"),
             "cuda", B=args.batch, T=args.seq_len, warmup=args.warmup, iters=args.iters)

    run_case("AAM + trapezoidal (Idea G dual-scan)",
             replace(base, mamba_enabled=True, mamba_pattern="AAM", mamba3_trapezoidal=True),
             "cuda", B=args.batch, T=args.seq_len, warmup=args.warmup, iters=args.iters)

    run_case("AAM + all Mamba-3 features",
             replace(base, mamba_enabled=True, mamba_pattern="AAM",
                     mamba3_qknorm=True, mamba3_bias=True,
                     mamba3_complex_rope=True, mamba3_trapezoidal=True),
             "cuda", B=args.batch, T=args.seq_len, warmup=args.warmup, iters=args.iters)


if __name__ == "__main__":
    main()
