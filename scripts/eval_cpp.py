"""
Evaluate a nanochat model on C++ code generation benchmarks.

Usage:
    NANOCHAT_CPP_TOKENIZER=1 .venv/bin/python -m scripts.eval_cpp \
        --checkpoint ~/.cache/nanochat/base_checkpoints/d12/model_000500.pt \
        --problems data/cpp_bench.jsonl \
        --num_samples 10 --temperature 0.8

    # Quick smoke test (2 samples, 3 easy problems):
    NANOCHAT_CPP_TOKENIZER=1 .venv/bin/python -m scripts.eval_cpp --quick

    # Compilation rate only (fast):
    NANOCHAT_CPP_TOKENIZER=1 .venv/bin/python -m scripts.eval_cpp --compile-only

    # Verify benchmark problems (no model needed):
    .venv/bin/python -m scripts.eval_cpp --verify
"""

import argparse
import json
import os
import sys
import time

import torch

from nanochat.cpp_eval import CppBenchmark, verify_problems


def load_problems(path: str, difficulty: str = None, max_problems: int = None) -> list[dict]:
    """Load problems from JSONL file, optionally filtering by difficulty."""
    with open(path, 'r') as f:
        problems = [json.loads(line) for line in f if line.strip()]
    if difficulty:
        problems = [p for p in problems if p.get('difficulty') == difficulty]
    if max_problems:
        problems = problems[:max_problems]
    return problems


def load_model_and_tokenizer(checkpoint_path: str, device: str):
    """Load model from checkpoint file and return (model, tokenizer)."""
    from nanochat.checkpoint_manager import build_model
    # Parse checkpoint path to get directory and step
    # Expected: .../model_000500.pt
    basename = os.path.basename(checkpoint_path)
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Extract step from filename
    import re
    match = re.match(r'model_(\d+)\.pt', basename)
    if match:
        step = int(match.group(1))
    else:
        raise ValueError(f"Cannot parse step from checkpoint filename: {basename}")

    model, tokenizer, meta = build_model(checkpoint_dir, step, torch.device(device), phase="eval")
    return model, tokenizer, meta


def print_results(results: dict):
    """Pretty-print benchmark results."""
    print("\n" + "=" * 70)
    print("C++ Code Generation Benchmark Results")
    print("=" * 70)

    # Per-problem table
    print(f"\n{'Problem':<25} {'Diff':<8} {'Compiled':<12} {'Passed':<12} {'pass@1':<10} {'pass@10':<10}")
    print("-" * 70)
    for p in results['problems']:
        print(f"{p['name']:<25} {p['difficulty']:<8} "
              f"{p['num_compiled']}/{p['num_samples']:<9} "
              f"{p['num_passed']}/{p['num_samples']:<9} "
              f"{p['pass_at_1']:<10.3f} "
              f"{p['pass_at_10']:<10.3f}")

    # Aggregate
    agg = results['aggregate']
    print("-" * 70)
    print(f"{'OVERALL':<25} {'':8} "
          f"{agg['overall_compilation_rate']:<12.1%} "
          f"{agg['total_passed']}/{agg['total_samples']:<9} "
          f"{agg['mean_pass_at_1']:<10.3f} "
          f"{agg['mean_pass_at_10']:<10.3f}")

    # By difficulty
    print("\nBy difficulty:")
    for diff, stats in results['by_difficulty'].items():
        print(f"  {diff:<10} compile={stats['mean_compilation_rate']:.1%}  "
              f"pass@1={stats['mean_pass_at_1']:.3f}  "
              f"({stats['num_problems']} problems)")

    print(f"\nTime: {results['elapsed_seconds']:.1f}s")
    print("=" * 70)


def run_compile_only(model, tokenizer, problems_path: str, device: str,
                     num_prompts: int = 5, num_samples: int = 10,
                     temperature: float = 0.8):
    """Quick compilation rate check: generate completions and check if they compile."""
    problems = load_problems(problems_path, max_problems=num_prompts)
    bench = CppBenchmark(model, tokenizer, device=device)

    total = 0
    compiled = 0
    print(f"\nCompilation Rate Check ({num_prompts} prompts x {num_samples} samples)")
    print("-" * 50)

    for p in problems:
        completions = bench.generate(
            p['prompt'], max_tokens=512,
            temperature=temperature, num_samples=num_samples
        )
        prob_compiled = 0
        for comp in completions:
            full_source = p['prompt'] + comp + "\n\n" + p['test']
            result = bench.compile_and_run(full_source)
            total += 1
            if result.compiled:
                compiled += 1
                prob_compiled += 1

        print(f"  {p['name']:<25} {prob_compiled}/{num_samples} compiled")

    rate = compiled / total if total > 0 else 0
    print("-" * 50)
    print(f"  Overall: {compiled}/{total} = {rate:.1%} compilation rate")
    return rate


def main():
    parser = argparse.ArgumentParser(description="C++ code generation benchmark")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (model_XXXXXX.pt)')
    parser.add_argument('--problems', type=str, default='data/cpp_bench.jsonl',
                        help='Path to benchmark problems JSONL')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of completions per problem')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature')
    parser.add_argument('--max_tokens', type=int, default=512,
                        help='Max tokens per completion')
    parser.add_argument('--difficulty', type=str, default=None,
                        choices=['easy', 'medium', 'hard'],
                        help='Filter problems by difficulty')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-sample details')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results JSON to this path')
    parser.add_argument('--quick', action='store_true',
                        help='Quick smoke test: 3 easy problems, 2 samples')
    parser.add_argument('--compile-only', action='store_true',
                        help='Only check compilation rate (faster)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify benchmark problems compile (no model needed)')

    args = parser.parse_args()

    # Verify mode: just check problems compile with reference solutions
    if args.verify:
        ok = verify_problems(args.problems)
        sys.exit(0 if ok else 1)

    # Find checkpoint
    if args.checkpoint is None:
        # Default: use d12 checkpoint
        import nanochat.common
        base_dir = nanochat.common.get_base_dir()
        ckpt_dir = os.path.join(base_dir, "base_checkpoints", "d12")
        if os.path.exists(ckpt_dir):
            # Find latest step
            from nanochat.checkpoint_manager import find_last_step
            step = find_last_step(ckpt_dir)
            args.checkpoint = os.path.join(ckpt_dir, f"model_{step:06d}.pt")
            print(f"Auto-detected checkpoint: {args.checkpoint}")
        else:
            print("ERROR: No checkpoint specified and no d12 checkpoint found.")
            print("Usage: python -m scripts.eval_cpp --checkpoint /path/to/model_XXXXXX.pt")
            sys.exit(1)

    # Quick mode overrides
    if args.quick:
        args.num_samples = 2
        args.difficulty = 'easy'

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    model, tokenizer, meta = load_model_and_tokenizer(args.checkpoint, device)
    print(f"Model loaded (step={meta.get('step', '?')}, config={meta.get('model_config', {})})")

    # Compile-only mode
    if args.compile_only:
        run_compile_only(model, tokenizer, args.problems, device,
                         num_samples=args.num_samples,
                         temperature=args.temperature)
        return

    # Load problems
    problems = load_problems(args.problems, difficulty=args.difficulty,
                             max_problems=3 if args.quick else None)
    print(f"Loaded {len(problems)} problems" +
          (f" (difficulty={args.difficulty})" if args.difficulty else ""))

    # Run benchmark
    bench = CppBenchmark(model, tokenizer, device=device)
    results = bench.run_benchmark(
        problems,
        num_samples=args.num_samples,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
    )

    # Print results
    print_results(results)

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
