"""
Evaluate the tool-call SFT model on C++ code generation benchmarks.

Unlike eval_cpp.py (which uses raw model.generate for code completion),
this script uses the Engine with ToolRuntime to support the full agent loop:
  - Model emits <THOUGHT_START>...<THOUGHT_END> (chain of thought)
  - Model emits <QUERY_TOOL> tool_call(...) <CODE_END> (tool use)
  - Engine injects <TOOL_RESULT> result <CODE_END> back into the generation
  - Model emits <CODE_START> final_code <CODE_END> (final answer)

The SFT model was trained on 5 data types:
  - docstring_search: think -> search -> think -> code
  - diff_compile: think -> compile -> see error -> think -> fix
  - humaneval_ask: think -> ask -> think -> solve
  - run_code: think -> run code -> observe output -> answer
  - no_tool: direct code generation

Usage:
    NANOCHAT_CPP_TOKENIZER=1 .venv/bin/python -m scripts.eval_agent \\
        --checkpoint ~/.cache/nanochat/sft_checkpoints/d16/model_022606.pt \\
        --num_samples 2 --temperature 0.6

    # Quick smoke test:
    NANOCHAT_CPP_TOKENIZER=1 .venv/bin/python -m scripts.eval_agent \\
        --checkpoint ~/.cache/nanochat/sft_checkpoints/d16/model_022606.pt \\
        --quick --verbose
"""

import argparse
import json
import math
import os
import re
import sys
import time

import torch

from nanochat.cpp_eval import CppBenchmark, CompileResult
from nanochat.engine import Engine
from nanochat.tool_runtime import ToolRuntime


def load_problems(path: str, difficulty: str = None, max_problems: int = None) -> list[dict]:
    """Load benchmark problems from JSONL file."""
    with open(path, "r") as f:
        problems = [json.loads(line) for line in f if line.strip()]
    if difficulty:
        problems = [p for p in problems if p.get("difficulty") == difficulty]
    if max_problems:
        problems = problems[:max_problems]
    return problems


def load_model_and_tokenizer(checkpoint_path: str, device: str):
    """Load model from a checkpoint .pt file path. Returns (model, tokenizer, meta)."""
    from nanochat.checkpoint_manager import build_model

    basename = os.path.basename(checkpoint_path)
    checkpoint_dir = os.path.dirname(checkpoint_path)

    match = re.match(r"model_(\d+)\.pt", basename)
    if not match:
        raise ValueError(f"Cannot parse step from checkpoint filename: {basename}")
    step = int(match.group(1))

    model, tokenizer, meta = build_model(checkpoint_dir, step, torch.device(device), phase="eval")
    return model, tokenizer, meta


def format_sft_prompt(tokenizer, problem_prompt: str) -> list[int]:
    """Format a benchmark problem prompt into the SFT token format.

    Format: <BOS>
    // Implement the following:
    {prompt}
    <THOUGHT_START>

    This primes the model to begin its chain-of-thought reasoning.
    """
    bos = tokenizer.get_bos_token_id()
    thought_start = tokenizer.encode_special("<THOUGHT_START>")

    # Build the prompt text (between BOS and THOUGHT_START)
    prompt_text = f"\n// Implement the following:\n{problem_prompt}\n"
    prompt_ids = tokenizer.encode(prompt_text)

    # Assemble: BOS + prompt text + THOUGHT_START
    tokens = [bos] + prompt_ids
    if thought_start is not None:
        tokens.append(thought_start)
    return tokens


def extract_code_block(tokenizer, token_ids: list[int]) -> str:
    """Extract the best <CODE_START>...<CODE_END> block from generated tokens.

    Strategy: find all code blocks, pick the longest non-trivial one (the model
    often generates multiple blocks; earlier ones tend to be better before
    the generation degenerates into repetitions).

    If no CODE_START/CODE_END markers are found, falls back to decoding
    everything after the last THOUGHT_END as the code.
    """
    code_start = tokenizer.encode_special("<CODE_START>")
    code_end = tokenizer.encode_special("<CODE_END>")

    # Find all <CODE_START>...<CODE_END> spans
    code_blocks = []
    i = 0
    while i < len(token_ids):
        if token_ids[i] == code_start:
            j = i + 1
            while j < len(token_ids) and token_ids[j] != code_end:
                j += 1
            if j < len(token_ids):
                block_tokens = token_ids[i + 1:j]
                if len(block_tokens) > 0:
                    code_blocks.append(block_tokens)
            i = j + 1
        else:
            i += 1

    if code_blocks:
        # Strategy: pick the first code block that looks like a complete function
        # (contains a closing brace), preferring longer blocks.
        # The model tends to generate good code early, then degenerate.
        for block in code_blocks:
            decoded = tokenizer.decode(block).strip()
            if "}" in decoded and len(decoded) > 10:
                return decoded
        # Fallback: pick the longest block
        best = max(code_blocks, key=len)
        return tokenizer.decode(best).strip()

    # Fallback: try to find content after the last THOUGHT_END
    thought_end = tokenizer.encode_special("<THOUGHT_END>")
    if thought_end is not None:
        last_te = -1
        for i, t in enumerate(token_ids):
            if t == thought_end:
                last_te = i
        if last_te >= 0:
            remaining = token_ids[last_te + 1:]
            # Filter out special tokens (EOS, etc.)
            eos = tokenizer.encode_special("<EOS>")
            remaining = [t for t in remaining if t != eos and t != code_start and t != code_end]
            if remaining:
                return tokenizer.decode(remaining).strip()

    # Last resort: decode everything
    return tokenizer.decode(token_ids).strip()


def fix_tokenizer_spacing(code: str) -> str:
    """Fix BERT-style tokenizer spacing artifacts in decoded C++ code.

    The CppTokenizer's decode inserts spaces around underscores (since _ is
    a separate token), producing e.g. 'is _ prime' instead of 'is_prime'.
    This fixes those artifacts to produce compilable C++ code.
    """
    # Fix spaces around underscores: "is _ prime" -> "is_prime"
    # Match: word_char SPACE _ SPACE word_char
    code = re.sub(r'(\w) _ (\w)', r'\1_\2', code)
    # Also handle: word_char _ SPACE word_char  and  word_char SPACE _ word_char
    code = re.sub(r'(\w)_ (\w)', r'\1_\2', code)
    code = re.sub(r'(\w) _(\w)', r'\1_\2', code)
    # Handle leading underscore: "_ name" -> "_name"
    code = re.sub(r'(?<!\w)_ (\w)', r'_\1', code)
    # Handle trailing underscore: "name _" at end of identifier part
    code = re.sub(r'(\w) _(?!\w)', r'\1_', code)
    # Fix double spaces that may result
    code = re.sub(r'  +', ' ', code)
    # Remove any special token text that leaked through decoding
    for tag in ["<THOUGHT_START>", "<THOUGHT_END>", "<CODE_START>", "<CODE_END>",
                "<QUERY_TOOL>", "<TOOL_RESULT>", "<BOS>", "<EOS>"]:
        code = code.replace(tag, "")
    return code


def strip_duplicate_signature(code: str, prompt: str) -> str:
    """If the generated code re-declares the function signature that is
    already in prompt, strip it so we don't get a redefinition error.

    The model often generates the full function (signature + body) inside
    <CODE_START>, but the prompt already has the signature with the opening brace.
    We detect this overlap and return only the function body.
    """
    # Normalize whitespace for comparison
    code_stripped = code.strip()
    prompt_stripped = prompt.strip()

    # Get the last line of the prompt (typically the function signature opening)
    prompt_lines = prompt_stripped.split("\n")
    # Find the line with the opening brace of the function
    sig_line = ""
    for line in reversed(prompt_lines):
        line_s = line.strip()
        if line_s and not line_s.startswith("//") and not line_s.startswith("#"):
            sig_line = line_s
            break

    if not sig_line:
        return code

    # Extract the function name from the signature line for matching
    # e.g. "bool is_prime(int n) {" -> "is_prime"
    func_match = re.search(r'\b(\w+)\s*\(', sig_line)
    if not func_match:
        return code

    func_name = func_match.group(1)

    # Check if the generated code starts with a line containing this function name
    code_lines = code_stripped.split("\n")
    for idx, line in enumerate(code_lines):
        line_s = line.strip()
        if not line_s or line_s.startswith("//"):
            continue
        # Check if this line contains the function signature
        if func_name in line_s and "(" in line_s and "{" in line_s:
            # This line is a duplicate signature - skip it and return the rest
            remaining = "\n".join(code_lines[idx + 1:])
            return remaining.strip()
        else:
            # First non-comment/empty line doesn't contain the signature
            break

    return code


# Module-level cache for detected device type
_cached_device_type = None


def _get_device_type():
    """Get device type, caching to avoid repeated detection prints."""
    global _cached_device_type
    if _cached_device_type is None:
        from nanochat.common import autodetect_device_type
        _cached_device_type = autodetect_device_type()
    return _cached_device_type


def generate_with_tools(engine, tokenizer, prompt_tokens, temperature=0.6,
                        top_k=50, max_tokens=1024, seed=42, verbose=False):
    """Run the Engine to generate a single sample with tool-call support.

    Returns the full token sequence (prompt + generated).
    """
    from contextlib import nullcontext
    device_type = _get_device_type()
    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        if device_type == "cuda"
        else nullcontext()
    )

    generated_tokens = list(prompt_tokens)  # start with prompt
    gen_kwargs = {
        "num_samples": 1,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "seed": seed,
    }

    if verbose:
        print("    [GEN] ", end="", flush=True)

    with autocast_ctx:
        for token_column, token_masks in engine.generate(prompt_tokens, **gen_kwargs):
            token = token_column[0]
            generated_tokens.append(token)
            if verbose:
                token_text = tokenizer.decode([token])
                print(token_text, end="", flush=True)

    if verbose:
        print()  # newline after generation

    return generated_tokens


def evaluate_problem_agent(engine, tokenizer, problem, num_samples=2,
                           temperature=0.6, max_tokens=1024, verbose=False):
    """Evaluate a single problem using the agent (Engine + tools).

    Returns a dict with problem-level results.
    """
    name = problem["name"]
    prompt = problem["prompt"]
    test_code = problem["test"]
    difficulty = problem.get("difficulty", "unknown")

    # Build the SFT prompt tokens
    prompt_tokens = format_sft_prompt(tokenizer, prompt)

    if verbose:
        prompt_text = tokenizer.decode(prompt_tokens)
        print(f"  Prompt ({len(prompt_tokens)} tokens): {prompt_text[:200]}...")

    num_compiled = 0
    num_passed = 0
    completions = []

    for i in range(num_samples):
        seed = 42 + i
        if verbose:
            print(f"  Sample {i+1}/{num_samples} (seed={seed}):")

        # Generate with tool support
        full_tokens = generate_with_tools(
            engine, tokenizer, prompt_tokens,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            verbose=verbose,
        )

        # Extract the code block
        # Only look at generated tokens (after prompt)
        gen_tokens = full_tokens[len(prompt_tokens):]
        code = extract_code_block(tokenizer, gen_tokens)

        # Fix tokenizer spacing artifacts (e.g. "is _ prime" -> "is_prime")
        code = fix_tokenizer_spacing(code)
        # Strip duplicate function signature if the model re-declared it
        code = strip_duplicate_signature(code, prompt)

        if verbose:
            print(f"    [CODE] ({len(code)} chars): {code[:200]}...")

        # Build full source: original prompt (C++ signature) + generated code + test
        full_source = prompt + code + "\n\n" + test_code

        # Compile and run
        result = CppBenchmark.compile_and_run(full_source)

        if result.compiled:
            num_compiled += 1
        if result.passed:
            num_passed += 1

        status = "PASS" if result.passed else ("COMPILE_FAIL" if not result.compiled else "RUNTIME_FAIL")
        if verbose:
            print(f"    [{status}]")
            if not result.compiled and result.compiler_output:
                err_lines = result.compiler_output.strip().split("\n")[:3]
                for line in err_lines:
                    print(f"      {line}")
            if result.compiled and not result.passed and result.error:
                print(f"      Runtime error: {result.error[:200]}")

        completions.append({
            "code": code[:300],
            "compiled": result.compiled,
            "passed": result.passed,
            "error": (result.compiler_output or result.error)[:200] if not result.passed else "",
        })

    # Compute pass@k
    n = num_samples
    c = num_passed
    p1 = CppBenchmark.pass_at_k(n, c, 1) if n >= 1 else 0.0
    p10 = CppBenchmark.pass_at_k(n, c, 10) if n >= 10 else 0.0
    comp_rate = num_compiled / n if n > 0 else 0.0

    return {
        "name": name,
        "difficulty": difficulty,
        "num_samples": n,
        "num_compiled": num_compiled,
        "num_passed": num_passed,
        "pass_at_1": p1,
        "pass_at_10": p10,
        "compilation_rate": comp_rate,
        "completions": completions,
    }


def print_results(results: dict):
    """Pretty-print benchmark results."""
    print("\n" + "=" * 75)
    print("Agent Eval: C++ Code Generation with Tool-Call Support")
    print("=" * 75)

    # Per-problem table
    print(f"\n{'Problem':<25} {'Diff':<8} {'Compiled':<12} {'Passed':<12} {'pass@1':<10}")
    print("-" * 75)
    for p in results["problems"]:
        print(
            f"{p['name']:<25} {p['difficulty']:<8} "
            f"{p['num_compiled']}/{p['num_samples']:<9} "
            f"{p['num_passed']}/{p['num_samples']:<9} "
            f"{p['pass_at_1']:<10.3f}"
        )

    # Aggregate
    agg = results["aggregate"]
    print("-" * 75)
    print(
        f"{'OVERALL':<25} {'':8} "
        f"{agg['overall_compilation_rate']:<12.1%} "
        f"{agg['total_passed']}/{agg['total_samples']:<9} "
        f"{agg['mean_pass_at_1']:<10.3f}"
    )

    # By difficulty
    print("\nBy difficulty:")
    for diff, stats in results["by_difficulty"].items():
        print(
            f"  {diff:<10} compile={stats['mean_compilation_rate']:.1%}  "
            f"pass@1={stats['mean_pass_at_1']:.3f}  "
            f"({stats['num_problems']} problems)"
        )

    # Tool usage stats
    if "tool_stats" in results:
        ts = results["tool_stats"]
        print(f"\nTool usage: {ts['total_tool_calls']} tool calls across {ts['problems_with_tools']} problems")

    print(f"\nTime: {results['elapsed_seconds']:.1f}s")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SFT model with tool-call support on C++ benchmarks"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to SFT model checkpoint (model_XXXXXX.pt)",
    )
    parser.add_argument(
        "--problems", type=str, default="data/cpp_bench.jsonl",
        help="Path to benchmark problems JSONL",
    )
    parser.add_argument(
        "--num_samples", type=int, default=5,
        help="Number of completions per problem",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_k", type=int, default=50,
        help="Top-k sampling parameter",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=1024,
        help="Max tokens per generation (including thought + tool calls + code)",
    )
    parser.add_argument(
        "--difficulty", type=str, default=None,
        choices=["easy", "medium", "hard"],
        help="Filter problems by difficulty",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (cuda or cpu)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-sample generation details",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save results JSON to this path",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick smoke test: 2 easy problems, 2 samples each",
    )
    parser.add_argument(
        "--no-tools", action="store_true",
        help="Disable tool runtime (model generates but tools are not executed)",
    )

    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.num_samples = 2
        args.difficulty = "easy"

    # Device setup
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, tokenizer, meta = load_model_and_tokenizer(args.checkpoint, device)
    step = meta.get("step", "?")
    config = meta.get("model_config", {})
    print(f"Model loaded: step={step}, n_layer={config.get('n_layer', '?')}, "
          f"n_embd={config.get('n_embd', '?')}, vocab={config.get('vocab_size', '?')}")

    # Set up tool runtime
    tool_runtime = None
    if not args.no_tools:
        tool_runtime = ToolRuntime(codebase_dir=".")
        print("Tool runtime enabled (search, compile, run, ask, read_file, test)")
    else:
        print("Tool runtime DISABLED (--no-tools)")

    # Create Engine with tool support
    engine = Engine(model, tokenizer, tool_runtime=tool_runtime)
    print("Engine created with tool-call state machine")

    # Load problems
    problems = load_problems(
        args.problems,
        difficulty=args.difficulty,
        max_problems=2 if args.quick else None,
    )
    print(f"Loaded {len(problems)} problems" +
          (f" (difficulty={args.difficulty})" if args.difficulty else ""))

    # Run evaluation
    all_results = []
    t0 = time.time()

    for i, problem in enumerate(problems):
        name = problem["name"]
        diff = problem.get("difficulty", "?")
        print(f"\n[{i+1}/{len(problems)}] {name} ({diff})")

        result = evaluate_problem_agent(
            engine, tokenizer, problem,
            num_samples=args.num_samples,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            verbose=args.verbose,
        )
        all_results.append(result)

        print(f"  => compile_rate={result['compilation_rate']:.0%}, "
              f"pass@1={result['pass_at_1']:.3f}, "
              f"passed={result['num_passed']}/{result['num_samples']}")

    elapsed = time.time() - t0

    # Aggregate results
    total_samples = sum(r["num_samples"] for r in all_results)
    total_compiled = sum(r["num_compiled"] for r in all_results)
    total_passed = sum(r["num_passed"] for r in all_results)

    # Per-difficulty breakdown
    by_difficulty = {}
    for r in all_results:
        d = r["difficulty"]
        if d not in by_difficulty:
            by_difficulty[d] = {"pass_at_1": [], "compilation_rate": []}
        by_difficulty[d]["pass_at_1"].append(r["pass_at_1"])
        by_difficulty[d]["compilation_rate"].append(r["compilation_rate"])

    difficulty_summary = {}
    for d, vals in by_difficulty.items():
        difficulty_summary[d] = {
            "mean_pass_at_1": sum(vals["pass_at_1"]) / len(vals["pass_at_1"]),
            "mean_compilation_rate": sum(vals["compilation_rate"]) / len(vals["compilation_rate"]),
            "num_problems": len(vals["pass_at_1"]),
        }

    results = {
        "problems": all_results,
        "aggregate": {
            "num_problems": len(all_results),
            "total_samples": total_samples,
            "total_compiled": total_compiled,
            "total_passed": total_passed,
            "overall_compilation_rate": total_compiled / total_samples if total_samples > 0 else 0,
            "mean_pass_at_1": (
                sum(r["pass_at_1"] for r in all_results) / len(all_results)
                if all_results else 0
            ),
        },
        "by_difficulty": difficulty_summary,
        "elapsed_seconds": elapsed,
        "config": {
            "checkpoint": args.checkpoint,
            "num_samples": args.num_samples,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "tools_enabled": not args.no_tools,
        },
    }

    # Print results
    print_results(results)

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
