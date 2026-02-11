"""
C++ code generation benchmark evaluator.

Evaluates a nanochat model on C++ coding problems by:
1. Generating code completions from function signature prompts
2. Compiling each completion with g++ (C++17)
3. Running the compiled binary to check correctness via assertions
4. Computing pass@k metrics using the unbiased estimator from the Codex paper

Usage (standalone test):
    python -m nanochat.cpp_eval

Usage (from eval script):
    from nanochat.cpp_eval import CppBenchmark
    bench = CppBenchmark(model, tokenizer, device='cuda')
    results = bench.run_benchmark(problems)
"""

import json
import math
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class CompileResult:
    """Result of compiling and running a C++ program."""
    compiled: bool
    passed: bool
    compiler_output: str = ""
    runtime_output: str = ""
    error: str = ""
    timeout: bool = False


@dataclass
class ProblemResult:
    """Result of evaluating a single problem."""
    name: str
    difficulty: str
    num_samples: int
    num_compiled: int
    num_passed: int
    pass_at_1: float = 0.0
    pass_at_10: float = 0.0
    compilation_rate: float = 0.0
    completions: list = field(default_factory=list)


class CppBenchmark:
    """Run HumanEval-style C++ problems against a nanochat model."""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def generate(self, prompt: str, max_tokens: int = 512,
                 temperature: float = 0.8, top_k: int = 50,
                 num_samples: int = 1, stop_tokens: Optional[list] = None) -> list[str]:
        """Generate code completions for a given prompt.

        Returns a list of completion strings (just the generated part, not the prompt).
        """
        completions = []
        prompt_ids = self.tokenizer.encode(prompt)

        for i in range(num_samples):
            generated_ids = []
            seed = 42 + i  # different seed per sample
            for token in self.model.generate(
                list(prompt_ids), max_tokens=max_tokens,
                temperature=temperature, top_k=top_k, seed=seed
            ):
                generated_ids.append(token)
                # Stop on common C++ function-end heuristics
                text_so_far = self.tokenizer.decode(generated_ids)
                # Stop if we see a second function definition or main()
                if '\nint main(' in text_so_far or '\nvoid main(' in text_so_far:
                    # Trim to before main
                    idx = text_so_far.find('\nint main(')
                    if idx == -1:
                        idx = text_so_far.find('\nvoid main(')
                    text_so_far = text_so_far[:idx]
                    completions.append(text_so_far)
                    break
                # Stop if we see a class definition after a function body
                if text_so_far.count('\n}') >= 2 and len(text_so_far) > 50:
                    # Multiple top-level closing braces likely means we went past the function
                    # Keep up to and including the first closing brace at column 0
                    lines = text_so_far.split('\n')
                    result_lines = []
                    brace_count = 0
                    found_end = False
                    for line in lines:
                        result_lines.append(line)
                        if line.strip() == '}':
                            brace_count += 1
                            if brace_count >= 1:
                                found_end = True
                                break
                    if found_end:
                        completions.append('\n'.join(result_lines))
                        break
            else:
                # Didn't break early, use full generation
                completions.append(self.tokenizer.decode(generated_ids))

        return completions

    @staticmethod
    def compile_and_run(full_source: str, timeout: float = 10.0) -> CompileResult:
        """Compile a C++ source string with g++ and run the resulting binary.

        Args:
            full_source: Complete C++ source code (includes + function + test main)
            timeout: Max seconds for compilation + execution

        Returns:
            CompileResult with compilation and execution status
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "solution.cpp")
            bin_path = os.path.join(tmpdir, "solution")

            with open(src_path, 'w') as f:
                f.write(full_source)

            # Compile
            try:
                compile_proc = subprocess.run(
                    ["g++", "-std=c++17", "-O1", "-o", bin_path, src_path],
                    capture_output=True, text=True, timeout=timeout
                )
            except subprocess.TimeoutExpired:
                return CompileResult(
                    compiled=False, passed=False,
                    error="Compilation timed out", timeout=True
                )

            if compile_proc.returncode != 0:
                return CompileResult(
                    compiled=False, passed=False,
                    compiler_output=compile_proc.stderr
                )

            # Run
            try:
                run_proc = subprocess.run(
                    [bin_path],
                    capture_output=True, text=True, timeout=timeout
                )
            except subprocess.TimeoutExpired:
                return CompileResult(
                    compiled=True, passed=False,
                    error="Execution timed out", timeout=True
                )

            passed = run_proc.returncode == 0
            return CompileResult(
                compiled=True, passed=passed,
                runtime_output=run_proc.stdout,
                error=run_proc.stderr if not passed else ""
            )

    @staticmethod
    def pass_at_k(n: int, c: int, k: int) -> float:
        """Unbiased estimator of pass@k.

        From "Evaluating Large Language Models Trained on Code" (Chen et al., 2021).

        Args:
            n: total number of samples
            c: number of correct samples
            k: k in pass@k

        Returns:
            Estimated probability that at least one of k samples passes.
        """
        if n - c < k:
            return 1.0
        return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))

    def evaluate_problem(self, problem: dict, num_samples: int = 10,
                         temperature: float = 0.8, max_tokens: int = 512,
                         verbose: bool = False) -> ProblemResult:
        """Evaluate a single C++ problem.

        Args:
            problem: dict with 'name', 'prompt', 'test', 'difficulty'
            num_samples: number of completions to generate
            temperature: sampling temperature
            max_tokens: max tokens per completion
            verbose: print per-sample details

        Returns:
            ProblemResult with pass@k metrics
        """
        name = problem['name']
        prompt = problem['prompt']
        test_code = problem['test']
        difficulty = problem.get('difficulty', 'unknown')

        # Generate completions
        completions = self.generate(
            prompt, max_tokens=max_tokens,
            temperature=temperature, num_samples=num_samples
        )

        num_compiled = 0
        num_passed = 0
        completion_details = []

        for i, completion in enumerate(completions):
            # Build full source: prompt + completion + test
            full_source = prompt + completion + "\n\n" + test_code

            result = self.compile_and_run(full_source)

            if result.compiled:
                num_compiled += 1
            if result.passed:
                num_passed += 1

            if verbose:
                status = "PASS" if result.passed else ("COMPILE_FAIL" if not result.compiled else "RUNTIME_FAIL")
                print(f"  Sample {i+1}/{num_samples}: {status}")
                if not result.compiled:
                    # Show first 3 lines of compiler error
                    err_lines = result.compiler_output.strip().split('\n')[:3]
                    for line in err_lines:
                        print(f"    {line}")

            completion_details.append({
                'completion': completion[:200],  # truncate for storage
                'compiled': result.compiled,
                'passed': result.passed,
                'error': result.error[:200] if result.error else '',
            })

        # Compute pass@k
        n = num_samples
        c = num_passed
        p1 = self.pass_at_k(n, c, 1) if n >= 1 else 0.0
        p10 = self.pass_at_k(n, c, 10) if n >= 10 else 0.0
        comp_rate = num_compiled / n if n > 0 else 0.0

        return ProblemResult(
            name=name,
            difficulty=difficulty,
            num_samples=n,
            num_compiled=num_compiled,
            num_passed=num_passed,
            pass_at_1=p1,
            pass_at_10=p10,
            compilation_rate=comp_rate,
            completions=completion_details,
        )

    def run_benchmark(self, problems: list[dict], num_samples: int = 10,
                      temperature: float = 0.8, max_tokens: int = 512,
                      verbose: bool = False) -> dict:
        """Run the full benchmark over a list of problems.

        Args:
            problems: list of problem dicts
            num_samples: completions per problem
            temperature: sampling temperature
            max_tokens: max tokens per completion
            verbose: print per-problem details

        Returns:
            dict with per-problem results and aggregate metrics
        """
        results = []
        t0 = time.time()

        for i, problem in enumerate(problems):
            name = problem['name']
            print(f"[{i+1}/{len(problems)}] Evaluating: {name} ({problem.get('difficulty', '?')})...")
            result = self.evaluate_problem(
                problem, num_samples=num_samples,
                temperature=temperature, max_tokens=max_tokens,
                verbose=verbose,
            )
            results.append(result)
            print(f"  => compile_rate={result.compilation_rate:.0%}, "
                  f"pass@1={result.pass_at_1:.3f}, "
                  f"passed={result.num_passed}/{result.num_samples}")

        elapsed = time.time() - t0

        # Aggregate
        total_samples = sum(r.num_samples for r in results)
        total_compiled = sum(r.num_compiled for r in results)
        total_passed = sum(r.num_passed for r in results)

        # Per-difficulty breakdown
        by_difficulty = {}
        for r in results:
            d = r.difficulty
            if d not in by_difficulty:
                by_difficulty[d] = {'pass_at_1': [], 'compilation_rate': []}
            by_difficulty[d]['pass_at_1'].append(r.pass_at_1)
            by_difficulty[d]['compilation_rate'].append(r.compilation_rate)

        difficulty_summary = {}
        for d, vals in by_difficulty.items():
            difficulty_summary[d] = {
                'mean_pass_at_1': sum(vals['pass_at_1']) / len(vals['pass_at_1']),
                'mean_compilation_rate': sum(vals['compilation_rate']) / len(vals['compilation_rate']),
                'num_problems': len(vals['pass_at_1']),
            }

        return {
            'problems': [
                {
                    'name': r.name,
                    'difficulty': r.difficulty,
                    'num_samples': r.num_samples,
                    'num_compiled': r.num_compiled,
                    'num_passed': r.num_passed,
                    'pass_at_1': r.pass_at_1,
                    'pass_at_10': r.pass_at_10,
                    'compilation_rate': r.compilation_rate,
                }
                for r in results
            ],
            'aggregate': {
                'num_problems': len(results),
                'total_samples': total_samples,
                'total_compiled': total_compiled,
                'total_passed': total_passed,
                'overall_compilation_rate': total_compiled / total_samples if total_samples > 0 else 0,
                'mean_pass_at_1': sum(r.pass_at_1 for r in results) / len(results) if results else 0,
                'mean_pass_at_10': sum(r.pass_at_10 for r in results) / len(results) if results else 0,
            },
            'by_difficulty': difficulty_summary,
            'elapsed_seconds': elapsed,
        }


def verify_problems(problems_path: str) -> bool:
    """Verify that all benchmark problems have valid test code by compiling
    them with a known-correct solution stub (empty, to just check syntax of tests).

    Actually, we compile the test code with a trivial prompt to check the test harness compiles.
    """
    with open(problems_path, 'r') as f:
        problems = [json.loads(line) for line in f if line.strip()]

    print(f"Verifying {len(problems)} problems from {problems_path}")
    all_ok = True
    for p in problems:
        # Build a source that includes the prompt (signature) + a minimal body + test
        # We can't easily auto-generate correct solutions, so just check compilation
        # of the test harness with a stub
        name = p['name']
        full = p['prompt'] + p.get('reference_solution', '// stub\n') + "\n\n" + p['test']
        result = CppBenchmark.compile_and_run(full)
        status = "PASS" if result.passed else ("COMPILE_FAIL" if not result.compiled else "RUNTIME_FAIL")
        if not result.passed:
            all_ok = False
            print(f"  FAIL [{status}]: {name}")
            if result.compiler_output:
                for line in result.compiler_output.strip().split('\n')[:5]:
                    print(f"    {line}")
            if result.error:
                print(f"    Runtime: {result.error[:200]}")
        else:
            print(f"  OK: {name}")

    return all_ok


def _truncate_to_function_body(completion: str) -> str:
    """Truncate generated code to just the function body.

    The prompt ends with an opening '{', so the completion should contain
    the body and closing '}'. We track brace depth and stop when it returns to 0.
    Also handles class/struct completions that may have multiple methods.
    """
    depth = 1  # The prompt's opening '{' starts us at depth 1
    result = []
    i = 0
    in_string = False
    string_char = None
    in_line_comment = False
    in_block_comment = False

    while i < len(completion):
        c = completion[i]

        # Track comments
        if not in_string and not in_block_comment and i + 1 < len(completion):
            if c == '/' and completion[i + 1] == '/':
                in_line_comment = True
            elif c == '/' and completion[i + 1] == '*':
                in_block_comment = True
        if in_line_comment and c == '\n':
            in_line_comment = False
        if in_block_comment and c == '*' and i + 1 < len(completion) and completion[i + 1] == '/':
            in_block_comment = False
            result.append(c)
            i += 1
            result.append(completion[i])
            i += 1
            continue

        # Track strings
        if not in_line_comment and not in_block_comment:
            if c in ('"', "'") and not in_string:
                in_string = True
                string_char = c
            elif c == string_char and in_string:
                # Check for escape
                num_backslashes = 0
                j = len(result) - 1
                while j >= 0 and result[j] == '\\':
                    num_backslashes += 1
                    j -= 1
                if num_backslashes % 2 == 0:
                    in_string = False

        # Track braces (only outside strings and comments)
        if not in_string and not in_line_comment and not in_block_comment:
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    result.append(c)
                    # Include any trailing newline
                    if i + 1 < len(completion) and completion[i + 1] == '\n':
                        result.append('\n')
                    return ''.join(result)

        result.append(c)
        i += 1

    return ''.join(result)


def evaluate_cpp_model(model, tokenizer, device, problems_path="data/cpp_bench.jsonl",
                       max_tokens=512, temperature=0.0):
    """Evaluate a model on C++ coding problems during training.

    Lightweight function designed for periodic evaluation in the training loop.
    Uses Engine for generation and CppBenchmark.compile_and_run() for testing.

    Args:
        model: GPT model instance
        tokenizer: tokenizer instance
        device: torch device
        problems_path: path to JSONL file with C++ problems
        max_tokens: max tokens to generate per problem
        temperature: sampling temperature (0.0 = greedy)

    Returns:
        dict with cpp_metric, cpp_compile_rate, cpp_pass_rate, cpp_by_difficulty, cpp_problems
    """
    from nanochat.engine import Engine
    from nanochat.common import print0

    # Load problems
    if not os.path.exists(problems_path):
        print0(f"WARNING: C++ benchmark not found at {problems_path}, skipping eval")
        return {"cpp_metric": 0.0, "cpp_compile_rate": 0.0, "cpp_pass_rate": 0.0}

    with open(problems_path, 'r') as f:
        problems = [json.loads(line) for line in f if line.strip()]

    print0(f"C++ eval: {len(problems)} problems, max_tokens={max_tokens}, temp={temperature}")

    engine = Engine(model, tokenizer)
    problem_results = []
    t0 = time.time()

    for i, problem in enumerate(problems):
        name = problem['name']
        prompt = problem['prompt']
        test_code = problem['test']
        difficulty = problem.get('difficulty', 'unknown')

        # Tokenize prompt and generate completion
        tokens = tokenizer(prompt, prepend="<|bos|>")
        prompt_len = len(tokens)

        try:
            result_tokens, _ = engine.generate_batch(
                tokens, num_samples=1, max_tokens=max_tokens, temperature=temperature
            )
            # Extract completion (tokens beyond the prompt)
            completion_tokens = result_tokens[0][prompt_len:]
            completion = tokenizer.decode(completion_tokens)
            # Truncate to just the function body to avoid duplicate definitions
            completion = _truncate_to_function_body(completion)
        except Exception as e:
            print0(f"  [{i+1}/{len(problems)}] {name}: generation error: {e}")
            problem_results.append({
                "name": name, "difficulty": difficulty,
                "compiled": False, "passed": False, "error": str(e)
            })
            continue

        # Assemble full source and compile+run
        full_source = prompt + completion + "\n\n" + test_code
        result = CppBenchmark.compile_and_run(full_source)

        status = "PASS" if result.passed else ("COMPILE_FAIL" if not result.compiled else "RUNTIME_FAIL")
        print0(f"  [{i+1}/{len(problems)}] {name} ({difficulty}): {status}")

        problem_results.append({
            "name": name, "difficulty": difficulty,
            "compiled": result.compiled, "passed": result.passed,
        })

    elapsed = time.time() - t0

    # Aggregate metrics
    num_problems = len(problem_results)
    num_compiled = sum(1 for r in problem_results if r["compiled"])
    num_passed = sum(1 for r in problem_results if r["passed"])
    compile_rate = num_compiled / num_problems if num_problems > 0 else 0.0
    pass_rate = num_passed / num_problems if num_problems > 0 else 0.0

    # Per-difficulty breakdown
    by_difficulty = {}
    for r in problem_results:
        d = r["difficulty"]
        if d not in by_difficulty:
            by_difficulty[d] = {"compiled": 0, "passed": 0, "total": 0}
        by_difficulty[d]["total"] += 1
        if r["compiled"]:
            by_difficulty[d]["compiled"] += 1
        if r["passed"]:
            by_difficulty[d]["passed"] += 1

    diff_summary = {}
    for d, vals in by_difficulty.items():
        diff_summary[d] = {
            "compile": vals["compiled"] / vals["total"],
            "pass": vals["passed"] / vals["total"],
            "n": vals["total"],
        }

    print0(f"C++ eval done in {elapsed:.1f}s: compile={compile_rate:.1%}, pass={pass_rate:.1%}")
    for d in ["easy", "medium", "hard"]:
        if d in diff_summary:
            s = diff_summary[d]
            print0(f"  {d}: compile={s['compile']:.1%}, pass={s['pass']:.1%} ({s['n']} problems)")

    return {
        "cpp_metric": pass_rate,
        "cpp_compile_rate": compile_rate,
        "cpp_pass_rate": pass_rate,
        "cpp_by_difficulty": diff_summary,
        "cpp_problems": problem_results,
        "cpp_eval_time": elapsed,
    }
