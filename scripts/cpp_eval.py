"""
C++ Code Evaluation for nanochat.

Evaluates C++ code generation using:
1. HumanEval-X C++ (164 problems) - comment/docstring to code completion
2. Metrics:
   - pass@k: Execution-based correctness (compile + run tests)
   - CodeBLEU: Syntax/semantic similarity to canonical solution

Usage:
    python -m scripts.cpp_eval [--checkpoint step] [--max-samples 20]

Requirements:
    pip install codebleu
    g++ must be available for compilation
"""
import os
import sys
import json
import argparse
import tempfile
import subprocess
import random
from pathlib import Path
from typing import Optional
from contextlib import nullcontext
from collections import defaultdict

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type
from nanochat.checkpoint_manager import load_checkpoint, _patch_missing_config_keys, _patch_missing_keys
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import HuggingFaceTokenizer


# -----------------------------------------------------------------------------
# Model Loading (with correct tokenizer)

def load_model_for_eval(model_tag: str, step: int, device):
    """Load model using HuggingFace tokenizer (for 32K vocab models)."""
    import logging
    logger = logging.getLogger(__name__)

    base_dir = get_base_dir()
    checkpoint_dir = os.path.join(base_dir, "base_checkpoints", model_tag)

    logger.info(f"Loading model from {checkpoint_dir} with step {step}")

    model_data, _, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)

    # Clean torch.compile prefix
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

    # Build model config
    model_config_kwargs = meta_data["model_config"]
    _patch_missing_config_keys(model_config_kwargs)
    model_config = GPTConfig(**model_config_kwargs)
    _patch_missing_keys(model_data, model_config)

    # Build model
    with torch.device("meta"):
        model = GPT(model_config)
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)
    model.eval()

    # Load HuggingFace tokenizer (32K vocab)
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    tokenizer = HuggingFaceTokenizer.from_directory(tokenizer_dir)

    # Verify vocab size match
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"], \
        f"Tokenizer vocab {tokenizer.get_vocab_size()} != model vocab {model_config_kwargs['vocab_size']}"

    logger.info(f"Loaded model: {model_config_kwargs['n_layer']} layers, {model_config_kwargs['vocab_size']} vocab")

    return model, tokenizer, meta_data


# -----------------------------------------------------------------------------
# HumanEval-X Data Loading

HUMANEVAL_CPP_PATH = "data/eval/humaneval_cpp.jsonl"
HUMANEVAL_CPP_URL = "https://raw.githubusercontent.com/THUDM/CodeGeeX/main/codegeex/benchmark/humaneval-x/cpp/data/humaneval_cpp.jsonl.gz"


def download_humaneval_cpp():
    """Download HumanEval-X C++ if not present."""
    project_root = Path(__file__).parent.parent
    local_path = project_root / HUMANEVAL_CPP_PATH

    if local_path.exists():
        return local_path

    local_path.parent.mkdir(parents=True, exist_ok=True)
    print0(f"Downloading HumanEval-X C++...")

    import gzip
    import urllib.request

    gz_path = local_path.with_suffix('.jsonl.gz')
    urllib.request.urlretrieve(HUMANEVAL_CPP_URL, gz_path)

    with gzip.open(gz_path, 'rb') as f_in:
        with open(local_path, 'wb') as f_out:
            f_out.write(f_in.read())

    gz_path.unlink()
    print0(f"Downloaded to {local_path}")
    return local_path


def load_humaneval_cpp():
    """Load HumanEval-X C++ dataset."""
    path = download_humaneval_cpp()
    examples = []
    with open(path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


# -----------------------------------------------------------------------------
# Code Generation

def generate_completion(model, tokenizer, prompt: str, max_new_tokens: int = 512,
                        temperature: float = 0.2, top_p: float = 0.95,
                        stop_tokens: list = None) -> str:
    """Generate code completion from the model."""
    device = next(model.parameters()).device

    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Get stop token ids
    stop_ids = set()
    if stop_tokens:
        for st in stop_tokens:
            try:
                st_ids = tokenizer.encode(st)
                if st_ids:
                    stop_ids.add(st_ids[0])
            except:
                pass

    # Generate
    generated = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Check stop conditions
            token_id = next_token.item()
            eos_id = getattr(tokenizer, 'eos_id', getattr(tokenizer, 'bos_token_id', None))
            if token_id in stop_ids or (eos_id is not None and token_id == eos_id):
                break

            generated.append(token_id)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    completion = tokenizer.decode(generated)
    return completion


def extract_function_body(completion: str, prompt: str) -> str:
    """Extract just the function body from completion."""
    # The completion should continue from where prompt ended
    # We need to find where the function ends (matching braces)

    full_code = prompt + completion

    # Find the opening brace of the function
    brace_start = prompt.rfind('{')
    if brace_start == -1:
        # No brace in prompt, look in completion
        brace_start = len(prompt) + completion.find('{')

    # Count braces to find function end
    depth = 0
    in_string = False
    in_char = False
    escape = False

    for i, c in enumerate(full_code[brace_start:], brace_start):
        if escape:
            escape = False
            continue
        if c == '\\':
            escape = True
            continue
        if c == '"' and not in_char:
            in_string = not in_string
        if c == "'" and not in_string:
            in_char = not in_char
        if in_string or in_char:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return full_code[len(prompt):i+1-len(prompt)+1]

    # If no matching brace found, return as-is
    return completion


# -----------------------------------------------------------------------------
# Execution-based Evaluation (pass@k)

def compile_and_run_cpp(code: str, test_code: str, timeout: int = 10) -> tuple[bool, str]:
    """Compile and run C++ code with tests. Returns (passed, error_msg)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "solution.cpp")
        exe_path = os.path.join(tmpdir, "solution")

        # Combine code and tests
        full_code = code + "\n" + test_code

        with open(src_path, 'w') as f:
            f.write(full_code)

        # Compile
        try:
            result = subprocess.run(
                ["g++", "-std=c++17", "-O2", "-o", exe_path, src_path],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                return False, f"Compilation error: {result.stderr[:500]}"
        except subprocess.TimeoutExpired:
            return False, "Compilation timeout"
        except FileNotFoundError:
            return False, "g++ not found"

        # Run
        try:
            result = subprocess.run(
                [exe_path],
                capture_output=True, text=True, timeout=timeout
            )
            if result.returncode != 0:
                return False, f"Runtime error: {result.stderr[:500]}"
            return True, "Passed"
        except subprocess.TimeoutExpired:
            return False, "Runtime timeout"
        except Exception as e:
            return False, f"Runtime exception: {str(e)[:200]}"


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Estimates pass@k from n samples with c correct.
    Uses the unbiased estimator from the Codex paper.
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


# -----------------------------------------------------------------------------
# CodeBLEU Evaluation

def compute_codebleu(predictions: list[str], references: list[str], lang: str = "cpp") -> dict:
    """Compute CodeBLEU score."""
    try:
        from codebleu import calc_codebleu
        result = calc_codebleu(references, predictions, lang=lang, weights=(0.25, 0.25, 0.25, 0.25))
        return result
    except ImportError:
        print0("Warning: codebleu not installed, skipping CodeBLEU")
        return {"codebleu": 0.0}
    except Exception as e:
        print0(f"Warning: CodeBLEU computation failed: {e}")
        return {"codebleu": 0.0}


# -----------------------------------------------------------------------------
# Main Evaluation

def evaluate_humaneval_cpp(model, tokenizer, device,
                           max_samples: int = -1,
                           num_samples_per_task: int = 5,
                           temperature: float = 0.2,
                           max_new_tokens: int = 512) -> dict:
    """
    Evaluate model on HumanEval-X C++.

    Args:
        model: The model to evaluate
        tokenizer: Tokenizer
        device: Device
        max_samples: Max tasks to evaluate (-1 = all 164)
        num_samples_per_task: Samples per task for pass@k estimation
        temperature: Sampling temperature
        max_new_tokens: Max tokens to generate

    Returns:
        Dictionary with pass@1, pass@5, codebleu, etc.
    """
    examples = load_humaneval_cpp()

    if max_samples > 0:
        random.seed(42)
        random.shuffle(examples)
        examples = examples[:max_samples]

    print0(f"\nEvaluating on {len(examples)} HumanEval-X C++ tasks")
    print0(f"Generating {num_samples_per_task} samples per task")
    print0(f"Temperature: {temperature}, Max tokens: {max_new_tokens}\n")

    results_per_task = {}
    all_predictions = []
    all_references = []

    for idx, ex in enumerate(examples):
        task_id = ex['task_id']
        prompt = ex['prompt']
        canonical = ex['canonical_solution']
        test = ex['test']

        print0(f"[{idx+1}/{len(examples)}] {task_id}...", end=' ')

        # Generate multiple samples
        samples = []
        passed_count = 0

        for s in range(num_samples_per_task):
            # Generate completion
            completion = generate_completion(
                model, tokenizer, prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop_tokens=["\n\n\n", "int main", "#include"]
            )

            # Build full code
            full_code = prompt + completion

            # Check if it compiles and passes tests
            passed, error = compile_and_run_cpp(full_code, test)
            if passed:
                passed_count += 1

            samples.append({
                'completion': completion,
                'passed': passed,
                'error': error
            })

        results_per_task[task_id] = {
            'samples': samples,
            'passed_count': passed_count,
            'total': num_samples_per_task
        }

        # Use first sample for CodeBLEU
        all_predictions.append(samples[0]['completion'])
        all_references.append(canonical)

        pass_rate = passed_count / num_samples_per_task
        print0(f"pass@1={int(passed_count>0)}, pass_rate={pass_rate:.2f}")

    # Calculate pass@k estimates
    pass_at_1_list = []
    pass_at_5_list = []

    for task_id, result in results_per_task.items():
        n = result['total']
        c = result['passed_count']
        pass_at_1_list.append(estimate_pass_at_k(n, c, 1))
        if n >= 5:
            pass_at_5_list.append(estimate_pass_at_k(n, c, 5))

    pass_at_1 = np.mean(pass_at_1_list)
    pass_at_5 = np.mean(pass_at_5_list) if pass_at_5_list else 0.0

    # Compute CodeBLEU
    codebleu_result = compute_codebleu(all_predictions, all_references, lang="cpp")

    # Summary
    results = {
        "pass@1": pass_at_1,
        "pass@5": pass_at_5,
        "codebleu": codebleu_result.get("codebleu", 0.0),
        "ngram_match": codebleu_result.get("ngram_match_score", 0.0),
        "syntax_match": codebleu_result.get("syntax_match_score", 0.0),
        "dataflow_match": codebleu_result.get("dataflow_match_score", 0.0),
        "num_tasks": len(examples),
        "samples_per_task": num_samples_per_task,
        "total_passed": sum(1 for r in results_per_task.values() if r['passed_count'] > 0),
    }

    return results


# -----------------------------------------------------------------------------
# CLI

def main():
    parser = argparse.ArgumentParser(description="C++ Code Evaluation")
    parser.add_argument('--checkpoint', type=int, default=None, help='Checkpoint step to load')
    parser.add_argument('--model-tag', type=str, default=None, help='Model tag')
    parser.add_argument('--max-samples', type=int, default=-1, help='Max tasks to evaluate (-1 = all)')
    parser.add_argument('--num-samples', type=int, default=5, help='Samples per task')
    parser.add_argument('--temperature', type=float, default=0.2, help='Sampling temperature')
    parser.add_argument('--max-tokens', type=int, default=512, help='Max tokens to generate')
    args = parser.parse_args()

    # Setup
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    # Load model
    print0("Loading model...")
    model, tokenizer, meta = load_model_for_eval(args.model_tag, args.checkpoint, device)
    step = meta.get('step', 'unknown')
    print0(f"Loaded model at step {step}")

    # Evaluate
    with autocast_ctx:
        results = evaluate_humaneval_cpp(
            model, tokenizer, device,
            max_samples=args.max_samples,
            num_samples_per_task=args.num_samples,
            temperature=args.temperature,
            max_new_tokens=args.max_tokens
        )

    # Print results
    print0("\n" + "="*60)
    print0("C++ Code Evaluation Results")
    print0("="*60)
    print0(f"Model step: {step}")
    print0(f"Tasks evaluated: {results['num_tasks']}")
    print0(f"Samples per task: {results['samples_per_task']}")
    print0("-"*60)
    print0(f"pass@1:          {results['pass@1']:.4f}")
    print0(f"pass@5:          {results['pass@5']:.4f}")
    print0(f"Tasks passed:    {results['total_passed']}/{results['num_tasks']}")
    print0("-"*60)
    print0(f"CodeBLEU:        {results['codebleu']:.4f}")
    print0(f"  N-gram match:  {results['ngram_match']:.4f}")
    print0(f"  Syntax match:  {results['syntax_match']:.4f}")
    print0(f"  Dataflow:      {results['dataflow_match']:.4f}")
    print0("="*60)

    # Save results
    base_dir = get_base_dir()
    output_dir = os.path.join(base_dir, "cpp_eval")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"humaneval_cpp_step{step}.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print0(f"\nResults saved to: {output_path}")

    compute_cleanup()


if __name__ == "__main__":
    main()
