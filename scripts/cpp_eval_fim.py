"""
C++ Code Evaluation using FIM (Fill-in-the-Middle) prompting.

Tests the model's ability to complete C++ functions given:
  - Prefix: docstring + includes + function signature
  - Suffix: closing brace }
  - Task: Fill in the function body (middle)

Format: <FIM_PREFIX> docstring+signature{ <FIM_SUFFIX> } <FIM_MIDDLE> [generate body]

Usage:
    python -m scripts.cpp_eval_fim --model-tag d16_400M_fim_cce_10b --checkpoint 20000 --max-samples 10
"""
import os
import sys
import json
import argparse
import tempfile
import subprocess
import random
from pathlib import Path
from contextlib import nullcontext

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type
from nanochat.checkpoint_manager import load_checkpoint, _patch_missing_config_keys, _patch_missing_keys
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import HuggingFaceTokenizer

# FIM token IDs (from C++ tokenizer)
FIM_PREFIX_ID = 4   # <FIM_PREFIX>
FIM_MIDDLE_ID = 5   # <FIM_MIDDLE>
FIM_SUFFIX_ID = 6   # <FIM_SUFFIX>
EOT_ID = 3          # <EOS>


# -----------------------------------------------------------------------------
# Model Loading

def load_model_for_eval(model_tag: str, step: int, device):
    """Load model using HuggingFace tokenizer (for 32K vocab models)."""
    import logging
    logger = logging.getLogger(__name__)

    base_dir = get_base_dir()
    checkpoint_dir = os.path.join(base_dir, "base_checkpoints", model_tag)

    logger.info(f"Loading model from {checkpoint_dir} with step {step}")

    model_data, _, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

    model_config_kwargs = meta_data["model_config"]
    _patch_missing_config_keys(model_config_kwargs)
    model_config = GPTConfig(**model_config_kwargs)
    _patch_missing_keys(model_data, model_config)

    with torch.device("meta"):
        model = GPT(model_config)
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)
    model.eval()

    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    tokenizer = HuggingFaceTokenizer.from_directory(tokenizer_dir)

    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"], \
        f"Tokenizer vocab {tokenizer.get_vocab_size()} != model vocab {model_config_kwargs['vocab_size']}"

    logger.info(f"Loaded model: {model_config_kwargs['n_layer']} layers, {model_config_kwargs['vocab_size']} vocab")

    return model, tokenizer, meta_data


# -----------------------------------------------------------------------------
# HumanEval-X Data

HUMANEVAL_CPP_PATH = "data/eval/humaneval_cpp.jsonl"

def load_humaneval_cpp():
    """Load HumanEval-X C++ dataset."""
    project_root = Path(__file__).parent.parent
    path = project_root / HUMANEVAL_CPP_PATH

    if not path.exists():
        raise FileNotFoundError(f"HumanEval-X C++ not found at {path}. Run cpp_eval.py first to download.")

    examples = []
    with open(path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


# -----------------------------------------------------------------------------
# FIM Generation

def generate_fim_completion(model, tokenizer, prefix: str, suffix: str,
                            max_new_tokens: int = 512, temperature: float = 0.2,
                            device=None) -> str:
    """
    Generate code using FIM format.

    Format: <FIM_PREFIX> prefix <FIM_SUFFIX> suffix <FIM_MIDDLE> [generate...]
    """
    if device is None:
        device = next(model.parameters()).device

    # Tokenize prefix and suffix
    prefix_tokens = tokenizer.encode(prefix)
    suffix_tokens = tokenizer.encode(suffix)

    # Build FIM input: <FIM_PREFIX> prefix <FIM_SUFFIX> suffix <FIM_MIDDLE>
    input_ids = [FIM_PREFIX_ID] + prefix_tokens + [FIM_SUFFIX_ID] + suffix_tokens + [FIM_MIDDLE_ID]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    generated = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_tensor)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > 0.95
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()

            # Stop on EOT or FIM tokens (indicates end of middle)
            if token_id in (EOT_ID, FIM_PREFIX_ID, FIM_SUFFIX_ID, FIM_MIDDLE_ID):
                break

            generated.append(token_id)
            input_tensor = torch.cat([input_tensor, next_token], dim=1)

            # Also stop if we generate a closing brace (end of function)
            if len(generated) > 5:
                recent = tokenizer.decode(generated[-10:])
                # Stop if we've closed the function
                if recent.count('}') > recent.count('{'):
                    break

    completion = tokenizer.decode(generated)
    return completion


def build_fim_prompt(example: dict) -> tuple[str, str]:
    """
    Build FIM prefix and suffix from HumanEval-X example.

    Prefix: docstring + includes + function signature + {
    Suffix: }

    Returns: (prefix, suffix)
    """
    prompt = example['prompt']

    # The prompt already ends with the function signature and {
    # We want everything up to and including the opening brace as prefix
    prefix = prompt.rstrip()  # Remove trailing whitespace

    # Suffix is just the closing brace
    suffix = "\n}"

    return prefix, suffix


# -----------------------------------------------------------------------------
# Execution-based Evaluation

def compile_and_run_cpp(code: str, test_code: str, timeout: int = 10) -> tuple[bool, str]:
    """Compile and run C++ code with tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "solution.cpp")
        exe_path = os.path.join(tmpdir, "solution")

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
                return False, f"Compilation error: {result.stderr[:300]}"
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
                return False, f"Runtime error: {result.stderr[:300]}"
            return True, "Passed"
        except subprocess.TimeoutExpired:
            return False, "Runtime timeout"
        except Exception as e:
            return False, f"Runtime exception: {str(e)[:200]}"


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator for pass@k."""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


# -----------------------------------------------------------------------------
# CodeBLEU

def compute_codebleu(predictions: list[str], references: list[str]) -> dict:
    """Compute CodeBLEU score."""
    try:
        from codebleu import calc_codebleu
        result = calc_codebleu(references, predictions, lang="cpp", weights=(0.25, 0.25, 0.25, 0.25))
        return result
    except Exception as e:
        print0(f"Warning: CodeBLEU failed: {e}")
        return {"codebleu": 0.0}


# -----------------------------------------------------------------------------
# Main Evaluation

def evaluate_humaneval_fim(model, tokenizer, device,
                           max_samples: int = -1,
                           num_samples_per_task: int = 5,
                           temperature: float = 0.2,
                           max_new_tokens: int = 512,
                           verbose: bool = False) -> dict:
    """
    Evaluate model on HumanEval-X C++ using FIM prompting.
    """
    examples = load_humaneval_cpp()

    if max_samples > 0:
        random.seed(42)
        random.shuffle(examples)
        examples = examples[:max_samples]

    print0(f"\n{'='*60}")
    print0(f"FIM-based C++ Evaluation")
    print0(f"{'='*60}")
    print0(f"Tasks: {len(examples)}")
    print0(f"Samples per task: {num_samples_per_task}")
    print0(f"Temperature: {temperature}")
    print0(f"Format: <FIM_PREFIX> docstring+sig {{ <FIM_SUFFIX> }} <FIM_MIDDLE>")
    print0(f"{'='*60}\n")

    results_per_task = {}
    all_predictions = []
    all_references = []

    for idx, ex in enumerate(examples):
        task_id = ex['task_id']
        canonical = ex['canonical_solution']
        test = ex['test']

        # Build FIM prompt
        prefix, suffix = build_fim_prompt(ex)

        if verbose:
            print0(f"\n[{idx+1}/{len(examples)}] {task_id}")
            print0(f"Prefix (last 100 chars): ...{prefix[-100:]}")
            print0(f"Suffix: {suffix}")
        else:
            print0(f"[{idx+1}/{len(examples)}] {task_id}...", end=' ')

        samples = []
        passed_count = 0

        for s in range(num_samples_per_task):
            # Generate with FIM
            completion = generate_fim_completion(
                model, tokenizer, prefix, suffix,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                device=device
            )

            # Clean up completion - remove extra closing braces
            # The completion should be just the function body
            completion = completion.strip()

            # Build full code: prefix + completion + suffix
            # But we need to handle the braces properly
            full_code = prefix + "\n" + completion

            # If completion doesn't end with }, add the suffix
            if not completion.rstrip().endswith('}'):
                full_code += suffix

            if verbose and s == 0:
                print0(f"Completion: {completion[:200]}...")

            # Test
            passed, error = compile_and_run_cpp(full_code, test)
            if passed:
                passed_count += 1

            samples.append({
                'completion': completion,
                'full_code': full_code,
                'passed': passed,
                'error': error
            })

        results_per_task[task_id] = {
            'samples': samples,
            'passed_count': passed_count,
            'total': num_samples_per_task
        }

        all_predictions.append(samples[0]['completion'])
        all_references.append(canonical)

        pass_rate = passed_count / num_samples_per_task
        status = "PASS" if passed_count > 0 else "FAIL"
        if not verbose:
            print0(f"{status} ({passed_count}/{num_samples_per_task})")

    # Calculate metrics
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

    # CodeBLEU
    codebleu_result = compute_codebleu(all_predictions, all_references)

    results = {
        "method": "FIM",
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
    parser = argparse.ArgumentParser(description="C++ Code Evaluation with FIM")
    parser.add_argument('--model-tag', type=str, required=True, help='Model tag')
    parser.add_argument('--checkpoint', type=int, required=True, help='Checkpoint step')
    parser.add_argument('--max-samples', type=int, default=-1, help='Max tasks (-1 = all)')
    parser.add_argument('--num-samples', type=int, default=5, help='Samples per task')
    parser.add_argument('--temperature', type=float, default=0.2, help='Temperature')
    parser.add_argument('--max-tokens', type=int, default=512, help='Max tokens')
    parser.add_argument('--verbose', action='store_true', help='Show detailed output')
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
        results = evaluate_humaneval_fim(
            model, tokenizer, device,
            max_samples=args.max_samples,
            num_samples_per_task=args.num_samples,
            temperature=args.temperature,
            max_new_tokens=args.max_tokens,
            verbose=args.verbose
        )

    # Print results
    print0("\n" + "="*60)
    print0("FIM-based C++ Evaluation Results")
    print0("="*60)
    print0(f"Model: {args.model_tag} @ step {step}")
    print0(f"Method: FIM prompting")
    print0(f"Tasks: {results['num_tasks']}, Samples/task: {results['samples_per_task']}")
    print0("-"*60)
    print0(f"pass@1:          {results['pass@1']:.4f}")
    print0(f"pass@5:          {results['pass@5']:.4f}")
    print0(f"Tasks passed:    {results['total_passed']}/{results['num_tasks']}")
    print0("-"*60)
    print0(f"CodeBLEU:        {results['codebleu']:.4f}")
    print0(f"  N-gram:        {results['ngram_match']:.4f}")
    print0(f"  Syntax:        {results['syntax_match']:.4f}")
    print0(f"  Dataflow:      {results['dataflow_match']:.4f}")
    print0("="*60)

    # Save
    base_dir = get_base_dir()
    output_dir = os.path.join(base_dir, "cpp_eval")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"humaneval_fim_step{step}.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print0(f"\nSaved to: {output_path}")

    compute_cleanup()


if __name__ == "__main__":
    main()
