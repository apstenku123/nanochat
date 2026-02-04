#!/usr/bin/env python3
"""
Prepare GSPO prompts from HumanEval-X C++ and other code generation benchmarks.

GSPO requires:
- Prompts (function signatures with docstrings)
- Test harnesses to evaluate correctness
- Canonical solutions for reference (optional)

Output format (JSONL):
{
    "prompt_id": "humaneval_cpp/0",
    "prompt": "// Complete the function...",
    "test_harness": "#include ... int main() { assert(...); }",
    "canonical_solution": "...",  # optional
    "metadata": {"source": "humaneval-x", ...}
}

Usage:
    python -m scripts.data.prepare_gspo_prompts \
        --humaneval data/humaneval_cpp/humaneval_cpp.jsonl \
        --output data/gspo_prompts.jsonl
"""

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class GSPOPrompt:
    """GSPO training prompt."""
    prompt_id: str
    prompt: str
    test_harness: str
    canonical_solution: Optional[str] = None
    metadata: Optional[dict] = None


def load_humaneval_cpp(path: str) -> list[GSPOPrompt]:
    """
    Load HumanEval-X C++ problems.

    Output format is pure C++ - the docstring/description is already
    in the prompt as a C++ block comment, and the function signature
    follows. The model just needs to complete the function body.
    """
    prompts = []

    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)
            task_id = data["task_id"]  # e.g., "CPP/0"
            prompt_id = f"humaneval_cpp/{task_id.split('/')[-1]}"

            # The prompt field is already pure C++ with:
            # - Block comment with docstring/examples
            # - #include statements
            # - Function signature ending with {
            prompt = data["prompt"]

            # Test harness is pure C++ with assertions
            test_harness = data["test"]

            # Canonical solution is the function body (no braces)
            canonical = data.get("canonical_solution", "")

            prompts.append(GSPOPrompt(
                prompt_id=prompt_id,
                prompt=prompt,  # Pure C++: docstring + includes + signature{
                test_harness=test_harness,  # Pure C++: main() with assert()
                canonical_solution=canonical,  # Pure C++: function body
                metadata={
                    "source": "humaneval-x",
                    "original_task_id": task_id,
                    "language": "cpp",
                },
            ))

    logger.info(f"Loaded {len(prompts)} HumanEval-X C++ problems")
    return prompts


def load_mbpp_cpp(path: str) -> list[GSPOPrompt]:
    """
    Load MBPP problems (Python) and convert to C++ format.

    Note: This is a basic conversion - for real use, you'd want
    manually translated MBPP problems.
    """
    prompts = []

    # MBPP is Python - we'd need C++ translations
    # For now, skip this and just note it's not available
    logger.warning("MBPP C++ conversion not implemented - skipping")

    return prompts


def create_fim_prompts(base_prompts: list[GSPOPrompt]) -> list[GSPOPrompt]:
    """
    Create FIM-style prompts from completion prompts.

    Uses standard FIM tokens:
        <|fim_prefix|>// docstring + signature{<|fim_suffix|>}<|fim_middle|>

    The model fills in the function body between { and }.
    This is pure C++ - the docstring is already a C++ block comment.
    """
    fim_prompts = []

    for p in base_prompts:
        # Only transform if it has a function signature ending with {
        if not re.search(r'\w+\s+\w+\s*\([^)]*\)\s*\{\s*$', p.prompt):
            continue

        # Create FIM version
        # prefix = everything including the opening {
        # suffix = closing } and newline
        # middle = function body (what the model generates)
        prefix = p.prompt.rstrip()
        suffix = "\n}\n"

        fim_prompt = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"

        fim_prompts.append(GSPOPrompt(
            prompt_id=f"{p.prompt_id}_fim",
            prompt=fim_prompt,
            test_harness=p.test_harness,
            canonical_solution=p.canonical_solution,
            metadata={
                **(p.metadata or {}),
                "fim_style": True,
            },
        ))

    logger.info(f"Created {len(fim_prompts)} FIM-style prompts")
    return fim_prompts


def validate_prompt(prompt: GSPOPrompt) -> bool:
    """Validate that a prompt has required fields and is valid C++."""
    if not prompt.prompt or not prompt.test_harness:
        return False

    # Check for basic C++ structure
    if not any(kw in prompt.prompt for kw in ["#include", "int ", "void ", "bool ", "string ", "vector<", "float ", "double "]):
        return False

    # Check test harness has assertions
    if "assert" not in prompt.test_harness and "EXPECT_" not in prompt.test_harness:
        return False

    return True


def write_prompts(prompts: list[GSPOPrompt], output_path: str):
    """Write prompts to JSONL file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    valid_count = 0
    with open(output_path, "w") as f:
        for p in prompts:
            if not validate_prompt(p):
                logger.warning(f"Skipping invalid prompt: {p.prompt_id}")
                continue

            data = asdict(p)
            f.write(json.dumps(data) + "\n")
            valid_count += 1

    logger.info(f"Wrote {valid_count} prompts to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare GSPO prompts")
    parser.add_argument(
        "--humaneval",
        type=str,
        default="data/humaneval_cpp/humaneval_cpp.jsonl",
        help="Path to HumanEval-X C++ JSONL",
    )
    parser.add_argument(
        "--mbpp",
        type=str,
        default=None,
        help="Path to MBPP C++ JSONL (optional)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/gspo_prompts.jsonl",
        help="Output path for GSPO prompts",
    )
    parser.add_argument(
        "--add-fim",
        action="store_true",
        help="Also create FIM-style prompts",
    )
    args = parser.parse_args()

    all_prompts = []

    # Load HumanEval-X C++
    if args.humaneval and os.path.exists(args.humaneval):
        prompts = load_humaneval_cpp(args.humaneval)
        all_prompts.extend(prompts)
    else:
        logger.warning(f"HumanEval file not found: {args.humaneval}")

    # Load MBPP C++ (if available)
    if args.mbpp and os.path.exists(args.mbpp):
        prompts = load_mbpp_cpp(args.mbpp)
        all_prompts.extend(prompts)

    # Optionally add FIM prompts
    if args.add_fim:
        fim_prompts = create_fim_prompts(all_prompts)
        all_prompts.extend(fim_prompts)

    # Write output
    write_prompts(all_prompts, args.output)

    # Summary
    sources = {}
    for p in all_prompts:
        source = (p.metadata or {}).get("source", "unknown")
        sources[source] = sources.get(source, 0) + 1

    logger.info(f"\nSummary:")
    for source, count in sources.items():
        logger.info(f"  {source}: {count} prompts")
    logger.info(f"  Total: {len(all_prompts)} prompts")


if __name__ == "__main__":
    main()
