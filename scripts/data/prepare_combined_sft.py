#!/usr/bin/env python3
"""
Prepare combined SFT dataset from multiple sources.

Sources:
1. diff_sft.jsonl - Code fixes from git diffs
2. docstring_pairs.jsonl - Function with docstring completion pairs

Output format (JSONL) - PURE C++ SYNTAX:
{
    "text": "// instruction in C++ comment\ncode...",
    "source": "diff_sft|docstring",
    "metadata": {...}
}

The model operates entirely in C++ token space:
- Instructions come as C++ comments (// or /* */)
- Output is raw C++ tokens (spacing handled by formatter)
- Tool calls to other models use comment syntax: // @need: function_name

Usage:
    python -m scripts.data.prepare_combined_sft \
        --diff-sft data/diff_sft.jsonl \
        --docstring-pairs data/docstring_pairs_full.jsonl \
        --output data/combined_sft.jsonl \
        --max-docstring 100000
"""

import argparse
import json
import logging
import os
import random
import re
from typing import Iterator, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences if present."""
    text = re.sub(r'^```(?:cpp|c\+\+|c)?\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?```$', '', text, flags=re.MULTILINE)
    return text.strip()


def format_as_cpp(instruction: str, code: str) -> str:
    """
    Format as pure C++ with instruction in comments.

    Input becomes C++ comment, output is raw C++ code.
    The model learns: comment -> code completion
    """
    # Clean the code
    code = strip_markdown_fences(code)

    # Format instruction as C++ comment block
    instruction_lines = instruction.strip().split('\n')
    comment_block = '\n'.join(f'// {line}' for line in instruction_lines)

    return f"{comment_block}\n{code}"


def extract_code_from_diff_instruction(instruction: str) -> tuple[str, str]:
    """
    Extract the 'before' code and commit message from diff instruction.

    Returns (before_code, commit_message)
    """
    # Extract commit message from first line
    lines = instruction.split('\n')
    commit_msg = ""
    for line in lines:
        if line.startswith("Fix the following"):
            # Extract description after colon if present
            if ':' in line:
                commit_msg = line.split(':', 1)[1].strip()
            break

    # Extract code from Before block
    before_match = re.search(r'Before:\s*```(?:cpp)?\n(.*?)```', instruction, re.DOTALL)
    before_code = before_match.group(1).strip() if before_match else ""

    return before_code, commit_msg


def load_diff_sft(path: str) -> Iterator[dict]:
    """
    Load diff SFT data as pure C++ format.

    Input format:
    {
        "instruction": "Fix the following C++ code:\n...\nBefore:\n```cpp\n...\n```",
        "response": "```cpp\n...\n```"
    }

    Output: Pure C++ with comment instruction
    // @fix: <commit_message>
    // Before:
    <before_code>
    // After:
    <after_code>
    """
    logger.info(f"Loading diff SFT from {path}...")

    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            instruction = data.get("instruction", "")
            response = data.get("response", "")

            if not instruction or not response:
                continue

            # Extract before code and commit message
            before_code, commit_msg = extract_code_from_diff_instruction(instruction)
            after_code = strip_markdown_fences(response)

            if not before_code or not after_code:
                continue

            # Format as pure C++ with comment-based instruction
            # Model learns: given buggy code with fix instruction -> produce fixed code
            if commit_msg:
                cpp_text = f"// @fix: {commit_msg}\n{before_code}\n// @fixed:\n{after_code}"
            else:
                cpp_text = f"// @fix\n{before_code}\n// @fixed:\n{after_code}"

            yield {
                "text": cpp_text,
                "source": "diff_sft",
                "metadata": {},
            }


def load_docstring_pairs(path: str, max_examples: int = 100000) -> Iterator[dict]:
    """
    Load docstring pairs data as pure C++ format.

    Input format:
    {
        "docstring": "reference code snippet",
        "signature": "condition/expression that starts the block",
        "body": "C/C++ implementation",
        "path": "file path"
    }

    Output: Pure C++ with FIM-style completion
    // @complete: <signature>
    // @ref: <reference_snippet>
    <signature> {
    <body>
    }
    """
    logger.info(f"Loading docstring pairs from {path}...")

    count = 0
    with open(path, "r") as f:
        for line in f:
            if count >= max_examples:
                break

            if not line.strip():
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            reference = data.get("docstring", "")
            signature = data.get("signature", "")
            body = data.get("body", "")

            if not (signature and body):
                continue

            # Skip if body is too short or too long
            if len(body) < 20 or len(body) > 5000:
                continue

            # Format reference as multi-line comment if it has newlines
            if '\n' in reference:
                ref_lines = reference.strip().split('\n')
                ref_comment = '/*\n' + '\n'.join(f' * {line}' for line in ref_lines[:10]) + '\n */'
            else:
                ref_comment = f'// @ref: {reference[:200]}'

            # Pure C++ format: comment instruction + complete code
            cpp_text = f"{ref_comment}\n{signature} {{\n{body}\n}}"

            yield {
                "text": cpp_text,
                "source": "docstring",
                "metadata": {},
            }
            count += 1


def load_fim_pairs(path: str, max_examples: int = 50000) -> Iterator[dict]:
    """
    Convert docstring pairs to FIM format using special tokens.

    Uses standard FIM tokens:
    <|fim_prefix|>...<|fim_suffix|>...<|fim_middle|>...
    """
    logger.info(f"Creating FIM pairs from {path}...")

    count = 0
    with open(path, "r") as f:
        for line in f:
            if count >= max_examples:
                break

            if not line.strip():
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            reference = data.get("docstring", "")
            signature = data.get("signature", "")
            body = data.get("body", "")

            if not (signature and body):
                continue

            # Skip if body is too short or too long
            if len(body) < 20 or len(body) > 3000:
                continue

            # FIM format: model fills in the body
            # Reference goes in comment before the code
            if reference:
                ref_comment = f"// {reference[:150].replace(chr(10), ' ')}\n"
            else:
                ref_comment = ""

            prefix = f"{ref_comment}{signature} {{"
            suffix = "\n}"
            middle = f"\n{body}"

            fim_text = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}"

            yield {
                "text": fim_text,
                "source": "fim",
                "metadata": {
                    "fim_style": True,
                },
            }
            count += 1


def write_combined(
    examples: list[dict],
    output_path: str,
    shuffle: bool = True,
):
    """Write combined dataset with optional shuffling."""
    if shuffle:
        random.shuffle(examples)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    logger.info(f"Wrote {len(examples)} examples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare combined SFT dataset")
    parser.add_argument(
        "--diff-sft",
        type=str,
        default="data/diff_sft.jsonl",
        help="Path to diff SFT data",
    )
    parser.add_argument(
        "--docstring-pairs",
        type=str,
        default="data/docstring_pairs_full.jsonl",
        help="Path to docstring pairs data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/combined_sft.jsonl",
        help="Output path",
    )
    parser.add_argument(
        "--max-docstring",
        type=int,
        default=100000,
        help="Max examples from docstring pairs",
    )
    parser.add_argument(
        "--max-fim",
        type=int,
        default=50000,
        help="Max FIM instruction examples",
    )
    parser.add_argument(
        "--include-fim",
        action="store_true",
        help="Include FIM-style instruction pairs",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Don't shuffle the output",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    all_examples = []

    # Load diff SFT
    if args.diff_sft and os.path.exists(args.diff_sft):
        diff_examples = list(load_diff_sft(args.diff_sft))
        all_examples.extend(diff_examples)
        logger.info(f"  Added {len(diff_examples)} diff SFT examples")
    else:
        logger.warning(f"Diff SFT file not found: {args.diff_sft}")

    # Load docstring pairs
    if args.docstring_pairs and os.path.exists(args.docstring_pairs):
        doc_examples = list(load_docstring_pairs(args.docstring_pairs, args.max_docstring))
        all_examples.extend(doc_examples)
        logger.info(f"  Added {len(doc_examples)} docstring examples")

        # Optionally add FIM instruction pairs
        if args.include_fim:
            fim_examples = list(load_fim_pairs(args.docstring_pairs, args.max_fim))
            all_examples.extend(fim_examples)
            logger.info(f"  Added {len(fim_examples)} FIM instruction examples")
    else:
        logger.warning(f"Docstring pairs file not found: {args.docstring_pairs}")

    # Write combined output
    write_combined(all_examples, args.output, shuffle=not args.no_shuffle)

    # Summary
    sources = {}
    for ex in all_examples:
        source = ex.get("source", "unknown")
        sources[source] = sources.get(source, 0) + 1

    logger.info(f"\nSummary:")
    for source, count in sorted(sources.items()):
        logger.info(f"  {source}: {count} examples")
    logger.info(f"  Total: {len(all_examples)} examples")


if __name__ == "__main__":
    main()
