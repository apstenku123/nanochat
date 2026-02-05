#!/usr/bin/env python3
"""
Generate tool-call SFT data from existing datasets.

Produces training examples where the model thinks in C++ comments,
makes tool calls as C++ function expressions, receives results,
and writes final code — all in pure C++ token space.

Five strategies:
A) Docstring → search + code  (from docstring_pairs_clean.jsonl)
B) Diff → compile + fix       (from diff_sft.jsonl)
C) HumanEval → ask + solve    (from gspo_prompts.jsonl)
D) No-tool direct code        (plain completion/repair without tools)
E) Code execution via run()   (model writes C++ to compute answers)

Output format (JSONL):
{
    "text": "<BOS>// task...<THOUGHT_START>// reasoning<THOUGHT_END>...<EOS>",
    "source": "docstring_search|diff_compile|humaneval_ask|no_tool"
}

Usage:
    python -m scripts.data.generate_tool_sft \
        --docstring-pairs data/docstring_pairs_clean.jsonl \
        --diff-sft data/diff_sft.jsonl \
        --gspo-prompts data/gspo_prompts.jsonl \
        --output data/tool_call_sft.jsonl
"""

import argparse
import json
import logging
import os
import random
import re
from typing import Iterator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Special tokens (these are literal strings that the tokenizer maps to IDs)
BOS = "<BOS>"
EOS = "<EOS>"
THOUGHT_START = "<THOUGHT_START>"
THOUGHT_END = "<THOUGHT_END>"
QUERY_TOOL = "<QUERY_TOOL>"
TOOL_RESULT = "<TOOL_RESULT>"
CODE_START = "<CODE_START>"
CODE_END = "<CODE_END>"
FIM_PREFIX = "<FIM_PREFIX>"
FIM_MIDDLE = "<FIM_MIDDLE>"
FIM_SUFFIX = "<FIM_SUFFIX>"


def escape_for_cpp_string(s: str) -> str:
    """Escape a string for use inside a C++ string literal in tool call."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def extract_key_terms(text: str) -> str:
    """Extract key C++ terms from a signature or docstring for search queries."""
    # Remove common noise words
    noise = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
             "to", "of", "in", "for", "on", "with", "at", "by", "from",
             "that", "this", "it", "and", "or", "not", "if", "else",
             "void", "int", "char", "bool", "float", "double", "const",
             "return", "static", "inline", "virtual", "override"}
    words = re.findall(r'[a-zA-Z_]\w+', text)
    keywords = [w for w in words if w.lower() not in noise and len(w) > 2]
    return " ".join(keywords[:6])


def truncate_code(code: str, max_lines: int = 20) -> str:
    """Truncate code to max lines, adding ... if truncated."""
    lines = code.split("\n")
    if len(lines) <= max_lines:
        return code
    return "\n".join(lines[:max_lines]) + "\n// ..."


def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences if present."""
    text = re.sub(r'^```(?:cpp|c\+\+|c)?\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?```$', '', text, flags=re.MULTILINE)
    return text.strip()


# ---------------------------------------------------------------------------
# Strategy A: Docstring → Search + Code
# ---------------------------------------------------------------------------

def load_docstring_index(path: str, max_records: int = 200000) -> list[dict]:
    """Load docstring pairs into memory for cross-referencing."""
    records = []
    with open(path) as f:
        for line in f:
            if len(records) >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_path_index(records: list[dict]) -> dict[str, list[int]]:
    """Build index from directory prefix to record indices for related search."""
    idx = {}
    for i, r in enumerate(records):
        path = r.get("path", "")
        # Use top 2 directory components as key
        parts = path.split("/")
        if len(parts) >= 2:
            key = "/".join(parts[:2])
        else:
            key = parts[0] if parts else ""
        idx.setdefault(key, []).append(i)
    return idx


def generate_docstring_search_examples(
    records: list[dict],
    path_index: dict[str, list[int]],
    max_examples: int = 100000,
    rng: random.Random = None,
) -> Iterator[dict]:
    """Strategy A: Wrap docstring→code in search tool narrative."""
    if rng is None:
        rng = random.Random(42)

    indices = list(range(len(records)))
    rng.shuffle(indices)

    count = 0
    for idx in indices:
        if count >= max_examples:
            break

        r = records[idx]
        docstring = r["docstring"]
        signature = r["signature"]
        body = r["body"]
        path = r.get("path", "unknown")

        # Skip if too long for 1024 token context
        total_chars = len(docstring) + len(signature) + len(body)
        if total_chars > 3000 or total_chars < 80:
            continue

        # Find a related record for the "search result"
        dir_key = "/".join(path.split("/")[:2]) if "/" in path else path
        related_indices = path_index.get(dir_key, [])
        related = None
        if len(related_indices) > 1:
            for ri in rng.sample(related_indices, min(5, len(related_indices))):
                if ri != idx:
                    related = records[ri]
                    break

        # Build search query from key terms
        query = extract_key_terms(signature + " " + docstring[:100])
        if not query:
            query = extract_key_terms(body[:100])
        if not query:
            continue

        # Build the search result (either related code or the docstring itself)
        if related:
            search_result = f"// Found in {related['path']}:\n{truncate_code(related['body'], 12)}"
        else:
            # Use docstring as the "reference" found
            doc_lines = docstring.strip().split("\n")
            search_result = "// Reference:\n" + "\n".join(
                f"// {line}" for line in doc_lines[:8]
            )

        # Build thought comments
        thought1 = f"// Need to implement: {signature.strip()[:80]}\n// Let me search for related patterns"
        thought2 = f"// Based on the reference, implementing the function"

        # Assemble the full text with special tokens
        text = (
            f"{BOS}\n"
            f"// Implement the following:\n"
            f"/* {docstring[:500].strip()} */\n"
            f"{signature.strip()}\n"
            f"{THOUGHT_START}\n{thought1}\n{THOUGHT_END}\n"
            f"{QUERY_TOOL} search(\"{escape_for_cpp_string(query)}\") {CODE_END}\n"
            f"{TOOL_RESULT}\n{search_result}\n{CODE_END}\n"
            f"{THOUGHT_START}\n{thought2}\n{THOUGHT_END}\n"
            f"{CODE_START}\n"
            f"{signature.strip()} {{\n{body.strip()}\n}}\n"
            f"{CODE_END}\n"
            f"{EOS}"
        )

        yield {"text": text, "source": "docstring_search"}
        count += 1

    logger.info(f"  Strategy A (docstring_search): generated {count} examples")


# ---------------------------------------------------------------------------
# Strategy B: Diff → Compile + Fix
# ---------------------------------------------------------------------------

def extract_before_after(instruction: str, response: str):
    """Extract before/after code from diff SFT format."""
    # Try Before/After format
    before_match = re.search(r'Before:\s*```(?:cpp)?\n(.*?)```', instruction, re.DOTALL)
    before_code = before_match.group(1).strip() if before_match else None

    after_code = strip_markdown_fences(response)

    # Extract commit message
    commit_msg = ""
    lines = instruction.split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("Fix the following"):
            if "\n" in instruction:
                # Second line is usually the commit message
                parts = instruction.split("\n", 2)
                if len(parts) >= 2:
                    commit_msg = parts[1].strip()
            break

    return before_code, after_code, commit_msg


def synthesize_compile_error(commit_msg: str, before_code: str) -> str:
    """Synthesize a plausible compiler error from the commit message and code."""
    msg_lower = commit_msg.lower()

    if any(w in msg_lower for w in ["undefined", "undeclared", "not declared"]):
        return "// error: use of undeclared identifier"
    elif any(w in msg_lower for w in ["type", "cast", "conversion"]):
        return "// error: invalid type conversion"
    elif any(w in msg_lower for w in ["null", "nullptr", "dereference"]):
        return "// warning: potential null pointer dereference"
    elif any(w in msg_lower for w in ["memory", "leak", "free", "delete"]):
        return "// warning: potential memory leak detected"
    elif any(w in msg_lower for w in ["overflow", "bounds", "range"]):
        return "// warning: array index out of bounds"
    elif any(w in msg_lower for w in ["unused", "unreachable"]):
        return "// warning: unused variable or unreachable code"
    elif any(w in msg_lower for w in ["thread", "race", "lock", "mutex"]):
        return "// warning: potential data race condition"
    elif any(w in msg_lower for w in ["virtual", "override", "vtable"]):
        return "// error: virtual function override mismatch"
    else:
        # Generic: reference the commit message
        return f"// warning: {commit_msg[:80]}"


def generate_diff_compile_examples(
    diff_path: str,
    max_examples: int = 50000,
    rng: random.Random = None,
) -> Iterator[dict]:
    """Strategy B: Wrap diff repairs in compile+fix tool narrative."""
    if rng is None:
        rng = random.Random(42)

    examples = []
    with open(diff_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))

    rng.shuffle(examples)
    count = 0

    for ex in examples:
        if count >= max_examples:
            break

        instruction = ex.get("instruction", "")
        response = ex.get("response", "")

        before_code, after_code, commit_msg = extract_before_after(instruction, response)
        if not before_code or not after_code:
            continue

        # Skip if too long
        total_chars = len(before_code) + len(after_code)
        if total_chars > 3000 or total_chars < 50:
            continue

        # Skip trivial renames
        if "rename" in instruction.lower()[:100] and before_code.count("\n") < 3:
            continue

        # Synthesize compile error
        compile_error = synthesize_compile_error(commit_msg or "fix bug", before_code)

        # Build thoughts
        thought1 = "// Let me try compiling this to identify the issue"
        if commit_msg:
            thought2 = f"// The issue is: {commit_msg[:100]}\n// Applying the fix"
        else:
            thought2 = "// Found the issue, applying the fix"

        text = (
            f"{BOS}\n"
            f"// Fix the following code:\n"
            f"{truncate_code(before_code, 30)}\n"
            f"{THOUGHT_START}\n{thought1}\n{THOUGHT_END}\n"
            f"{QUERY_TOOL} compile(\"{escape_for_cpp_string(truncate_code(before_code, 15))}\") {CODE_END}\n"
            f"{TOOL_RESULT}\n{compile_error}\n{CODE_END}\n"
            f"{THOUGHT_START}\n{thought2}\n{THOUGHT_END}\n"
            f"{CODE_START}\n"
            f"{after_code.strip()}\n"
            f"{CODE_END}\n"
            f"{EOS}"
        )

        yield {"text": text, "source": "diff_compile"}
        count += 1

    logger.info(f"  Strategy B (diff_compile): generated {count} examples")


# ---------------------------------------------------------------------------
# Strategy C: HumanEval → Ask + Solve
# ---------------------------------------------------------------------------

def extract_problem_hint(prompt: str, solution: str) -> str:
    """Generate an algorithm hint from the HumanEval problem."""
    # Extract the function docstring
    doc_match = re.search(r'/\*(.*?)\*/', prompt, re.DOTALL)
    if doc_match:
        doc = doc_match.group(1).strip()
        # First sentence as hint
        sentences = doc.split(".")
        if sentences:
            return f"// Approach: {sentences[0].strip()[:120]}"

    return "// Use a straightforward iterative approach"


def generate_humaneval_ask_examples(
    gspo_path: str,
    max_examples: int = 2000,
    rng: random.Random = None,
) -> Iterator[dict]:
    """Strategy C: Wrap HumanEval in ask+solve narrative."""
    if rng is None:
        rng = random.Random(42)

    prompts = []
    with open(gspo_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line))

    count = 0
    # Repeat with different temperatures/augmentations
    num_repeats = max(1, max_examples // max(len(prompts), 1))

    for repeat in range(num_repeats):
        rng.shuffle(prompts)
        for p in prompts:
            if count >= max_examples:
                break

            prompt = p.get("prompt", "")
            solution = p.get("canonical_solution", "")
            if not prompt or not solution:
                continue

            # Skip if too long
            if len(prompt) + len(solution) > 2500:
                continue

            # Generate hint
            hint = extract_problem_hint(prompt, solution)

            # Extract function name for query
            func_match = re.search(r'(\w+)\s*\(', prompt.split("\n")[-1])
            func_name = func_match.group(1) if func_match else "the function"

            # Vary the ask query across repeats
            ask_queries = [
                f"What algorithm should I use for {func_name}?",
                f"What is the best approach for implementing {func_name}?",
                f"Explain the key edge cases for {func_name}",
                f"What data structure works best for {func_name}?",
            ]
            ask_query = ask_queries[repeat % len(ask_queries)]

            thought1 = f"// Let me think about the approach for {func_name}"
            thought2 = "// Got it, implementing the solution"

            text = (
                f"{BOS}\n"
                f"{prompt.strip()}\n"
                f"{THOUGHT_START}\n{thought1}\n{THOUGHT_END}\n"
                f"{QUERY_TOOL} ask(\"{escape_for_cpp_string(ask_query)}\") {CODE_END}\n"
                f"{TOOL_RESULT}\n{hint}\n{CODE_END}\n"
                f"{THOUGHT_START}\n{thought2}\n{THOUGHT_END}\n"
                f"{CODE_START}\n"
                f"{solution.strip()}\n"
                f"{CODE_END}\n"
                f"{EOS}"
            )

            yield {"text": text, "source": "humaneval_ask"}
            count += 1

    logger.info(f"  Strategy C (humaneval_ask): generated {count} examples")


# ---------------------------------------------------------------------------
# Strategy E: Code execution via run() — model writes C++ to compute things
# ---------------------------------------------------------------------------

# Templates for code-execution training examples
# Each template has a Python "result" function that computes the expected output
# (avoids forking g++ 10k times during data generation)
import math

_WORDS_LONG = ["strawberry", "banana", "mississippi", "programming",
               "concatenation", "parallelism", "polymorphism",
               "implementation", "infrastructure", "communication",
               "characteristic", "acknowledgement", "representative",
               "understanding", "international", "extraordinary"]
_WORDS_SHORT = ["hello", "world", "algorithm", "template",
                "namespace", "iterator", "polymorphism", "compiler",
                "debugger", "profiler", "optimizer", "database"]
_WORDS_VERY_LONG = ["supercalifragilisticexpialidocious", "antidisestablishmentarianism",
                    "electroencephalography", "incomprehensible",
                    "onomatopoeia", "serendipity", "cryptocurrency"]

_RUN_TEMPLATES = [
    # String counting
    {
        "task": "// How many times does the letter '{letter}' appear in \"{word}\"?",
        "thought1": "// I need to count occurrences — let me write C++ code to compute this",
        "code": '    string s = "{word}";\n    int count = 0;\n    for (char c : s) if (c == \'{letter}\') count++;\n    cout << count << endl;',
        "gen": lambda rng: {"word": rng.choice(_WORDS_LONG), "letter": rng.choice(list("abcdefghijklmnopqrstuvwxyz"))},
        "result": lambda p: str(p["word"].count(p["letter"])),
    },
    # String length
    {
        "task": "// What is the length of \"{word}\"?",
        "thought1": "// Let me compute the string length",
        "code": '    string s = "{word}";\n    cout << s.length() << endl;',
        "gen": lambda rng: {"word": rng.choice(_WORDS_VERY_LONG + _WORDS_LONG)},
        "result": lambda p: str(len(p["word"])),
    },
    # Reverse string
    {
        "task": "// What is \"{word}\" reversed?",
        "thought1": "// Let me reverse the string with C++ code",
        "code": '    string s = "{word}";\n    reverse(s.begin(), s.end());\n    cout << s << endl;',
        "gen": lambda rng: {"word": rng.choice(_WORDS_SHORT)},
        "result": lambda p: p["word"][::-1],
    },
    # Math computation
    {
        "task": "// Compute {a} * {b} + {c}",
        "thought1": "// Let me compute this with C++ code",
        "code": '    cout << {a} * {b} + {c} << endl;',
        "gen": lambda rng: {"a": rng.randint(100, 9999), "b": rng.randint(100, 9999), "c": rng.randint(1, 999)},
        "result": lambda p: str(p["a"] * p["b"] + p["c"]),
    },
    # Fibonacci
    {
        "task": "// What is the {n}th Fibonacci number?",
        "thought1": "// Let me compute Fibonacci with a loop",
        "code": '    int n = {n};\n    long long a = 0, b = 1;\n    for (int i = 0; i < n; i++) {{ long long t = a + b; a = b; b = t; }}\n    cout << a << endl;',
        "gen": lambda rng: {"n": rng.randint(5, 40)},
        "result": lambda p: str((lambda n: (lambda f: f(f, n))(lambda self, n: 0 if n == 0 else 1 if n == 1 else (lambda: (a := [0, 1], [a.append(a[-1] + a[-2]) for _ in range(n - 1)], a[n])[-1])()))(p["n"])),
    },
    # Character at position
    {
        "task": "// What is the {pos}th character of \"{word}\"?",
        "thought1": "// Let me index into the string",
        "code": '    string s = "{word}";\n    int pos = {pos};\n    if (pos < (int)s.size()) cout << s[pos] << endl;\n    else cout << "(out of range)" << endl;',
        "gen": lambda rng: {"word": rng.choice(_WORDS_SHORT), "pos": rng.randint(0, 7)},
        "result": lambda p: p["word"][p["pos"]] if p["pos"] < len(p["word"]) else "(out of range)",
    },
    # Sort characters
    {
        "task": "// Sort the characters of \"{word}\" alphabetically",
        "thought1": "// Let me sort with std::sort",
        "code": '    string s = "{word}";\n    sort(s.begin(), s.end());\n    cout << s << endl;',
        "gen": lambda rng: {"word": rng.choice(_WORDS_SHORT)},
        "result": lambda p: "".join(sorted(p["word"])),
    },
    # GCD
    {
        "task": "// What is the GCD of {a} and {b}?",
        "thought1": "// Let me compute GCD using Euclidean algorithm",
        "code": '    int a = {a}, b = {b};\n    while (b) {{ int t = b; b = a % b; a = t; }}\n    cout << a << endl;',
        "gen": lambda rng: {"a": rng.randint(10, 9999), "b": rng.randint(10, 9999)},
        "result": lambda p: str(math.gcd(p["a"], p["b"])),
    },
    # Count vowels
    {
        "task": "// How many vowels are in \"{word}\"?",
        "thought1": "// Let me count vowels with C++ code",
        "code": '    string s = "{word}";\n    string vowels = "aeiouAEIOU";\n    int count = 0;\n    for (char c : s) if (vowels.find(c) != string::npos) count++;\n    cout << count << endl;',
        "gen": lambda rng: {"word": rng.choice(_WORDS_LONG)},
        "result": lambda p: str(sum(1 for c in p["word"] if c in "aeiouAEIOU")),
    },
    # Is palindrome
    {
        "task": "// Is \"{word}\" a palindrome?",
        "thought1": "// Let me check palindrome property",
        "code": '    string s = "{word}";\n    string r = s;\n    reverse(r.begin(), r.end());\n    cout << (s == r ? "yes" : "no") << endl;',
        "gen": lambda rng: {"word": rng.choice(["racecar", "level", "hello", "madam", "world",
                                                 "rotator", "deified", "civic", "algorithm"])},
        "result": lambda p: "yes" if p["word"] == p["word"][::-1] else "no",
    },
]


def _fib(n):
    """Compute nth Fibonacci number."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

# Fix Fibonacci result to use clean function
_RUN_TEMPLATES[4]["result"] = lambda p: str(_fib(p["n"]))


def generate_run_tool_examples(
    max_examples: int = 10000,
    rng: random.Random = None,
) -> Iterator[dict]:
    """Strategy E: Model writes C++ code and runs it via run() tool.

    Results are computed in Python (same math as the C++ code would produce)
    to avoid forking g++ for each example.
    """
    if rng is None:
        rng = random.Random(42)

    count = 0
    while count < max_examples:
        template = rng.choice(_RUN_TEMPLATES)
        params = template["gen"](rng)

        task = template["task"].format(**params)
        thought1 = template["thought1"]
        code_body = template["code"].format(**params)
        output = template["result"](params)

        # The code that gets passed to run() as a string argument
        run_code = code_body.replace("\n", "\\n").replace('"', '\\"')

        thought2 = f"// The result is {output}"

        text = (
            f"{BOS}\n"
            f"{task}\n"
            f"{THOUGHT_START}\n{thought1}\n{THOUGHT_END}\n"
            f"{QUERY_TOOL} run(\"{run_code}\") {CODE_END}\n"
            f"{TOOL_RESULT}\n// Output: {output}\n{CODE_END}\n"
            f"{THOUGHT_START}\n{thought2}\n{THOUGHT_END}\n"
            f"{CODE_START}\n"
            f"// Answer: {output}\n"
            f"{CODE_END}\n"
            f"{EOS}"
        )

        yield {"text": text, "source": "run_code"}
        count += 1

    logger.info(f"  Strategy E (run_code): generated {count} examples")


# ---------------------------------------------------------------------------
# Strategy D: No-tool direct code (docstring → code, diff → fix, FIM)
# ---------------------------------------------------------------------------

def generate_no_tool_examples(
    docstring_records: list[dict],
    diff_path: str,
    max_docstring: int = 30000,
    max_diff: int = 15000,
    max_fim: int = 5000,
    rng: random.Random = None,
) -> Iterator[dict]:
    """Strategy D: Plain code completion/repair without tool calls."""
    if rng is None:
        rng = random.Random(42)

    # D.1: Docstring → code (no tools, just think + code)
    indices = list(range(len(docstring_records)))
    rng.shuffle(indices)
    count_doc = 0
    for idx in indices:
        if count_doc >= max_docstring:
            break
        r = docstring_records[idx]
        docstring = r["docstring"]
        signature = r["signature"]
        body = r["body"]

        total_chars = len(docstring) + len(signature) + len(body)
        if total_chars > 2500 or total_chars < 60:
            continue

        # Simple: think + code, no tools
        thought = f"// Implementing {signature.strip()[:60]}"

        text = (
            f"{BOS}\n"
            f"/* {docstring[:400].strip()} */\n"
            f"{signature.strip()}\n"
            f"{THOUGHT_START}\n{thought}\n{THOUGHT_END}\n"
            f"{CODE_START}\n"
            f"{signature.strip()} {{\n{body.strip()}\n}}\n"
            f"{CODE_END}\n"
            f"{EOS}"
        )

        yield {"text": text, "source": "no_tool"}
        count_doc += 1

    # D.2: Diff → direct fix (no compile tool)
    diff_examples = []
    with open(diff_path) as f:
        for line in f:
            line = line.strip()
            if line:
                diff_examples.append(json.loads(line))
    rng.shuffle(diff_examples)
    count_diff = 0
    for ex in diff_examples:
        if count_diff >= max_diff:
            break
        instruction = ex.get("instruction", "")
        response = ex.get("response", "")
        before_code, after_code, commit_msg = extract_before_after(instruction, response)
        if not before_code or not after_code:
            continue
        total_chars = len(before_code) + len(after_code)
        if total_chars > 2500 or total_chars < 50:
            continue

        if commit_msg:
            thought = f"// Fix: {commit_msg[:100]}"
        else:
            thought = "// Applying the fix directly"

        text = (
            f"{BOS}\n"
            f"// Fix the following code:\n"
            f"{truncate_code(before_code, 25)}\n"
            f"{THOUGHT_START}\n{thought}\n{THOUGHT_END}\n"
            f"{CODE_START}\n"
            f"{after_code.strip()}\n"
            f"{CODE_END}\n"
            f"{EOS}"
        )

        yield {"text": text, "source": "no_tool"}
        count_diff += 1

    # D.3: FIM from docstring pairs (model already knows FIM from pretraining)
    rng.shuffle(indices)
    count_fim = 0
    for idx in indices:
        if count_fim >= max_fim:
            break
        r = docstring_records[idx]
        signature = r["signature"]
        body = r["body"]
        docstring = r["docstring"]

        if len(body) < 30 or len(body) > 2000:
            continue

        # FIM format: prefix + suffix + middle
        doc_comment = f"// {docstring[:120].replace(chr(10), ' ')}"
        prefix = f"{doc_comment}\n{signature.strip()} {{"
        suffix = "\n}"
        middle = f"\n{body.strip()}"

        text = (
            f"{BOS}\n"
            f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}{middle}\n"
            f"{EOS}"
        )

        yield {"text": text, "source": "no_tool_fim"}
        count_fim += 1

    logger.info(f"  Strategy D (no_tool): {count_doc} docstring + {count_diff} diff + {count_fim} FIM")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate tool-call SFT data")
    parser.add_argument("--docstring-pairs", type=str,
                        default="data/docstring_pairs_clean.jsonl")
    parser.add_argument("--diff-sft", type=str,
                        default="data/diff_sft.jsonl")
    parser.add_argument("--gspo-prompts", type=str,
                        default="data/gspo_prompts.jsonl")
    parser.add_argument("--output", type=str,
                        default="data/tool_call_sft.jsonl")
    # Strategy limits
    parser.add_argument("--max-docstring-search", type=int, default=100000)
    parser.add_argument("--max-diff-compile", type=int, default=50000)
    parser.add_argument("--max-humaneval-ask", type=int, default=2000)
    parser.add_argument("--max-no-tool-docstring", type=int, default=30000)
    parser.add_argument("--max-no-tool-diff", type=int, default=15000)
    parser.add_argument("--max-no-tool-fim", type=int, default=5000)
    parser.add_argument("--max-run-code", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load docstring records into memory (for cross-referencing)
    logger.info("Loading docstring pairs...")
    doc_records = load_docstring_index(args.docstring_pairs, max_records=300000)
    logger.info(f"  Loaded {len(doc_records):,} docstring records")
    path_index = build_path_index(doc_records)
    logger.info(f"  Built path index with {len(path_index)} directory groups")

    # Generate all strategies
    all_examples = []

    logger.info("Generating Strategy A: docstring → search + code...")
    for ex in generate_docstring_search_examples(
        doc_records, path_index, args.max_docstring_search, rng
    ):
        all_examples.append(ex)

    logger.info("Generating Strategy B: diff → compile + fix...")
    for ex in generate_diff_compile_examples(
        args.diff_sft, args.max_diff_compile, rng
    ):
        all_examples.append(ex)

    if os.path.exists(args.gspo_prompts):
        logger.info("Generating Strategy C: HumanEval → ask + solve...")
        for ex in generate_humaneval_ask_examples(
            args.gspo_prompts, args.max_humaneval_ask, rng
        ):
            all_examples.append(ex)

    logger.info("Generating Strategy D: no-tool direct code...")
    for ex in generate_no_tool_examples(
        doc_records, args.diff_sft,
        max_docstring=args.max_no_tool_docstring,
        max_diff=args.max_no_tool_diff,
        max_fim=args.max_no_tool_fim,
        rng=rng,
    ):
        all_examples.append(ex)

    logger.info("Generating Strategy E: code execution via run()...")
    for ex in generate_run_tool_examples(
        max_examples=args.max_run_code,
        rng=rng,
    ):
        all_examples.append(ex)

    # Shuffle
    rng.shuffle(all_examples)

    # Write output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    # Summary
    source_counts = {}
    for ex in all_examples:
        src = ex.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    logger.info(f"\nGeneration complete: {len(all_examples):,} total examples")
    for src, count in sorted(source_counts.items()):
        logger.info(f"  {src}: {count:,}")
    logger.info(f"Output: {args.output}")


if __name__ == "__main__":
    main()
