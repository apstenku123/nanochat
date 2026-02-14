"""
Deep tests for scripts/data/generate_tool_sft.py data generation quality.
"""
import math
import random
import json
import tempfile
import os
import re

import pytest

# Import the generation functions
from scripts.data.generate_tool_sft import (
    escape_for_cpp_string,
    extract_key_terms,
    truncate_code,
    strip_markdown_fences,
    extract_before_after,
    synthesize_compile_error,
    generate_run_tool_examples,
    generate_docstring_search_examples,
    generate_diff_compile_examples,
    generate_humaneval_ask_examples,
    generate_no_tool_examples,
    build_path_index,
    _fib,
    _RUN_TEMPLATES,
    BOS, EOS, THOUGHT_START, THOUGHT_END, QUERY_TOOL, TOOL_RESULT,
    CODE_START, CODE_END, FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX,
)


class TestEscapeForCppString:
    def test_basic_escape(self):
        assert escape_for_cpp_string('hello "world"') == 'hello \\"world\\"'

    def test_backslash_escape(self):
        assert escape_for_cpp_string("a\\b") == "a\\\\b"

    def test_newline_escape(self):
        assert escape_for_cpp_string("line1\nline2") == "line1\\nline2"

    def test_combined(self):
        result = escape_for_cpp_string('path\\to\n"file"')
        assert result == 'path\\\\to\\n\\"file\\"'

    def test_empty(self):
        assert escape_for_cpp_string("") == ""

    def test_no_escape_needed(self):
        assert escape_for_cpp_string("hello world") == "hello world"

    def test_tab_not_escaped(self):
        """BUG: tabs are NOT escaped but would break C++ string literals."""
        result = escape_for_cpp_string("col1\tcol2")
        # Tab is not escaped — this would produce a valid C++ string
        # with a literal tab, which is fine actually. Not a bug.
        assert result == "col1\tcol2"

    def test_null_byte(self):
        """Null byte should be escaped for C++ strings."""
        result = escape_for_cpp_string("hello\x00world")
        assert "\\0" in result
        assert "\x00" not in result

    def test_carriage_return(self):
        """Carriage return should be escaped."""
        result = escape_for_cpp_string("line1\rline2")
        assert "\\r" in result
        assert "\r" not in result


class TestExtractKeyTerms:
    def test_filters_noise_words(self):
        terms = extract_key_terms("the void function to sort the array")
        assert "the" not in terms
        assert "void" not in terms

    def test_extracts_cpp_identifiers(self):
        terms = extract_key_terms("std::vector<int>::push_back implementation")
        words = terms.split()
        assert any("vector" in w for w in words)
        assert any("push_back" in w for w in words)

    def test_limits_to_6_terms(self):
        terms = extract_key_terms("word1 word2 word3 word4 word5 word6 word7 word8 word9 word10")
        assert len(terms.split()) <= 6

    def test_empty_string(self):
        terms = extract_key_terms("")
        assert terms == ""

    def test_all_noise_words(self):
        terms = extract_key_terms("the a an is are to of in for on")
        assert terms == ""

    def test_short_words_filtered(self):
        """Words <= 2 chars are filtered out."""
        terms = extract_key_terms("is a do go")
        assert terms == ""


class TestTruncateCode:
    def test_no_truncation(self):
        code = "line1\nline2\nline3"
        assert truncate_code(code, 5) == code

    def test_truncation(self):
        code = "\n".join(f"line{i}" for i in range(30))
        result = truncate_code(code, 10)
        lines = result.split("\n")
        assert len(lines) == 11  # 10 lines + "// ..."
        assert lines[-1] == "// ..."

    def test_exact_limit(self):
        code = "\n".join(f"line{i}" for i in range(20))
        result = truncate_code(code, 20)
        assert result == code


class TestStripMarkdownFences:
    def test_cpp_fence(self):
        text = "```cpp\nint x = 0;\n```"
        assert strip_markdown_fences(text) == "int x = 0;"

    def test_no_fence(self):
        text = "int x = 0;"
        assert strip_markdown_fences(text) == text

    def test_cplusplus_fence(self):
        text = "```c++\nint x = 0;\n```"
        assert strip_markdown_fences(text) == "int x = 0;"

    def test_plain_fence(self):
        text = "```\nint x = 0;\n```"
        assert strip_markdown_fences(text) == "int x = 0;"


class TestExtractBeforeAfter:
    def test_standard_format(self):
        instruction = """Fix the following code:
commit message here
Before:
```cpp
int f() { return; }
```"""
        response = "```cpp\nint f() { return 0; }\n```"
        before, after, msg = extract_before_after(instruction, response)
        assert before == "int f() { return; }"
        assert after == "int f() { return 0; }"

    def test_no_before_block(self):
        """No Before: block → before_code is None."""
        instruction = "Fix this:\nint f() { return; }"
        response = "int f() { return 0; }"
        before, after, msg = extract_before_after(instruction, response)
        assert before is None
        assert after == "int f() { return 0; }"

    def test_commit_message_extraction(self):
        instruction = "Fix the following code:\nfix null pointer dereference\nBefore:\n```cpp\nint* p;\n```"
        response = "```cpp\nint* p = nullptr;\n```"
        before, after, msg = extract_before_after(instruction, response)
        assert "null pointer" in msg.lower()


class TestSynthesizeCompileError:
    def test_undefined_error(self):
        result = synthesize_compile_error("fix undefined variable", "int x;")
        assert "undeclared" in result.lower()

    def test_type_error(self):
        result = synthesize_compile_error("fix type conversion", "int x = 1.0;")
        assert "type" in result.lower()

    def test_null_error(self):
        result = synthesize_compile_error("fix null dereference", "int* p;")
        assert "null" in result.lower()

    def test_generic_fallback(self):
        result = synthesize_compile_error("refactor code structure", "int x;")
        assert "refactor" in result.lower()

    def test_empty_message(self):
        result = synthesize_compile_error("", "int x;")
        assert result.startswith("//")


class TestFibonacci:
    """Test the _fib helper and Fibonacci template."""

    def test_fib_0(self):
        assert _fib(0) == 0

    def test_fib_1(self):
        assert _fib(1) == 1

    def test_fib_10(self):
        assert _fib(10) == 55

    def test_fib_40(self):
        assert _fib(40) == 102334155

    def test_fib_matches_template(self):
        """Template result lambda should match _fib."""
        fib_template = _RUN_TEMPLATES[4]
        for n in [0, 1, 5, 10, 20, 40]:
            expected = str(_fib(n))
            actual = fib_template["result"]({"n": n})
            assert actual == expected, f"Fib({n}): {actual} != {expected}"


class TestRunTemplateCorrectness:
    """Verify each template's result function produces correct output."""

    def test_string_count(self):
        tmpl = _RUN_TEMPLATES[0]
        result = tmpl["result"]({"word": "strawberry", "letter": "r"})
        assert result == "3"

    def test_string_count_zero(self):
        tmpl = _RUN_TEMPLATES[0]
        result = tmpl["result"]({"word": "hello", "letter": "z"})
        assert result == "0"

    def test_string_length(self):
        tmpl = _RUN_TEMPLATES[1]
        result = tmpl["result"]({"word": "supercalifragilisticexpialidocious"})
        assert result == "34"

    def test_reverse_string(self):
        tmpl = _RUN_TEMPLATES[2]
        result = tmpl["result"]({"word": "hello"})
        assert result == "olleh"

    def test_math_computation(self):
        tmpl = _RUN_TEMPLATES[3]
        result = tmpl["result"]({"a": 100, "b": 200, "c": 50})
        assert result == str(100 * 200 + 50)

    def test_char_at_position(self):
        tmpl = _RUN_TEMPLATES[5]
        result = tmpl["result"]({"word": "hello", "pos": 0})
        assert result == "h"

    def test_char_at_position_oob(self):
        tmpl = _RUN_TEMPLATES[5]
        result = tmpl["result"]({"word": "hi", "pos": 5})
        assert result == "(out of range)"

    def test_sort_chars(self):
        tmpl = _RUN_TEMPLATES[6]
        result = tmpl["result"]({"word": "dcba"})
        assert result == "abcd"

    def test_gcd(self):
        tmpl = _RUN_TEMPLATES[7]
        result = tmpl["result"]({"a": 12, "b": 8})
        assert result == str(math.gcd(12, 8))

    def test_count_vowels(self):
        tmpl = _RUN_TEMPLATES[8]
        result = tmpl["result"]({"word": "hello"})
        assert result == "2"  # e, o

    def test_is_palindrome_yes(self):
        tmpl = _RUN_TEMPLATES[9]
        result = tmpl["result"]({"word": "racecar"})
        assert result == "yes"

    def test_is_palindrome_no(self):
        tmpl = _RUN_TEMPLATES[9]
        result = tmpl["result"]({"word": "hello"})
        assert result == "no"


class TestRunTemplateCodeFormat:
    """Test that template code strings are valid for C++ compilation."""

    def test_all_templates_have_required_keys(self):
        for i, tmpl in enumerate(_RUN_TEMPLATES):
            assert "task" in tmpl, f"Template {i} missing 'task'"
            assert "thought1" in tmpl, f"Template {i} missing 'thought1'"
            assert "code" in tmpl, f"Template {i} missing 'code'"
            assert "gen" in tmpl, f"Template {i} missing 'gen'"
            assert "result" in tmpl, f"Template {i} missing 'result'"

    def test_all_templates_gen_doesnt_crash(self):
        rng = random.Random(42)
        for i, tmpl in enumerate(_RUN_TEMPLATES):
            for _ in range(10):
                try:
                    params = tmpl["gen"](rng)
                except Exception as e:
                    pytest.fail(f"Template {i} gen() crashed: {e}")

    def test_all_templates_result_matches_gen(self):
        """Verify result function works with gen-produced params."""
        rng = random.Random(42)
        for i, tmpl in enumerate(_RUN_TEMPLATES):
            for _ in range(5):
                params = tmpl["gen"](rng)
                try:
                    result = tmpl["result"](params)
                    assert isinstance(result, str), f"Template {i} result not string: {type(result)}"
                except Exception as e:
                    pytest.fail(f"Template {i} result() crashed with params={params}: {e}")

    def test_code_format_string_works(self):
        """Verify code .format(**params) doesn't crash."""
        rng = random.Random(42)
        for i, tmpl in enumerate(_RUN_TEMPLATES):
            for _ in range(5):
                params = tmpl["gen"](rng)
                try:
                    code = tmpl["code"].format(**params)
                    assert isinstance(code, str)
                except Exception as e:
                    pytest.fail(f"Template {i} code.format() crashed: {e}")

    def test_task_format_string_works(self):
        """Verify task .format(**params) doesn't crash."""
        rng = random.Random(42)
        for i, tmpl in enumerate(_RUN_TEMPLATES):
            for _ in range(5):
                params = tmpl["gen"](rng)
                try:
                    task = tmpl["task"].format(**params)
                    assert isinstance(task, str)
                except Exception as e:
                    pytest.fail(f"Template {i} task.format() crashed: {e}")


class TestGenerateRunExamples:
    """Test the generate_run_tool_examples function."""

    def test_generates_correct_count(self):
        examples = list(generate_run_tool_examples(max_examples=50, rng=random.Random(42)))
        assert len(examples) == 50

    def test_all_have_required_fields(self):
        for ex in generate_run_tool_examples(max_examples=20, rng=random.Random(42)):
            assert "text" in ex
            assert "source" in ex
            assert ex["source"] == "run_code"

    def test_text_has_special_tokens(self):
        """Every example should have the full special token structure."""
        for ex in generate_run_tool_examples(max_examples=20, rng=random.Random(42)):
            text = ex["text"]
            assert text.startswith(BOS), f"Missing BOS: {text[:50]}"
            assert text.endswith(EOS), f"Missing EOS: {text[-50:]}"
            assert THOUGHT_START in text, f"Missing THOUGHT_START"
            assert THOUGHT_END in text, f"Missing THOUGHT_END"
            assert QUERY_TOOL in text, f"Missing QUERY_TOOL"
            assert TOOL_RESULT in text, f"Missing TOOL_RESULT"
            assert CODE_START in text, f"Missing CODE_START"
            assert CODE_END in text, f"Missing CODE_END"

    def test_special_token_ordering(self):
        """Tokens should appear in correct order."""
        for ex in generate_run_tool_examples(max_examples=20, rng=random.Random(42)):
            text = ex["text"]
            # BOS < THOUGHT_START < THOUGHT_END < QUERY_TOOL < TOOL_RESULT < CODE_START < CODE_END < EOS
            positions = {
                "BOS": text.index(BOS),
                "THOUGHT_START_1": text.index(THOUGHT_START),
                "THOUGHT_END_1": text.index(THOUGHT_END),
                "QUERY_TOOL": text.index(QUERY_TOOL),
                "TOOL_RESULT": text.index(TOOL_RESULT),
                "CODE_START": text.index(CODE_START),
                "EOS": text.rindex(EOS),
            }
            assert positions["BOS"] < positions["THOUGHT_START_1"]
            assert positions["THOUGHT_START_1"] < positions["THOUGHT_END_1"]
            assert positions["THOUGHT_END_1"] < positions["QUERY_TOOL"]
            assert positions["QUERY_TOOL"] < positions["TOOL_RESULT"]
            assert positions["TOOL_RESULT"] < positions["CODE_START"]
            assert positions["CODE_START"] < positions["EOS"]

    def test_run_code_escaping(self):
        """BUG CHECK: The run() code body is escaped for embedding in tool call string.
        Verify the escaping produces valid results."""
        for ex in generate_run_tool_examples(max_examples=100, rng=random.Random(42)):
            text = ex["text"]
            # Extract the run("...") call
            match = re.search(r'run\("(.+?)"\)', text, re.DOTALL)
            if match:
                run_arg = match.group(1)
                # Should not have unescaped newlines (they should be \\n)
                assert "\n" not in run_arg or "\\n" in text, f"Unescaped newline in run arg"


class TestGenerateWithDataFiles:
    """Test data generation with actual JSONL inputs."""

    def _make_docstring_jsonl(self, path, n=10):
        records = []
        for i in range(n):
            r = {
                "docstring": f"Compute the sum of two integers with index {i}.",
                "signature": f"int add_{i}(int a, int b)",
                "body": f"return a + b + {i};",
                "path": f"project/src/math_{i}.cpp",
            }
            records.append(r)
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return records

    def _make_diff_jsonl(self, path, n=5):
        records = []
        for i in range(n):
            r = {
                "instruction": f"Fix the following bug:\nfix undefined variable x{i}\nBefore:\n```cpp\nint f() {{ return x{i}; }}\n```",
                "response": f"```cpp\nint f() {{ int x{i} = 0; return x{i}; }}\n```",
            }
            records.append(r)
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return records

    def _make_gspo_jsonl(self, path, n=3):
        records = []
        for i in range(n):
            r = {
                "prompt": f"/* Compute factorial of n */\nint factorial(int n)",
                "canonical_solution": f"    if (n <= 1) return 1;\n    return n * factorial(n-1);",
            }
            records.append(r)
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return records

    def test_strategy_a_docstring_search(self, tmp_path):
        doc_path = tmp_path / "docs.jsonl"
        records = self._make_docstring_jsonl(str(doc_path))
        path_index = build_path_index(records)

        examples = list(generate_docstring_search_examples(
            records, path_index, max_examples=5, rng=random.Random(42)
        ))
        assert len(examples) > 0
        for ex in examples:
            assert ex["source"] == "docstring_search"
            assert QUERY_TOOL in ex["text"]
            assert "search(" in ex["text"]

    def test_strategy_b_diff_compile(self, tmp_path):
        diff_path = tmp_path / "diffs.jsonl"
        self._make_diff_jsonl(str(diff_path))

        examples = list(generate_diff_compile_examples(
            str(diff_path), max_examples=5, rng=random.Random(42)
        ))
        assert len(examples) > 0
        for ex in examples:
            assert ex["source"] == "diff_compile"
            assert "compile(" in ex["text"]

    def test_strategy_c_humaneval(self, tmp_path):
        gspo_path = tmp_path / "gspo.jsonl"
        self._make_gspo_jsonl(str(gspo_path))

        examples = list(generate_humaneval_ask_examples(
            str(gspo_path), max_examples=5, rng=random.Random(42)
        ))
        assert len(examples) > 0
        for ex in examples:
            assert ex["source"] == "humaneval_ask"
            assert "ask(" in ex["text"]

    def test_strategy_d_no_tool(self, tmp_path):
        doc_path = tmp_path / "docs.jsonl"
        diff_path = tmp_path / "diffs.jsonl"
        records = self._make_docstring_jsonl(str(doc_path))
        self._make_diff_jsonl(str(diff_path))

        examples = list(generate_no_tool_examples(
            records, str(diff_path),
            max_docstring=5, max_diff=3, max_fim=3,
            rng=random.Random(42)
        ))
        assert len(examples) > 0
        sources = [ex["source"] for ex in examples]
        assert "no_tool" in sources

    def test_no_nested_bos_eos(self):
        """Generated text should have exactly 1 BOS at start and 1 EOS at end."""
        for ex in generate_run_tool_examples(max_examples=50, rng=random.Random(42)):
            text = ex["text"]
            assert text.count(BOS) == 1, f"Multiple BOS tokens"
            assert text.count(EOS) == 1, f"Multiple EOS tokens"

    def test_code_end_count_matches_structure(self):
        """CODE_END should appear the right number of times."""
        for ex in generate_run_tool_examples(max_examples=20, rng=random.Random(42)):
            text = ex["text"]
            # Should have exactly 3 CODE_END: after QUERY_TOOL, after TOOL_RESULT, after CODE_START
            code_end_count = text.count(CODE_END)
            assert code_end_count == 3, f"Expected 3 CODE_END, got {code_end_count} in: {text[:200]}"
