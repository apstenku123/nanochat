"""
Tests for tool-call SFT masking correctness and special-token protocol.
"""

import pytest
from nanochat.tool_sft_dataset import (
    compute_loss_mask,
    BOS_ID, EOS_ID, CODE_START_ID, CODE_END_ID,
    THOUGHT_START_ID, THOUGHT_END_ID, QUERY_TOOL_ID, TOOL_RESULT_ID,
    FIM_PREFIX_ID, FIM_MIDDLE_ID, FIM_SUFFIX_ID,
)


class TestComputeLossMask:
    """Tests for the compute_loss_mask function."""

    def test_instruction_prefix_masked(self):
        """Tokens before first response token should be masked (0)."""
        # BOS + some instruction tokens + THOUGHT_START + content + THOUGHT_END
        tokens = [BOS_ID, 100, 101, 102, THOUGHT_START_ID, 200, 201, THOUGHT_END_ID]
        mask = compute_loss_mask(tokens)

        # BOS and instruction (100, 101, 102) → 0
        assert mask[0] == 0  # BOS
        assert mask[1] == 0  # instruction
        assert mask[2] == 0  # instruction
        assert mask[3] == 0  # instruction
        # THOUGHT_START onwards → 1
        assert mask[4] == 1  # THOUGHT_START
        assert mask[5] == 1  # content
        assert mask[6] == 1  # content
        assert mask[7] == 1  # THOUGHT_END

    def test_thought_block_trained(self):
        """THOUGHT_START...THOUGHT_END blocks should be trained (mask=1)."""
        tokens = [THOUGHT_START_ID, 100, 101, 102, THOUGHT_END_ID]
        mask = compute_loss_mask(tokens)
        assert all(m == 1 for m in mask)

    def test_query_tool_block_trained(self):
        """QUERY_TOOL...CODE_END blocks should be trained (mask=1)."""
        tokens = [QUERY_TOOL_ID, 100, 101, CODE_END_ID]
        mask = compute_loss_mask(tokens)
        assert all(m == 1 for m in mask)

    def test_tool_result_block_masked(self):
        """TOOL_RESULT...CODE_END blocks should NOT be trained (mask=0)."""
        tokens = [
            THOUGHT_START_ID, 100, THOUGHT_END_ID,  # thought (trained)
            TOOL_RESULT_ID, 200, 201, 202, CODE_END_ID,  # tool result (masked)
            CODE_START_ID, 300, CODE_END_ID,  # code (trained)
        ]
        mask = compute_loss_mask(tokens)

        # Thought block: all 1
        assert mask[0] == 1  # THOUGHT_START
        assert mask[1] == 1  # content
        assert mask[2] == 1  # THOUGHT_END

        # Tool result block: all 0
        assert mask[3] == 0  # TOOL_RESULT
        assert mask[4] == 0  # result content
        assert mask[5] == 0  # result content
        assert mask[6] == 0  # result content
        assert mask[7] == 0  # CODE_END closing tool result

        # Code block: all 1
        assert mask[8] == 1  # CODE_START
        assert mask[9] == 1  # code content
        assert mask[10] == 1  # CODE_END

    def test_code_start_block_trained(self):
        """CODE_START...CODE_END blocks should be trained (mask=1)."""
        tokens = [CODE_START_ID, 100, 101, 102, CODE_END_ID]
        mask = compute_loss_mask(tokens)
        assert all(m == 1 for m in mask)

    def test_bos_eos_always_masked(self):
        """BOS and EOS tokens should always be masked (0)."""
        tokens = [BOS_ID, THOUGHT_START_ID, 100, THOUGHT_END_ID, EOS_ID]
        mask = compute_loss_mask(tokens)
        assert mask[0] == 0  # BOS (instruction prefix)
        assert mask[4] == 0  # EOS

    def test_full_sequence_masking(self):
        """Full realistic sequence with instruction, thought, tool call, result, code."""
        tokens = [
            BOS_ID,              # 0: mask=0 (BOS)
            100, 101,            # 1-2: mask=0 (instruction)
            THOUGHT_START_ID,    # 3: mask=1 (thought)
            200, 201,            # 4-5: mask=1 (thought content)
            THOUGHT_END_ID,      # 6: mask=1 (thought end)
            QUERY_TOOL_ID,       # 7: mask=1 (tool call)
            300, 301,            # 8-9: mask=1 (call expression)
            CODE_END_ID,         # 10: mask=1 (end of tool call)
            TOOL_RESULT_ID,      # 11: mask=0 (result injected)
            400, 401, 402,       # 12-14: mask=0 (result content)
            CODE_END_ID,         # 15: mask=0 (end of result)
            THOUGHT_START_ID,    # 16: mask=1 (second thought)
            500,                 # 17: mask=1 (thought content)
            THOUGHT_END_ID,      # 18: mask=1 (thought end)
            CODE_START_ID,       # 19: mask=1 (final code)
            600, 601,            # 20-21: mask=1 (code)
            CODE_END_ID,         # 22: mask=1 (code end)
            EOS_ID,              # 23: mask=0 (EOS)
        ]
        mask = compute_loss_mask(tokens)

        expected = [
            0,     # BOS
            0, 0,  # instruction
            1,     # THOUGHT_START
            1, 1,  # thought content
            1,     # THOUGHT_END
            1,     # QUERY_TOOL
            1, 1,  # tool call expr
            1,     # CODE_END (tool call)
            0,     # TOOL_RESULT
            0, 0, 0,  # result content
            0,     # CODE_END (tool result)
            1,     # THOUGHT_START
            1,     # thought content
            1,     # THOUGHT_END
            1,     # CODE_START
            1, 1,  # code
            1,     # CODE_END (code)
            0,     # EOS
        ]
        assert mask == expected, f"\nGot:      {mask}\nExpected: {expected}"

    def test_multiple_tool_calls(self):
        """Sequence with two tool calls, both results should be masked."""
        tokens = [
            THOUGHT_START_ID, 100, THOUGHT_END_ID,
            # First tool call
            QUERY_TOOL_ID, 200, CODE_END_ID,
            TOOL_RESULT_ID, 300, 301, CODE_END_ID,
            # Second tool call
            QUERY_TOOL_ID, 400, CODE_END_ID,
            TOOL_RESULT_ID, 500, 501, CODE_END_ID,
            # Final code
            CODE_START_ID, 600, CODE_END_ID,
        ]
        mask = compute_loss_mask(tokens)

        # First tool call: trained
        assert mask[3] == 1  # QUERY_TOOL
        assert mask[4] == 1  # expression
        assert mask[5] == 1  # CODE_END

        # First result: masked
        assert mask[6] == 0  # TOOL_RESULT
        assert mask[7] == 0  # result
        assert mask[8] == 0  # result
        assert mask[9] == 0  # CODE_END

        # Second tool call: trained
        assert mask[10] == 1  # QUERY_TOOL
        assert mask[11] == 1  # expression
        assert mask[12] == 1  # CODE_END

        # Second result: masked
        assert mask[13] == 0  # TOOL_RESULT
        assert mask[14] == 0  # result
        assert mask[15] == 0  # result
        assert mask[16] == 0  # CODE_END

        # Final code: trained
        assert mask[17] == 1  # CODE_START
        assert mask[18] == 1  # code
        assert mask[19] == 1  # CODE_END

    def test_no_response_tokens_all_masked(self):
        """If no response tokens exist, everything is masked."""
        tokens = [BOS_ID, 100, 101, 102, EOS_ID]
        mask = compute_loss_mask(tokens)
        assert all(m == 0 for m in mask)

    def test_fim_masking(self):
        """FIM sequences: only content after FIM_MIDDLE is trained."""
        tokens = [
            FIM_PREFIX_ID, 100, 101,  # prefix (masked)
            FIM_SUFFIX_ID, 200, 201,  # suffix (masked)
            FIM_MIDDLE_ID, 300, 301,  # middle content (TRAINED)
            EOS_ID,                   # EOS (masked)
        ]
        mask = compute_loss_mask(tokens)

        # Prefix and suffix parts: masked
        assert mask[0] == 0  # FIM_PREFIX
        assert mask[1] == 0  # prefix content
        assert mask[2] == 0  # prefix content
        assert mask[3] == 0  # FIM_SUFFIX
        assert mask[4] == 0  # suffix content
        assert mask[5] == 0  # suffix content

        # FIM_MIDDLE and content: trained
        assert mask[6] == 1  # FIM_MIDDLE
        assert mask[7] == 1  # infill content
        assert mask[8] == 1  # infill content

        # EOS: masked
        assert mask[9] == 0

    def test_empty_thought_block(self):
        """Empty thought block should still be trained."""
        tokens = [THOUGHT_START_ID, THOUGHT_END_ID, CODE_START_ID, 100, CODE_END_ID]
        mask = compute_loss_mask(tokens)
        assert mask == [1, 1, 1, 1, 1]

    def test_run_tool_pattern(self):
        """Strategy E pattern: thought → run() tool call → result → thought → code."""
        tokens = [
            BOS_ID,              # instruction
            100, 101,            # task comment
            THOUGHT_START_ID,    # think about computation
            200, 201,
            THOUGHT_END_ID,
            QUERY_TOOL_ID,       # run("code")
            300, 301,
            CODE_END_ID,
            TOOL_RESULT_ID,      # // Output: 42
            400, 401,
            CODE_END_ID,
            THOUGHT_START_ID,    # summarize result
            500,
            THOUGHT_END_ID,
            CODE_START_ID,       # final answer
            600,
            CODE_END_ID,
            EOS_ID,
        ]
        mask = compute_loss_mask(tokens)

        # Instruction prefix
        assert mask[0] == 0  # BOS
        assert mask[1] == 0  # task
        assert mask[2] == 0  # task

        # Thought + tool call: trained
        for i in range(3, 11):  # THOUGHT_START through CODE_END of tool call
            assert mask[i] == 1, f"Position {i} should be 1, got {mask[i]}"

        # Tool result: masked
        assert mask[11] == 0  # TOOL_RESULT
        assert mask[12] == 0  # result content
        assert mask[13] == 0  # result content
        assert mask[14] == 0  # CODE_END

        # Second thought + code: trained
        for i in range(15, 21):
            assert mask[i] == 1, f"Position {i} should be 1, got {mask[i]}"

        # EOS: masked
        assert mask[21] == 0


class TestToolRuntime:
    """Tests for tool_runtime.py argument parsing and dispatch."""

    def test_parse_string_arg(self):
        from nanochat.tool_runtime import ToolRuntime
        rt = ToolRuntime()
        args = rt._parse_args('"hello world"')
        assert args == ["hello world"]

    def test_parse_int_arg(self):
        from nanochat.tool_runtime import ToolRuntime
        rt = ToolRuntime()
        args = rt._parse_args("42")
        assert args == [42]

    def test_parse_mixed_args(self):
        from nanochat.tool_runtime import ToolRuntime
        rt = ToolRuntime()
        args = rt._parse_args('"path/to/file.cpp", 10, 20')
        assert args == ["path/to/file.cpp", 10, 20]

    def test_parse_empty_args(self):
        from nanochat.tool_runtime import ToolRuntime
        rt = ToolRuntime()
        args = rt._parse_args("")
        assert args == []

    def test_parse_escape_sequences(self):
        from nanochat.tool_runtime import ToolRuntime
        rt = ToolRuntime()
        args = rt._parse_args(r'"hello\nworld"')
        assert args == ["hello\nworld"]

    def test_unknown_tool(self):
        from nanochat.tool_runtime import ToolRuntime
        rt = ToolRuntime()
        result = rt.execute("unknown_func()")
        assert "error" in result
        assert "unknown tool" in result

    def test_invalid_expression(self):
        from nanochat.tool_runtime import ToolRuntime
        rt = ToolRuntime()
        result = rt.execute("not a function call")
        assert result is None

    def test_read_file_outside_codebase(self):
        from nanochat.tool_runtime import ToolRuntime
        rt = ToolRuntime(codebase_dir="/tmp/test_codebase_nonexistent")
        result = rt.execute('read_file("/etc/passwd")')
        assert "error" in result

    def test_compile_valid_code(self):
        from nanochat.tool_runtime import ToolRuntime
        import shutil
        if shutil.which("g++") is None:
            pytest.skip("g++ not available")
        rt = ToolRuntime()
        result = rt.execute('compile("int main() { return 0; }")')
        assert "successful" in result.lower() or "warning" in result.lower()

    def test_compile_invalid_code(self):
        from nanochat.tool_runtime import ToolRuntime
        import shutil
        if shutil.which("g++") is None:
            pytest.skip("g++ not available")
        rt = ToolRuntime()
        result = rt.execute('compile("int main() { undefined_var = 1; }")')
        assert "error" in result.lower()

    def test_run_simple_program(self):
        from nanochat.tool_runtime import ToolRuntime
        import shutil
        if shutil.which("g++") is None:
            pytest.skip("g++ not available")
        rt = ToolRuntime()
        result = rt.execute('run("#include <iostream>\\nint main() { std::cout << 42; return 0; }")')
        assert "42" in result

    def test_run_auto_wrap_main(self):
        from nanochat.tool_runtime import ToolRuntime
        import shutil
        if shutil.which("g++") is None:
            pytest.skip("g++ not available")
        rt = ToolRuntime()
        result = rt.execute('run("cout << 7 * 6 << endl;")')
        assert "42" in result
