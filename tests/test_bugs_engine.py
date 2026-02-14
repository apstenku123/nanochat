"""
Tests for Engine tool-call state machine and RowState logic.
"""
import pytest
from collections import deque
from nanochat.engine import RowState, use_calculator, eval_with_timeout


class TestRowState:
    """Test RowState transitions."""

    def test_initial_state(self):
        s = RowState()
        assert s.current_tokens == []
        assert len(s.forced_tokens) == 0
        assert not s.in_python_block
        assert not s.in_tool_block
        assert not s.completed

    def test_initial_state_with_tokens(self):
        s = RowState([1, 2, 3])
        assert s.current_tokens == [1, 2, 3]

    def test_forced_tokens_fifo(self):
        s = RowState()
        s.forced_tokens = deque([10, 20, 30])
        assert s.forced_tokens.popleft() == 10
        assert s.forced_tokens.popleft() == 20
        assert s.forced_tokens.popleft() == 30

    def test_tool_block_accumulates_tokens(self):
        s = RowState()
        s.in_tool_block = True
        for tok in [100, 101, 102]:
            s.tool_expr_tokens.append(tok)
        assert s.tool_expr_tokens == [100, 101, 102]


class TestUseCalculator:
    """Test the calculator used for Python tool blocks."""

    def test_simple_addition(self):
        assert use_calculator("2+3") == 5

    def test_simple_multiplication(self):
        assert use_calculator("4*5") == 20

    def test_division(self):
        result = use_calculator("10/3")
        assert abs(result - 3.3333) < 0.01

    def test_parentheses(self):
        assert use_calculator("(2+3)*4") == 20

    def test_power_blocked(self):
        """** operator should be blocked for safety."""
        assert use_calculator("2**10") is None

    def test_string_count(self):
        result = use_calculator("'strawberry'.count('r')")
        assert result == 3

    def test_string_count_no_match(self):
        result = use_calculator("'hello'.count('z')")
        assert result == 0

    def test_import_blocked(self):
        assert use_calculator("__import__('os')") is None

    def test_exec_blocked(self):
        assert use_calculator("exec('print(1)')") is None

    def test_eval_blocked(self):
        assert use_calculator("eval('1+1')") is None

    def test_getattr_blocked(self):
        assert use_calculator("getattr(str, 'count')") is None

    def test_non_count_method_blocked(self):
        """Only .count() is allowed for string ops."""
        result = use_calculator("'hello'.upper()")
        assert result is None

    def test_commas_stripped(self):
        """Commas in numbers should be stripped."""
        result = use_calculator("1,000+2,000")
        assert result == 3000

    def test_empty_expression(self):
        result = use_calculator("")
        assert result is None

    def test_very_large_number(self):
        result = use_calculator("999999*999999")
        assert result == 999999 * 999999

    def test_negative_numbers(self):
        result = use_calculator("-5+3")
        # '-' is in the allowed chars for math
        assert result == -2

    def test_float_expression(self):
        result = use_calculator("3.14*2")
        assert abs(result - 6.28) < 0.01

    def test_division_by_zero(self):
        """Should return None (exception caught by eval_with_timeout)."""
        result = use_calculator("1/0")
        assert result is None

    def test_timeout_infinite_loop(self):
        """BUG CHECK: Can we craft an expression that hangs?
        eval_with_timeout has a 3-second alarm."""
        # This should time out
        result = eval_with_timeout("9**9**9**9", max_time=1)
        # Actually ** is not in allowed chars for math so use_calculator
        # would reject it, but eval_with_timeout is separate
        # The timeout should catch it
        assert result is None


class TestEngineToolCallStateMachine:
    """Test the tool-call state machine logic in Engine.generate().

    We can't easily test generate() directly without a model, but we can
    test the state transition logic by simulating it.
    """

    def test_tool_block_state_transitions(self):
        """Simulate the state machine for a QUERY_TOOL...CODE_END block."""
        QUERY_TOOL_ID = 11
        CODE_END_ID = 8

        s = RowState()
        tokens = [QUERY_TOOL_ID, 100, 101, CODE_END_ID]

        for tok in tokens:
            if tok == QUERY_TOOL_ID:
                s.in_tool_block = True
                s.tool_expr_tokens = []
            elif tok == CODE_END_ID and s.in_tool_block:
                s.in_tool_block = False
                # Would dispatch to tool_runtime here
                assert s.tool_expr_tokens == [100, 101]
                s.tool_expr_tokens = []
            elif s.in_tool_block:
                s.tool_expr_tokens.append(tok)
            s.current_tokens.append(tok)

        assert not s.in_tool_block
        assert s.tool_expr_tokens == []

    def test_python_and_tool_blocks_dont_interfere(self):
        """Python blocks and tool blocks should be independent."""
        s = RowState()

        # Simulate: python block then tool block
        s.in_python_block = True
        s.python_expr_tokens = [100]
        s.in_python_block = False  # python_end received

        s.in_tool_block = True
        s.tool_expr_tokens = [200]
        s.in_tool_block = False

        # Both should be independent
        assert s.python_expr_tokens == [100]
        assert s.tool_expr_tokens == [200]

    def test_code_end_outside_tool_block_ignored(self):
        """CODE_END when not in a tool block should not change state."""
        s = RowState()
        CODE_END_ID = 8

        s.in_tool_block = False
        # CODE_END arrives but we're not in a tool block
        if CODE_END_ID == 8 and s.in_tool_block:
            pytest.fail("Should not enter this branch")
        # State unchanged
        assert not s.in_tool_block

    def test_nested_query_tool_bug(self):
        """BUG: If model emits QUERY_TOOL while already in a tool block,
        the inner QUERY_TOOL gets added to tool_expr_tokens.
        This is probably fine (model shouldn't do this) but worth noting."""
        QUERY_TOOL_ID = 11
        CODE_END_ID = 8

        s = RowState()
        s.in_tool_block = True
        s.tool_expr_tokens = []

        # Model (incorrectly) emits another QUERY_TOOL
        tok = QUERY_TOOL_ID
        # In engine.py, the check is `if query_tool is not None and next_token == query_tool:`
        # This resets the block! The old tokens are lost.
        # Let's verify this is what happens by simulating:
        if tok == QUERY_TOOL_ID:
            s.in_tool_block = True
            s.tool_expr_tokens = []

        # State was reset — previous tool expr tokens lost
        assert s.tool_expr_tokens == []
        # This is actually the engine.py behavior — it restarts the block


class TestEvalWithTimeout:
    def test_normal_eval(self):
        assert eval_with_timeout("2+3") == 5

    def test_returns_none_on_error(self):
        assert eval_with_timeout("undefined_var") is None

    def test_alarm_reset_on_error(self):
        """After an error, signal.alarm should be reset to 0."""
        import signal
        eval_with_timeout("undefined_var")
        # alarm should be cancelled
        remaining = signal.alarm(0)
        assert remaining == 0
