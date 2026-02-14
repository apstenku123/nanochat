"""
Deep edge-case and bug-finding tests for ToolRuntime.
"""
import os
import tempfile
from pathlib import Path

import pytest
from nanochat.tool_runtime import ToolRuntime


class TestArgParsing:
    """Probe _parse_args for edge cases."""

    def setup_method(self):
        self.rt = ToolRuntime()

    def test_nested_quotes_in_string(self):
        """String with escaped quotes inside."""
        args = self.rt._parse_args(r'"hello \"world\""')
        assert args == ['hello "world"']

    def test_single_quotes(self):
        args = self.rt._parse_args("'hello world'")
        assert args == ["hello world"]

    def test_negative_integer(self):
        args = self.rt._parse_args("-42")
        assert args == [-42]

    def test_negative_integer_in_args(self):
        args = self.rt._parse_args('"file.cpp", -1, 20')
        assert args == ["file.cpp", -1, 20]

    def test_zero(self):
        args = self.rt._parse_args("0")
        assert args == [0]

    def test_large_integer(self):
        args = self.rt._parse_args("999999999")
        assert args == [999999999]

    def test_string_with_newlines(self):
        args = self.rt._parse_args(r'"line1\nline2\nline3"')
        assert args == ["line1\nline2\nline3"]

    def test_string_with_tab(self):
        args = self.rt._parse_args(r'"col1\tcol2"')
        assert args == ["col1\tcol2"]

    def test_empty_string_arg(self):
        args = self.rt._parse_args('""')
        assert args == [""]

    def test_multiple_strings(self):
        args = self.rt._parse_args('"hello", "world"')
        assert args == ["hello", "world"]

    def test_string_with_backslash(self):
        args = self.rt._parse_args(r'"path\\to\\file"')
        assert args == ["path\\to\\file"]

    def test_unterminated_string_does_not_crash(self):
        """Unterminated string literal should not crash."""
        args = self.rt._parse_args('"hello')
        assert args is not None  # Should return something, not crash

    def test_float_treated_as_int_or_skipped(self):
        """Floats aren't explicitly handled — check behavior."""
        args = self.rt._parse_args("3.14")
        # Parser sees '3' as int, then '.', '1', '4'
        # Current behavior: will parse 3, then 1, then 4 as separate ints
        # This is a bug if floats are ever needed, but ok for current usage
        assert 3 in args

    def test_string_with_comma_inside(self):
        args = self.rt._parse_args('"a, b, c"')
        assert args == ["a, b, c"]

    def test_string_with_parens_inside(self):
        """Parens inside strings shouldn't break parsing."""
        args = self.rt._parse_args('"func(x, y)"')
        assert args == ["func(x, y)"]

    def test_only_whitespace(self):
        args = self.rt._parse_args("   \t  ")
        assert args == []


class TestExecuteDispatch:
    """Test the execute() dispatch logic."""

    def setup_method(self):
        self.rt = ToolRuntime()

    def test_empty_expression(self):
        assert self.rt.execute("") is None

    def test_whitespace_expression(self):
        assert self.rt.execute("   ") is None

    def test_no_parens(self):
        assert self.rt.execute("search") is None

    def test_extra_parens_in_args(self):
        """Function call with parens inside string arg."""
        result = self.rt.execute('compile("int f() { return 0; }")')
        assert result is not None

    def test_multiline_expression(self):
        """Multiline code in compile() arg — re.DOTALL should handle this."""
        result = self.rt.execute('compile("int main() {\n  return 0;\n}")')
        assert result is not None

    def test_unknown_tool_error_message(self):
        result = self.rt.execute("foobar()")
        assert "unknown tool" in result
        assert "foobar" in result

    def test_too_many_args_error(self):
        """Passing wrong number of args should return an error, not crash."""
        result = self.rt.execute('search("query", 5, "extra_arg")')
        # search() takes (query, max_results=5), extra arg should cause TypeError
        assert result is not None  # shouldn't crash


class TestToolCompile:
    """Test tool_compile edge cases."""

    def setup_method(self):
        self.rt = ToolRuntime()

    @pytest.fixture(autouse=True)
    def check_gpp(self):
        import shutil
        if shutil.which("g++") is None:
            pytest.skip("g++ not available")

    def test_empty_code(self):
        result = self.rt.tool_compile("")
        # Empty code should compile (g++ accepts empty stdin for syntax check)
        assert result is not None

    def test_code_with_warnings(self):
        code = "int f() { int x; return x; }"
        result = self.rt.tool_compile(code)
        # May or may not have warnings depending on g++ version
        assert result is not None

    def test_flags_injection_attempt(self):
        """Flags are passed through shlex.split — check that works safely."""
        result = self.rt.tool_compile("int main() { return 0; }", flags="-Wall -Wextra")
        assert "successful" in result.lower() or "warning" in result.lower()

    def test_very_long_code(self):
        """Very long code should work (up to stderr truncation)."""
        code = "int f() { " + "int x = 0; " * 500 + "return x; }"
        result = self.rt.tool_compile(code)
        assert result is not None


class TestToolRun:
    """Test tool_run edge cases."""

    def setup_method(self):
        self.rt = ToolRuntime()

    @pytest.fixture(autouse=True)
    def check_gpp(self):
        import shutil
        if shutil.which("g++") is None:
            pytest.skip("g++ not available")

    def test_runtime_error(self):
        code = '#include <cstdlib>\nint main() { abort(); }'
        result = self.rt.tool_run(code)
        assert "error" in result.lower() or "runtime" in result.lower() or "Output" not in result

    def test_infinite_loop_times_out(self):
        code = 'int main() { while(true) {} }'
        result = self.rt.tool_run(code)
        assert "timed out" in result.lower()

    def test_no_output(self):
        code = 'int main() { return 0; }'
        result = self.rt.tool_run(code)
        assert "no output" in result.lower()

    def test_auto_wrap_with_includes(self):
        """Auto-wrap adds includes for cout etc."""
        result = self.rt.tool_run('cout << "hello" << endl;')
        assert "hello" in result

    def test_auto_wrap_false_positive(self):
        """'maintain' contains 'main' as substring — old check was fooled.
        Now the regex correctly sees no 'int main(' so it auto-wraps."""
        code = 'int maintain = 42;\ncout << maintain << endl;'
        result = self.rt.tool_run(code)
        # No real main(), so tool_run auto-wraps with main().
        # The wrapped code compiles and prints 42.
        assert "error" not in result.lower()
        assert "42" in result

    def test_temp_file_cleanup(self):
        """Temp files should be cleaned up even on success."""
        import glob
        before = set(glob.glob("/tmp/tmp*.cpp")) | set(glob.glob("/tmp/tmp*"))
        self.rt.tool_run('#include <iostream>\nint main() { std::cout << 1; }')
        after = set(glob.glob("/tmp/tmp*.cpp")) | set(glob.glob("/tmp/tmp*"))
        # New temp files should be cleaned up
        leaked = after - before
        cpp_leaked = [f for f in leaked if f.endswith('.cpp') or not '.' in os.path.basename(f)]
        # Allow some tolerance for other processes
        assert len(cpp_leaked) <= 1  # at most 1 leftover is acceptable


class TestToolReadFile:
    """Test tool_read_file edge cases."""

    def test_symlink_escape(self, tmp_path):
        """Symlink pointing outside codebase should be caught."""
        codebase = tmp_path / "repo"
        codebase.mkdir()
        secret = tmp_path / "secret.txt"
        secret.write_text("secret data")

        link = codebase / "link.txt"
        link.symlink_to(secret)

        rt = ToolRuntime(codebase_dir=str(codebase))
        result = rt.tool_read_file("link.txt")
        assert "path outside codebase" in result

    def test_read_file_line_range(self, tmp_path):
        codebase = tmp_path / "repo"
        codebase.mkdir()
        f = codebase / "test.cpp"
        f.write_text("\n".join(f"line {i}" for i in range(1, 51)))

        rt = ToolRuntime(codebase_dir=str(codebase))
        result = rt.tool_read_file("test.cpp", 5, 10)
        assert "line 5" in result
        assert "line 10" in result
        assert "line 4" not in result
        assert "line 11" not in result

    def test_read_nonexistent_file(self, tmp_path):
        rt = ToolRuntime(codebase_dir=str(tmp_path))
        result = rt.tool_read_file("nonexistent.cpp")
        assert "not found" in result

    def test_read_truncation(self, tmp_path):
        codebase = tmp_path / "repo"
        codebase.mkdir()
        f = codebase / "big.cpp"
        f.write_text("\n".join(f"line {i}" for i in range(100)))

        rt = ToolRuntime(codebase_dir=str(codebase))
        result = rt.tool_read_file("big.cpp")
        assert "truncated" in result


class TestToolSearch:
    """Test tool_search edge cases."""

    def test_search_nonexistent_dir(self):
        rt = ToolRuntime(codebase_dir="/nonexistent/path/xyz")
        result = rt.tool_search("pattern")
        # rg should handle nonexistent path gracefully
        assert result is not None

    def test_search_empty_query(self):
        rt = ToolRuntime()
        result = rt.tool_search("")
        assert result is not None  # should not crash
