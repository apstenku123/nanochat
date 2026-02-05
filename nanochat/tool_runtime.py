"""
Tool runtime for C++ code-native agent.

Implements tool backends that the engine dispatches to when the model
emits <QUERY_TOOL> function_call(...) <CODE_END>.

Each tool is a Python function that takes parsed arguments and returns
a string result (which gets injected as <TOOL_RESULT> ... <CODE_END>).
"""

import os
import re
import subprocess
import shlex
from typing import Optional


class ToolRuntime:
    """Manages tool dispatch and execution for the C++ agent.

    Args:
        codebase_dir: Root directory for search/read_file operations.
        gemma_endpoint: URL for the Gemma 270M ask() tool (or None to disable).
        compile_cmd: Compiler command (default: g++).
    """

    def __init__(
        self,
        codebase_dir: str = ".",
        gemma_endpoint: Optional[str] = None,
        compile_cmd: str = "g++",
    ):
        self.codebase_dir = os.path.abspath(codebase_dir)
        self.gemma_endpoint = gemma_endpoint
        self.compile_cmd = compile_cmd

        self._dispatch = {
            "search": self.tool_search,
            "ask": self.tool_ask,
            "read_file": self.tool_read_file,
            "compile": self.tool_compile,
            "test": self.tool_test,
            "run": self.tool_run,
        }

    def execute(self, expr: str) -> Optional[str]:
        """Parse a C++ function call expression and dispatch to the appropriate tool.

        Args:
            expr: A string like 'search("query")' or 'read_file("path", 10, 20)'.

        Returns:
            Tool result as a string, or None if the expression is invalid.
        """
        expr = expr.strip()
        if not expr:
            return None

        # Parse: function_name(arg1, arg2, ...)
        match = re.match(r'(\w+)\s*\((.*)\)$', expr, re.DOTALL)
        if not match:
            return None

        func_name = match.group(1)
        args_str = match.group(2).strip()

        handler = self._dispatch.get(func_name)
        if handler is None:
            return f"// error: unknown tool '{func_name}'"

        # Parse arguments
        args = self._parse_args(args_str)
        if args is None:
            return f"// error: failed to parse arguments for {func_name}"

        try:
            return handler(*args)
        except TypeError as e:
            return f"// error: {func_name}: {e}"
        except Exception as e:
            return f"// error: {func_name} failed: {e}"

    def _parse_args(self, args_str: str) -> Optional[list]:
        """Parse C++ function call arguments.

        Handles:
        - String literals: "hello", 'hello'
        - Integers: 42
        - Empty args: ()
        """
        if not args_str:
            return []

        args = []
        i = 0
        while i < len(args_str):
            c = args_str[i]
            if c in ' \t\n':
                i += 1
                continue
            if c == ',':
                i += 1
                continue

            # String literal
            if c in ('"', "'"):
                quote = c
                j = i + 1
                s = []
                while j < len(args_str):
                    ch = args_str[j]
                    if ch == '\\' and j + 1 < len(args_str):
                        nc = args_str[j + 1]
                        if nc == 'n':
                            s.append('\n')
                        elif nc == 't':
                            s.append('\t')
                        elif nc == '\\':
                            s.append('\\')
                        elif nc == quote:
                            s.append(quote)
                        else:
                            s.append(nc)
                        j += 2
                    elif ch == quote:
                        j += 1
                        break
                    else:
                        s.append(ch)
                        j += 1
                args.append(''.join(s))
                i = j
                continue

            # Integer
            if c.isdigit() or (c == '-' and i + 1 < len(args_str) and args_str[i + 1].isdigit()):
                j = i + 1 if c == '-' else i
                while j < len(args_str) and args_str[j].isdigit():
                    j += 1
                args.append(int(args_str[i:j]))
                i = j
                continue

            # Unknown character — skip
            i += 1

        return args

    # --- Tool implementations ---

    def tool_search(self, query: str, max_results: int = 5) -> str:
        """Search the codebase using ripgrep."""
        try:
            result = subprocess.run(
                ["rg", "--max-count", str(max_results), "--no-heading",
                 "-n", "--type", "cpp", "--type", "c", query, self.codebase_dir],
                capture_output=True, text=True, timeout=10,
            )
            output = result.stdout.strip()
            if not output:
                return f"// No results found for: {query}"
            # Limit output length
            lines = output.split("\n")
            if len(lines) > max_results * 3:
                lines = lines[:max_results * 3]
            return "\n".join(f"// {line}" for line in lines)
        except FileNotFoundError:
            return "// error: ripgrep (rg) not found"
        except subprocess.TimeoutExpired:
            return "// error: search timed out"

    def tool_ask(self, prompt: str) -> str:
        """Query the Gemma 270M router model for quick answers."""
        if self.gemma_endpoint is None:
            # Placeholder: return a generic helpful comment
            return f"// (ask tool not connected) Query: {prompt[:100]}"

        try:
            import urllib.request
            import json
            data = json.dumps({"prompt": prompt, "max_tokens": 200}).encode()
            req = urllib.request.Request(
                self.gemma_endpoint,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
                return f"// {result.get('text', result.get('response', str(result)))}"
        except Exception as e:
            return f"// error: ask failed: {e}"

    def tool_read_file(self, path: str, start: int = 0, end: int = 0) -> str:
        """Read a file or portion of a file."""
        # Resolve relative to codebase dir
        full_path = os.path.join(self.codebase_dir, path) if not os.path.isabs(path) else path

        # Security: don't allow reading outside codebase
        full_path = os.path.abspath(full_path)
        if not full_path.startswith(self.codebase_dir):
            return f"// error: path outside codebase: {path}"

        if not os.path.exists(full_path):
            return f"// error: file not found: {path}"

        try:
            with open(full_path) as f:
                lines = f.readlines()

            if start > 0 or end > 0:
                # 1-indexed line range
                start_idx = max(0, start - 1)
                end_idx = end if end > 0 else len(lines)
                lines = lines[start_idx:end_idx]

            # Limit output
            if len(lines) > 30:
                lines = lines[:30]
                lines.append("// ... (truncated)\n")

            return "".join(lines).rstrip()
        except Exception as e:
            return f"// error: read failed: {e}"

    def tool_compile(self, code: str, flags: str = "") -> str:
        """Try to compile a C++ code snippet (syntax check only)."""
        cmd = [self.compile_cmd, "-fsyntax-only", "-x", "c++", "-std=c++17", "-"]
        if flags:
            cmd.extend(shlex.split(flags))

        try:
            result = subprocess.run(
                cmd,
                input=code, capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                if result.stderr.strip():
                    return f"// Compiled with warnings:\n// {result.stderr.strip()[:300]}"
                return "// Compilation successful"
            else:
                errors = result.stderr.strip()[:500]
                return f"// Compilation errors:\n// {errors}"
        except FileNotFoundError:
            return f"// error: compiler '{self.compile_cmd}' not found"
        except subprocess.TimeoutExpired:
            return "// error: compilation timed out"

    def tool_run(self, code: str) -> str:
        """Compile and run a C++ code snippet, return stdout.

        The model generates C++ code that gets compiled and executed.
        This enables the model to compute things it can't do natively
        (counting, string ops, math) by writing and running C++ code.

        The code should be a complete program with main() or a snippet
        that will be wrapped in a minimal main() if needed.
        """
        import tempfile

        # If the code doesn't have main(), wrap it
        if "main" not in code:
            code = f'#include <iostream>\n#include <string>\n#include <algorithm>\nusing namespace std;\nint main() {{\n{code}\n    return 0;\n}}'

        try:
            with tempfile.NamedTemporaryFile(suffix=".cpp", mode="w", delete=False) as src:
                src.write(code)
                src_path = src.name
            out_path = src_path.replace(".cpp", "")

            # Compile
            comp = subprocess.run(
                [self.compile_cmd, "-std=c++17", "-o", out_path, src_path],
                capture_output=True, text=True, timeout=10,
            )
            if comp.returncode != 0:
                return f"// Compile error:\n// {comp.stderr.strip()[:300]}"

            # Run with timeout and memory limit
            run = subprocess.run(
                [out_path],
                capture_output=True, text=True, timeout=5,
            )

            # Clean up
            os.unlink(src_path)
            if os.path.exists(out_path):
                os.unlink(out_path)

            output = run.stdout.strip()
            if run.returncode != 0:
                stderr = run.stderr.strip()[:200]
                return f"// Runtime error:\n// {stderr}\n// stdout: {output[:200]}"

            if not output:
                return "// (no output)"
            return f"// Output: {output[:500]}"

        except subprocess.TimeoutExpired:
            return "// error: execution timed out (5s limit)"
        except FileNotFoundError:
            return f"// error: compiler '{self.compile_cmd}' not found"
        except Exception as e:
            return f"// error: run failed: {e}"
        finally:
            # Cleanup on any path
            for p in [src_path, out_path]:
                try:
                    if os.path.exists(p):
                        os.unlink(p)
                except Exception:
                    pass

    def tool_test(self, test_name: str) -> str:
        """Run a test (placeholder — needs project-specific configuration)."""
        return f"// test '{test_name}': not configured (set up test runner in tool_runtime)"
