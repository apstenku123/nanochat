"""
C++ code verifier using g++ compiler.

Compiles C++ code snippets and optionally runs them with test harnesses.
Used by the GRPO RL training loop to compute rewards for generated code.
"""

import subprocess
import tempfile
import os
import re
from typing import Optional


def verify_cpp(
    code: str,
    test_code: str = "",
    timeout: float = 10.0,
    std: str = "c++20",
    extra_flags: Optional[list] = None,
) -> dict:
    """Compile and optionally run C++ code.

    Args:
        code: The C++ source code to verify.
        test_code: Optional test harness that #includes or uses the code.
                   If provided, it is appended after the code with a main().
        timeout: Maximum seconds for compilation and execution.
        std: C++ standard to use (e.g. "c++17", "c++20").
        extra_flags: Additional g++ flags (e.g. ["-O2", "-fsanitize=address"]).

    Returns:
        dict with keys:
            compile_ok (bool): Whether compilation succeeded.
            warnings (str): Compiler warnings (if any).
            run_ok (bool | None): Whether execution succeeded (None if not run).
            stdout (str): Program stdout.
            stderr (str): Compiler + runtime stderr combined.
            returncode (int | None): Program exit code (None if not run).
            tests_passed (int): Number of tests passed (parsed from output).
            tests_total (int): Total number of tests (parsed from output).
    """
    result = dict(
        compile_ok=False,
        warnings="",
        run_ok=None,
        stdout="",
        stderr="",
        returncode=None,
        tests_passed=0,
        tests_total=0,
    )

    if extra_flags is None:
        extra_flags = []

    with tempfile.TemporaryDirectory(prefix="nanochat_verify_") as tmpdir:
        src_path = os.path.join(tmpdir, "solution.cpp")
        bin_path = os.path.join(tmpdir, "solution")

        # Build the full source: code + optional test harness
        full_source = code
        if test_code:
            full_source = code + "\n\n" + test_code

        with open(src_path, "w") as f:
            f.write(full_source)

        # Decide whether to compile-only or compile+link
        has_main = "int main" in full_source
        if has_main:
            compile_cmd = [
                "g++", f"-std={std}", "-Wall", "-Wextra",
                "-o", bin_path, src_path,
            ] + extra_flags
        else:
            # No main: compile only (syntax/type check), don't link
            compile_cmd = [
                "g++", f"-std={std}", "-Wall", "-Wextra",
                "-c", "-o", os.path.join(tmpdir, "solution.o"), src_path,
            ] + extra_flags

        try:
            comp = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            result["stderr"] = f"Compilation timed out after {timeout}s"
            return result
        except FileNotFoundError:
            result["stderr"] = "g++ not found"
            return result

        result["stderr"] = comp.stderr
        result["warnings"] = comp.stderr if comp.returncode == 0 else ""

        if comp.returncode != 0:
            # Compilation failed
            return result

        result["compile_ok"] = True

        # If no main function, skip execution
        if not has_main:
            return result

        # Run the binary
        try:
            run = subprocess.run(
                [bin_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            result["run_ok"] = False
            result["stderr"] += f"\nExecution timed out after {timeout}s"
            return result

        result["returncode"] = run.returncode
        result["stdout"] = run.stdout
        result["stderr"] += run.stderr
        result["run_ok"] = run.returncode == 0

        # Parse test results from stdout
        # Expected format: "PASSED X/Y" or individual "PASS"/"FAIL" lines
        passed_match = re.search(r"PASSED\s+(\d+)/(\d+)", run.stdout)
        if passed_match:
            result["tests_passed"] = int(passed_match.group(1))
            result["tests_total"] = int(passed_match.group(2))
        else:
            # Count individual PASS/FAIL lines
            pass_count = len(re.findall(r"^PASS", run.stdout, re.MULTILINE))
            fail_count = len(re.findall(r"^FAIL", run.stdout, re.MULTILINE))
            total = pass_count + fail_count
            if total > 0:
                result["tests_passed"] = pass_count
                result["tests_total"] = total

    return result


def compute_reward(result: dict) -> float:
    """Convert verification result to a scalar reward in [0, 1].

    Reward schedule:
        0.0 — does not compile
        0.3 — compiles but runtime error (or timeout)
        0.5 — compiles and runs cleanly (no tests provided)
        0.3 + 0.7 * (tests_passed / tests_total) — with test results
    """
    if not result["compile_ok"]:
        return 0.0

    # Not executed (e.g. no main)
    if result["run_ok"] is None:
        return 0.3

    if result["tests_total"] > 0:
        frac = result["tests_passed"] / result["tests_total"]
        return 0.3 + 0.7 * frac

    if result["run_ok"]:
        return 0.5

    # Compiled but runtime error
    return 0.3


# -------------------------------------------------------------------------
# Quick self-test
# -------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Test 1: correct code with main ===")
    r1 = verify_cpp('#include <cstdio>\nint main() { printf("hello\\n"); return 0; }')
    print(f"  compile_ok={r1['compile_ok']}, run_ok={r1['run_ok']}, stdout={r1['stdout'].strip()!r}")
    print(f"  reward={compute_reward(r1):.2f}")
    assert r1["compile_ok"] and r1["run_ok"] and compute_reward(r1) == 0.5

    print("=== Test 2: compile error ===")
    r2 = verify_cpp("int main() { undefined_func(); }")
    print(f"  compile_ok={r2['compile_ok']}")
    print(f"  reward={compute_reward(r2):.2f}")
    assert not r2["compile_ok"] and compute_reward(r2) == 0.0

    print("=== Test 3: runtime error ===")
    r3 = verify_cpp('#include <cstdlib>\nint main() { abort(); }')
    print(f"  compile_ok={r3['compile_ok']}, run_ok={r3['run_ok']}, returncode={r3['returncode']}")
    print(f"  reward={compute_reward(r3):.2f}")
    assert r3["compile_ok"] and not r3["run_ok"] and compute_reward(r3) == 0.3

    print("=== Test 4: code with test harness ===")
    code = "int add(int a, int b) { return a + b; }"
    test = """
#include <cstdio>
int main() {
    int passed = 0, total = 3;
    if (add(1, 2) == 3) { printf("PASS\\n"); passed++; } else { printf("FAIL\\n"); }
    if (add(0, 0) == 0) { printf("PASS\\n"); passed++; } else { printf("FAIL\\n"); }
    if (add(-1, 1) == 0) { printf("PASS\\n"); passed++; } else { printf("FAIL\\n"); }
    printf("PASSED %d/%d\\n", passed, total);
    return passed == total ? 0 : 1;
}
"""
    r4 = verify_cpp(code, test_code=test)
    print(f"  compile_ok={r4['compile_ok']}, run_ok={r4['run_ok']}, tests={r4['tests_passed']}/{r4['tests_total']}")
    print(f"  reward={compute_reward(r4):.2f}")
    assert r4["compile_ok"] and r4["run_ok"] and r4["tests_passed"] == 3 and compute_reward(r4) == 1.0

    print("=== Test 5: no main, no test (compile-only) ===")
    r5 = verify_cpp("int add(int a, int b) { return a + b; }")
    print(f"  compile_ok={r5['compile_ok']}, run_ok={r5['run_ok']}")
    print(f"  reward={compute_reward(r5):.2f}")
    assert r5["compile_ok"] and r5["run_ok"] is None and compute_reward(r5) == 0.3

    print("\nAll verifier tests passed!")
