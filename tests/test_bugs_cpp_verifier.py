"""
Deep edge-case tests for cpp_verifier.py.
"""
import pytest
import shutil

from nanochat.cpp_verifier import verify_cpp, compute_reward


@pytest.fixture(autouse=True)
def check_gpp():
    if shutil.which("g++") is None:
        pytest.skip("g++ not available")


class TestVerifyCpp:

    def test_empty_code(self):
        """Empty string should be valid C++ (nothing to compile)."""
        result = verify_cpp("")
        assert result["compile_ok"]

    def test_just_whitespace(self):
        result = verify_cpp("   \n\n\n  ")
        assert result["compile_ok"]

    def test_syntax_error(self):
        result = verify_cpp("int main() { return")
        assert not result["compile_ok"]
        assert result["stderr"]

    def test_int_main_in_comment_false_positive(self):
        """'int main' in a comment should NOT trigger linking.
        The regex r'\\bint\\s+main\\s*\\(' correctly ignores comments."""
        code = """
// This function is not int main
// int main() is documented here
int add(int a, int b) { return a + b; }
"""
        result = verify_cpp(code)
        # No real main() exists, so verify_cpp should do compile-only
        assert result["compile_ok"]

    def test_int_main_substring_false_positive(self):
        """'int maintain()' should NOT match as main — compile-only, no linking."""
        code = 'int maintain() { return 0; }'
        result = verify_cpp(code)
        # No real main(), so verify_cpp should do compile-only successfully
        assert result["compile_ok"]

    def test_compile_with_sanitizer_flags(self):
        code = '#include <cstdio>\nint main() { printf("hi\\n"); return 0; }'
        result = verify_cpp(code, extra_flags=["-fsanitize=address"])
        # May or may not work depending on asan availability
        assert result is not None

    def test_compile_timeout(self):
        """Very short timeout should trigger timeout handling."""
        code = '#include <cstdio>\nint main() { printf("hi\\n"); return 0; }'
        result = verify_cpp(code, timeout=0.001)
        # Should either compile fast enough or timeout
        assert result is not None

    def test_runtime_segfault(self):
        code = """
#include <cstdlib>
int main() {
    int* p = nullptr;
    return *p;
}
"""
        result = verify_cpp(code)
        assert result["compile_ok"]
        assert result["run_ok"] is False
        assert result["returncode"] != 0

    def test_runtime_exit_code_nonzero(self):
        code = 'int main() { return 42; }'
        result = verify_cpp(code)
        assert result["compile_ok"]
        assert result["run_ok"] is False
        assert result["returncode"] == 42

    def test_test_result_parsing_passed(self):
        code = '''
#include <cstdio>
int main() {
    printf("PASSED 3/5\\n");
    return 0;
}
'''
        result = verify_cpp(code)
        assert result["tests_passed"] == 3
        assert result["tests_total"] == 5

    def test_test_result_parsing_individual(self):
        code = '''
#include <cstdio>
int main() {
    printf("PASS\\nPASS\\nFAIL\\n");
    return 0;
}
'''
        result = verify_cpp(code)
        assert result["tests_passed"] == 2
        assert result["tests_total"] == 3

    def test_test_result_parsing_no_tests(self):
        code = '#include <cstdio>\nint main() { printf("hello\\n"); return 0; }'
        result = verify_cpp(code)
        assert result["tests_passed"] == 0
        assert result["tests_total"] == 0

    def test_large_stdout(self):
        """Program with lots of output should be captured."""
        code = '''
#include <cstdio>
int main() {
    for (int i = 0; i < 10000; i++) printf("line %d\\n", i);
    return 0;
}
'''
        result = verify_cpp(code)
        assert result["compile_ok"]
        assert result["run_ok"]
        assert len(result["stdout"]) > 0

    def test_c20_features(self):
        """Default standard is c++20."""
        code = '''
#include <span>
#include <vector>
void f(std::span<int> s) {}
int main() {
    std::vector<int> v = {1,2,3};
    f(v);
    return 0;
}
'''
        result = verify_cpp(code)
        assert result["compile_ok"]

    def test_compile_only_no_main(self):
        """Code without main should compile-only (no link, no run)."""
        code = "int add(int a, int b) { return a + b; }"
        result = verify_cpp(code)
        assert result["compile_ok"]
        assert result["run_ok"] is None  # Not run


class TestComputeReward:

    def test_compile_fail(self):
        assert compute_reward({"compile_ok": False, "run_ok": None, "tests_passed": 0, "tests_total": 0}) == 0.0

    def test_compile_only(self):
        assert compute_reward({"compile_ok": True, "run_ok": None, "tests_passed": 0, "tests_total": 0}) == 0.3

    def test_runs_no_tests(self):
        assert compute_reward({"compile_ok": True, "run_ok": True, "tests_passed": 0, "tests_total": 0}) == 0.5

    def test_runs_runtime_error(self):
        assert compute_reward({"compile_ok": True, "run_ok": False, "tests_passed": 0, "tests_total": 0}) == 0.3

    def test_all_tests_pass(self):
        assert compute_reward({"compile_ok": True, "run_ok": True, "tests_passed": 5, "tests_total": 5}) == 1.0

    def test_half_tests_pass(self):
        reward = compute_reward({"compile_ok": True, "run_ok": True, "tests_passed": 2, "tests_total": 4})
        assert abs(reward - 0.65) < 0.01  # 0.3 + 0.7 * 0.5

    def test_no_tests_pass(self):
        reward = compute_reward({"compile_ok": True, "run_ok": True, "tests_passed": 0, "tests_total": 5})
        assert abs(reward - 0.3) < 0.01  # 0.3 + 0.7 * 0

    def test_tests_with_runtime_error(self):
        """Tests exist but program crashed — still use test fraction."""
        reward = compute_reward({"compile_ok": True, "run_ok": False, "tests_passed": 3, "tests_total": 5})
        expected = 0.3 + 0.7 * (3/5)
        assert abs(reward - expected) < 0.01
