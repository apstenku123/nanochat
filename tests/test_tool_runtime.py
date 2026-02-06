from pathlib import Path

from nanochat.tool_runtime import ToolRuntime


def test_read_file_rejects_paths_outside_codebase(tmp_path: Path):
    codebase = tmp_path / "repo"
    codebase.mkdir()
    inside = codebase / "main.cpp"
    inside.write_text("int main() { return 0; }\n")

    sibling = tmp_path / "repo2"
    sibling.mkdir()
    secret = sibling / "secret.cpp"
    secret.write_text("top secret\n")

    rt = ToolRuntime(codebase_dir=str(codebase))

    assert "int main()" in rt.tool_read_file("main.cpp")
    assert "path outside codebase" in rt.tool_read_file("../repo2/secret.cpp")
    assert "path outside codebase" in rt.tool_read_file(str(secret))


def test_tool_run_handles_missing_compiler_without_cleanup_crash(tmp_path: Path):
    rt = ToolRuntime(codebase_dir=str(tmp_path), compile_cmd="__missing_compiler__")
    out = rt.tool_run("int main() { return 0; }")
    assert "compiler '__missing_compiler__' not found" in out
