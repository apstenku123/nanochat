#!/usr/bin/env python3
"""Fast diff extraction using 'git log -p' (bulk blob fetch, no per-commit overhead).

Usage: python extract_diffs_fast.py [repo_dir] [output.jsonl] [--max-commits N]
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

MAX_DIFF_BYTES = 50_000
MIN_DIFF_BYTES = 100
MAX_COMMITS = int(os.environ.get("MAX_COMMITS_PER_REPO", "0"))

SKIP_MSG_PATTERNS = re.compile(
    r"(auto-generated|bump.version|merge.pull.request|"
    r"update.submodule|dependabot|renovate|"
    r"cherry.picked from|Revert \"Revert)",
    re.IGNORECASE,
)

CPP_EXTENSIONS = {".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hxx"}

COMMIT_SEP = "---COMMIT_SEP---"


def has_cpp_extension(path: str) -> bool:
    return any(path.endswith(ext) for ext in CPP_EXTENSIONS)


def parse_git_log_stream(proc, repo_name: str, out_file, stats: dict):
    """Parse streaming output of git log -p --no-merges."""
    current_hash = None
    current_msg_lines = []
    current_diff_lines = []
    in_msg = False
    in_diff = False
    commit_count = 0
    extracted = 0

    def flush_commit():
        nonlocal extracted
        if current_hash is None:
            return

        msg = "\n".join(current_msg_lines).strip()
        if SKIP_MSG_PATTERNS.search(msg):
            stats["skipped_msg"] += 1
            return

        # Filter diff to only C/C++ files
        cpp_diff_lines = []
        in_cpp_file = False
        for line in current_diff_lines:
            if line.startswith("diff --git"):
                # Check if this file has a C++ extension
                match = re.match(r"diff --git a/(.+?) b/(.+)", line)
                if match and has_cpp_extension(match.group(2)):
                    in_cpp_file = True
                    cpp_diff_lines.append(line)
                else:
                    in_cpp_file = False
            elif in_cpp_file:
                cpp_diff_lines.append(line)

        if not cpp_diff_lines:
            stats["skipped_empty"] += 1
            return

        diff = "\n".join(cpp_diff_lines)
        diff_bytes = len(diff.encode("utf-8", errors="replace"))

        if diff_bytes > MAX_DIFF_BYTES or diff_bytes < MIN_DIFF_BYTES:
            stats["skipped_size"] += 1
            return

        files = re.findall(r"^diff --git a/(.+?) b/", diff, re.MULTILINE)
        ins = sum(1 for l in cpp_diff_lines if l.startswith("+") and not l.startswith("+++"))
        dels = sum(1 for l in cpp_diff_lines if l.startswith("-") and not l.startswith("---"))

        record = {
            "repo": repo_name,
            "commit_hash": current_hash,
            "commit_msg": msg,
            "diff": diff,
            "files_changed": files,
            "insertions": ins,
            "deletions": dels,
        }
        out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        extracted += 1
        stats["extracted"] += 1
        stats["total_bytes"] += diff_bytes

    for raw_line in proc.stdout:
        line = raw_line.rstrip("\n")

        if line.startswith("commit ") and len(line) == 47:  # "commit " + 40 hex
            # Flush previous commit
            flush_commit()
            commit_count += 1
            current_hash = line[7:]
            current_msg_lines = []
            current_diff_lines = []
            in_msg = True
            in_diff = False

            if commit_count % 10000 == 0:
                print(f"  {repo_name}: {commit_count} commits, {extracted} extracted", flush=True)
            if MAX_COMMITS > 0 and commit_count >= MAX_COMMITS:
                break
            continue

        if current_hash is None:
            continue

        if in_msg and line.startswith("diff --git"):
            in_msg = False
            in_diff = True
            current_diff_lines.append(line)
        elif in_msg:
            # Skip Author:/Date: header lines
            if not (line.startswith("Author:") or line.startswith("Date:") or line.startswith("Merge:")):
                current_msg_lines.append(line.lstrip())
        elif in_diff:
            current_diff_lines.append(line)

    # Flush last commit
    flush_commit()
    stats["total_commits"] += commit_count
    print(f"  {repo_name}: done. {commit_count} commits, {extracted} extracted", flush=True)


def extract_repo(repo_path: Path, out_file, stats: dict):
    repo_name = repo_path.name
    print(f"Processing {repo_name}...", flush=True)

    cmd = [
        "git", "log", "--all", "--no-merges", "-p",
        "--format=commit %H%nAuthor: %an%nDate: %ad%n%n%B",
        "--diff-filter=ACMR",  # Added, Copied, Modified, Renamed only
    ]

    proc = subprocess.Popen(
        cmd, cwd=repo_path, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, bufsize=1,
        errors="replace",
    )

    try:
        parse_git_log_stream(proc, repo_name, out_file, stats)
    finally:
        proc.terminate()
        proc.wait()


def main():
    repo_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/cpp_full_history")
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/cpp_diffs_fast.jsonl")

    if not repo_dir.exists():
        print(f"Repo directory {repo_dir} does not exist")
        sys.exit(1)

    repos = sorted([d for d in repo_dir.iterdir() if (d / ".git").exists()])
    if not repos:
        print(f"No git repos found in {repo_dir}")
        sys.exit(1)

    print(f"Found {len(repos)} repos: {[r.name for r in repos]}")

    stats = {"total_commits": 0, "extracted": 0, "skipped_msg": 0,
             "skipped_size": 0, "skipped_empty": 0, "total_bytes": 0}

    with open(output_path, "w") as f:
        for repo in repos:
            extract_repo(repo, f, stats)

    print("\n=== Statistics ===")
    print(f"Total non-merge commits scanned: {stats['total_commits']}")
    print(f"Diffs extracted: {stats['extracted']}")
    print(f"Skipped (empty/no C++): {stats['skipped_empty']}")
    print(f"Skipped (too large/small): {stats['skipped_size']}")
    print(f"Skipped (auto-generated msg): {stats['skipped_msg']}")
    print(f"Total diff bytes: {stats['total_bytes']:,}")
    est_tokens = stats['total_bytes'] // 4
    print(f"Estimated tokens (~4 bytes/token): {est_tokens:,}")
    print(f"Output: {output_path} ({output_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
