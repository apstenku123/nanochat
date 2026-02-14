"""Extract git commit history as raw JSONL for the Rust cpp-chunker commit mode.

Extracts per-file commit diffs from git repos and outputs JSONL with:
  {old_content, new_content, diff, subject, body, filepath, repo}

The Rust cpp-chunker --commit-mode then processes these with tree-sitter to
extract function/class chains and produce training documents.

Usage:
    # Extract raw commit data
    python3 scripts/data/extract_git_history.py \
        --repo ~/data/cpp_raw/opencv \
        --output /mnt/nvme/nanochat_data/opencv_commits.jsonl \
        --max_commits 50000

    # Process with Rust tool
    ./cpp-chunker --commit-mode --inputs opencv_commits.jsonl \
        --output opencv_training.jsonl --max-tokens 16384
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

# C/C++ file extensions
CPP_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".c++",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".h++",
    ".inl",
    ".inc",
    ".ipp",
    ".tcc",
    ".tpp",
}

# Files/paths to skip
SKIP_PATTERNS = {
    "test/",
    "tests/",
    "testing/",
    "unittest/",
    "benchmarks/",
    "third_party/",
    "3rdparty/",
    "vendor/",
    "external/",
    "deps/",
    "generated/",
    "auto_generated/",
    "cmake-build",
    ".pb.h",
    ".pb.cc",
    "_generated.h",
    ".gen.cc",
    ".gen.h",
}

MAX_DIFF_CHARS = 50000
MAX_FILES_PER_COMMIT = 5
MIN_DIFF_CHARS = 50


def is_cpp_file(path: str) -> bool:
    return Path(path).suffix.lower() in CPP_EXTENSIONS


def should_skip_path(path: str) -> bool:
    path_lower = path.lower()
    return any(p in path_lower for p in SKIP_PATTERNS)


def run_git(repo_path: str, args: list[str], timeout: int = 60) -> Optional[str]:
    cmd = ["git", "-C", repo_path] + args
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, errors="replace"
        )
        if result.returncode != 0:
            return None
        return result.stdout
    except (subprocess.TimeoutExpired, OSError):
        return None


def get_commit_list(repo_path: str, max_commits: int = 0) -> list[str]:
    args = ["log", "--format=%H", "--no-merges", "--diff-filter=M"]
    if max_commits > 0:
        args.extend(["-n", str(max_commits)])
    output = run_git(repo_path, args, timeout=120)
    if not output:
        return []
    return output.strip().split("\n")


def get_commit_info(repo_path: str, commit_hash: str) -> Optional[dict]:
    fmt = "%H%n%s%n%b"
    output = run_git(repo_path, ["show", "-s", f"--format={fmt}", commit_hash])
    if not output:
        return None
    lines = output.strip().split("\n")
    if len(lines) < 2:
        return None
    return {
        "hash": lines[0],
        "subject": lines[1],
        "body": "\n".join(lines[2:]).strip() if len(lines) > 2 else "",
    }


def get_commit_diffs(repo_path: str, commit_hash: str) -> Optional[list[dict]]:
    """Get per-file diffs for C/C++ files in a commit."""
    name_status = run_git(
        repo_path, ["diff-tree", "--no-commit-id", "-r", "--name-status", commit_hash]
    )
    if not name_status:
        return None

    files = []
    for line in name_status.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status = parts[0][0]
        filepath = parts[-1]
        if not is_cpp_file(filepath):
            continue
        if should_skip_path(filepath):
            continue
        if status not in ("M",):
            continue
        files.append(filepath)

    if not files or len(files) > MAX_FILES_PER_COMMIT:
        return None

    results = []
    for filepath in files:
        old_content = run_git(
            repo_path, ["show", f"{commit_hash}^:{filepath}"], timeout=30
        )
        new_content = run_git(
            repo_path, ["show", f"{commit_hash}:{filepath}"], timeout=30
        )
        diff = run_git(
            repo_path,
            ["diff", f"{commit_hash}^", commit_hash, "--", filepath],
            timeout=30,
        )

        if old_content is None or new_content is None or diff is None:
            continue
        diff_len = len(diff)
        if diff_len < MIN_DIFF_CHARS or diff_len > MAX_DIFF_CHARS:
            continue
        if len(old_content) > 200000 or len(new_content) > 200000:
            continue

        results.append(
            {
                "filepath": filepath,
                "old_content": old_content,
                "new_content": new_content,
                "diff": diff,
            }
        )

    return results if results else None


def process_repo(
    repo_path: str,
    output_file,
    max_commits: int = 0,
    repo_name: str = "",
) -> dict:
    if not repo_name:
        repo_name = Path(repo_path).name

    stats = {"repo": repo_name, "commits_checked": 0, "records_written": 0}

    depth_output = run_git(repo_path, ["rev-list", "--count", "HEAD"])
    if depth_output:
        commit_count = int(depth_output.strip())
        if commit_count <= 1:
            print(f"  [{repo_name}] Shallow clone (1 commit), skipping")
            return stats
        print(f"  [{repo_name}] {commit_count:,} commits available")
    else:
        print(f"  [{repo_name}] Cannot count commits, skipping")
        return stats

    commits = get_commit_list(repo_path, max_commits)
    if not commits:
        print(f"  [{repo_name}] No commits found")
        return stats

    for i, commit_hash in enumerate(commits):
        if i > 0 and i % 1000 == 0:
            print(
                f"  [{repo_name}] Processed {i:,}/{len(commits):,} commits, "
                f"{stats['records_written']:,} records"
            )

        stats["commits_checked"] += 1

        commit_info = get_commit_info(repo_path, commit_hash)
        if not commit_info:
            continue

        subject = commit_info["subject"].lower()
        if any(
            skip in subject
            for skip in [
                "merge branch",
                "merge pull request",
                "update submodule",
                "bump version",
                "auto-generated",
                "clang-format",
                "fix whitespace",
                "fix typo in comment",
            ]
        ):
            continue

        file_diffs = get_commit_diffs(repo_path, commit_hash)
        if not file_diffs:
            continue

        for fd in file_diffs:
            record = {
                "old_content": fd["old_content"],
                "new_content": fd["new_content"],
                "diff": fd["diff"],
                "subject": commit_info["subject"],
                "body": commit_info["body"],
                "filepath": fd["filepath"],
                "repo": repo_name,
            }
            output_file.write(json.dumps(record) + "\n")
            stats["records_written"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract raw git commit data as JSONL for cpp-chunker --commit-mode"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--repo", help="Path to a single git repository")
    group.add_argument("--repo_dir", help="Directory containing multiple repos")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument(
        "--max_commits", type=int, default=0, help="Max commits per repo (0 = all)"
    )
    args = parser.parse_args()

    repos = []
    if args.repo:
        repos.append(args.repo)
    else:
        for entry in sorted(os.listdir(args.repo_dir)):
            path = os.path.join(args.repo_dir, entry)
            if os.path.isdir(os.path.join(path, ".git")):
                repos.append(path)

    print(f"Found {len(repos)} repositories")
    print(f"Max commits per repo: {args.max_commits or 'all'}")
    print(f"Output: {args.output}")
    print()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    total_records = 0
    start_time = time.time()

    with open(args.output, "w") as f:
        for i, repo_path in enumerate(repos):
            repo_name = Path(repo_path).name
            print(f"[{i + 1}/{len(repos)}] {repo_name}...")

            try:
                stats = process_repo(repo_path, f, args.max_commits, repo_name)
                total_records += stats["records_written"]
                print(f"  [{repo_name}] {stats['records_written']:,} records")
            except Exception as e:
                print(f"  [{repo_name}] ERROR: {e}")

    elapsed = time.time() - start_time
    output_size = os.path.getsize(args.output)
    print("\n=== SUMMARY ===")
    print(f"Repos: {len(repos)}")
    print(f"Total records: {total_records:,}")
    print(f"Time: {elapsed:.0f}s ({elapsed / 60:.1f}m)")
    print(f"Output: {args.output} ({output_size / (1024**3):.2f} GB)")


if __name__ == "__main__":
    main()
