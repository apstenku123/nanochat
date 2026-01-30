#!/usr/bin/env python3
"""Extract C++ commit diffs from git repos as JSONL for training."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Limits
MAX_DIFF_BYTES = 50_000
MIN_DIFF_BYTES = 100
MAX_COMMITS = int(os.environ.get("MAX_COMMITS_PER_REPO", "0"))  # 0 = unlimited

# File extensions to include in diffs
CPP_EXTS = ("*.cpp", "*.cc", "*.cxx", "*.c", "*.h", "*.hpp", "*.hxx")

# Patterns in commit messages to skip
SKIP_MSG_PATTERNS = re.compile(
    r"(auto-generated|bump.version|merge.pull.request|"
    r"update.submodule|dependabot|renovate|"
    r"Merge branch|Merge remote|Merge tag|"
    r"cherry.picked from|Revert \"Revert)",
    re.IGNORECASE,
)


def git(*args, cwd=None, timeout=60):
    """Run git command, return stdout or None on error."""
    try:
        r = subprocess.run(
            ["git"] + list(args),
            cwd=cwd, capture_output=True, text=True, timeout=timeout,
        )
        if r.returncode != 0:
            return None
        return r.stdout
    except (subprocess.TimeoutExpired, Exception):
        return None


def extract_repo(repo_path: Path, out_file, stats: dict):
    repo_name = repo_path.name
    print(f"Processing {repo_name}...", flush=True)

    # Get all commit hashes (non-merge only)
    raw = git("log", "--all", "--no-merges", "--format=%H", cwd=repo_path, timeout=300)
    if not raw:
        print(f"  No commits found in {repo_name}")
        return

    hashes = raw.strip().split("\n")
    total = len(hashes)
    if MAX_COMMITS > 0:
        hashes = hashes[:MAX_COMMITS]
    print(f"  {total} non-merge commits, processing {len(hashes)}", flush=True)

    extracted = 0
    skipped_msg = 0
    skipped_size = 0
    skipped_empty = 0

    for i, h in enumerate(hashes):
        if i > 0 and i % 5000 == 0:
            print(f"  {repo_name}: {i}/{len(hashes)} commits, {extracted} extracted", flush=True)

        # Get commit message
        msg = git("log", "-1", "--format=%B", h, cwd=repo_path, timeout=10)
        if not msg:
            continue
        msg = msg.strip()

        if SKIP_MSG_PATTERNS.search(msg):
            skipped_msg += 1
            continue

        # Get diff (C/C++ files only)
        diff_args = ["diff", f"{h}~1", h, "--"] + list(CPP_EXTS)
        diff = git(*diff_args, cwd=repo_path, timeout=30)
        if diff is None:
            # Might be first commit; try --root
            diff = git("diff", "--root", h, "--", *CPP_EXTS, cwd=repo_path, timeout=30)
        if not diff:
            skipped_empty += 1
            continue

        diff_bytes = len(diff.encode("utf-8", errors="replace"))
        if diff_bytes > MAX_DIFF_BYTES or diff_bytes < MIN_DIFF_BYTES:
            skipped_size += 1
            continue

        # Parse changed files from diff header
        files = re.findall(r"^diff --git a/(.+?) b/", diff, re.MULTILINE)

        # Count insertions/deletions
        ins = sum(1 for line in diff.split("\n") if line.startswith("+") and not line.startswith("+++"))
        dels = sum(1 for line in diff.split("\n") if line.startswith("-") and not line.startswith("---"))

        record = {
            "repo": repo_name,
            "commit_hash": h,
            "commit_msg": msg,
            "diff": diff,
            "files_changed": files,
            "insertions": ins,
            "deletions": dels,
        }
        out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        extracted += 1
        stats["total_bytes"] += diff_bytes

    stats["total_commits"] += total
    stats["extracted"] += extracted
    stats["skipped_msg"] += skipped_msg
    stats["skipped_size"] += skipped_size
    stats["skipped_empty"] += skipped_empty
    print(f"  {repo_name}: extracted {extracted} diffs "
          f"(skipped: {skipped_empty} empty, {skipped_size} size, {skipped_msg} message)")


def main():
    repo_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/cpp_full_history")
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/cpp_diffs.jsonl")

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
