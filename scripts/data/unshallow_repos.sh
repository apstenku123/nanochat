#!/bin/bash
# Unshallow key C/C++ repos for git history extraction
# Only unshallows repos that are worth the disk/time cost
#
# Usage: bash scripts/data/unshallow_repos.sh

set -euo pipefail

TARGET_DIR="${1:-$HOME/data/cpp_raw}"

# These repos have rich C++ commit history worth extracting
# Ordered by expected value (quality * quantity of C++ commits)
REPOS=(
    # Medium-size repos with high-quality C++ commits
    opencv          # ~40K commits, great C++ with algos
    rocksdb         # ~15K commits, database C++
    clickhouse      # ~50K commits, modern C++
    grpc            # ~20K commits, networking C++
    protobuf        # ~15K commits, serialization
    folly           # ~8K commits, Facebook C++ libs
    abseil-cpp      # ~5K commits, Google C++ libs
    fmt             # ~5K commits, formatting lib
    spdlog          # ~3K commits, logging
    json            # ~8K commits, nlohmann json
    godot           # ~60K commits, game engine
    blender         # ~100K commits, 3D graphics
    duckdb          # ~15K commits, analytics DB
    envoy           # ~20K commits, proxy server

    # Large C repos
    redis           # ~15K commits, data structure server
    curl            # ~30K commits, networking
    ffmpeg          # ~100K commits, multimedia
    openssl         # ~30K commits, crypto
    postgres        # ~80K commits, database

    # Very large (unshallow if time allows)
    # tensorflow    # ~150K commits — very large
    # pytorch       # ~100K commits — very large
    # llvm-project  # ~500K commits — huge
    # linux         # ~1.3M commits — enormous
)

echo "=== Unshallowing repos for history extraction ==="
echo "Target: $TARGET_DIR"
echo "Repos: ${#REPOS[@]}"
echo ""

unshallowed=0
skipped=0
failed=0

for repo in "${REPOS[@]}"; do
    dir="$TARGET_DIR/$repo"
    if [ ! -d "$dir/.git" ]; then
        echo "[SKIP] $repo (not found)"
        ((skipped++)) || true
        continue
    fi

    # Check if already unshallowed
    depth=$(cd "$dir" && git rev-list --count HEAD 2>/dev/null || echo "0")
    if [ "$depth" -gt 10 ]; then
        echo "[ALREADY] $repo ($depth commits)"
        ((skipped++)) || true
        continue
    fi

    echo -n "[UNSHALLOW] $repo ... "
    if (cd "$dir" && git fetch --unshallow 2>/dev/null); then
        new_depth=$(cd "$dir" && git rev-list --count HEAD 2>/dev/null || echo "?")
        echo "OK ($new_depth commits)"
        ((unshallowed++)) || true
    else
        echo "FAIL"
        ((failed++)) || true
    fi
done

echo ""
echo "=== Summary ==="
echo "Unshallowed: $unshallowed"
echo "Skipped: $skipped"
echo "Failed: $failed"
