#!/bin/bash
# V3: Download more large C++ codebases to reach 3B+ tokens
set -euo pipefail

DATA_DIR="${1:-/home/dave/Downloads/source/nanochat/data/cpp_raw}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "=== Downloading additional C++ sources (v3) ==="

clone_if_missing() {
    local dir="$1" url="$2"
    shift 2
    if [ ! -d "$dir" ]; then
        echo "--- Cloning $dir ---"
        git clone "$@" "$url" "$dir" || echo "WARN: Failed to clone $dir, continuing..."
    else
        echo "--- $dir already exists, skipping ---"
    fi
}

# GCC - huge C/C++ compiler source
clone_if_missing gcc-mirror   https://github.com/gcc-mirror/gcc.git          --depth=1

# Game engines / graphics
clone_if_missing godot        https://github.com/godotengine/godot.git       --depth=1
clone_if_missing filament     https://github.com/google/filament.git         --depth=1
clone_if_missing imgui        https://github.com/ocornut/imgui.git           --depth=1
clone_if_missing SDL          https://github.com/libsdl-org/SDL.git          --depth=1
clone_if_missing vulkan-hpp   https://github.com/KhronosGroup/Vulkan-Hpp.git --depth=1

# Databases
clone_if_missing rocksdb      https://github.com/facebook/rocksdb.git        --depth=1
clone_if_missing leveldb      https://github.com/google/leveldb.git          --depth=1
clone_if_missing duckdb       https://github.com/duckdb/duckdb.git           --depth=1
clone_if_missing clickhouse   https://github.com/ClickHouse/ClickHouse.git   --depth=1
clone_if_missing mariadb      https://github.com/MariaDB/server.git          --depth=1
clone_if_missing mysql        https://github.com/mysql/mysql-server.git      --depth=1
clone_if_missing postgres     https://github.com/postgres/postgres.git       --depth=1
clone_if_missing redis        https://github.com/redis/redis.git             --depth=1
clone_if_missing mongodb-cxx  https://github.com/mongodb/mongo-cxx-driver.git --depth=1

# Networking / infrastructure
clone_if_missing envoy        https://github.com/envoyproxy/envoy.git        --depth=1
clone_if_missing seastar      https://github.com/scylladb/seastar.git        --depth=1
clone_if_missing cpprestsdk   https://github.com/microsoft/cpprestsdk.git    --depth=1
clone_if_missing asio         https://github.com/chriskohlhoff/asio.git      --depth=1

# Serialization / IPC
clone_if_missing flatbuffers  https://github.com/google/flatbuffers.git      --depth=1
clone_if_missing capnproto    https://github.com/capnproto/capnproto.git     --depth=1
clone_if_missing arrow        https://github.com/apache/arrow.git            --depth=1

# Build tools / compilers / linkers
clone_if_missing mold         https://github.com/rui314/mold.git             --depth=1
clone_if_missing tbb          https://github.com/oneapi-src/oneTBB.git       --depth=1
clone_if_missing swift        https://github.com/swiftlang/swift.git         --depth=1

# Scientific / math
clone_if_missing z3           https://github.com/Z3Prover/z3.git             --depth=1
clone_if_missing VTK          https://github.com/Kitware/VTK.git             --depth=1
clone_if_missing ITK          https://github.com/InsightSoftwareConsortium/ITK.git --depth=1
clone_if_missing dealii       https://github.com/dealii/dealii.git           --depth=1
clone_if_missing openexr      https://github.com/AcademySoftwareFoundation/openexr.git --depth=1

# Graphics / VFX
clone_if_missing mesa         https://gitlab.freedesktop.org/mesa/mesa.git   --depth=1
clone_if_missing blender      https://projects.blender.org/blender/blender.git --depth=1

# Search
clone_if_missing re2          https://github.com/google/re2.git              --depth=1

# Storage
clone_if_missing ceph         https://github.com/ceph/ceph.git               --depth=1

echo ""
echo "=== V3 Downloads complete ==="
du -sh "$DATA_DIR"
