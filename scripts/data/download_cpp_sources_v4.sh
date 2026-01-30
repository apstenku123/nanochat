#!/bin/bash
# V4: More C++ sources to push toward 3B tokens
set -euo pipefail

DATA_DIR="${1:-/home/dave/Downloads/source/nanochat/data/cpp_raw}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "=== Downloading additional C++ sources (v4) ==="

clone_if_missing() {
    local dir="$1" url="$2"
    shift 2
    if [ ! -d "$dir" ]; then
        echo "--- Cloning $dir ---"
        git clone "$@" "$url" "$dir" || echo "WARN: Failed to clone $dir"
    else
        echo "--- $dir already exists, skipping ---"
    fi
}

# Language runtimes (large C/C++ codebases)
clone_if_missing mono            https://github.com/mono/mono.git                --depth=1
clone_if_missing dotnet-runtime  https://github.com/dotnet/runtime.git           --depth=1
clone_if_missing openjdk         https://github.com/openjdk/jdk.git              --depth=1
clone_if_missing ruby            https://github.com/ruby/ruby.git                --depth=1
clone_if_missing cpython         https://github.com/python/cpython.git           --depth=1
clone_if_missing php-src         https://github.com/php/php-src.git              --depth=1
clone_if_missing node            https://github.com/nodejs/node.git              --depth=1

# Databases
clone_if_missing scylladb        https://github.com/scylladb/scylladb.git        --depth=1
clone_if_missing foundationdb    https://github.com/apple/foundationdb.git       --depth=1

# Networking / security
clone_if_missing nginx           https://github.com/nginx/nginx.git              --depth=1
clone_if_missing curl            https://github.com/curl/curl.git                --depth=1
clone_if_missing openssl         https://github.com/openssl/openssl.git          --depth=1
clone_if_missing libuv           https://github.com/libuv/libuv.git              --depth=1
clone_if_missing wolfssl         https://github.com/wolfSSL/wolfssl.git          --depth=1

# Media
clone_if_missing ffmpeg          https://github.com/FFmpeg/FFmpeg.git            --depth=1
clone_if_missing opus            https://github.com/xiph/opus.git                --depth=1

# Science / numerics
clone_if_missing openblas        https://github.com/OpenMathLib/OpenBLAS.git     --depth=1

# Graphics / GUI
clone_if_missing glfw            https://github.com/glfw/glfw.git                --depth=1
clone_if_missing SFML            https://github.com/SFML/SFML.git                --depth=1
clone_if_missing wxWidgets       https://github.com/wxWidgets/wxWidgets.git      --depth=1

# OS / embedded
clone_if_missing freebsd-src     https://github.com/freebsd/freebsd-src.git      --depth=1
clone_if_missing zephyr          https://github.com/zephyrproject-rtos/zephyr.git --depth=1
clone_if_missing FreeRTOS        https://github.com/FreeRTOS/FreeRTOS.git        --depth=1

# C++ libraries
clone_if_missing spdlog          https://github.com/gabime/spdlog.git            --depth=1
clone_if_missing benchmark       https://github.com/google/benchmark.git         --depth=1
clone_if_missing entt            https://github.com/skypjack/entt.git            --depth=1
clone_if_missing range-v3        https://github.com/ericniebler/range-v3.git     --depth=1
clone_if_missing taskflow        https://github.com/taskflow/taskflow.git        --depth=1

# ML / AI (C++ parts)
clone_if_missing onnxruntime     https://github.com/microsoft/onnxruntime.git    --depth=1
clone_if_missing tvm             https://github.com/apache/tvm.git               --depth=1
clone_if_missing xgboost         https://github.com/dmlc/xgboost.git            --depth=1

echo ""
echo "=== V4 Downloads complete ==="
du -sh "$DATA_DIR"
