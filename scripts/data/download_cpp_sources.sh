#!/bin/bash
# Download C++ source code for tokenizer training (M0: ~2-5GB target)
# Sources: LLVM, Boost, Linux kernel headers, stb, json.hpp, fmt
set -euo pipefail

DATA_DIR="${1:-/home/dave/Downloads/source/nanochat/data/cpp_raw}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "=== Downloading C++ sources to $DATA_DIR ==="

# 1. LLVM/Clang (modern C++, ~800MB of C++ files)
if [ ! -d "llvm-project" ]; then
    echo "--- Cloning LLVM 19.x (shallow) ---"
    git clone --depth=1 --branch=llvmorg-19.1.0 https://github.com/llvm/llvm-project.git
else
    echo "--- LLVM already exists, skipping ---"
fi

# 2. Boost (advanced C++ templates, ~300MB)
if [ ! -d "boost" ]; then
    echo "--- Cloning Boost 1.86.0 (shallow) ---"
    git clone --depth=1 --branch=boost-1.86.0 --recurse-submodules --shallow-submodules https://github.com/boostorg/boost.git
else
    echo "--- Boost already exists, skipping ---"
fi

# 3. Linux kernel (C, systems programming, headers only for M0)
if [ ! -d "linux" ]; then
    echo "--- Cloning Linux kernel v6.10 (shallow) ---"
    git clone --depth=1 --branch=v6.10 https://github.com/torvalds/linux.git
else
    echo "--- Linux already exists, skipping ---"
fi

# 4. fmt (modern C++ formatting library, small but high quality)
if [ ! -d "fmt" ]; then
    echo "--- Cloning fmt 11.0.0 ---"
    git clone --depth=1 --branch=11.0.0 https://github.com/fmtlib/fmt.git
else
    echo "--- fmt already exists, skipping ---"
fi

# 5. googletest (C++ testing patterns)
if [ ! -d "googletest" ]; then
    echo "--- Cloning googletest v1.15.0 ---"
    git clone --depth=1 --branch=v1.15.0 https://github.com/google/googletest.git
else
    echo "--- googletest already exists, skipping ---"
fi

# 6. abseil-cpp (Google C++ common libraries)
if [ ! -d "abseil-cpp" ]; then
    echo "--- Cloning abseil-cpp ---"
    git clone --depth=1 https://github.com/abseil/abseil-cpp.git
else
    echo "--- abseil-cpp already exists, skipping ---"
fi

# 7. folly (Facebook C++ library)
if [ ! -d "folly" ]; then
    echo "--- Cloning folly ---"
    git clone --depth=1 https://github.com/facebook/folly.git
else
    echo "--- folly already exists, skipping ---"
fi

# 8. grpc (Google RPC framework, large C++ codebase)
if [ ! -d "grpc" ]; then
    echo "--- Cloning grpc v1.67.0 ---"
    git clone --depth=1 --branch=v1.67.0 https://github.com/grpc/grpc.git
else
    echo "--- grpc already exists, skipping ---"
fi

echo ""
echo "=== Downloads complete ==="
echo "Counting C/C++ files..."
find "$DATA_DIR" -type f \( -name "*.cpp" -o -name "*.cc" -o -name "*.cxx" -o -name "*.c" -o -name "*.h" -o -name "*.hpp" -o -name "*.hxx" \) | wc -l
echo "Total size:"
du -sh "$DATA_DIR"
