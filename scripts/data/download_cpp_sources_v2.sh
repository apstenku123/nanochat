#!/bin/bash
# Download additional C++ sources to scale training data toward 3B+ tokens
# V2: Adds new repos on top of existing ones from v1
set -euo pipefail

DATA_DIR="${1:-/home/dave/Downloads/source/nanochat/data/cpp_raw}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "=== Downloading additional C++ sources to $DATA_DIR ==="

clone_if_missing() {
    local dir="$1" url="$2"
    shift 2
    if [ ! -d "$dir" ]; then
        echo "--- Cloning $dir ---"
        git clone "$@" "$url" "$dir"
    else
        echo "--- $dir already exists, skipping ---"
    fi
}

# New sources (v2)
clone_if_missing qtbase      https://github.com/qt/qtbase.git          --depth=1 --branch=v6.7.0
clone_if_missing cuda-samples https://github.com/NVIDIA/cuda-samples.git --depth=1
clone_if_missing opencv       https://github.com/opencv/opencv.git      --depth=1 --branch=4.10.0
clone_if_missing eigen        https://gitlab.com/libeigen/eigen.git     --depth=1 --branch=3.4.0
clone_if_missing json         https://github.com/nlohmann/json.git      --depth=1
clone_if_missing STL          https://github.com/microsoft/STL.git      --depth=1
clone_if_missing protobuf     https://github.com/protocolbuffers/protobuf.git --depth=1 --branch=v27.0
clone_if_missing CMake        https://github.com/Kitware/CMake.git      --depth=1 --branch=v3.30.0
clone_if_missing Catch2       https://github.com/catchorg/Catch2.git    --depth=1
clone_if_missing poco         https://github.com/pocoproject/poco.git   --depth=1
clone_if_missing bitcoin      https://github.com/bitcoin/bitcoin.git    --depth=1
clone_if_missing tensorflow   https://github.com/tensorflow/tensorflow.git --depth=1
clone_if_missing pytorch      https://github.com/pytorch/pytorch.git    --depth=1

echo ""
echo "=== Downloads complete ==="
echo "Counting C/C++ files in new repos..."
for d in qtbase cuda-samples opencv eigen json STL protobuf CMake Catch2 poco bitcoin tensorflow pytorch; do
    if [ -d "$d" ]; then
        count=$(find "$d" -type f \( -name "*.cpp" -o -name "*.cc" -o -name "*.cxx" -o -name "*.c" -o -name "*.h" -o -name "*.hpp" -o -name "*.hxx" -o -name "*.cu" -o -name "*.cuh" \) | wc -l)
        echo "  $d: $count files"
    fi
done
echo "Total size:"
du -sh "$DATA_DIR"
