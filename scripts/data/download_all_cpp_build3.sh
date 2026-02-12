#!/bin/bash
# Combined download of all C++ repos for clang-based indexing on build3
# Runs clones in parallel batches to maximize bandwidth
set -euo pipefail

DATA_DIR="${1:-/home/dave/data/cpp_raw}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

MAX_PARALLEL=8  # parallel git clones

clone_bg() {
    local dir="$1" url="$2"
    shift 2
    if [ -d "$dir" ]; then
        echo "SKIP: $dir exists"
        return 0
    fi
    echo "START: $dir"
    git clone "$@" "$url" "$dir" 2>&1 | tail -1 && echo "DONE: $dir" || echo "FAIL: $dir"
}

wait_for_slots() {
    while [ "$(jobs -r | wc -l)" -ge "$MAX_PARALLEL" ]; do
        sleep 2
    done
}

echo "=== Downloading all C++ repos to $DATA_DIR ==="
echo "Max parallel: $MAX_PARALLEL"
echo "Start time: $(date)"

# === V1: Core repos ===
wait_for_slots; clone_bg llvm-project   https://github.com/llvm/llvm-project.git     --depth=1 &
wait_for_slots; clone_bg linux          https://github.com/torvalds/linux.git         --depth=1 &
wait_for_slots; clone_bg boost          https://github.com/boostorg/boost.git         --depth=1 --recurse-submodules --shallow-submodules &
wait_for_slots; clone_bg fmt            https://github.com/fmtlib/fmt.git             --depth=1 &
wait_for_slots; clone_bg googletest     https://github.com/google/googletest.git      --depth=1 &
wait_for_slots; clone_bg abseil-cpp     https://github.com/abseil/abseil-cpp.git      --depth=1 &
wait_for_slots; clone_bg folly          https://github.com/facebook/folly.git         --depth=1 &
wait_for_slots; clone_bg grpc           https://github.com/grpc/grpc.git              --depth=1 &

# === V2: Frameworks and ML ===
wait_for_slots; clone_bg qtbase         https://github.com/qt/qtbase.git              --depth=1 &
wait_for_slots; clone_bg cuda-samples   https://github.com/NVIDIA/cuda-samples.git    --depth=1 &
wait_for_slots; clone_bg opencv         https://github.com/opencv/opencv.git          --depth=1 &
wait_for_slots; clone_bg eigen          https://gitlab.com/libeigen/eigen.git         --depth=1 &
wait_for_slots; clone_bg json           https://github.com/nlohmann/json.git          --depth=1 &
wait_for_slots; clone_bg STL            https://github.com/microsoft/STL.git          --depth=1 &
wait_for_slots; clone_bg protobuf       https://github.com/protocolbuffers/protobuf.git --depth=1 &
wait_for_slots; clone_bg CMake          https://github.com/Kitware/CMake.git          --depth=1 &
wait_for_slots; clone_bg Catch2         https://github.com/catchorg/Catch2.git        --depth=1 &
wait_for_slots; clone_bg poco           https://github.com/pocoproject/poco.git       --depth=1 &
wait_for_slots; clone_bg bitcoin        https://github.com/bitcoin/bitcoin.git        --depth=1 &
wait_for_slots; clone_bg tensorflow     https://github.com/tensorflow/tensorflow.git  --depth=1 &
wait_for_slots; clone_bg pytorch        https://github.com/pytorch/pytorch.git        --depth=1 &

# === V3: Compilers, databases, graphics ===
wait_for_slots; clone_bg gcc-mirror     https://github.com/gcc-mirror/gcc.git         --depth=1 &
wait_for_slots; clone_bg godot          https://github.com/godotengine/godot.git      --depth=1 &
wait_for_slots; clone_bg filament       https://github.com/google/filament.git        --depth=1 &
wait_for_slots; clone_bg imgui          https://github.com/ocornut/imgui.git          --depth=1 &
wait_for_slots; clone_bg SDL            https://github.com/libsdl-org/SDL.git         --depth=1 &
wait_for_slots; clone_bg vulkan-hpp     https://github.com/KhronosGroup/Vulkan-Hpp.git --depth=1 &
wait_for_slots; clone_bg rocksdb        https://github.com/facebook/rocksdb.git       --depth=1 &
wait_for_slots; clone_bg leveldb        https://github.com/google/leveldb.git         --depth=1 &
wait_for_slots; clone_bg duckdb         https://github.com/duckdb/duckdb.git          --depth=1 &
wait_for_slots; clone_bg clickhouse     https://github.com/ClickHouse/ClickHouse.git  --depth=1 &
wait_for_slots; clone_bg mariadb        https://github.com/MariaDB/server.git         --depth=1 &
wait_for_slots; clone_bg mysql          https://github.com/mysql/mysql-server.git     --depth=1 &
wait_for_slots; clone_bg postgres       https://github.com/postgres/postgres.git      --depth=1 &
wait_for_slots; clone_bg redis          https://github.com/redis/redis.git            --depth=1 &
wait_for_slots; clone_bg mongodb-cxx    https://github.com/mongodb/mongo-cxx-driver.git --depth=1 &
wait_for_slots; clone_bg envoy          https://github.com/envoyproxy/envoy.git       --depth=1 &
wait_for_slots; clone_bg seastar        https://github.com/scylladb/seastar.git       --depth=1 &
wait_for_slots; clone_bg cpprestsdk     https://github.com/microsoft/cpprestsdk.git   --depth=1 &
wait_for_slots; clone_bg asio           https://github.com/chriskohlhoff/asio.git     --depth=1 &
wait_for_slots; clone_bg flatbuffers    https://github.com/google/flatbuffers.git     --depth=1 &
wait_for_slots; clone_bg capnproto      https://github.com/capnproto/capnproto.git    --depth=1 &
wait_for_slots; clone_bg arrow          https://github.com/apache/arrow.git           --depth=1 &
wait_for_slots; clone_bg mold           https://github.com/rui314/mold.git            --depth=1 &
wait_for_slots; clone_bg tbb            https://github.com/oneapi-src/oneTBB.git      --depth=1 &
wait_for_slots; clone_bg swift          https://github.com/swiftlang/swift.git        --depth=1 &
wait_for_slots; clone_bg z3             https://github.com/Z3Prover/z3.git            --depth=1 &
wait_for_slots; clone_bg VTK            https://github.com/Kitware/VTK.git            --depth=1 &
wait_for_slots; clone_bg ITK            https://github.com/InsightSoftwareConsortium/ITK.git --depth=1 &
wait_for_slots; clone_bg dealii         https://github.com/dealii/dealii.git          --depth=1 &
wait_for_slots; clone_bg openexr        https://github.com/AcademySoftwareFoundation/openexr.git --depth=1 &
wait_for_slots; clone_bg mesa           https://gitlab.freedesktop.org/mesa/mesa.git  --depth=1 &
wait_for_slots; clone_bg blender        https://projects.blender.org/blender/blender.git --depth=1 &
wait_for_slots; clone_bg re2            https://github.com/google/re2.git             --depth=1 &
wait_for_slots; clone_bg ceph           https://github.com/ceph/ceph.git              --depth=1 &

# === V4: Runtimes, networking, OS ===
wait_for_slots; clone_bg mono           https://github.com/mono/mono.git              --depth=1 &
wait_for_slots; clone_bg dotnet-runtime https://github.com/dotnet/runtime.git         --depth=1 &
wait_for_slots; clone_bg openjdk        https://github.com/openjdk/jdk.git            --depth=1 &
wait_for_slots; clone_bg ruby           https://github.com/ruby/ruby.git              --depth=1 &
wait_for_slots; clone_bg cpython        https://github.com/python/cpython.git         --depth=1 &
wait_for_slots; clone_bg php-src        https://github.com/php/php-src.git            --depth=1 &
wait_for_slots; clone_bg node           https://github.com/nodejs/node.git            --depth=1 &
wait_for_slots; clone_bg scylladb       https://github.com/scylladb/scylladb.git      --depth=1 &
wait_for_slots; clone_bg foundationdb   https://github.com/apple/foundationdb.git     --depth=1 &
wait_for_slots; clone_bg nginx          https://github.com/nginx/nginx.git            --depth=1 &
wait_for_slots; clone_bg curl           https://github.com/curl/curl.git              --depth=1 &
wait_for_slots; clone_bg openssl        https://github.com/openssl/openssl.git        --depth=1 &
wait_for_slots; clone_bg libuv          https://github.com/libuv/libuv.git            --depth=1 &
wait_for_slots; clone_bg wolfssl        https://github.com/wolfSSL/wolfssl.git        --depth=1 &
wait_for_slots; clone_bg ffmpeg         https://github.com/FFmpeg/FFmpeg.git          --depth=1 &
wait_for_slots; clone_bg opus           https://github.com/xiph/opus.git              --depth=1 &
wait_for_slots; clone_bg openblas       https://github.com/OpenMathLib/OpenBLAS.git   --depth=1 &
wait_for_slots; clone_bg glfw           https://github.com/glfw/glfw.git              --depth=1 &
wait_for_slots; clone_bg SFML           https://github.com/SFML/SFML.git             --depth=1 &
wait_for_slots; clone_bg wxWidgets      https://github.com/wxWidgets/wxWidgets.git    --depth=1 &
wait_for_slots; clone_bg freebsd-src    https://github.com/freebsd/freebsd-src.git    --depth=1 &
wait_for_slots; clone_bg zephyr         https://github.com/zephyrproject-rtos/zephyr.git --depth=1 &
wait_for_slots; clone_bg FreeRTOS       https://github.com/FreeRTOS/FreeRTOS.git      --depth=1 &
wait_for_slots; clone_bg spdlog         https://github.com/gabime/spdlog.git          --depth=1 &
wait_for_slots; clone_bg benchmark      https://github.com/google/benchmark.git       --depth=1 &
wait_for_slots; clone_bg entt           https://github.com/skypjack/entt.git          --depth=1 &
wait_for_slots; clone_bg range-v3       https://github.com/ericniebler/range-v3.git   --depth=1 &
wait_for_slots; clone_bg taskflow       https://github.com/taskflow/taskflow.git      --depth=1 &
wait_for_slots; clone_bg onnxruntime    https://github.com/microsoft/onnxruntime.git  --depth=1 &
wait_for_slots; clone_bg tvm            https://github.com/apache/tvm.git             --depth=1 &
wait_for_slots; clone_bg xgboost       https://github.com/dmlc/xgboost.git            --depth=1 &

# === Extra: Kernel/driver focused repos (user specifically wants bottom-up HAL→driver→subsystem) ===
wait_for_slots; clone_bg dpdk           https://github.com/DPDK/dpdk.git              --depth=1 &
wait_for_slots; clone_bg qemu           https://github.com/qemu/qemu.git              --depth=1 &
wait_for_slots; clone_bg xen            https://github.com/xen-project/xen.git        --depth=1 &
wait_for_slots; clone_bg u-boot         https://github.com/u-boot/u-boot.git          --depth=1 &
wait_for_slots; clone_bg opensbi        https://github.com/riscv-software-src/opensbi.git --depth=1 &
wait_for_slots; clone_bg windows-driver-samples https://github.com/microsoft/Windows-driver-samples.git --depth=1 &
wait_for_slots; clone_bg ntfs-3g        https://github.com/tuxera/ntfs-3g.git         --depth=1 &

echo ""
echo "Waiting for all clones to finish..."
wait

echo ""
echo "=== All downloads complete ==="
echo "End time: $(date)"
echo ""
echo "Counting C/C++ source files..."
find "$DATA_DIR" -type f \( -name "*.cpp" -o -name "*.cc" -o -name "*.cxx" -o -name "*.c" -o -name "*.h" -o -name "*.hpp" \) | wc -l
echo "Total disk usage:"
du -sh "$DATA_DIR"
