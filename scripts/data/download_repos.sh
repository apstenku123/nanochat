#!/bin/bash
# Download additional C/C++ open source repositories to build3
# Usage: bash scripts/data/download_repos.sh [target_dir] [--dry-run]
#
# Downloads repos as shallow clones (depth=1) to ~/data/cpp_raw/
# Skips repos that already exist.

set -euo pipefail

TARGET_DIR="${1:-$HOME/data/cpp_raw}"
DRY_RUN="${2:-}"

mkdir -p "$TARGET_DIR"

DOWNLOADED=0
SKIPPED=0
FAILED=0

clone_repo() {
    local url="$1"
    local name="$2"

    if [ -d "$TARGET_DIR/$name" ]; then
        ((SKIPPED++)) || true
        return 0
    fi

    if [ "$DRY_RUN" = "--dry-run" ]; then
        echo "[DRY-RUN] $url -> $name"
        return 0
    fi

    echo -n "[CLONE] $name ... "
    if git clone --depth=1 --recurse-submodules --shallow-submodules -q \
        "$url" "$TARGET_DIR/$name" 2>/dev/null; then
        echo "OK"
        ((DOWNLOADED++)) || true
    else
        echo "FAIL"
        ((FAILED++)) || true
        rm -rf "$TARGET_DIR/$name" 2>/dev/null || true
    fi
}

echo "=== Downloading C/C++ repos to $TARGET_DIR ==="
echo "Start: $(date)"

# --- Operating Systems ---
clone_repo https://github.com/haiku/haiku haiku
clone_repo https://github.com/reactos/reactos reactos
clone_repo https://github.com/openbsd/src openbsd
clone_repo https://github.com/NetBSD/src netbsd
clone_repo https://github.com/SerenityOS/serenity serenity

# --- ML/AI (C++) ---
clone_repo https://github.com/ggerganov/llama.cpp llama.cpp
clone_repo https://github.com/ggerganov/whisper.cpp whisper.cpp
clone_repo https://github.com/ggerganov/ggml ggml
clone_repo https://github.com/facebookresearch/faiss faiss
clone_repo https://github.com/mlpack/mlpack mlpack
clone_repo https://github.com/BVLC/caffe caffe
clone_repo https://github.com/apache/mxnet mxnet
clone_repo https://github.com/flashlight/flashlight flashlight
clone_repo https://github.com/microsoft/CNTK cntk
clone_repo https://github.com/PaddlePaddle/Paddle paddle
clone_repo https://github.com/google/mediapipe mediapipe

# --- Databases ---
clone_repo https://github.com/sqlite/sqlite sqlite
clone_repo https://github.com/dragonflydb/dragonfly dragonfly
clone_repo https://github.com/valkey-io/valkey valkey
clone_repo https://github.com/wiredtiger/wiredtiger wiredtiger
clone_repo https://github.com/memcached/memcached memcached

# --- Scientific/HPC ---
clone_repo https://github.com/trilinos/Trilinos trilinos
clone_repo https://github.com/mfem/mfem mfem
clone_repo https://github.com/lammps/lammps lammps
clone_repo https://github.com/OpenFOAM/OpenFOAM-dev openfoam
clone_repo https://github.com/STEllAR-GROUP/hpx hpx
clone_repo https://github.com/kokkos/kokkos kokkos
clone_repo https://github.com/AMReX-Codes/amrex amrex
clone_repo https://github.com/ceres-solver/ceres-solver ceres-solver
clone_repo https://github.com/PointCloudLibrary/pcl pcl
clone_repo https://github.com/CGAL/cgal cgal

# --- Graphics/Rendering ---
clone_repo https://github.com/assimp/assimp assimp
clone_repo https://github.com/bulletphysics/bullet3 bullet3
clone_repo https://github.com/OGRECave/ogre ogre
clone_repo https://github.com/mosra/magnum magnum
clone_repo https://github.com/bkaradzic/bgfx bgfx
clone_repo https://github.com/AcademySoftwareFoundation/openvdb openvdb
clone_repo https://github.com/google/skia skia
clone_repo https://github.com/cocos2d/cocos2d-x cocos2d-x
clone_repo https://github.com/KhronosGroup/Vulkan-Samples vulkan-samples
clone_repo https://github.com/KhronosGroup/SPIRV-Cross spirv-cross
clone_repo https://github.com/KhronosGroup/glslang glslang

# --- Networking ---
clone_repo https://github.com/libevent/libevent libevent
clone_repo https://github.com/haproxy/haproxy haproxy
clone_repo https://github.com/squid-cache/squid squid
clone_repo https://github.com/OpenVPN/openvpn openvpn
clone_repo https://github.com/nmap/nmap nmap
clone_repo https://github.com/zeromq/libzmq libzmq
clone_repo https://github.com/apache/thrift thrift
clone_repo https://github.com/uNetworking/uWebSockets uwebsockets
clone_repo https://github.com/facebook/proxygen proxygen
clone_repo https://github.com/simdjson/simdjson simdjson
clone_repo https://github.com/Tencent/rapidjson rapidjson
clone_repo https://github.com/msgpack/msgpack-c msgpack-c

# --- Core C Libraries ---
clone_repo https://sourceware.org/git/glibc.git glibc
clone_repo https://sourceware.org/git/binutils-gdb.git binutils-gdb
clone_repo https://github.com/git/git git-src
clone_repo https://github.com/tmux/tmux tmux
clone_repo https://github.com/vim/vim vim
clone_repo https://github.com/neovim/neovim neovim
clone_repo https://github.com/lua/lua lua
clone_repo https://github.com/jqlang/jq jq
clone_repo https://github.com/madler/zlib zlib
clone_repo https://github.com/lz4/lz4 lz4
clone_repo https://github.com/facebook/zstd zstd
clone_repo https://github.com/tukaani-project/xz xz
clone_repo https://github.com/glennrp/libpng libpng
clone_repo https://github.com/libjpeg-turbo/libjpeg-turbo libjpeg-turbo
clone_repo https://github.com/harfbuzz/harfbuzz harfbuzz
clone_repo https://github.com/freetype/freetype freetype
clone_repo https://github.com/GNOME/glib glib
clone_repo https://github.com/GNOME/gtk gtk
clone_repo https://github.com/systemd/systemd systemd
clone_repo https://github.com/util-linux/util-linux util-linux
clone_repo https://github.com/strace/strace strace
clone_repo https://github.com/bminor/musl musl
clone_repo https://github.com/bminor/bash bash-src

# --- Embedded/RTOS ---
clone_repo https://github.com/apache/nuttx nuttx
clone_repo https://github.com/RIOT-OS/RIOT riot-os
clone_repo https://github.com/ChibiOS/ChibiOS chibios
clone_repo https://github.com/ARMmbed/mbed-os mbed-os
clone_repo https://github.com/contiki-os/contiki contiki
clone_repo https://github.com/espressif/esp-idf esp-idf

# --- Crypto/Security ---
clone_repo https://github.com/Mbed-TLS/mbedtls mbedtls
clone_repo https://github.com/jedisct1/libsodium libsodium
clone_repo https://github.com/gpg/gnupg gnupg
clone_repo https://github.com/wireshark/wireshark wireshark
clone_repo https://github.com/veracrypt/VeraCrypt veracrypt

# --- C++ Libraries ---
clone_repo https://github.com/g-truc/glm glm
clone_repo https://github.com/nothings/stb stb
clone_repo https://github.com/USCiLab/cereal cereal
clone_repo https://github.com/microsoft/GSL gsl
clone_repo https://github.com/microsoft/terminal windows-terminal
clone_repo https://github.com/pybind/pybind11 pybind11
clone_repo https://github.com/doctest/doctest doctest
clone_repo https://github.com/include-what-you-use/include-what-you-use iwyu

# --- KDE Frameworks ---
for fw in kcoreaddons kconfig ki18n kio kwidgetsaddons kxmlgui knotifications; do
    clone_repo "https://invent.kde.org/frameworks/$fw.git" "kde-$fw"
done
clone_repo https://invent.kde.org/graphics/krita.git krita
clone_repo https://invent.kde.org/utilities/kate.git kate

# --- Audio/Video ---
clone_repo https://github.com/gstreamer/gstreamer gstreamer
clone_repo https://github.com/mpv-player/mpv mpv
clone_repo https://github.com/xiph/flac flac
clone_repo https://github.com/HandBrake/HandBrake handbrake

# --- ASWF (Academy Software Foundation) ---
clone_repo https://github.com/AcademySoftwareFoundation/MaterialX materialx
clone_repo https://github.com/AcademySoftwareFoundation/opensubdiv opensubdiv
clone_repo https://github.com/AcademySoftwareFoundation/OpenShadingLanguage osl

echo ""
echo "=== Summary ==="
echo "Downloaded: $DOWNLOADED | Skipped: $SKIPPED | Failed: $FAILED"
echo "Total repos: $(ls -d $TARGET_DIR/*/ 2>/dev/null | wc -l)"
echo "Total size: $(du -sh $TARGET_DIR 2>/dev/null | cut -f1)"
echo "End: $(date)"
