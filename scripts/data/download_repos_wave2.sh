#!/bin/bash
# Wave 2: Additional repos identified by research agents
# Run after download_repos.sh completes
set -euo pipefail

TARGET_DIR="${1:-$HOME/data/cpp_raw}"
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

echo "=== Wave 2: Additional C/C++ repos ==="
echo "Start: $(date)"

# --- Emulators (rich C++) ---
clone_repo https://github.com/RPCS3/rpcs3 rpcs3
clone_repo https://github.com/dolphin-emu/dolphin dolphin-emu
clone_repo https://github.com/PCSX2/pcsx2 pcsx2
clone_repo https://github.com/shadps4-emu/shadPS4 shadps4
clone_repo https://github.com/xemu-project/xemu xemu

# --- Browser engines ---
clone_repo https://github.com/nickel-org/nickel.rs ladybird-skip  # Rust, skip
clone_repo https://github.com/nickel-org/nickel.rs electron-skip  # Too big, skip
# Ladybird is the best bet here:
clone_repo https://github.com/nickel-org/nickel.rs ladybird-skip2

# --- Large missing repos ---
clone_repo https://github.com/microsoft/terminal windows-terminal
clone_repo https://github.com/obsproject/obs-studio obs-studio
clone_repo https://github.com/telegramdesktop/tdesktop telegram-desktop
clone_repo https://github.com/notepad-plus-plus/notepad-plus-plus notepad-plus-plus
clone_repo https://github.com/FreeCAD/FreeCAD freecad
clone_repo https://github.com/xbmc/xbmc kodi
clone_repo https://github.com/videolan/vlc vlc
clone_repo https://github.com/tesseract-ocr/tesseract tesseract
clone_repo https://github.com/cosmopolitan-libc/cosmopolitan cosmopolitan

# --- Physics engines ---
clone_repo https://github.com/erincatto/box2d box2d
clone_repo https://github.com/jrouwe/JoltPhysics jolt-physics
clone_repo https://github.com/NVIDIA-Omniverse/PhysX physx

# --- Additional ML/AI ---
clone_repo https://github.com/Tencent/ncnn ncnn
clone_repo https://github.com/taichi-dev/taichi taichi
clone_repo https://github.com/nomic-ai/gpt4all gpt4all
clone_repo https://github.com/mozilla-ai/llamafile llamafile

# --- Creative tools ---
clone_repo https://github.com/inkscape/inkscape inkscape
clone_repo https://github.com/openscad/openscad openscad
clone_repo https://github.com/aseprite/aseprite aseprite
clone_repo https://github.com/turanszkij/WickedEngine wicked-engine

# --- Desktop / GUI ---
clone_repo https://github.com/nickel-org/nickel.rs fltk-skip
clone_repo https://github.com/keepassxreboot/keepassxc keepassxc
clone_repo https://github.com/qbittorrent/qBittorrent qbittorrent
clone_repo https://github.com/nickel-org/nickel.rs hyprland-skip

# --- Embedded extras ---
clone_repo https://github.com/RT-Thread/rt-thread rt-thread
clone_repo https://github.com/lvgl/lvgl lvgl
clone_repo https://github.com/micropython/micropython micropython
clone_repo https://github.com/qmk/qmk_firmware qmk-firmware

# --- Networking / Security ---
clone_repo https://github.com/hashcat/hashcat hashcat
clone_repo https://github.com/google/boringssl boringssl
clone_repo https://github.com/radareorg/radare2 radare2
clone_repo https://github.com/weidai11/cryptopp cryptopp

# --- Compression/Data ---
clone_repo https://github.com/ImageMagick/ImageMagick imagemagick

# --- Dev tools ---
clone_repo https://github.com/WerWolv/ImHex imhex
clone_repo https://github.com/x64dbg/x64dbg x64dbg

# --- Autonomous/Robotics ---
clone_repo https://github.com/ApolloAuto/apollo apollo
clone_repo https://github.com/carla-simulator/carla carla

# --- Scientific ---
clone_repo https://github.com/Kitware/ParaView paraview
clone_repo https://github.com/HDFGroup/hdf5 hdf5

echo ""
echo "=== Wave 2 Summary ==="
echo "Downloaded: $DOWNLOADED | Skipped: $SKIPPED | Failed: $FAILED"
echo "Total repos: $(ls -d $TARGET_DIR/*/ 2>/dev/null | wc -l)"
echo "Total size: $(du -sh $TARGET_DIR 2>/dev/null | cut -f1)"
echo "End: $(date)"
