#!/usr/bin/env bash
# Download key C++ repos with full history (partial clone for efficiency).
# Uses --filter=blob:none so git fetches tree history but blobs on demand.
set -euo pipefail

DEST_DIR="${1:-data/cpp_full_history}"
mkdir -p "$DEST_DIR"

# Ordered smallest-first so we get usable data quickly.
declare -A REPOS=(
  [bitcoin]="https://github.com/bitcoin/bitcoin.git"
  [protobuf]="https://github.com/protocolbuffers/protobuf.git"
  [grpc]="https://github.com/grpc/grpc.git"
  [opencv]="https://github.com/opencv/opencv.git"
  [qtbase]="https://github.com/qt/qtbase.git"
  [boost]="https://github.com/boostorg/boost.git"
  [llvm-project]="https://github.com/llvm/llvm-project.git"
  [linux]="https://github.com/torvalds/linux.git"
)

# Clone order: small repos first
ORDER=(bitcoin protobuf grpc opencv qtbase boost llvm-project linux)

for name in "${ORDER[@]}"; do
  url="${REPOS[$name]}"
  dest="$DEST_DIR/$name"
  if [ -d "$dest/.git" ]; then
    echo "=== $name already cloned, skipping ==="
    continue
  fi
  echo "=== Cloning $name (partial clone) ==="
  rm -rf "$dest"
  git clone --filter=blob:none "$url" "$dest" &
  # Run up to 2 clones in parallel
  if (( $(jobs -r | wc -l) >= 2 )); then
    wait -n
  fi
done

wait
echo "=== All clones complete ==="
