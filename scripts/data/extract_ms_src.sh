#!/bin/bash
# Extract MS SDK/DDK archives and collect C/C++ files into JSONL
# Usage: bash scripts/data/extract_ms_src.sh

set -euo pipefail

SRC_DIR="$HOME/data/ms_src"
EXTRACT_DIR="$HOME/data/ms_src_extracted"
OUTPUT="$HOME/data/ms_src_cpp.jsonl"

mkdir -p "$EXTRACT_DIR"

echo "=== Extracting MS SDK/DDK archives ==="
echo "Source: $SRC_DIR"
echo "Extract to: $EXTRACT_DIR"

# Extract all 7z archives
for archive in "$SRC_DIR"/*.7z; do
    name=$(basename "$archive" .7z)
    dest="$EXTRACT_DIR/$name"
    if [ -d "$dest" ]; then
        echo "[SKIP] $name (already extracted)"
        continue
    fi
    echo "[EXTRACT] $name ..."
    mkdir -p "$dest"
    7z x -o"$dest" "$archive" -y > /dev/null 2>&1 || echo "[WARN] Partial extract: $name"
done

echo ""
echo "=== Collecting C/C++ files into JSONL ==="

# Find all C/C++ files and write as JSONL
> "$OUTPUT"
count=0
find "$EXTRACT_DIR" \( \
    -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' \
    -o -name '*.h' -o -name '*.hh' -o -name '*.hpp' -o -name '*.hxx' \
    -o -name '*.inl' -o -name '*.inc' \
\) -type f -size +50c -size -500k | while read -r filepath; do
    # Read file, escape for JSON, write as {"text": "..."}
    text=$(cat "$filepath" 2>/dev/null | python3 -c "
import sys, json
text = sys.stdin.read()
if len(text) > 50:
    print(json.dumps({'text': text}))
" 2>/dev/null)
    if [ -n "$text" ]; then
        echo "$text" >> "$OUTPUT"
        ((count++)) || true
        if [ $((count % 10000)) -eq 0 ]; then
            echo "  Collected $count files..."
        fi
    fi
done

total=$(wc -l < "$OUTPUT")
size=$(du -sh "$OUTPUT" | cut -f1)
echo ""
echo "=== Summary ==="
echo "Total C/C++ files: $total"
echo "Output: $OUTPUT ($size)"
echo "Disk usage:"
du -sh "$EXTRACT_DIR"
