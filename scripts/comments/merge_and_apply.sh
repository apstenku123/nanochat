#!/bin/bash
set -e

DATA="/home/dave/data"
TRANSLATED="$DATA/comments_translated.jsonl"
RETRY="$DATA/comments_translated_retry24.jsonl"
MERGED="$DATA/comments_translated_merged.jsonl"
BACKUP="$DATA/cpp_raw_backup_comments"
APPLY_SCRIPT="/home/dave/source/nanochat/scripts/comments/apply_translations.py"
VENV="/home/dave/venv_translate/bin/python"

echo "=== MERGE AND APPLY TRANSLATIONS ==="
echo "Date: $(date)"
echo ""

# Step 1: Merge main + retry
echo "--- Step 1: Merging translations ---"
MAIN_COUNT=$(wc -l < "$TRANSLATED")
echo "Main translations: $MAIN_COUNT"

if [ -f "$RETRY" ] && [ -s "$RETRY" ]; then
    RETRY_COUNT=$(wc -l < "$RETRY")
    echo "Retry translations: $RETRY_COUNT"
    cat "$TRANSLATED" "$RETRY" > "$MERGED"
else
    echo "No retry file, using main only"
    cp "$TRANSLATED" "$MERGED"
fi

TOTAL=$(wc -l < "$MERGED")
echo "Total merged: $TOTAL"

# Step 2: Check for duplicates and quality
echo ""
echo "--- Step 2: Quality check ---"
python3 << 'PYEOF'
import json

records = []
seen = set()
dupes = 0
non_ascii = 0
low_conf = 0
min_conf = 1.0
total = 0

with open("/home/dave/data/comments_translated_merged.jsonl") as f:
    for line in f:
        r = json.loads(line.strip())
        total += 1
        if r["id"] in seen:
            dupes += 1
        seen.add(r["id"])
        if not r["translated"].isascii():
            non_ascii += 1
        if r["confidence"] < 0.5:
            low_conf += 1
        min_conf = min(min_conf, r["confidence"])

print(f"Total: {total}")
print(f"Unique: {len(seen)}, Duplicates: {dupes}")
print(f"Non-ASCII: {non_ascii}")
print(f"Low confidence (<0.5): {low_conf}")
print(f"Min confidence: {min_conf:.3f}")
PYEOF

# Step 3: Validate byte offsets
echo ""
echo "--- Step 3: Validate byte offsets ---"
$VENV "$APPLY_SCRIPT" --input "$MERGED" --validate --dry-run

# Step 4: Apply translations
echo ""
echo "--- Step 4: Apply translations ---"
$VENV "$APPLY_SCRIPT" --input "$MERGED" --backup-dir "$BACKUP"

echo ""
echo "=== MERGE AND APPLY COMPLETE ==="
echo "Date: $(date)"
