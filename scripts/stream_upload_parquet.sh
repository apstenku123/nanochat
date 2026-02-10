#!/bin/bash
# Stream-upload parquet shards to GCS as they are created.
# Usage: ./scripts/stream_upload_parquet.sh <local_parquet_dir> <gcs_dest>
# Example: ./scripts/stream_upload_parquet.sh data/cpp_chunked_parquet gs://nanochat-training-data-2026/parquet/cpp_chunked_16k
set -euo pipefail

LOCAL_DIR="${1:?Usage: $0 <local_parquet_dir> <gcs_dest>}"
GCS_DEST="${2:?Usage: $0 <local_parquet_dir> <gcs_dest>}"

echo "[upload] Watching $LOCAL_DIR -> $GCS_DEST"

UPLOADED=""

while true; do
    # Find all parquet files (train shards + val_shard)
    for f in "$LOCAL_DIR"/shard_*.parquet "$LOCAL_DIR"/val_shard.parquet; do
        [ -f "$f" ] || continue
        base=$(basename "$f")
        # Skip if already uploaded
        if echo "$UPLOADED" | grep -qF "$base"; then
            continue
        fi
        echo "[upload] Uploading $base..."
        gcloud storage cp "$f" "$GCS_DEST/$base" 2>&1
        UPLOADED="$UPLOADED $base"
        echo "[upload] Done: $base ($(echo $UPLOADED | wc -w) files uploaded)"
    done

    # Check if _COMPLETE sentinel exists
    if [ -f "$LOCAL_DIR/_COMPLETE" ]; then
        # Upload sentinel
        gcloud storage cp "$LOCAL_DIR/_COMPLETE" "$GCS_DEST/_COMPLETE" 2>&1
        echo "[upload] All shards uploaded. _COMPLETE sentinel sent."
        break
    fi

    sleep 15
done

echo "[upload] Pipeline complete. $(echo $UPLOADED | wc -w) total files uploaded to $GCS_DEST"
