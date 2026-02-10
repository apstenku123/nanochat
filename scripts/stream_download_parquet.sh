#!/bin/bash
# Stream-download parquet shards from GCS as they arrive.
# Usage: ./scripts/stream_download_parquet.sh <gcs_source> <local_dest>
# Example: ./scripts/stream_download_parquet.sh gs://nanochat-training-data-2026/parquet/cpp_chunked_16k /home/dave/data/parquet
set -euo pipefail

GCS_SRC="${1:?Usage: $0 <gcs_source> <local_dest>}"
LOCAL_DIR="${2:?Usage: $0 <gcs_source> <local_dest>}"

mkdir -p "$LOCAL_DIR"
echo "[download] Watching $GCS_SRC -> $LOCAL_DIR"

DOWNLOADED=""

while true; do
    # List remote parquet files
    REMOTE_FILES=$(gcloud storage ls "$GCS_SRC/shard_*.parquet" "$GCS_SRC/val_shard.parquet" 2>/dev/null || true)

    for remote in $REMOTE_FILES; do
        base=$(basename "$remote")
        local_path="$LOCAL_DIR/$base"
        # Skip if already downloaded
        if [ -f "$local_path" ]; then
            continue
        fi
        echo "[download] Downloading $base..."
        gcloud storage cp "$remote" "$local_path" 2>&1
        DOWNLOADED="$DOWNLOADED $base"
        echo "[download] Done: $base ($(echo $DOWNLOADED | wc -w) files downloaded)"
    done

    # Check for _COMPLETE sentinel
    if gcloud storage ls "$GCS_SRC/_COMPLETE" >/dev/null 2>&1; then
        gcloud storage cp "$GCS_SRC/_COMPLETE" "$LOCAL_DIR/_COMPLETE" 2>&1
        echo "[download] All shards downloaded. _COMPLETE sentinel received."
        break
    fi

    sleep 15
done

echo "[download] Pipeline complete. $(echo $DOWNLOADED | wc -w) total files downloaded to $LOCAL_DIR"
