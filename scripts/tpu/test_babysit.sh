#!/bin/bash
# Test script for babysit_tpu.sh - validates each function against live infrastructure
# Run FROM YOUR LOCAL MACHINE
#
# Usage: bash scripts/tpu/test_babysit.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NANOCHAT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECT="alpine-aspect-459819-m4"
GCS_BUCKET="gs://nanochat-training-data-2026"

PASS=0
FAIL=0

test_result() {
    local name="$1"
    local expected="$2"
    local actual="$3"
    if [ "$expected" = "$actual" ]; then
        echo "  PASS: $name (got: $actual)"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $name (expected: $expected, got: $actual)"
        FAIL=$((FAIL + 1))
    fi
}

test_contains() {
    local name="$1"
    local expected="$2"
    local actual="$3"
    if echo "$actual" | grep -q "$expected"; then
        echo "  PASS: $name (contains: $expected)"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $name (expected to contain: $expected, got: $actual)"
        FAIL=$((FAIL + 1))
    fi
}

echo "============================================"
echo "Babysit TPU - Integration Tests"
echo "============================================"
echo ""

# ─── Test 1: get_tpu_state ────────────────────────────────────────────────
echo "=== Test 1: TPU State Detection ==="

STATE=$(gcloud compute tpus tpu-vm describe nanochat-v6e-small \
    --zone=asia-northeast1-b --project=$PROJECT \
    --format="value(state)" 2>/dev/null || echo "NOT_FOUND")
test_result "v6e-small state" "READY" "$STATE"

STATE=$(gcloud compute tpus tpu-vm describe nanochat-tpu \
    --zone=us-west4-a --project=$PROJECT \
    --format="value(state)" 2>/dev/null || echo "NOT_FOUND")
test_result "v5e state" "READY" "$STATE"

# Test non-existent TPU
STATE=$(gcloud compute tpus tpu-vm describe nonexistent-tpu-12345 \
    --zone=us-west4-a --project=$PROJECT \
    --format="value(state)" 2>/dev/null || echo "NOT_FOUND")
test_result "nonexistent TPU state" "NOT_FOUND" "$STATE"

echo ""

# ─── Test 2: GCS Checkpoint Discovery ─────────────────────────────────────
echo "=== Test 2: GCS Checkpoint Discovery ==="

# v6e-small should have step 5000
LATEST=-1
while IFS= read -r line; do
    if [[ "$line" =~ model_([0-9]+)\.pt ]]; then
        step=${BASH_REMATCH[1]}
        step=$((10#$step))
        if (( step > LATEST )); then LATEST=$step; fi
    fi
done < <(gsutil ls "gs://nanochat-training-data-2026/checkpoints/v6e/base_checkpoints/d16/model_*.pt" 2>/dev/null || true)
test_result "v6e checkpoint (should be 5000)" "5000" "$LATEST"

# v6e-longctx should have nothing (new, 65k tokenizer)
LATEST=-1
while IFS= read -r line; do
    if [[ "$line" =~ model_([0-9]+)\.pt ]]; then
        step=${BASH_REMATCH[1]}
        step=$((10#$step))
        if (( step > LATEST )); then LATEST=$step; fi
    fi
done < <(gsutil ls "gs://nanochat-training-data-2026/checkpoints/v6e-longctx/base_checkpoints/d16/model_*.pt" 2>/dev/null || true)
test_result "v6e-longctx checkpoint (should be -1)" "-1" "$LATEST"

echo ""

# ─── Test 3: Training Process Detection ───────────────────────────────────
echo "=== Test 3: Training Process Detection ==="

RESULT=$(gcloud compute tpus tpu-vm ssh nanochat-v6e-small \
    --zone=asia-northeast1-b --project=$PROJECT \
    --command="pgrep -f 'python3.*scripts[.]base_train' > /dev/null 2>&1 && echo 'BABYSIT_RUNNING' || echo 'BABYSIT_STOPPED'" 2>/dev/null || echo "BABYSIT_SSH_FAILED")
if echo "$RESULT" | grep -q "BABYSIT_RUNNING"; then STATUS="RUNNING"
elif echo "$RESULT" | grep -q "BABYSIT_STOPPED"; then STATUS="STOPPED"
else STATUS="SSH_FAILED"; fi
test_result "v6e-small training detection" "RUNNING" "$STATUS"

RESULT=$(gcloud compute tpus tpu-vm ssh nanochat-v6e-longctx \
    --zone=asia-northeast1-b --project=$PROJECT \
    --command="pgrep -f 'python3.*scripts[.]base_train' > /dev/null 2>&1 && echo 'BABYSIT_RUNNING' || echo 'BABYSIT_STOPPED'" 2>/dev/null || echo "BABYSIT_SSH_FAILED")
if echo "$RESULT" | grep -q "BABYSIT_RUNNING"; then STATUS="RUNNING"
elif echo "$RESULT" | grep -q "BABYSIT_STOPPED"; then STATUS="STOPPED"
else STATUS="SSH_FAILED"; fi
test_result "v6e-longctx training detection (should be STOPPED)" "STOPPED" "$STATUS"

echo ""

# ─── Test 4: Profile Configuration ───────────────────────────────────────
echo "=== Test 4: Profile Configuration Parsing ==="

# Source the babysit script's profile configs by parsing them
for profile in v6e-small v6e-longctx v5e; do
    # Extract TPU_NAME from the script
    TPU_NAME=$(grep -A2 "\"$profile\")" "$SCRIPT_DIR/babysit_tpu.sh" | grep "TPU_NAME=" | head -1 | cut -d'"' -f2)
    if [ -n "$TPU_NAME" ]; then
        echo "  PASS: Profile '$profile' -> TPU_NAME=$TPU_NAME"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: Profile '$profile' -> could not extract TPU_NAME"
        FAIL=$((FAIL + 1))
    fi
done

echo ""

# ─── Test 5: GCS Upload Fix Verification ─────────────────────────────────
echo "=== Test 5: Checkpoint Manager GCS Upload Fix ==="

# Verify the fixed checkpoint_manager.py checks return codes
if grep -q "result.returncode == 0" "$NANOCHAT_DIR/nanochat/checkpoint_manager.py"; then
    echo "  PASS: _upload_to_gcs checks returncode"
    PASS=$((PASS + 1))
else
    echo "  FAIL: _upload_to_gcs does not check returncode"
    FAIL=$((FAIL + 1))
fi

if grep -q "max_retries" "$NANOCHAT_DIR/nanochat/checkpoint_manager.py"; then
    echo "  PASS: _upload_to_gcs has retry logic"
    PASS=$((PASS + 1))
else
    echo "  FAIL: _upload_to_gcs missing retry logic"
    FAIL=$((FAIL + 1))
fi

if grep -q "_download_from_gcs" "$NANOCHAT_DIR/nanochat/checkpoint_manager.py"; then
    echo "  PASS: _download_from_gcs function exists"
    PASS=$((PASS + 1))
else
    echo "  FAIL: _download_from_gcs function missing"
    FAIL=$((FAIL + 1))
fi

if grep -q "resume_from_step == -2" "$NANOCHAT_DIR/scripts/base_train.py"; then
    echo "  PASS: base_train.py supports auto-detect (-2)"
    PASS=$((PASS + 1))
else
    echo "  FAIL: base_train.py missing auto-detect support"
    FAIL=$((FAIL + 1))
fi

echo ""

# ─── Test 6: Babysit Script Syntax ───────────────────────────────────────
echo "=== Test 6: Script Syntax Validation ==="

if bash -n "$SCRIPT_DIR/babysit_tpu.sh" 2>&1; then
    echo "  PASS: babysit_tpu.sh syntax valid"
    PASS=$((PASS + 1))
else
    echo "  FAIL: babysit_tpu.sh has syntax errors"
    FAIL=$((FAIL + 1))
fi

echo ""

# ─── Summary ─────────────────────────────────────────────────────────────
echo "============================================"
echo "Results: $PASS passed, $FAIL failed"
echo "============================================"
if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
