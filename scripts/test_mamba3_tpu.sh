#!/bin/bash
# Test Mamba-3 features and AAM pattern on TPU v6e-4
# Uses small total_batch_size to minimize grad accum overhead (1 step = 1 fwd+bwd)

source ~/venv311/bin/activate
cd ~/nanochat
export XLA_NO_SPECIAL_SCALARS=1
export NANOCHAT_BASE_DIR=/home/dave/data

# total_batch_size=4096 = 1 device_batch * 1024 seq * 4 chips => grad_accum=1
BASE_ARGS="--num_iterations=5 --device_batch_size=1 --max_seq_len=1024 --total_batch_size=4096 --kernel=current --no_compile --run=dummy --data_dir=/home/dave/data/parquet --streaming_data"
MAMBA_ARGS="--mamba --mamba_pattern=M --depth=8"

run_test() {
    local name="$1"
    shift
    echo "=============================================="
    echo "TEST: $name"
    echo "=============================================="
    python3 -u -m scripts.base_train $@ 2>&1 | grep -E "step |tok/sec|loss:|ERROR|NaN|Traceback|File|assert|Error|WARNING:" || true
    echo ""
}

run_test "1. Mamba-2 baseline (d=8, all-M)" $BASE_ARGS $MAMBA_ARGS
run_test "2. QK-norm (Phase 2a)" $BASE_ARGS $MAMBA_ARGS --mamba3_qknorm
run_test "3. Bias (Phase 2b)" $BASE_ARGS $MAMBA_ARGS --mamba3_bias
run_test "4. Complex RoPE (Phase 2c)" $BASE_ARGS $MAMBA_ARGS --mamba3_complex_rope
run_test "5. All Phase 2" $BASE_ARGS $MAMBA_ARGS --mamba3_qknorm --mamba3_bias --mamba3_complex_rope
run_test "6. Trapezoidal (Phase 3)" $BASE_ARGS $MAMBA_ARGS --mamba3_trapezoidal
run_test "7. ALL Mamba-3 features" $BASE_ARGS $MAMBA_ARGS --mamba3_qknorm --mamba3_bias --mamba3_complex_rope --mamba3_trapezoidal
run_test "8. AAM d=24 hybrid" $BASE_ARGS --mamba --mamba_pattern=AAM --depth=24

echo "=============================================="
echo "ALL TESTS COMPLETE"
echo "=============================================="
