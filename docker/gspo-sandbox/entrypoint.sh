#!/bin/bash
# GSPO Sandbox Entrypoint
# Compiles and runs C++ code with strict resource limits
#
# Accepts code via:
#   1. File mount at /sandbox/code.cpp
#   2. Stdin (echo "code" | docker run ...)
#   3. Environment variable CODE_CONTENT
#
# Output format (JSON):
#   { "success": bool, "stdout": str, "stderr": str, "exit_code": int,
#     "compile_time_ms": int, "run_time_ms": int, "timeout": bool }

set -o pipefail

# Configuration from environment
COMPILER="${COMPILER:-g++}"
CPP_STANDARD="${CPP_STANDARD:-20}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-10}"
OPTIMIZATION_LEVEL="${OPTIMIZATION_LEVEL:--O2}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"
CODE_FILE="/sandbox/code.cpp"
BINARY_FILE="/sandbox/a.out"
TEST_FILE="${TEST_FILE:-}"
TEST_CONTENT="${TEST_CONTENT:-}"

# JSON output helper
json_output() {
    local success="$1"
    local stdout="$2"
    local stderr="$3"
    local exit_code="$4"
    local compile_time="$5"
    local run_time="$6"
    local timeout="$7"

    # Escape JSON strings
    stdout=$(echo "$stdout" | jq -Rs .)
    stderr=$(echo "$stderr" | jq -Rs .)

    cat <<EOF
{
    "success": $success,
    "stdout": $stdout,
    "stderr": $stderr,
    "exit_code": $exit_code,
    "compile_time_ms": $compile_time,
    "run_time_ms": $run_time,
    "timeout": $timeout
}
EOF
}

# Get code content
get_code() {
    if [[ -n "${CODE_CONTENT:-}" ]]; then
        echo "$CODE_CONTENT" > "$CODE_FILE"
    elif [[ ! -f "$CODE_FILE" ]]; then
        # Try to read from stdin
        if [[ ! -t 0 ]]; then
            cat > "$CODE_FILE"
        fi
    fi

    if [[ ! -f "$CODE_FILE" ]] || [[ ! -s "$CODE_FILE" ]]; then
        json_output false '""' '"No code provided. Mount code at /sandbox/code.cpp or pass via stdin/CODE_CONTENT"' 1 0 0 false
        exit 1
    fi
}

# Get test content if provided
get_test() {
    if [[ -n "${TEST_CONTENT:-}" ]]; then
        TEST_FILE="/sandbox/test.cpp"
        echo "$TEST_CONTENT" > "$TEST_FILE"
    fi
}

# Compile code
compile_code() {
    local start_time end_time
    start_time=$(date +%s%3N)

    local compile_flags="-std=c++${CPP_STANDARD} ${OPTIMIZATION_LEVEL} -Wall -Wextra ${EXTRA_FLAGS}"

    # If test file exists, compile both
    if [[ -n "$TEST_FILE" ]] && [[ -f "$TEST_FILE" ]]; then
        compile_flags="$compile_flags -lgtest -lgtest_main -pthread"
        compile_stderr=$($COMPILER $compile_flags "$CODE_FILE" "$TEST_FILE" -o "$BINARY_FILE" 2>&1)
    else
        compile_stderr=$($COMPILER $compile_flags "$CODE_FILE" -o "$BINARY_FILE" 2>&1)
    fi
    compile_exit=$?

    end_time=$(date +%s%3N)
    compile_time=$((end_time - start_time))

    if [[ $compile_exit -ne 0 ]]; then
        json_output false '""' "\"Compilation failed: $compile_stderr\"" $compile_exit $compile_time 0 false
        exit 0
    fi

    echo $compile_time
}

# Run code with timeout
run_code() {
    local compile_time="$1"
    local start_time end_time run_time
    local run_stdout run_stderr run_exit timeout_occurred

    start_time=$(date +%s%3N)
    timeout_occurred=false

    # Create a temporary file for output
    local stdout_file=$(mktemp)
    local stderr_file=$(mktemp)

    # Run with timeout and resource limits
    timeout --signal=KILL "${TIMEOUT_SECONDS}s" \
        /usr/bin/env -i PATH=/usr/bin:/bin HOME=/sandbox \
        "$BINARY_FILE" >"$stdout_file" 2>"$stderr_file"
    run_exit=$?

    end_time=$(date +%s%3N)
    run_time=$((end_time - start_time))

    # Check for timeout (exit code 137 = SIGKILL from timeout)
    if [[ $run_exit -eq 137 ]] || [[ $run_exit -eq 124 ]]; then
        timeout_occurred=true
        run_stderr="Execution timed out after ${TIMEOUT_SECONDS} seconds"
    fi

    run_stdout=$(cat "$stdout_file" | head -c 1048576)  # Limit to 1MB
    run_stderr_content=$(cat "$stderr_file" | head -c 1048576)

    if [[ -n "$run_stderr_content" ]]; then
        run_stderr="$run_stderr_content"
    fi

    rm -f "$stdout_file" "$stderr_file"

    # Determine success
    local success=false
    if [[ $run_exit -eq 0 ]] && [[ "$timeout_occurred" == "false" ]]; then
        success=true
    fi

    json_output $success "\"$run_stdout\"" "\"$run_stderr\"" $run_exit $compile_time $run_time $timeout_occurred
}

# Main
main() {
    get_code
    get_test

    compile_time=$(compile_code)
    if [[ $? -ne 0 ]]; then
        exit 0
    fi

    run_code "$compile_time"
}

main "$@"
