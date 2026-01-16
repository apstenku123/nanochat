#!/bin/bash
# Kill the training run

if [ -f run.pid ]; then
    PID=$(cat run.pid)
    if kill -0 $PID 2>/dev/null; then
        kill $PID
        echo "Killed process $PID"
    else
        echo "Process $PID not running"
    fi
    rm -f run.pid
else
    echo "No run.pid file found"
fi
