#!/bin/bash
# open_wave_dashboard.sh — Launch TPU monitoring widgets in Wave Terminal
# Opens TPU load widgets for all active TPUs + the monitoring dashboard

WSH="/home/dave/.local/share/waveterm/bin/wsh"

echo "Opening TPU monitoring dashboard in Wave Terminal..."

# Launch the main monitoring dashboard in a terminal block
$WSH run -c "bash /home/dave/source/nanochat/scripts/tpu/monitor_tpus.sh --watch" &

# Launch TPU load widgets (shows training status, logs, progress bars)
for widget in tpu-load-mhcengram tpu-load-v6e8hybrid tpu-load-v6e8mtp tpu-load-longctx tpu-load-mhc tpu-load-engram; do
  $WSH launch "$widget" 2>/dev/null &
done

wait
echo "Dashboard widgets launched!"
