#!/bin/bash
# open_wave_dashboard.sh — Launch TPU monitoring widgets in Wave Terminal
# Opens sysinfo widgets for all active TPUs + the monitoring dashboard

WSH="/home/dave/.local/share/waveterm/bin/wsh"

echo "Opening TPU monitoring dashboard in Wave Terminal..."

# Launch the main monitoring dashboard in a terminal block
$WSH run -c "bash /home/dave/source/nanochat/scripts/tpu/monitor_tpus.sh --watch" &

# Launch sysinfo widgets for each connected TPU
for widget in tpu-sysinfo-mhcengram tpu-sysinfo-v6e8hybrid tpu-sysinfo-v6e8mtp tpu-sysinfo-longctx tpu-sysinfo-mhc tpu-sysinfo-engram; do
  $WSH launch "$widget" 2>/dev/null &
done

wait
echo "Dashboard widgets launched!"
