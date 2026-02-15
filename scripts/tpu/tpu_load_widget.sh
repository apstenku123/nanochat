#!/bin/bash
# tpu_load_widget.sh — Real-time TPU training monitor for Wave Terminal
# Runs on the TPU VM itself, shows training status + system metrics
# Usage: ssh tpu-xxx 'bash nanochat/scripts/tpu/tpu_load_widget.sh'
#   or:  bash scripts/tpu/tpu_load_widget.sh --remote tpu-mhc

set -uo pipefail

# Colors (safe to use even without TERM set)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# If --remote <host>, SSH to the host and run this script there
if [[ "${1:-}" == "--remote" ]]; then
  HOST="${2:?Usage: $0 --remote <ssh-host>}"
  RATE="${3:-10}"
  # Use -tt to force tty allocation even without local tty
  exec ssh -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -tt "$HOST" \
    "export TERM=xterm-256color; bash ~/nanochat/scripts/tpu/tpu_load_widget.sh --rate $RATE"
fi

RATE=10
[[ "${1:-}" == "--rate" ]] && RATE="${2:-10}"

# Set TERM if not set
: "${TERM:=xterm-256color}"
export TERM

show_status() {
  local hostname
  hostname=$(hostname -s 2>/dev/null || echo "unknown")

  # Get TPU type from metadata
  local tpu_type
  tpu_type=$(curl -s -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type" 2>/dev/null || echo "?")

  # Count TPU chips
  local chip_count
  chip_count=$(ls -d /dev/vfio/[0-9]* 2>/dev/null | wc -l)

  # Training process info
  local train_line
  train_line=$(ps aux 2>/dev/null | grep -E 'scripts\.(base_train|sft_train)' | grep -v grep | head -1 || true)

  # Header
  echo -e "${BOLD}${CYAN}═══ $hostname ($tpu_type, ${chip_count} chips) ═══${NC}  $(date '+%H:%M:%S')"
  echo ""

  if [[ -n "$train_line" ]]; then
    local pid cpu_pct mem_pct rss_kb
    pid=$(echo "$train_line" | awk '{print $2}')
    cpu_pct=$(echo "$train_line" | awk '{print $3}')
    mem_pct=$(echo "$train_line" | awk '{print $4}')
    rss_kb=$(echo "$train_line" | awk '{print $6}')
    local rss_gb
    rss_gb=$(awk "BEGIN{printf \"%.1f\", $rss_kb/1048576}")

    # Uptime of training process
    local elapsed
    elapsed=$(ps -p "$pid" -o etime= 2>/dev/null | tr -d ' ' || echo "?")

    # Determine training type
    local train_type="pretrain"
    echo "$train_line" | grep -q "sft_train" && train_type="SFT"

    # Extract key flags
    local detail=""
    if [[ "$train_type" == "SFT" ]]; then
      local data_file
      data_file=$(echo "$train_line" | grep -oP '(?<=--data )\S+' | xargs basename 2>/dev/null || echo "")
      [[ -n "$data_file" ]] && detail="$data_file"
    else
      local run_name
      run_name=$(echo "$train_line" | grep -oP '(?<=--run[= ])\S+' || echo "")
      [[ -n "$run_name" ]] && detail="run=$run_name"
    fi

    echo -e "  ${GREEN}● TRAINING${NC}  ${BOLD}$train_type${NC}  ${DIM}pid:$pid  uptime:$elapsed${NC}"
    [[ -n "$detail" ]] && echo -e "  ${DIM}↳ $detail${NC}"
    echo -e "  ${DIM}cpu: ${cpu_pct}%  mem: ${mem_pct}%  rss: ${rss_gb}G${NC}"
  else
    echo -e "  ${RED}● IDLE${NC}  No training process detected"
  fi

  echo ""

  # Latest log lines
  local newest
  newest=$(ls -t ~/train*.log ~/sft_*.log 2>/dev/null | head -1 || true)
  if [[ -n "$newest" ]]; then
    local logname
    logname=$(basename "$newest")
    echo -e "${BOLD}Log:${NC} ${DIM}$logname${NC}"

    # Show last 5 lines
    tail -5 "$newest" 2>/dev/null | while IFS= read -r line; do
      echo -e "  ${DIM}$line${NC}"
    done

    # Calculate step progress if visible
    local last_step_line
    last_step_line=$(tail -1 "$newest" 2>/dev/null || true)
    if echo "$last_step_line" | grep -qP 'step \d+/\d+'; then
      local current total pct
      current=$(echo "$last_step_line" | grep -oP 'step \K\d+')
      total=$(echo "$last_step_line" | grep -oP 'step \d+/\K\d+')
      if [[ -n "$current" && -n "$total" && "$total" -gt 0 ]]; then
        pct=$(awk "BEGIN{printf \"%.1f\", $current*100/$total}")
        # Progress bar
        local bar_width=30
        local filled=$(awk "BEGIN{printf \"%d\", $pct*$bar_width/100}")
        local empty=$((bar_width - filled))
        local bar=""
        for ((i=0; i<filled; i++)); do bar+="█"; done
        for ((i=0; i<empty; i++)); do bar+="░"; done
        echo ""
        echo -e "  ${BOLD}Progress:${NC} [${GREEN}${bar}${NC}] ${pct}%  (${current}/${total})"
      fi
    fi
  else
    echo -e "${DIM}No log files found${NC}"
  fi

  echo ""

  # System memory summary
  local mem_total mem_used mem_avail
  read -r _ mem_total mem_used _ _ _ mem_avail < <(free -g | grep Mem)
  echo -e "${BOLD}System:${NC} RAM ${mem_used}G/${mem_total}G used, ${mem_avail}G available"

  # Load average
  local loadavg
  loadavg=$(awk '{print $1, $2, $3}' /proc/loadavg 2>/dev/null)
  local ncpu
  ncpu=$(nproc 2>/dev/null || echo "?")
  echo -e "  ${DIM}load: $loadavg  cpus: $ncpu${NC}"

  # Disk
  local disk_info
  disk_info=$(df -h / 2>/dev/null | tail -1 | awk '{print $3 "/" $2 " (" $5 ")"}')
  echo -e "  ${DIM}disk: $disk_info${NC}"
}

# Main loop
while true; do
  # Use ANSI escape to clear screen (works without TERM properly set)
  printf '\033[2J\033[H'
  show_status
  echo ""
  echo -e "${DIM}Refreshing every ${RATE}s...${NC}"
  sleep "$RATE"
done
