#!/bin/bash
# monitor_tpus.sh — Dashboard for all nanochat TPU VMs
# Shows: TPU state, training status, last log line, resource usage
#
# Usage:
#   ./monitor_tpus.sh              # One-shot status check
#   ./monitor_tpus.sh --watch      # Continuous refresh (every 60s)
#   ./monitor_tpus.sh --logs       # Tail latest training logs from all active TPUs
#   ./monitor_tpus.sh --json       # JSON output for programmatic use

set -euo pipefail

PROJECT="alpine-aspect-459819-m4"
WSH="/home/dave/.local/share/waveterm/bin/wsh"

# TPU definitions: name:zone:ssh_alias:type
TPUS=(
  "nanochat-v6e-mhc-engram:europe-west4-a:tpu-mhc-engram:v6e-4"
  "nanochat-v6e8-hybrid-eu:europe-west4-a:tpu-v6e8-hybrid:v6e-8"
  "nanochat-v6e8-mtp:europe-west4-a:tpu-v6e8-mtp:v6e-8"
  "nanochat-v6e-longctx:asia-northeast1-b:tpu-longctx:v6e-4"
  "nanochat-v6e-mhc:asia-northeast1-b:tpu-mhc:v6e-4"
  "nanochat-v6e-engram:asia-northeast1-b:tpu-engram:v6e-4"
)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

check_tpu() {
  local name="$1" zone="$2" alias="$3" type="$4"

  # Get TPU state from gcloud
  local state
  state=$(gcloud compute tpus tpu-vm describe "$name" \
    --zone="$zone" --project="$PROJECT" \
    --format='value(state)' 2>/dev/null || echo "NOT_FOUND")

  local state_color="$RED"
  [[ "$state" == "READY" ]] && state_color="$GREEN"
  [[ "$state" == "CREATING" || "$state" == "STARTING" ]] && state_color="$YELLOW"

  # Short display name
  local short="${name#nanochat-}"

  if [[ "$state" != "READY" ]]; then
    printf "${BOLD}%-22s${NC} ${state_color}%-12s${NC} %-6s %s\n" \
      "$short" "$state" "$type" "—"
    return
  fi

  # SSH check: is training running?
  local train_info
  train_info=$(ssh -o ConnectTimeout=5 -o BatchMode=yes "$alias" \
    'ps aux 2>/dev/null | grep -E "scripts\.(base_train|sft_train)" | grep -v grep | head -1' 2>/dev/null || echo "")

  local train_status="${RED}IDLE${NC}"
  local train_detail=""
  if [[ -n "$train_info" ]]; then
    train_status="${GREEN}TRAINING${NC}"
    # Extract the script name and key flags
    if echo "$train_info" | grep -q "sft_train"; then
      train_detail="SFT"
      local data
      data=$(echo "$train_info" | grep -oP '(?<=--data )\S+' | xargs basename 2>/dev/null || echo "")
      [[ -n "$data" ]] && train_detail="SFT: $data"
    else
      train_detail="pretrain"
      local run
      run=$(echo "$train_info" | grep -oP '(?<=--run=)\S+' || echo "")
      [[ -n "$run" ]] && train_detail="pretrain: $run"
    fi
  fi

  # Get last log line from most recently modified log file
  local last_log
  last_log=$(ssh -o ConnectTimeout=5 -o BatchMode=yes "$alias" \
    'newest=$(ls -t ~/train*.log ~/sft_*.log 2>/dev/null | head -1)
     [ -n "$newest" ] && tail -1 "$newest" 2>/dev/null || echo "no log"' 2>/dev/null || echo "no log")
  # Truncate long log lines
  last_log="${last_log:0:80}"

  # Get CPU usage
  local cpu_pct
  cpu_pct=$(ssh -o ConnectTimeout=5 -o BatchMode=yes "$alias" \
    "top -bn1 2>/dev/null | grep 'Cpu(s)' | awk '{print \$2}'" 2>/dev/null || echo "?")

  printf "${BOLD}%-22s${NC} ${state_color}%-12s${NC} %-6s ${train_status}  ${DIM}cpu:${cpu_pct}%%${NC}\n" \
    "$short" "$state" "$type"
  if [[ -n "$train_detail" ]]; then
    printf "  ${CYAN}↳ %s${NC}\n" "$train_detail"
  fi
  if [[ -n "$last_log" && "$last_log" != "no log" ]]; then
    printf "  ${DIM}↳ %s${NC}\n" "$last_log"
  fi
}

show_dashboard() {
  echo -e "${BOLD}╔══════════════════════════════════════════════════════════════════╗${NC}"
  echo -e "${BOLD}║  nanochat TPU Dashboard — $(date '+%Y-%m-%d %H:%M:%S')              ║${NC}"
  echo -e "${BOLD}╚══════════════════════════════════════════════════════════════════╝${NC}"
  echo ""
  printf "${BOLD}%-22s %-12s %-6s %s${NC}\n" "TPU" "STATE" "TYPE" "STATUS"
  printf "%-22s %-12s %-6s %s\n" "───────────────────" "──────────" "──────" "─────────────────"

  for entry in "${TPUS[@]}"; do
    IFS=':' read -r name zone alias type <<< "$entry"
    check_tpu "$name" "$zone" "$alias" "$type"
  done
  echo ""
}

show_logs() {
  echo -e "${BOLD}Training Logs (last 20 lines from active TPUs)${NC}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  for entry in "${TPUS[@]}"; do
    IFS=':' read -r name zone alias type <<< "$entry"
    local short="${name#nanochat-}"

    # Check if SSH reachable + has a log file
    local log_content
    log_content=$(ssh -o ConnectTimeout=5 -o BatchMode=yes "$alias" \
      'newest=$(ls -t ~/train*.log ~/sft_*.log 2>/dev/null | head -1)
       [ -n "$newest" ] && echo "=== $newest ===" && tail -20 "$newest" 2>/dev/null' 2>/dev/null || echo "")

    if [[ -n "$log_content" ]]; then
      echo ""
      echo -e "${BOLD}${CYAN}━━━ $short ($type) ━━━${NC}"
      echo "$log_content"
    fi
  done
}

show_json() {
  echo "["
  local first=true
  for entry in "${TPUS[@]}"; do
    IFS=':' read -r name zone alias type <<< "$entry"
    local state
    state=$(gcloud compute tpus tpu-vm describe "$name" \
      --zone="$zone" --project="$PROJECT" \
      --format='value(state)' 2>/dev/null || echo "NOT_FOUND")

    local training="false"
    local process=""
    if [[ "$state" == "READY" ]]; then
      process=$(ssh -o ConnectTimeout=5 -o BatchMode=yes "$alias" \
        'ps aux 2>/dev/null | grep -E "scripts\.(base_train|sft_train)" | grep -v grep | head -1' 2>/dev/null || echo "")
      [[ -n "$process" ]] && training="true"
    fi

    $first || echo ","
    first=false
    cat <<EOF
  {"name": "$name", "zone": "$zone", "type": "$type", "state": "$state", "training": $training}
EOF
  done
  echo "]"
}

# Main
case "${1:-}" in
  --watch)
    while true; do
      clear
      show_dashboard
      echo -e "${DIM}Refreshing every 60s... (Ctrl+C to stop)${NC}"
      sleep 60
    done
    ;;
  --logs)
    show_logs
    ;;
  --json)
    show_json
    ;;
  *)
    show_dashboard
    ;;
esac
