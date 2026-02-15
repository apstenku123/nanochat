#!/bin/bash
# watchdog.sh — Training failure detection with alerts
# Runs as a background daemon, checks all TPUs every 5 minutes.
# Sends desktop notifications and optional email when training dies.
#
# Usage:
#   ./watchdog.sh                    # Run in foreground
#   nohup ./watchdog.sh &            # Run as daemon
#   ./watchdog.sh --once             # Single check, exit
#   ./watchdog.sh --setup-gcp-alert  # Create GCP Cloud Monitoring alert policy

set -euo pipefail

PROJECT="alpine-aspect-459819-m4"
CHECK_INTERVAL=300  # 5 minutes
ALERT_EMAIL="${ALERT_EMAIL:-dave@cppcode.online}"
WSH="/home/dave/.local/share/waveterm/bin/wsh"
LOGFILE="${HOME}/.nanochat_watchdog.log"
STATE_DIR="${HOME}/.nanochat_watchdog_state"

mkdir -p "$STATE_DIR"

# TPU definitions: name:zone:ssh_alias:expected_training
# Set expected_training to "yes" if this TPU should be actively training
# Update these as you start/stop experiments
WATCHED_TPUS=(
  "nanochat-v6e-mhc-engram:europe-west4-a:tpu-mhc-engram:yes"
  "nanochat-v6e8-mtp:europe-west4-a:tpu-v6e8-mtp:yes"
  "nanochat-v6e-mhc:asia-northeast1-b:tpu-mhc:yes"
)

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"
}

alert() {
  local title="$1" message="$2" severity="${3:-warning}"

  log "ALERT [$severity]: $title — $message"

  # Desktop notification
  if command -v notify-send &>/dev/null; then
    local urgency="normal"
    [[ "$severity" == "critical" ]] && urgency="critical"
    notify-send -u "$urgency" "🔴 $title" "$message" 2>/dev/null || true
  fi

  # Wave Terminal notification
  if [[ -x "$WSH" ]]; then
    "$WSH" notify -t "$title" "$message" 2>/dev/null || true
  fi

  # Optional: healthchecks.io ping (set HEALTHCHECKS_URL env var)
  if [[ -n "${HEALTHCHECKS_URL:-}" ]]; then
    curl -fsS -m 10 --retry 3 "${HEALTHCHECKS_URL}/fail" \
      --data-raw "$title: $message" 2>/dev/null || true
  fi
}

check_tpu_training() {
  local name="$1" zone="$2" alias="$3" expected="$4"
  local state_file="$STATE_DIR/${name}.state"
  local prev_state
  prev_state=$(cat "$state_file" 2>/dev/null || echo "unknown")

  # Check TPU VM state
  local tpu_state
  tpu_state=$(gcloud compute tpus tpu-vm describe "$name" \
    --zone="$zone" --project="$PROJECT" \
    --format='value(state)' 2>/dev/null || echo "UNREACHABLE")

  local short="${name#nanochat-}"

  if [[ "$tpu_state" == "PREEMPTED" ]]; then
    if [[ "$prev_state" != "PREEMPTED" ]]; then
      alert "TPU Preempted: $short" \
        "TPU $name in $zone was preempted. Babysitter should auto-recover." \
        "critical"
    fi
    echo "PREEMPTED" > "$state_file"
    return
  fi

  if [[ "$tpu_state" != "READY" ]]; then
    if [[ "$prev_state" != "$tpu_state" ]]; then
      alert "TPU State: $short" \
        "TPU $name is $tpu_state (expected READY)" \
        "warning"
    fi
    echo "$tpu_state" > "$state_file"
    return
  fi

  # TPU is READY — check if training is running
  if [[ "$expected" != "yes" ]]; then
    echo "READY_IDLE_OK" > "$state_file"
    return
  fi

  local is_training
  is_training=$(ssh -o ConnectTimeout=10 -o BatchMode=yes "$alias" \
    'pgrep -f "scripts\.(base_train|sft_train)" > /dev/null 2>&1 && echo "YES" || echo "NO"' 2>/dev/null || echo "SSH_FAIL")

  if [[ "$is_training" == "YES" ]]; then
    # Training is running — check for stale log (no update in 30 min)
    local log_age
    log_age=$(ssh -o ConnectTimeout=5 -o BatchMode=yes "$alias" \
      'newest=$(ls -t ~/train*.log ~/sft_*.log 2>/dev/null | head -1)
       if [ -n "$newest" ]; then
         echo $(( $(date +%s) - $(stat -c %Y "$newest") ))
       else
         echo 999999
       fi' 2>/dev/null || echo "999999")

    if [[ "$log_age" -gt 1800 ]]; then
      if [[ "$prev_state" != "STALE_LOG" ]]; then
        alert "Training Stalled: $short" \
          "Training process alive but log hasn't updated in $((log_age/60)) minutes" \
          "warning"
      fi
      echo "STALE_LOG" > "$state_file"
    else
      echo "TRAINING_OK" > "$state_file"
    fi
  elif [[ "$is_training" == "NO" ]]; then
    if [[ "$prev_state" != "TRAINING_DEAD" ]]; then
      # Check if it just finished (exit code 0) or crashed
      local last_line
      last_line=$(ssh -o ConnectTimeout=5 -o BatchMode=yes "$alias" \
        'for f in ~/train.log ~/sft_*.log ~/train_*.log; do
           [ -f "$f" ] && tail -3 "$f" 2>/dev/null && break
         done' 2>/dev/null || echo "no log")

      alert "Training Died: $short" \
        "Expected training on $name but process not found. Last log: ${last_line:0:200}" \
        "critical"
    fi
    echo "TRAINING_DEAD" > "$state_file"
  else
    if [[ "$prev_state" != "SSH_FAIL" ]]; then
      alert "SSH Failed: $short" \
        "Cannot SSH to $name ($alias) — network issue?" \
        "warning"
    fi
    echo "SSH_FAIL" > "$state_file"
  fi
}

run_checks() {
  log "--- Watchdog check ---"
  for entry in "${WATCHED_TPUS[@]}"; do
    IFS=':' read -r name zone alias expected <<< "$entry"
    check_tpu_training "$name" "$zone" "$alias" "$expected"
  done

  # Ping healthchecks.io success (dead man's switch)
  if [[ -n "${HEALTHCHECKS_URL:-}" ]]; then
    curl -fsS -m 10 --retry 3 "$HEALTHCHECKS_URL" 2>/dev/null || true
  fi
}

setup_gcp_alert() {
  echo "Setting up GCP Cloud Monitoring alert for idle TPUs..."

  # Create email notification channel
  local channel_id
  channel_id=$(gcloud alpha monitoring channels list \
    --project="$PROJECT" \
    --filter="type='email' AND labels.email_address='$ALERT_EMAIL'" \
    --format='value(name)' 2>/dev/null | head -1)

  if [[ -z "$channel_id" ]]; then
    echo "Creating notification channel for $ALERT_EMAIL..."
    channel_id=$(gcloud alpha monitoring channels create \
      --project="$PROJECT" \
      --display-name="nanochat-alerts" \
      --description="Alert channel for nanochat TPU training" \
      --type=email \
      --channel-labels=email_address="$ALERT_EMAIL" \
      --format='value(name)' 2>/dev/null)
    echo "Created channel: $channel_id"
  else
    echo "Using existing channel: $channel_id"
  fi

  # Create alert policy for TensorCore idle > 1 hour
  local policy_file
  policy_file=$(mktemp /tmp/tpu-alert-XXXXX.json)
  cat > "$policy_file" <<POLICY
{
  "displayName": "nanochat: TPU TensorCore Idle > 1 Hour",
  "combiner": "OR",
  "conditions": [
    {
      "displayName": "TensorCore idle duration exceeds 1 hour",
      "conditionThreshold": {
        "filter": "metric.type=\"tpu.googleapis.com/tpu/tensorcore/idle_duration\" AND resource.type=\"tpu_worker\"",
        "comparison": "COMPARISON_GT",
        "thresholdValue": 3600,
        "duration": "300s",
        "trigger": {
          "count": 1
        },
        "aggregations": [
          {
            "alignmentPeriod": "300s",
            "perSeriesAligner": "ALIGN_MAX"
          }
        ]
      }
    }
  ],
  "notificationChannels": ["$channel_id"],
  "documentation": {
    "content": "A TPU TensorCore has been idle for more than 1 hour. Check if training has crashed or completed.",
    "mimeType": "text/markdown"
  },
  "alertStrategy": {
    "autoClose": "1800s"
  }
}
POLICY

  gcloud alpha monitoring policies create \
    --project="$PROJECT" \
    --policy-from-file="$policy_file" 2>/dev/null && \
    echo "Alert policy created successfully!" || \
    echo "Failed to create alert policy (may already exist)"

  rm -f "$policy_file"
  echo ""
  echo "Done! You'll receive email at $ALERT_EMAIL when any TPU TensorCore is idle > 1 hour."
}

# Main
case "${1:-}" in
  --once)
    run_checks
    ;;
  --setup-gcp-alert)
    setup_gcp_alert
    ;;
  *)
    log "Watchdog started (interval: ${CHECK_INTERVAL}s)"
    log "Watching: ${WATCHED_TPUS[*]}"
    while true; do
      run_checks
      sleep "$CHECK_INTERVAL"
    done
    ;;
esac
