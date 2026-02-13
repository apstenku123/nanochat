#!/usr/bin/env bash
# ==============================================================
# Setup script for ashg.lists@gmail.com
# A100 GPU VM: small-a100-40gb (34.133.4.26)
# Project: alpine-aspect-459819-m4 | Zone: us-central1-f
# ==============================================================
#
# What this does:
#   1. Checks/installs gcloud CLI
#   2. Authenticates your Google account
#   3. Configures SSH so you can: ssh a100
#   4. You get passwordless sudo automatically (via OS Login)
#
# Usage:
#   chmod +x setup_a100_access.sh
#   ./setup_a100_access.sh
#
# After running, just:
#   ssh a100           # connect to the machine
#   sudo su -          # no password needed
# ==============================================================

set -euo pipefail

PROJECT="alpine-aspect-459819-m4"
ZONE="us-central1-f"
INSTANCE="small-a100-40gb"
EXTERNAL_IP="34.133.4.26"

echo "=== A100 VM Access Setup ==="
echo ""

# --- Step 1: Check for gcloud ---
if ! command -v gcloud &>/dev/null; then
    echo "gcloud CLI not found. Installing..."
    if [[ "$(uname)" == "Darwin" ]]; then
        echo "On macOS, install via: brew install --cask google-cloud-sdk"
        echo "Or: https://cloud.google.com/sdk/docs/install"
        exit 1
    elif [[ "$(uname)" == "Linux" ]]; then
        echo "Installing gcloud CLI..."
        curl -fsSL https://sdk.cloud.google.com | bash -s -- --disable-prompts
        export PATH="$HOME/google-cloud-sdk/bin:$PATH"
    fi
fi

echo "[OK] gcloud CLI found: $(command -v gcloud)"

# --- Step 2: Authenticate ---
CURRENT_ACCOUNT=$(gcloud config get-value account 2>/dev/null || true)
if [[ "$CURRENT_ACCOUNT" != "ashg.lists@gmail.com" ]]; then
    echo ""
    echo "Logging in as ashg.lists@gmail.com..."
    gcloud auth login ashg.lists@gmail.com
fi
echo "[OK] Authenticated as: $(gcloud config get-value account 2>/dev/null)"

# --- Step 3: Set project ---
gcloud config set project "$PROJECT" 2>/dev/null
echo "[OK] Project set to: $PROJECT"

# --- Step 4: Generate SSH keys and push to OS Login ---
echo ""
echo "Setting up SSH keys via OS Login..."
# This generates a key pair if needed and registers it with Google
gcloud compute os-login ssh-keys add --key-file="$HOME/.ssh/google_compute_engine.pub" 2>/dev/null || {
    # If no key exists yet, gcloud compute ssh will create one
    echo "Keys will be generated on first SSH connection."
}

# --- Step 5: Configure SSH for easy access ---
echo ""
echo "Configuring SSH..."

# Get the OS Login username (Google maps email to a POSIX username)
OSLOGIN_USER=$(gcloud compute os-login describe-profile --format="value(posixAccounts[0].username)" 2>/dev/null || echo "")

# Create/update SSH config
SSH_CONFIG="$HOME/.ssh/config"
mkdir -p "$HOME/.ssh"

# Remove old entry if exists
if [[ -f "$SSH_CONFIG" ]]; then
    # Remove existing a100 block
    sed -i.bak '/^# --- A100 VM START ---$/,/^# --- A100 VM END ---$/d' "$SSH_CONFIG" 2>/dev/null || true
fi

cat >> "$SSH_CONFIG" <<SSHEOF

# --- A100 VM START ---
Host a100
    HostName $EXTERNAL_IP
    User ${OSLOGIN_USER:-ext_ashg_lists_gmail_com}
    IdentityFile ~/.ssh/google_compute_engine
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    LogLevel ERROR
# --- A100 VM END ---
SSHEOF

chmod 600 "$SSH_CONFIG"
echo "[OK] SSH config updated: $SSH_CONFIG"

# --- Step 6: Test connection ---
echo ""
echo "=== Setup Complete ==="
echo ""
echo "You now have access to:"
echo "  Instance: $INSTANCE"
echo "  IP:       $EXTERNAL_IP"
echo "  Zone:     $ZONE"
echo ""
echo "Connect using ANY of these methods:"
echo ""
echo "  1. Easy way:     ssh a100"
echo "  2. gcloud way:   gcloud compute ssh $INSTANCE --zone=$ZONE"
echo ""
echo "Sudo works without password:"
echo "  sudo su -"
echo "  sudo apt install ..."
echo ""
echo "Want to test now? Running: gcloud compute ssh $INSTANCE --zone=$ZONE --command='whoami && sudo whoami'"
read -p "Press Enter to test, or Ctrl+C to skip... "
gcloud compute ssh "$INSTANCE" --zone="$ZONE" --command="echo 'SSH: OK (logged in as $(whoami))' && sudo echo 'SUDO: OK (passwordless)'"
