#!/bin/bash
set -e

REPO_DIR="/opt/gpus/instance-config"
LOG_FILE="/var/log/gpus-agent.log"
PYTHON="/venv/main/bin/python"

# Determine group name: prefer running process arg, fallback to env var
GROUP=$(pgrep -a -f "setup.py" 2>/dev/null | grep -v grep | awk '{print $NF}' | head -1)
if [ -z "$GROUP" ]; then
    GROUP="${INSTANCE_GROUP_NAME:-}"
fi
if [ -z "$GROUP" ]; then
    echo "ERROR: Cannot determine group name. Set INSTANCE_GROUP_NAME or have setup.py running." >&2
    exit 1
fi

echo "[restart] Pulling latest code..."
git -C "$REPO_DIR" pull --ff-only

echo "[restart] Stopping existing setup.py (group=$GROUP)..."
pkill -f "setup.py" 2>/dev/null || true
sleep 1

echo "[restart] Starting setup.py..."
nohup "$PYTHON" "$REPO_DIR/setup.py" "$GROUP" >> "$LOG_FILE" 2>&1 &
echo "[restart] Started PID=$! group=$GROUP, log=$LOG_FILE"
