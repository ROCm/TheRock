#!/bin/bash
# Script to stop memory monitoring for build process
# This script gracefully stops the memory monitor and displays its output

set -euo pipefail

# Set default values if environment variables are not set
BUILD_DIR="${BUILD_DIR:-build}"
JOB_NAME="${JOB_NAME:-default}"
MONITOR_PID="${MONITOR_PID:-}"

# Check if PID was provided
if [ -z "${MONITOR_PID}" ]; then
  echo "Error: MONITOR_PID environment variable not set" >&2
  echo "Usage: MONITOR_PID=<pid> ${0}" >&2
  exit 1
fi

echo "Stopping memory monitor (PID: ${MONITOR_PID})"

# Send interrupt signal to stop monitoring gracefully
if kill -0 "${MONITOR_PID}" 2>/dev/null; then
  kill -SIGINT "${MONITOR_PID}"
  sleep 2
  # Force kill if still running
  kill -9 "${MONITOR_PID}" 2>/dev/null || true
fi

echo "Memory monitor stopped"

# Display the monitor output
if [ -f "${BUILD_DIR}/logs/monitor_output_${JOB_NAME}.txt" ]; then
  echo "=== Memory Monitor Output ==="
  cat "${BUILD_DIR}/logs/monitor_output_${JOB_NAME}.txt"
fi
