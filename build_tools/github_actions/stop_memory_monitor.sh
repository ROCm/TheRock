#!/bin/bash
# Script to stop memory monitoring for build process
# This script gracefully stops the memory monitor and displays its output

set -euo pipefail

# Set default values if environment variables are not set
BUILD_DIR="${BUILD_DIR:-build}"
GITHUB_JOB_NAME="${GITHUB_JOB_NAME:-default}"
MONITOR_PID="${MONITOR_PID:-}"
PID_FILE="${BUILD_DIR}/logs/monitor_pid_${GITHUB_JOB_NAME}.txt"
STOP_SIGNAL_FILE="${BUILD_DIR}/logs/stop_monitor_${GITHUB_JOB_NAME}.signal"
MONITOR_OUTPUT="${BUILD_DIR}/logs/monitor_output_${GITHUB_JOB_NAME}.txt"

# Detect Windows-like environments (MSYS/MINGW/CYGWIN)
UNAME_OUT="$(uname -s 2>/dev/null || true)"
case "${UNAME_OUT}" in
  MINGW*|MSYS*|CYGWIN*)
    IS_WINDOWS=1
    ;;
  *)
    IS_WINDOWS=0
    ;;
esac

# Fallback to PID file if MONITOR_PID not provided
if [ -z "${MONITOR_PID}" ] && [ -f "${PID_FILE}" ]; then
  MONITOR_PID=$(cat "${PID_FILE}")
fi

# Check if PID was provided
if [ -z "${MONITOR_PID}" ]; then
  echo "Error: MONITOR_PID not set and PID file missing: ${PID_FILE}" >&2
  exit 1
fi

echo "Stopping memory monitor (PID: ${MONITOR_PID})"

if [ "${IS_WINDOWS}" -eq 1 ]; then
  # Create the stop signal file to trigger graceful shutdown
  touch "${STOP_SIGNAL_FILE}"
  echo "Stop signal file created: ${STOP_SIGNAL_FILE}"

  # Wait for graceful shutdown (up to 5 seconds)
  for i in {1..10}; do
    if ! kill -0 "${MONITOR_PID}" 2>/dev/null; then
      echo "Memory monitor stopped gracefully"
      break
    fi
    sleep 0.5
  done

  # If still running, force kill the process
  if kill -0 "${MONITOR_PID}" 2>/dev/null; then
    echo "Graceful shutdown timed out, forcing termination"
    # Use taskkill on Windows for more reliable termination
    if command -v taskkill &> /dev/null; then
      taskkill //F //PID "${MONITOR_PID}" 2>/dev/null || true
    else
      kill -9 "${MONITOR_PID}" 2>/dev/null || true
    fi
  fi
else
  # Send interrupt signal to stop monitoring gracefully
  if kill -0 "${MONITOR_PID}" 2>/dev/null; then
    kill -SIGINT "${MONITOR_PID}"
    sleep 2
    # Force kill if still running
    kill -9 "${MONITOR_PID}" 2>/dev/null || true
  fi
fi

echo "Memory monitor stopped"

# Display the monitor output
if [ -f "${MONITOR_OUTPUT}" ]; then
  echo "=== Memory Monitor Output ==="
  cat "${MONITOR_OUTPUT}"
fi
