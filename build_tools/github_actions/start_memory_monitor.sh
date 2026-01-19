#!/bin/bash
# Script to start memory monitoring for build process
# This script starts the memory monitor in the background and captures its PID

set -euo pipefail

# Discover repository root
REPO_ROOT="$(git rev-parse --show-toplevel)"

# Set default values if environment variables are not set
BUILD_DIR="${BUILD_DIR:-build}"
GITHUB_JOB_NAME="${GITHUB_JOB_NAME:-default}"
PHASE="${PHASE:-Build Phase}"
MEMORY_MONITOR_INTERVAL="${MEMORY_MONITOR_INTERVAL:-30}"

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

# Create logs directory in BUILD_DIR
mkdir -p "${BUILD_DIR}/logs"

# Get the parent process PID (the shell running this script's parent)
PARENT_PID=$PPID

# Set max runtime to 24 hours (matching workflow timeout) or use environment variable override
# This ensures the monitor won't outlive the workflow even if other safeguards fail
MAX_RUN_TIME=${MAX_RUN_TIME:-$((24 * 3600))}

# Prepare log file paths
LOG_FILE="${BUILD_DIR}/logs/build_memory_log_${GITHUB_JOB_NAME}.jsonl"
STOP_SIGNAL_FILE="${BUILD_DIR}/logs/stop_monitor_${GITHUB_JOB_NAME}.signal"
MONITOR_OUTPUT="${BUILD_DIR}/logs/monitor_output_${GITHUB_JOB_NAME}.txt"
PID_FILE="${BUILD_DIR}/logs/monitor_pid_${GITHUB_JOB_NAME}.txt"

# Start memory monitor in background
if [ "${IS_WINDOWS}" -eq 1 ]; then
  python "${REPO_ROOT}/build_tools/memory_monitor.py" \
    --phase "${PHASE}" \
    --interval "${MEMORY_MONITOR_INTERVAL}" \
    --log-file "${LOG_FILE}" \
    --stop-signal-file "${STOP_SIGNAL_FILE}" \
    --background \
    --max-runtime "${MAX_RUN_TIME}" \
    > "${MONITOR_OUTPUT}" 2>&1 &
else
  python "${REPO_ROOT}/build_tools/memory_monitor.py" \
    --phase "${PHASE}" \
    --interval "${MEMORY_MONITOR_INTERVAL}" \
    --log-file "${LOG_FILE}" \
    --parent-pid "${PARENT_PID}" \
    --background \
    --max-runtime "${MAX_RUN_TIME}" \
    > "${MONITOR_OUTPUT}" 2>&1 &
fi

# Capture PID
MONITOR_PID=${!}
echo "Memory monitoring started with PID: ${MONITOR_PID}" >&2

# Export PID for use in other scripts
export MONITOR_PID
echo "${MONITOR_PID}" > "${PID_FILE}"

# Wait for the background process to fully start and stabilize
# This grace period helps ensure the Python process has initialized and prevents
# race conditions where the script exits before the monitor is actually running
sleep 2

# Output the PID to stdout for capture by calling scripts
echo "${MONITOR_PID}"
