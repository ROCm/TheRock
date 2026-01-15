#!/bin/bash
# Script to start memory monitoring for build process
# This script starts the memory monitor in the background and captures its PID

set -euo pipefail

# Discover repository root
REPO_ROOT="$(git rev-parse --show-toplevel)"

# Set default values if environment variables are not set
BUILD_DIR="${BUILD_DIR:-build}"
JOB_NAME="${JOB_NAME:-default}"
PHASE="${PHASE:-Build Phase}"
MEMORY_MONITOR_INTERVAL="${MEMORY_MONITOR_INTERVAL:-30}"

# Create logs directory in BUILD_DIR
mkdir -p "${BUILD_DIR}/logs"

# Get the parent process PID (the shell running this script's parent)
PARENT_PID=$PPID

# Set max runtime to 24 hours (matching workflow timeout) or use environment variable override
# This ensures the monitor won't outlive the workflow even if other safeguards fail
MAX_RUN_TIME=${MAX_RUN_TIME:-$((24 * 3600))}

# Start memory monitor in background
python "${REPO_ROOT}/build_tools/memory_monitor.py" \
  --phase "${PHASE}" \
  --interval "${MEMORY_MONITOR_INTERVAL}" \
  --log-file "${BUILD_DIR}/logs/build_memory_log_${JOB_NAME}.jsonl" \
  --parent-pid "${PARENT_PID}" \
  --background \
  --max-runtime "${MAX_RUN_TIME}" \
  > "${BUILD_DIR}/logs/monitor_output_${JOB_NAME}.txt" 2>&1 &

# Capture PID
MONITOR_PID=${!}
echo "Memory monitoring started with PID: ${MONITOR_PID}" >&2

# Export PID for use in other scripts
export MONITOR_PID
echo "${MONITOR_PID}" > "${BUILD_DIR}/logs/monitor_pid_${JOB_NAME}.txt"

# Wait for the background process to fully start and stabilize
# This grace period helps ensure the Python process has initialized and prevents
# race conditions where the script exits before the monitor is actually running
sleep 2

# Output the PID to stdout for capture by calling scripts
echo "${MONITOR_PID}"
