#!/usr/bin/env bash
# Start memory monitoring for Windows builds
# shell environment and avoid CUID generation issues in HIP compilation

set -euo pipefail

# Set default values if environment variables are not set
MEMORY_MONITOR_INTERVAL="${MEMORY_MONITOR_INTERVAL:-30}"
PHASE="${PHASE:-Build Phase}"

# Create logs directory
mkdir -p "${BUILD_DIR}/logs"

# Prepare log file paths with Windows-style paths converted if needed
LOG_FILE="${BUILD_DIR}/logs/build_memory_log_${GITHUB_JOB}.jsonl"
STOP_SIGNAL_FILE="${BUILD_DIR}/logs/stop_monitor_${GITHUB_JOB}.signal"
MONITOR_OUTPUT="${BUILD_DIR}/logs/monitor_output_${GITHUB_JOB}.txt"
PID_FILE="${BUILD_DIR}/logs/memory_monitor.pid"

# Set max runtime to 24 hours (matching workflow timeout) or use environment variable override
# This ensures the monitor won't outlive the workflow even if other safeguards fail
MAX_RUN_TIME=${MAX_RUN_TIME:-$((24 * 3600))}

# Start the memory monitor in the background
python build_tools/memory_monitor.py \
  --phase "${PHASE}" \
  --interval "${MEMORY_MONITOR_INTERVAL}" \
  --log-file "${LOG_FILE}" \
  --stop-signal-file "${STOP_SIGNAL_FILE}" \
  --max-runtime "${MAX_RUN_TIME}" \
  --background \
  > "${MONITOR_OUTPUT}" 2>&1 &

# Save the PID
MONITOR_PID=${!}
echo "${MONITOR_PID}" > "${PID_FILE}"

echo "Memory monitoring started with PID: ${MONITOR_PID}"
