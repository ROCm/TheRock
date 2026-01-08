#!/usr/bin/env bash
# Stop memory monitoring for Windows builds
# shell environment and avoid CUID generation issues in HIP compilation

set -euo pipefail

MonitorMemory="${1:-false}"

# Stop memory monitoring if it was started
if [[ "${MonitorMemory}" == "true" ]]; then
  PID_FILE="${BUILD_DIR}/logs/memory_monitor.pid"
  STOP_SIGNAL_FILE="${BUILD_DIR}/logs/stop_monitor_${GITHUB_JOB}.signal"
  MONITOR_OUTPUT="${BUILD_DIR}/logs/monitor_output_${GITHUB_JOB}.txt"

  if [[ -f "${PID_FILE}" ]]; then
    MONITOR_PID=$(cat "${PID_FILE}")
    echo "Stopping memory monitor (PID: ${MONITOR_PID})"

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
  fi

  # Display the monitor output file if it exists
  if [[ -f "${MONITOR_OUTPUT}" ]]; then
    echo ""
    echo "=== Memory Monitor Output ==="
    cat "${MONITOR_OUTPUT}"
    echo "=== End of Memory Monitor Output ==="
  else
    echo "Warning: Monitor output file not found: ${MONITOR_OUTPUT}"
  fi
else
  echo "Memory monitoring was disabled"
fi
