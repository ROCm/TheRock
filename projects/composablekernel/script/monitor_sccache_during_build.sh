#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Continuous monitoring script for sccache during builds
# Usage: ./monitor_sccache_during_build.sh [log_prefix] &

LOG_PREFIX=${1:-"sccache_monitor"}

# Include stage name in log filename if available
STAGE_SUFFIX=""
if [ -n "${STAGE_NAME}" ]; then
    # Convert stage name to filename-safe format (replace spaces and special chars with underscores)
    STAGE_SAFE=$(echo "${STAGE_NAME}" | sed 's/[^a-zA-Z0-9]/_/g' | sed 's/__*/_/g' | sed 's/^_\|_$//g')
    STAGE_SUFFIX="_${STAGE_SAFE}"
fi

MONITOR_LOG="logs/${LOG_PREFIX}_$(date +%Y%m%d_%H%M%S)${STAGE_SUFFIX}.log"
MONITOR_INTERVAL=30  # seconds

echo "Starting sccache monitoring - logging to $MONITOR_LOG"
echo "Monitor interval: $MONITOR_INTERVAL seconds"

# Function to log with timestamp
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MONITOR_LOG"
}

# Function to get sccache stats safely
get_sccache_stats() {
    if command -v sccache &> /dev/null; then
        sccache --show-stats 2>/dev/null || echo "sccache stats unavailable"
    else
        echo "sccache command not found"
    fi
}

# Function to check if sccache server is running
is_sccache_running() {
    if command -v sccache &> /dev/null; then
        sccache --show-stats &> /dev/null
        return $?
    else
        return 1
    fi
}

# Function to test Redis connectivity
test_redis_connectivity() {
    # Use SCCACHE_REDIS if set, otherwise construct from CK_SCCACHE
    local REDIS_URL=""
    if [ -n "${SCCACHE_REDIS}" ]; then
        REDIS_URL="${SCCACHE_REDIS}"
    elif [ -n "${CK_SCCACHE}" ]; then
        REDIS_URL="redis://${CK_SCCACHE}"
    fi
    
    if [ -n "${REDIS_URL}" ]; then
        local start_time=$(date +%s%N)
        local response=$(timeout 5 redis-cli -u "${REDIS_URL}" ping 2>&1) || response="TIMEOUT"
        local end_time=$(date +%s%N)
        local latency=$(( (end_time - start_time) / 1000000 ))
        echo "Redis: $response (${latency}ms)"
    else
        echo "Redis: No Redis URL available"
    fi
}

# Gets the last sccache stats before exiting
cleanup() {
    log_with_timestamp "=== FINAL SCCACHE STATS EXIT ==="
    log_with_timestamp "$(get_sccache_stats)"
    echo "=== CONTINUOUS MONITORING STOPPED ==="
    # List monitoring logs
    echo "=== MONITORING LOGS ==="
    ls -la logs/*monitor*.log 2>/dev/null || echo "No monitoring logs found"
}
trap cleanup EXIT

log_with_timestamp "=== SCCACHE MONITORING STARTED ==="
log_with_timestamp "PID: $$"
log_with_timestamp "Node: ${NODE_NAME:-$(hostname)}"
log_with_timestamp "Stage: ${STAGE_NAME:-unknown}"
log_with_timestamp "WORKSPACE_PATH: ${WORKSPACE:-not set}"
log_with_timestamp "SCCACHE_C_CUSTOM_CACHE_BUSTER: ${SCCACHE_C_CUSTOM_CACHE_BUSTER:-not set}"
log_with_timestamp "CK_SCCACHE: ${CK_SCCACHE:-not set}"

# Initial state
log_with_timestamp "=== INITIAL STATE ==="
# Reset sscache stats
sccache --zero-stats
log_with_timestamp "$(get_sccache_stats) $(test_redis_connectivity)"

# Monitor loop
while true; do
    sleep $MONITOR_INTERVAL
    
    # Check if sccache server is still running
    if ! is_sccache_running; then
        log_with_timestamp "WARNING: sccache server not running!"
    fi
    
    # Get current stats
    current_stats=$(get_sccache_stats)
    redis_status=$(test_redis_connectivity)
    
    # Log current cache hit information
    log_with_timestamp "$(get_sccache_stats) $(test_redis_connectivity)"
    
    # Check for Redis latency issues
    if echo "$redis_status" | grep -E "[0-9]{3,}" > /dev/null; then  # >100ms latency
        log_with_timestamp "HIGH REDIS LATENCY detected"
    fi
    
    # Check for Redis connection failures
    if echo "$redis_status" | grep -E "(TIMEOUT|Connection refused|No route)" > /dev/null; then
        log_with_timestamp "REDIS CONNECTION FAILURE detected"
    fi
done