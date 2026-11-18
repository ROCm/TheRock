#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Unattended driver for the fingerprint cache benchmark.
#
# Default MODE=verify: configure-only (NO builds). Verifies the fingerprints are
# reproducible, then changes each subproject one at a time, re-runs *configure*
# (seconds), and records which fingerprints move -- the invalidation blast radius
# you need to draft topology-based reuse rules. Runs in minutes.
#
# MODE=build runs real cold/warm/per-component builds with wall-clock timings
# (hours). MODE=simulate mutates fingerprints without touching sources (seconds).
#
# Run from the TheRock repo root after fetch_sources.py. ClickHouse creds are
# optional (CLICKHOUSE_URL/_USER/_PASSWORD) for the cloud columns.
#
# Tunables (env): MODE (verify|build|simulate), FAMILIES (gfx110X-all),
# MAX_COMPONENTS (0=all), BUILD_TARGET (build mode only), OUTPUT.
set -euo pipefail

MODE="${MODE:-verify}"
FAMILIES="${FAMILIES:-gfx110X-all}"
MAX_COMPONENTS="${MAX_COMPONENTS:-0}"
OUTPUT="${OUTPUT:-build/fingerprint_cache_report}"
LOG="${LOG:-fingerprint_benchmark.log}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
SCRIPT=build_tools/benchmarks/fingerprint_cache_benchmark.py
[ -f "$SCRIPT" ] || { echo "Run from the TheRock repo root." >&2; exit 1; }

log "=== fingerprint cache benchmark: mode=$MODE families=$FAMILIES ==="
[ -n "${CLICKHOUSE_URL:-}" ] && log "cloud: ClickHouse enabled" || log "cloud: local only"

if [ -f build_tools/setup_ccache.py ]; then
  eval "$(python3 build_tools/setup_ccache.py 2>/dev/null || true)"
fi
export CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache

ARGS=("$MODE" --families "$FAMILIES" --output "$OUTPUT")
[ "$MODE" != "simulate" ] && ARGS+=(--max-components "$MAX_COMPONENTS")
[ "$MODE" = "build" ] && [ -n "${BUILD_TARGET:-}" ] && ARGS+=(--build-target "$BUILD_TARGET")
[ "$MODE" = "build" ] && [ "${SKIP_CLEAN:-0}" != "1" ] && { log "cold build: removing build/"; rm -rf build; }

log "running (report is rewritten after every step; open the .html anytime)"
python3 "$SCRIPT" "${ARGS[@]}" 2>&1 | tee -a "$LOG"
log "=== done === Report: $OUTPUT.md and $OUTPUT.html"
