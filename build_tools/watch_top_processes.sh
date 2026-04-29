#!/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Reports top processes periodically. Meant to be run in the background before
# starting a build with high latency tasks so that some progress can be seen.

function cycle() {
  echo ""
  echo ":::: TOP PROCESSES $(date) ::::"
  ps --no-headers aux | sort -nrk 3,3 | head -n 5
  echo ""
  echo ""
}

while true; do
  # Capture content and print in one write to attempt to avoid tearing on
  # rapidly moving terminals.
  report="$(cycle)"
  echo "$report" 1>&2
  sleep 30
done
