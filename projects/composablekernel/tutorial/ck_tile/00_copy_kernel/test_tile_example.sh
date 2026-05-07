#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set -euo pipefail

BIN="${BIN:-../../../build/bin/tile_tutorial_copy_kernel}"
WARMUP="${WARMUP:-20}"
REPEAT="${REPEAT:-100}"
VALIDATE="${VALIDATE:-1}"

MS=(128 256 512 1024)
NS=(64 256 1024 2048 4096)
PRECS=(fp16 fp32)

echo "Using BIN=$BIN"
echo "WARMUP=$WARMUP REPEAT=$REPEAT VALIDATE=$VALIDATE"

failures=0

for prec in "${PRECS[@]}"; do
  for m in "${MS[@]}"; do
    for n in "${NS[@]}"; do
      echo "=============================================="
      echo "Running: prec=$prec m=$m n=$n"
      set +e
      out="$("$BIN" -prec="$prec" -m="$m" -n="$n" -warmup="$WARMUP" -repeat="$REPEAT" -v="$VALIDATE" 2>&1)"
      rc=$?
      set -e

      echo "$out"
      if [[ $rc -ne 0 ]]; then
        echo "RUN ERROR (rc=$rc) for m=$m n=$n prec=$prec"
        ((failures++)) || true
        continue
      fi

      if [[ "$VALIDATE" == "1" ]]; then
        if ! grep -q "valid:y" <<<"$out"; then
          echo "VALIDATION FAILED for m=$m n=$n prec=$prec"
          ((failures++)) || true
        fi
      fi
    done
  done
done

echo "=============================================="
if [[ $failures -eq 0 ]]; then
  echo "All runs passed"
else
  echo "$failures runs failed"
fi