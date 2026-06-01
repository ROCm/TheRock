#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

BIN=./bin/tile_example_gemm_weight_preshuffle
PREC=fp8
VERBOSITY=2

# List of all (m, n, k) triplets
ARGS_LIST=(
  "1 2048 5120"
  "1 5120 1024"
  "2 2048 5120"
  "2 5120 1024"
  "3 2048 5120"
  "3 5120 1024"
  "4 2048 5120"
  "4 5120 1024"
  "5 2048 5120"
  "5 5120 1024"
  "6 2048 5120"
  "6 5120 1024"
  "7 2048 5120"
  "7 5120 1024"
  "8 2048 5120"
  "8 5120 1024"
  "9 2048 5120"
  "9 5120 1024"
  "10 2048 5120"
  "10 5120 1024"
  "11 2048 5120"
  "11 5120 1024"
  "12 2048 5120"
  "12 5120 1024"
  "13 2048 5120"
  "13 5120 1024"
  "14 2048 5120"
  "14 5120 1024"
  "15 2048 5120"
  "15 5120 1024"
  "16 64 128"
  "16 64 256"
  "16 2048 5120"
  "16 5120 1024"
  "512 768 640"
  "1024 1792 896"
  "1536 2816 1152"
  "2048 5120 1024"
  "2048 5120 8192"
  "2048 7168 8192"
  "2048 8192 3584"
  "16384 7168 8192"
  "16384 8192 3584"
)

# Output file
OUTPUT_FILE="gemm_profile_results.csv"

# Output header
echo "m,n,k,Pipeline,Time_ms,TFlops,GBps,Verification" > "$OUTPUT_FILE"

# Loop over each argument set
for args in "${ARGS_LIST[@]}"; do
  read -r m n k <<< "$args"

  echo "Testing: m=$m, n=$n, k=$k"
  OUTPUT=$($BIN -m=$m -n=$n -k=$k -prec=$PREC -v=$VERBOSITY 2>/dev/null)

  # Extract pipeline information
  # Format: "Launching kernel with args: gemm_fp8_pipeline_AGmemBGmemCRegV2_128x256x256x256_16x16x128_16x16_0x0x0"
  PIPELINE=$(echo "$OUTPUT" | grep "Launching kernel with args:" | sed -n 's/.*Launching kernel with args: \(.*\)/\1/p')

  # Extract TFlops and GB/s from the output
  # Format: "Run Gemm kernel with M=3840 N=4096 K=2048 ... : 0.042338 ms, 1521.67 TFlops, 1126.89 GB/s,"
  PERF_LINE=$(echo "$OUTPUT" | grep "TFlops")

  # Extract verification result
  # Format: "The GPU verification result is:correct" (note: no space after colon)
  VERIFICATION=$(echo "$OUTPUT" | grep "The GPU verification result is:" | sed -n 's/.*The GPU verification result is:\(.*\)/\1/p')

  if [ -n "$PERF_LINE" ]; then
    # Extract execution time in ms
    TIME_MS=$(echo "$PERF_LINE" | grep -o '[0-9]\+\.[0-9]\+ ms' | grep -o '[0-9]\+\.[0-9]\+')
    # Extract TFlops value - more robust regex
    TFLOPS=$(echo "$PERF_LINE" | grep -o '[0-9]\+\.[0-9]\+ TFlops' | grep -o '[0-9]\+\.[0-9]\+')
    # Extract GB/s value - more robust regex
    GBPS=$(echo "$PERF_LINE" | grep -o '[0-9]\+\.[0-9]\+ GB/s' | grep -o '[0-9]\+\.[0-9]\+')

    # Use extracted pipeline or default if not found
    if [ -z "$PIPELINE" ]; then
      PIPELINE="gemm_basic"
    fi

    # Print to terminal
    echo "  Pipeline: $PIPELINE"
    echo "  Time: ${TIME_MS} ms"
    echo "  TFlops: ${TFLOPS}"
    echo "  GB/s: ${GBPS}"
    echo "  Verification: ${VERIFICATION:-N/A}"

    
    # Save to CSV file
    echo "$m,$n,$k,$PIPELINE,$TIME_MS,$TFLOPS,$GBPS,$VERIFICATION" >> "$OUTPUT_FILE"
  else
    echo "  ERROR: Could not parse performance data"
    echo ""
    echo "$m,$n,$k,$PIPELINE,,,,$VERIFICATION" >> "$OUTPUT_FILE"
  fi
done

echo "=========================================="
echo "Profile completed!"
echo "Results saved to: $OUTPUT_FILE"
echo "Total tests run: ${#ARGS_LIST[@]}"
echo "=========================================="