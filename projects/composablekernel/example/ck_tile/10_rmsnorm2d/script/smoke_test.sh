#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT


EXE="$(find . -name tile_rmsnorm2d_fwd -type f | head -n 1)"

total=0
valid=0

run_case() {
  cmd="$EXE -prec_i=$1 -fadd=$2 -s=$3 $4 -m=$5 -n=$6 $7"
  echo "[CMD] $cmd"
  output=$($cmd 2>&1)
  echo "$output"
  if echo "$output" | grep -q "valid:y"; then
    valid=$((valid + 1))
  fi
  total=$((total + 1))
}

fquant_list=(
  ""
  "-fquant=1 -prec_o=int8"
  "-fquant=2 -prec_o=int8"
  "-fquant=1 -prec_o=fp8"
  "-fquant=2 -prec_o=fp8"
  "-fquant=1 -prec_o=int8 -save_unquant=1"
  "-fquant=2 -prec_o=int8 -save_unquant=1"
  "-fquant=1 -prec_o=fp8 -save_unquant=1"
  "-fquant=2 -prec_o=fp8 -save_unquant=1"
)

m_n_list=(
  "99 13" "17 16" "1 100" "4 128" "80 127"
  "7 599" "19 512" "11 510" "91 636"
  "31 1024" "8 1501" "3 1826" "5 2040"
  "7 2734" "1 3182" "9 4096" "3 8192"
)

### Add special stride test ###
m_n_stride_list=(
  "22 255 -x_stride=256 -xr_stride=256 -y_stride=256 -yr_stride=256"
  "33 313 -x_stride=1000 -xr_stride=1000 -y_stride=1000 -yr_stride=1000"
  "171 676 -x_stride=818 -xr_stride=818 -y_stride=818 -yr_stride=818"
  "12 768 -x_stride=800 -xr_stride=800 -y_stride=800 -yr_stride=800"
  "100 766 -x_stride=812 -xr_stride=812 -y_stride=812 -yr_stride=812"
  "64 1000 -x_stride=1004 -xr_stride=1004 -y_stride=1004 -yr_stride=1004"
)

for fquant in "${fquant_list[@]}"; do
  for pr_i in "fp16" "bf16"; do
    for fadd in "0" "1"; do
      for s in "0" "1"; do
        for pair in "${m_n_list[@]}"; do
          m=$(echo $pair | cut -d ' ' -f1)
          n=$(echo $pair | cut -d ' ' -f2)
          run_case "$pr_i" "$fadd" "$s" "$fquant" "$m" "$n" ""
        done

        ### Running tests with stride ###
        for triple in "${m_n_stride_list[@]}"; do
          m=$(echo $triple | cut -d ' ' -f1)
          n=$(echo $triple | cut -d ' ' -f2)
          stride_args=$(echo $triple | cut -d ' ' -f3-)
          run_case "$pr_i" "$fadd" "$s" "$fquant" "$m" "$n" "$stride_args"
        done
      done
    done
  done
done

# Special two-pass only
for pr_i in "fp16" "bf16"; do
  for fadd in "0" "1"; do
    for s in "0" "1"; do
      run_case "$pr_i" "$fadd" "$s" "" "1" "10547" ""
    done
  done
done

# Summary
echo "=============================="
echo "Total cases: $total"
echo "Valid cases: $valid"
accuracy=$(awk "BEGIN {printf \"%.2f\", ($valid / $total) * 100}")
echo "Accuracy: $accuracy%"
echo "=============================="
