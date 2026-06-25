#!/bin/sh
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT


# TODO: run this script from CK root or build directory
EXE="$(find . -name tile_example_fmha_fwd_v3 -type f | head -n 1)"
VALID=0

for causal in 0 1 ; do
for prec in "fp16" "bf16" ; do
for hdim in 128 ; do
for perm in 0 ; do

$EXE -prec=$prec -b=32 -h=16        -s=512   -d=$hdim -causal=$causal -iperm=$perm -operm=$perm -v=$VALID
$EXE -prec=$prec -b=16 -h=16        -s=1024  -d=$hdim -causal=$causal -iperm=$perm -operm=$perm -v=$VALID
$EXE -prec=$prec -b=8  -h=16        -s=2048  -d=$hdim -causal=$causal -iperm=$perm -operm=$perm -v=$VALID
$EXE -prec=$prec -b=4  -h=16        -s=4096  -d=$hdim -causal=$causal -iperm=$perm -operm=$perm -v=$VALID
$EXE -prec=$prec -b=2  -h=16        -s=8192  -d=$hdim -causal=$causal -iperm=$perm -operm=$perm -v=$VALID
$EXE -prec=$prec -b=1  -h=16        -s=16384 -d=$hdim -causal=$causal -iperm=$perm -operm=$perm -v=$VALID
                                          
$EXE -prec=$prec -b=1  -h=64        -s=16384 -d=$hdim -causal=$causal -iperm=$perm -operm=$perm -v=$VALID
$EXE -prec=$prec -b=1  -h=16 -h_k=1 -s=65536 -d=$hdim -causal=$causal -iperm=$perm -operm=$perm -v=$VALID
$EXE -prec=$prec -b=1  -h=40        -s=37200 -d=$hdim -causal=$causal -iperm=$perm -operm=$perm -v=$VALID

done
done
done
done

# Padding benchmark comparisons for v3 (batch mode only)
# ==== V3 Padding Benchmarks: batch mode (baseline vs low/med/high pad) ====
prec="fp16"
base_v3_args="-prec=$prec -b=4 -h=16 -d=128 -s=1024 -mask=0 -iperm=0 -operm=0 -v=$VALID"

# baseline (no pad)
$EXE $base_v3_args

# low pad (≈90–95% effective)
$EXE $base_v3_args -q_eff_lens=1024,960,992,896 -kv_eff_lens=1024,960,992,896

# medium pad (≈60–75% effective)
$EXE $base_v3_args -q_eff_lens=896,768,512,640 -kv_eff_lens=896,768,512,640

# high pad (≈30–40% effective)
$EXE $base_v3_args -q_eff_lens=512,384,256,320 -kv_eff_lens=512,384,256,320
