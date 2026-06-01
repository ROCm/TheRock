#!/bin/sh
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# TODO: run this script from CK root or build directory
EXE="$(find . -name tile_example_fmha_fwd -type f | head -n 1)"
VALID=0

for prec in "fp16" "bf16" ; do
for perm in 0 1 ; do
for hdim in 64 128 256 ; do

nhead=$((2048 / $hdim))     # follow fav2 setup
$EXE -prec=$prec -b=32 -h=$nhead -d=$hdim -s=512   -iperm=$perm -operm=$perm -kname=1 -v=$VALID ; sleep 3
$EXE -prec=$prec -b=16 -h=$nhead -d=$hdim -s=1024  -iperm=$perm -operm=$perm -kname=1 -v=$VALID ; sleep 3
$EXE -prec=$prec -b=8  -h=$nhead -d=$hdim -s=2048  -iperm=$perm -operm=$perm -kname=1 -v=$VALID ; sleep 3
$EXE -prec=$prec -b=4  -h=$nhead -d=$hdim -s=4096  -iperm=$perm -operm=$perm -kname=1 -v=$VALID ; sleep 3
$EXE -prec=$prec -b=2  -h=$nhead -d=$hdim -s=8192  -iperm=$perm -operm=$perm -kname=1 -v=$VALID ; sleep 3
$EXE -prec=$prec -b=1  -h=$nhead -d=$hdim -s=16384 -iperm=$perm -operm=$perm -kname=1 -v=$VALID ; sleep 3

done
done
done

#Padding Benchmarks: batch mode (baseline vs low/med/high pad)
prec="fp16"
base_batch_args="-prec=$prec -mode=0 -b=4 -h=16 -h_k=16 -d=128 -s=1024 -bias=n -mask=0 -lse=0 -iperm=0 -operm=0 -vlayout=r -kname=1 -v=$VALID"

# baseline (no pad)
$EXE $base_batch_args

# low pad (≈90–95% effective)
$EXE $base_batch_args -q_eff_lens=1024,960,992,896 -kv_eff_lens=1024,960,992,896

# medium pad (≈60–75% effective)
$EXE $base_batch_args -q_eff_lens=896,768,512,640 -kv_eff_lens=896,768,512,640

# high pad (≈30–40% effective)
$EXE $base_batch_args -q_eff_lens=512,384,256,320 -kv_eff_lens=512,384,256,320

# Padding Benchmarks: group mode (baseline vs low/med/high physical pad)
seqlens_q="1024,768,512,256"
seqlens_k="1024,768,512,256"
base_group_args="-prec=$prec -mode=1 -b=4 -h=16 -h_k=16 -d=128 -s=$seqlens_q -s_k=$seqlens_k -bias=n -mask=0 -lse=0 -iperm=0 -operm=0 -vlayout=r -kname=1 -v=$VALID"

# baseline (no physical pad)
$EXE $base_group_args

# low physical pad
$EXE $base_group_args -s_qpad=1152,896,576,320 -s_kpad=1152,896,576,320

# medium physical pad
$EXE $base_group_args -s_qpad=1536,1152,768,384 -s_kpad=1536,1152,768,384

# high physical pad
$EXE $base_group_args -s_qpad=2048,1536,1024,512 -s_kpad=2048,1536,1024,512
