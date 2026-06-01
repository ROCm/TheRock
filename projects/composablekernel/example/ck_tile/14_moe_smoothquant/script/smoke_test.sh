#!/bin/sh
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

EXE=build/bin/tile_example_moe_smoothquant

for pr_i in "fp16" "bf16" ; do
for pr_o in "int8" "fp8" ; do
$EXE -prec_i=$pr_i -prec_o=$pr_o -t=99  -h=13
$EXE -prec_i=$pr_i -prec_o=$pr_o -t=17  -h=16
$EXE -prec_i=$pr_i -prec_o=$pr_o -t=1   -h=100
$EXE -prec_i=$pr_i -prec_o=$pr_o -t=4   -h=128
$EXE -prec_i=$pr_i -prec_o=$pr_o -t=80  -h=127
$EXE -prec_i=$pr_i -prec_o=$pr_o -t=22  -h=255 -stride=256
$EXE -prec_i=$pr_i -prec_o=$pr_o -t=7   -h=599
$EXE -prec_i=$pr_i -prec_o=$pr_o -t=19  -h=512
$EXE -prec_i=$pr_i -prec_o=$pr_o -t=33  -h=313 -stride=1000
$EXE -prec_i=$pr_i -prec_o=$pr_o -t=11  -h=510
$EXE -prec_i=$pr_i -prec_o=$pr_o -t=171 -h=676 -stride=818
$EXE -prec_i=$pr_i -prec_o=$pr_o -t=91  -h=636
$EXE -prec_i=$pr_i -prec_o=$pr_o -t=12  -h=768 -stride=800
$EXE -prec_i=$pr_i -prec_o=$pr_o -t=100 -h=766 -stride=812
$EXE -prec_i=$pr_i -prec_o=$pr_o -t=31  -h=1024
$EXE -prec_i=$pr_i -prec_o=$pr_o -t=64  -h=1000 -stride=1004
$EXE -prec_i=$pr_i -prec_o=$pr_o -t=8   -h=1501
$EXE -prec_i=$pr_i -prec_o=$pr_o -t=3   -h=1826
$EXE -prec_i=$pr_i -prec_o=$pr_o -t=5   -h=2040
$EXE -prec_i=$pr_i -prec_o=$pr_o -t=7   -h=2734
$EXE -prec_i=$pr_i -prec_o=$pr_o -t=1   -h=3182
$EXE -prec_i=$pr_i -prec_o=$pr_o -t=9   -h=4096
$EXE -prec_i=$pr_i -prec_o=$pr_o -t=3   -h=8192
$EXE -prec_i=$pr_i -prec_o=$pr_o -t=1   -h=10547
$EXE -prec_i=$pr_i -prec_o=$pr_o -t=3   -h=17134
done
done
