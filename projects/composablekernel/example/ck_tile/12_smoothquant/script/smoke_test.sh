#!/bin/sh
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

EXE="$(find . -name tile_smoothquant -type f | head -n 1)"

for pr_i in "fp16" "bf16" ; do
$EXE -prec=$pr_i -m=99  -n=13
$EXE -prec=$pr_i -m=17  -n=16
$EXE -prec=$pr_i -m=1   -n=100
$EXE -prec=$pr_i -m=4   -n=128
$EXE -prec=$pr_i -m=80  -n=127
$EXE -prec=$pr_i -m=22  -n=255 -stride=256
$EXE -prec=$pr_i -m=7   -n=599
$EXE -prec=$pr_i -m=19  -n=512
$EXE -prec=$pr_i -m=33  -n=313 -stride=1000
$EXE -prec=$pr_i -m=11  -n=510
$EXE -prec=$pr_i -m=171 -n=676 -stride=818
$EXE -prec=$pr_i -m=91  -n=636
$EXE -prec=$pr_i -m=12  -n=768 -stride=800
$EXE -prec=$pr_i -m=100 -n=766 -stride=812
$EXE -prec=$pr_i -m=31  -n=1024
$EXE -prec=$pr_i -m=64  -n=1000 -stride=1004
$EXE -prec=$pr_i -m=8   -n=1501
$EXE -prec=$pr_i -m=3   -n=1826
$EXE -prec=$pr_i -m=5   -n=2040
$EXE -prec=$pr_i -m=7   -n=2734
$EXE -prec=$pr_i -m=1   -n=3182
$EXE -prec=$pr_i -m=9   -n=4096
$EXE -prec=$pr_i -m=3   -n=8192
$EXE -prec=$pr_i -m=1   -n=10547
$EXE -prec=$pr_i -m=3   -n=17134
done
