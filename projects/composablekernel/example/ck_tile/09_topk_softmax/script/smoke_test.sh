#!/bin/sh
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT


EXE=./build/bin/tile_example_topk_softmax

for pr_i in "fp16" "bf16" ; do
$EXE -pr_i=$pr_i -t=80 -e=17
$EXE -pr_i=$pr_i -t=111 -e=117
$EXE -pr_i=$pr_i -t=1000 -e=55
$EXE -pr_i=$pr_i -t=99 -e=180
$EXE -pr_i=$pr_i -t=175 -e=64 -k=8
$EXE -pr_i=$pr_i -t=65 -e=8 -k=2
$EXE -pr_i=$pr_i -t=1 -e=25
$EXE -pr_i=$pr_i -t=31 -e=19 -k=15
$EXE -pr_i=$pr_i -t=81 -e=37 -k=7
$EXE -pr_i=$pr_i -t=199 -e=128 -k=13
$EXE -pr_i=$pr_i -t=23 -e=1 -k=1
$EXE -pr_i=$pr_i -t=127 -e=99 -k=19 -st_i=233 -st_o=31
$EXE -pr_i=$pr_i -t=71 -e=11 -k=11 -st_i=30 -st_o=12
$EXE -pr_i=$pr_i -t=1 -e=1 -k=1
$EXE -pr_i=$pr_i -t=99 -e=2 -k=1 -st_i=11 -st_o=5
$EXE -pr_i=$pr_i -t=333 -e=99 -k=13 -st_i=191 -st_o=17
done
