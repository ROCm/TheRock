#!/bin/sh
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT


EXE=./build/bin/tile_example_batched_transpose

for C in "64" "256" "1024" "4096" "16384"; do
for W in "64" "256" "1024" "4096" "16384"; do
for pr in "fp8" "fp16" "bf16"; do
for pipeline in "0" "1"; do

$EXE -pipeline=$pipeline -pr=$pr -N=1 -C=$C -H=1 -W=$W -layout_in='NCHW' -layout_out='NHWC'

done
done
done
done