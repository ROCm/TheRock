#!/bin/sh
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT


EXE=./build/bin/tile_example_batched_transpose

for pr in "fp8" "fp16" "bf16"; do
for pipeline in "0" "1"; do
$EXE -pr=$pr -pipeline=$pipeline -N=1 -C=32 -H=1 -W=32 -layout_in='NCHW' -layout_out='NHWC'
$EXE -pr=$pr -pipeline=$pipeline -N=1 -C=64 -H=1 -W=64 -layout_in='NCHW' -layout_out='NHWC'
$EXE -pr=$pr -pipeline=$pipeline -N=2 -C=12 -H=1 -W=32 -layout_in='NHWC' -layout_out='NCHW'
$EXE -pr=$pr -pipeline=$pipeline -N=3 -C=1334 -H=1 -W=37 -layout_in='NHWC' -layout_out='NCHW'
$EXE -pr=$pr -pipeline=$pipeline -N=4 -C=27 -H=1 -W=32 -layout_in='NCHW' -layout_out='NHWC'
$EXE -pr=$pr -pipeline=$pipeline -N=5 -C=1234 -H=1 -W=12 -layout_in='NCHW' -layout_out='NHWC'
$EXE -pr=$pr -pipeline=$pipeline -N=1 -C=1 -H=1 -W=1 -layout_in='NCHW' -layout_out='NHWC'
$EXE -pr=$pr -pipeline=$pipeline -N=1 -C=1 -H=1 -W=1 -layout_in='NHWC' -layout_out='NCHW'
$EXE -pr=$pr -pipeline=$pipeline -N=128 -C=1024 -H=64 -W=64 -layout_in='NCHW' -layout_out='NHWC'
$EXE -pr=$pr -pipeline=$pipeline -N=128 -C=1024 -H=64 -W=64 -layout_in='NHWC' -layout_out='NCHW'
$EXE -pr=$pr -pipeline=$pipeline -N=16 -C=64 -H=32 -W=128 -layout_in='NCHW' -layout_out='NHWC'
$EXE -pr=$pr -pipeline=$pipeline -N=16 -C=64 -H=128 -W=32 -layout_in='NHWC' -layout_out='NCHW'
$EXE -pr=$pr -pipeline=$pipeline -N=1 -C=2048 -H=1 -W=1 -layout_in='NCHW' -layout_out='NHWC'
$EXE -pr=$pr -pipeline=$pipeline -N=1 -C=2048 -H=1 -W=1 -layout_in='NHWC' -layout_out='NCHW'
$EXE -pr=$pr -pipeline=$pipeline -N=1 -C=1 -H=1024 -W=1024 -layout_in='NCHW' -layout_out='NHWC'
$EXE -pr=$pr -pipeline=$pipeline -N=1 -C=1 -H=1024 -W=1024 -layout_in='NHWC' -layout_out='NCHW'
$EXE -pr=$pr -pipeline=$pipeline -N=8 -C=16 -H=8 -W=16 -layout_in='NCHW' -layout_out='NHWC'
$EXE -pr=$pr -pipeline=$pipeline -N=8 -C=16 -H=8 -W=16 -layout_in='NHWC' -layout_out='NCHW'
$EXE -pr=$pr -pipeline=$pipeline -N=1 -C=64 -H=1 -W=1024 -layout_in='NCHW' -layout_out='NHWC'
$EXE -pr=$pr -pipeline=$pipeline -N=1 -C=64 -H=1024 -W=1 -layout_in='NHWC' -layout_out='NCHW'

done
done
