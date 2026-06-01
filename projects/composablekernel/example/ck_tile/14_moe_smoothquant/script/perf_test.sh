# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT


EXE=build/bin/tile_example_moe_smoothquant

$EXE -t=1 -h=1  -v=1 -prec=bf16 -repeat=1000
$EXE -t=700 -h=80  -v=1 -prec=bf16 -repeat=1000
$EXE -t=700 -h=128  -v=1 -prec=bf16 -repeat=1000
$EXE -t=700 -h=144  -v=1 -prec=bf16 -repeat=1000
$EXE -t=700 -h=168  -v=1 -prec=bf16 -repeat=1000
$EXE -t=700 -h=184  -v=1 -prec=bf16 -repeat=1000
$EXE -t=700 -h=256  -v=1 -prec=bf16 -repeat=1000
$EXE -t=700 -h=288  -v=1 -prec=bf16 -repeat=1000
$EXE -t=700 -h=344  -v=1 -prec=bf16 -repeat=1000
$EXE -t=700 -h=376  -v=1 -prec=bf16 -repeat=1000
$EXE -t=700 -h=448  -v=1 -prec=bf16 -repeat=1000
$EXE -t=700 -h=512  -v=1 -prec=bf16 -repeat=1000
$EXE -t=700 -h=924  -v=1 -prec=bf16 -repeat=1000
$EXE -t=700 -h=1024  -v=1 -prec=bf16 -repeat=1000
$EXE -t=700 -h=1078  -v=1 -prec=bf16 -repeat=1000
$EXE -t=700 -h=1996  -v=1 -prec=bf16 -repeat=1000
$EXE -t=700 -h=4080  -v=1 -prec=bf16 -repeat=1000

$EXE -t=700 -h=80  -v=1  -prec=fp16 -repeat=1000
$EXE -t=700 -h=128  -v=1 -prec=fp16 -repeat=1000
$EXE -t=700 -h=144  -v=1 -prec=fp16 -repeat=1000
$EXE -t=700 -h=168  -v=1 -prec=fp16 -repeat=1000
$EXE -t=700 -h=184  -v=1 -prec=fp16 -repeat=1000
$EXE -t=700 -h=256  -v=1 -prec=fp16 -repeat=1000
$EXE -t=700 -h=288  -v=1 -prec=fp16 -repeat=1000
$EXE -t=700 -h=344  -v=1 -prec=fp16 -repeat=1000
$EXE -t=700 -h=376  -v=1 -prec=fp16 -repeat=1000
$EXE -t=700 -h=448  -v=1 -prec=fp16 -repeat=1000
$EXE -t=700 -h=512  -v=1 -prec=fp16 -repeat=1000
$EXE -t=700 -h=924  -v=1 -prec=fp16 -repeat=1000
$EXE -t=700 -h=1024  -v=1 -prec=fp16 -repeat=1000
$EXE -t=700 -h=1078  -v=1 -prec=fp16 -repeat=1000
$EXE -t=700 -h=1996  -v=1 -prec=fp16 -repeat=1000
$EXE -t=700 -h=4080  -v=1 -prec=fp16 -repeat=1000