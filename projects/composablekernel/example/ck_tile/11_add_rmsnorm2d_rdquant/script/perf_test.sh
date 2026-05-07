#!/bin/sh
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

EXE="$(find . -name tile_add_rmsnorm2d_rdquant_fwd -type f | head -n 1)"

$EXE -m=1 -n=1 -e=1e-12 -v=1 -prec=bf16 -repeat=1000
$EXE -m=700 -n=80 -e=1e-12 -v=1 -prec=bf16 -repeat=1000
$EXE -m=700 -n=128 -e=1e-12 -v=1 -prec=bf16 -repeat=1000
$EXE -m=700 -n=144 -e=1e-12 -v=1 -prec=bf16 -repeat=1000
$EXE -m=700 -n=168 -e=1e-12 -v=1 -prec=bf16 -repeat=1000
$EXE -m=700 -n=184 -e=1e-12 -v=1 -prec=bf16 -repeat=1000
$EXE -m=700 -n=256 -e=1e-12 -v=1 -prec=bf16 -repeat=1000
$EXE -m=700 -n=288 -e=1e-12 -v=1 -prec=bf16 -repeat=1000
$EXE -m=700 -n=344 -e=1e-12 -v=1 -prec=bf16 -repeat=1000
$EXE -m=700 -n=376 -e=1e-12 -v=1 -prec=bf16 -repeat=1000
$EXE -m=700 -n=448 -e=1e-12 -v=1 -prec=bf16 -repeat=1000
$EXE -m=700 -n=512 -e=1e-12 -v=1 -prec=bf16 -repeat=1000
$EXE -m=700 -n=924 -e=1e-12 -v=1 -prec=bf16 -repeat=1000
$EXE -m=700 -n=1024 -e=1e-12 -v=1 -prec=bf16 -repeat=1000
$EXE -m=700 -n=1078 -e=1e-12 -v=1 -prec=bf16 -repeat=1000
$EXE -m=700 -n=1996 -e=1e-12 -v=1 -prec=bf16 -repeat=1000
$EXE -m=700 -n=4080 -e=1e-12 -v=1 -prec=bf16 -repeat=1000

$EXE -m=700 -n=80 -e=1e-12 -v=1  -prec=fp16 -repeat=1000
$EXE -m=700 -n=128 -e=1e-12 -v=1 -prec=fp16 -repeat=1000
$EXE -m=700 -n=144 -e=1e-12 -v=1 -prec=fp16 -repeat=1000
$EXE -m=700 -n=168 -e=1e-12 -v=1 -prec=fp16 -repeat=1000
$EXE -m=700 -n=184 -e=1e-12 -v=1 -prec=fp16 -repeat=1000
$EXE -m=700 -n=256 -e=1e-12 -v=1 -prec=fp16 -repeat=1000
$EXE -m=700 -n=288 -e=1e-12 -v=1 -prec=fp16 -repeat=1000
$EXE -m=700 -n=344 -e=1e-12 -v=1 -prec=fp16 -repeat=1000
$EXE -m=700 -n=376 -e=1e-12 -v=1 -prec=fp16 -repeat=1000
$EXE -m=700 -n=448 -e=1e-12 -v=1 -prec=fp16 -repeat=1000
$EXE -m=700 -n=512 -e=1e-12 -v=1 -prec=fp16 -repeat=1000
$EXE -m=700 -n=924 -e=1e-12 -v=1 -prec=fp16 -repeat=1000
$EXE -m=700 -n=1024 -e=1e-12 -v=1 -prec=fp16 -repeat=1000
$EXE -m=700 -n=1078 -e=1e-12 -v=1 -prec=fp16 -repeat=1000
$EXE -m=700 -n=1996 -e=1e-12 -v=1 -prec=fp16 -repeat=1000
$EXE -m=700 -n=4080 -e=1e-12 -v=1 -prec=fp16 -repeat=1000