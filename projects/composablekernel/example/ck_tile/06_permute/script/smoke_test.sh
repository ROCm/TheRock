#!/bin/sh
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# TODO: run this script from CK root
BUILD=build
EXE=$BUILD/bin/tile_example_permute
COMMON_ARGS='-v=1 -warmup=0 -repeat=1'
# mode=0
# export HIP_VISIBLE_DEVICES=4
if [ $# -ge 1 ] ; then
    set -x
fi

$EXE -prec=fp16 -shape=3,6,4,32,16,2,8 -perm=0,1,4,2,5,3,6  $COMMON_ARGS
$EXE -prec=fp16 -shape=5,10,4,32,8,2,8 -perm=0,1,4,2,5,3,6  $COMMON_ARGS
$EXE -prec=fp16 -shape=3,8,4,16,16,4,8 -perm=0,1,4,2,5,3,6  $COMMON_ARGS
$EXE -prec=fp16 -shape=3,6,4,32,16,2,8 -perm=0,1,2,4,5,3,6  $COMMON_ARGS
$EXE -prec=fp16 -shape=5,10,4,32,8,2,8 -perm=0,1,2,4,5,3,6  $COMMON_ARGS
$EXE -prec=fp16 -shape=3,8,4,16,16,4,8 -perm=0,1,2,4,5,3,6  $COMMON_ARGS
$EXE -prec=fp16 -shape=2,8,16,8,4,8 -perm=0,1,3,4,2,5  $COMMON_ARGS
$EXE -prec=fp16 -shape=1,24,32,16,2,8 -perm=0,1,3,4,2,5  $COMMON_ARGS

echo "------------------------------------------------------------------"

for prec in "fp8" "fp16" "fp32" ; do

$EXE -prec=$prec -shape=3,8 -perm=1,0 $COMMON_ARGS
$EXE -prec=$prec -shape=48,6,8 -perm=2,1,0  $COMMON_ARGS
$EXE -prec=$prec -shape=24,128,3 -perm=0,2,1  $COMMON_ARGS
$EXE -prec=$prec -shape=4,10,7,6 -perm=0,2,3,1  $COMMON_ARGS
$EXE -prec=$prec -shape=8,24,36,10 -perm=3,1,2,0  $COMMON_ARGS
$EXE -prec=$prec -shape=8,1,36,4 -perm=2,1,0,3  $COMMON_ARGS
$EXE -prec=$prec -shape=5,10,16,2,36,4 -perm=4,5,2,1,0,3  $COMMON_ARGS
$EXE -prec=$prec -shape=2,32,8,3,6,2,5,4 -perm=5,2,4,7,1,6,3,0  $COMMON_ARGS
echo "------------------------------------------------------------------"
done
