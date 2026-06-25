#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# TODO: run this script from CK root or build directory
EXE="$(find . -name tile_example_fmha_fwd -type f | head -n 1)"
KNAME=1

export CK_WARMUP=0
export CK_REPEAT=1

COMMON_ARGS='-v=1 -warmup=0 -repeat=1'
# mode=0
# export HIP_VISIBLE_DEVICES=4

TEST_SPLITKV=0
TEST_APPENDKV=0
# options:
#    -s: run splitkv tests
#    -a: run appendkv tests
while getopts ":sa" opt; do
    case "${opt}" in
        s)
            TEST_SPLITKV=1
            ;;
        a)
            TEST_APPENDKV=1
            ;;
        *)
            ;;
    esac
done

run_fp16_bf16_tests() {
    local NUM_SPLITS="1"
    local PAGE_BLOCK_SIZE="0"
    local CACHE_BATCH_IDX="0"

    if [ $TEST_SPLITKV -eq 1 ] ; then
        NUM_SPLITS="$NUM_SPLITS 2 3"
        PAGE_BLOCK_SIZE="$PAGE_BLOCK_SIZE 128"
        CACHE_BATCH_IDX="$CACHE_BATCH_IDX 1"
    fi

    for prec in "fp16"; do 
    for mode in 1 0 ; do
    for perm in 0 1 ; do
    for vlayout in "r" "c" ; do
    for batch in 1 4; do
    for head in 1; do
    for h_k in 1; do
    for q_seq in 128 512 ; do
    for kv_seq in 128 1024; do
    for hdim in 32 64 128 256; do #256 
    for lse in 0 1 ; do
    for bias in "e" ; do
    for p_drop in 0.0 0.2; do # 0.0   
    for mask in "t:2,0,4" "b:1,0,2"; do
    for num_splits in $NUM_SPLITS ; do
    for page_block_size in $PAGE_BLOCK_SIZE ; do
    for cache_batch_idx in $CACHE_BATCH_IDX ; do

    # $EXE -prec=$prec -mode=$mode -b=1 -h=1 -d=$hdim -s=1024 -bias=$bias -p_drop=$p_drop -lse=$lse -iperm=$perm -operm=$perm -vlayout=$vlayout -num_splits=$num_splits -page_block_size=$page_block_size -kname=$KNAME $COMMON_ARGS  
    $EXE -prec=$prec -mode=$mode -b=$batch -h=$head -h_k=$h_k -d=16 -d_v=$hdim -s=$q_seq -s_k=$kv_seq -bias=$bias -p_drop=$p_drop -lse=$lse -iperm=$perm -operm=$perm -vlayout=$vlayout -num_splits=$num_splits -page_block_size=$page_block_size -cache_batch_idx=$cache_batch_idx -kname=$KNAME $COMMON_ARGS -mask=$mask

    done ; done ; done ; done ; done
    done ; done ; done ; done ; done
    done ; done ; done ; done ; done
    done ; done
}


set -x

run_fp16_bf16_tests

set +x
