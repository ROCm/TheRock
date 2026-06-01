#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# TODO: run this script from CK root or build directory
set -euo pipefail

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
EXE_NAME=tile_example_fmha_fwd
EXE="$(find . -name $EXE_NAME -type f | head -n 1)"
KNAME=1
GPU_arch=$GPU_arch
if [ -z "$GPU_arch" ] ; then
    GPU_arch=$(rocminfo | grep -E 'Name:\s+gfx' | head -n1 | awk '{print $2}')
fi

export CK_WARMUP=0
export CK_REPEAT=1

CURR_FAILS_FILE=${CURR_FAILS_FILE:-"fmha_fwd_fails_$GPU_arch.txt"}
rm -f $CURR_FAILS_FILE
touch $CURR_FAILS_FILE
KNOWN_FAILS_FILE=${KNOWN_FAILS_FILE:-"$SCRIPT_DIR/fmha_fwd_known_fails_$GPU_arch.txt"}

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

run_exe() {
    set +ex
    $EXE $@
    local ret=$?
    if [ $ret -ne 0 ] ; then
        echo "$EXE_NAME $*" >> $CURR_FAILS_FILE
    fi
    set -ex
}

run_fp16_bf16_tests() {
    local NUM_SPLITS="1"
    local PAGE_BLOCK_SIZE="0"
    local CACHE_BATCH_IDX="0"

    if [ $TEST_SPLITKV -eq 1 ] ; then
        NUM_SPLITS="$NUM_SPLITS 2 3"
        PAGE_BLOCK_SIZE="$PAGE_BLOCK_SIZE 128"
        CACHE_BATCH_IDX="$CACHE_BATCH_IDX 1"
    fi

    for prec in "fp16" "bf16" ; do
    for mode in 1 0 ; do
    for perm in 0 1 ; do
    for hdim in 32 64 128 256 ; do
    for lse in 0 1 ; do
    for bias in "n" "e" "a" ; do
    for p_drop in 0.0 0.2 ; do
    for num_splits in $NUM_SPLITS ; do
    for page_block_size in $PAGE_BLOCK_SIZE ; do
    for cache_batch_idx in $CACHE_BATCH_IDX ; do

    # run_exe -prec=$prec -mode=$mode -b=1 -h=1 -d=$hdim -s=1024 -bias=$bias -p_drop=$p_drop -lse=$lse -iperm=$perm -operm=$perm  -num_splits=$num_splits -page_block_size=$page_block_size -kname=$KNAME $COMMON_ARGS
    run_exe -prec=$prec -mode=$mode -b=2 -h=2 -h_k=1 -d=16    -d_v=$hdim -s=55   -s_k=256            -bias=$bias -p_drop=$p_drop -lse=$lse -iperm=$perm -operm=$perm                -num_splits=$num_splits -page_block_size=$page_block_size -cache_batch_idx=$cache_batch_idx -kname=$KNAME $COMMON_ARGS
    run_exe -prec=$prec -mode=$mode -b=1 -h=3        -d=$hdim            -s=100  -s_k=51             -bias=$bias -p_drop=$p_drop -lse=$lse -iperm=$perm -operm=$perm                -num_splits=$num_splits -page_block_size=$page_block_size -cache_batch_idx=$cache_batch_idx -kname=$KNAME $COMMON_ARGS
    run_exe -prec=$prec -mode=$mode -b=2 -h=1        -d=16    -d_v=$hdim -s=99   -s_k=256            -bias=$bias -p_drop=$p_drop -lse=$lse -iperm=$perm -operm=$perm -mask=1        -num_splits=$num_splits -page_block_size=$page_block_size -cache_batch_idx=$cache_batch_idx -kname=$KNAME $COMMON_ARGS
    run_exe -prec=$prec -mode=$mode -b=1 -h=2 -h_k=1 -d=$hdim            -s=1024 -s_k=256            -bias=$bias -p_drop=$p_drop -lse=$lse -iperm=$perm -operm=$perm -mask=2        -num_splits=$num_splits -page_block_size=$page_block_size -cache_batch_idx=$cache_batch_idx -kname=$KNAME $COMMON_ARGS
    run_exe -prec=$prec -mode=$mode -b=2 -h=1        -d=$hdim -d_v=24    -s=3    -s_k=99             -bias=$bias -p_drop=$p_drop -lse=$lse -iperm=$perm -operm=$perm -mask=2        -num_splits=$num_splits -page_block_size=$page_block_size -cache_batch_idx=$cache_batch_idx -kname=$KNAME $COMMON_ARGS
    run_exe -prec=$prec -mode=$mode -b=3 -h=2 -h_k=1 -d=$hdim            -s=200  -s_k=520            -bias=$bias -p_drop=$p_drop -lse=$lse -iperm=$perm -operm=$perm -mask=t:128,30 -num_splits=$num_splits -page_block_size=$page_block_size -cache_batch_idx=$cache_batch_idx -kname=$KNAME $COMMON_ARGS
    run_exe -prec=$prec -mode=$mode -b=2 -h=1        -d=$hdim            -s=99   -s_k=32             -bias=$bias -p_drop=$p_drop -lse=$lse -iperm=$perm -operm=$perm -mask=b:4,35   -num_splits=$num_splits -page_block_size=$page_block_size -cache_batch_idx=$cache_batch_idx -kname=$KNAME $COMMON_ARGS
    run_exe -prec=$prec -mode=$mode -b=1 -h=2 -h_k=1 -d=$hdim            -s=33   -s_k=0              -bias=$bias -p_drop=$p_drop -lse=$lse -iperm=$perm -operm=$perm -mask=2        -num_splits=$num_splits -page_block_size=$page_block_size -cache_batch_idx=$cache_batch_idx -kname=$KNAME $COMMON_ARGS
    run_exe -prec=$prec -mode=$mode -b=1 -h=2 -h_k=1 -d=$hdim            -s=1    -s_k=10  -s_kpad=32 -bias=$bias -p_drop=$p_drop -lse=$lse -iperm=$perm -operm=$perm -mask=2        -num_splits=$num_splits -page_block_size=$page_block_size -cache_batch_idx=$cache_batch_idx -kname=$KNAME $COMMON_ARGS

    done ; done ; done ; done ; done
    done ; done ; done ; done ; done
}

run_fp8bf16_tests() {
    for perm in 0 1 ; do
    for b in 1 2 ; do
    for hdim in 64 128 256 ; do

    $EXE -prec=fp8bf16 -init=3 -b=$b -h=1 -d=$hdim -s=128 -iperm=$perm -operm=$perm -vlayout=r -qscale=1 -kname=$KNAME $COMMON_ARGS

    done ; done ; done
}

run_fp8fp32_tests() {
    for perm in 0 1 ; do
    for b in 1 2 ; do
    for hdim in 128 ; do

    $EXE -prec=fp8fp32 -init=3 -b=$b -h=1 -d=$hdim -s=128 -iperm=$perm -operm=$perm -vlayout=r -qscale=1 -kname=$KNAME $COMMON_ARGS

    done ; done ; done
}

run_fp16_appendkv_tests() {
    for s in $(seq 63 1 65) ; do
    for s_k in 65 129 ; do
    for s_knew in 0 64 $s_k ; do
    for hdim in 32 64 128 256 ; do
    for ri in 0 1 ; do
    for rdim in 0 16 32 $hdim ; do
    for page_block_size in 0 128 ; do
    for cache_batch_idx in 0 1 ; do

    run_exe -prec=fp16 -b=3 -h=3 -d=$hdim -s=$s -s_k=$s_k -s_knew=$s_knew -rotary_dim=$rdim -rotary_interleaved=$ri -page_block_size=$page_block_size -cache_batch_idx=$cache_batch_idx -iperm=1 -operm=1 -kname=1 $COMMON_ARGS

    done ; done ; done ; done ; done
    done ; done ; done
}

run_padding_smoke_tests() {
    # Padding-only smoke tests for batch/group mode using COMMON_ARGS
    local prec="fp16"

    # Batch mode: padding via effective lengths (exclude PAD)
    # Use lse=1 to select a non-trload kernel and avoid overly strict tolerance mismatches
    local base_batch="-prec=$prec -mode=0 -b=4 -h=16 -h_k=16 -d=128 -s=1024 -bias=n -mask=0 -lse=1 -iperm=0 -operm=0 -vlayout=r -kname=$KNAME $COMMON_ARGS"
    # low pad (≈90–95% effective)
    $EXE $base_batch -q_eff_lens=1024,960,992,896 -kv_eff_lens=1024,960,992,896
    # medium pad (≈60–75% effective)
    $EXE $base_batch -q_eff_lens=896,768,512,640 -kv_eff_lens=896,768,512,640
    # high pad (≈30–40% effective)
    $EXE $base_batch -q_eff_lens=512,384,256,320 -kv_eff_lens=512,384,256,320

    # Group mode: padding via physical stride along seqlen
    local seqlens_q="1024,768,512,256"
    local seqlens_k="1024,768,512,256"
    local base_group="-prec=$prec -mode=1 -b=4 -h=16 -h_k=16 -d=128 -s=$seqlens_q -s_k=$seqlens_k -bias=n -mask=0 -lse=0 -iperm=0 -operm=0 -vlayout=r -kname=$KNAME $COMMON_ARGS"
    # low physical pad
    $EXE $base_group -s_qpad=1152,896,576,320 -s_kpad=1152,896,576,320
    # medium physical pad
    $EXE $base_group -s_qpad=1536,1152,768,384 -s_kpad=1536,1152,768,384
    # high physical pad
    $EXE $base_group -s_qpad=2048,1536,1024,512 -s_kpad=2048,1536,1024,512
}

run_padding_basic_boundary_tests() {
    # Basic padding and boundary tests (reference: smoke_test_fwd_pad.sh)
    local prec
    local perm

    # Group mode: Q&K padded with per-batch different strides
    for prec in fp16 bf16 ; do
      for perm in 0 1 ; do
        $EXE -prec=$prec -mode=1 -b=2 -h=2 -h_k=1 -d=16 -d_v=32 \
             -s=55 -s_k=256 -s_qpad=64,60 -s_kpad=272,260 \
             -bias=n -p_drop=0.0 -lse=0 -iperm=$perm -operm=$perm \
             -num_splits=1 -page_block_size=0 -cache_batch_idx=0 -kname=$KNAME $COMMON_ARGS
      done
    done

    # slightly larger, uneven padding strides
    for prec in fp16 bf16 ; do
      for perm in 0 1 ; do
        $EXE -prec=$prec -mode=1 -b=3 -h=2 -h_k=1 -d=64 -d_v=64 \
             -s=50,60,40 -s_k=128,256,192 -s_qpad=64,64,64 -s_kpad=160,288,224 \
             -bias=n -p_drop=0.0 -lse=1 -iperm=$perm -operm=$perm \
             -num_splits=1 -page_block_size=0 -cache_batch_idx=0 -kname=$KNAME $COMMON_ARGS
      done
    done

    # only K padded; Q unpadded
    for prec in fp16 bf16 ; do
      for perm in 0 1 ; do
        $EXE -prec=$prec -mode=1 -b=2 -h=2 -h_k=1 -d=32 -d_v=64 \
             -s=55 -s_k=256 -s_kpad=272,260 \
             -bias=n -p_drop=0.0 -lse=1 -iperm=$perm -operm=$perm \
             -num_splits=1 -page_block_size=0 -cache_batch_idx=0 -kname=$KNAME $COMMON_ARGS
      done
    done

    # use cu_seqlen overrides to skip tail PAD
    for prec in fp16 bf16 ; do
      for perm in 0 1 ; do
        $EXE -prec=$prec -mode=0 -b=4 -h=8 -h_k=8 -d=128 -s=3 -s_k=3 \
             -q_eff_lens=1,2,1,2 -kv_eff_lens=1,2,1,2 \
             -bias=n -p_drop=0.0 -lse=1 -iperm=$perm -operm=$perm \
             -num_splits=1 -page_block_size=0 -cache_batch_idx=0 -kname=$KNAME $COMMON_ARGS

        $EXE -prec=$prec -mode=0 -b=2 -h=2 -h_k=1 -d=32 -d_v=64 -s=64 -s_k=256 \
             -q_eff_lens=55,60 -kv_eff_lens=200,256 \
             -bias=n -p_drop=0.0 -lse=0 -iperm=$perm -operm=$perm \
             -num_splits=1 -page_block_size=0 -cache_batch_idx=0 -kname=$KNAME $COMMON_ARGS
      done
    done

    # no padding (equal), mixed Q/KV, all len=1
    for prec in fp16 bf16 ; do
      $EXE -prec=$prec -mode=0 -b=4 -h=8 -d=64 -s=128 -s_k=128 \
           -q_eff_lens=128,128,128,128 -kv_eff_lens=128,128,128,128 \
           -bias=n -p_drop=0.0 -lse=1 -kname=$KNAME $COMMON_ARGS

      $EXE -prec=$prec -mode=0 -b=4 -h=8 -d=64 -s=128 -s_k=128 \
           -q_eff_lens=10,20,30,40 -kv_eff_lens=40,30,20,10 \
           -bias=n -p_drop=0.0 -lse=1 -kname=$KNAME $COMMON_ARGS

      $EXE -prec=$prec -mode=0 -b=4 -h=8 -d=64 -s=128 -s_k=128 \
           -q_eff_lens=1,1,1,1 -kv_eff_lens=1,1,1,1 \
           -bias=n -p_drop=0.0 -lse=1 -kname=$KNAME $COMMON_ARGS
    done

    # highly variable logical lengths
    for prec in fp16 bf16 ; do
      $EXE -prec=$prec -mode=1 -b=4 -h=4 -d=32 \
           -s=1,127,3,65 -s_k=1,127,3,65 -s_kpad=128 \
           -bias=n -p_drop=0.0 -lse=1 -kname=$KNAME $COMMON_ARGS
    done

    # GQA + Alibi + Causal mask (keep vlayout row-major for fp16/bf16
    for prec in fp16 bf16 ; do
      $EXE -prec=$prec -mode=1 -b=2 -h=16 -h_k=4 -d=128 \
           -s=256,129 -s_k=256,129 -s_kpad=256 \
           -bias=a -mask=t -lse=1 -iperm=0 -operm=0 -vlayout=r \
           -kname=$KNAME $COMMON_ARGS
    done
}

set -x

run_fp16_bf16_tests
run_padding_smoke_tests
run_padding_basic_boundary_tests
run_fp8bf16_tests
run_fp8fp32_tests

if [ $TEST_APPENDKV -eq 1 ] ; then
    run_fp16_appendkv_tests
fi

set +x

new_fails_count=0
known_fails_count=0
if [ -f $KNOWN_FAILS_FILE ] ; then
    echo "Comparing current fails ($CURR_FAILS_FILE) against known fails ($KNOWN_FAILS_FILE):"
    while IFS= read -r line; do
        if grep -Fxq "$line" $KNOWN_FAILS_FILE; then
            echo "Known fail: $line"
            known_fails_count=$(($known_fails_count + 1))
        else
            echo "New fail: $line"
            new_fails_count=$(($new_fails_count + 1))
        fi
    done < $CURR_FAILS_FILE
else
    new_fails_count=$(wc -l < $CURR_FAILS_FILE)
    echo "No known fails file, all fails ($new_fails_count) are new:"
    cat $CURR_FAILS_FILE
fi
echo "New fails count: $new_fails_count; Known fails count: $known_fails_count"
exit $(($new_fails_count != 0))
