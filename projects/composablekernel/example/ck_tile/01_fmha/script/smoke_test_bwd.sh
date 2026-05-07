#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# TODO: run this script from CK root or build directory
set -euo pipefail

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
EXE_NAME=tile_example_fmha_bwd
EXE="$(find . -name $EXE_NAME -type f | head -n 1)"
KNAME=1
GPU_arch=${GPU_arch:-""}
if [ -z "$GPU_arch" ] ; then
    GPU_arch=$(rocminfo | grep -E 'Name:\s+gfx' | head -n1 | awk '{print $2}')
fi

export CK_WARMUP=0
export CK_REPEAT=1

CURR_FAILS_FILE=${CURR_FAILS_FILE:-"fmha_bwd_fails_$GPU_arch.txt"}
rm -f $CURR_FAILS_FILE
touch $CURR_FAILS_FILE
KNOWN_FAILS_FILE=${KNOWN_FAILS_FILE:-"$SCRIPT_DIR/fmha_bwd_known_fails_$GPU_arch.txt"}

COMMON_ARGS='-v=1'

run_exe() {
    set +ex
    $EXE $@
    local ret=$?
    if [ $ret -ne 0 ] ; then
        echo "$EXE_NAME $*" >> $CURR_FAILS_FILE
    fi
    set -ex
}

test_h_s_mask() {
    run_exe -b=1 -h=4 -h_k=2 -s=259                         $@
    run_exe -b=2 -h=2        -s=516 -s_k=253                $@
    run_exe -b=1 -h=4 -h_k=1 -s=500 -s_k=251 -mask=1        $@
    run_exe -b=1 -h=2        -s=900 -s_k=258 -mask=2        $@
    run_exe -b=2 -h=1        -s=987 -s_k=219 -mask=t:128,30 $@
    run_exe -b=2 -h=3 -h_k=1 -s=244 -s_k=499 -mask=b:4,35   $@
}

set -x
# main tests
for prec in "fp16" "bf16" ; do
for perm in 0 1 ; do
for hdim in 32 64 128 256 ; do
for mode in 0 1 ; do
for bias in "n" "a" ; do
for dbias in 0 ; do
for p_drop in 0.0 0.2 ; do
for deterministic in 0 ; do
test_h_s_mask -prec=$prec -d=$hdim -bias=$bias -dbias=$dbias -p_drop=$p_drop -iperm=$perm -operm=$perm -deterministic=$deterministic -v=1 -mode=$mode -kname=$KNAME $COMMON_ARGS
done
done
done
done
done
done
done
done

# additional cases
for hdim in 40 48 72 96 ; do
test_h_s_mask -prec=fp16 -d=$hdim -bias=a -dbias=0 -p_drop=0.2 -iperm=0 -operm=0 -deterministic=0 -v=1 -mode=1 -kname=$KNAME $COMMON_ARGS
test_h_s_mask -prec=bf16 -d=$hdim -bias=n -dbias=0 -p_drop=0   -iperm=1 -operm=1 -deterministic=0 -v=1 -mode=1 -kname=$KNAME $COMMON_ARGS
test_h_s_mask -prec=bf16 -d=$hdim -bias=a -dbias=0 -p_drop=0.2 -iperm=1 -operm=1 -deterministic=0 -v=1 -mode=1 -kname=$KNAME $COMMON_ARGS
done
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
