#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT


EXE="$(find . -name tile_example_flatmm_basic -type f | head -n 1)"
KNAME=1

export CK_WARMUP=0
export CK_REPEAT=1

COMMON_ARGS='-v=2 -warmup=0 -repeat=1'

run_tests() {
    for m in 128 1024; do
        for n in 128 2048; do
            for k in 128 4096; do

                $EXE -m=$m -n=$n -k=$k -stride_a=0 -stride_b=0 -stride_c=0 -prec=$1 $COMMON_ARGS
                if [ $? -eq 0 ]; then
                    echo "Success: Test with m=$m, n=$n, k=$k executed successfully."
                else
                    echo "Error: Test with m=$m, n=$n, k=$k failed to execute properly."
                    # Optionally, exit or break if you need to halt further execution
                    # exit 1
                fi

            done
        done
    done
}

set -x

run_tests "bf16"
run_tests "fp16"

set +x
