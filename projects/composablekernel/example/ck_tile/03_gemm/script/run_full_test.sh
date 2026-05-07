#!/bin/bash 
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

#
# in order to run this script you'd first need to build the tile_example_gemm executables in ../build/bin/
#
# run the script as "./run_full_test.sh <tag for your test environment> <branch name> <host name> <gpu_arch>
# input arguments: 
# environment tag  : a string describing the specifics of your test environment
# branch name      : name of the branch in git repo (git status | grep -e 'On branch')
# host name        : $hostname
# gpu architecture: e.g., gfx90a, or gfx942, etc.

# get the command line arguments:
export env_type=$1
echo 'Environment type: ' $env_type
export branch=$2
echo 'Branch name: ' $branch
export host_name=$3
echo 'Host name: ' $host_name
export GPU_arch=$4
echo 'GPU_arch: ' $GPU_arch

function print_log_header(){
    rm -f $1;
    echo 'On branch ' $3 &> $1;
    echo 'Node name: ' $4 >> $1;
    # get GPU architecture and compute units from rocminfo
    echo -n "GPU_arch: " >> $1; rocminfo | grep "Name:" | grep "gfx" >> $1;
    rocminfo | grep "Compute Unit:" >> $1;
    hipcc --version | grep -e 'HIP version'  >> $1;
    echo 'Environment type: ' $2 >> $1;
    /opt/rocm/bin/amdclang++ --version | grep -e 'InstalledDir' >> $1;
}

# run verification tests
for dtype in fp16 bf16 fp8 bf8; do
    example/ck_tile/03_gemm/script/benchmark_basic_$dtype.sh
done
example/ck_tile/03_gemm/script/smoke_test_mem_pipeline.sh

# run performance benchmarks
for dtype in fp16 bf16 fp8 bf8; do
    export gemm_log="perf_tile_gemm_mem_pipeline_${dtype}_${GPU_arch}.log"
    print_log_header $gemm_log $env_type $branch $host_name
    example/ck_tile/03_gemm/script/benchmark_mem_pipeline_$dtype.sh 2>&1 | tee -a $gemm_log
done
