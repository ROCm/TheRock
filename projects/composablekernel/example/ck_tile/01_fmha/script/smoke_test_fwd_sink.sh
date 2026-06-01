#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# TODO: run this script from CK root or build directory
#EXE="/code/composable_kernel/build/bin/tile_example_fmha_fwd"
set -euo pipefail

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
EXE_NAME=tile_example_fmha_fwd
EXE="$(find . -name $EXE_NAME -type f | head -n 1)"
KNAME=1
GPU_arch=$GPU_arch
if [ -z "$GPU_arch" ] ; then
    GPU_arch=$(rocminfo | grep -E 'Name:\s+gfx' | head -n1 | awk '{print $2}')
fi
set -x

COMMON_ARGS='-v=1 -warmup=0 -repeat=1'


$EXE -prec=fp16 -mode=0 -b=1 -h=1 -d=128 -d_v=128 -s=512 -s_k=512 -bias=n -lse=0 -iperm=0 -operm=0 -vlayout=r -num_splits=1 -page_block_size=128 -cache_batch_idx=0  -kname=1 -v=1 -warmup=0 -repeat=1 -mask=t:2,0,2

# window_size[2,0], sink_size = 2

#    x=1/y=3                 
#    1 * * * * * * *           1 * * * * * * *  
#    1 1 * * * * * *           1 1 * * * * * *
#    1 1 1 * * * * *   ---->   1 1 1 * * * * * 
#    * 1 1 1 * * * *           1 1 1 1 * * * * 
#    * * 1 1 1 * * *           1 1 1 1 1 * * * 
#    * * * 1 1 1 * *           1 1 * 1 1 1 * * 
#    * * * * 1 1 1 *           1 1 * * 1 1 1 *
#    * * * * * 1 1 1           1 1 * * * 1 1 1
#    l=2/r=0(tl)               l=2/r=0/s=2(tl)

$EXE -prec=fp16 -mode=0 -b=1 -h=1 -d=128 -d_v=128 -s=1024 -s_k=1024 -bias=n -lse=0 -iperm=0 -operm=0 -vlayout=r -num_splits=1 -page_block_size=128 -cache_batch_idx=0  -kname=1 -v=1 -warmup=0 -repeat=1 -mask=t:0,3,2 #-mask=b:3,0,2

#    x=4/y=1                   
#    1 1 1 1 * * * *           1 1 1 1 * * * * 
#    * 1 1 1 1 * * *           1 1 1 1 1 * * *
#    * * 1 1 1 1 * *   ---->   1 1 1 1 1 1 * *
#    * * * 1 1 1 1 *           1 1 * 1 1 1 1 *
#    * * * * 1 1 1 1           1 1 * * 1 1 1 1 
#    l=0/r=3(tl)               l=0/r=3/s=2(tl)
#    l=3/r=0(br)               l=3/r=0/s=2(br)  


$EXE -prec=fp16 -mode=0 -b=1 -h=1 -d=128 -d_v=128 -s=4096 -s_k=4096 -bias=n -lse=0 -iperm=0 -operm=0 -vlayout=r -num_splits=1 -page_block_size=128 -cache_batch_idx=0  -kname=1 -v=1 -warmup=0 -repeat=1 -mask=b:1,0,2

#    x=4/y=-1          
#    * * 1 1 * * * *           1 1 1 1 * * * * 
#    * * * 1 1 * * *           1 1 * 1 1 * * *
#    * * * * 1 1 * *   ---->   1 1 * * 1 1 * *
#    * * * * * 1 1 *           1 1 * * * 1 1 *
#    * * * * * * 1 1           1 1 * * * * 1 1 
#    l=1/r=0(br)               l=1/r=0/s=2(br)


$EXE -prec=fp16 -mode=1 -b=1 -h=1 -d=128 -d_v=128 -s=8192 -s_k=8192 -bias=n -lse=0 -iperm=0 -operm=0 -vlayout=r -num_splits=1 -page_block_size=128 -cache_batch_idx=0  -kname=1 -v=1 -warmup=0 -repeat=1 -mask=b:2,0,2

#    x=-1/y=5 
     
#    * * * * * *               * * * * * *
#    * * * * * *               * * * * * *
#    1 * * * * *               1 * * * * *
#    1 1 * * * *               1 1 * * * *
#    1 1 1 * * *       ---->   1 1 1 * * *
#    * 1 1 1 * *               1 1 1 1 * *
#    * * 1 1 1 *               1 1 1 1 1 *  
#    * * * 1 1 1               1 1 * 1 1 1
#    l=2/r=0(br)               l=2/r=0/s=2(br)


$EXE -prec=fp16 -mode=1 -b=1 -h=1 -d=128 -d_v=128 -s=16384 -s_k=16384 -bias=n -lse=0 -iperm=0 -operm=0 -vlayout=r -num_splits=1 -page_block_size=128 -cache_batch_idx=0  -kname=1 -v=1 -warmup=0 -repeat=1 -mask=b:-1,1,2
#      x=-1/y=8
#    * * * * *               * * * * *    
#    * * * * *               * * * * * 
#    1 * * * *      ---->    1 * * * * 
#    1 1 * * *               1 1 * * * 
#    1 1 1 * *               1 1 1 * * 
#    1 1 1 1 *               1 1 1 1 * 
#    1 1 1 1 1               1 1 1 1 1 
#    1 1 1 1 1               1 1 1 1 1 
#    l=2/r=0(br)             l=2/r=0/s=2(br)
     
