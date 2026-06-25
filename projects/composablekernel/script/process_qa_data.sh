#!/bin/bash 
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

#
# in order to run this script you'd need the following python packages:

#pip3 install --upgrade pip
#pip3 install sqlalchemy pymysql pandas sshtunnel

# you would also need to set up some environment variables in order to 
# post your new test results to the database and compare them to the baseline
# please contact Illia.Silin@amd.com for more details

#process results
python3 process_perf_data.py perf_gemm.log
python3 process_perf_data.py perf_resnet50_N256.log
python3 process_perf_data.py perf_resnet50_N4.log
python3 process_perf_data.py perf_batched_gemm.log
python3 process_perf_data.py perf_grouped_gemm.log
python3 process_perf_data.py perf_grouped_conv_fwd.log
python3 process_perf_data.py perf_grouped_conv_bwd_data.log
python3 process_perf_data.py perf_grouped_conv_bwd_weight.log
python3 process_perf_data.py perf_gemm_bilinear.log
python3 process_perf_data.py perf_reduction.log
python3 process_perf_data.py perf_splitK_gemm.log
python3 process_perf_data.py perf_onnx_gemm.log
python3 process_perf_data.py perf_mixed_gemm.log

file=./perf_onnx_gemm_gfx10.log
if [ -e "$file" ]; then
    python3 process_perf_data.py perf_onnx_gemm_gfx10.log
fi
file=./perf_onnx_gemm_gfx11.log
if [ -e "$file" ]; then
    python3 process_perf_data.py perf_onnx_gemm_gfx11.log
fi
file=./perf_onnx_gemm_gfx12.log
if [ -e "$file" ]; then
    python3 process_perf_data.py perf_onnx_gemm_gfx12.log
fi
file=./perf_fmha_fwd_gfx942.log
if [ -e "$file" ]; then
    python3 process_perf_data.py perf_fmha_fwd_gfx942.log
fi
file=./perf_fmha_bwd_gfx942.log
if [ -e "$file" ]; then
    python3 process_perf_data.py perf_fmha_bwd_gfx942.log
fi
file=./perf_fmha_fwd_gfx90a.log
if [ -e "$file" ]; then
    python3 process_perf_data.py perf_fmha_fwd_gfx90a.log
fi
file=./perf_fmha_bwd_gfx90a.log
if [ -e "$file" ]; then
    python3 process_perf_data.py perf_fmha_bwd_gfx90a.log
fi

for gpu in "gfx90a" "gfx942"; do
    for dtype in "fp16" "bf16" "fp8" "bf8"; do
        file=./perf_tile_gemm_mem_pipeline_${dtype}_${gpu}.log
        if [ -e "$file" ]; then
            python3 process_perf_data.py perf_tile_gemm_mem_pipeline_${dtype}_${gpu}.log
        fi
    done
done
