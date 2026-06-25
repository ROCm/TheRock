// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "common.hpp"

// kernel data types
using InKernelDataType       = FP32;
using WeiKernelDataType      = FP32;
using AccDataType            = FP32;
using CShuffleDataType       = FP32;
using BiasKernelDataType     = FP32;
using ResidualKernelDataType = FP32;
using OutKernelDataType      = FP32;

// tensor data types
using InUserDataType  = InKernelDataType;
using WeiUserDataType = WeiKernelDataType;
using OutUserDataType = OutKernelDataType;

using InElementOp  = PassThrough;
using WeiElementOp = PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::AddReluAdd;

#include "run_grouped_conv_fwd_bias_relu_add_example.inc"

int main(int argc, char* argv[])
{
    if(ck::is_gfx11_supported() || ck::is_gfx12_supported())
    {
        return 0;
    }
    return !run_grouped_conv_fwd_bias_relu_add_example(argc, argv);
}
