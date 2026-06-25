// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "common_wmma.hpp"
#include "ck/host_utility/device_prop.hpp"

// kernel data types
using InKernelDataType       = FP16;
using WeiKernelDataType      = FP16;
using AccDataType            = FP32;
using CShuffleDataType       = FP16;
using BiasKernelDataType     = FP16;
using ResidualKernelDataType = FP16;
using OutKernelDataType      = FP16;

// tensor data types
using InUserDataType  = InKernelDataType;
using WeiUserDataType = WeiKernelDataType;
using OutUserDataType = OutKernelDataType;

using InElementOp  = PassThrough;
using WeiElementOp = PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::AddReluAdd;

#include "run_grouped_conv_fwd_bias_relu_add_wmma_example.inc"

int main(int argc, char* argv[])
{
    bool is_supported = ck::is_gfx11_supported();
    if(!is_supported)
    {
        std::cout << "WARNING: wmma example not supported on the platform " << ck::get_device_name()
                  << std::endl;
        return 0;
    }
    return !run_grouped_conv_fwd_bias_relu_add_example(argc, argv);
}
