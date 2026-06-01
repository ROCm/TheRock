// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "common.hpp"

using DataType   = F16;
using BundleType = F64;

static_assert(sizeof(BundleType) % sizeof(DataType) == 0);

// clang-format off
using DevicePermuteInstance = ck::tensor_operation::device::DevicePermuteImpl
// ######| NumDim|     InData|     OutData| Elementwise| Block|  NPer|  HPer|  WPer|   InBlock|      InBlockTransfer|           InBlockTransfer|       Src|       Dst|             Src|             Dst|
// ######|       |       Type|        Type|   Operation|  Size| Block| Block| Block| LdsExtraW| ThreadClusterLengths| ThreadClusterArrangeOrder| VectorDim| VectorDim| ScalarPerVector| ScalarPerVector|
// ######|       |           |            |            |      |      |      |      |          |                     |                          |          |          |                |                |
// ######|       |           |            |            |      |      |      |      |          |                     |                          |          |          |                |                |
         <       3, BundleType, BundleType, PassThrough,   256,     1,    32,    32,         5,         S<1, 32,  8>,                S<0, 1, 2>,         2,         1,               4,               1>;
// clang-format on

#include "run_permute_bundle_example.inc"

int main(int argc, char* argv[])
{
    bool time_kernel = false;

    if(argc == 1)
    {
        // use default
    }
    else if(argc == 2)
    {
        time_kernel = std::stoi(argv[1]);
    }
    else
    {
        printf("arg1: time kernel (0=no, 1=yes, default=0)\n");
        exit(0);
    }

    return !run_permute_bundle_example({1, 80, 32000}, {0, 2, 1}, time_kernel);
}
