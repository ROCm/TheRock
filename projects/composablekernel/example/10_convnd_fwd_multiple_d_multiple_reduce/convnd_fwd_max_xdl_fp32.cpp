// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "common.hpp"

using ADataType         = FP32;
using BDataType         = FP32;
using AccDataType       = FP32;
using CShuffleDataType  = FP32;
using DsDataType        = ck::Tuple<>;
using EDataType         = FP32;
using ReduceAccDataType = FP32;
using R0DataType        = FP32;
using RsDataType        = ck::Tuple<R0DataType>;

#include "run_convnd_fwd_max_example.inc"

int main(int argc, char* argv[])
{
    if(ck::is_gfx11_supported() || ck::is_gfx12_supported())
    {
        return 0;
    }
    return !run_convnd_fwd_max_example(argc, argv);
}
