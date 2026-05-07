// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "convnd_fwd_activ_multi_ab_common.hpp"

using DataType    = float;
using AccDataType = float;
using InDataType  = DataType;
using WeiDataType = DataType;
using OutDataType = DataType;
using ADataTypes  = ck::Tuple<DataType, DataType>;
using BDataTypes  = ck::Tuple<DataType, DataType>;

using InElementOp  = ck::tensor_operation::element_wise::ScaleAdd;
using WeiElementOp = ck::tensor_operation::element_wise::ScaleAdd;

using DeviceGroupedConvNDActivInstance = DeviceGroupedConvNDMultiABFwdInstance<DataType,
                                                                               AccDataType,
                                                                               ADataTypes,
                                                                               BDataTypes,
                                                                               InElementOp,
                                                                               WeiElementOp>;

#include "../run_convnd_activ_example.inc"

int main(int argc, char* argv[])
{

    if(ck::is_gfx11_supported() || ck::is_gfx12_supported())
    {
        std::cout << "FP32 are not supported on gfx11 and gfx12" << std::endl;
        return 0;
    }

    return !run_convnd_example(argc, argv);
}
