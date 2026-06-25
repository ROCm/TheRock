// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_wmma_cshuffle_v3.hpp"

using ADataType        = ck::f8_t;
using BDataType        = ck::f8_t;
using AccDataType      = float;
using CShuffleDataType = ck::bhalf_t;
using CDataType        = ck::bhalf_t;
using ComputeTypeA     = ck::f8_t;
using ComputeTypeB     = ck::f8_t;

using ALayout = Col;
using BLayout = Col;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

// clang-format off
using DeviceGemmV2Instance = ck::tensor_operation::device::DeviceGemm_Wmma_CShuffleV3<
    ALayout, BLayout, CLayout,
    ADataType, BDataType, CDataType, AccDataType, CShuffleDataType,
    PassThrough, PassThrough, PassThrough, GemmDefault,
    128,
    128, 64, 64,
    16, 16, // AK1, BK1
    16, 16,
    4, 2,
    S<4, 32, 1>, S<0, 2, 1>, S<0, 2, 1>,
    1, 4, 16, 0,
    S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>,
    2, 16, 16, 0,
    1, 1, S<1, 32, 1, 4>, 8,
    ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v1,
    ComputeTypeA, ComputeTypeB>;
// clang-format on

using ReferenceComputeType  = ck::f8_t;
using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                        BDataType,
                                                                        CDataType,
                                                                        AccDataType,
                                                                        AElementOp,
                                                                        BElementOp,
                                                                        CElementOp,
                                                                        ReferenceComputeType,
                                                                        ReferenceComputeType>;

#include "run_gemm_example_v2.inc"

int main(int argc, char* argv[])
{
    if(!ck::is_gfx12_supported())
    {
        std::cout << "This kernel support gfx12 only" << std::endl;

        return 0;
    }
    return !run_gemm_splitk_example(argc, argv);
}
