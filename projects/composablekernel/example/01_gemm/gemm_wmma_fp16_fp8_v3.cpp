// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_wmma_cshuffle_v3.hpp"

using ADataType        = ck::half_t;
using BDataType        = ck::f8_t;
using AccDataType      = float;
using CShuffleDataType = ck::half_t;
using CDataType        = ck::half_t;

using ALayout = Row;
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
    AElementOp, BElementOp, CElementOp, GemmDefault,
    256,
    128, 128, 32,
    8, 8,
    16, 16,
    4, 2,
    S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>,
    2, 8, 8, 1,
    S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>,
    2, 8, 8, 1,
    1, 1, S<1, 32, 1, 8>, 8,
    ck::BlockGemmPipelineScheduler::Interwave, ck::BlockGemmPipelineVersion::v1>;
// clang-format on

using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                        BDataType,
                                                                        CDataType,
                                                                        AccDataType,
                                                                        PassThrough,
                                                                        PassThrough,
                                                                        PassThrough>;

#include "run_gemm_example_v2.inc"

int main(int argc, char* argv[]) { return !run_gemm_splitk_example(argc, argv); }
