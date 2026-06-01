// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "utils/ckb_conv_test_configs.hpp"
#include "utils/ckb_conv_test_utils.hpp"
#include "utils/conv_algorithm_type_utils.hpp"

namespace {

using namespace ck_tile::builder::test_utils;

// 2D FP8 NHWGC (channels-last) with Pipeline V1 and DEFAULT
TEST(FwdConvInstances,
     Create_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_Instance_2D_FP8_ChannelsLast)
{
    using enum ck_tile::builder::ConvDirection;
    using enum ck_tile::builder::DataType;
    using enum ck_tile::builder::TensorLayout;

    constexpr ConvSignature FwdConvSignature{.spatial_dim            = 2,
                                             .direction              = FORWARD,
                                             .data_type              = FP8,
                                             .accumulation_data_type = FP32,
                                             .input  = {.config = {.layout = NHWGC}},
                                             .weight = {.config = {.layout = GKYXC}},
                                             .output = {.config = {.layout = NHWGK}}};

    constexpr auto FwdConvAlgorithm =
        ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle{}
            .with_thread_block(FwdThreadBlock_256_256x128x32)
            .with_gemm_config(FwdGemmParams_Xdl_4x2_per_wave)
            .with_transfer(FwdTransfer_4x64x1_fp8)
            .with_specializations(ConvFwdSpecialization::DEFAULT, GemmSpecialization::MNKPadding)
            .with_prefetch_config(1, 1, PipelineScheduler::DEFAULT);

    using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;

    const auto expected_transfer_parameters = to_string(FwdConvAlgorithm);
    run_test<Builder>({"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle",
                       expected_transfer_parameters,
                       "Default",
                       "NHWGC,GKYXC,EmptyTuple,NHWGK",
                       "PassThrough,PassThrough,PassThrough",
                       "MNKPadding"});
}

} // namespace
