// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "utils/ckb_conv_test_configs.hpp"
#include "utils/ckb_conv_test_utils.hpp"
#include "utils/conv_algorithm_type_utils.hpp"

namespace {

using namespace ck_tile::builder::test_utils;

// 1D I8 (channels-last) with and DEFAULT specialization
// (not supported on gfx11 and gfx12)
#if !defined(__gfx11__) && !defined(__gfx12__)
TEST(FwdConvInstances,
     Create_DeviceGroupedConvFwdMultipleD_Wmma_CShuffle_Instance_1D_FP32_ChannelsFirst_scale)
{
    using enum ck_tile::builder::ConvDirection;
    using enum ck_tile::builder::DataType;
    using enum ck_tile::builder::TensorLayout;

    constexpr ConvSignature FwdConvSignature{.spatial_dim            = 1,
                                             .direction              = FORWARD,
                                             .data_type              = I8,
                                             .accumulation_data_type = I32,
                                             .input                  = {.config = {.layout = GNWC}},
                                             .weight                 = {.config = {.layout = GKXC}},
                                             .output = {.config = {.layout = GNWK}}};

    constexpr auto FwdConvAlgorithm =
        ConvAlgorithm_DeviceGroupedConvFwdMultipleD_Wmma_CShuffle{}
            .with_thread_block(FwdThreadBlock_128_64x64x64)
            .with_gemm_config(FwdGemmParams_Wmma_2x1_per_wave)
            .with_transfer(FwdTransfer_4x32x1)
            .with_specializations(ConvFwdSpecialization::DEFAULT, GemmSpecialization::MNKPadding)
            .with_prefetch_config(1, 0, PipelineScheduler::DEFAULT);

    using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;

    const auto expected_transfer_parameters = to_string(FwdConvAlgorithm);
    run_test<Builder>({"DeviceGroupedConvFwdMultipleD_Wmma_CShuffle",
                       expected_transfer_parameters,
                       "GNWC,GKXC,EmptyTuple,GNWK",
                       "PassThrough,PassThrough,PassThrough",
                       "Default"});
}
#endif

} // namespace
