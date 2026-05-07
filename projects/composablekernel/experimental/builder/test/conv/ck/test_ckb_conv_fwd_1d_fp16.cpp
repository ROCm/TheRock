// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "utils/ckb_conv_test_configs.hpp"
#include "utils/ckb_conv_test_utils.hpp"
#include "utils/conv_algorithm_type_utils.hpp"

namespace {

using namespace ck_tile::builder::test_utils;

// 1D FP16 (channels-last) with DEFAULT specialization
TEST(FwdConvInstances,
     Create_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_Instance_1D_FP16_ChannelsFirst)
{
    using enum ck_tile::builder::ConvDirection;
    using enum ck_tile::builder::DataType;
    using enum ck_tile::builder::TensorLayout;

    constexpr ConvSignature FwdConvSignature{.spatial_dim            = 1,
                                             .direction              = FORWARD,
                                             .data_type              = FP16,
                                             .accumulation_data_type = FP32,
                                             .input                  = {.config = {.layout = NWGC}},
                                             .weight                 = {.config = {.layout = GKXC}},
                                             .output = {.config = {.layout = NWGK}}};

    constexpr auto FwdConvAlgorithm =
        ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle{}
            .with_thread_block(FwdThreadBlock_64_64x32x32)
            .with_gemm_config(FwdGemmParams_Xdl_2x1_per_wave)
            .with_transfer(FwdTransfer_4x16x1)
            .with_specializations(ConvFwdSpecialization::DEFAULT, GemmSpecialization::MNKPadding)
            .with_prefetch_config(1, 2, PipelineScheduler::DEFAULT);

    using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;

    const auto expected_transfer_parameters = to_string(FwdConvAlgorithm);
    run_test<Builder>({"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle",
                       expected_transfer_parameters,
                       "NWGC,GKXC,EmptyTuple,NWGK",
                       "PassThrough,PassThrough,PassThrough",
                       "MNKPadding",
                       "Default"});
}

} // namespace
