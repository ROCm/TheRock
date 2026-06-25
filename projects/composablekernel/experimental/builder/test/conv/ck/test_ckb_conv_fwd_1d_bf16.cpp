// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "utils/ckb_conv_test_configs.hpp"
#include "utils/ckb_conv_test_utils.hpp"
#include "utils/conv_algorithm_type_utils.hpp"

namespace {

using namespace ck_tile::builder::test_utils;

// 1D BF16 (channels-first) with Pipeline V2 and FILTER_1X1_STRIDE1_PAD0 specialization and SCALE
// elementwise op
TEST(FwdConvInstances,
     Create_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3_Instance_1D_BF16_ChannelsFirst_scale)
{
    using enum ck_tile::builder::ConvDirection;
    using enum ck_tile::builder::DataType;
    using enum ck_tile::builder::TensorLayout;
    using enum ck_tile::builder::ElementwiseOperation;

    constexpr ConvSignature FwdConvSignature{
        .spatial_dim            = 1,
        .direction              = FORWARD,
        .data_type              = BF16,
        .accumulation_data_type = FP32,
        .input                  = {.config = {.layout = NGCW}},
        .weight                 = {.config = {.layout = GKXC}},
        .output = {.config = {.layout = NGKW}, .operation = {.elementwise_operation = SCALE}}};

    constexpr auto FwdConvAlgorithm =
        ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3{}
            .with_thread_block(FwdThreadBlock_256_256x256x32)
            .with_gemm_config(FwdGemmParams_Xdl_4x4_per_wave)
            .with_transfer(FwdTransfer_4x64x1)
            .with_specializations(ConvFwdSpecialization::FILTER_1X1_STRIDE1_PAD0,
                                  GemmSpecialization::MNKPadding)
            .with_block_gemm(BlockGemmDesc_v2_intrawave);

    using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;

    const auto expected_transfer_parameters = to_string(FwdConvAlgorithm);
    run_test<Builder>({"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3",
                       expected_transfer_parameters,
                       "NGCW,GKXC,EmptyTuple,NGKW",
                       "PassThrough,PassThrough,Scale",
                       "Filter1x1Stride1Pad0",
                       "MNKPadding",
                       "Intrawave",
                       "v2"});
}

} // namespace
