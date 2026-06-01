// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "utils/ckb_conv_tile_test_configs.hpp"
#include "utils/ckb_conv_test_utils.hpp"

namespace {

using namespace ck_tile::builder::test_utils;

TEST(FwdConvInstances, Create_ConvAlgorithm_Tile_GroupedConvolutionKernel_2D_FP16_NHWGC)
{
    constexpr ConvSignature FwdConvSignature{.spatial_dim = 2,
                                             .direction   = ConvDirection::BACKWARD_WEIGHT,
                                             .data_type   = DataType::FP16,
                                             .accumulation_data_type = DataType::FP32,
                                             .input  = {.config = {.layout = TensorLayout::NHWGC}},
                                             .weight = {.config = {.layout = TensorLayout::GKYXC}},
                                             .output = {.config = {.layout = TensorLayout::NHWGK}}};

    constexpr auto FwdConvAlgorithm =
        ConvAlgorithm_Tile_GroupedConvolutionKernel{}
            .with_tile_specializations(TileConvSpecialization::DEFAULT)
            .with_tile_thread_block(FwdTileThreadBlock_64x64x64)
            .with_tile_block_gemm(TileBlockGemmDesc_16x16_v3_intrawave)
            .with_tile_transfer(FwdTileTransfer_4x4x4)
            .with_tile_optimizations(TileOptimizations{
                .num_groups_to_merge = 1, .split_image = false, .explicit_gemm = false});

    using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;
    run_ck_tile_test<Builder>({
        "grouped_convolution_backward_weight",
        "fp16",
        "NHWGC_GKYXC_NHWGK",
        "64x64x64",
        "2x2",
        "16x16x16",
        //    "4x4x4", // TODO: Enable this check
        "Default",
        "Intrawave",
        "CShuffleEpilogue",
        "pipeline_AgBgCrCompV3",
        "DoubleSmemBuffer_0",
        "NumWaveGroups_1",
        "MergedGroups_1",
        "SplitImage_0",
        "ExplicitGemm_0",
    });
}

} // namespace
