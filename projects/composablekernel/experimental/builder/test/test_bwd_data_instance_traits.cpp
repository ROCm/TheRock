// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "ck/ck.hpp"
#include "ck_tile/builder/reflect/instance_traits.hpp"
#include "ck_tile/builder/reflect/instance_traits_tile_grouped_convolution_backward_data.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"

namespace {

TEST(InstanceTraits, TileInstanceStringReturnsCorrectFormat)
{
    using GroupedConvTraitsType =
        ck_tile::GroupedConvTraits<2 /*NDimSpatial*/,
                                   ck_tile::ConvolutionSpecialization::Default /*ConvSpec*/,
                                   ck_tile::tensor_layout::convolution::NHWGC /*InLayout*/,
                                   ck_tile::tensor_layout::convolution::GKYXC /*WeiLayout*/,
                                   ck_tile::tuple<> /*DsLayout*/,
                                   ck_tile::tensor_layout::convolution::NHWGK /*OutLayout*/,
                                   4 /*VectorSizeA*/,
                                   4 /*VectorSizeB*/,
                                   4 /*VectorSizeC*/,
                                   1 /*NumGroupsToMerge*/,
                                   false /*EnableSplitImage*/,
                                   false /*ExplicitGemm*/>;

    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<128 /*M_Tile*/, 128 /*N_Tile*/, 32 /*K_Tile*/>,
        ck_tile::sequence<4 /*M_Warp*/, 1 /*N_Warp*/, 1 /*K_Warp*/>,
        ck_tile::sequence<16 /*M_Warp_Tile*/, 16 /*N_Warp_Tile*/, 16 /*K_Warp_Tile*/>>;

    using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<
        GemmShape,
        GroupedConvTraitsType::FixedGemmParams::TilePartitionerGroupNum,
        GroupedConvTraitsType::FixedGemmParams::TilePartitionerM01>;

    using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<
        GroupedConvTraitsType::FixedGemmParams::kPadM,
        GroupedConvTraitsType::FixedGemmParams::kPadN,
        GroupedConvTraitsType::FixedGemmParams::kPadK,
        false /*DoubleSmemBuffer*/,
        typename GroupedConvTraitsType::AsLayoutBwdData,
        typename GroupedConvTraitsType::BsLayoutBwdData,
        typename GroupedConvTraitsType::CLayoutBwdData,
        GroupedConvTraitsType::FixedGemmParams::TransposeC,
        GroupedConvTraitsType::FixedGemmParams::UseStructuredSparsity,
        GroupedConvTraitsType::FixedGemmParams::Persistent,
        1 /*NumWaveGroups*/>;

    using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<
        ck_tile::bf16_t /*OutDataType*/,
        ck_tile::bf16_t /*WeiDataType*/,
        float /*AccDataType*/,
        GemmShape,
        GemmUniversalTraits,
        ck_tile::GemmPipelineScheduler::Intrawave /*scheduler*/,
        ck_tile::element_wise::PassThrough /*AElementwiseOperation*/,
        ck_tile::element_wise::PassThrough /*BElementwiseOperation*/,
        ck_tile::bf16_t /*InDataType*/,
        GroupedConvTraitsType::FixedGemmParams::FixedVectorSize,
        GroupedConvTraitsType::VectorSizeA,
        GroupedConvTraitsType::VectorSizeB>;

    using GemmPipeline = typename ck_tile::GemmPipelineAgBgCrCompV3<UniversalGemmProblem>;

    using ConvEpilogue = ck_tile::CShuffleEpilogue<
        ck_tile::CShuffleEpilogueProblem<ck_tile::bf16_t /*OutDataType*/,
                                         ck_tile::bf16_t /*WeiDataType*/,
                                         ck_tile::tuple<> /*DsDataType*/,
                                         float /*AccDataType*/,
                                         ck_tile::bf16_t /*InDataType*/,
                                         typename GroupedConvTraitsType::ImplicitGemmDsLayout,
                                         typename GroupedConvTraitsType::FixedGemmParams::ELayout,
                                         ck_tile::element_wise::PassThrough /*CDElementWise*/,
                                         128 /*MPerBlock*/,
                                         128 /*NPerBlock*/,
                                         4 /*M_Warp*/,
                                         1 /*N_Warp*/,
                                         16 /*M_Warp_Tile*/,
                                         16 /*N_Warp_Tile*/,
                                         16 /*K_Warp_Tile*/,
                                         GroupedConvTraitsType::FixedGemmParams::TransposeC,
                                         1 /*kNumWaveGroups*/,
                                         GroupedConvTraitsType::FixedGemmParams::FixedVectorSize,
                                         GroupedConvTraitsType::VectorSizeC>>;

    using GroupedConvBwdDataKernel =
        ck_tile::GroupedConvolutionBackwardDataKernel<GroupedConvTraitsType,
                                                      TilePartitioner,
                                                      GemmPipeline,
                                                      ConvEpilogue>;

    std::string instance_str = ck_tile::reflect::instance_string<GroupedConvBwdDataKernel>();

    std::string expected_str = "GroupedConvolutionBackwardDataKernel"
                               "<2"           // NDimSpatial
                               ",Default"     // ConvSpecialization
                               ",NHWGC"       // InLayout
                               ",GKYXC"       // WeiLayout
                               ",EmptyTuple"  // DsLayout
                               ",NHWGK"       // OutLayout
                               ",4"           // VectorSizeA
                               ",4"           // VectorSizeB
                               ",4"           // VectorSizeC
                               ",1"           // NumGroupsToMerge
                               ",0"           // EnableSplitImage
                               ",0"           // ExplicitGemm
                               ",128"         // MPerBlock
                               ",128"         // NPerBlock
                               ",32"          // KPerBlock
                               ",4"           // MWarp
                               ",1"           // NWarp
                               ",1"           // KWarp
                               ",16"          // MWarpTile
                               ",16"          // NWarpTile
                               ",16"          // KWarpTile
                               ",bf16"        // ADataType
                               ",bf16"        // BDataType
                               ",COMPUTE_V3"  // BlkGemmPipelineVer
                               ",Intrawave"   // BlkGemmPipeSched
                               ",0"           // DoubleSmemBuffer
                               ",1"           // NumWaveGroups
                               ",fp32"        // AccDataType
                               ",bf16"        // EDataType
                               ",EmptyTuple"  // DsDataType
                               ",PassThrough" // CDEElementwiseOperation
                               ">";

    EXPECT_EQ(instance_str, expected_str);
}

} // anonymous namespace
