// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck_tile/builder/reflect/instance_traits.hpp"
#include "ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_weight_xdl_cshuffle.hpp"
#include "ck_tile/builder/reflect/instance_traits_tile_grouped_convolution_backward_weight.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"

namespace {

TEST(InstanceTraits, BwdWeightXdlCShuffleInstanceStringReturnsCorrectFormat)
{
    using DeviceInstance = ck::tensor_operation::device::DeviceGroupedConvBwdWeight_Xdl_CShuffle<
        2,                                               // NDimSpatial
        ck::tensor_layout::convolution::GNHWC,           // InLayout
        ck::tensor_layout::convolution::GKYXC,           // WeiLayout
        ck::tensor_layout::convolution::GNHWK,           // OutLayout
        ck::half_t,                                      // InDataType
        ck::half_t,                                      // WeiDataType
        ck::half_t,                                      // OutDataType
        float,                                           // AccDataType
        ck::tensor_operation::element_wise::PassThrough, // InElementwiseOperation
        ck::tensor_operation::element_wise::PassThrough, // WeiElementwiseOperation
        ck::tensor_operation::element_wise::PassThrough, // OutElementwiseOperation
        ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::
            Default,            // ConvBackwardWeightSpecialization
        256,                    // BlockSize
        128,                    // MPerBlock
        128,                    // NPerBlock
        4,                      // K0PerBlock
        8,                      // K1
        32,                     // MPerXDL
        32,                     // NPerXDL
        2,                      // MXdlPerWave
        2,                      // NXdlPerWave
        ck::Sequence<4, 64, 1>, // ABlockTransferThreadClusterLengths_K0_M_K1
        ck::Sequence<1, 0, 2>,  // ABlockTransferThreadClusterArrangeOrder
        ck::Sequence<1, 0, 2>,  // ABlockTransferSrcAccessOrder
        2,                      // ABlockTransferSrcVectorDim
        8,                      // ABlockTransferSrcScalarPerVector
        8,                      // ABlockTransferDstScalarPerVector_K1
        false,                  // ABlockLdsAddExtraM
        ck::Sequence<4, 64, 1>, // BBlockTransferThreadClusterLengths_K0_N_K1
        ck::Sequence<1, 0, 2>,  // BBlockTransferThreadClusterArrangeOrder
        ck::Sequence<1, 0, 2>,  // BBlockTransferSrcAccessOrder
        2,                      // BBlockTransferSrcVectorDim
        8,                      // BBlockTransferSrcScalarPerVector
        8,                      // BBlockTransferDstScalarPerVector_K1
        false,                  // BBlockLdsAddExtraN
        1,                      // CShuffleMXdlPerWavePerShuffle
        1,                      // CShuffleNXdlPerWavePerShuffle
        ck::Sequence<1,
                     32,
                     1,
                     8>, // CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        8,               // CBlockTransferScalarPerVector_NWaveNPerXdl
        ck::half_t,      // ComputeTypeA
        ck::half_t,      // ComputeTypeB
        1,               // MaxTransposeTransferSrcScalarPerVector
        1>;              // MaxTransposeTransferDstScalarPerVector

    std::string instance_str = ck_tile::reflect::instance_string<DeviceInstance>();

    std::string expected_str = "DeviceGroupedConvBwdWeight_Xdl_CShuffle"
                               "<2"             // NDimSpatial
                               ",GNHWC"         // InLayout
                               ",GKYXC"         // WeiLayout
                               ",GNHWK"         // OutLayout
                               ",fp16"          // InDataType
                               ",fp16"          // WeiDataType
                               ",fp16"          // OutDataType
                               ",fp32"          // AccDataType
                               ",PassThrough"   // InElementwiseOperation
                               ",PassThrough"   // WeiElementwiseOperation
                               ",PassThrough"   // OutElementwiseOperation
                               ",Default"       // ConvBackwardWeightSpecialization
                               ",256"           // BlockSize
                               ",128"           // MPerBlock
                               ",128"           // NPerBlock
                               ",4"             // K0PerBlock
                               ",8"             // K1
                               ",32"            // MPerXDL
                               ",32"            // NPerXDL
                               ",2"             // MXdlPerWave
                               ",2"             // NXdlPerWave
                               ",Seq(4,64,1)"   // ABlockTransferThreadClusterLengths_K0_M_K1
                               ",Seq(1,0,2)"    // ABlockTransferThreadClusterArrangeOrder
                               ",Seq(1,0,2)"    // ABlockTransferSrcAccessOrder
                               ",2"             // ABlockTransferSrcVectorDim
                               ",8"             // ABlockTransferSrcScalarPerVector
                               ",8"             // ABlockTransferDstScalarPerVector_K1
                               ",false"         // ABlockLdsAddExtraM
                               ",Seq(4,64,1)"   // BBlockTransferThreadClusterLengths_K0_N_K1
                               ",Seq(1,0,2)"    // BBlockTransferThreadClusterArrangeOrder
                               ",Seq(1,0,2)"    // BBlockTransferSrcAccessOrder
                               ",2"             // BBlockTransferSrcVectorDim
                               ",8"             // BBlockTransferSrcScalarPerVector
                               ",8"             // BBlockTransferDstScalarPerVector_K1
                               ",false"         // BBlockLdsAddExtraN
                               ",1"             // CShuffleMXdlPerWavePerShuffle
                               ",1"             // CShuffleNXdlPerWavePerShuffle
                               ",Seq(1,32,1,8)" // CBlockTransferClusterLengths
                               ",8"             // CBlockTransferScalarPerVector_NWaveNPerXdl
                               ",fp16"          // ComputeTypeA
                               ",fp16"          // ComputeTypeB
                               ",1"             // MaxTransposeTransferSrcScalarPerVector
                               ",1>";           // MaxTransposeTransferDstScalarPerVector

    EXPECT_EQ(instance_str, expected_str);
}

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
        typename GroupedConvTraitsType::AsLayoutBwdWeight,
        typename GroupedConvTraitsType::BsLayoutBwdWeight,
        typename GroupedConvTraitsType::CLayoutBwdWeight,
        GroupedConvTraitsType::FixedGemmParams::TransposeC,
        GroupedConvTraitsType::FixedGemmParams::UseStructuredSparsity,
        GroupedConvTraitsType::FixedGemmParams::Persistent,
        1 /*NumWaveGroups*/>;

    using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<
        ck_tile::bf16_t /*OutDataType*/,
        ck_tile::bf16_t /*InDataType*/,
        float /*AccDataType*/,
        GemmShape,
        GemmUniversalTraits,
        ck_tile::GemmPipelineScheduler::Intrawave /*scheduler*/,
        ck_tile::element_wise::PassThrough /*AElementwiseOperation*/,
        ck_tile::element_wise::PassThrough /*BElementwiseOperation*/,
        ck_tile::bf16_t /*WeiDataType*/,
        GroupedConvTraitsType::FixedGemmParams::FixedVectorSize,
        GroupedConvTraitsType::VectorSizeA,
        GroupedConvTraitsType::VectorSizeB>;

    using GemmPipeline = typename ck_tile::GemmPipelineAgBgCrCompV3<UniversalGemmProblem>;

    using ConvEpilogue = ck_tile::CShuffleEpilogue<
        ck_tile::CShuffleEpilogueProblem<ck_tile::bf16_t /*OutDataType*/,
                                         ck_tile::bf16_t /*InDataType*/,
                                         ck_tile::tuple<> /*DsDataType*/,
                                         float /*AccDataType*/,
                                         ck_tile::bf16_t /*WeiDataType*/,
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

    using GroupedConvBwdWeiKernel =
        ck_tile::GroupedConvolutionBackwardWeightKernel<GroupedConvTraitsType,
                                                        TilePartitioner,
                                                        GemmPipeline,
                                                        ConvEpilogue>;

    std::string instance_str = ck_tile::reflect::instance_string<GroupedConvBwdWeiKernel>();

    std::string expected_str = "GroupedConvolutionBackwardWeightKernel"
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
