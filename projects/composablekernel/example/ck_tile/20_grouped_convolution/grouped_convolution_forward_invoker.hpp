// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Regular grouped convolution invoker (no split-image)
// This invoker demonstrates regular convolution without split-image.
// It always uses Kernel<false> (split-image disabled).
// For large images that require split-image, use
// grouped_convolution_forward_split_image_invoker.hpp

#pragma once

#include "grouped_convolution_utils.hpp"

struct GroupedConvolutionForwardInvoker
{
    template <ck_tile::index_t NDimSpatial,
              typename ConvConfig,
              typename InDataType,
              typename WeiDataType,
              typename AccDataType,
              typename OutDataType,
              typename InLayout,
              typename WeiLayout,
              typename OutLayout,
              typename DsDataType    = ck_tile::tuple<>,
              typename DsLayout      = ck_tile::tuple<>,
              typename CDElementWise = ck_tile::element_wise::PassThrough>
    static float grouped_conv_fwd(const ck_tile::GroupedConvFwdHostArgs<CDElementWise>& args,
                                  const ck_tile::stream_config& s)
    {
        // Implicit GEMM Traits
        using GemmShape = ck_tile::TileGemmShape<
            ck_tile::sequence<ConvConfig::M_Tile, ConvConfig::N_Tile, ConvConfig::K_Tile>,
            ck_tile::sequence<ConvConfig::M_Warp, ConvConfig::N_Warp, ConvConfig::K_Warp>,
            ck_tile::sequence<ConvConfig::M_Warp_Tile,
                              ConvConfig::N_Warp_Tile,
                              ConvConfig::K_Warp_Tile>>;

        constexpr auto ConvSpec     = ck_tile::ConvolutionSpecialization::Default;
        using GroupedConvTraitsType = ck_tile::GroupedConvTraits<NDimSpatial,
                                                                 ConvSpec,
                                                                 InLayout,
                                                                 WeiLayout,
                                                                 DsLayout,
                                                                 OutLayout,
                                                                 ConvConfig::VectorSizeA,
                                                                 ConvConfig::VectorSizeB,
                                                                 ConvConfig::VectorSizeC,
                                                                 ConvConfig::NumGroupsToMerge>;

        using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<
            GemmShape,
            GroupedConvTraitsType::FixedGemmParams::TilePartitionerGroupNum,
            GroupedConvTraitsType::FixedGemmParams::TilePartitionerM01>;

        using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<
            GroupedConvTraitsType::FixedGemmParams::kPadM,
            GroupedConvTraitsType::FixedGemmParams::kPadN,
            GroupedConvTraitsType::FixedGemmParams::kPadK,
            ConvConfig::DoubleSmemBuffer,
            typename GroupedConvTraitsType::AsLayoutFwd,
            typename GroupedConvTraitsType::BsLayoutFwd,
            typename GroupedConvTraitsType::CLayoutFwd,
            GroupedConvTraitsType::FixedGemmParams::TransposeC,
            GroupedConvTraitsType::FixedGemmParams::UseStructuredSparsity,
            GroupedConvTraitsType::FixedGemmParams::Persistent,
            ConvConfig::NumWaveGroups>;
        constexpr auto scheduler = ConvConfig::Scheduler;

        // =====================================================================
        // Regular Convolution: Simple, no split-image
        // =====================================================================

        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<
            InDataType,
            WeiDataType,
            AccDataType,
            GemmShape,
            GemmUniversalTraits,
            scheduler,
            ck_tile::element_wise::PassThrough,
            ck_tile::element_wise::PassThrough,
            OutDataType,
            GroupedConvTraitsType::FixedGemmParams::FixedVectorSize,
            GroupedConvTraitsType::VectorSizeA,
            GroupedConvTraitsType::VectorSizeB>;

        using GemmPipeline = typename PipelineTypeTraits<
            ConvConfig::Pipeline>::template GemmPipeline<UniversalGemmProblem>;

        using ConvEpilogue = ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<
            InDataType,
            WeiDataType,
            DsDataType,
            AccDataType,
            OutDataType,
            typename GroupedConvTraitsType::ImplicitGemmDsLayout,
            typename GroupedConvTraitsType::FixedGemmParams::ELayout,
            CDElementWise,
            TilePartitioner::MPerBlock,
            TilePartitioner::NPerBlock,
            ConvConfig::M_Warp,
            ConvConfig::N_Warp,
            ConvConfig::M_Warp_Tile,
            ConvConfig::N_Warp_Tile,
            ConvConfig::K_Warp_Tile,
            GroupedConvTraitsType::FixedGemmParams::TransposeC,
            ConvConfig::NumWaveGroups,
            GroupedConvTraitsType::FixedGemmParams::FixedVectorSize,
            GroupedConvTraitsType::VectorSizeC>>;

        using Kernel = ck_tile::GroupedConvolutionForwardKernel<GroupedConvTraitsType,
                                                                TilePartitioner,
                                                                GemmPipeline,
                                                                ConvEpilogue>;
        auto kargs   = Kernel::MakeKernelArgs(args);

        const dim3 grids  = Kernel::GridSize(kargs);
        const dim3 blocks = Kernel::BlockSize();

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping conv!\n");
        }

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel with args: " << Kernel::GetName() << '\n'
                      << "shape: " << GemmShape::GetName() << '\n'
                      << "problem: " << UniversalGemmProblem::GetName() << '\n'
                      << "pipeline: " << GemmPipeline::GetName() << '\n'
                      << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                      << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                      << '\n'
                      << "Vector size A: " << GemmPipeline::GetVectorSizeA()
                      << ", Vector size B: " << GemmPipeline::GetVectorSizeB()
                      << ", Vector size C: " << ConvEpilogue::GetVectorSizeC() << std::endl;
        }

        return ck_tile::launch_kernel(
            s, ck_tile::make_kernel<ConvConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
    }
};
