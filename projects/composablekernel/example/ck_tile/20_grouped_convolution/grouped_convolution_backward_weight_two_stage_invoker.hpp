// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include "grouped_convolution_utils.hpp"

struct GroupedConvolutionBackwardWeightTwoStageInvoker
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
              typename DsDataType     = ck_tile::tuple<>,
              typename DsLayout       = ck_tile::tuple<>,
              typename CDEElementWise = ck_tile::element_wise::PassThrough>
    static InvokerResult grouped_conv_bwd_weight(const ck_tile::GroupedConvBwdWeightHostArgs& args,
                                                 const ck_tile::stream_config& s)
    {
        using WorkspaceDataType = float;
        // Force Vector Size C to 1 for two stage to check main
        // two stage use case
        constexpr ck_tile::index_t VectorSizeC = 1;

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
                                                                 VectorSizeC,
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
            typename GroupedConvTraitsType::AsLayoutBwdWeight,
            typename GroupedConvTraitsType::BsLayoutBwdWeight,
            typename GroupedConvTraitsType::CLayoutBwdWeight,
            GroupedConvTraitsType::FixedGemmParams::TransposeC,
            GroupedConvTraitsType::FixedGemmParams::UseStructuredSparsity,
            GroupedConvTraitsType::FixedGemmParams::Persistent,
            ConvConfig::NumWaveGroups>;

        constexpr auto scheduler = ConvConfig::Scheduler;

        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<
            OutDataType,
            InDataType,
            AccDataType,
            GemmShape,
            GemmUniversalTraits,
            scheduler,
            ck_tile::element_wise::PassThrough,
            ck_tile::element_wise::PassThrough,
            WeiDataType,
            GroupedConvTraitsType::FixedGemmParams::FixedVectorSize,
            GroupedConvTraitsType::VectorSizeA,
            GroupedConvTraitsType::VectorSizeB>;

        using GemmPipeline = typename PipelineTypeTraits<
            ConvConfig::Pipeline>::template GemmPipeline<UniversalGemmProblem>;

        using ConvEpilogue = ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<
            OutDataType, // A: Out
            InDataType,  // B: In
            DsDataType,
            AccDataType,
            WorkspaceDataType, // C: Workspace  normally Out
            typename GroupedConvTraitsType::ImplicitGemmDsLayout,
            typename GroupedConvTraitsType::FixedGemmParams::ELayout,
            CDEElementWise,
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

        using Kernel = ck_tile::GroupedConvolutionBackwardWeightKernel<GroupedConvTraitsType,
                                                                       TilePartitioner,
                                                                       GemmPipeline,
                                                                       ConvEpilogue>;

        const ck_tile::index_t spatial_lengths_accum =
            std::accumulate(args.filter_spatial_lengths_.begin(),
                            args.filter_spatial_lengths_.end(),
                            1,
                            std::multiplies<ck_tile::index_t>());
        ck_tile::DeviceMem ws_m_n_dev_buf(args.G_ * args.K_ * args.C_ * spatial_lengths_accum *
                                          sizeof(WorkspaceDataType));
        ck_tile::GroupedConvBwdWeightHostArgs ws_args = ck_tile::GroupedConvBwdWeightHostArgs(args);
        auto c_ptr                                    = ws_args.wei_ptr;
        ws_args.wei_ptr                               = ws_m_n_dev_buf.GetDeviceBuffer();

        const auto kargs  = Kernel::MakeKernelArgs(ws_args);
        const dim3 grids  = Kernel::GridSize(kargs);
        const dim3 blocks = Kernel::BlockSize();

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping conv!\n");
        }

        using XElementwiseOperation = ck_tile::element_wise::UnaryConvert;
        using BlockTile             = ck_tile::sequence<2048>;
        using BlockWarps            = ck_tile::sequence<8>;
        using WarpTile              = ck_tile::sequence<64>;

        using ElementwiseShape =
            ck_tile::ElementWiseShape<BlockWarps, BlockTile, WarpTile, WorkspaceDataType>;
        using Problem = ck_tile::ElementWisePipelineProblem<WorkspaceDataType,
                                                            WorkspaceDataType,
                                                            WeiDataType,
                                                            ElementwiseShape,
                                                            XElementwiseOperation>;
        using ElementwiseKernel =
            ck_tile::ElementWiseKernel<Problem, ck_tile::ElementWiseDefaultPolicy>;

        ck_tile::index_t total_elements     = 1;
        std::vector<ck_tile::index_t> shape = {
            static_cast<ck_tile::index_t>(args.G_ * args.K_),
            static_cast<ck_tile::index_t>(args.C_ * spatial_lengths_accum)};

        for(auto d : shape)
            total_elements *= d;

        const ck_tile::index_t kBlockSize = ElementwiseKernel::BlockSize();

        constexpr ck_tile::index_t elements_per_block = BlockTile::at(ck_tile::number<0>{});
        ck_tile::index_t kGridSize = (total_elements + elements_per_block - 1) / elements_per_block;

        auto input_tensors = ck_tile::make_tuple(static_cast<WorkspaceDataType*>(ws_args.wei_ptr));
        auto input_size    = ck_tile::make_tuple(shape[0], shape[1]);

        // Check if the kernel configuration is supported
        if(!ElementwiseKernel::IsSupportedArgument(input_size))
        {
            throw std::runtime_error(
                "Wrong! Elementwise arguments not supported! Skipping gemm!\n");
        }

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel with args: " << Kernel::GetName() << '\n'
                      << "shape: " << GemmShape::GetName() << '\n'
                      << "pipeline: " << GemmPipeline::GetName() << '\n'
                      << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                      << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                      << '\n'
                      << "Vector size A: " << GemmPipeline::GetVectorSizeA()
                      << ", Vector size B: " << GemmPipeline::GetVectorSizeB()
                      << ", Vector size C: " << ConvEpilogue::GetVectorSizeC() << std::endl;
        }

        auto preprocess = [&]() {
            if(args.k_batch > 1)
                ck_tile::hip_check_error(
                    hipMemsetAsync(ws_args.wei_ptr,
                                   0,
                                   shape[0] * shape[1] * sizeof(WorkspaceDataType),
                                   s.stream_id_));
        };

        float ave_time = ck_tile::launch_kernel_time_mask(
            s,
            preprocess,
            ck_tile::make_kernel<ConvConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs),
            ck_tile::make_kernel<ConvConfig::kBlockPerCu>(
                ElementwiseKernel{},
                kGridSize,
                kBlockSize,
                0,
                input_size,
                ck_tile::make_tuple(shape[1], 1), // Input Stride
                ck_tile::make_tuple(shape[1], 1), // Output Stride
                input_tensors,
                static_cast<WeiDataType*>(c_ptr)));
        return InvokerResult{ave_time, kargs.k_batch};
    }
};
