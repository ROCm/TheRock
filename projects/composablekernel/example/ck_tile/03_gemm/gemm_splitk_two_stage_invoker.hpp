// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include "gemm_utils.hpp"
#include "ck_tile/ops/elementwise.hpp"

template <typename PrecType_, typename WorkspaceType_>
struct GemmConfigTwoStage : public GemmConfigComputeV3<PrecType_>
{
    using WorkspaceType = ck_tile::remove_cvref_t<WorkspaceType_>;
};

template <typename PrecType_, typename WorkspaceType_>
struct GemmConfigTwoStage_Wmma : public GemmConfigComputeV3_WMMA<PrecType_>
{
    using WorkspaceType = ck_tile::remove_cvref_t<WorkspaceType_>;
};

struct SplitKTwoStageInvoker
{
    template <typename GemmConfig,
              typename ADataType,
              typename BDataType,
              typename DsDataType,
              typename AccDataType,
              typename CDataType,
              typename ALayout,
              typename BLayout,
              typename DsLayout,
              typename ELayout,
              bool Persistent,
              typename CDEElementWise>
    static float gemm(const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s)

    {
        using GemmShape = ck_tile::TileGemmShape<
            ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
            ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
            ck_tile::
                sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile>,
            GemmConfig::PermuteA,
            GemmConfig::PermuteB>;

        using TilePartitioner =
            ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                       GemmConfig::TileParitionerGroupNum,
                                                       GemmConfig::TileParitionerM01>;

        using GemmUniversalTraits =
            ck_tile::TileGemmUniversalTraits<GemmConfig::kPadM,
                                             GemmConfig::kPadN,
                                             GemmConfig::kPadK,
                                             GemmConfig::DoubleSmemBuffer,
                                             ALayout,
                                             BLayout,
                                             ELayout,
                                             GemmConfig::TransposeC,
                                             GemmConfig::UseStructuredSparsity,
                                             Persistent,
                                             GemmConfig::NumWaveGroups,
                                             GemmConfig::Preshuffle>;
        constexpr auto scheduler = GemmConfig::Scheduler;

        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                           BDataType,
                                                                           AccDataType,
                                                                           GemmShape,
                                                                           GemmUniversalTraits,
                                                                           scheduler>;
        using WorkspaceType        = ck_tile::remove_cvref_t<typename GemmConfig::WorkspaceType>;

        using GemmPipeline = typename PipelineTypeTraits<
            GemmConfig::Pipeline>::template GemmPipeline<UniversalGemmProblem>;

        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             DsDataType,
                                             AccDataType,
                                             WorkspaceType,
                                             DsLayout,
                                             ELayout,
                                             CDEElementWise,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             GemmConfig::M_Warp,
                                             GemmConfig::N_Warp,
                                             GemmConfig::M_Warp_Tile,
                                             GemmConfig::N_Warp_Tile,
                                             GemmConfig::K_Warp_Tile,
                                             UniversalGemmProblem::TransposeC,
                                             GemmConfig::NumWaveGroups>>;

        using GemmKernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;

        ck_tile::DeviceMem ws_m_n_dev_buf(args.M * args.N * sizeof(WorkspaceType));
        ck_tile::GemmHostArgs ws_args = ck_tile::GemmHostArgs(args);
        auto c_ptr                    = ws_args.c_ptr;
        ws_args.c_ptr                 = ws_m_n_dev_buf.GetDeviceBuffer();
        auto gemm_kargs               = GemmKernel::MakeKernelArgs(ws_args);

        const dim3 grids  = Persistent ? GemmKernel::MaxOccupancyGridSize(s)
                                       : GemmKernel::GridSize(args.M, args.N, args.k_batch);
        const dim3 blocks = GemmKernel::BlockSize();

        if(!GemmKernel::IsSupportedArgument(gemm_kargs))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
        }

        using XElementwiseOperation = ck_tile::element_wise::UnaryConvert;
        using BlockTile             = ck_tile::sequence<2048>;
        using BlockWarps            = ck_tile::sequence<8>;
        using WarpTile              = ck_tile::sequence<64>;

        using ElementwiseShape =
            ck_tile::ElementWiseShape<BlockWarps, BlockTile, WarpTile, WorkspaceType>;
        using Problem = ck_tile::ElementWisePipelineProblem<WorkspaceType,
                                                            WorkspaceType,
                                                            CDataType,
                                                            ElementwiseShape,
                                                            XElementwiseOperation>;
        using ElementwiseKernel =
            ck_tile::ElementWiseKernel<Problem, ck_tile::ElementWiseDefaultPolicy>;

        ck_tile::index_t total_elements     = 1;
        std::vector<ck_tile::index_t> shape = {args.M, args.N};

        for(auto d : shape)
            total_elements *= d;

        const ck_tile::index_t kBlockSize      = ElementwiseKernel::BlockSize();
        constexpr ck_tile::index_t kBlockPerCu = 1;

        constexpr ck_tile::index_t elements_per_block = BlockTile::at(ck_tile::number<0>{});
        ck_tile::index_t kGridSize = (total_elements + elements_per_block - 1) / elements_per_block;

        auto input_tensors = ck_tile::make_tuple(static_cast<WorkspaceType*>(ws_args.c_ptr));
        auto input_size    = ck_tile::make_tuple(args.M, args.N);

        // Check if the kernel configuration is supported
        if(!ElementwiseKernel::IsSupportedArgument(input_size))
        {
            throw std::runtime_error(
                "Wrong! Elementwise arguments not supported! Skipping gemm!\n");
        }

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel with args: " << GemmKernel::GetName() << '\n'
                      << "shape: " << GemmShape::GetName() << '\n'
                      << "problem: " << UniversalGemmProblem::GetName() << '\n'
                      << "pipeline: " << GemmPipeline::GetName() << '\n'
                      << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                      << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                      << std::endl;
        }

        // Declare rotating_mem_ptr here so it stays in scope until it is needed
        std::unique_ptr<ck_tile::RotatingMemWrapper<ADataType, BDataType>> rotating_mem_ptr;
        std::function<void()> preprocess;

        auto clear_gemm_output = [&]() {
            if(args.k_batch > 1)
                hipGetErrorString(hipMemsetAsync(
                    ws_args.c_ptr, 0, args.M * args.N * sizeof(WorkspaceType), s.stream_id_));
        };

        if(s.flush_cache_)
        {
            std::cout << "Flushing cache..." << std::endl;

            ck_tile::HostTensor<ADataType> a_m(ck_tile::host_tensor_descriptor(
                args.M, args.K, args.stride_A, is_row_major(ALayout{})));
            ck_tile::HostTensor<BDataType> b_n(ck_tile::host_tensor_descriptor(
                args.K, args.N, args.stride_B, is_row_major(BLayout{})));

            auto size_a_buffer = a_m.get_element_space_size_in_bytes();
            auto size_b_buffer = b_n.get_element_space_size_in_bytes();

            rotating_mem_ptr = std::make_unique<ck_tile::RotatingMemWrapper<ADataType, BDataType>>(
                gemm_kargs.as_ptr[0],
                gemm_kargs.bs_ptr[0],
                s.rotating_count_,
                size_a_buffer,
                size_b_buffer);
            rotating_mem_ptr->Print();

            preprocess = [&]() {
                ck_tile::flush_icache();
                rotating_mem_ptr->Next();
                clear_gemm_output();
            };
        }
        else
        {
            preprocess = clear_gemm_output;
        }

        return ck_tile::launch_kernel_time_mask(
            s,
            preprocess,
            ck_tile::make_kernel<GemmConfig::kBlockPerCu>(
                GemmKernel{}, grids, blocks, 0, gemm_kargs),
            ck_tile::make_kernel<kBlockPerCu>(ElementwiseKernel{},
                                              kGridSize,
                                              kBlockSize,
                                              0,
                                              input_size,
                                              ck_tile::make_tuple(args.N, 1), // Input Stride
                                              ck_tile::make_tuple(args.N, 1), // Output Stride
                                              input_tensors,
                                              static_cast<CDataType*>(c_ptr)));
    }
};
