// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include "gemm_utils.hpp"

struct BasicInvoker
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
              typename CLayout,
              bool Persistent,
              typename CDEElementWise>
    static float gemm(const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s)
    {
        if constexpr(Persistent)
        {
            std::cout << "WARNING: Ignoring persistent kernel option for basic gemm." << std::endl;
        }

        // This part comes from the Codegen
        constexpr ck_tile::index_t M_Tile = 256;
        constexpr ck_tile::index_t N_Tile = 256;
        constexpr ck_tile::index_t K_Tile = 64;

#if CK_TILE_USE_WMMA
        constexpr ck_tile::index_t M_Warp = 4;
        constexpr ck_tile::index_t N_Warp = 2;
        constexpr ck_tile::index_t K_Warp = 1;

        constexpr ck_tile::index_t M_Warp_Tile = 16;
        constexpr ck_tile::index_t N_Warp_Tile = 16;
        constexpr ck_tile::index_t K_Warp_Tile = 16;
#else
        constexpr ck_tile::index_t M_Warp = 2;
        constexpr ck_tile::index_t N_Warp = 2;
        constexpr ck_tile::index_t K_Warp = 1;

        constexpr ck_tile::index_t M_Warp_Tile = 32;
        constexpr ck_tile::index_t N_Warp_Tile = 32;
        constexpr ck_tile::index_t K_Warp_Tile = 16;
#endif

        using CodegenGemmShape =
            ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                                   ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                                   ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

        using TilePartitioner = ck_tile::GemmTile1DPartitioner<CodegenGemmShape>;

        using CodegenGemmTraits = ck_tile::TileGemmTraits<GemmConfig::kPadM,
                                                          GemmConfig::kPadN,
                                                          GemmConfig::kPadK,
                                                          ALayout,
                                                          BLayout,
                                                          CLayout>;

        using CodegenPipelineProblem = ck_tile::GemmPipelineProblem<ADataType,
                                                                    BDataType,
                                                                    AccDataType,
                                                                    CodegenGemmShape,
                                                                    CodegenGemmTraits>;

        using CodegenGemmPipeline = ck_tile::GemmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>;

        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             ck_tile::tuple<>,
                                             AccDataType,
                                             CDataType,
                                             ck_tile::tuple<>,
                                             CLayout,
                                             ck_tile::element_wise::PassThrough,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             M_Warp,
                                             N_Warp,
                                             M_Warp_Tile,
                                             N_Warp_Tile,
                                             K_Warp_Tile,
                                             CodegenPipelineProblem::TransposeC>>;

        // ToDo: Will add the codegen part to test different pipeline policies in GEMM.
        // Now we only use the BlockGemmASmemBSmemCRegV1DefaultPolicy.
        using Kernel = ck_tile::GemmKernel<TilePartitioner, CodegenGemmPipeline, GemmEpilogue>;
        auto kargs   = Kernel::MakeKernelArgs(args);

        const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch);
        const dim3 blocks = Kernel::BlockSize();

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
        }

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel with args: " << Kernel::GetName() << '\n'
                      << "shape: " << CodegenGemmShape::GetName() << '\n'
                      << "problem: " << CodegenPipelineProblem::GetName() << '\n'
                      << "pipeline: " << CodegenGemmPipeline::GetName() << '\n'
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
                    args.e_ptr, 0, args.M * args.N * sizeof(CDataType), s.stream_id_));
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
                kargs.as_ptr[0], kargs.bs_ptr[0], s.rotating_count_, size_a_buffer, size_b_buffer);
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
            ck_tile::make_kernel<GemmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
    }
};
