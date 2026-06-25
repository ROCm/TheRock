// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <type_traits>

#include "ck_tile/host.hpp"
#include "mx_flatmm.hpp"

template <typename Layout>
using is_row_major_t = ck_tile::bool_constant<
    std::is_same_v<ck_tile::remove_cvref_t<Layout>, ck_tile::tensor_layout::gemm::RowMajor>>;

template <typename FlatmmConfig,
          typename ADataType,
          typename BDataType,
          typename DsDatatype,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout,
          typename ScaleM,
          typename ScaleN,
          bool persistent,
          typename CDEElementWise,
          bool Splitk,
          bool HasHotLoop,
          ck_tile::TailNumber TailNum>
float mx_flatmm_calc(const ck_tile::ScaleFlatmmHostArgs<ScaleM, ScaleN>& args,
                     const ck_tile::stream_config& s)
{
    using FlatmmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<FlatmmConfig::M_Tile, FlatmmConfig::N_Tile, FlatmmConfig::K_Tile>,
        ck_tile::sequence<FlatmmConfig::M_Warp, FlatmmConfig::N_Warp, FlatmmConfig::K_Warp>,
        ck_tile::sequence<FlatmmConfig::M_Warp_Tile,
                          FlatmmConfig::N_Warp_Tile,
                          FlatmmConfig::K_Warp_Tile>>;

    using MXGemmTraits = ck_tile::TileGemmUniversalTraits<FlatmmConfig::kPadM,
                                                          FlatmmConfig::kPadN,
                                                          FlatmmConfig::kPadK,
                                                          FlatmmConfig::DoubleSmemBuffer,
                                                          ALayout,
                                                          BLayout,
                                                          CLayout,
                                                          FlatmmConfig::TransposeC,
                                                          FlatmmConfig::UseStructuredSparsity,
                                                          persistent,
                                                          FlatmmConfig::NumWaveGroups,
                                                          true>;

    using ComputeDataType = ADataType;
    static_assert(sizeof(ComputeDataType) >= sizeof(BDataType),
                  "mixed_prec_flatmm requires ADataType is a wider type than BDataType");

    constexpr auto scheduler = FlatmmConfig::Scheduler;
    ck_tile::ignore          = Splitk;

    constexpr int BlockedXDLN_PerWarp = 2; // determined by scale shuffle pattern

    using MXPipelineProblem = ck_tile::MXFlatmmPipelineProblem<ADataType,
                                                               BDataType,
                                                               AccDataType,
                                                               FlatmmShape,
                                                               MXGemmTraits,
                                                               scheduler,
                                                               HasHotLoop,
                                                               TailNum>;

    using MXFlatmmPipeline = ck_tile::MXFlatmmPipelineAGmemBGmemCRegV1<MXPipelineProblem>;

    using TilePartitioner =
        ck_tile::GemmSpatiallyLocalTilePartitioner<FlatmmShape,
                                                   FlatmmConfig::TileParitionerGroupNum,
                                                   FlatmmConfig::TileParitionerM01>;
    using GemmEpilogue =
        ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<ComputeDataType,
                                                                   ComputeDataType,
                                                                   DsDatatype,
                                                                   AccDataType,
                                                                   CDataType,
                                                                   DsLayout,
                                                                   CLayout,
                                                                   CDEElementWise,
                                                                   TilePartitioner::MPerBlock,
                                                                   TilePartitioner::NPerBlock,
                                                                   FlatmmConfig::M_Warp,
                                                                   FlatmmConfig::N_Warp,
                                                                   FlatmmConfig::M_Warp_Tile,
                                                                   FlatmmConfig::N_Warp_Tile,
                                                                   FlatmmConfig::K_Warp_Tile,
                                                                   MXPipelineProblem::TransposeC,
                                                                   FlatmmConfig::NumWaveGroups,
                                                                   false, // FixedVectorSize
                                                                   1,     // VectorSizeC
                                                                   FlatmmConfig::TiledMMAPermuteN,
                                                                   BlockedXDLN_PerWarp>>;

    using Kernel = ck_tile::MXFlatmmKernel<TilePartitioner, MXFlatmmPipeline, GemmEpilogue>;

    auto kargs = Kernel::MakeKernelArgs(args);

    const dim3 grids      = Kernel::GridSize(kargs);
    constexpr dim3 blocks = Kernel::BlockSize();

    if(!Kernel::IsSupportedArgument(kargs))
        throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");

    if(s.log_level_ > 0)
    {
        std::cout << "Launching kernel with args:" << FlatmmShape::GetName() << "\n"
                  << "Shape: " << FlatmmShape::GetName() << "\n"
                  << "problem: " << MXPipelineProblem::GetName() << "\n"
                  << "pipeline: " << MXFlatmmPipeline::GetName() << "\n"
                  << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                  << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                  << std::endl;
    }

    // Declare rotating_mem_ptr here so it stays in scope until it is needed
    std::unique_ptr<ck_tile::RotatingMemWrapper<ADataType, BDataType>> rotating_mem_ptr;
    std::function<void()> preprocess;

    auto clear_gemm_output = [&]() {
        if(args.k_batch > 1)
            hipGetErrorString(
                hipMemsetAsync(args.e_ptr, 0, args.M * args.N * sizeof(CDataType), s.stream_id_));
    };

    if(s.flush_cache_)
    {
        std::cout << "Flushing cache..." << std::endl;
        constexpr ck_tile::index_t APackedSize = ck_tile::numeric_traits<ADataType>::PackedSize;
        constexpr ck_tile::index_t BPackedSize = ck_tile::numeric_traits<BDataType>::PackedSize;

        ck_tile::HostTensor<ADataType> a_m(ck_tile::host_tensor_descriptor(
            args.M, args.K, args.stride_A, is_row_major_t<ALayout>{}));
        ck_tile::HostTensor<BDataType> b_n(ck_tile::host_tensor_descriptor(
            args.K, args.N, args.stride_B, is_row_major_t<BLayout>{}));

        auto size_a_buffer = a_m.get_element_space_size_in_bytes() / APackedSize;
        auto size_b_buffer = b_n.get_element_space_size_in_bytes() / BPackedSize;

        rotating_mem_ptr = std::make_unique<ck_tile::RotatingMemWrapper<ADataType, BDataType>>(
            kargs.a_ptr, kargs.b_ptr, s.rotating_count_, size_a_buffer, size_b_buffer);
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
        ck_tile::make_kernel<FlatmmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
}
