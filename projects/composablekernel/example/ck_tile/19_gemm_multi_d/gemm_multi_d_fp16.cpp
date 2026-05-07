// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>
#include <memory>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/host.hpp"
#include "gemm_multi_d_fp16.hpp"
#include "utils.hpp"

template <typename GemmConfig,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename AccDataType,
          typename EDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout,
          typename CDEElementWise = ck_tile::element_wise::PassThrough>
auto gemm_multi_d(const gemm_multi_d_kargs& args, const ck_tile::stream_config& s) -> float
{
    constexpr ck_tile::index_t M_Tile = GemmConfig::M_Tile;
    constexpr ck_tile::index_t N_Tile = GemmConfig::N_Tile;
    constexpr ck_tile::index_t K_Tile = GemmConfig::K_Tile;

    constexpr ck_tile::index_t M_Warp = GemmConfig::M_Warp;
    constexpr ck_tile::index_t N_Warp = GemmConfig::N_Warp;
    constexpr ck_tile::index_t K_Warp = GemmConfig::K_Warp;

    constexpr ck_tile::index_t M_Warp_Tile = GemmConfig::M_Warp_Tile;
    constexpr ck_tile::index_t N_Warp_Tile = GemmConfig::N_Warp_Tile;
    constexpr ck_tile::index_t K_Warp_Tile = GemmConfig::K_Warp_Tile;

    constexpr bool DoubleSmemBuffer = GemmConfig::DoubleSmemBuffer;
    constexpr bool kPadM            = false;
    constexpr bool kPadN            = false;
    constexpr bool kPadK            = false;

    constexpr bool TransposeC = false;

    constexpr int kBlockPerCu                         = 1;
    constexpr ck_tile::index_t TileParitionerGroupNum = 8;
    constexpr ck_tile::index_t TileParitionerM01      = 4;

    using GemmShape =
        ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                               ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                               ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

    using TilePartitioner = ck_tile::
        GemmSpatiallyLocalTilePartitioner<GemmShape, TileParitionerGroupNum, TileParitionerM01>;

    using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<kPadM,
                                                                 kPadN,
                                                                 kPadK,
                                                                 DoubleSmemBuffer,
                                                                 ALayout,
                                                                 BLayout,
                                                                 CLayout,
                                                                 TransposeC>;

    constexpr auto scheduler = GemmConfig::Scheduler;

    using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                       BDataType,
                                                                       AccDataType,
                                                                       GemmShape,
                                                                       GemmUniversalTraits,
                                                                       scheduler>;

    using GemmPipeline = typename PipelineTypeTraits<GemmConfig::Pipeline>::template GemmPipeline<
        UniversalGemmProblem>;

    using GemmEpilogue = ck_tile::CShuffleEpilogue<
        ck_tile::CShuffleEpilogueProblem<ADataType,
                                         BDataType,
                                         DsDataType,
                                         AccDataType,
                                         EDataType,
                                         DsLayout,
                                         CLayout,
                                         CDEElementWise,
                                         TilePartitioner::MPerBlock,
                                         TilePartitioner::NPerBlock,
                                         M_Warp,
                                         N_Warp,
                                         M_Warp_Tile,
                                         N_Warp_Tile,
                                         K_Warp_Tile,
                                         UniversalGemmProblem::TransposeC>>;

    using Kernel = ck_tile::GemmKernelMultiD<TilePartitioner, GemmPipeline, GemmEpilogue>;
    auto kargs   = Kernel::MakeKernelArgs(args);

    const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch);
    const dim3 blocks = Kernel::BlockSize();

    if(!Kernel::IsSupportedArgument(kargs))
    {
        throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
    }

    if(s.log_level_ > 0)
    {
        std::cout << "Launching kernel with args:" << " grid: {" << grids.x << ", " << grids.y
                  << ", " << grids.z << "}" << ", blocks: {" << blocks.x << ", " << blocks.y << ", "
                  << blocks.z << "}" << std::endl;
    }

    return ck_tile::launch_kernel(
        s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
}

#include "run_gemm_multi_d_fp16_example.inc"

int main(int argc, char* argv[])
{
#if CK_TILE_USE_WMMA
    return !run_multiple_d_gemm_example<GemmConfigV3_Wmma>(argc, argv);
#else
    return !run_multiple_d_gemm_example<GemmConfigV3>(argc, argv);
#endif
}
