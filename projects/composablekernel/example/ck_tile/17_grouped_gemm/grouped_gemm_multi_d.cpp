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
#include "grouped_gemm_multi_d.hpp"

template <typename GemmConfig,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename AccDataType,
          typename EDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename CDEElementWise>
float grouped_gemm_multi_d(const std::vector<grouped_gemm_multi_d_kargs>& gemm_descs,
                           const ck_tile::stream_config& s,
                           void* kargs_ptr)
{

    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
        ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
        ck_tile::
            sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile>>;
    using TilePartitioner =
        ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                   GemmConfig::TileParitionerGroupNum,
                                                   GemmConfig::TileParitionerM01>;

    using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<GemmConfig::kPadM,
                                                                 GemmConfig::kPadN,
                                                                 GemmConfig::kPadK,
                                                                 GemmConfig::DoubleSmemBuffer,
                                                                 ALayout,
                                                                 BLayout,
                                                                 ELayout,
                                                                 GemmConfig::TransposeC>;
    constexpr auto scheduler  = GemmConfig::Scheduler;

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
                                         ELayout,
                                         CDEElementWise,
                                         TilePartitioner::MPerBlock,
                                         TilePartitioner::NPerBlock,
                                         GemmConfig::M_Warp,
                                         GemmConfig::N_Warp,
                                         GemmConfig::M_Warp_Tile,
                                         GemmConfig::N_Warp_Tile,
                                         GemmConfig::K_Warp_Tile,
                                         UniversalGemmProblem::TransposeC>>;

    using Kernel = ck_tile::GroupedGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
    auto kargs   = Kernel::MakeKargs(gemm_descs);
    if(!Kernel::IsSupportedArgument(kargs))
    {
        throw std::runtime_error("Kernel arguments not supported!");
    }

    const dim3 blocks = Kernel::BlockSize();
    const dim3 grids  = Kernel::GridSize(gemm_descs);

    HIP_CHECK_ERROR(hipMemcpyWithStream(kargs_ptr,
                                        kargs.data(),
                                        get_workspace_size(gemm_descs),
                                        hipMemcpyHostToDevice,
                                        s.stream_id_));

    if(s.log_level_ > 0)
    {
        std::cout << "Launching kernel: " << Kernel::GetName() << " with args:" << " grid: { "
                  << grids.x << ", " << grids.y << ", " << grids.z << "}" << ", blocks: {"
                  << blocks.x << ", " << blocks.y << ", " << blocks.z << "}" << std::endl;
    }

    return ck_tile::launch_kernel(s,
                                  ck_tile::make_kernel<GemmConfig::kBlockPerCu>(
                                      Kernel{},
                                      grids,
                                      blocks,
                                      0,
                                      ck_tile::cast_pointer_to_constant_address_space(kargs_ptr),
                                      gemm_descs.size()));
}

template <typename GemmConfig,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename AccDataType,
          typename EDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename CDEElementWise>
float grouped_gemm_multi_d_tileloop(const ck_tile::stream_config& s,
                                    const ck_tile::index_t num_groups,
                                    void* kargs_ptr)
{
    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
        ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
        ck_tile::
            sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile>>;
    using TilePartitioner =
        ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                   GemmConfig::TileParitionerGroupNum,
                                                   GemmConfig::TileParitionerM01>;
    using GemmUniversalTraits =
        ck_tile::PersistentTileGemmUniversalTraits<GemmConfig::kPadM,
                                                   GemmConfig::kPadN,
                                                   GemmConfig::kPadK,
                                                   GemmConfig::DoubleSmemBuffer,
                                                   ALayout,
                                                   BLayout,
                                                   ELayout>;

    constexpr auto scheduler = GemmConfig::Scheduler;

    // We create the GEMM pipeline without specifying hotloop or tailnumber.
    // These are automatically run inside the kernel based on the given input data.
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
                                         ELayout,
                                         CDEElementWise,
                                         TilePartitioner::MPerBlock,
                                         TilePartitioner::NPerBlock,
                                         GemmConfig::M_Warp,
                                         GemmConfig::N_Warp,
                                         GemmConfig::M_Warp_Tile,
                                         GemmConfig::N_Warp_Tile,
                                         GemmConfig::K_Warp_Tile,
                                         UniversalGemmProblem::TransposeC>>;
    using Kernel      = ck_tile::GroupedGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
    const dim3 blocks = Kernel::BlockSize();
    const dim3 grids  = Kernel::MaxOccupancyGridSize(s);

    if(s.log_level_ > 0)
    {
        std::cout << "Launching kernel: " << Kernel::GetName() << " with args:" << " grid: {"
                  << grids.x << ", " << grids.y << ", " << grids.z << "}" << ", blocks: {"
                  << blocks.x << ", " << blocks.y << ", " << blocks.z << "}" << std::endl;
    }

    return ck_tile::launch_kernel(s,
                                  ck_tile::make_kernel<GemmConfig::kBlockPerCu>(
                                      Kernel{},
                                      grids,
                                      blocks,
                                      0,
                                      ck_tile::cast_pointer_to_constant_address_space(kargs_ptr),
                                      num_groups));
}

#include "run_grouped_gemm_multi_d_example.inc"

int main(int argc, char* argv[])
{
#if CK_TILE_USE_WMMA
    return !run_grouped_gemm_multi_d_example<GemmConfigV3_Wmma>(argc, argv);
#else
    return !run_grouped_gemm_multi_d_example<GemmConfigV3>(argc, argv) ||
           !run_grouped_gemm_multi_d_example<GemmConfigMemory>(argc, argv) ||
           !run_grouped_gemm_multi_d_example<GemmConfigV4>(argc, argv);
#endif
}
