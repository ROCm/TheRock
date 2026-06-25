// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm_quant.hpp"

template <typename GemmConfig,
          typename ALayout,
          typename AQLayout,
          typename BLayout,
          typename BQLayout,
          typename CLayout,
          typename ADataType,
          typename AQDataType,
          typename BDataType,
          typename BQDataType,
          typename AccDataType,
          typename CDataType,
          typename QuantGroupSize,
          ck_tile::QuantType QuantMode = ck_tile::QuantType::BQuantGrouped>
float grouped_gemm(const std::vector<grouped_gemm_kargs>& gemm_descs,
                   const ck_tile::stream_config& s,
                   void* kargs_ptr)
{
    constexpr ck_tile::index_t TileParitionerGroupNum = 8;
    constexpr ck_tile::index_t TileParitionerM01      = 4;

    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
        ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
        ck_tile::
            sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile>>;
    using TilePartitioner = ck_tile::
        GemmSpatiallyLocalTilePartitioner<GemmShape, TileParitionerGroupNum, TileParitionerM01>;

    using Traits              = ck_tile::TileGemmTraits<GemmConfig::kPadM,
                                                        GemmConfig::kPadN,
                                                        GemmConfig::kPadK,
                                                        ALayout,
                                                        BLayout,
                                                        CLayout>;
    using GemmUniversalTraits = ck_tile::TileGemmQuantTraits<GemmConfig::kPadM,
                                                             GemmConfig::kPadN,
                                                             GemmConfig::kPadK,
                                                             false, // PreshuffleQuant
                                                             GemmConfig::PreshuffleB,
                                                             ALayout,
                                                             BLayout,
                                                             CLayout,
                                                             QuantMode,
                                                             AQLayout,
                                                             BQLayout,
                                                             GemmConfig::TransposeC,
                                                             GemmConfig::DoubleSmemBuffer,
                                                             GemmConfig::Persistent>;
    using GemmPipelineProblem =
        ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, GemmShape, Traits>;

    using BaseGemmPipeline =
        GemmQuantConfig<QuantMode>::template BaseGemmPipeline<GemmPipelineProblem,
                                                              GemmConfig::PreshuffleB>;

    const ck_tile::index_t k_grain = gemm_descs[0].k_batch * GemmConfig::K_Tile;
    const ck_tile::index_t K_split = (gemm_descs[0].K + k_grain - 1) / k_grain * GemmConfig::K_Tile;

    const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
    const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
    const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

    float ave_time{0};

    const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {
        constexpr bool has_hot_loop_v = has_hot_loop_.value;
        constexpr auto tail_number_v  = tail_number_.value;
        constexpr auto scheduler      = GemmConfig::Scheduler;

        constexpr bool UseGroupedQuant = QuantMode == ck_tile::QuantType::AQuantGrouped ||
                                         QuantMode == ck_tile::QuantType::BQuantGrouped;
        using QuantGemmProblem = std::conditional_t<
            UseGroupedQuant,
            std::conditional_t<QuantMode == ck_tile::QuantType::AQuantGrouped,
                               ck_tile::GemmAQuantPipelineProblem<ADataType,
                                                                  AQDataType,
                                                                  BDataType,
                                                                  AccDataType,
                                                                  GemmShape,
                                                                  GemmUniversalTraits,
                                                                  QuantGroupSize,
                                                                  GemmConfig::TransposeC,
                                                                  BDataType,
                                                                  scheduler,
                                                                  has_hot_loop_v,
                                                                  tail_number_v>,
                               ck_tile::GemmBQuantPipelineProblem<ADataType,
                                                                  BDataType,
                                                                  BQDataType,
                                                                  AccDataType,
                                                                  GemmShape,
                                                                  GemmUniversalTraits,
                                                                  QuantGroupSize,
                                                                  ADataType,
                                                                  scheduler,
                                                                  has_hot_loop_v,
                                                                  tail_number_v>>,
            ck_tile::GemmRowColTensorQuantPipelineProblem<ADataType,
                                                          BDataType,
                                                          AccDataType,
                                                          AccDataType,
                                                          GemmShape,
                                                          GemmUniversalTraits,
                                                          GemmConfig::TransposeC,
                                                          BDataType,
                                                          scheduler,
                                                          has_hot_loop_v,
                                                          tail_number_v>>;

        using GemmPipeline =
            GemmQuantConfig<QuantMode>::template GemmPipeline<QuantGemmProblem,
                                                              GemmConfig::PreshuffleB>;

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
                                             GemmConfig::M_Warp,
                                             GemmConfig::N_Warp,
                                             GemmConfig::M_Warp_Tile,
                                             GemmConfig::N_Warp_Tile,
                                             GemmConfig::K_Warp_Tile,
                                             QuantGemmProblem::TransposeC>>;

        using Kernel = ck_tile::QuantGroupedGemmKernel<TilePartitioner,
                                                       GemmPipeline,
                                                       GemmEpilogue,
                                                       GemmUniversalTraits::kQuantType>;
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
            std::cout << "Launching kernel: " << Kernel::GetName() << " with args:" << " grid: {"
                      << grids.x << ", " << grids.y << ", " << grids.z << "}" << ", blocks: {"
                      << blocks.x << ", " << blocks.y << ", " << blocks.z << "}" << std::endl;
        }

        return ave_time = ck_tile::launch_kernel(
                   s,
                   ck_tile::make_kernel<GemmConfig::kBlockPerCu>(
                       Kernel{},
                       grids,
                       blocks,
                       0,
                       ck_tile::cast_pointer_to_constant_address_space(kargs_ptr),
                       gemm_descs.size()));
    };

    return ave_time = BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
}

template <typename GemmConfig,
          typename ALayout,
          typename AQLayout,
          typename BLayout,
          typename BQLayout,
          typename CLayout,
          typename ADataType,
          typename AQDataType,
          typename BDataType,
          typename BQDataType,
          typename AccDataType,
          typename CDataType,
          typename QuantGroupSize,
          ck_tile::QuantType QuantMode = ck_tile::QuantType::BQuantGrouped>
float grouped_gemm_tileloop(const ck_tile::stream_config& s,
                            const ck_tile::index_t num_groups,
                            void* kargs_ptr)
{
    constexpr ck_tile::index_t TileParitionerGroupNum = 8;
    constexpr ck_tile::index_t TileParitionerM01      = 4;

    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
        ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
        ck_tile::
            sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile>>;
    using TilePartitioner = ck_tile::
        GemmSpatiallyLocalTilePartitioner<GemmShape, TileParitionerGroupNum, TileParitionerM01>;

    using GemmUniversalTraits = ck_tile::TileGemmQuantTraits<GemmConfig::kPadM,
                                                             GemmConfig::kPadN,
                                                             GemmConfig::kPadK,
                                                             false, // PreshuffleQuant
                                                             GemmConfig::PreshuffleB,
                                                             ALayout,
                                                             BLayout,
                                                             CLayout,
                                                             QuantMode,
                                                             AQLayout,
                                                             BQLayout,
                                                             GemmConfig::TransposeC,
                                                             GemmConfig::DoubleSmemBuffer,
                                                             GemmConfig::Persistent>;

    constexpr auto scheduler = GemmConfig::Scheduler;

    constexpr bool UseGroupedQuant = QuantMode == ck_tile::QuantType::AQuantGrouped ||
                                     QuantMode == ck_tile::QuantType::BQuantGrouped;

    using QuantGemmProblem = std::conditional_t<
        UseGroupedQuant,
        std::conditional_t<QuantMode == ck_tile::QuantType::AQuantGrouped,
                           ck_tile::GemmAQuantPipelineProblem<ADataType,
                                                              AQDataType,
                                                              BDataType,
                                                              AccDataType,
                                                              GemmShape,
                                                              GemmUniversalTraits,
                                                              QuantGroupSize,
                                                              GemmConfig::TransposeC>,
                           ck_tile::GemmBQuantPipelineProblem<ADataType,
                                                              BDataType,
                                                              BQDataType,
                                                              AccDataType,
                                                              GemmShape,
                                                              GemmUniversalTraits,
                                                              QuantGroupSize>>,
        ck_tile::GemmRowColTensorQuantPipelineProblem<ADataType,
                                                      BDataType,
                                                      AccDataType,
                                                      AccDataType,
                                                      GemmShape,
                                                      GemmUniversalTraits,
                                                      GemmConfig::TransposeC,
                                                      BDataType,
                                                      scheduler>>;

    using GemmPipeline = GemmQuantConfig<QuantMode>::template GemmPipeline<QuantGemmProblem,
                                                                           GemmConfig::PreshuffleB>;

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
                                         GemmConfig::M_Warp,
                                         GemmConfig::N_Warp,
                                         GemmConfig::M_Warp_Tile,
                                         GemmConfig::N_Warp_Tile,
                                         GemmConfig::K_Warp_Tile,
                                         QuantGemmProblem::TransposeC>>;
    using Kernel      = ck_tile::QuantGroupedGemmKernel<TilePartitioner,
                                                        GemmPipeline,
                                                        GemmEpilogue,
                                                        GemmUniversalTraits::kQuantType>;
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
