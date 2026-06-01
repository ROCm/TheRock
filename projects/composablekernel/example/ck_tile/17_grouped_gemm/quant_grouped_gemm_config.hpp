// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

template <typename DataType>
struct GemmTypeConfig;

template <>
struct GemmTypeConfig<ck_tile::fp8_t>
{
    using ADataType   = ck_tile::fp8_t;
    using BDataType   = ck_tile::fp8_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
};
template <>
struct GemmTypeConfig<ck_tile::bf8_t>
{
    using ADataType   = ck_tile::bf8_t;
    using BDataType   = ck_tile::bf8_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
};

template <bool Persistent_>
struct GemmConfigBase
{
    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;
    static constexpr bool kPadK = false;

    static constexpr bool PermuteA = false;
    static constexpr bool PermuteB = false;

    static constexpr bool TransposeC            = false;
    static constexpr bool UseStructuredSparsity = false;

    static constexpr int kBlockPerCu                         = 1;
    static constexpr ck_tile::index_t TileParitionerGroupNum = 8;
    static constexpr ck_tile::index_t TileParitionerM01      = 4;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Intrawave;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr bool PreshuffleB               = false;
    static constexpr bool Persistent                = Persistent_;
};

template <typename PrecType, bool Persistent>
struct GemmConfigComputeV3_2 : public GemmConfigBase<Persistent>
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile>();
};

template <typename PrecType, bool Persistent>
struct GemmConfigPreshuffleB_Bquant_prefill : public GemmConfigBase<Persistent>
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile, true>();

    static constexpr bool PreshuffleB      = true;
    static constexpr bool DoubleSmemBuffer = true;
};

template <ck_tile::QuantType QuantMode>
struct GemmQuantConfig;

template <>
struct GemmQuantConfig<ck_tile::QuantType::TensorQuant>
{
    template <typename PrecType, bool Persistent>
    using GemmConfig = GemmConfigComputeV3_2<PrecType, Persistent>;

    template <typename GemmProblem, bool PreshuffleB = false>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<GemmProblem>;

    template <typename GemmProblem, bool PreshuffleB = false>
    using BaseGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV3<GemmProblem>;
};

template <>
struct GemmQuantConfig<ck_tile::QuantType::RowColQuant>
{
    template <typename PrecType, bool Persistent>
    using GemmConfig = GemmConfigComputeV3_2<PrecType, Persistent>;

    template <typename GemmProblem, bool PreshuffleB = false>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<GemmProblem>;

    template <typename GemmProblem, bool PreshuffleB = false>
    using BaseGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV3<GemmProblem>;
};

template <>
struct GemmQuantConfig<ck_tile::QuantType::AQuantGrouped>
{
    template <typename PrecType, bool Persistent>
    using GemmConfig = GemmConfigComputeV3_2<PrecType, Persistent>;

    template <typename GemmProblem, bool PreshuffleB = false>
    using GemmPipeline = ck_tile::AQuantGemmPipelineAgBgCrCompV3<GemmProblem>;

    template <typename GemmProblem, bool PreshuffleB = false>
    using BaseGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV3<GemmProblem>;
};

template <>
struct GemmQuantConfig<ck_tile::QuantType::BQuantGrouped>
{
    template <typename PrecType, bool Persistent>
    using GemmConfig = GemmConfigPreshuffleB_Bquant_prefill<PrecType, Persistent>;

    template <typename GemmProblem, bool PreshuffleB = false>
    using GemmPipeline = std::conditional_t<PreshuffleB == true,
                                            ck_tile::WPQuantBPipelineAgBgCrV2<GemmProblem>,
                                            ck_tile::BQuantGemmPipelineAgBgCrCompV3<GemmProblem>>;

    template <typename GemmProblem, bool PreshuffleB = false>
    using BaseGemmPipeline =
        std::conditional_t<PreshuffleB == true,
                           ck_tile::BaseWeightPreshufflePipelineAGmemBGmemCRegV2<GemmProblem>,
                           ck_tile::BaseGemmPipelineAgBgCrCompV3<GemmProblem>>;
};

using grouped_gemm_kargs = ck_tile::QuantGroupedGemmHostArgs;

inline std::size_t get_workspace_size(const std::vector<grouped_gemm_kargs>& gemm_descs)
{
    return gemm_descs.size() * sizeof(ck_tile::QuantGemmTransKernelArg);
}
