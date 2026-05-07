// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <tuple>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/utility/json_dump.hpp"

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

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile>();
};

template <ck_tile::QuantType QuantMode>
struct GemmQuantConfig;

// ABQuant specialization for GemmQuantConfig
template <>
struct GemmQuantConfig<ck_tile::QuantType::ABQuantGrouped>
{
    template <typename PrecType, bool Persistent>
    using GemmConfig = GemmConfigComputeV3_2<PrecType, Persistent>;

    template <typename GemmProblem, bool PreshuffleB = false>
    using GemmPipeline = ck_tile::ABQuantGemmPipelineAgBgCrCompV3<GemmProblem>;

    template <typename GemmProblem, bool PreshuffleB = false>
    using BaseGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV3<GemmProblem>;
};

using grouped_gemm_kargs = ck_tile::QuantGroupedGemmHostArgs;

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("Ms", "", "M dimensions - empty by default.")
        .insert("Ns", "", "N dimensions - empty by default.")
        .insert("Ks", "", "K dimensions - empty by default.")
        .insert(
            "stride_As",
            "",
            "Tensor A strides - it is empty by default.") // stride_As/stride_Bs/stride_Cs/stride_AQs/stride_BQs
                                                          // can be set to zero if
                                                          // Ms/Ns/Ks is not empty
        .insert("stride_Bs", "", "Tensor B strides - it is empty by default.")
        .insert("stride_Cs", "", "Tensor C strides - it is empty by default.")
        .insert("stride_AQs", "", "Tensor AQ strides - it is empty by default.")
        .insert("stride_BQs", "", "Tensor BQ strides - it is empty by default.")
        .insert("a_layout", "R", "A tensor data layout - Row by default.")
        .insert("b_layout", "C", "B tensor data layout - Row by default.")
        .insert("c_layout", "R", "C tensor data layout - Row by default.")
        .insert("validate", "1", "0. No validation, 1. Validation on CPU.")
        .insert("prec", "fp8", "data type. fp16/bf16/fp8/bf8")
        .insert("warmup", "10", "number of iterations before benchmark the kernel.")
        .insert("repeat", "100", "number of iterations to benchmark the kernel.")
        .insert("group_count", "8", "group count.")
        .insert("kbatch", "1", "kbatch for SplitK")
        .insert("init", "0", "0. Random, 2. One(s) (Constant)")
        .insert("persistent", "0", "Kernel persistency. 0: non-persistent. 1: persistent.")
        .insert("bquant_group_size", "1x1x128", "BQuant group size. 1x1x128 (default) or 1x128x128")
        .insert("json", "0", "0: No Json, 1: Dump Results in Json format")
        .insert("jsonfile", "abquant_grouped_gemm.json", "json file name to dump results");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

inline std::size_t get_workspace_size(const std::vector<grouped_gemm_kargs>& gemm_descs)
{
    return gemm_descs.size() * sizeof(ck_tile::QuantGemmTransKernelArg);
}

// Forward declaration of the non-persistent version
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
          typename AQuantGroupSize,
          typename BQuantGroupSize,
          ck_tile::QuantType QuantMode = ck_tile::QuantType::ABQuantGrouped>
float grouped_gemm_abquant(const std::vector<grouped_gemm_kargs>& gemm_descs,
                           const ck_tile::stream_config& s,
                           void* kargs_ptr);

// Forward declaration of the tileloop version for persistent kernels
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
          typename AQuantGroupSize,
          typename BQuantGroupSize,
          ck_tile::QuantType QuantMode = ck_tile::QuantType::ABQuantGrouped>
float grouped_gemm_tileloop(const ck_tile::stream_config& s,
                            const ck_tile::index_t num_groups,
                            void* kargs_ptr);
