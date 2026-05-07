// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <tuple>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/utility/json_dump.hpp"

struct GemmConfigBase
{
    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;
    static constexpr bool kPadK = false;

    static constexpr bool TransposeC = false;

    static constexpr int kBlockPerCu                         = 1;
    static constexpr ck_tile::index_t TileParitionerGroupNum = 8;
    static constexpr ck_tile::index_t TileParitionerM01      = 4;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Intrawave;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;
    static constexpr bool Preshuffle = false; // currently preshuffle == true is not supported yet
    static constexpr bool Persistent = false; // currently persistent == true is not supported yet
    static constexpr bool DoubleSmemBuffer =
        false; // currently double smem buffer == true is not supported yet
};

struct GemmConfigMemory : public GemmConfigBase
{
    // Memory friendly for Interwave scheduler
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 32;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 8;

    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr bool Persistent                = true;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::MEMORY;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Interwave;
};

struct GemmConfigV3 : public GemmConfigBase
{
    // Compute friendly for Intrawave scheduler
    static constexpr ck_tile::index_t M_Tile = 256;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr bool Persistent                = true;
    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Intrawave;
};
struct GemmConfigV4 : public GemmConfigBase
{
    // Compute friendly for Intrawave scheduler
    // Using the ping pong reader in the lds level
    static constexpr ck_tile::index_t M_Tile = 256;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 32;

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr bool Persistent                = true;
    static constexpr bool DoubleSmemBuffer          = true;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V4;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Intrawave;
};

struct GemmConfigV3_Wmma : public GemmConfigBase
{
    // Compute friendly for Intrawave scheduler
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Intrawave;
};

template <ck_tile::GemmPipeline PipelineId>
struct PipelineTypeTraits;

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::MEMORY>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrMem<PipelineProblem>;
    template <typename PipelineProblem>
    using UniversalGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrMem<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::COMPUTE_V3>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<PipelineProblem>;
    template <typename PipelineProblem>
    using UniversalGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV3<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::COMPUTE_V4>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV4<PipelineProblem>;
    template <typename PipelineProblem>
    using UniversalGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV4<PipelineProblem>;
};

template <typename DataType>
struct GemmMultiDTypeConfig;

template <>
struct GemmMultiDTypeConfig<ck_tile::half_t>
{
    using ADataType   = ck_tile::half_t;
    using BDataType   = ck_tile::half_t;
    using D0DataType  = ck_tile::half_t;
    using D1DataType  = ck_tile::half_t;
    using EDataType   = ck_tile::half_t;
    using DsDataType  = ck_tile::tuple<D0DataType, D1DataType>;
    using AccDataType = float;
};

template <>
struct GemmMultiDTypeConfig<ck_tile::bf16_t>
{
    using ADataType   = ck_tile::bf16_t;
    using BDataType   = ck_tile::bf16_t;
    using D0DataType  = ck_tile::bf16_t;
    using D1DataType  = ck_tile::bf16_t;
    using EDataType   = ck_tile::bf16_t;
    using DsDataType  = ck_tile::tuple<D0DataType, D1DataType>;
    using AccDataType = float;
};

// Deduce the number of D tensors from the DsDataType tuple size
// All precision configs have the same number of D tensors, so we can use any one
constexpr std::size_t NumDTensor = GemmMultiDTypeConfig<ck_tile::bf16_t>::DsDataType::size();

using grouped_gemm_multi_d_kargs = ck_tile::GroupedGemmHostArgs<NumDTensor>;

std::pair<bool, ck_tile::ArgParser> create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("Ms", "", "M dimensions - empty by default.")
        .insert("Ns", "", "N dimensions - empty by default.")
        .insert("Ks", "", "K dimensions - empty by default.")
        .insert("stride_As", "", "Tensor A strides - it is empty by default.")
        .insert("stride_Bs", "", "Tensor B strides - it is empty by default.")
        .insert("stride_Ds", "", "Tensor Ds strides - it is empty by default.")
        .insert("stride_Es", "", "Tensor E strides - it is empty by default.")
        .insert("a_layout", "R", "A tensor data layout - Row by default.")
        .insert("b_layout", "C", "B tensor data layout - Row by default.")
        .insert("ds_layout", "R", "Ds tensor data layout - Row by default.")
        .insert("e_layout", "R", "E tensor data layout - Row by default.")
        .insert("validate", "1", "0. No validation, 1. Validation on CPU.")
        .insert("prec", "bf16", "data type. fp16/bf16")
        .insert("warmup", "10", "number of iterations before benchmark the kernel.")
        .insert("repeat", "100", "number of iterations to benchmark the kernel.")
        .insert("group_count", "8", "group count.")
        .insert("kbatch", "1", "kbatch for SplitK")
        .insert("json", "0", "0: No Json, 1: Dump Results in Json format")
        .insert("jsonfile", "grouped_gemm.json", "json file name to dump results");

    bool result = arg_parser.parse(argc, argv);
    return std::make_pair(result, arg_parser);
}

inline std::size_t get_workspace_size(const std::vector<grouped_gemm_multi_d_kargs>& gemm_descs)
{
    return gemm_descs.size() * sizeof(ck_tile::GemmTransKernelArg<NumDTensor>);
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
float grouped_gemm_multi_d(const std::vector<grouped_gemm_multi_d_kargs>& gemm_descs,
                           const ck_tile::stream_config& s,
                           void* kargs_ptr);
