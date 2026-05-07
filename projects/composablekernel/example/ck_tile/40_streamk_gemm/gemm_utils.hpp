// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"

struct GemmConfigurationBase
{
    static constexpr bool PAD_M = true;
    static constexpr bool PAD_N = true;
    static constexpr bool PAD_K = true;

    static constexpr bool PERMUTE_A = false;
    static constexpr bool PERMUTE_B = false;

    static constexpr bool TRANSPOSE_C             = false;
    static constexpr bool USE_STRUCTURED_SPARSITY = false;

    static constexpr int BLOCK_PER_CU                 = 1;
    static constexpr auto SCHEDULER                   = ck_tile::GemmPipelineScheduler::Intrawave;
    static constexpr ck_tile::index_t NUM_WAVE_GROUPS = 1;
    static constexpr bool PRESHUFFLE                  = false;
    static constexpr bool DOUBLE_SMEM_BUFFER          = false;
};

template <typename PrecisionType, bool IsPersistent>
struct GemmConfigurationMemoryInterwave : public GemmConfigurationBase
{
    static constexpr ck_tile::index_t M_TILE = 256;
    static constexpr ck_tile::index_t N_TILE = 256;
    static constexpr ck_tile::index_t K_TILE = 16;

    static constexpr ck_tile::index_t M_WARP = 2;
    static constexpr ck_tile::index_t N_WARP = 2;
    static constexpr ck_tile::index_t K_WARP = 1;

    static constexpr ck_tile::index_t M_WARP_TILE = 32;
    static constexpr ck_tile::index_t N_WARP_TILE = 32;
    static constexpr ck_tile::index_t K_WARP_TILE = sizeof(PrecisionType) == 2 ? 8 : 16;

    static constexpr bool PERSISTENT = IsPersistent;
    static constexpr auto SCHEDULER  = ck_tile::GemmPipelineScheduler::Intrawave;
};

template <typename ADataType_, typename BDataType_ = ADataType_, typename CDataType_ = ADataType_>
struct StreamKGemmTypeConfiguration
{
    using ADataType   = ADataType_;
    using BDataType   = BDataType_;
    using AccDataType = float;
    using CDataType   = CDataType_;
};

auto createArgs(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "512", "m dimension")
        .insert("n", "512", "n dimension")
        .insert("k", "512", "k dimension")
        .insert("a_layout", "R", "A tensor data layout - Row by default")
        .insert("b_layout", "C", "B tensor data layout - Column by default")
        .insert("c_layout", "R", "C tensor data layout - Row by default")
        .insert("reduction_strategy",
                "atomic",
                "strategy for storing results in C tensor - atomic/reduction")
        .insert("persistent_dp",
                "0",
                "0. Non-persistent data-parallel section, 1 Fully persistent kernel.")
        .insert("stride_a", "0", "Tensor A stride")
        .insert("stride_b", "0", "Tensor B stride")
        .insert("stride_c", "0", "Tensor C stride")
        .insert("v", "2", "0. No validation, 1. Validation on CPU, 2. Validation on GPU")
        .insert("prec", "fp16", "data type. fp16/bf16/fp8/bf8")
        .insert("warmup", "50", "number of iterations before benchmarking the kernel")
        .insert("repeat", "100", "number of iterations to benchmark the kernel")
        .insert("timer", "gpu", "gpu:gpu timer, cpu:cpu timer")
        .insert("init", "0", "0:random, 1:linear, 2:constant(1)")
        .insert("flush_cache", "true", "flush cache before running the kernel, defaults to true");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}
