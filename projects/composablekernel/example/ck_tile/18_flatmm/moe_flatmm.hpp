// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <tuple>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/moe_flatmm.hpp"

template <typename DataType>
struct FlatmmConfig32
{
    static constexpr ck_tile::index_t M_Tile = 64;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(DataType);

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = sizeof(DataType) == 2 ? 16 : 32;

    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;
    static constexpr bool kPadK = false;

    static constexpr bool TransposeC            = false;
    static constexpr bool UseStructuredSparsity = false;

    static constexpr int kBlockPerCu                = 1;
    static constexpr int TileParitionerGroupNum     = 8;
    static constexpr int TileParitionerM01          = 4;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Default;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr bool TiledMMAPermuteN = false; // disable PermuteN when NWarpTile != 16
};

template <typename DataType>
struct FlatmmConfig32_950 : public FlatmmConfig32<DataType>
{
    static constexpr ck_tile::index_t K_Warp_Tile = sizeof(DataType) == 2 ? 16 : 64;
};

// GEMM config with 16x16 warp tile
template <typename DataType>
struct FlatmmConfig16
{
    static constexpr ck_tile::index_t M_Tile = 64;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(DataType);

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = sizeof(DataType) == 2 ? 32 : 64;

    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;
    static constexpr bool kPadK = false;

    static constexpr bool TransposeC            = false;
    static constexpr bool UseStructuredSparsity = false;

    static constexpr int kBlockPerCu                = 1;
    static constexpr int TileParitionerGroupNum     = 8;
    static constexpr int TileParitionerM01          = 4;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Default;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
    static constexpr bool DoubleSmemBuffer          = false;

    static constexpr int N_Repeat          = N_Tile / N_Warp_Tile / N_Warp;
    static constexpr bool TiledMMAPermuteN = false;
};

template <typename DataType>
struct FlatmmConfig16_950 : public FlatmmConfig16<DataType>
{
    static constexpr ck_tile::index_t N_Tile      = 256;
    static constexpr ck_tile::index_t K_Tile      = 256 / sizeof(DataType);
    static constexpr ck_tile::index_t K_Warp_Tile = sizeof(DataType) == 2 ? 32 : 128;
    static constexpr int kBlockPerCu              = 1;

    static constexpr int N_Repeat =
        N_Tile / FlatmmConfig16<DataType>::N_Warp_Tile / FlatmmConfig16<DataType>::N_Warp;
    static constexpr bool TiledMMAPermuteN = false; // N_Repeat % 2 == 0;
};

template <typename ADataType>
struct GemmBasicTypeConfig;

template <>
struct GemmBasicTypeConfig<ck_tile::half_t>
{
    using ADataType   = ck_tile::half_t;
    using BDataType   = ck_tile::half_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
    // ToDo: Add more bias config to support different categories of GEMM.
};

template <>
struct GemmBasicTypeConfig<ck_tile::bf16_t>
{
    using ADataType   = ck_tile::bf16_t;
    using BDataType   = ck_tile::bf16_t;
    using AccDataType = float;
    using CDataType   = ck_tile::bf16_t;
};
template <>
struct GemmBasicTypeConfig<ck_tile::fp8_t>
{
    using ADataType   = ck_tile::fp8_t;
    using BDataType   = ck_tile::fp8_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
    // ToDo: Add more bias config to support different categories of GEMM.
};

template <>
struct GemmBasicTypeConfig<ck_tile::bf8_t>
{
    using ADataType   = ck_tile::bf8_t;
    using BDataType   = ck_tile::bf8_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
};

template <typename T>
struct is_8bit_type
    : std::bool_constant<std::is_same_v<T, ck_tile::fp8_t> || std::is_same_v<T, ck_tile::bf8_t>>
{
};

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("experts", "8", "Num of experts - 8 by default")
        .insert("NumTokens", "128", "M dimensions - 128 by default.")
        .insert("TopK", "3", "Top K - 3 by default.")
        .insert("N", "4096", "N dimensions - 4096 by default.")
        .insert("K", "4096", "K dimensions - 4096 by default.")
        .insert("stride_A", "", "Tensor A strides - it is empty by default.")
        .insert("stride_B", "", "Tensor B strides - it is empty by default.")
        .insert("stride_C", "", "Tensor C strides - it is empty by default.")
        .insert("a_layout", "R", "A tensor data layout - Row by default.")
        .insert("b_layout", "C", "B tensor data layout - Col by default.")
        .insert("c_layout", "R", "C tensor data layout - Row by default.")
        .insert("gemm_kind",
                "gemm1_gate_only",
                "Gemm kind in FFN network [gemm1_gate_only | gemm1_gate_up | gemm2] - "
                "gemm1_gate_only by default.")
        .insert("validate", "1", "0. No validation, 1. Validation on CPU.")
        .insert("warmup", "50", "number of iterations before benchmark the kernel")
        .insert("prec", "fp16", "data type. fp16/bf16/fp8/bf8")
        .insert(
            "warp_tile", "0", "0: 16x16, 1: 32x32, 2: 16x16x128 (950 only), 3: 32x32x64 (950 only)")
        .insert("repeat", "10", "number of iterations to benchmark the kernel.");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}
