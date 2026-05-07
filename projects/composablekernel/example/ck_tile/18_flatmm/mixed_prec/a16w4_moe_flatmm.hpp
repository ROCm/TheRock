// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <tuple>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/moe_flatmm.hpp"

// GEMM config with 16x16 warp tile
struct A16W4_FlatmmConfig16
{
    static constexpr ck_tile::index_t M_Tile = 32;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 256;

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;

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

struct A16W4_FlatmmConfig16_950 : public A16W4_FlatmmConfig16
{
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr int kBlockPerCu         = 1;

    static constexpr int N_Repeat =
        N_Tile / A16W4_FlatmmConfig16::N_Warp_Tile / A16W4_FlatmmConfig16::N_Warp;
    static constexpr bool TiledMMAPermuteN = false;
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
                "gemm1_gate_up",
                "Gemm kind in FFN network [gemm1_gate_up | gemm2 | gemm1_split_k] - "
                "gemm1_gate_up by default.")
        .insert("validate", "1", "0. No validation, 1. Validation on CPU.")
        .insert("warmup", "50", "number of iterations before benchmark the kernel")
        .insert("mixed_prec",
                "bf16xfp4",
                "data type for activation and weight, support: bf16xfp4, fp16xfp4")
        .insert("init", "0", "0:random, 1:constant(1)")
        .insert("warp_tile",
                "0",
                "0: 16x16, 1: 16x16 (950 only, may use a larger tile than warp_tile=0)")
        .insert("repeat", "10", "number of iterations to benchmark the kernel.")
        .insert("k_batch", "1", "parallism to control splik-k.");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}
