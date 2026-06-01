// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/flatmm.hpp"
#include "ck_tile/ops/gemm.hpp"

// GEMM config with 32x132 warp tile
template <typename DataType>
struct FlatmmConfig32
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
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
    static constexpr ck_tile::index_t M_Tile = 128;
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
    static constexpr bool TiledMMAPermuteN = N_Repeat % 4 == 0;
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
    static constexpr bool TiledMMAPermuteN = N_Repeat % 4 == 0;
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

template <typename FlatmmConfig,
          typename ADataType,
          typename BDataType,
          typename DsDatatype,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ScaleM,
          typename ScaleN,
          bool persistent,
          typename CDEElementWise>
float flatmm_calc(const ck_tile::ScaleFlatmmHostArgs<ScaleM, ScaleN>& args,
                  const ck_tile::stream_config& s);
