// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <tuple>
#include <stdexcept>
#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/host/check_err.hpp"
#include "ck_tile/host/reference/reference_gemm.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm_quant.hpp"

// Forward declarations for quant type-specific implementations
template <ck_tile::QuantType QT>
struct QuantTypeTraits;

template <typename TTuple, size_t Index, typename DefaultType, typename Enable = void>
struct SafeTupleElement
{
    using type = DefaultType;
};

template <typename TTuple, size_t Index, typename DefaultType>
struct SafeTupleElement<TTuple,
                        Index,
                        DefaultType,
                        std::enable_if_t<(Index < std::tuple_size_v<TTuple>)>>
{
    using type = std::tuple_element_t<Index, TTuple>;
};

template <typename TTuple, size_t Index, typename DefaultType>
using SafeTupleElement_t = typename SafeTupleElement<TTuple, Index, DefaultType>::type;

// Base class for common quant gemm functionality
template <typename Tuple, typename Derived>
class TestCkTileGemmQuantBase : public ::testing::Test
{
    protected:
    using ALayout                   = std::tuple_element_t<0, Tuple>;
    using BLayout                   = std::tuple_element_t<1, Tuple>;
    using CLayout                   = std::tuple_element_t<2, Tuple>;
    using AQLayout                  = std::tuple_element_t<3, Tuple>;
    using ADataType                 = std::tuple_element_t<4, Tuple>;
    using BDataType                 = std::tuple_element_t<5, Tuple>;
    using QDataType                 = std::tuple_element_t<6, Tuple>;
    using CDataType                 = std::tuple_element_t<7, Tuple>;
    static constexpr auto QuantType = std::tuple_element_t<8, Tuple>::value;
    using GemmConfig                = std::tuple_element_t<9, Tuple>;
    using QuantGroupSize            = std::tuple_element_t<10, Tuple>;
    using AQuantGroupSize           = QuantGroupSize;
    using BQuantGroupSize           = SafeTupleElement_t<Tuple, 11, QuantGroupSize>;
    using BQLayout                  = SafeTupleElement_t<Tuple, 12, AQLayout>;
    using AccDataType               = float; // accumulate always in float

    // Get the quant-type specific data types from traits
    using QuantTraits     = QuantTypeTraits<QuantType>;
    using ComputeDataType = typename QuantTraits::template ComputeDataType<ADataType, BDataType>;

    static constexpr ck_tile::index_t M_Tile = GemmConfig::M_Tile;
    static constexpr ck_tile::index_t N_Tile = GemmConfig::N_Tile;
    static constexpr ck_tile::index_t K_Tile = GemmConfig::K_Tile;

    static constexpr ck_tile::index_t M_Warp = GemmConfig::M_Warp;
    static constexpr ck_tile::index_t N_Warp = GemmConfig::N_Warp;
    static constexpr ck_tile::index_t K_Warp = GemmConfig::K_Warp;

    static constexpr ck_tile::index_t M_Warp_Tile = GemmConfig::M_Warp_Tile;
    static constexpr ck_tile::index_t N_Warp_Tile = GemmConfig::N_Warp_Tile;
    static constexpr ck_tile::index_t K_Warp_Tile = GemmConfig::K_Warp_Tile;
    static constexpr bool PreshuffleQuant         = GemmConfig::PreshuffleQuant;
    static constexpr bool PreshuffleB             = GemmConfig::PreshuffleB;
    static constexpr bool TiledMMAPermuteN        = GemmConfig::TiledMMAPermuteN;
    static constexpr bool DoubleSmemBuffer        = GemmConfig::DoubleSmemBuffer;

    static constexpr bool kPadM = GemmConfig::kPadM;
    static constexpr bool kPadN = GemmConfig::kPadN;
    static constexpr bool kPadK = GemmConfig::kPadK;

    public:
    void SetUp() override { static_cast<Derived*>(this)->SetUpQuantTypeSpecific(); }

    void TearDown() override { static_cast<Derived*>(this)->TearDownQuantTypeSpecific(); }

    // Common test execution logic
    void invoke_quant_gemm(const ck_tile::QuantGemmHostArgs& args, const ck_tile::stream_config& s)
    {
        // WP pipeline requires per-thread tile size aligned to Problem::VectorLoadSize.
        // static_assert((WG::kM * WG::kK * sizeof(ADataType) * MIterPerWarp / WaveSize) %
        // VectorLoadSize == 0). gfx9 cards match the requirements but it fails on gfx12. so we only
        // need to check the limitation on RDNA cards, i.e. assume wave size is 32.
        constexpr ck_tile::index_t WaveSize     = 32;
        constexpr ck_tile::index_t MIterPerWarp = M_Tile / (M_Warp * M_Warp_Tile);
        constexpr bool SupportVectorSize16 =
            (M_Warp_Tile * K_Warp_Tile * sizeof(ADataType) * MIterPerWarp / WaveSize) % 16 == 0;
        constexpr int VectorSize = PreshuffleB ? (SupportVectorSize16 ? 16 : 8) : 16;
        using CodegenGemmShape =
            ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                                   ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                                   ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

        using TilePartitioner = ck_tile::GemmTile1DPartitioner<CodegenGemmShape>;

        using CodegenGemmTraits = ck_tile::TileGemmQuantTraits<kPadM,
                                                               kPadN,
                                                               kPadK,
                                                               PreshuffleQuant,
                                                               PreshuffleB,
                                                               ALayout,
                                                               BLayout,
                                                               CLayout,
                                                               QuantType,
                                                               AQLayout,
                                                               BQLayout,
                                                               GemmConfig::TransposeC,
                                                               DoubleSmemBuffer,
                                                               false,
                                                               VectorSize>;

        // Let the derived class create the appropriate pipeline and epilogue
        static_cast<Derived*>(this)
            ->template run_quant_gemm_impl<CodegenGemmShape, TilePartitioner, CodegenGemmTraits>(
                args, s);
    }

    void RunTest(ck_tile::index_t M, ck_tile::index_t N, ck_tile::index_t K)
    {
        // Generate test data and run the kernel
        static_cast<Derived*>(this)->run_test_with_validation(M, N, K);
    }

    // Helper function to check layout
    template <typename Layout>
    static constexpr auto is_row_major(Layout)
    {
        return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(Layout{})>,
                                                     ck_tile::tensor_layout::gemm::RowMajor>>{};
    }

    // Tolerance calculation function for validation
    template <typename ADataType_, typename BDataType_, typename AccDataType_, typename CDataType_>
    auto calculate_rtol_atol(const ck_tile::index_t K,
                             const ck_tile::index_t kbatch,
                             const float max_accumulated_value)
    {
        using ComputeType = std::conditional_t<
            std::is_same_v<BDataType_, ck_tile::pk_fp4_raw_t>,
            ADataType_,
            std::conditional_t<sizeof(ADataType_) < sizeof(BDataType_), ADataType_, BDataType_>>;
        // Calculate thresholds
        const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType_, AccDataType_>(
            ck_tile::integer_divide_ceil(K, kbatch));
        const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType_, AccDataType_>(
            max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));
        // Calculate error due to split_k accumulation
        const auto rtol_split_k =
            ck_tile::get_relative_threshold<CDataType_, CDataType_, CDataType_>(kbatch);
        const auto atol_split_k =
            ck_tile::get_absolute_threshold<CDataType_, CDataType_, CDataType_>(
                max_accumulated_value, kbatch);
        // Use higher threshold
        return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
    }
};

// Define generic QuantTypeTraits template (will be specialized)
template <ck_tile::QuantType QT>
struct QuantTypeTraits
{
    static_assert(QT == ck_tile::QuantType::ABQuantGrouped ||
                      QT == ck_tile::QuantType::AQuantGrouped ||
                      QT == ck_tile::QuantType::BQuantGrouped ||
                      QT == ck_tile::QuantType::RowColQuant ||
                      QT == ck_tile::QuantType::TensorQuant,
                  "Unsupported quantization type");
};

// Specialization for AQuantGrouped
template <>
struct QuantTypeTraits<ck_tile::QuantType::AQuantGrouped>
{
    template <typename ADataType, typename BDataType>
    using ComputeDataType = BDataType; // For AQuant, compute type is BDataType

    static constexpr const char* name = "aquant";
};

// Specialization for BQuantGrouped
template <>
struct QuantTypeTraits<ck_tile::QuantType::BQuantGrouped>
{
    template <typename ADataType, typename BDataType>
    using ComputeDataType = ADataType; // For BQuant, compute type is ADataType

    static constexpr const char* name = "bquant";
};

// Specialization for ABQuantGrouped
template <>
struct QuantTypeTraits<ck_tile::QuantType::ABQuantGrouped>
{
    template <typename ADataType, typename BDataType>
    using ComputeDataType = BDataType; // For AQuant, compute type is BDataType

    static constexpr const char* name = "abquant";
};

// Specialization for RowColQuant
template <>
struct QuantTypeTraits<ck_tile::QuantType::RowColQuant>
{
    template <typename ADataType, typename BDataType>
    using ComputeDataType = ADataType; // For RowColQuant, compute type is ADataType

    static constexpr const char* name = "rowcol";
};

// Specialization for TensorQuant
template <>
struct QuantTypeTraits<ck_tile::QuantType::TensorQuant>
{
    template <typename ADataType, typename BDataType>
    using ComputeDataType = ADataType; // For TensorQuant, compute type is ADataType

    static constexpr const char* name = "tensor";
};
