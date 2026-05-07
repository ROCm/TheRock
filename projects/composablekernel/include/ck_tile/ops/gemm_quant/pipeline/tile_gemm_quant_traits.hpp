// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include <cstdint>

namespace ck_tile {

enum struct QuantType : std::uint16_t
{
    AQuantGrouped  = 0,
    BQuantGrouped  = 1,
    RowColQuant    = 2,
    TensorQuant    = 3,
    ABQuantGrouped = 4
};

inline std::string quant_type_to_string(QuantType quant_type)
{
    switch(quant_type)
    {
    case QuantType::AQuantGrouped: return "AQuantGrouped";
    case QuantType::BQuantGrouped: return "BQuantGrouped";
    case QuantType::RowColQuant: return "RowColQuant";
    case QuantType::TensorQuant: return "TensorQuant";
    case QuantType::ABQuantGrouped: return "ABQuantGrouped";
    default: return "Unknown";
    }
}

template <bool kPadM_,
          bool kPadN_,
          bool kPadK_,
          bool PreshuffleQuant_,
          bool PreshuffleB_,
          typename ALayout_,
          typename BLayout_,
          typename CLayout_,
          QuantType QuantType_,
          typename AQLayout_        = ALayout_,
          typename BQLayout_        = BLayout_,
          bool TransposeC_          = false,
          bool DoubleSmemBuffer_    = false,
          bool UsePersistentKernel_ = false,
          int VectorSize_           = 16>
struct TileGemmQuantTraits
{
    static constexpr bool kPadM = kPadM_;
    static constexpr bool kPadN = kPadN_;
    static constexpr bool kPadK = kPadK_;

    static constexpr QuantType kQuantType = QuantType_;

    static constexpr int _VectorSize       = VectorSize_;
    static constexpr bool DoubleSmemBuffer = DoubleSmemBuffer_;

    using ALayout  = ALayout_;
    using BLayout  = BLayout_;
    using CLayout  = CLayout_;
    using AQLayout = AQLayout_;
    using BQLayout = BQLayout_;

    // TODO: It should be replaced to single value
    using AsLayout = ALayout_;
    using BsLayout = BLayout_;

    static constexpr bool TransposeC            = TransposeC_;
    static constexpr bool UseStructuredSparsity = false;
    static constexpr index_t NumWaveGroups      = 1;
    static constexpr bool UsePersistentKernel   = UsePersistentKernel_;

    static constexpr bool PreshuffleQuant = PreshuffleQuant_;
    static constexpr bool PreshuffleB     = PreshuffleB_;
};

} // namespace ck_tile
