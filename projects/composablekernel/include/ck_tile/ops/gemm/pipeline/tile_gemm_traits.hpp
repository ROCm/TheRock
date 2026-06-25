// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <bool kPadM_,
          bool kPadN_,
          bool kPadK_,
          typename AsLayout_,
          typename BsLayout_,
          typename CLayout_,
          index_t NumWaveGroups_ = 1>
struct TileGemmTraits
{
    static constexpr bool kPadM = kPadM_;
    static constexpr bool kPadN = kPadN_;
    static constexpr bool kPadK = kPadK_;

    // TODO this can't be hardcoded here! Should be in policy!
    static constexpr int _VectorSize = 16;

    using AsLayout = AsLayout_;
    using BsLayout = BsLayout_;
    using CLayout  = CLayout_;

    static constexpr bool TransposeC            = false;
    static constexpr bool UseStructuredSparsity = false;
    static constexpr index_t NumWaveGroups      = NumWaveGroups_;
};

template <bool kPadM_,
          bool kPadN_,
          bool kPadK_,
          bool DoubleSmemBuffer_,
          typename AsLayout_,
          typename BsLayout_,
          typename CLayout_,
          bool TransposeC_            = false,
          bool UseStructuredSparsity_ = false,
          bool UsePersistentKernel_   = false,
          index_t NumWaveGroups_      = 1,
          bool Preshuffle_            = false,
          int VectorSize_             = 16>
struct TileGemmUniversalTraits
{
    static constexpr bool kPadM            = kPadM_;
    static constexpr bool kPadN            = kPadN_;
    static constexpr bool kPadK            = kPadK_;
    static constexpr int _VectorSize       = VectorSize_;
    static constexpr bool DoubleSmemBuffer = DoubleSmemBuffer_;

    using AsLayout                   = AsLayout_;
    using BsLayout                   = BsLayout_;
    using CLayout                    = CLayout_;
    static constexpr bool TransposeC = TransposeC_;

    static constexpr bool UseStructuredSparsity = UseStructuredSparsity_;
    static constexpr bool UsePersistentKernel   = UsePersistentKernel_;
    static constexpr index_t NumWaveGroups      = NumWaveGroups_;
    static constexpr bool Preshuffle            = Preshuffle_;
};

template <bool kPadM_,
          bool kPadN_,
          bool kPadK_,
          bool DoubleSmemBuffer_,
          typename AsLayout_,
          typename BsLayout_,
          typename CLayout_,
          bool TransposeC_            = false,
          bool UseStructuredSparsity_ = false>
using PersistentTileGemmUniversalTraits = TileGemmUniversalTraits<kPadM_,
                                                                  kPadN_,
                                                                  kPadK_,
                                                                  DoubleSmemBuffer_,
                                                                  AsLayout_,
                                                                  BsLayout_,
                                                                  CLayout_,
                                                                  TransposeC_,
                                                                  UseStructuredSparsity_,
                                                                  true>;

} // namespace ck_tile
