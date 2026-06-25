// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
namespace ck_tile {
template <typename Arch, typename ADType, typename BDType, typename CDType>
struct WmmaTraitsBase;

// GFX11 specialization
template <typename ADType, typename BDType, typename CDType>
struct WmmaTraitsBase<gfx11_t, ADType, BDType, CDType>
{
    using ArchType = gfx11_t;

    using ADataType = ADType;
    using BDataType = BDType;
    using CDataType = CDType;

    using AVecType = ext_vector_t<ADataType, 16>;
    using BVecType = ext_vector_t<BDataType, 16>;
    using CVecType = ext_vector_t<CDataType, 8>;

    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 16;

    static constexpr index_t kAMBlock = 1;
    static constexpr index_t kBNBlock = 1;

    static constexpr index_t kRepeat      = 2;
    static constexpr index_t kAMLane      = 16;
    static constexpr index_t kBNLane      = 16;
    static constexpr index_t kABK0PerLane = 1;
    static constexpr index_t kABKLane     = 1;
    static constexpr index_t kABK1PerLane = 16;

    static constexpr index_t kCMLane     = 2;
    static constexpr index_t kCNLane     = 16;
    static constexpr index_t kCM0PerLane = 8;
    static constexpr index_t kCM1PerLane = 1;

    using kABPs2RHssMajor = sequence<0, 2, 1>;
    using kABPs2RHssMinor = sequence<0, 1, 0>;
    using kABYs2RHsMajor  = sequence<2, 2>;
    using kABYs2RHsMinor  = sequence<0, 2>;

    using kCPs2RHssMajor = sequence<1, 2>;
    using kCPs2RHssMinor = sequence<1, 0>;
    using kCYs2RHsMajor  = sequence<1, 1>;
    using kCYs2RHsMinor  = sequence<0, 2>;

    using kCTPs2RHssMajor = sequence<2, 1>;
    using kCTPs2RHssMinor = sequence<1, 0>;
    using kCTYs2RHsMajor  = sequence<2, 2>;
    using kCTYs2RHsMinor  = sequence<0, 2>;
};

// GFX12 specialization
template <typename ADType, typename BDType, typename CDType>
struct WmmaTraitsBase<gfx12_t, ADType, BDType, CDType>
{
    using ArchType = gfx12_t;

    using ADataType = ADType;
    using BDataType = BDType;
    using CDataType = CDType;

    using AVecType = ext_vector_t<ADataType, 8>;
    using BVecType = ext_vector_t<BDataType, 8>;
    using CVecType = ext_vector_t<CDataType, 8>;

    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 16;

    static constexpr index_t kAMBlock = 1;
    static constexpr index_t kBNBlock = 1;

    static constexpr index_t kRepeat      = 1;
    static constexpr index_t kAMLane      = 16;
    static constexpr index_t kBNLane      = 16;
    static constexpr index_t kABK0PerLane = 1;
    static constexpr index_t kABKLane     = 2;
    static constexpr index_t kABK1PerLane = 8;

    static constexpr index_t kCMLane     = 2;
    static constexpr index_t kCNLane     = 16;
    static constexpr index_t kCM0PerLane = 1;
    static constexpr index_t kCM1PerLane = 8;

    using kABPs2RHssMajor = sequence<2, 1>;
    using kABPs2RHssMinor = sequence<1, 0>;
    using kABYs2RHsMajor  = sequence<2, 2>;
    using kABYs2RHsMinor  = sequence<0, 2>;

    using kCPs2RHssMajor = sequence<1, 2>;
    using kCPs2RHssMinor = sequence<1, 0>;
    using kCYs2RHsMajor  = sequence<1, 1>;
    using kCYs2RHsMinor  = sequence<0, 2>;

    using kCTPs2RHssMajor = sequence<2, 1>;
    using kCTPs2RHssMinor = sequence<1, 0>;
    using kCTYs2RHsMajor  = sequence<2, 2>;
    using kCTYs2RHsMinor  = sequence<0, 2>;
};
} // namespace ck_tile
