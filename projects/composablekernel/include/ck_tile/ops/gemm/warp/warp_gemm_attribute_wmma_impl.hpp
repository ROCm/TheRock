// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/config.hpp"

namespace ck_tile {

// Base traits for WMMA operations
template <typename Arch,
          typename AType,
          typename BType,
          typename CType,
          index_t M,
          index_t N,
          index_t K>
struct WmmaTraits;

// Generic WMMA implementation using traits
template <typename Traits>
struct WarpGemmAttributeWmmaImpl
{
    using TraitsType = Traits;
    using ADataType  = typename Traits::ADataType;
    using BDataType  = typename Traits::BDataType;
    using CDataType  = typename Traits::CDataType;

    using AVecType = typename Traits::AVecType;
    using BVecType = typename Traits::BVecType;
    using CVecType = typename Traits::CVecType;

    // Forward all static constants and type aliases
    static constexpr index_t kM = Traits::kM;
    static constexpr index_t kN = Traits::kN;
    static constexpr index_t kK = Traits::kK;

    static constexpr index_t kAMBlock = Traits::kAMBlock;
    static constexpr index_t kBNBlock = Traits::kBNBlock;

    static constexpr index_t kRepeat      = Traits::kRepeat;
    static constexpr index_t kAMLane      = Traits::kAMLane;
    static constexpr index_t kBNLane      = Traits::kBNLane;
    static constexpr index_t kABK0PerLane = Traits::kABK0PerLane;
    static constexpr index_t kABKLane     = Traits::kABKLane;
    static constexpr index_t kABK1PerLane = Traits::kABK1PerLane;

    static constexpr index_t kCMLane     = Traits::kCMLane;
    static constexpr index_t kCNLane     = Traits::kCNLane;
    static constexpr index_t kCM0PerLane = Traits::kCM0PerLane;
    static constexpr index_t kCM1PerLane = Traits::kCM1PerLane;

    using kABPs2RHssMajor = typename Traits::kABPs2RHssMajor;
    using kABPs2RHssMinor = typename Traits::kABPs2RHssMinor;
    using kABYs2RHsMajor  = typename Traits::kABYs2RHsMajor;
    using kABYs2RHsMinor  = typename Traits::kABYs2RHsMinor;

    using kCPs2RHssMajor = typename Traits::kCPs2RHssMajor;
    using kCPs2RHssMinor = typename Traits::kCPs2RHssMinor;
    using kCYs2RHsMajor  = typename Traits::kCYs2RHsMajor;
    using kCYs2RHsMinor  = typename Traits::kCYs2RHsMinor;

    using kCTPs2RHssMajor = typename Traits::kCTPs2RHssMajor;
    using kCTPs2RHssMinor = typename Traits::kCTPs2RHssMinor;
    using kCTYs2RHsMajor  = typename Traits::kCTYs2RHsMajor;
    using kCTYs2RHsMinor  = typename Traits::kCTYs2RHsMinor;

    // c_vec += a_vec * b_vec
    template <bool clamp = false, bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        c_vec = Traits::template wmma_intrinsic<clamp>(a_vec, b_vec, c_vec);
    }

    // c_vec = a_vec * b_vec
    template <bool clamp = false>
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        return bit_cast<CVecType>(
            Traits::template wmma_intrinsic<clamp>(a_vec, b_vec, CVecType{0.f}));
    }
};

using DeviceIp = remove_cvref_t<decltype(ck_tile::get_device_arch())>;
using WarpGemmAttributeWmmaImpl_f32_16x16x16_f16_f16 =
    WarpGemmAttributeWmmaImpl<WmmaTraits<DeviceIp, fp16_t, fp16_t, float, 16, 16, 16>>;

using WarpGemmAttributeWmmaImpl_f32_16x16x16_bf16_bf16 =
    WarpGemmAttributeWmmaImpl<WmmaTraits<DeviceIp, bf16_t, bf16_t, float, 16, 16, 16>>;

using WarpGemmAttributeWmmaImpl_i32_16x16x16_i8_i8 =
    WarpGemmAttributeWmmaImpl<WmmaTraits<DeviceIp, int8_t, int8_t, int32_t, 16, 16, 16>>;

using WarpGemmAttributeWmmaImpl_f32_16x16x16_f8_f8 =
    WarpGemmAttributeWmmaImpl<WmmaTraits<gfx12_t, fp8_t, fp8_t, float, 16, 16, 16>>;

using WarpGemmAttributeWmmaImpl_f32_16x16x16_bf8_bf8 =
    WarpGemmAttributeWmmaImpl<WmmaTraits<gfx12_t, bf8_t, bf8_t, float, 16, 16, 16>>;

using WarpGemmAttributeWmmaImpl_f32_16x16x16_f8_bf8 =
    WarpGemmAttributeWmmaImpl<WmmaTraits<gfx12_t, fp8_t, bf8_t, float, 16, 16, 16>>;

using WarpGemmAttributeWmmaImpl_f32_16x16x16_bf8_f8 =
    WarpGemmAttributeWmmaImpl<WmmaTraits<gfx12_t, bf8_t, fp8_t, float, 16, 16, 16>>;

template <typename Arch,
          typename AType,
          typename BType,
          typename CType,
          index_t warp_m,
          index_t warp_n,
          index_t warp_k>
struct has_wmma_traits
{
    template <typename T>
    static auto
    test(int) -> decltype(std::declval<
                              typename WmmaTraits<T, AType, BType, CType, warp_m, warp_n, warp_k>::
                                  ADataType>(),
                          std::true_type{});

    template <typename>
    static std::false_type test(...);

    static constexpr bool value = decltype(test<Arch>(0))::value;
};

template <typename Arch,
          typename AType,
          typename BType,
          typename CType,
          index_t warp_m,
          index_t warp_n,
          index_t warp_k>
constexpr bool has_wmma_traits_v =
    has_wmma_traits<Arch, AType, BType, CType, warp_m, warp_n, warp_k>::value;
} // namespace ck_tile
