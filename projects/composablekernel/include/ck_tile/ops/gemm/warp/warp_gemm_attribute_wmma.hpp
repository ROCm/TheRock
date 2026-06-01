// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl.hpp"

namespace ck_tile {

// TODO: currently only support 16 bit input, which means only support tr16_b128; will use ADataType
// to determine the layout in the future
template <typename Impl>
struct AWarpDstrEncodingTrait
{
    using type = tile_distribution_encoding<
        sequence<Impl::kRepeat>,
        tuple<sequence<Impl::kAMLane>,
              sequence<Impl::kABK0PerLane, Impl::kABKLane, Impl::kABK1PerLane>>,
        tuple<typename Impl::kABPs2RHssMajor>,
        tuple<typename Impl::kABPs2RHssMinor>,
        typename Impl::kABYs2RHsMajor,
        typename Impl::kABYs2RHsMinor>;
};

template <typename Impl>
struct BWarpDstrEncodingTrait
{
    using type = tile_distribution_encoding<
        sequence<Impl::kRepeat>,
        tuple<sequence<Impl::kBNLane>,
              sequence<Impl::kABK0PerLane, Impl::kABKLane, Impl::kABK1PerLane>>,
        tuple<typename Impl::kABPs2RHssMajor>,
        tuple<typename Impl::kABPs2RHssMinor>,
        typename Impl::kABYs2RHsMajor,
        typename Impl::kABYs2RHsMinor>;
};

template <typename Impl>
struct CWarpDstrEncodingTrait
{
    using type = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane>,
              sequence<Impl::kCNLane>>,
        tuple<typename Impl::kCPs2RHssMajor>,
        tuple<typename Impl::kCPs2RHssMinor>,
        typename Impl::kCYs2RHsMajor,
        typename Impl::kCYs2RHsMinor>;
};

template <typename Impl>
struct CTransposedWarpDstrEncodingTrait
{
    using type = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kCNLane>,
              sequence<Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane>>,
        tuple<typename Impl::kCTPs2RHssMajor>,
        tuple<typename Impl::kCTPs2RHssMinor>,
        typename Impl::kCTYs2RHsMajor,
        typename Impl::kCTYs2RHsMinor>;
};

template <typename WarpGemmAttributeWmmaImpl_, bool kTransC = false>
struct WarpGemmAttributeWmma
{
    using Impl = remove_cvref_t<WarpGemmAttributeWmmaImpl_>;

    // When kTransC is true and A/B types differ, we need an impl with swapped types
    using TransposedImpl =
        std::conditional_t<kTransC &&
                               !std::is_same_v<typename Impl::ADataType, typename Impl::BDataType>,
                           WarpGemmAttributeWmmaImpl<WmmaTraits<typename Impl::TraitsType::ArchType,
                                                                typename Impl::BDataType,
                                                                typename Impl::ADataType,
                                                                typename Impl::CDataType,
                                                                Impl::kM,
                                                                Impl::kN,
                                                                Impl::kK>>,
                           Impl>;

    using ADataType = typename Impl::ADataType;
    using BDataType = typename Impl::BDataType;
    using CDataType = typename Impl::CDataType;

    using AVecType = typename Impl::AVecType;
    using BVecType = typename Impl::BVecType;
    using CVecType = typename Impl::CVecType;

    static constexpr index_t kM          = Impl::kM;
    static constexpr index_t kN          = Impl::kN;
    static constexpr index_t kK          = Impl::kK;
    static constexpr index_t kCMLane     = Impl::kCMLane;
    static constexpr index_t kKPerThread = Impl::kABK0PerLane * Impl::kABK1PerLane;

    CK_TILE_HOST_DEVICE static constexpr auto get_num_of_access() { return 1; }

    // 16 bit input, kAMLane = 16, kABK0PerLane = 4, kABKLane = 2, kABK1PerLane = 2
    // 8  bit input, kAMLane = 16, kABK0PerLane = 2, kABKLane = 2, kABK1PerLane = 4
    using AWarpDstrEncoding = typename AWarpDstrEncodingTrait<Impl>::type;
    using BWarpDstrEncoding = typename BWarpDstrEncodingTrait<Impl>::type;

    // kCM0PerLane = 1, kCMLane = 2, kCM1PerLane = 2, kCNLane = 16
    using CWarpDstrEncoding =
        std::conditional_t<kTransC,
                           typename CTransposedWarpDstrEncodingTrait<Impl>::type,
                           typename CWarpDstrEncodingTrait<Impl>::type>;

    // c_vec += a_vec * b_vec
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        if constexpr(kTransC)
        {
            TransposedImpl{}(c_vec, b_vec, a_vec, bool_constant<post_nop_>{});
        }
        else
        {
            Impl{}(c_vec, a_vec, b_vec, bool_constant<post_nop_>{});
        }
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        if constexpr(kTransC)
        {
            return TransposedImpl{}(b_vec, a_vec);
        }
        else
        {
            return Impl{}(a_vec, b_vec);
        }
    }
};

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          index_t M_Warp_Tile,
          index_t N_Warp_Tile,
          index_t K_Warp_Tile>
CK_TILE_HOST bool check_wmma_supported()
{
    if(is_gfx12_supported())
    {
        return has_wmma_traits_v<gfx12_t,
                                 ADataType,
                                 BDataType,
                                 AccDataType,
                                 M_Warp_Tile,
                                 N_Warp_Tile,
                                 K_Warp_Tile>;
    }
    else if(is_gfx11_supported())
    {
        return has_wmma_traits_v<gfx11_t,
                                 ADataType,
                                 BDataType,
                                 AccDataType,
                                 M_Warp_Tile,
                                 N_Warp_Tile,
                                 K_Warp_Tile>;
    }
    else
    {
        return false;
    }
}

} // namespace ck_tile
