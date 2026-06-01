// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma_impl.hpp"

namespace ck_tile {

// Number of groups of consecutive elements to fill in a ABKLane
enum class WGAttrNumAccessEnum
{
    Single  = 1,
    Double  = 2,
    Quad    = 4,
    Invalid = -1
};

template <typename WarpGemmAttributeMfmaImpl_,
          WGAttrNumAccessEnum AttrNumAccess_ = WGAttrNumAccessEnum::Single>
struct WarpGemmAttributeMfma
{
    using Impl                           = remove_cvref_t<WarpGemmAttributeMfmaImpl_>;
    static constexpr auto AttrNumAccess  = AttrNumAccess_;
    static constexpr auto AttrNumAccessV = static_cast<index_t>(AttrNumAccess);

    using ADataType = typename Impl::ADataType;
    using BDataType = typename Impl::BDataType;
    using CDataType = typename Impl::CDataType;

    using AVecType = typename Impl::AVecType;
    using BVecType = typename Impl::BVecType;
    using CVecType = typename Impl::CVecType;

    static constexpr index_t kM          = Impl::kM;
    static constexpr index_t kN          = Impl::kN;
    static constexpr index_t kK          = Impl::kK;
    static constexpr index_t kKPerThread = Impl::kABKPerLane;
    static constexpr index_t kCMLane     = Impl::kCMLane;

    CK_TILE_HOST_DEVICE static constexpr auto get_num_of_access() { return 1; }

    static_assert(Impl::kAMBlock == 1 && Impl::kBNBlock == 1,
                  "Multi-block WarpGemmAttributeMfmaImpl is not supported");

    template <index_t kMNLane>
    static constexpr auto get_warp_dstr_encoding()
    {
        static_assert(kKPerThread % AttrNumAccessV == 0,
                      "kKPerThread must be divisible by NumAccess");
        if constexpr(AttrNumAccessV == 1)
            return tile_distribution_encoding<
                sequence<>,
                tuple<sequence<kMNLane>, sequence<Impl::kABKLane, Impl::kABKPerLane>>,
                tuple<sequence<2, 1>>,
                tuple<sequence<0, 0>>,
                sequence<2>,
                sequence<1>>{};
        else
            return tile_distribution_encoding<
                sequence<>,
                tuple<sequence<kMNLane>,
                      sequence<AttrNumAccessV, Impl::kABKLane, Impl::kABKPerLane / AttrNumAccessV>>,
                tuple<sequence<2, 1>>,
                tuple<sequence<1, 0>>,
                sequence<2, 2>,
                sequence<0, 2>>{};
    }
    using AWarpDstrEncoding = decltype(get_warp_dstr_encoding<Impl::kAMLane>());
    using BWarpDstrEncoding = decltype(get_warp_dstr_encoding<Impl::kBNLane>());

    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane>,
              sequence<Impl::kCNLane>>,
        tuple<sequence<1, 2>>,
        tuple<sequence<1, 0>>,
        sequence<1, 1>,
        sequence<0, 2>>;

    // c_vec += a_vec * b_vec
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        Impl{}(c_vec, a_vec, b_vec, bool_constant<post_nop_>{});
    }

    // c_vec += a_vec * b_vec
    template <index_t opselA, index_t opselB, bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const int32_t& a_scale,
                                   const BVecType& b_vec,
                                   const int32_t& b_scale,
                                   bool_constant<post_nop_> = {}) const
    {
        Impl{}.template operator()<opselA, opselB>(
            c_vec, a_vec, a_scale, b_vec, b_scale, bool_constant<post_nop_>{});
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        return Impl{}(a_vec, b_vec);
    }

    // c_vec = a_vec * b_vec
    template <index_t opselA, index_t opselB>
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec,
                                       const int32_t& a_scale,
                                       const BVecType& b_vec,
                                       const int32_t& b_scale) const
    {
        return Impl{}.template operator()<opselA, opselB>(a_vec, a_scale, b_vec, b_scale);
    }
};

template <typename WarpGemmAttributeMfmaImpl_,
          index_t kKIter,
          WGAttrNumAccessEnum AttrNumAccess_ = WGAttrNumAccessEnum::Single>
struct WarpGemmAttributeMfmaIterateK
{
    static_assert(kKIter > 0, "wrong!");

    using Impl                           = remove_cvref_t<WarpGemmAttributeMfmaImpl_>;
    static constexpr auto AttrNumAccess  = AttrNumAccess_;
    static constexpr auto AttrNumAccessV = static_cast<index_t>(AttrNumAccess);

    using ADataType = typename Impl::ADataType;
    using BDataType = typename Impl::BDataType;
    using CDataType = typename Impl::CDataType;

    using AVecType =
        ext_vector_t<ADataType, vector_traits<typename Impl::AVecType>::vector_size * kKIter>;
    using BVecType =
        ext_vector_t<BDataType, vector_traits<typename Impl::BVecType>::vector_size * kKIter>;
    using CVecType = typename Impl::CVecType;

    static constexpr index_t kM          = Impl::kM;
    static constexpr index_t kN          = Impl::kN;
    static constexpr index_t kK          = Impl::kK * kKIter;
    static constexpr index_t kKPerThread = Impl::kABKPerLane * kKIter;
    static constexpr index_t kCMLane     = Impl::kCMLane;

    CK_TILE_HOST_DEVICE static constexpr auto get_num_of_access() { return kKIter; }

    static_assert(Impl::kAMBlock == 1 || Impl::kBNBlock == 1,
                  "Multi-block on both M & N directions is not supported");

    template <index_t kMNLane, index_t kMNBlock, index_t kNMBlock>
    CK_TILE_DEVICE static constexpr auto get_warp_dstr_encoding()
    {
        if constexpr(kMNBlock == 1 && kNMBlock == 1)
        {
            static_assert(kKPerThread % AttrNumAccessV == 0,
                          "kKPerThread must be divisible by NumAccess");
            if constexpr(AttrNumAccessV == 1)
                return tile_distribution_encoding<
                    sequence<>,
                    tuple<sequence<kMNLane>, sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
                    tuple<sequence<2, 1>>,
                    tuple<sequence<0, 0>>,
                    sequence<2>,
                    sequence<1>>{};
            else
                return tile_distribution_encoding<
                    sequence<>,
                    tuple<sequence<kMNLane>,
                          sequence<AttrNumAccessV,
                                   Impl::kABKLane,
                                   Impl::kABKPerLane * kKIter / AttrNumAccessV>>,
                    tuple<sequence<2, 1>>,
                    tuple<sequence<1, 0>>,
                    sequence<2, 2>,
                    sequence<0, 2>>{};
        }
        else if constexpr(kMNBlock == 1 && 1 < kNMBlock)
        {
            static_assert(AttrNumAccessV == 1,
                          "Multiple access is not supported when using multi-block");
            // each M/N blocks share the same data
            return tile_distribution_encoding<
                sequence<kNMBlock>,
                tuple<sequence<kMNLane>, sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
                tuple<sequence<0, 2, 1>>,
                tuple<sequence<0, 0, 0>>,
                sequence<2>,
                sequence<1>>{};
        }
        else if constexpr(1 < kMNBlock && kNMBlock == 1)
        {
            static_assert(AttrNumAccessV == 1,
                          "Multiple access is not supported when using multi-block");
            // single block to multi-block thread mapping
            return tile_distribution_encoding<
                sequence<>,
                tuple<sequence<kMNBlock, kMNLane>,
                      sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
                tuple<sequence<1, 2, 1>>,
                tuple<sequence<0, 0, 1>>,
                sequence<2>,
                sequence<1>>{};
        }
    }

    CK_TILE_DEVICE static constexpr auto get_cwarp_dstr_encoding()
    {
        if constexpr(Impl::kAMBlock == 1 && Impl::kBNBlock == 1)
        {
            return tile_distribution_encoding<
                sequence<>,
                tuple<sequence<Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane>,
                      sequence<Impl::kCNLane>>,
                tuple<sequence<1, 2>>,
                tuple<sequence<1, 0>>,
                sequence<1, 1>,
                sequence<0, 2>>{};
        }
        else if constexpr(Impl::kAMBlock == 1 && 1 < Impl::kBNBlock)
        {
            return tile_distribution_encoding<
                sequence<>,
                tuple<sequence<Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane>,
                      sequence<Impl::kBNBlock * Impl::kCNLane>>,
                tuple<sequence<1, 2>>,
                tuple<sequence<1, 0>>,
                sequence<1, 1>,
                sequence<0, 2>>{};
        }
        else if constexpr(1 < Impl::kAMBlock && Impl::kBNBlock == 1)
        {
            return tile_distribution_encoding<
                sequence<>,
                tuple<
                    sequence<Impl::kCM0PerLane, Impl::kAMBlock * Impl::kCMLane, Impl::kCM1PerLane>,
                    sequence<Impl::kCNLane>>,
                tuple<sequence<1, 2>>,
                tuple<sequence<1, 0>>,
                sequence<1, 1>,
                sequence<0, 2>>{};
        }
    }

    using AWarpDstrEncoding =
        decltype(get_warp_dstr_encoding<Impl::kAMLane, Impl::kAMBlock, Impl::kBNBlock>());
    using BWarpDstrEncoding =
        decltype(get_warp_dstr_encoding<Impl::kBNLane, Impl::kBNBlock, Impl::kAMBlock>());
    using CWarpDstrEncoding = decltype(get_cwarp_dstr_encoding());

    // c_vec += a_vec * b_vec
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        static_for<0, kKIter, 1>{}([&](auto iKIter) { operator()(c_vec, a_vec, b_vec, iKIter); });
    }

    template <index_t iKIter, bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   number<iKIter>,
                                   bool_constant<post_nop_> = {}) const
    {
        using buf_a = thread_buffer<typename Impl::AVecType, kKIter>;
        using buf_b = thread_buffer<typename Impl::BVecType, kKIter>;

        static_assert(iKIter < kKIter);

        Impl{}(c_vec,
               reinterpret_cast<const buf_a&>(a_vec)
                   .template get_as<typename Impl::AVecType>()[iKIter],
               reinterpret_cast<const buf_b&>(b_vec)
                   .template get_as<typename Impl::BVecType>()[iKIter],
               bool_constant<post_nop_>{});
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        constexpr auto I0 = number<0>{};
        using buf_a       = thread_buffer<typename Impl::AVecType, kKIter>;
        using buf_b       = thread_buffer<typename Impl::BVecType, kKIter>;

        // c = a * b
        auto c_vec = Impl{}(
            reinterpret_cast<const buf_a&>(a_vec).template get_as<typename Impl::AVecType>()[I0],
            reinterpret_cast<const buf_b&>(b_vec).template get_as<typename Impl::BVecType>()[I0]);

        // c += a * b
        static_for<1, kKIter, 1>{}([&](auto iKIter) { operator()(c_vec, a_vec, b_vec, iKIter); });

        return c_vec;
    }
};

template <typename WarpGemmAttributeMfmaImpl_,
          WGAttrNumAccessEnum AttrNumAccess_ = WGAttrNumAccessEnum::Single>
struct WarpGemmAttributeMfmaTransposedCDistribution
{
    using Impl                           = remove_cvref_t<WarpGemmAttributeMfmaImpl_>;
    static constexpr auto AttrNumAccess  = AttrNumAccess_;
    static constexpr auto AttrNumAccessV = static_cast<index_t>(AttrNumAccess);

    using ADataType = typename Impl::BDataType;
    using BDataType = typename Impl::ADataType;
    using CDataType = typename Impl::CDataType;

    using AVecType = typename Impl::BVecType;
    using BVecType = typename Impl::AVecType;
    using CVecType = typename Impl::CVecType;

    static constexpr index_t kM          = Impl::kN;
    static constexpr index_t kN          = Impl::kM;
    static constexpr index_t kK          = Impl::kK;
    static constexpr index_t kKPerThread = Impl::kABKPerLane;
    static constexpr index_t kCMLane     = Impl::kCMLane;

    CK_TILE_HOST_DEVICE static constexpr auto get_num_of_access() { return 1; }

    static_assert(Impl::kAMBlock == 1 && Impl::kBNBlock == 1,
                  "Multi-block WarpGemmAttributeMfmaImpl is not supported");

    using AWarpDstrEncoding =
        typename WarpGemmAttributeMfma<Impl, AttrNumAccess>::BWarpDstrEncoding;
    using BWarpDstrEncoding =
        typename WarpGemmAttributeMfma<Impl, AttrNumAccess>::AWarpDstrEncoding;

    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kCNLane>,
              sequence<Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<1, 0>>,
        sequence<2, 2>,
        sequence<0, 2>>;

    // c_vec += a_vec * b_vec
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        // swap A and B
        Impl{}(c_vec, b_vec, a_vec, bool_constant<post_nop_>{});
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        // swap A and B
        return Impl{}(b_vec, a_vec);
    }
};

template <typename WarpGemmAttributeMfmaImpl_, index_t SFactor_ = 2>
struct WarpGemmAttributeMfmaTransposedCDistribution_SwizzleB
{
    using Impl = remove_cvref_t<WarpGemmAttributeMfmaImpl_>;

    using ADataType = typename Impl::BDataType;
    using BDataType = typename Impl::ADataType;
    using CDataType = typename Impl::CDataType;

    using AVecType = typename Impl::BVecType;
    using BVecType = typename Impl::AVecType;
    using CVecType = typename Impl::CVecType;

    static constexpr index_t kM          = Impl::kN;
    static constexpr index_t kN          = Impl::kM;
    static constexpr index_t kK          = Impl::kK;
    static constexpr index_t kKPerThread = Impl::kABKPerLane;
    static constexpr index_t SFactor     = SFactor_; // group how many CM1 together

    CK_TILE_HOST_DEVICE static constexpr auto get_num_of_access() { return 1; }

    static_assert(Impl::kAMBlock == 1 && Impl::kBNBlock == 1,
                  "Multi-block WarpGemmAttributeMfmaImpl is not supported");

    using AWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kBNLane>, sequence<Impl::kABKLane, Impl::kABKPerLane>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<0, 0>>,
        sequence<2>,
        sequence<1>>;
#if 0
    using BWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kAMLane / (Impl::kABKPerLane * Impl::kABKLane * 2),
                       Impl::kABKLane,
                       2,
                       Impl::kABKPerLane>,
              sequence<Impl::kABKLane, Impl::kABKPerLane>>,
        tuple<sequence<2, 1, 1, 1, 1>>,
        tuple<sequence<0, 0, 2, 1, 3>>,
        sequence<2>,
        sequence<1>>;

    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kCNLane>,
              sequence<Impl::kCM0PerLane / 2, Impl::kCMLane, Impl::kCM1PerLane * 2>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<1, 0>>,
        sequence<2, 2>,
        sequence<0, 2>>;
#else
    // TODO: more test not only 32x32
    using BWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kAMLane / (Impl::kCMLane * SFactor * Impl::kCM1PerLane),
                       Impl::kCMLane,
                       SFactor,
                       Impl::kCM1PerLane>,
              sequence<Impl::kABKLane, Impl::kABKPerLane>>,
        tuple<sequence<2, 1, 1, 1, 1>>,
        tuple<sequence<0, 0, 2, 1, 3>>,
        sequence<2>,
        sequence<1>>;

    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kCNLane>,
              sequence<Impl::kCM0PerLane / SFactor, Impl::kCMLane, Impl::kCM1PerLane * SFactor>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<1, 0>>,
        sequence<2, 2>,
        sequence<0, 2>>;
#endif
    template <bool post_nop_ = false>
    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        // swap A and B
        Impl{}(c_vec, b_vec, a_vec, bool_constant<post_nop_>{});
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        // swap A and B
        return Impl{}(b_vec, a_vec);
    }
};

template <typename WarpGemmAttributeMfmaImpl_,
          index_t kKIter,
          WGAttrNumAccessEnum AttrNumAccess_ = WGAttrNumAccessEnum::Single>
struct WarpGemmAttributeMfmaIterateKAndTransposedCDistribution
{
    using Impl                          = remove_cvref_t<WarpGemmAttributeMfmaImpl_>;
    static constexpr auto AttrNumAccess = AttrNumAccess_;

    // swap A and B
    using ADataType = typename Impl::BDataType;
    using BDataType = typename Impl::ADataType;
    using CDataType = typename Impl::CDataType;

    using AVecType =
        ext_vector_t<ADataType, vector_traits<typename Impl::AVecType>::vector_size * kKIter>;
    using BVecType =
        ext_vector_t<BDataType, vector_traits<typename Impl::BVecType>::vector_size * kKIter>;
    using CVecType = typename Impl::CVecType;

    static constexpr index_t kM          = Impl::kN;
    static constexpr index_t kN          = Impl::kM;
    static constexpr index_t kK          = Impl::kK * kKIter;
    static constexpr index_t kKPerThread = Impl::kABKPerLane * kKIter;

    CK_TILE_HOST_DEVICE static constexpr auto get_num_of_access() { return kKIter; }

    static_assert(Impl::kAMBlock == 1 || Impl::kBNBlock == 1,
                  "Multi-block on both M & N directions is not supported");

    CK_TILE_DEVICE static constexpr auto get_cwarp_dstr_encoding()
    {
        if constexpr(Impl::kAMBlock == 1 && Impl::kBNBlock == 1)
        {
            return tile_distribution_encoding<
                sequence<>,
                tuple<sequence<Impl::kCNLane>,
                      sequence<Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane>>,
                tuple<sequence<2, 1>>,
                tuple<sequence<1, 0>>,
                sequence<2, 2>,
                sequence<0, 2>>{};
        }
        else if constexpr(Impl::kAMBlock == 1 && 1 < Impl::kBNBlock)
        {
            return tile_distribution_encoding<
                sequence<>,
                tuple<sequence<Impl::kBNBlock * Impl::kCNLane>,
                      sequence<Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane>>,
                tuple<sequence<2, 1>>,
                tuple<sequence<1, 0>>,
                sequence<2, 2>,
                sequence<0, 2>>{};
        }
        else if constexpr(1 < Impl::kAMBlock && Impl::kBNBlock == 1)
        {
            return tile_distribution_encoding<
                sequence<>,
                tuple<
                    sequence<Impl::kCNLane>,
                    sequence<Impl::kCM0PerLane, Impl::kAMBlock * Impl::kCMLane, Impl::kCM1PerLane>>,
                tuple<sequence<2, 1>>,
                tuple<sequence<1, 0>>,
                sequence<2, 2>,
                sequence<0, 2>>{};
        }
    }

    using AWarpDstrEncoding =
        typename WarpGemmAttributeMfmaIterateK<Impl, kKIter, AttrNumAccess>::BWarpDstrEncoding;
    using BWarpDstrEncoding =
        typename WarpGemmAttributeMfmaIterateK<Impl, kKIter, AttrNumAccess>::AWarpDstrEncoding;
    using CWarpDstrEncoding = decltype(get_cwarp_dstr_encoding());

    // c_vec += a_vec * b_vec
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        static_for<0, kKIter, 1>{}([&](auto iKIter) { operator()(c_vec, a_vec, b_vec, iKIter); });
    }

    template <index_t iKIter, bool post_nop_ = false>
    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   number<iKIter>,
                                   bool_constant<post_nop_> = {}) const
    {
        using buf_a = thread_buffer<typename Impl::AVecType, kKIter>;
        using buf_b = thread_buffer<typename Impl::BVecType, kKIter>;

        static_assert(iKIter < kKIter);
        // swap A and B, value and type
        Impl{}(c_vec,
               reinterpret_cast<const buf_b&>(b_vec)
                   .template get_as<typename Impl::BVecType>()[iKIter],
               reinterpret_cast<const buf_a&>(a_vec)
                   .template get_as<typename Impl::AVecType>()[iKIter],
               bool_constant<post_nop_>{});
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        constexpr auto I0 = number<0>{};
        using buf_a       = thread_buffer<typename Impl::AVecType, kKIter>;
        using buf_b       = thread_buffer<typename Impl::BVecType, kKIter>;

        // swap A and B, value and type
        auto c_vec = Impl{}(
            reinterpret_cast<const buf_b&>(b_vec).template get_as<typename Impl::BVecType>()[I0],
            reinterpret_cast<const buf_a&>(a_vec).template get_as<typename Impl::AVecType>()[I0]);

        static_for<1, kKIter, 1>{}([&](auto iKIter) { operator()(c_vec, a_vec, b_vec, iKIter); });

        return c_vec;
    }
};

template <typename WarpGemmAttributeMfmaImpl_, index_t kKIter, index_t SFactor_ = 2>
struct WarpGemmAttributeMfmaIterateKAndTransposedCDistribution_SwizzleB
{
    using Impl = remove_cvref_t<WarpGemmAttributeMfmaImpl_>;

    // swap A and B
    using ADataType = typename Impl::BDataType;
    using BDataType = typename Impl::ADataType;
    using CDataType = typename Impl::CDataType;

    using AVecType =
        ext_vector_t<ADataType, vector_traits<typename Impl::AVecType>::vector_size * kKIter>;
    using BVecType =
        ext_vector_t<BDataType, vector_traits<typename Impl::BVecType>::vector_size * kKIter>;
    using CVecType = typename Impl::CVecType;

    static constexpr index_t kM          = Impl::kN;
    static constexpr index_t kN          = Impl::kM;
    static constexpr index_t kK          = Impl::kK * kKIter;
    static constexpr index_t kKPerThread = Impl::kABKPerLane * kKIter;
    static constexpr index_t SFactor     = SFactor_; // group how many CM1 together

    CK_TILE_HOST_DEVICE static constexpr auto get_num_of_access() { return kKIter; }

    static_assert(Impl::kAMBlock == 1 && Impl::kBNBlock == 1,
                  "Multi-block WarpGemmAttributeMfmaImpl is not supported");

    using AWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kBNLane>, sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<0, 0>>,
        sequence<2>,
        sequence<1>>;
#if 0
    using BWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kAMLane / (Impl::kABKPerLane * Impl::kABKLane * 2),
                       Impl::kABKLane,
                       2,
                       Impl::kABKPerLane>,
              sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
        tuple<sequence<2, 1, 1, 1, 1>>,
        tuple<sequence<0, 0, 2, 1, 3>>,
        sequence<2>,
        sequence<1>>;

    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kCNLane>,
              sequence<Impl::kCM0PerLane / 2, Impl::kCMLane, Impl::kCM1PerLane * 2>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<1, 0>>,
        sequence<2, 2>,
        sequence<0, 2>>;
#else
    // TODO: more test not only 32x32
    using BWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kAMLane / (Impl::kCMLane * SFactor * Impl::kCM1PerLane),
                       Impl::kCMLane,
                       SFactor,
                       Impl::kCM1PerLane>,
              sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
        tuple<sequence<2, 1, 1, 1, 1>>,
        tuple<sequence<0, 0, 2, 1, 3>>,
        sequence<2>,
        sequence<1>>;

    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kCNLane>,
              sequence<Impl::kCM0PerLane / SFactor, Impl::kCMLane, Impl::kCM1PerLane * SFactor>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<1, 0>>,
        sequence<2, 2>,
        sequence<0, 2>>;
#endif
    // c_vec += a_vec * b_vec
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        // swap A and B, value and type
        static_for<0, kKIter, 1>{}([&](auto iKIter) { operator()(c_vec, a_vec, b_vec, iKIter); });
    }

    template <index_t iKIter, bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   number<iKIter>,
                                   bool_constant<post_nop_> = {}) const
    {
        using buf_a = thread_buffer<typename Impl::AVecType, kKIter>;
        using buf_b = thread_buffer<typename Impl::BVecType, kKIter>;

        static_assert(iKIter < kKIter);
        // swap A and B, value and type
        Impl{}(c_vec,
               reinterpret_cast<const buf_b&>(b_vec)
                   .template get_as<typename Impl::BVecType>()[iKIter],
               reinterpret_cast<const buf_a&>(a_vec)
                   .template get_as<typename Impl::AVecType>()[iKIter],
               bool_constant<post_nop_>{});
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        using buf_a       = thread_buffer<typename Impl::AVecType, kKIter>;
        using buf_b       = thread_buffer<typename Impl::BVecType, kKIter>;
        constexpr auto I0 = number<0>{};

        // swap A and B, value and type
        auto c_vec = Impl{}(
            reinterpret_cast<const buf_b&>(b_vec).template get_as<typename Impl::BVecType>()[I0],
            reinterpret_cast<const buf_a&>(a_vec).template get_as<typename Impl::AVecType>()[I0]);

        static_for<1, kKIter, 1>{}([&](auto iKIter) { operator()(c_vec, a_vec, b_vec, iKIter); });

        return c_vec;
    }
};

template <typename WarpGemmAttributeMfmaImpl_, index_t kKIter, index_t SFactor_ = 2>
struct WarpGemmAttributeMfmaIterateK_SwizzleA
{
    using Impl = remove_cvref_t<WarpGemmAttributeMfmaImpl_>;

    using ADataType = typename Impl::ADataType;
    using BDataType = typename Impl::BDataType;
    using CDataType = typename Impl::CDataType;

    using AVecType =
        ext_vector_t<ADataType, vector_traits<typename Impl::AVecType>::vector_size * kKIter>;
    using BVecType =
        ext_vector_t<BDataType, vector_traits<typename Impl::BVecType>::vector_size * kKIter>;
    using CVecType = typename Impl::CVecType;

    static constexpr index_t kM          = Impl::kM;
    static constexpr index_t kN          = Impl::kN;
    static constexpr index_t kK          = Impl::kK * kKIter;
    static constexpr index_t kKPerThread = Impl::kABKPerLane * kKIter;
    static constexpr index_t SFactor     = SFactor_; // group how many CM1 together

    CK_TILE_HOST_DEVICE static constexpr auto get_num_of_access() { return kKIter; }

    static_assert(Impl::kAMBlock == 1 && Impl::kBNBlock == 1,
                  "Multi-block WarpGemmAttributeMfmaImpl is not supported");

    using AWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kAMLane / (Impl::kCMLane * SFactor * Impl::kCM1PerLane),
                       Impl::kCMLane,
                       SFactor,
                       Impl::kCM1PerLane>,
              sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
        tuple<sequence<2, 1, 1, 1, 1>>,
        tuple<sequence<0, 0, 2, 1, 3>>,
        sequence<2>,
        sequence<1>>;

    using BWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kBNLane>, sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<0, 0>>,
        sequence<2>,
        sequence<1>>;

    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kCM0PerLane / SFactor, Impl::kCMLane, Impl::kCM1PerLane * SFactor>,
              sequence<Impl::kCNLane>>,
        tuple<sequence<1, 2>>,
        tuple<sequence<1, 0>>,
        sequence<1, 1>,
        sequence<0, 2>>;

    // c_vec += a_vec * b_vec
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        static_for<0, kKIter, 1>{}([&](auto iKIter) { operator()(c_vec, a_vec, b_vec, iKIter); });
    }

    template <index_t iKIter, bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   number<iKIter>,
                                   bool_constant<post_nop_> = {}) const
    {
        using buf_a = thread_buffer<typename Impl::AVecType, kKIter>;
        using buf_b = thread_buffer<typename Impl::BVecType, kKIter>;

        static_assert(iKIter < kKIter);

        Impl{}(c_vec,
               reinterpret_cast<const buf_a&>(a_vec)
                   .template get_as<typename Impl::AVecType>()[iKIter],
               reinterpret_cast<const buf_b&>(b_vec)
                   .template get_as<typename Impl::BVecType>()[iKIter],
               bool_constant<post_nop_>{});
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        constexpr auto I0 = number<0>{};
        using buf_a       = thread_buffer<typename Impl::AVecType, kKIter>;
        using buf_b       = thread_buffer<typename Impl::BVecType, kKIter>;

        auto c_vec = Impl{}(
            reinterpret_cast<const buf_a&>(a_vec).template get_as<typename Impl::AVecType>()[I0],
            reinterpret_cast<const buf_b&>(b_vec).template get_as<typename Impl::BVecType>()[I0]);

        static_for<1, kKIter, 1>{}([&](auto iKIter) { operator()(c_vec, a_vec, b_vec, iKIter); });

        return c_vec;
    }
};

} // namespace ck_tile
