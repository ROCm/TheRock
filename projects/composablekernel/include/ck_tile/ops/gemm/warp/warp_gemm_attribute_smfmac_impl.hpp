// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "warp_gemm_attribute_mfma_impl.hpp"

namespace ck_tile {

// fp16 2:4 structured sparsity

template <WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
struct WarpGemmAttributeSmfmacImplF16F16F32M32N32K16
{
    static constexpr WGAttrCtlEnum Ctrl = Ctrl_;
    using ADataType                     = fp16_t;
    using BDataType                     = fp16_t;
    using IdxDataType                   = int32_t;
    using CDataType                     = float;

    using AVecType = ext_vector_t<fp16_t, 4>;
    using BVecType = ext_vector_t<fp16_t, 8>;
    using CVecType = ext_vector_t<float, 16>;

    static constexpr index_t kM = 32;
    static constexpr index_t kN = 32;
    static constexpr index_t kK = 16;

    static constexpr index_t kAMBlock = 1;
    static constexpr index_t kBNBlock = 1;

    static constexpr index_t kAMLane     = 32;
    static constexpr index_t kBNLane     = 32;
    static constexpr index_t kABKLane    = 2;
    static constexpr index_t kABKPerLane = 8;

    static constexpr index_t kCMLane     = 2;
    static constexpr index_t kCNLane     = 32;
    static constexpr index_t kCM0PerLane = 4;
    static constexpr index_t kCM1PerLane = 4;

    static constexpr index_t CompressionRatio = 2;

    // c_vec += a_vec * b_vec[idx]
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   const int32_t& idx,
                                   bool_constant<post_nop_> = {}) const
    {
#if defined(__gfx94_) or defined(__gfx95_)
        c_vec = __builtin_amdgcn_smfmac_f32_32x32x16_f16(a_vec, b_vec, c_vec, idx, 0, 0);
#else
        ck_tile::ignore = c_vec;
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = idx;
#endif
    }
};

template <WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
struct WarpGemmAttributeSmfmacImplF16F16F32M16N16K32
{
    static constexpr WGAttrCtlEnum Ctrl = Ctrl_;
    using ADataType                     = fp16_t;
    using BDataType                     = fp16_t;
    using IdxDataType                   = int32_t;
    using CDataType                     = float;

    using AVecType = ext_vector_t<fp16_t, 4>;
    using BVecType = ext_vector_t<fp16_t, 8>;
    using CVecType = ext_vector_t<float, 4>;

    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 32;

    static constexpr index_t kAMBlock = 1;
    static constexpr index_t kBNBlock = 1;

    static constexpr index_t kAMLane     = 16;
    static constexpr index_t kBNLane     = 16;
    static constexpr index_t kABKLane    = 4;
    static constexpr index_t kABKPerLane = 8;

    static constexpr index_t kCMLane     = 4;
    static constexpr index_t kCNLane     = 16;
    static constexpr index_t kCM0PerLane = 1;
    static constexpr index_t kCM1PerLane = 4;

    static constexpr index_t CompressionRatio = 2;

    // c_vec += a_vec * b_vec[idx]
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   const int32_t& idx,
                                   bool_constant<post_nop_> = {}) const
    {
#if defined(__gfx94_) or defined(__gfx95_)
        c_vec = __builtin_amdgcn_smfmac_f32_16x16x32_f16(a_vec, b_vec, c_vec, idx, 0, 0);
#else
        ck_tile::ignore = c_vec;
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = idx;
#endif
    }
};

} // namespace ck_tile
