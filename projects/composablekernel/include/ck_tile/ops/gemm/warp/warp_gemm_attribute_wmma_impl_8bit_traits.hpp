// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "warp_gemm_attribute_wmma_impl_base_traits.hpp"
namespace ck_tile {
// int8 specialization - GFX11
template <>
struct WmmaTraits<gfx11_t, int8_t, int8_t, int32_t, 16, 16, 16>
    : WmmaTraitsBase<gfx11_t, int8_t, int8_t, int32_t>
{
    using ArchType = gfx11_t;

    template <bool clamp = false>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx11__
        return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, // neg_a
                                                          bit_cast<int32x4_t>(a_vec),
                                                          true, // neg_b
                                                          bit_cast<int32x4_t>(b_vec),
                                                          bit_cast<int32x8_t>(c_vec),
                                                          clamp);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

// int8 specialization - GFX12
template <>
struct WmmaTraits<gfx12_t, int8_t, int8_t, int32_t, 16, 16, 16>
    : WmmaTraitsBase<gfx12_t, int8_t, int8_t, int32_t>
{
    using ArchType = gfx12_t;

    template <bool clamp = false>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx12__
        return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, // neg_a
                                                                bit_cast<int32x2_t>(a_vec),
                                                                true, // neg_b
                                                                bit_cast<int32x2_t>(b_vec),
                                                                bit_cast<int32x8_t>(c_vec),
                                                                clamp);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

// fp8/bf8 specialization - GFX12
template <>
struct WmmaTraits<gfx12_t, fp8_t, fp8_t, float, 16, 16, 16>
    : WmmaTraitsBase<gfx12_t, fp8_t, fp8_t, float>
{
    using ArchType = gfx12_t;

    template <bool clamp = false>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx12__
        return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
            bit_cast<int32x2_t>(a_vec), bit_cast<int32x2_t>(b_vec), bit_cast<fp32x8_t>(c_vec));
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

template <>
struct WmmaTraits<gfx12_t, bf8_t, bf8_t, float, 16, 16, 16>
    : WmmaTraitsBase<gfx12_t, bf8_t, bf8_t, float>
{
    using ArchType = gfx12_t;

    template <bool clamp = false>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx12__
        return __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(
            bit_cast<int32x2_t>(a_vec), bit_cast<int32x2_t>(b_vec), bit_cast<fp32x8_t>(c_vec));
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

template <>
struct WmmaTraits<gfx12_t, fp8_t, bf8_t, float, 16, 16, 16>
    : WmmaTraitsBase<gfx12_t, fp8_t, bf8_t, float>
{
    using ArchType = gfx12_t;

    template <bool clamp = false>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx12__
        return __builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w32_gfx12(
            bit_cast<int32x2_t>(a_vec), bit_cast<int32x2_t>(b_vec), bit_cast<fp32x8_t>(c_vec));
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

template <>
struct WmmaTraits<gfx12_t, bf8_t, fp8_t, float, 16, 16, 16>
    : WmmaTraitsBase<gfx12_t, bf8_t, fp8_t, float>
{
    template <bool clamp = false>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx12__
        return __builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w32_gfx12(
            bit_cast<int32x2_t>(a_vec), bit_cast<int32x2_t>(b_vec), bit_cast<fp32x8_t>(c_vec));
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};
} // namespace ck_tile
