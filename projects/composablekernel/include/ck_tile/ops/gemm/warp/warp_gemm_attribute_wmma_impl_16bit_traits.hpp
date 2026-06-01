// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "warp_gemm_attribute_wmma_impl_base_traits.hpp"
namespace ck_tile {
// fp16 specialization - GFX11
template <>
struct WmmaTraits<gfx11_t, fp16_t, fp16_t, float, 16, 16, 16>
    : WmmaTraitsBase<gfx11_t, fp16_t, fp16_t, float>
{
    using ArchType = gfx11_t;

    template <bool clamp = false>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx11__
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_vec, b_vec, c_vec);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0.f};
#endif
    }
};

// bf16 specialization - GFX11
template <>
struct WmmaTraits<gfx11_t, bf16_t, bf16_t, float, 16, 16, 16>
    : WmmaTraitsBase<gfx11_t, bf16_t, bf16_t, float>
{
    using ArchType = gfx11_t;

    template <bool clamp = false>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx11__
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_vec, b_vec, c_vec);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0.f};
#endif
    }
};

// fp16 specialization - GFX12
template <>
struct WmmaTraits<gfx12_t, fp16_t, fp16_t, float, 16, 16, 16>
    : WmmaTraitsBase<gfx12_t, fp16_t, fp16_t, float>
{
    using ArchType = gfx12_t;

    template <bool clamp = false>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx12__
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_vec, b_vec, c_vec);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0.f};
#endif
    }
};

// bf16 specialization - GFX12
template <>
struct WmmaTraits<gfx12_t, bf16_t, bf16_t, float, 16, 16, 16>
    : WmmaTraitsBase<gfx12_t, bf16_t, bf16_t, float>
{
    using ArchType = gfx12_t;

    template <bool clamp = false>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx12__
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a_vec, b_vec, c_vec);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0.f};
#endif
    }
};
} // namespace ck_tile
