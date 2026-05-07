// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// TODO: refactor warp-gemm
// currently there is a discrepency for vav/vva if we need transpose C/D
// e.g. if we want A:agpr, B:vgpr, we have to use vva in WGAttrEnum
// because we swap the A/B pointer in _impl code (but not known this info here)
enum class WGAttrCtlEnum
{
    Default_ = 0,
    Raw_vvv  = 1, // c-vgpr, a-vgpr, b-vgpr
    Raw_vaa  = 2, // c-vgpr, a-agpr, b-agpr
    Raw_vav  = 3, // c-vgpr, a-agpr, b-vgpr
    Raw_vva  = 4, // c-vgpr, a-vgpr, b-agpr
    Raw_avv  = 5, // c-agpr, a-vgpr, b-vgpr
    // raw_a_a_a = 3,  // c-agpr, a-agpr, b-agpr
};

#define DISPATCH_MFMA_(mfma_, dmod_, amod_, bmod_, cmod_)       \
    if constexpr(post_nop_)                                     \
    {                                                           \
        asm volatile(mfma_ " %0, %1, %2, %3 ; yyy\n"            \
                           "s_nop 3"                            \
                     : dmod_(c_vec)                             \
                     : amod_(a_vec), bmod_(b_vec), cmod_(c_vec) \
                     :);                                        \
    }                                                           \
    else                                                        \
    {                                                           \
        asm volatile(mfma_ " %0, %1, %2, %3\n"                  \
                     : dmod_(c_vec)                             \
                     : amod_(a_vec), bmod_(b_vec), cmod_(c_vec) \
                     :);                                        \
    }

#define DISPATCH_MFMA_CTRL_(mfma_, ctrl_)              \
    if constexpr(ctrl_ == WGAttrCtlEnum::Raw_vvv)      \
    {                                                  \
        DISPATCH_MFMA_(mfma_, "+v", "v", "v", "v")     \
    }                                                  \
    else if constexpr(ctrl_ == WGAttrCtlEnum::Raw_vaa) \
    {                                                  \
        DISPATCH_MFMA_(mfma_, "+v", "a", "a", "v")     \
    }                                                  \
    else if constexpr(ctrl_ == WGAttrCtlEnum::Raw_vav) \
    {                                                  \
        DISPATCH_MFMA_(mfma_, "+v", "a", "v", "v")     \
    }                                                  \
    else if constexpr(ctrl_ == WGAttrCtlEnum::Raw_vva) \
    {                                                  \
        DISPATCH_MFMA_(mfma_, "+v", "v", "a", "v")     \
    }                                                  \
    else if constexpr(ctrl_ == WGAttrCtlEnum::Raw_avv) \
    {                                                  \
        DISPATCH_MFMA_(mfma_, "+a", "v", "v", "a")     \
    }

// F32
template <WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
struct WarpGemmAttributeMfmaImplF32F32F32M16N16K4
{
    static constexpr WGAttrCtlEnum Ctrl = Ctrl_;

    using ADataType = float;
    using BDataType = float;
    using CDataType = float;

    using AVecType = ext_vector_t<ADataType, 1>;
    using BVecType = ext_vector_t<BDataType, 1>;
    using CVecType = ext_vector_t<CDataType, 4>;

    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 4;

    static constexpr index_t kAMBlock = 1;
    static constexpr index_t kBNBlock = 1;

    static constexpr index_t kAMLane     = 16;
    static constexpr index_t kBNLane     = 16;
    static constexpr index_t kABKLane    = 4;
    static constexpr index_t kABKPerLane = 1;

    static constexpr index_t kCMLane     = 4;
    static constexpr index_t kCNLane     = 16;
    static constexpr index_t kCM0PerLane = 1;
    static constexpr index_t kCM1PerLane = 4;

    // c_vec += a_vec * b_vec
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        DISPATCH_MFMA_CTRL_("v_mfma_f32_16x16x4f32", Ctrl)
        else
        {
#if defined(__gfx9__)
            c_vec = __builtin_amdgcn_mfma_f32_16x16x4f32(a_vec[0], b_vec[0], c_vec, 0, 0, 0);
#else
            ck_tile::ignore = c_vec;
            ck_tile::ignore = a_vec;
            ck_tile::ignore = b_vec;
#endif
        }
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx9__)
        return bit_cast<CVecType>(
            __builtin_amdgcn_mfma_f32_16x16x4f32(a_vec[0], b_vec[0], CVecType{0.f}, 0, 0, 0));
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        return CVecType{0.f};
#endif
    }
};

template <WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
struct WarpGemmAttributeMfmaImplF32F32F32M32N32K2
{
    static constexpr WGAttrCtlEnum Ctrl = Ctrl_;

    using ADataType = float;
    using BDataType = float;
    using CDataType = float;

    using AVecType = ext_vector_t<ADataType, 1>;
    using BVecType = ext_vector_t<BDataType, 1>;
    using CVecType = ext_vector_t<CDataType, 16>;

    static constexpr index_t kM = 32;
    static constexpr index_t kN = 32;
    static constexpr index_t kK = 2;

    static constexpr index_t kAMBlock = 1;
    static constexpr index_t kBNBlock = 1;

    static constexpr index_t kAMLane     = 32;
    static constexpr index_t kBNLane     = 32;
    static constexpr index_t kABKLane    = 2;
    static constexpr index_t kABKPerLane = 1;

    static constexpr index_t kCMLane     = 2;
    static constexpr index_t kCNLane     = 32;
    static constexpr index_t kCM0PerLane = 4;
    static constexpr index_t kCM1PerLane = 4;

    // c_vec += a_vec * b_vec
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        DISPATCH_MFMA_CTRL_("v_mfma_f32_32x32x2f32", Ctrl)
        else
        {
#if defined(__gfx9__)
            c_vec = __builtin_amdgcn_mfma_f32_32x32x2f32(a_vec[0], b_vec[0], c_vec, 0, 0, 0);
#else
            ck_tile::ignore = c_vec;
            ck_tile::ignore = a_vec;
            ck_tile::ignore = b_vec;
#endif
        }
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx9__)
        return bit_cast<CVecType>(
            __builtin_amdgcn_mfma_f32_32x32x2f32(a_vec[0], b_vec[0], CVecType{0.f}, 0, 0, 0));
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        return CVecType{0.f};
#endif
    }
};

// V_MFMA_F32_16x16x32_BF16
template <WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
struct WarpGemmAttributeMfmaImplBf16Bf16F32M16N16K32
{
    static constexpr WGAttrCtlEnum Ctrl = Ctrl_;
    using ADataType                     = bf16_t;
    using BDataType                     = bf16_t;
    using CDataType                     = float;

    using AVecType = ext_vector_t<bf16_t, 8>;
    using BVecType = ext_vector_t<bf16_t, 8>;
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

    // c_vec += a_vec * b_vec
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        DISPATCH_MFMA_CTRL_("v_mfma_f32_16x16x32_bf16", Ctrl)
        else
        {
#if defined(__gfx950__)
            c_vec = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a_vec, b_vec, c_vec, 0, 0, 0);
#else
            ck_tile::ignore = c_vec;
            ck_tile::ignore = a_vec;
            ck_tile::ignore = b_vec;
#endif
        }
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx950__)
        return bit_cast<CVecType>(
            __builtin_amdgcn_mfma_f32_16x16x32_bf16(a_vec, b_vec, fp32x4_t{0.f}, 0, 0, 0));
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        return CVecType{0.f};
#endif
    }
};
// FP16
template <WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
struct WarpGemmAttributeMfmaImplF16F16F32M32N32K8
{
    static constexpr WGAttrCtlEnum Ctrl = Ctrl_;
    using ADataType                     = fp16_t;
    using BDataType                     = fp16_t;
    using CDataType                     = float;

    using AVecType = ext_vector_t<fp16_t, 4>;
    using BVecType = ext_vector_t<fp16_t, 4>;
    using CVecType = ext_vector_t<float, 16>;

    static constexpr index_t kM = 32;
    static constexpr index_t kN = 32;
    static constexpr index_t kK = 8;

    static constexpr index_t kAMBlock = 1;
    static constexpr index_t kBNBlock = 1;

    static constexpr index_t kAMLane     = 32;
    static constexpr index_t kBNLane     = 32;
    static constexpr index_t kABKLane    = 2;
    static constexpr index_t kABKPerLane = 4;

    static constexpr index_t kCMLane     = 2;
    static constexpr index_t kCNLane     = 32;
    static constexpr index_t kCM0PerLane = 4;
    static constexpr index_t kCM1PerLane = 4;

    // c_vec += a_vec * b_vec
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        DISPATCH_MFMA_CTRL_("v_mfma_f32_32x32x8f16", Ctrl)
        else
        {
#if defined(__gfx9__)
            c_vec = __builtin_amdgcn_mfma_f32_32x32x8f16(a_vec, b_vec, c_vec, 0, 0, 0);
#else
            ck_tile::ignore = c_vec;
            ck_tile::ignore = a_vec;
            ck_tile::ignore = b_vec;
#endif
        }
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx9__)
        return bit_cast<CVecType>(
            __builtin_amdgcn_mfma_f32_32x32x8f16(a_vec, b_vec, fp32x16_t{0.f}, 0, 0, 0));
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        return CVecType{0.f};
#endif
    }
};

template <WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
struct WarpGemmAttributeMfmaImplF16F16F32M16N16K16
{
    static constexpr WGAttrCtlEnum Ctrl = Ctrl_;
    using ADataType                     = fp16_t;
    using BDataType                     = fp16_t;
    using CDataType                     = float;

    using AVecType = ext_vector_t<fp16_t, 4>;
    using BVecType = ext_vector_t<fp16_t, 4>;
    using CVecType = ext_vector_t<float, 4>;

    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 16;

    static constexpr index_t kAMBlock = 1;
    static constexpr index_t kBNBlock = 1;

    static constexpr index_t kAMLane     = 16;
    static constexpr index_t kBNLane     = 16;
    static constexpr index_t kABKLane    = 4;
    static constexpr index_t kABKPerLane = 4;

    static constexpr index_t kCMLane     = 4;
    static constexpr index_t kCNLane     = 16;
    static constexpr index_t kCM0PerLane = 1;
    static constexpr index_t kCM1PerLane = 4;

    // c_vec += a_vec * b_vec
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        DISPATCH_MFMA_CTRL_("v_mfma_f32_16x16x16f16", Ctrl)
        else
        {
#if defined(__gfx9__)
            c_vec = __builtin_amdgcn_mfma_f32_16x16x16f16(a_vec, b_vec, c_vec, 0, 0, 0);
#else
            ck_tile::ignore = c_vec;
            ck_tile::ignore = a_vec;
            ck_tile::ignore = b_vec;
#endif
        }
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx9__)
        return bit_cast<CVecType>(
            __builtin_amdgcn_mfma_f32_16x16x16f16(a_vec, b_vec, fp32x4_t{0.f}, 0, 0, 0));
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        return CVecType{0.f};
#endif
    }
};

template <WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
struct WarpGemmAttributeMfmaImplF16F16F32M16N16K32
{
    static constexpr WGAttrCtlEnum Ctrl = Ctrl_;
    using ADataType                     = fp16_t;
    using BDataType                     = fp16_t;
    using CDataType                     = float;

    using AVecType = ext_vector_t<fp16_t, 8>;
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

    // c_vec += a_vec * b_vec
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        DISPATCH_MFMA_CTRL_("v_mfma_f32_16x16x32f16", Ctrl)
        else
        {
#if defined(__gfx950__)
            c_vec = __builtin_amdgcn_mfma_f32_16x16x32_f16(a_vec, b_vec, c_vec, 0, 0, 0);
#else
            ck_tile::ignore = c_vec;
            ck_tile::ignore = a_vec;
            ck_tile::ignore = b_vec;
#endif
        }
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx950__)
        return bit_cast<CVecType>(
            __builtin_amdgcn_mfma_f32_16x16x32_f16(a_vec, b_vec, fp32x4_t{0.f}, 0, 0, 0));
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        return CVecType{0.f};
#endif
    }
};

template <WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
struct WarpGemmAttributeMfmaImplF16F16F32M4N64K4
{
    static constexpr WGAttrCtlEnum Ctrl = Ctrl_;
    using ADataType                     = fp16_t;
    using BDataType                     = fp16_t;
    using CDataType                     = float;

    using AVecType = ext_vector_t<fp16_t, 4>;
    using BVecType = ext_vector_t<fp16_t, 4>;
    using CVecType = ext_vector_t<float, 4>;

    static constexpr index_t kM = 4;
    static constexpr index_t kN = 64;
    static constexpr index_t kK = 4;

    static constexpr index_t kAMBlock = 1;
    static constexpr index_t kBNBlock = 16;

    // we only write down single block (4 threads) thread mapping here
    static constexpr index_t kAMLane     = 4;
    static constexpr index_t kBNLane     = 4;
    static constexpr index_t kABKLane    = 1;
    static constexpr index_t kABKPerLane = 4;

    static constexpr index_t kCMLane     = 1;
    static constexpr index_t kCNLane     = 4;
    static constexpr index_t kCM0PerLane = 1;
    static constexpr index_t kCM1PerLane = 4;

    // c_vec += a_vec * b_vec
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        DISPATCH_MFMA_CTRL_("v_mfma_f32_4x4x4f16", Ctrl)
        else
        {
#if defined(__gfx9__)
            c_vec = __builtin_amdgcn_mfma_f32_4x4x4f16(a_vec, b_vec, c_vec, 0, 0, 0);
#else
            ignore = c_vec;
            ignore = a_vec;
            ignore = b_vec;
#endif
        }
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx9__)
        return bit_cast<CVecType>(
            __builtin_amdgcn_mfma_f32_4x4x4f16(a_vec, b_vec, fp32x4_t{0.f}, 0, 0, 0));
#else
        ignore = a_vec;
        ignore = b_vec;
        return CVecType{0.f};
#endif
    }
};

template <WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
struct WarpGemmAttributeMfmaImplF16F16F32M64N4K4
{
    static constexpr WGAttrCtlEnum Ctrl = Ctrl_;
    using ADataType                     = fp16_t;
    using BDataType                     = fp16_t;
    using CDataType                     = float;

    using AVecType = ext_vector_t<fp16_t, 4>;
    using BVecType = ext_vector_t<fp16_t, 4>;
    using CVecType = ext_vector_t<float, 4>;

    static constexpr index_t kM = 64;
    static constexpr index_t kN = 4;
    static constexpr index_t kK = 4;

    static constexpr index_t kAMBlock = 16;
    static constexpr index_t kBNBlock = 1;

    // we only write down single block (4 threads) thread mapping here
    static constexpr index_t kAMLane     = 4;
    static constexpr index_t kBNLane     = 4;
    static constexpr index_t kABKLane    = 1;
    static constexpr index_t kABKPerLane = 4;

    static constexpr index_t kCMLane     = 1;
    static constexpr index_t kCNLane     = 4;
    static constexpr index_t kCM0PerLane = 1;
    static constexpr index_t kCM1PerLane = 4;

    // c_vec += a_vec * b_vec
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        DISPATCH_MFMA_CTRL_("v_mfma_f32_4x4x4f16", Ctrl)
        else
        {
#if defined(__gfx9__)
            c_vec = __builtin_amdgcn_mfma_f32_4x4x4f16(a_vec, b_vec, c_vec, 0, 0, 0);
#else
            ignore = c_vec;
            ignore = a_vec;
            ignore = b_vec;
#endif
        }
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx9__)
        return bit_cast<CVecType>(
            __builtin_amdgcn_mfma_f32_4x4x4f16(a_vec, b_vec, fp32x4_t{0.f}, 0, 0, 0));
#else
        ignore = a_vec;
        ignore = b_vec;
        return CVecType{0.f};
#endif
    }
};

// Bf16
template <WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
struct WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K8
{
    static constexpr WGAttrCtlEnum Ctrl = Ctrl_;
    using ADataType                     = bf16_t;
    using BDataType                     = bf16_t;
    using CDataType                     = float;

    using AVecType = ext_vector_t<bf16_t, 4>;
    using BVecType = ext_vector_t<bf16_t, 4>;
    using CVecType = ext_vector_t<float, 16>;

    static constexpr index_t kM = 32;
    static constexpr index_t kN = 32;
    static constexpr index_t kK = 8;

    static constexpr index_t kAMBlock = 1;
    static constexpr index_t kBNBlock = 1;

    static constexpr index_t kAMLane     = 32;
    static constexpr index_t kBNLane     = 32;
    static constexpr index_t kABKLane    = 2;
    static constexpr index_t kABKPerLane = 4;

    static constexpr index_t kCMLane     = 2;
    static constexpr index_t kCNLane     = 32;
    static constexpr index_t kCM0PerLane = 4;
    static constexpr index_t kCM1PerLane = 4;

    // c_vec += a_vec * b_vec
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        DISPATCH_MFMA_CTRL_("v_mfma_f32_32x32x8bf16_1k", Ctrl)
        else
        {
#if defined(__gfx90a__) || defined(__gfx94__)
            c_vec = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a_vec, b_vec, c_vec, 0, 0, 0);
#elif defined(__gfx908__)
            static_for<0, 2, 1>{}([&](auto k) {
                c_vec = __builtin_amdgcn_mfma_f32_32x32x4bf16(
                    reinterpret_cast<const thread_buffer<ADataType, 4>&>(a_vec)
                        .template get_as<ext_vector_t<bf16_t, 2>>()[number<k>{}],
                    reinterpret_cast<const thread_buffer<BDataType, 4>&>(b_vec)
                        .template get_as<ext_vector_t<bf16_t, 2>>()[number<k>{}],
                    c_vec,
                    0,
                    0,
                    0);
            });
#else
            ck_tile::ignore = c_vec;
            ck_tile::ignore = a_vec;
            ck_tile::ignore = b_vec;
#endif
        }
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx90a__) || defined(__gfx94__)
        return bit_cast<CVecType>(
            __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a_vec, b_vec, fp32x16_t{0.f}, 0, 0, 0));
#elif defined(__gfx908__)
        CVecType c_vec{0.f};
        static_for<0, 2, 1>{}([&](auto k) {
            c_vec = __builtin_amdgcn_mfma_f32_32x32x4bf16(
                reinterpret_cast<const thread_buffer<ADataType, 4>&>(a_vec)
                    .template get_as<ext_vector_t<bf16_t, 2>>()[number<k>{}],
                reinterpret_cast<const thread_buffer<BDataType, 4>&>(b_vec)
                    .template get_as<ext_vector_t<bf16_t, 2>>()[number<k>{}],
                c_vec,
                0,
                0,
                0);
        });
        return c_vec;
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        return CVecType{0.f};
#endif
    }
};

template <WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
struct WarpGemmAttributeMfmaImplBf16Bf16F32M16N16K16
{
    static constexpr WGAttrCtlEnum Ctrl = Ctrl_;
    using ADataType                     = bf16_t;
    using BDataType                     = bf16_t;
    using CDataType                     = float;

    using AVecType = ext_vector_t<bf16_t, 4>;
    using BVecType = ext_vector_t<bf16_t, 4>;
    using CVecType = ext_vector_t<float, 4>;

    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 16;

    static constexpr index_t kAMBlock = 1;
    static constexpr index_t kBNBlock = 1;

    static constexpr index_t kAMLane     = 16;
    static constexpr index_t kBNLane     = 16;
    static constexpr index_t kABKLane    = 4;
    static constexpr index_t kABKPerLane = 4;

    static constexpr index_t kCMLane     = 4;
    static constexpr index_t kCNLane     = 16;
    static constexpr index_t kCM0PerLane = 1;
    static constexpr index_t kCM1PerLane = 4;

    // c_vec += a_vec * b_vec
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        DISPATCH_MFMA_CTRL_("v_mfma_f32_16x16x16bf16_1k", Ctrl)
        {
#if defined(__gfx90a__) || defined(__gfx94__)
            c_vec = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_vec, b_vec, c_vec, 0, 0, 0);
#elif defined(__gfx908__)
            static_for<0, 2, 1>{}([&](auto k) {
                c_vec = __builtin_amdgcn_mfma_f32_16x16x8bf16(
                    reinterpret_cast<const thread_buffer<ADataType, 4>&>(a_vec)
                        .template get_as<ext_vector_t<bf16_t, 2>>()[number<k>{}],
                    reinterpret_cast<const thread_buffer<BDataType, 4>&>(b_vec)
                        .template get_as<ext_vector_t<bf16_t, 2>>()[number<k>{}],
                    c_vec,
                    0,
                    0,
                    0);
            });
#else
            ck_tile::ignore = c_vec;
            ck_tile::ignore = a_vec;
            ck_tile::ignore = b_vec;
#endif
        }
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx90a__) || defined(__gfx94__)
        return bit_cast<CVecType>(
            __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_vec, b_vec, fp32x4_t{0.f}, 0, 0, 0));
#elif defined(__gfx908__)
        CVecType c_vec{0.f};
        static_for<0, 2, 1>{}([&](auto k) {
            c_vec = __builtin_amdgcn_mfma_f32_16x16x8bf16(
                reinterpret_cast<const thread_buffer<ADataType, 4>&>(a_vec)
                    .template get_as<ext_vector_t<bf16_t, 2>>()[number<k>{}],
                reinterpret_cast<const thread_buffer<BDataType, 4>&>(b_vec)
                    .template get_as<ext_vector_t<bf16_t, 2>>()[number<k>{}],
                c_vec,
                0,
                0,
                0);
        });
        return c_vec;
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        return CVecType{0.f};
#endif
    }
};

template <WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
struct WarpGemmAttributeMfmaImplBf16Bf16F32M4N64K4
{
    static constexpr WGAttrCtlEnum Ctrl = Ctrl_;
    using ADataType                     = bf16_t;
    using BDataType                     = bf16_t;
    using CDataType                     = float;

    using AVecType = ext_vector_t<bf16_t, 4>;
    using BVecType = ext_vector_t<bf16_t, 4>;
    using CVecType = ext_vector_t<float, 4>;

    static constexpr index_t kM = 4;
    static constexpr index_t kN = 64;
    static constexpr index_t kK = 4;

    static constexpr index_t kAMBlock = 1;
    static constexpr index_t kBNBlock = 16;

    // we only write down single block (4 threads) thread mapping here
    static constexpr index_t kAMLane     = 4;
    static constexpr index_t kBNLane     = 4;
    static constexpr index_t kABKLane    = 1;
    static constexpr index_t kABKPerLane = 4;

    static constexpr index_t kCMLane     = 1;
    static constexpr index_t kCNLane     = 4;
    static constexpr index_t kCM0PerLane = 1;
    static constexpr index_t kCM1PerLane = 4;

    // c_vec += a_vec * b_vec
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        DISPATCH_MFMA_CTRL_("v_mfma_f32_4x4x4bf16_1k", Ctrl)
        else
        {
#if defined(__gfx90a__) || defined(__gfx94__)
            c_vec = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(a_vec, b_vec, c_vec, 0, 0, 0);
#elif defined(__gfx908__)
            static_for<0, 2, 1>{}([&](auto k) {
                c_vec = __builtin_amdgcn_mfma_f32_4x4x2bf16(
                    reinterpret_cast<const thread_buffer<ADataType, 4>&>(a_vec)
                        .template get_as<ext_vector_t<bf16_t, 2>>()[number<k>{}],
                    reinterpret_cast<const thread_buffer<BDataType, 4>&>(b_vec)
                        .template get_as<ext_vector_t<bf16_t, 2>>()[number<k>{}],
                    c_vec,
                    0,
                    0,
                    0);
            });
#else
            ignore = c_vec;
            ignore = a_vec;
            ignore = b_vec;
#endif
        }
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx90a__) || defined(__gfx94__)
        return bit_cast<CVecType>(
            __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(a_vec, b_vec, fp32x4_t{0.f}, 0, 0, 0));
#elif defined(__gfx908__)
        CVecType c_vec{0.f};
        static_for<0, 2, 1>{}([&](auto k) {
            c_vec = __builtin_amdgcn_mfma_f32_4x4x2bf16(
                reinterpret_cast<const thread_buffer<ADataType, 4>&>(a_vec)
                    .template get_as<ext_vector_t<bf16_t, 2>>()[number<k>{}],
                reinterpret_cast<const thread_buffer<BDataType, 4>&>(b_vec)
                    .template get_as<ext_vector_t<bf16_t, 2>>()[number<k>{}],
                c_vec,
                0,
                0,
                0);
        });
        return c_vec;
#else
        ignore = a_vec;
        ignore = b_vec;
        return CVecType{0.f};
#endif
    }
};

template <WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
struct WarpGemmAttributeMfmaImplBf16Bf16F32M64N4K4
{
    static constexpr WGAttrCtlEnum Ctrl = Ctrl_;
    using ADataType                     = bf16_t;
    using BDataType                     = bf16_t;
    using CDataType                     = float;

    using AVecType = ext_vector_t<bf16_t, 4>;
    using BVecType = ext_vector_t<bf16_t, 4>;
    using CVecType = ext_vector_t<float, 4>;

    static constexpr index_t kM = 64;
    static constexpr index_t kN = 4;
    static constexpr index_t kK = 4;

    static constexpr index_t kAMBlock = 16;
    static constexpr index_t kBNBlock = 1;

    // we only write down single block (4 threads) thread mapping here
    static constexpr index_t kAMLane     = 4;
    static constexpr index_t kBNLane     = 4;
    static constexpr index_t kABKLane    = 1;
    static constexpr index_t kABKPerLane = 4;

    static constexpr index_t kCMLane     = 1;
    static constexpr index_t kCNLane     = 4;
    static constexpr index_t kCM0PerLane = 1;
    static constexpr index_t kCM1PerLane = 4;

    // c_vec += a_vec * b_vec
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        DISPATCH_MFMA_CTRL_("v_mfma_f32_4x4x4bf16_1k", Ctrl)
        else
        {
#if defined(__gfx90a__) || defined(__gfx94__)
            c_vec = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(a_vec, b_vec, c_vec, 0, 0, 0);
#elif defined(__gfx908__)
            static_for<0, 2, 1>{}([&](auto k) {
                c_vec = __builtin_amdgcn_mfma_f32_4x4x2bf16(
                    reinterpret_cast<const thread_buffer<ADataType, 4>&>(a_vec)
                        .template get_as<ext_vector_t<bf16_t, 2>>()[number<k>{}],
                    reinterpret_cast<const thread_buffer<BDataType, 4>&>(b_vec)
                        .template get_as<ext_vector_t<bf16_t, 2>>()[number<k>{}],
                    c_vec,
                    0,
                    0,
                    0);
            });
#else
            ignore = c_vec;
            ignore = a_vec;
            ignore = b_vec;
#endif
        }
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx90a__) || defined(__gfx94__)
        return bit_cast<CVecType>(
            __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(a_vec, b_vec, fp32x4_t{0.f}, 0, 0, 0));
#elif defined(__gfx908__)
        CVecType c_vec{0.f};
        static_for<0, 2, 1>{}([&](auto k) {
            c_vec = __builtin_amdgcn_mfma_f32_4x4x2bf16(
                reinterpret_cast<const thread_buffer<ADataType, 4>&>(a_vec)
                    .template get_as<ext_vector_t<bf16_t, 2>>()[number<k>{}],
                reinterpret_cast<const thread_buffer<BDataType, 4>&>(b_vec)
                    .template get_as<ext_vector_t<bf16_t, 2>>()[number<k>{}],
                c_vec,
                0,
                0,
                0);
        });
        return c_vec;
#else
        ignore = a_vec;
        ignore = b_vec;
        return CVecType{0.f};
#endif
    }
};

// gfx950
template <WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
struct WarpGemmAttributeMfmaImplF16F16F32M32N32K16
{
    static constexpr WGAttrCtlEnum Ctrl = Ctrl_;
    using ADataType                     = fp16_t;
    using BDataType                     = fp16_t;
    using CDataType                     = float;

    using AVecType = ext_vector_t<fp16_t, 8>;
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

    // c_vec += a_vec * b_vec
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        DISPATCH_MFMA_CTRL_("v_mfma_f32_32x32x16_f16", Ctrl)
        else
        {
#if defined(__gfx950__)
            c_vec = __builtin_amdgcn_mfma_f32_32x32x16_f16(a_vec, b_vec, c_vec, 0, 0, 0);
#elif defined(__gfx90a__) || defined(__gfx94__)
            static_for<0, 2, 1>{}([&](auto k) {
                c_vec = __builtin_amdgcn_mfma_f32_32x32x8f16(
                    reinterpret_cast<const thread_buffer<ADataType, 8>&>(a_vec)
                        .template get_as<ext_vector_t<fp16_t, 4>>()[number<k>{}],
                    reinterpret_cast<const thread_buffer<BDataType, 8>&>(b_vec)
                        .template get_as<ext_vector_t<fp16_t, 4>>()[number<k>{}],
                    c_vec,
                    0,
                    0,
                    0);
            });
#elif defined(__gfx908__)
            static_for<0, 4, 1>{}([&](auto k) {
                c_vec = __builtin_amdgcn_mfma_f32_32x32x4f16(
                    reinterpret_cast<const thread_buffer<ADataType, 4>&>(a_vec)
                        .template get_as<ext_vector_t<fp16_t, 2>>()[number<k>{}],
                    reinterpret_cast<const thread_buffer<BDataType, 4>&>(b_vec)
                        .template get_as<ext_vector_t<fp16_t, 2>>()[number<k>{}],
                    c_vec,
                    0,
                    0,
                    0);
            });
#else
            ck_tile::ignore = c_vec;
            ck_tile::ignore = a_vec;
            ck_tile::ignore = b_vec;
#endif
        }
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx950__)
        return __builtin_amdgcn_mfma_f32_32x32x16_f16(a_vec, b_vec, fp32x16_t{0.f}, 0, 0, 0);
#elif defined(__gfx90a__) || defined(__gfx94__)
        CVecType c_vec{0.f};
        static_for<0, 2, 1>{}([&](auto k) {
            c_vec = __builtin_amdgcn_mfma_f32_32x32x8f16(
                reinterpret_cast<const thread_buffer<ADataType, 8>&>(a_vec)
                    .template get_as<ext_vector_t<fp16_t, 4>>()[number<k>{}],
                reinterpret_cast<const thread_buffer<BDataType, 8>&>(b_vec)
                    .template get_as<ext_vector_t<fp16_t, 4>>()[number<k>{}],
                c_vec,
                0,
                0,
                0);
        });
        return c_vec;
#elif defined(__gfx908__)
        CVecType c_vec{0.f};
        static_for<0, 4, 1>{}([&](auto k) {
            c_vec = __builtin_amdgcn_mfma_f32_32x32x4f16(
                reinterpret_cast<const thread_buffer<ADataType, 4>&>(a_vec)
                    .template get_as<ext_vector_t<fp16_t, 2>>()[number<k>{}],
                reinterpret_cast<const thread_buffer<BDataType, 4>&>(b_vec)
                    .template get_as<ext_vector_t<fp16_t, 2>>()[number<k>{}],
                c_vec,
                0,
                0,
                0);
        });
        return c_vec;
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        return CVecType{0.f};
#endif
    }
};

template <WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
struct WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K16
{
    static constexpr WGAttrCtlEnum Ctrl = Ctrl_;
    using ADataType                     = bf16_t;
    using BDataType                     = bf16_t;
    using CDataType                     = float;

    using AVecType = ext_vector_t<bf16_t, 8>;
    using BVecType = ext_vector_t<bf16_t, 8>;
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

    // c_vec += a_vec * b_vec
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        DISPATCH_MFMA_CTRL_("v_mfma_f32_32x32x16_bf16", Ctrl)
        else
        {
#if defined(__gfx950__)
            c_vec = __builtin_amdgcn_mfma_f32_32x32x16_bf16(a_vec, b_vec, c_vec, 0, 0, 0);
#elif defined(__gfx90a__) || defined(__gfx94__)
            static_for<0, 2, 1>{}([&](auto k) {
                c_vec = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(
                    reinterpret_cast<const thread_buffer<ADataType, 8>&>(a_vec)
                        .template get_as<ext_vector_t<bf16_t, 4>>()[number<k>{}],
                    reinterpret_cast<const thread_buffer<BDataType, 8>&>(b_vec)
                        .template get_as<ext_vector_t<bf16_t, 4>>()[number<k>{}],
                    c_vec,
                    0,
                    0,
                    0);
            });
#elif defined(__gfx908__)
            static_for<0, 4, 1>{}([&](auto k) {
                c_vec = __builtin_amdgcn_mfma_f32_32x32x4bf16(
                    reinterpret_cast<const thread_buffer<ADataType, 4>&>(a_vec)
                        .template get_as<ext_vector_t<bf16_t, 2>>()[number<k>{}],
                    reinterpret_cast<const thread_buffer<BDataType, 4>&>(b_vec)
                        .template get_as<ext_vector_t<bf16_t, 2>>()[number<k>{}],
                    c_vec,
                    0,
                    0,
                    0);
            });
#else
            ck_tile::ignore = c_vec;
            ck_tile::ignore = a_vec;
            ck_tile::ignore = b_vec;
#endif
        }
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx950__)
        return __builtin_amdgcn_mfma_f32_32x32x16_bf16(a_vec, b_vec, fp32x16_t{0.f}, 0, 0, 0);
#elif defined(__gfx90a__) || defined(__gfx94__)
        CVecType c_vec{0.f};
        static_for<0, 2, 1>{}([&](auto k) {
            c_vec = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(
                reinterpret_cast<const thread_buffer<ADataType, 8>&>(a_vec)
                    .template get_as<ext_vector_t<bf16_t, 4>>()[number<k>{}],
                reinterpret_cast<const thread_buffer<BDataType, 8>&>(b_vec)
                    .template get_as<ext_vector_t<bf16_t, 4>>()[number<k>{}],
                c_vec,
                0,
                0,
                0);
        });
        return c_vec;
#elif defined(__gfx908__)
        CVecType c_vec{0.f};
        static_for<0, 4, 1>{}([&](auto k) {
            c_vec = __builtin_amdgcn_mfma_f32_32x32x4bf16(
                reinterpret_cast<const thread_buffer<ADataType, 4>&>(a_vec)
                    .template get_as<ext_vector_t<bf16_t, 2>>()[number<k>{}],
                reinterpret_cast<const thread_buffer<BDataType, 4>&>(b_vec)
                    .template get_as<ext_vector_t<bf16_t, 2>>()[number<k>{}],
                c_vec,
                0,
                0,
                0);
        });
        return c_vec;
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        return CVecType{0.f};
#endif
    }
};

// FP8
template <typename AType_, typename BType_, WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
struct WarpGemmAttributeMfmaImpl_f32_16x16x32_f8_base
{
    static constexpr WGAttrCtlEnum Ctrl = Ctrl_;
    using ADataType                     = AType_;
    using BDataType                     = BType_;
    using CDataType                     = float;

    using AVecType = ext_vector_t<ADataType, 8>;
    using BVecType = ext_vector_t<BDataType, 8>;
    using CVecType = ext_vector_t<CDataType, 4>;

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

    // c_vec += a_vec * b_vec
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        if constexpr(Ctrl == WGAttrCtlEnum::Raw_vvv)
        {
            if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, fp8_t>)
            {
                DISPATCH_MFMA_("mfma_f32_16x16x32_fp8_fp8", "+v", "v", "v", "v")
            }
            else if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, bf8_t>)
            {
                DISPATCH_MFMA_("mfma_f32_16x16x32_fp8_bf8", "+v", "v", "v", "v")
            }
            else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, fp8_t>)
            {
                DISPATCH_MFMA_("mfma_f32_16x16x32_bf8_fp8", "+v", "v", "v", "v")
            }
            else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, bf8_t>)
            {
                DISPATCH_MFMA_("mfma_f32_16x16x32_bf8_bf8", "+v", "v", "v", "v")
            }
        }
        else if constexpr(Ctrl == WGAttrCtlEnum::Raw_vaa)
        {
            if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, fp8_t>)
            {
                DISPATCH_MFMA_("mfma_f32_16x16x32_fp8_fp8", "+v", "a", "a", "v")
            }
            else if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, bf8_t>)
            {
                DISPATCH_MFMA_("mfma_f32_16x16x32_fp8_bf8", "+v", "a", "a", "v")
            }
            else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, fp8_t>)
            {
                DISPATCH_MFMA_("mfma_f32_16x16x32_bf8_fp8", "+v", "a", "a", "v")
            }
            else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, bf8_t>)
            {
                DISPATCH_MFMA_("mfma_f32_16x16x32_bf8_bf8", "+v", "a", "a", "v")
            }
        }
        else if constexpr(Ctrl == WGAttrCtlEnum::Raw_vav)
        {
            if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, fp8_t>)
            {
                DISPATCH_MFMA_("mfma_f32_16x16x32_fp8_fp8", "+v", "a", "v", "v")
            }
            else if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, bf8_t>)
            {
                DISPATCH_MFMA_("mfma_f32_16x16x32_fp8_bf8", "+v", "a", "v", "v")
            }
            else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, fp8_t>)
            {
                DISPATCH_MFMA_("mfma_f32_16x16x32_bf8_fp8", "+v", "a", "v", "v")
            }
            else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, bf8_t>)
            {
                DISPATCH_MFMA_("mfma_f32_16x16x32_bf8_bf8", "+v", "a", "v", "v")
            }
        }
        else if constexpr(Ctrl == WGAttrCtlEnum::Raw_vva)
        {
            if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, fp8_t>)
            {
                DISPATCH_MFMA_("mfma_f32_16x16x32_fp8_fp8", "+v", "v", "a", "v")
            }
            else if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, bf8_t>)
            {
                DISPATCH_MFMA_("mfma_f32_16x16x32_fp8_bf8", "+v", "v", "a", "v")
            }
            else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, fp8_t>)
            {
                DISPATCH_MFMA_("mfma_f32_16x16x32_bf8_fp8", "+v", "v", "a", "v")
            }
            else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, bf8_t>)
            {
                DISPATCH_MFMA_("mfma_f32_16x16x32_bf8_bf8", "+v", "v", "a", "v")
            }
        }
        else
        {
#if defined(__gfx94__) or defined(__gfx95__)
            if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, fp8_t>)
                c_vec = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
                    bit_cast<int64_t>(a_vec), bit_cast<int64_t>(b_vec), c_vec, 0, 0, 0);
            else if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, bf8_t>)
                c_vec = __builtin_amdgcn_mfma_f32_16x16x32_fp8_bf8(
                    bit_cast<int64_t>(a_vec), bit_cast<int64_t>(b_vec), c_vec, 0, 0, 0);
            else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, fp8_t>)
                c_vec = __builtin_amdgcn_mfma_f32_16x16x32_bf8_fp8(
                    bit_cast<int64_t>(a_vec), bit_cast<int64_t>(b_vec), c_vec, 0, 0, 0);
            else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, bf8_t>)
                c_vec = __builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8(
                    bit_cast<int64_t>(a_vec), bit_cast<int64_t>(b_vec), c_vec, 0, 0, 0);
#else
            ck_tile::ignore = c_vec;
            ck_tile::ignore = a_vec;
            ck_tile::ignore = b_vec;
#endif
        }
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx94__) or defined(__gfx95__)
        if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, fp8_t>)
            return bit_cast<CVecType>(__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
                bit_cast<int64_t>(a_vec), bit_cast<int64_t>(b_vec), CVecType{0.f}, 0, 0, 0));
        else if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, bf8_t>)
            return bit_cast<CVecType>(__builtin_amdgcn_mfma_f32_16x16x32_fp8_bf8(
                bit_cast<int64_t>(a_vec), bit_cast<int64_t>(b_vec), CVecType{0.f}, 0, 0, 0));
        else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, fp8_t>)
            return bit_cast<CVecType>(__builtin_amdgcn_mfma_f32_16x16x32_bf8_fp8(
                bit_cast<int64_t>(a_vec), bit_cast<int64_t>(b_vec), CVecType{0.f}, 0, 0, 0));
        else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, bf8_t>)
            return bit_cast<CVecType>(__builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8(
                bit_cast<int64_t>(a_vec), bit_cast<int64_t>(b_vec), CVecType{0.f}, 0, 0, 0));
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        return CVecType{0.f};
#endif
    }
};

template <typename AType_, typename BType_, WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
struct WarpGemmAttributeMfmaImpl_f32_32x32x16_f8_base
{
    static constexpr WGAttrCtlEnum Ctrl = Ctrl_;
    using ADataType                     = AType_;
    using BDataType                     = BType_;
    using CDataType                     = float;

    using AVecType = ext_vector_t<ADataType, 8>;
    using BVecType = ext_vector_t<BDataType, 8>;
    using CVecType = ext_vector_t<CDataType, 16>;

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

    // c_vec += a_vec * b_vec
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        if constexpr(Ctrl == WGAttrCtlEnum::Raw_vvv)
        {
            if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, fp8_t>)
            {
                DISPATCH_MFMA_("mfma_f32_32x32x16_fp8_fp8", "+v", "v", "v", "v")
            }
            else if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, bf8_t>)
            {
                DISPATCH_MFMA_("mfma_f32_32x32x16_fp8_bf8", "+v", "v", "v", "v")
            }
            else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, fp8_t>)
            {
                DISPATCH_MFMA_("mfma_f32_32x32x16_bf8_fp8", "+v", "v", "v", "v")
            }
            else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, bf8_t>)
            {
                DISPATCH_MFMA_("mfma_f32_32x32x16_bf8_bf8", "+v", "v", "v", "v")
            }
        }
        else if constexpr(Ctrl == WGAttrCtlEnum::Raw_vaa)
        {
            if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, fp8_t>)
            {
                DISPATCH_MFMA_("mfma_f32_32x32x16_fp8_fp8", "+v", "a", "a", "v")
            }
            else if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, bf8_t>)
            {
                DISPATCH_MFMA_("mfma_f32_32x32x16_fp8_bf8", "+v", "a", "a", "v")
            }
            else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, fp8_t>)
            {
                DISPATCH_MFMA_("mfma_f32_32x32x16_bf8_fp8", "+v", "a", "a", "v")
            }
            else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, bf8_t>)
            {
                DISPATCH_MFMA_("mfma_f32_32x32x16_bf8_bf8", "+v", "a", "a", "v")
            }
        }
        else if constexpr(Ctrl == WGAttrCtlEnum::Raw_vav)
        {
            if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, fp8_t>)
            {
                DISPATCH_MFMA_("mfma_f32_32x32x16_fp8_fp8", "+v", "a", "v", "v")
            }
            else if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, bf8_t>)
            {
                DISPATCH_MFMA_("mfma_f32_32x32x16_fp8_bf8", "+v", "a", "v", "v")
            }
            else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, fp8_t>)
            {
                DISPATCH_MFMA_("mfma_f32_32x32x16_bf8_fp8", "+v", "a", "v", "v")
            }
            else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, bf8_t>)
            {
                DISPATCH_MFMA_("mfma_f32_32x32x16_bf8_bf8", "+v", "a", "v", "v")
            }
        }
        else if constexpr(Ctrl == WGAttrCtlEnum::Raw_vva)
        {
            if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, fp8_t>)
            {
                DISPATCH_MFMA_("mfma_f32_32x32x16_fp8_fp8", "+v", "v", "a", "v")
            }
            else if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, bf8_t>)
            {
                DISPATCH_MFMA_("mfma_f32_32x32x16_fp8_bf8", "+v", "v", "a", "v")
            }
            else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, fp8_t>)
            {
                DISPATCH_MFMA_("mfma_f32_32x32x16_bf8_fp8", "+v", "v", "a", "v")
            }
            else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, bf8_t>)
            {
                DISPATCH_MFMA_("mfma_f32_32x32x16_bf8_bf8", "+v", "v", "a", "v")
            }
        }
        else
        {
#if defined(__gfx94__) or defined(__gfx95__)
            if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, fp8_t>)
                c_vec = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
                    bit_cast<int64_t>(a_vec), bit_cast<int64_t>(b_vec), c_vec, 0, 0, 0);
            else if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, bf8_t>)
                c_vec = __builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8(
                    bit_cast<int64_t>(a_vec), bit_cast<int64_t>(b_vec), c_vec, 0, 0, 0);
            else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, fp8_t>)
                c_vec = __builtin_amdgcn_mfma_f32_32x32x16_bf8_fp8(
                    bit_cast<int64_t>(a_vec), bit_cast<int64_t>(b_vec), c_vec, 0, 0, 0);
            else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, bf8_t>)
                c_vec = __builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8(
                    bit_cast<int64_t>(a_vec), bit_cast<int64_t>(b_vec), c_vec, 0, 0, 0);
#elif defined(__gfx908__) || defined(__gfx90a__)
            static_for<0, 8, 1>{}([&](auto k) {
                float a_f32 =
                    type_convert<float>(reinterpret_cast<const thread_buffer<ADataType, 8>&>(a_vec)
                                            .template get_as<ADataType>()[number<k>{}]);
                float b_f32 =
                    type_convert<float>(reinterpret_cast<const thread_buffer<BDataType, 8>&>(b_vec)
                                            .template get_as<BDataType>()[number<k>{}]);

                c_vec = __builtin_amdgcn_mfma_f32_32x32x2f32(a_f32, b_f32, c_vec, 0, 0, 0);
            });
#else
            ck_tile::ignore = c_vec;
            ck_tile::ignore = a_vec;
            ck_tile::ignore = b_vec;
#endif
        }
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx94__) or defined(__gfx95__)
        if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, fp8_t>)
            return bit_cast<CVecType>(__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
                bit_cast<int64_t>(a_vec), bit_cast<int64_t>(b_vec), CVecType{0.f}, 0, 0, 0));
        else if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, bf8_t>)
            return bit_cast<CVecType>(__builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8(
                bit_cast<int64_t>(a_vec), bit_cast<int64_t>(b_vec), CVecType{0.f}, 0, 0, 0));
        else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, fp8_t>)
            return bit_cast<CVecType>(__builtin_amdgcn_mfma_f32_32x32x16_bf8_fp8(
                bit_cast<int64_t>(a_vec), bit_cast<int64_t>(b_vec), CVecType{0.f}, 0, 0, 0));
        else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, bf8_t>)
            return bit_cast<CVecType>(__builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8(
                bit_cast<int64_t>(a_vec), bit_cast<int64_t>(b_vec), CVecType{0.f}, 0, 0, 0));
#elif defined(__gfx908__) || defined(__gfx90a__)
        CVecType c_vec{0.f};
        static_for<0, 8, 1>{}([&](auto k) {
            float a_f32 =
                type_convert<float>(reinterpret_cast<const thread_buffer<ADataType, 8>&>(a_vec)
                                        .template get_as<ADataType>()[number<k>{}]);
            float b_f32 =
                type_convert<float>(reinterpret_cast<const thread_buffer<BDataType, 8>&>(b_vec)
                                        .template get_as<BDataType>()[number<k>{}]);

            c_vec = __builtin_amdgcn_mfma_f32_32x32x2f32(a_f32, b_f32, c_vec, 0, 0, 0);
        });
        return c_vec;
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        return CVecType{0.f};
#endif
    }
};

template <WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
using WarpGemmAttributeMfmaImpl_f32_32x32x16_fp8_fp8 =
    WarpGemmAttributeMfmaImpl_f32_32x32x16_f8_base<fp8_t, fp8_t, Ctrl_>;
template <WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
using WarpGemmAttributeMfmaImpl_f32_16x16x32_fp8_fp8 =
    WarpGemmAttributeMfmaImpl_f32_16x16x32_f8_base<fp8_t, fp8_t, Ctrl_>;
template <WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
using WarpGemmAttributeMfmaImpl_f32_32x32x16_fp8_bf8 =
    WarpGemmAttributeMfmaImpl_f32_32x32x16_f8_base<fp8_t, bf8_t, Ctrl_>;
template <WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
using WarpGemmAttributeMfmaImpl_f32_16x16x32_fp8_bf8 =
    WarpGemmAttributeMfmaImpl_f32_16x16x32_f8_base<fp8_t, bf8_t, Ctrl_>;

template <WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
using WarpGemmAttributeMfmaImpl_f32_16x16x32_bf8_bf8 =
    WarpGemmAttributeMfmaImpl_f32_16x16x32_f8_base<bf8_t, bf8_t, Ctrl_>;

template <WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
using WarpGemmAttributeMfmaImpl_f32_32x32x16_bf8_fp8 =
    WarpGemmAttributeMfmaImpl_f32_32x32x16_f8_base<bf8_t, fp8_t, Ctrl_>;

template <WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
using WarpGemmAttributeMfmaImpl_f32_32x32x16_bf8_bf8 =
    WarpGemmAttributeMfmaImpl_f32_32x32x16_f8_base<bf8_t, bf8_t, Ctrl_>;

template <typename AType_, typename BType_, WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
struct WarpGemmAttributeMfmaImpl_f32_16x16x128_f8f6f4
{
    static constexpr WGAttrCtlEnum Ctrl = Ctrl_;
    using ADataType                     = AType_;
    using BDataType                     = BType_;
    using CDataType                     = float;

    using AVecType = ext_vector_t<ADataType, 32 / numeric_traits<ADataType>::PackedSize>;
    using BVecType = ext_vector_t<BDataType, 32 / numeric_traits<BDataType>::PackedSize>;
    using CVecType = ext_vector_t<CDataType, 4>;

    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 128;

    static constexpr index_t kAMBlock = 1;
    static constexpr index_t kBNBlock = 1;

    static constexpr index_t kAMLane     = 16;
    static constexpr index_t kBNLane     = 16;
    static constexpr index_t kABKLane    = 4;
    static constexpr index_t kABKPerLane = 32;

    static constexpr index_t kCMLane     = 4;
    static constexpr index_t kCNLane     = 16;
    static constexpr index_t kCM0PerLane = 1;
    static constexpr index_t kCM1PerLane = 4;

    // c_vec += a_vec * b_vec
    template <index_t opselA, index_t opselB, bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const int32_t& a_scale,
                                   const BVecType& b_vec,
                                   const int32_t& b_scale,
                                   bool_constant<post_nop_> = {}) const
    {
#if defined(__gfx950__)
        auto dtype2conf = [](auto dtype) {
            if constexpr(std::is_same_v<decltype(dtype), fp8_t>)
                return make_tuple(number<0>{}, int32x8_t{});
            else if constexpr(std::is_same_v<decltype(dtype), bf8_t>)
                return make_tuple(number<1>{}, int32x8_t{});
            // else if e2m3 => make_tuple(number<2>{}, int32x6_t{})
            // else if e3m2 => make_tuple(number<3>{}, int32x6_t{})
            else if constexpr(std::is_same_v<decltype(dtype), pk_fp4_t>)
                return make_tuple(number<4>{}, int32x4_t{});
            else
                static_assert(false, "Unsupported data type for mfma scale");
        };
        auto dtype2code = [&](auto dtype) { return dtype2conf(dtype)(number<0>{}); };
        auto dtype2vec  = [&](auto dtype) { return dtype2conf(dtype)(number<1>{}); };
        auto arg256     = [&](auto x) {
            if constexpr(sizeof(x) == 16)
                return int32x8_t{x[0], x[1], x[2], x[3], 0, 0, 0, 0};
            else if constexpr(sizeof(x) == 24)
                return int32x8_t{x[0], x[1], x[2], x[3], x[4], x[5], 0, 0};
            else if constexpr(sizeof(x) == 32)
                return x;
            else
                static_assert(false, "Unexpected vector size for mfma scale");
        };

        auto arg_a         = bit_cast<decltype(dtype2vec(ADataType{}))>(a_vec);
        auto arg_b         = bit_cast<decltype(dtype2vec(BDataType{}))>(b_vec);
        constexpr int cbsz = decltype(dtype2code(ADataType{}))::value;
        constexpr int blgp = decltype(dtype2code(BDataType{}))::value;
        c_vec              = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            arg256(arg_a), arg256(arg_b), c_vec, cbsz, blgp, opselA, a_scale, opselB, b_scale);
#else
        ck_tile::ignore = c_vec;
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = a_scale;
        ck_tile::ignore = b_scale;
#endif
    }

    // c_vec = a_vec * b_vec
    template <index_t opselA, index_t opselB>
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec,
                                       const int32_t& a_scale,
                                       const BVecType& b_vec,
                                       const int32_t& b_scale) const
    {
        CVecType c_vec{0.f};
        operator()<opselA, opselB>(c_vec, a_vec, a_scale, b_vec, b_scale);
        return c_vec;
    }

    // c_vec += a_vec * b_vec
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        operator()<0, 0>(c_vec, a_vec, 0, b_vec, 0);
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        return operator()<0, 0>(a_vec, 0, b_vec, 0);
    }
};

template <typename AType_, typename BType_, WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
struct WarpGemmAttributeMfmaImpl_f32_32x32x64_f8_bf8_base
{
    static constexpr WGAttrCtlEnum Ctrl = Ctrl_;
    using ADataType                     = AType_;
    using BDataType                     = BType_;
    using CDataType                     = float;

    using AVecType = ext_vector_t<ADataType, 32>;
    using BVecType = ext_vector_t<BDataType, 32>;
    using CVecType = ext_vector_t<CDataType, 16>;

    static constexpr index_t kM = 32;
    static constexpr index_t kN = 32;
    static constexpr index_t kK = 64;

    static constexpr index_t kAMBlock = 1;
    static constexpr index_t kBNBlock = 1;

    static constexpr index_t kAMLane     = 32;
    static constexpr index_t kBNLane     = 32;
    static constexpr index_t kABKLane    = 2;
    static constexpr index_t kABKPerLane = 32;

    static constexpr index_t kCMLane     = 2;
    static constexpr index_t kCNLane     = 32;
    static constexpr index_t kCM0PerLane = 4;
    static constexpr index_t kCM1PerLane = 4;

    // c_vec += a_vec * b_vec
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        //__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a, b, c, cbsz, blgp, opsel, scale_a,
        // opsel, scale_b)
#if defined(__gfx950__)
        if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, fp8_t>)
            c_vec = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                a_vec, b_vec, c_vec, 0, 0, 0, 0, 0, 0);
        else if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, bf8_t>)
            c_vec = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                a_vec, b_vec, c_vec, 0, 1, 0, 0, 0, 0);
        else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, fp8_t>)
            c_vec = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                a_vec, b_vec, c_vec, 1, 0, 0, 0, 0, 0);
        else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, bf8_t>)
            c_vec = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                a_vec, b_vec, c_vec, 1, 1, 0, 0, 0, 0);
#else
        ck_tile::ignore = c_vec;
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
#endif
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx950__)
        if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, fp8_t>)
            return bit_cast<CVecType>(__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                a_vec, b_vec, CVecType{0.f}, 0, 0, 0, 0, 0, 0));
        else if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, bf8_t>)
            return bit_cast<CVecType>(__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                a_vec, b_vec, CVecType{0.f}, 0, 1, 0, 0, 0, 0));
        else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, fp8_t>)
            return bit_cast<CVecType>(__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                a_vec, b_vec, CVecType{0.f}, 1, 0, 0, 0, 0, 0));
        else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, bf8_t>)
            return bit_cast<CVecType>(__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                a_vec, b_vec, CVecType{0.f}, 1, 1, 0, 0, 0, 0));
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        return CVecType{0.f};
#endif
    }
};

template <WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
using WarpGemmAttributeMfmaImpl_f32_32x32x64_fp8_fp8 =
    WarpGemmAttributeMfmaImpl_f32_32x32x64_f8_bf8_base<fp8_t, fp8_t, Ctrl_>;

template <WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
using WarpGemmAttributeMfmaImpl_f32_32x32x64_fp8_bf8 =
    WarpGemmAttributeMfmaImpl_f32_32x32x64_f8_bf8_base<fp8_t, bf8_t, Ctrl_>;

template <WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
using WarpGemmAttributeMfmaImpl_f32_32x32x64_bf8_fp8 =
    WarpGemmAttributeMfmaImpl_f32_32x32x64_f8_bf8_base<bf8_t, fp8_t, Ctrl_>;

template <WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
using WarpGemmAttributeMfmaImpl_f32_32x32x64_bf8_bf8 =
    WarpGemmAttributeMfmaImpl_f32_32x32x64_f8_bf8_base<bf8_t, bf8_t, Ctrl_>;

// int8
template <WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
struct WarpGemmAttributeMfmaImpl_i32_32x32x16_i8
{
    static constexpr WGAttrCtlEnum Ctrl = Ctrl_;
    using ADataType                     = int8_t;
    using BDataType                     = int8_t;
    using CDataType                     = int32_t;

    using AVecType = ext_vector_t<ADataType, 8>;
    using BVecType = ext_vector_t<BDataType, 8>;
    using CVecType = ext_vector_t<CDataType, 16>;

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

    // c_vec += a_vec * b_vec
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        DISPATCH_MFMA_CTRL_("v_mfma_i32_32x32x16_i8", Ctrl)
        else
        {
#if defined(__gfx94__) or defined(__gfx95__)
            c_vec = __builtin_amdgcn_mfma_i32_32x32x16_i8(
                bit_cast<int64_t>(a_vec), bit_cast<int64_t>(b_vec), c_vec, 0, 0, 0);
#elif defined(__gfx908__) || defined(__gfx90a__)
            static_for<0, 8, 1>{}([&](auto k) {
                float a_f32 =
                    type_convert<float>(reinterpret_cast<const thread_buffer<ADataType, 8>&>(a_vec)
                                            .template get_as<ADataType>()[number<k>{}]);
                float b_f32 =
                    type_convert<float>(reinterpret_cast<const thread_buffer<BDataType, 8>&>(b_vec)
                                            .template get_as<BDataType>()[number<k>{}]);

                c_vec = __builtin_amdgcn_mfma_f32_32x32x2f32(a_f32, b_f32, c_vec, 0, 0, 0);
            });
#else
            ck_tile::ignore = c_vec;
            ck_tile::ignore = a_vec;
            ck_tile::ignore = b_vec;
#endif
        }
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        CVecType c_vec{0};
        operator()(c_vec, a_vec, b_vec);
        return c_vec;
    }
};

template <WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
struct WarpGemmAttributeMfmaImpl_i32_16x16x32_i8
{
    static constexpr WGAttrCtlEnum Ctrl = Ctrl_;
    using ADataType                     = int8_t;
    using BDataType                     = int8_t;
    using CDataType                     = int32_t;

    using AVecType = ext_vector_t<ADataType, 8>;
    using BVecType = ext_vector_t<BDataType, 8>;
    using CVecType = ext_vector_t<CDataType, 4>;

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
    static constexpr index_t kCM1PerLane = 4; // write to 4x AccVGPRs

    // c_vec += a_vec * b_vec
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        DISPATCH_MFMA_CTRL_("v_mfma_i32_16x16x32_i8", Ctrl)
        else
        {
#if defined(__gfx94__) or defined(__gfx95__)
            c_vec = __builtin_amdgcn_mfma_i32_16x16x32_i8(
                bit_cast<int64_t>(a_vec), bit_cast<int64_t>(b_vec), c_vec, 0, 0, 0);
#else
            ck_tile::ignore = c_vec;
            ck_tile::ignore = a_vec;
            ck_tile::ignore = b_vec;
#endif
        }
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        CVecType c_vec{0};
        operator()(c_vec, a_vec, b_vec);
        return c_vec;
    }
};

template <WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
struct WarpGemmAttributeMfmaImpl_i32_16x16x64_i8
{
    static constexpr WGAttrCtlEnum Ctrl = Ctrl_;
    using ADataType                     = int8_t;
    using BDataType                     = int8_t;
    using CDataType                     = int32_t;

    using AVecType = ext_vector_t<ADataType, 16>;
    using BVecType = ext_vector_t<BDataType, 16>;
    using CVecType = ext_vector_t<CDataType, 4>;

    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 64;

    static constexpr index_t kAMBlock = 1;
    static constexpr index_t kBNBlock = 1;

    static constexpr index_t kAMLane     = 16;
    static constexpr index_t kBNLane     = 16;
    static constexpr index_t kABKLane    = 4;
    static constexpr index_t kABKPerLane = 16;

    static constexpr index_t kCMLane     = 4;
    static constexpr index_t kCNLane     = 16;
    static constexpr index_t kCM0PerLane = 1;
    static constexpr index_t kCM1PerLane = 4; // write to 4x AccVGPRs

    // c_vec += a_vec * b_vec
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        DISPATCH_MFMA_CTRL_("v_mfma_i32_16x16x64_i8", Ctrl)
        else
        {
#if defined(__gfx95__)
            c_vec = __builtin_amdgcn_mfma_i32_16x16x64_i8(
                bit_cast<int64_t>(a_vec), bit_cast<int64_t>(b_vec), c_vec, 0, 0, 0);
#else
            ck_tile::ignore = c_vec;
            ck_tile::ignore = a_vec;
            ck_tile::ignore = b_vec;
#endif
        }
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        CVecType c_vec{0};
        operator()(c_vec, a_vec, b_vec);
        return c_vec;
    }
};

template <WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
struct WarpGemmAttributeMfmaImpl_i32_32x32x32_i8
{
    static constexpr WGAttrCtlEnum Ctrl = Ctrl_;
    using ADataType                     = int8_t;
    using BDataType                     = int8_t;
    using CDataType                     = int32_t;

    using AVecType = ext_vector_t<ADataType, 16>;
    using BVecType = ext_vector_t<BDataType, 16>;
    using CVecType = ext_vector_t<CDataType, 16>;

    static constexpr index_t kM = 32;
    static constexpr index_t kN = 32;
    static constexpr index_t kK = 32;

    static constexpr index_t kAMBlock = 1;
    static constexpr index_t kBNBlock = 1;

    static constexpr index_t kAMLane     = 32;
    static constexpr index_t kBNLane     = 32;
    static constexpr index_t kABKLane    = 2;
    static constexpr index_t kABKPerLane = 16;

    static constexpr index_t kCMLane     = 2;
    static constexpr index_t kCNLane     = 32;
    static constexpr index_t kCM0PerLane = 4;
    static constexpr index_t kCM1PerLane = 4;

    // c_vec += a_vec * b_vec
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        DISPATCH_MFMA_CTRL_("v_mfma_i32_32x32x32_i8", Ctrl)
        else
        {
#if defined(__gfx95__)
            c_vec = __builtin_amdgcn_mfma_i32_32x32x32_i8(
                a_vec, bit_cast<int64_t>(b_vec), c_vec, 0, 0, 0);
#else
            ck_tile::ignore = c_vec;
            ck_tile::ignore = a_vec;
            ck_tile::ignore = b_vec;
#endif
        }
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        CVecType c_vec{0};
        operator()(c_vec, a_vec, b_vec);
        return c_vec;
    }
};

#undef DISPATCH_MFMA_

} // namespace ck_tile
