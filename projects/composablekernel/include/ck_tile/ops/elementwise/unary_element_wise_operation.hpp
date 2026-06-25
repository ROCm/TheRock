// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include <cstdint>
#include <type_traits>

#define CONSTEXPR_LOOKUP_TABLE_FOR_BF16 1
#define CONSTEXPR_LOOKUP_TABLE_FOR_FP8 0
#define CONSTEXPR_LOOKUP_TABLE_FOR_BF8 0

namespace ck_tile {
namespace element_wise {

// Generalized constexpr lookup table generator
template <typename T, std::size_t N, typename F, std::size_t... Is>
constexpr std::array<T, N> make_lookup_table_impl(F&& func, std::index_sequence<Is...>)
{
    return {func(Is)...};
}

template <typename T, std::size_t N, typename F>
constexpr std::array<T, N> make_lookup_table(F&& func)
{
    return make_lookup_table_impl<T, N>(std::forward<F>(func), std::make_index_sequence<N>{});
}

/**
 * @brief Fast int4x4 to fp16x8_t data type conversion based on paper
 * "Who Says Elephants Can't Run: Bringing Large Scale MoE Models into Cloud Scale Production"
 * @see https://arxiv.org/abs/2211.10017
 * @see
 * https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
 *
 * This function converts 4 4-bit integers into 4 fp16 values.
 * @note `int q` contains 4 bytes, low 4 bits of each byte represent an int4.
 * @note This function assumes pk_int4_t has a bias of 8, meaning 0b0000 is converted to fp16(-8)
 * @note The output ordering differs from input ordering. For example, when input is 0x76543210,
 *       the output sequence will be fp16(7, 3, 6, 2, 5, 1, 4, 0). Therefore, the input tensor
 *       must be preprocessed with permute_vectors_i4x4_b on the host side before using this
 * function.
 *
 * @see permute_vectors_i4x4_b
 */
CK_TILE_DEVICE fp16x4_t i4_to_half4(int q)
{
    const int LO = 0x000f000f;
    const int HI = 0x00f000f0;
    const int EX = 0x64006400;

    int lo;
    int hi;
    // Extract the two int4 at low bit and create two fp16 number.
    asm volatile("v_and_or_b32 %0, %1, %2, %3" : "=v"(lo) : "v"(q), "v"(LO), "v"(EX));
    // Extract the two int4 at hight bit and create two fp16 number.
    asm volatile("v_and_or_b32 %0, %1, %2, %3" : "=v"(hi) : "v"(q), "v"(HI), "v"(EX));

    const int SUB = 0xE408E408; // half2 {-1032, -1032}
    const int MUL = 0x2c002c00; // half2 {1 / 16, 1 / 16}
    const int ADD = 0xd480d480; // half2 {-72, -72}

    fp16x4_t res;

    // for two fp16 from lowbit, subtract 1032 to get correct fp16 value
    asm volatile("v_pk_add_f16 %0, %1, %2"
                 : "=v"(res.lo)
                 : "v"(bit_cast<fp16x2_t>(lo)), "v"(bit_cast<fp16x2_t>(SUB)));

    // for two fp16 from highbit, divide 16 and subtract 72 to get correct fp16 value
    asm volatile(
        "v_pk_fma_f16 %0, %1, %2, %3"
        : "=v"(res.hi)
        : "v"(bit_cast<fp16x2_t>(hi)), "v"(bit_cast<fp16x2_t>(MUL)), "v"(bit_cast<fp16x2_t>(ADD)));

    return res;
}

/**
 * @brief This function dequantizes 4 int4 values into 4 fp16 values and applies scaling.
 *
 * @note `int q` contains 4 bytes, low 4 bits of each byte represent an int4.
 * @note This function assumes pk_int4_t has a bias of 8, meaning 0b0000 is converted to fp16(-8)
 * @note The output ordering differs from input ordering. For example, when input is 0x76543210,
 *       the output sequence will be fp16(7, 3, 6, 2, 5, 1, 4, 0). Therefore, the input tensor
 *       must be preprocessed with permute_vectors_i4x4_b on the host side before using this
 * function.
 *
 * @see permute_vectors_i4x4_b
 */
CK_TILE_DEVICE fp16x4_t i4_to_half4_scale(int q, const fp16x2_t& scale)
{
    const int LO = 0x000f000f;
    const int HI = 0x00f000f0;
    const int EX = 0x64006400;

    int lo;
    int hi;
    // Extract the two int4 at low bit and create two fp16 number.
    asm volatile("v_and_or_b32 %0, %1, %2, %3" : "=v"(lo) : "v"(q), "v"(LO), "v"(EX));
    // Extract the two int4 at hight bit and create two fp16 number.
    asm volatile("v_and_or_b32 %0, %1, %2, %3" : "=v"(hi) : "v"(q), "v"(HI), "v"(EX));

    const int SUB = 0xE408E408; // half2 {-1032, -1032}
    const int MUL = 0x2c002c00; // half2 {1 / 16, 1 / 16}
    const int ADD = 0xd480d480; // half2 {-72, -72}

    fp16x4_t res;

    asm volatile("v_pk_add_f16 %0, %1, %2"
                 : "=v"(res.lo)
                 : "v"(bit_cast<fp16x2_t>(lo)), "v"(bit_cast<fp16x2_t>(SUB)));

    asm volatile(
        "v_pk_fma_f16 %0, %1, %2, %3"
        : "=v"(res.hi)
        : "v"(bit_cast<fp16x2_t>(hi)), "v"(bit_cast<fp16x2_t>(MUL)), "v"(bit_cast<fp16x2_t>(ADD)));

    asm volatile("v_pk_mul_f16 %0, %1, %2" : "=v"(res.lo) : "v"(res.lo), "v"(scale));

    asm volatile("v_pk_mul_f16 %0, %1, %2" : "=v"(res.hi) : "v"(res.hi), "v"(scale));

    return res;
}

/**
 * @brief This function converts 4 4-bit integers into 4 bf16 values.
 *
 * @note `int q` contains 4 bytes, low 4 bits of each byte represent an int4.
 * @note This function assumes pk_int4_t has a bias of 8, meaning 0b0000 is converted to bf16(-8)
 * @note The output ordering differs from input ordering. For example, when input is 0x76543210,
 *       the output sequence will be bf16(7, 3, 6, 2, 5, 1, 4, 0). Therefore, the input tensor
 *       must be preprocessed with permute_vectors_i4x4_b on the host side before using this
 * function.
 *
 * @see permute_vectors_i4x4_b
 */
CK_TILE_DEVICE bf16x4_t i4_to_bhalf4(int q)
{
#if !CONSTEXPR_LOOKUP_TABLE_FOR_BF16
    // This approach fails validation in GEMM tests.
    uint32_t i8s = (q & 0xf) | ((q & 0xf0) << 4) | ((q & 0xf00) << 8) | ((q & 0xf000) << 12);

    static constexpr uint32_t fp32_base = 0x4B000000;

    float fp32_intermediates[4];

    uint32_t* fp32_intermediates_casted = reinterpret_cast<uint32_t*>(fp32_intermediates);

    fp32_intermediates_casted[0] = __byte_perm(i8s, fp32_base, 0x7650);
    fp32_intermediates_casted[1] = __byte_perm(i8s, fp32_base, 0x7651);
    fp32_intermediates_casted[2] = __byte_perm(i8s, fp32_base, 0x7652);
    fp32_intermediates_casted[3] = __byte_perm(i8s, fp32_base, 0x7653);

    fp32_intermediates[0] -= 8388616.f;
    fp32_intermediates[1] -= 8388616.f;
    fp32_intermediates[2] -= 8388616.f;
    fp32_intermediates[3] -= 8388616.f;

    bf16x4_t res;
    res.lo = bit_cast<bf16x2_t>(
        __byte_perm(fp32_intermediates_casted[1], fp32_intermediates_casted[0], 0x7632));
    res.hi = bit_cast<bf16x2_t>(
        __byte_perm(fp32_intermediates_casted[3], fp32_intermediates_casted[2], 0x7632));

    return res;
#else
    // Lookup table for bf16_t values corresponding to int4 values -8 to 7
    constexpr auto bf16_lookup_table = make_lookup_table<bf16_t, 16>(
        [](int i) { return bit_cast<bf16_t>(float_to_bf16_rtn_raw(i - 8)); });

    return bf16x4_t{bf16_lookup_table[(q >> 0) & 0xf],
                    bf16_lookup_table[(q >> 16) & 0xf],
                    bf16_lookup_table[(q >> 4) & 0xf],
                    bf16_lookup_table[(q >> 20) & 0xf]};
#endif
}

#if !CONSTEXPR_LOOKUP_TABLE_FOR_FP8
/**
 * @brief This function converts 8 packed 4-bit integers into 8 fp8 values.
 *
 * @note `int q` contains 4 bytes, each byte represents 2 int4.
 * @note This function assumes pk_int4_t has a bias of 8, meaning 0b0000 is converted to fp8(-8)
 * @note The output ordering differs from input ordering. For example, when input is 0x76543210,
 *       the output sequence will be fp8(7, 3, 6, 2, 5, 1, 4, 0). Therefore, the input tensor
 *       must be preprocessed with permute_vectors_i4x4_b on the host side before using this
 * function.
 *
 * @see permute_vectors_i4x4_b
 */
CK_TILE_DEVICE fp8x8_t amd_assembly_i4_to_fp8x8(int a)
{
#if CK_TILE_USE_OCP_FP8
    // register values [3, 2, 1, 0]
    static constexpr uint32_t reg0 = 0xcaccced0;
    // register values [7, 6, 5, 4]
    static constexpr uint32_t reg1 = 0xb8c0c4c8;
    // register values [-1, -2, -3, -4]
    static constexpr uint32_t reg2 = 0x44403800;
    // register values [-5, -6, -7, -8]
    static constexpr uint32_t reg3 = 0x4e4c4a48;
#else
    // register values [3, 2, 1, 0]
    static constexpr uint32_t reg0 = 0xd2d4d6d8;
    // register values [7, 6, 5, 4]
    static constexpr uint32_t reg1 = 0xc0c8ccd0;
    // register values [-1, -2, -3, -4]
    static constexpr uint32_t reg2 = 0x4C484000;
    // register values [-5, -6, -7, -8]
    static constexpr uint32_t reg3 = 0x56545250;
#endif

    uint32_t tmp_pos, tmp_neg, tmp_res_even, tmp_res_odd, final_sel;

    // ---- Lower 4 int4 values (even positions) ----
    // Extract dictionary indices: low 3 bits of each byte (values 0..7).
    uint32_t dict_sel = a & 0x07070707;
    // sign bit is bit[2] of each nibble after bias; shift to isolate per-byte sign.
    uint32_t sign = a >> 1;
    // Build final selector:
    //   - bit 2 of each byte (0x04) selects negative vs positive table
    //   - 0x03020100 selects byte lanes [0,1,2,3] in order
    final_sel = (sign & 0x04040404) | 0x03020100;
    // Lookup positive and negative fp8 codes from the small register tables.
    tmp_pos = __builtin_amdgcn_perm(reg1, reg0, dict_sel);
    tmp_neg = __builtin_amdgcn_perm(reg3, reg2, dict_sel);
    // Select per-lane between tmp_pos and tmp_neg using the sign-derived selector.
    tmp_res_even = __builtin_amdgcn_perm(tmp_neg, tmp_pos, final_sel);

    // ---- Upper 4 int4 values (odd positions) ----
    // Shift to bring the high-nibble int4s into place and repeat the process.
    a >>= 4;
    dict_sel  = a & 0x07070707;
    sign      = a >> 1;
    final_sel = (sign & 0x04040404) | 0x03020100;

    tmp_pos           = __builtin_amdgcn_perm(reg1, reg0, dict_sel);
    tmp_neg           = __builtin_amdgcn_perm(reg3, reg2, dict_sel);
    tmp_res_odd       = __builtin_amdgcn_perm(tmp_neg, tmp_pos, final_sel);
    auto tmp_res_low  = __builtin_amdgcn_perm(tmp_res_odd, tmp_res_even, 0x06040200);
    auto tmp_res_high = __builtin_amdgcn_perm(tmp_res_odd, tmp_res_even, 0x07050301);

    return bit_cast<fp8x8_t>((static_cast<uint64_t>(tmp_res_high) << 32) | tmp_res_low);
}
#else
CK_TILE_DEVICE fp8x4_t i4_to_fp8x4(int q)
{
    // The approach below can be used once this compiler issue is resolved:
    // "constexpr bit cast involving type 'unsigned _BitInt(8)' is not yet supported"
    // Lookup table for fp8_t values corresponding to int4 values -8 to 7
    constexpr auto fp8_lookup_table = make_lookup_table<fp8_t, 16>(
        [](int i) { return impl::cast_to_f8<float, fp8_t, true, false>(i - 8, 0); });

    return fp8x4_t{fp8_lookup_table[(q >> 0) & 0xf],
                   fp8_lookup_table[(q >> 16) & 0xf],
                   fp8_lookup_table[(q >> 4) & 0xf],
                   fp8_lookup_table[(q >> 20) & 0xf]};
}
#endif

CK_TILE_DEVICE float amd_assembly_fp8_to_fp32(uint32_t src)
{
    float res;
    asm volatile("v_cvt_f32_fp8 %0, %1, src0_sel:BYTE_0" : "=v"(res) : "v"(src));
    return res;
}

CK_TILE_DEVICE float amd_assembly_bf8_to_fp32(uint32_t src)
{
    float res;
    asm volatile("v_cvt_f32_bf8 %0, %1, src0_sel:BYTE_0" : "=v"(res) : "v"(src));
    return res;
}

#if !CONSTEXPR_LOOKUP_TABLE_FOR_BF8
/**
 * @brief This function converts 8 packed 4-bit integers into 8 bf8 values.
 *
 * @note `int q` contains 4 bytes, each byte represents 2 int4.
 * @note This function assumes pk_int4_t has a bias of 8, meaning 0b0000 is converted to bf8(-8)
 * @note The output ordering differs from input ordering. For example, when input is 0x76543210,
 *       the output sequence will be bf8(7, 3, 6, 2, 5, 1, 4, 0). Therefore, the input tensor
 *       must be preprocessed with permute_vectors_i4x4_b on the host side before using this
 * function.
 *
 * @see permute_vectors_i4x4_b
 */
CK_TILE_DEVICE bf8x8_t amd_assembly_i4_to_bf8x8(uint32_t a)
{
#if CK_TILE_USE_OCP_FP8
    // register values [3, 2, 1, 0]
    static constexpr uint32_t reg0 = 0Xc5c6c7c8;
    // register values [7, 6, 5, 4]
    static constexpr uint32_t reg1 = 0Xbcc0c2c4;
    // register values [11, 10, 9, 8]
    static constexpr uint32_t reg2 = 0X42403c00;
    // register values [15, 14, 13, 12]
    static constexpr uint32_t reg3 = 0X47464544;
#else
    // register values [3, 2, 1, 0]
    static constexpr uint32_t reg0 = 0Xc9cacbcc;
    // register values [7, 6, 5, 4]
    static constexpr uint32_t reg1 = 0Xc0c4c6c8;
    // register values [11, 10, 9, 8]
    static constexpr uint32_t reg2 = 0X46444000;
    // register values [15, 14, 13, 12]
    static constexpr uint32_t reg3 = 0X4b4a4948;
#endif

    uint32_t tmp_pos, tmp_neg, tmp_res_even, tmp_res_odd, final_sel;

    // ---- Lower 4 int4 values (even positions) ----
    // Extract dictionary indices: low 3 bits of each byte (values 0..7).
    uint32_t dict_sel = a & 0x07070707;

    // sign bit is bit[2] of each nibble after bias; shift to isolate per-byte sign.
    uint32_t sign = a >> 1;
    // Build final selector:
    //   - bit 2 of each byte (0x04) selects negative vs positive table
    //   - 0x03020100 selects byte lanes [0,1,2,3] in order
    final_sel = (sign & 0x04040404) | 0x03020100;

    // Lookup positive and negative fp8 codes from the small register tables.
    tmp_pos = __builtin_amdgcn_perm(reg1, reg0, dict_sel);
    tmp_neg = __builtin_amdgcn_perm(reg3, reg2, dict_sel);
    // Select per-lane between tmp_pos and tmp_neg using the sign-derived selector.
    tmp_res_even = __builtin_amdgcn_perm(tmp_neg, tmp_pos, final_sel);

    // ---- Upper 4 int4 values (odd positions) ----
    // Shift to bring the high-nibble int4s into place and repeat the process.
    a >>= 4;
    dict_sel  = a & 0x07070707;
    sign      = a >> 1;
    final_sel = (sign & 0x04040404) | 0x03020100;

    tmp_pos           = __builtin_amdgcn_perm(reg1, reg0, dict_sel);
    tmp_neg           = __builtin_amdgcn_perm(reg3, reg2, dict_sel);
    tmp_res_odd       = __builtin_amdgcn_perm(tmp_neg, tmp_pos, final_sel);
    auto tmp_res_low  = __builtin_amdgcn_perm(tmp_res_odd, tmp_res_even, 0x06040200);
    auto tmp_res_high = __builtin_amdgcn_perm(tmp_res_odd, tmp_res_even, 0x07050301);

    return bit_cast<bf8x8_t>((static_cast<uint64_t>(tmp_res_high) << 32) | tmp_res_low);
}
#else
CK_TILE_DEVICE bf8x4_t i4_to_bf8x4(int q)
{
    // The approach below can be used once this compiler issue is resolved:
    // "constexpr bit cast involving type 'unsigned _BitInt(8)' is not yet supported"
    // Lookup table for bf8_t values corresponding to int4 values -8 to 7
    constexpr auto bf8_lookup_table = make_lookup_table<bf8_t, 16>(
        [](int i) { return impl::cast_to_f8<float, bf8_t, true, false>(i - 8, 0); });

    return bf8x4_t{bf8_lookup_table[(q >> 0) & 0xf],
                   bf8_lookup_table[(q >> 16) & 0xf],
                   bf8_lookup_table[(q >> 4) & 0xf],
                   bf8_lookup_table[(q >> 20) & 0xf]};
}
#endif

struct PassThroughPack8
{
    static constexpr const char* name = "PassThroughPack8";

    template <typename Y, typename X>
    CK_TILE_HOST_DEVICE void operator()(Y& y, const X& x) const;

    CK_TILE_HOST_DEVICE constexpr void operator()(fp16x8_t& y, const pk_int4x4_t& x) const
    {
        y.lo = i4_to_half4(bit_cast<int>(x));
        y.hi = i4_to_half4(bit_cast<int>(x) >> 8);
    }

    CK_TILE_HOST_DEVICE constexpr void operator()(bf16x8_t& y, const pk_int4x4_t& x) const
    {
        y.lo = i4_to_bhalf4(bit_cast<int>(x));
        y.hi = i4_to_bhalf4(bit_cast<int>(x) >> 8);
    }

    CK_TILE_HOST_DEVICE constexpr void operator()(fp8x8_t& y, const pk_int4x4_t& x) const
    {
#if !CONSTEXPR_LOOKUP_TABLE_FOR_FP8
        y = amd_assembly_i4_to_fp8x8(bit_cast<uint32_t>(x));
#else
        y.lo = i4_to_fp8x4(bit_cast<int>(x));
        y.hi = i4_to_fp8x4(bit_cast<int>(x) >> 8);
#endif
    }

    CK_TILE_HOST_DEVICE constexpr void operator()(bf8x8_t& y, const pk_int4x4_t& x) const
    {
#if !CONSTEXPR_LOOKUP_TABLE_FOR_BF8
        y = amd_assembly_i4_to_bf8x8(bit_cast<uint32_t>(x));
#else
        y.lo = i4_to_bf8x4(bit_cast<int>(x));
        y.hi = i4_to_bf8x4(bit_cast<int>(x) >> 8);
#endif
    }
    constexpr const static bool is_pack8_invocable = true;
};

struct DequantPack8
{
    static constexpr const char* name = "DequantPack8";

    template <typename Y, typename X, typename Z>
    CK_TILE_HOST_DEVICE void operator()(Y& y, const X& x, const Z& z) const;

    CK_TILE_HOST_DEVICE constexpr void
    operator()(fp16x8_t& y, const pk_int4x4_t& x, const fp16x2_t& z) const
    {
        y.lo = i4_to_half4_scale(bit_cast<int>(x), z);
        y.hi = i4_to_half4_scale(bit_cast<int>(x) >> 8, z);
    }

    constexpr const static bool is_pack8_invocable = true;
};

struct PassThroughPack2
{
    static constexpr const char* name = "PassThroughPack2";

    template <typename Y, typename X>
    CK_TILE_HOST_DEVICE void operator()(Y& y, const X& x) const;

#if 0
    CK_TILE_HOST_DEVICE constexpr void operator()(ck_tile::fp16x2_t& y, const ck_tile::f8x2_t& x) const
    {
        auto t = type_convert<float2_t>(x);
        y      = type_convert<fp16x2_t>(t);
    }
#endif

    CK_TILE_HOST_DEVICE constexpr void operator()(fp16x2_t& y, const pk_int4_t& x) const
    {
        uint8_t x_u8 = bit_cast<uint8_t>(x);
        uint8_t x_l  = (x_u8 & 0x0f) >> 0;
        uint8_t x_h  = (x_u8 & 0xf0) >> 4;

        y.lo = type_convert<half_t>(x_l);
        y.hi = type_convert<half_t>(x_h);
    }

    constexpr const static bool is_pack2_invocable = true;
};

struct PassThrough
{
    static constexpr const char* name = "PassThrough";

    template <class T>
    using raw_t = std::remove_cv_t<std::remove_reference_t<T>>;

    template <class Y, class X>
    CK_TILE_HOST_DEVICE void operator()(Y&& y, const X& x) const
    {
        /*  Only do the assignment when
            - y is an *l-value*   and
            - y is *not* const     */
        if constexpr(std::is_lvalue_reference_v<Y&&> && !std::is_const_v<raw_t<Y>>)
        {
            y = ck_tile::type_convert<raw_t<Y>>(x);
        }
        /*  otherwise (r-value or const)     → do nothing  */
    }

    template <typename Y, typename X>
    CK_TILE_HOST_DEVICE void operator()(Y& y, const X& x) const
    {
        y = ck_tile::type_convert<raw_t<Y>>(x);
    }

    template <typename E, typename C, typename... Ds>
    CK_TILE_HOST_DEVICE auto operator()(E& e, const C& c, const Ds&...) const -> void
    {
        // Just assign e with c
        if constexpr(std::is_same_v<E, C>)
        {
            e = c;
        }
        else
        {
            e = ck_tile::type_convert<E>(c);
        }
    }
};

struct AddScale
{
    static constexpr const char* name = "AddScale";

    template <typename E, typename... As>
    CK_TILE_HOST_DEVICE constexpr void operator()(E& a, const As&... as) const
    {
        // Start with the base value c
        float result = ck_tile::type_convert<float>(0.0f);

        // Add by each D parameter using fold expression
        ((result += ck_tile::type_convert<float>(as)), ...);

        a = ck_tile::type_convert<E>(scale * result);
    }

    float scale = 1.0;
};

struct MultiDMultiply
{
    static constexpr const char* name = "MultiDMultiply";

    template <typename E, typename C, typename... Ds>
    CK_TILE_HOST_DEVICE auto operator()(E& e, const C& c, const Ds&... ds) const -> void
    {
        // Start with the base value c
        float result = ck_tile::type_convert<float>(c);

        // Multiply by each D parameter using fold expression
        ((result *= ck_tile::type_convert<float>(ds)), ...);

        e = ck_tile::type_convert<E>(result);
    }
};

struct MultiDAdd
{
    static constexpr const char* name = "MultiDAdd";

    template <typename E, typename C, typename... Ds>
    CK_TILE_HOST_DEVICE auto operator()(E& e, const C& c, const Ds&... ds) const -> void
    {
        // Start with the base value c
        float result = ck_tile::type_convert<float>(c);

        // Add by each D parameter using fold expression
        ((result += ck_tile::type_convert<float>(ds)), ...);

        e = ck_tile::type_convert<E>(result);
    }
};

struct UnaryConvert
{
    static constexpr const char* name = "UnaryConvert";

    template <typename Y, typename X>
    CK_TILE_HOST_DEVICE void operator()(Y& y, const X& x) const
    {
        y = type_convert<Y>(x);
    }
};

#if 0
struct ConvertBF16RTN
{
    // convert to bf16 using round to nearest (rtn)
    template <typename Y, typename X>
    CK_TILE_HOST_DEVICE void operator()(Y& y, const X& x) const
    {
        // check Y datatype
        static_assert(std::is_same_v<Y, ck_tile::bf16_t>, "Data type is not supported by this operation!");

        // check X datatype
        static_assert(std::is_same_v<X, float> || std::is_same_v<X, ck_tile::fp16_t>,
                      "Data type is not supported by this operation!");

        y = bf16_convert_rtn<Y>(x);
    }
};

struct ConvertF8SR
{
    // convert to fp8 using stochastic rounding (SR)
    template <typename Y, typename X>
    CK_TILE_HOST_DEVICE void operator()(Y& y, const X& x) const
    {
        // check Y datatype
        static_assert(std::is_same_v<Y, ck_tile::fp8_t> || std::is_same_v<Y, ck_tile::bf8_t>,
                      "Data type is not supported by this operation!");

        // check X datatype
        static_assert(std::is_same_v<X, float> || std::is_same_v<X, ck_tile::fp16_t>,
                      "Data type is not supported by this operation!");

        y = f8_convert_sr<Y>(x);
    }
};

struct ConvertF8RNE
{
    // convert to fp8 using rounding to nearest even
    template <typename Y, typename X>
    CK_TILE_HOST_DEVICE void operator()(Y& y, const X& x) const
    {
        // check Y datatype
        static_assert(std::is_same_v<Y, ck_tile::fp8_t> || std::is_same_v<Y, ck_tile::bf8_t>,
                      "Data type is not supported by this operation!");

        // check X datatype
        static_assert(std::is_same_v<X, float> || std::is_same_v<X, ck_tile::fp16_t>,
                      "Data type is not supported by this operation!");

        y = f8_convert_rne<Y>(x);
    }
};
#endif

struct Scale
{
    static constexpr const char* name = "Scale";

    CK_TILE_HOST_DEVICE Scale(float scale = 1.f) : scale_(scale) {}

    template <typename Y, typename X>
    CK_TILE_HOST_DEVICE void operator()(Y& y, const X& x) const
    {
        y = ck_tile::type_convert<Y>(ck_tile::type_convert<float>(x) * scale_);
    }

    template <>
    CK_TILE_HOST_DEVICE void
    operator()<ck_tile::fp16_t, ck_tile::fp16_t>(ck_tile::fp16_t& y, const ck_tile::fp16_t& x) const
    {
        y = ck_tile::type_convert<ck_tile::fp16_t>(scale_) * x;
    };

    template <>
    CK_TILE_HOST_DEVICE void
    operator()<ck_tile::bf16_t, ck_tile::bf16_t>(ck_tile::bf16_t& y, const ck_tile::bf16_t& x) const
    {
        const float x_tmp = ck_tile::type_convert<float>(x);
        const float y_tmp = scale_ * x_tmp;
        y                 = ck_tile::type_convert<ck_tile::bf16_t>(y_tmp);
    };

    template <>
    CK_TILE_HOST_DEVICE void operator()<float, float>(float& y, const float& x) const
    {
        y = scale_ * x;
    };

    template <>
    CK_TILE_HOST_DEVICE void operator()<double, double>(double& y, const double& x) const
    {
        y = scale_ * x;
    };

    template <>
    CK_TILE_HOST_DEVICE void operator()<int8_t, int8_t>(int8_t& y, const int8_t& x) const
    {
        y = ck_tile::type_convert<int8_t>(scale_ * ck_tile::type_convert<float>(x));
    };

    float scale_;
};

struct ScaleAndResetNaNToMinusInfinity
{
    static constexpr const char* name = "ScaleAndResetNaNToMinusInfinity";

    CK_TILE_HOST_DEVICE ScaleAndResetNaNToMinusInfinity(float scale) : scale_(scale) {}

    template <typename Y, typename X>
    CK_TILE_HOST_DEVICE void operator()(Y& y, const X& x) const;

    template <>
    CK_TILE_HOST_DEVICE void operator()<float, float>(float& y, const float& x) const
    {
        y = ck_tile::isnan(x) ? -numeric<float>::infinity() : scale_ * x;
    };

    float scale_;
};

struct UnaryDivide
{
    static constexpr const char* name = "UnaryDivide";

    CK_TILE_HOST_DEVICE UnaryDivide(const int32_t divider = 1) : divider_(divider) {}

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        y = x / type_convert<T>(divider_);
    };

    int32_t divider_ = 1;
};

struct UnarySquare
{
    static constexpr const char* name = "UnarySquare";

    template <typename Y, typename X>
    CK_TILE_HOST_DEVICE void operator()(Y& y, const X& x) const
    {
        static_assert(std::is_same_v<X, float> || std::is_same_v<X, ck_tile::fp16_t> ||
                          std::is_same_v<X, double> || std::is_same_v<X, int32_t> ||
                          std::is_same_v<X, int8_t>
#ifdef CK_TILE_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
                          || std::is_same_v<X, int4_t>
#endif
                      ,
                      "Data type is not supported by this operation!");
        y = x * x;
    };
};

struct UnaryAbs
{
    static constexpr const char* name = "UnaryAbs";

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int32_t> ||
                          std::is_same_v<T, int8_t>,
                      "Data type is not supported by this operation!");

        y = ck_tile::abs(x);
    };
};

struct UnarySqrt
{
    static constexpr const char* name = "UnarySqrt";

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                      "Data type is not supported by this operation!");

        y = ck_tile::sqrt(x);
    };
};

struct Relu
{
    static constexpr const char* name = "Relu";

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int32_t> ||
                          std::is_same_v<T, int8_t>,
                      "Data type is not supported by this operation!");
        y = x > 0 ? x : 0;
    }

    template <>
    CK_TILE_HOST_DEVICE void operator()(ck_tile::bf16_t& y, const ck_tile::bf16_t& x) const
    {
        float x_f32 = ck_tile::type_convert<float>(x);
        float y_f32 = x_f32 > 0 ? x_f32 : 0;
        y           = ck_tile::type_convert<ck_tile::bf16_t>(y_f32);
    }
};

// Fast GeLU
// https://paperswithcode.com/method/gelu
// y = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
// host code use higher accuracy "exp" and "div"
// gpu code use lower accuracy "_ocml_exp_f32" and "rcp" function
struct FastGelu
{
    static constexpr const char* name = "FastGelu";

    template <typename Y, typename X>
    CK_TILE_HOST void operator()(Y& y, const X& x) const;

    template <typename Y, typename X>
    CK_TILE_DEVICE void operator()(Y& y, const X& x) const;

    template <>
    CK_TILE_HOST void operator()<float, float>(float& y, const float& x) const
    {
        // const float u   = -2.f * x * (0.035677f * x * x + 0.797885f);
        const float c1  = -2.0 * 0.035677f;
        const float c2  = -2.0 * 0.797885f;
        const float u   = x * (c1 * x * x + c2);
        const float emu = exp(u);
        y               = x / (1.f + emu);
    }

    // device code, use lower precision "__ocml_exp_f32" and "rcp"
    template <>
    CK_TILE_DEVICE void operator()<float, float>(float& y, const float& x) const
    {
        // const float u   = 2.f * x * (0.035677f * x * x + 0.797885f);
        const float c1  = -2.0 * 0.035677f;
        const float c2  = -2.0 * 0.797885f;
        const float u   = x * (c1 * x * x + c2);
        const float emu = __ocml_exp_f32(u);

        y = x * ck_tile::rcp(1.f + emu);
    }

    template <>
    CK_TILE_HOST void operator()<ck_tile::fp16_t, ck_tile::fp16_t>(ck_tile::fp16_t& y,
                                                                   const ck_tile::fp16_t& x) const
    {
        float y_f;

        this->operator()<float, float>(y_f, type_convert<float>(x));

        y = type_convert<ck_tile::fp16_t>(y_f);
    }

    template <>
    CK_TILE_DEVICE void operator()<ck_tile::fp16_t, ck_tile::fp16_t>(ck_tile::fp16_t& y,
                                                                     const ck_tile::fp16_t& x) const
    {
        float y_f;

        this->operator()<float, float>(y_f, type_convert<float>(x));

        y = type_convert<ck_tile::fp16_t>(y_f);
    }

    template <>
    CK_TILE_HOST void operator()<ck_tile::fp16_t, float>(ck_tile::fp16_t& y, const float& x) const
    {
        float y_f;

        this->operator()<float, float>(y_f, x);

        y = type_convert<ck_tile::fp16_t>(y_f);
    }

    template <>
    CK_TILE_DEVICE void operator()<ck_tile::fp16_t, float>(ck_tile::fp16_t& y, const float& x) const
    {
        float y_f;

        this->operator()<float, float>(y_f, x);

        y = type_convert<ck_tile::fp16_t>(y_f);
    }

    template <>
    CK_TILE_HOST void operator()<ck_tile::bf16_t, float>(ck_tile::bf16_t& y, const float& x) const
    {
        float y_f;

        this->operator()<float, float>(y_f, x);

        y = type_convert<ck_tile::bf16_t>(y_f);
    }

    template <>
    CK_TILE_DEVICE void operator()<ck_tile::bf16_t, float>(ck_tile::bf16_t& y, const float& x) const
    {
        float y_f;

        this->operator()<float, float>(y_f, x);

        y = type_convert<ck_tile::bf16_t>(y_f);
    }

    template <>
    CK_TILE_DEVICE void operator()<ck_tile::bf16_t, ck_tile::bf16_t>(ck_tile::bf16_t& y,
                                                                     const ck_tile::bf16_t& x) const
    {
        float y_f;

        this->operator()<float, float>(y_f, type_convert<float>(x));

        y = type_convert<ck_tile::bf16_t>(y_f);
    }

    template <>
    CK_TILE_HOST void operator()<ck_tile::bf16_t, ck_tile::bf16_t>(ck_tile::bf16_t& y,
                                                                   const ck_tile::bf16_t& x) const
    {
        float y_f;

        this->operator()<float, float>(y_f, type_convert<float>(x));

        y = type_convert<ck_tile::bf16_t>(y_f);
    }
};

struct FastGeluAsm
{
    static constexpr const char* name = "FastGeluAsm";

    template <typename Y, typename X>
    CK_TILE_HOST void operator()(Y& y, const X& x) const;

    template <typename Y, typename X>
    CK_TILE_DEVICE void operator()(Y& y, const X& x) const;

    template <>
    CK_TILE_HOST void operator()<float, float>(float& y, const float& x) const
    {
        // const float u   = -2.f * x * (0.035677f * x * x + 0.797885f);
        const float c1  = -2.0 * 0.035677f;
        const float c2  = -2.0 * 0.797885f;
        const float u   = x * (c1 * x * x + c2);
        const float emu = exp(u);
        y               = x / (1.f + emu);
    }

    // device code, use lower precision "__ocml_exp_f32" and "rcp"
    template <>
    CK_TILE_DEVICE void operator()<float, float>(float& y, const float& x) const
    {
        const uint32_t c1     = 0xbd92220c; // -2.0 * 0.035677f;
        const float c2        = -2.0 * 0.797885f;
        const uint32_t log2e_ = 0x3fb8aa3b; // log2e_v<float>;
        float tmp;

        asm volatile("v_mul_f32 %[v_tmp], %[v_x], %[v_x]        ; x*x\n"
                     "v_fma_f32 %[v_tmp], %[v_tmp], %[s_c1], %[v_c2]  ; c1*x*x+c2\n"
                     "v_mul_f32 %[v_tmp], %[v_tmp], %[v_x]      ; x*(c1*x*x+c2)\n"
                     "v_mul_f32 %[v_tmp], %[v_tmp], %[s_log2e]  ; log2e*x*(c1*x*x+c2)\n"
                     "v_exp_f32 %[v_tmp], %[v_tmp]              ; emu = exp2(log2e*x*(c1*x*x+c2))\n"
                     "s_nop 0                                   ; hazard for exp\n"
                     "v_add_f32 %[v_tmp], %[v_tmp], 1.0         ; emu+1.0f\n"
                     "v_rcp_f32 %[v_tmp], %[v_tmp]              ; 1/(emu+1.0f)\n"
                     "s_nop 0                                   ; hazard for rcp \n"
                     "v_mul_f32 %[v_y], %[v_tmp], %[v_x]        ; x * 1/(emu+1f)\n"
                     : [v_y] "=v"(y), [v_tmp] "+v"(tmp)
                     : [v_x] "v"(x), [s_c1] "s"(c1), [v_c2] "v"(c2), [s_log2e] "s"(log2e_)
                     :);
    }

    template <>
    CK_TILE_HOST void operator()<fp32x2_t, fp32x2_t>(fp32x2_t& y, const fp32x2_t& x) const
    {
        const float c1   = -2.0 * 0.035677f;
        const float c2   = -2.0 * 0.797885f;
        const float u0   = x.x * (c1 * x.x * x.x + c2);
        const float emu0 = exp(u0);
        y.x              = x.x / (1.f + emu0);
        const float u1   = x.y * (c1 * x.y * x.y + c2);
        const float emu1 = exp(u1);
        y.y              = x.y / (1.f + emu1);
    }

    // this is packed verion to remove data hazard for trans
    template <>
    CK_TILE_DEVICE void operator()<fp32x2_t, fp32x2_t>(fp32x2_t& y, const fp32x2_t& x) const
    {
        const uint32_t c1     = 0xbd92220c; // -2.0 * 0.035677f;
        float c2              = -2.0 * 0.797885f;
        const uint32_t log2e_ = 0x3fb8aa3b; // log2e_v<float>;
        float tmp0, tmp1;
        float y0 = x.x, y1 = x.y;

        asm volatile(
            "v_mul_f32 %[v_tmp0], %[v_y0], %[v_y0]        ; x*x\n"
            "v_mul_f32 %[v_tmp1], %[v_y1], %[v_y1]        ; x*x\n"
            "v_fma_f32 %[v_tmp0], %[v_tmp0], %[s_c1], %[v_c2]  ; c1*x*x+c2\n"
            "v_fma_f32 %[v_tmp1], %[v_tmp1], %[s_c1], %[v_c2]  ; c1*x*x+c2\n"
            "v_mul_f32 %[v_tmp0], %[v_tmp0], %[v_y0]      ; x*(c1*x*x+c2)\n"
            "v_mul_f32 %[v_tmp1], %[v_tmp1], %[v_y1]      ; x*(c1*x*x+c2)\n"
            "v_mul_f32 %[v_tmp0], %[v_tmp0], %[s_log2e]  ; log2e*x*(c1*x*x+c2)\n"
            "v_mul_f32 %[v_tmp1], %[v_tmp1], %[s_log2e]  ; log2e*x*(c1*x*x+c2)\n"
            "v_exp_f32 %[v_tmp0], %[v_tmp0]              ; emu = exp2(log2e*x*(c1*x*x+c2))\n"
            "v_exp_f32 %[v_tmp1], %[v_tmp1]              ; emu = exp2(log2e*x*(c1*x*x+c2))\n"
            "v_add_f32 %[v_tmp0], %[v_tmp0], 1.0         ; emu+1.0f\n"
            "v_add_f32 %[v_tmp1], %[v_tmp1], 1.0         ; emu+1.0f\n"
            "v_rcp_f32 %[v_tmp0], %[v_tmp0]              ; 1/(emu+1.0f)\n"
            "v_rcp_f32 %[v_tmp1], %[v_tmp1]              ; 1/(emu+1.0f)\n"
            "v_mul_f32 %[v_y0], %[v_tmp0], %[v_y0]        ; x * 1/(emu+1f)\n"
            "v_mul_f32 %[v_y1], %[v_tmp1], %[v_y1]        ; x * 1/(emu+1f)\n"
            : [v_y0] "+v"(y0),
              [v_y1] "+v"(y1),
              [v_c2] "+v"(c2),
              // NOTE! it is totally possible that c2/y0/y1 share same register, they are all local
              // tmp variables we need to expicitly hint compiler they may read+write, to allow
              // allocate different register , the side effect is c2=** may issue for every such
              // inline asm block
              [v_tmp0] "+v"(tmp0),
              [v_tmp1] "+v"(tmp1)
            : [s_c1] "s"(c1), [s_log2e] "s"(log2e_)
            :);
        y.x = y0;
        y.y = y1;
    }
};

// https://paperswithcode.com/method/gelu
// y = 0.5*x*(1+erf(x/sqrt(2)))
struct Gelu
{
    static constexpr const char* name = "Gelu";

    template <typename Y, typename X>
    CK_TILE_HOST_DEVICE void operator()(Y& y, const X& x) const;

    template <>
    CK_TILE_HOST_DEVICE void operator()<float, float>(float& y, const float& x) const
    {
        y = 0.5f * x * (1.f + erf(float(0.70710678118f * x)));
    }

    template <>
    CK_TILE_HOST_DEVICE void
    operator()<ck_tile::fp16_t, ck_tile::fp16_t>(ck_tile::fp16_t& y, const ck_tile::fp16_t& x) const
    {
        y = ck_tile::fp16_t(0.5) * x *
            (ck_tile::fp16_t(1) + ck_tile::fp16_t(erf(float(0.70710678118f * x))));
    }
};

struct Sigmoid
{
    static constexpr const char* name = "Sigmoid";

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");
        constexpr T one = type_convert<T>(1);
        y               = one / (one + ck_tile::exp(-x));
    };
};

struct Silu
{
    static constexpr const char* name = "Silu";

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");
        constexpr T one = type_convert<T>(1);
        y               = x * (one / (one + ck_tile::exp(-x)));
    };

    template <>
    CK_TILE_HOST_DEVICE void operator()<fp32x2_t>(fp32x2_t& y, const fp32x2_t& x) const
    {
        constexpr auto one = type_convert<float>(1);
        y[0]               = x[0] * __builtin_amdgcn_rcpf(one + ck_tile::exp(-x[0]));
        y[1]               = x[1] * __builtin_amdgcn_rcpf(one + ck_tile::exp(-x[1]));
    };
};

#if 0
// Silu, the formular is not so good to do inline asm (dependency)
// we put the code here purposely if in the future ppl want to try
struct SiluAsm
{
    template <typename T>
    CK_TILE_HOST void operator()(T& y, T& x) const
    {
        static_assert(std::is_same_v<T, float>, "Data type is not supported by this operation!");
        constexpr T one = type_convert<T>(1);
        y               = x * (one / (one + ck_tile::exp(-x)));
    };

    template <typename T>
    CK_TILE_DEVICE void operator()(T& y, T& x) const
    {
        static_assert(std::is_same_v<T, float>, "Data type is not supported by this operation!");

        const uint32_t log2e_neg_ = 0x3fb8aa3b | 0x80000000; // log2e_v<float> * -1;

        // NOTE: x/y can't be same register before inline asm
        // "+v" as y, "v" as x is not enought, x/y stil maybe put to same register
        T tmp = x;
        asm volatile("v_mul_f32 %[v_y], %[s_log2e], %[v_x]\n"
                     "v_exp_f32 %[v_y], %[v_y]\n"
                     "s_nop 0           ; hazard for exp\n"
                     "v_add_f32 %[v_y], %[v_y], 1.0\n"
                     "v_rcp_f32 %[v_y], %[v_y]\n"
                     "s_nop 0           ; hazard for rcp\n"
                     "v_mul_f32 %[v_y], %[v_x], %[v_y]\n"
                     : [v_y] "+v"(y), [v_x] "+v"(tmp)
                     : [s_log2e] "s"(log2e_neg_)
                     :);
    };

    template <>
    CK_TILE_HOST void operator()<fp32x2_t>(fp32x2_t& y, fp32x2_t& x) const
    {
        constexpr auto one = type_convert<float>(1);
        y[0]               = x[0] * (one / (one + ck_tile::exp(-x[0])));
        y[1]               = x[1] * (one / (one + ck_tile::exp(-x[1])));
    };

    template <>
    CK_TILE_DEVICE void operator()<fp32x2_t>(fp32x2_t& y, fp32x2_t& x) const
    {
        const uint32_t log2e_neg_ = 0x3fb8aa3b | 0x80000000; // log2e_v<float> * -1;

        // NOTE: x/y can't be same register before inline asm
        // float tmp0 = x[0], tmp1 = x[1];
        asm volatile("v_mul_f32 %[v_y0], %[s_log2e], %[v_x0]\n"
                     "v_mul_f32 %[v_y1], %[s_log2e], %[v_x1]\n"
                     "v_exp_f32 %[v_y0], %[v_y0]\n"
                     "v_exp_f32 %[v_y1], %[v_y1]\n"
                     "v_add_f32 %[v_y0], %[v_y0], 1.0\n"
                     "v_add_f32 %[v_y1], %[v_y1], 1.0\n"
                     "v_rcp_f32 %[v_y0], %[v_y0]\n"
                     "v_rcp_f32 %[v_y1], %[v_y1]\n"
                     "v_mul_f32 %[v_y0], %[v_x0], %[v_y0]\n"
                     "v_mul_f32 %[v_y1], %[v_x1], %[v_y1]\n"
                     : [v_y0] "+v"(y[0]), [v_y1] "+v"(y[1]), [v_x0] "+v"(x[0]), [v_x1] "+v"(x[1])
                     : [s_log2e] "s"(log2e_neg_)
                     :);
    };
};
#endif

struct TanH
{
    static constexpr const char* name = "TanH";

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        y = ck_tile::tanh(x);
    };
};

struct ACos
{
    static constexpr const char* name = "ACos";

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        y = ck_tile::acos(x);
    };
};

struct Neg
{
    static constexpr const char* name = "Neg";

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        y = ck_tile::neg(x);
    };
};

struct ATan
{
    static constexpr const char* name = "ATan";

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        y = ck_tile::atan(x);
    };
};

struct Sin
{
    static constexpr const char* name = "Sin";

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        y = ck_tile::sin(x);
    };
};

struct ASinH
{
    static constexpr const char* name = "ASinH";

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        y = ck_tile::asinh(x);
    };
};

struct Cos
{
    static constexpr const char* name = "Cos";

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        y = ck_tile::cos(x);
    };
};

struct ACosH
{
    static constexpr const char* name = "ACosH";

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        y = ck_tile::acosh(x);
    };
};

struct Tan
{
    static constexpr const char* name = "Tan";

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        y = ck_tile::tan(x);
    };
};

struct ATanH
{
    static constexpr const char* name = "ATanH";

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        y = ck_tile::atanh(x);
    };
};

struct SinH
{
    static constexpr const char* name = "SinH";

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        y = ck_tile::sinh(x);
    };
};

struct Ceil
{
    static constexpr const char* name = "Ceil";

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        y = ck_tile::ceil(x);
    };
};

struct Exp
{
    static constexpr const char* name = "Exp";

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        y = ck_tile::exp(x);
    };
};

struct CosH
{
    static constexpr const char* name = "CosH";

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        y = ck_tile::cosh(x);
    };
};

struct Floor
{
    static constexpr const char* name = "Floor";

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        y = ck_tile::floor(x);
    };
};

struct Log
{
    static constexpr const char* name = "Log";

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        y = ck_tile::log(x);
    };
};

struct ASin
{
    static constexpr const char* name = "ASin";

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        y = ck_tile::asin(x);
    };
};

struct Rcp
{
    static constexpr const char* name = "Rcp";

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        y = ck_tile::rcp(x);
    };
};

struct Swish
{
    static constexpr const char* name = "Swish";

    Swish(float beta = 1.0f) : beta_(beta) {}

    template <typename Y, typename X>
    CK_TILE_HOST_DEVICE void operator()(Y& y, const X& x) const
    {
        static_assert(std::is_same_v<X, float> || std::is_same_v<X, double> ||
                          std::is_same_v<X, ck_tile::fp16_t>,
                      "Data type is not supported by this operation!");

        static_assert(std::is_same_v<Y, float> || std::is_same_v<Y, double> ||
                          std::is_same_v<Y, ck_tile::fp16_t>,
                      "Data type is not supported by this operation!");

        float bx = -beta_ * type_convert<float>(x);
        y        = type_convert<Y>(x / (1.f + ck_tile::exp(bx)));
    };

    const float beta_;
};

struct SoftRelu
{
    static constexpr const char* name = "SoftRelu";

    SoftRelu(float alpha = 1.f) : alpha_(alpha){};

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int32_t> ||
                          std::is_same_v<T, int8_t>,
                      "Data type is not supported by this operation!");
        T casted_alpha  = type_convert<T>(alpha_);
        constexpr T one = type_convert<T>(1);
        y               = ck_tile::log(one + ck_tile::exp(x * casted_alpha)) / casted_alpha;
    }
    const float alpha_;
};

struct Power
{
    static constexpr const char* name = "Power";

    Power(float alpha = 0.f, float beta = 1.f, float gamma = 2.f)
        : alpha_(alpha), beta_(beta), gamma_(gamma){};

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int32_t> ||
                          std::is_same_v<T, int8_t>,
                      "Data type is not supported by this operation!");
        T casted_alpha     = type_convert<T>(alpha_);
        T casted_beta      = type_convert<T>(beta_);
        T casted_gamma     = type_convert<T>(gamma_);
        T shifted_scaled_x = casted_alpha + casted_beta * x;
        y                  = ck_tile::pow(shifted_scaled_x, casted_gamma);
    }
    const float alpha_;
    const float beta_;
    const float gamma_;
};

struct ClippedRelu
{
    static constexpr const char* name = "ClippedRelu";

    ClippedRelu(float alpha = 0.f, float beta = 1.f) : alpha_(alpha), beta_(beta){};

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int32_t> ||
                          std::is_same_v<T, int8_t>,
                      "Data type is not supported by this operation!");
        T casted_alpha = type_convert<T>(alpha_);
        T casted_beta  = type_convert<T>(beta_);
        y              = ck_tile::min(casted_beta, ck_tile::max(casted_alpha, x));
    }
    const float alpha_;
    const float beta_;
};

struct LeakyRelu
{
    static constexpr const char* name = "LeakyRelu";

    LeakyRelu(float alpha = 0.01f) : alpha_(alpha){};

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int32_t> ||
                          std::is_same_v<T, int8_t>,
                      "Data type is not supported by this operation!");
        T casted_alpha = type_convert<T>(alpha_);
        y              = x >= 0 ? x : x * casted_alpha;
    }
    const float alpha_;
};

struct Elu
{
    static constexpr const char* name = "Elu";

    Elu(float alpha = 1.f) : alpha_(alpha){};

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int32_t> ||
                          std::is_same_v<T, int8_t>,
                      "Data type is not supported by this operation!");
        T casted_alpha = type_convert<T>(alpha_);
        y              = x > 0 ? x : casted_alpha * ck_tile::expm1(x);
    }
    const float alpha_;
};

struct Logistic
{
    static constexpr const char* name = "Logistic";

    Logistic(float alpha = 1.f) : alpha_(alpha){};

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(T& y, const T& x) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int32_t> ||
                          std::is_same_v<T, int8_t>,
                      "Data type is not supported by this operation!");
        T casted_alpha  = type_convert<T>(alpha_);
        constexpr T one = type_convert<T>(1);
        y               = casted_alpha / (one + ck_tile::exp(-x) * casted_alpha);
    }
    const float alpha_;
};

struct Clamp
{
    CK_TILE_HOST_DEVICE Clamp(float lower = std::numeric_limits<float>::lowest(),
                              float upper = std::numeric_limits<float>::max())
        : lower_(lower), upper_(upper) {};

    template <typename T>
    CK_TILE_HOST_DEVICE constexpr void operator()(T& y, const T& x) const
    {
        T lower = ck_tile::type_convert<T>(lower_);
        T upper = ck_tile::type_convert<T>(upper_);
        y       = ck_tile::clamp(x, lower, upper);
    }

    float lower_, upper_;
};

struct ConvInvscale
{
    static constexpr const char* name = "ConvInvscale";

    CK_TILE_HOST_DEVICE
    ConvInvscale(float scale_in = 1.f, float scale_wei = 1.f, float scale_out = 1.f)
        : scale_in_(scale_in), scale_wei_(scale_wei), scale_out_(scale_out)
    {
    }

    template <typename E, typename C>
    CK_TILE_HOST_DEVICE void operator()(E& e, const C& c) const;

    template <>
    CK_TILE_HOST_DEVICE void operator()<ck_tile::fp8_t, float>(ck_tile::fp8_t& e,
                                                               const float& c) const
    {
        e = type_convert<ck_tile::fp8_t>(c / scale_in_ / scale_wei_ / scale_out_);
    };

    float scale_in_;
    float scale_wei_;
    float scale_out_;
};

struct ConvScale
{
    static constexpr const char* name = "ConvScale";

    CK_TILE_HOST_DEVICE
    ConvScale(float scale_in = 1.f, float scale_wei = 1.f, float scale_out = 1.f)
        : scale_in_(scale_in), scale_wei_(scale_wei), scale_out_(scale_out)
    {
    }

    template <typename E, typename C>
    CK_TILE_HOST_DEVICE void operator()(E& e, const C& c) const;

    template <>
    CK_TILE_HOST_DEVICE void operator()<ck_tile::fp8_t, float>(ck_tile::fp8_t& e,
                                                               const float& c) const
    {
        e = type_convert<ck_tile::fp8_t>(c * scale_in_ * scale_wei_ * scale_out_);
    };

    float scale_in_;
    float scale_wei_;
    float scale_out_;
};

struct ConvScaleRelu
{
    static constexpr const char* name = "ConvScaleRelu";

    CK_TILE_HOST_DEVICE
    ConvScaleRelu(float scale_in = 1.f, float scale_wei = 1.f, float scale_out = 1.f)
        : scale_in_(scale_in), scale_wei_(scale_wei), scale_out_(scale_out)
    {
    }

    template <typename E, typename C>
    CK_TILE_HOST_DEVICE void operator()(E& e, const C& c) const;

    template <>
    CK_TILE_HOST_DEVICE void operator()<ck_tile::fp8_t, float>(ck_tile::fp8_t& e,
                                                               const float& c) const
    {
        float x;
        Relu{}.template operator()<float>(x, c * scale_in_ * scale_wei_);
        e = type_convert<ck_tile::fp8_t>(x * scale_out_);
    };

    float scale_in_;
    float scale_wei_;
    float scale_out_;
};

template <typename DstType, typename SrcType>
struct Cast
{
    static constexpr const char* name = "Cast";

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(DstType& y, const SrcType& x) const
    {
        y = ck_tile::type_convert<DstType>(x);
    };
};

/**
 * @brief Compose two unary element-wise functions into one.
 *
 *
 * @note The Ds tensor can be used by at most one of the composed functions.
 * This holds even if compositions are chained:
 * In `Compose<FA, Compose<FB, FC>>`, only one of `FA`, `FB`, or `FC` can use
 * the Ds tensor.
 *
 * @tparam FuncA    The first function to be applied.
 * @tparam FuncB    The second function to be applied.
 * @tparam FuncADs  Whether `FuncA` uses the Ds tensor.
 * @tparam FuncBDs  Whether `FuncB` uses the Ds tensor.
 */
template <typename FuncA, typename FuncB, bool FuncADs = false, bool FuncBDs = false>
struct Compose
{
    static_assert(!(FuncADs && FuncBDs), "Only one composed function may use the Ds tensor.");

    CK_TILE_HOST_DEVICE Compose(FuncA func_a_ = FuncA{}, FuncB func_b_ = FuncB{})
        : func_a(func_a_), func_b(func_b_)
    {
    }

    template <typename AIn, typename BOut, typename AOut = AIn, typename... ADs>
    CK_TILE_HOST_DEVICE constexpr void operator()(BOut& y, const AIn& x, const ADs&... ds) const
    {
        AOut tmp;
        if constexpr(FuncADs)
        {
            func_a(tmp, x, ds...);
            func_b(y, tmp);
        }
        else if constexpr(FuncBDs)
        {
            func_a(tmp, x);
            func_b(y, tmp, ds...);
        }
        else
        {
            func_a(tmp, x);
            func_b(y, tmp);
        }
    }

    const FuncA func_a;
    const FuncB func_b;
};

// support fastconvert of int8 to fp16
#if 0
template <typename InputDataType, typename OutputDataType, index_t RegPackNumber>
struct FastNumericArrayConverter
{
};

template <>
struct FastNumericArrayConverter<uint8_t, ck_tile::fp16_t, 4>
{
    using InputArray  = vector_type<uint8_t, 4>;
    using OutputArray = vector_type<ck_tile::fp16_t, 4>;

    CK_TILE_DEVICE static OutputArray convert(InputArray const& Input)
    {
        OutputArray Output;

        uint32_t* half_2       = reinterpret_cast<uint32_t*>(&Output);
        uint32_t const uint8_4 = reinterpret_cast<uint32_t const&>(Input);

        static constexpr uint32_t byte_selector_01 = 0x05010500;
        static constexpr uint32_t byte_selector_23 = 0x05030502;
        static constexpr uint32_t fp16_adder       = 0x64646464;
        half_2[0] = __builtin_amdgcn_perm(fp16_adder, uint8_4, byte_selector_01);
        half_2[1] = __builtin_amdgcn_perm(fp16_adder, uint8_4, byte_selector_23);

        static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;
        asm volatile("v_pk_add_f16 %0, %1, %2 neg_lo:[0,1] neg_hi:[0,1]"
                     : "=v"(half_2[0])
                     : "v"(half_2[0]), "s"(I8s_TO_F16s_MAGIC_NUM));
        asm volatile("v_pk_add_f16 %0, %1, %2 neg_lo:[0,1] neg_hi:[0,1]"
                     : "=v"(half_2[1])
                     : "v"(half_2[1]), "s"(I8s_TO_F16s_MAGIC_NUM));

        return Output;
    }

    CK_TILE_DEVICE OutputArray operator()(InputArray const& Input) { return convert(Input); }
};

template <index_t N>
struct FastNumericArrayConverter<uint8_t, ck_tile::fp16_t, N>
{
    static constexpr int VEC_WIDTH = 4;
    static_assert(!(N % VEC_WIDTH), "N must be multiple of 4.");

    using InputArray  = vector_type<uint8_t, N>;
    using OutputArray = vector_type<ck_tile::fp16_t, N>;

    CK_TILE_DEVICE static OutputArray convert(InputArray const& Input)
    {
        FastNumericArrayConverter<uint8_t, ck_tile::fp16_t, 4> converter;

        OutputArray Output;

        using Vec_InputArray  = vector_type<uint8_t, 4>;
        using Vec_OutputArray = vector_type<ck_tile::fp16_t, 4>;

        Vec_OutputArray* half_4_ptr       = reinterpret_cast<Vec_OutputArray*>(&Output);
        Vec_InputArray const* uint8_4_ptr = reinterpret_cast<Vec_InputArray const*>(&Input);

        static_for<0, N / VEC_WIDTH, 1>{}(
            [&](auto i) { half_4_ptr[i] = converter(uint8_4_ptr[i]); });

        return Output;
    }

    CK_TILE_DEVICE OutputArray operator()(InputArray const& Input) { return convert(Input); }
};
#endif

} // namespace element_wise
} // namespace ck_tile
