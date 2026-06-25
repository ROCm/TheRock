// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/f8_utils.hpp"
#include "ck/utility/get_id.hpp"
#include "ck/utility/mxf4_utils.hpp"
#include "ck/utility/mxf6_utils.hpp"
#include "ck/utility/random_gen.hpp"
#include "ck/utility/array.hpp"
#include "ck/utility/amd_inline_asm.hpp"
#include "ck/utility/type.hpp"

namespace ck {
// Define the common macro for MI300 models
#if defined(__gfx942__) || defined(__gfx950__)
#define __gfx94__
#endif

namespace {
namespace details {

[[maybe_unused]] __host__ half2_t pk_add_f16(const half2_t& x, const half2_t& y)
{
    half2_t vector_res;

    vector_res.x = x.x + y.x;
    vector_res.y = x.y + y.y;

    return vector_res;
}

[[maybe_unused]] __device__ half2_t pk_add_f16(const half2_t& x, const half2_t& y)
{
    return amd_assembly_pk_add_f16(x, y);
}
} // namespace details
} // namespace

#if defined(__gfx950__)
inline __device__ bhalf_t static_cast_float_to_bf16(float x)
{
    union
    {
        uint16_t uint16;
        __bf16 bf16;
    } out;
    out.bf16 = static_cast<__bf16>(x);
    return out.uint16;
}
#endif

// Declare a template function for bf16 conversion using RTN
template <typename Y, typename X>
__host__ __device__ constexpr Y bf16_convert_rtn(X x);

// Convert fp32 to bf16 with RTN if higher precision is needed
template <>
inline __host__ __device__ constexpr bhalf_t bf16_convert_rtn<bhalf_t, float>(float x)
{
#if defined(__gfx950__)
    return static_cast_float_to_bf16(x);
#else
    // Nan check
    if(x != x)
    {
        return uint16_t(0x7FC0);
    }

    union
    {
        float fp32;
        uint32_t int32;
    } u = {x};

    const uint32_t first_bf16_mantisa_bit = ((u.int32 >> 16) & 1);
    constexpr uint32_t rounding_bias      = uint32_t((1 << 15) - 1);

    return uint16_t((u.int32 + first_bf16_mantisa_bit + rounding_bias) >> 16);
#endif
}

// convert fp16 to bfp16 via fp32 with RTN if higher precision is needed
template <>
inline __host__ __device__ constexpr bhalf_t bf16_convert_rtn<bhalf_t, half_t>(half_t x)
{
    float x_fp32 = static_cast<float>(x);

    return bf16_convert_rtn<bhalf_t>(x_fp32);
}

// Convert X to Y, both X and Y are non-const data types.
template <typename Y,
          typename X,
          ck::enable_if_t<!(ck::is_const_v<Y> || ck::is_const_v<X>), bool> = false>
__host__ __device__ constexpr Y type_convert(X x)
{
    static_assert(!ck::is_reference_v<Y> && !ck::is_reference_v<X>);

    return static_cast<Y>(x);
}

// Convert X to Y, either X or Y is a const data type.
template <typename Y,
          typename X,
          ck::enable_if_t<ck::is_const_v<Y> || ck::is_const_v<X>, bool> = false>
__host__ __device__ constexpr Y type_convert(X x)
{
    static_assert(!ck::is_reference_v<Y> && !ck::is_reference_v<X>);

    using NonConstY = ck::remove_const_t<Y>;
    using NonConstX = ck::remove_const_t<X>;
    return static_cast<Y>(type_convert<NonConstY, NonConstX>(x));
}

// convert bfp16 to fp32
template <>
inline __host__ __device__ constexpr float type_convert<float, bhalf_t>(bhalf_t x)
{
    union
    {
        uint32_t int32;
        float fp32;
    } u = {uint32_t(x) << 16};

    return u.fp32;
}

// convert fp32 to bfp16, round to nearest even
template <>
inline __host__ __device__ constexpr bhalf_t type_convert<bhalf_t, float>(float x)
{
#if CK_USE_RNE_BF16_CONVERSION
    return bf16_convert_rtn<bhalf_t>(x);
#else
    return uint16_t(static_cast<uint32_t>(x) >> 16);
#endif
}

// convert bfp16 to fp16 via fp32
template <>
inline __host__ __device__ constexpr half_t type_convert<half_t, bhalf_t>(bhalf_t x)
{
    float x_fp32 = type_convert<float>(x);

    return static_cast<half_t>(x_fp32);
}

// convert fp16 to bfp16 via fp32
template <>
inline __host__ __device__ constexpr bhalf_t type_convert<bhalf_t, half_t>(half_t x)
{
    float x_fp32 = static_cast<float>(x);

    return type_convert<bhalf_t>(x_fp32);
}

// convert bfp16 to int8 via fp32
template <>
inline __host__ __device__ constexpr int8_t type_convert<int8_t, bhalf_t>(bhalf_t x)
{
    float x_fp32 = type_convert<float>(x);

    return static_cast<int8_t>(x_fp32);
}

// convert int8 to bfp16 via fp32
template <>
inline __host__ __device__ constexpr bhalf_t type_convert<bhalf_t, int8_t>(int8_t x)
{
    float x_fp32 = static_cast<float>(x);

    return type_convert<bhalf_t>(x_fp32);
}

template <>
inline __host__ __device__ constexpr f8_ocp_t type_convert<f8_ocp_t, int>(int x)
{
    return f8_ocp_t{type_convert<f8_ocp_t::data_type>(x)};
}

template <>
inline __host__ __device__ constexpr bf8_ocp_t type_convert<bf8_ocp_t, int>(int x)
{
    return bf8_ocp_t{type_convert<bf8_ocp_t::data_type>(x)};
}

template <typename Y, enable_if_t<is_same_v<Y, ck::tf32_t>, bool> = false>
inline __host__ __device__ constexpr float type_convert(float x)
{
    union
    {
        float fp32;
        uint32_t int32;
    } u = {x};

    u.int32 = u.int32 & 0xffffe000;
    return u.fp32;
}

// Convert X to Y
template <typename Y, typename X>
__host__ __device__ constexpr Y type_convert_sp(X x)
{
    static_assert(!ck::is_reference_v<Y> && !ck::is_reference_v<X>);
    return static_cast<Y>(x);
}

template <>
inline __host__ __device__ constexpr int type_convert_sp<int, float>(float x)
{
    union
    {
        float fp32;
        int int32;
    } u = {x};

    return u.int32;
}

template <>
inline __host__ __device__ constexpr float type_convert_sp<float, int>(int x)
{
    union
    {
        int int32;
        float fp32;
    } u = {x};

    return u.fp32;
}

template <>
inline __host__ __device__ constexpr int type_convert_sp<int, half_t>(half_t x)
{
    union
    {
        half_t fp16;
        int int32;
    } u = {x};

    return u.int32;
}

template <>
inline __host__ __device__ constexpr half_t type_convert_sp<half_t, int>(int x)
{
    union
    {
        int int32;
        half_t fp16;
    } u = {x};

    return u.fp16;
}

template <>
inline __host__ __device__ constexpr int type_convert_sp<int, f8_t>(f8_t x)
{
    union
    {
        f8_t fp8;
        int int32;
    } u = {x};

    return u.int32;
}

template <>
inline __host__ __device__ constexpr f8_t type_convert_sp<f8_t, int>(int x)
{
    union
    {
        int int32;
        f8_t fp8;
    } u = {x};

    return u.fp8;
}

template <>
inline __host__ __device__ constexpr int type_convert_sp<int, bhalf_t>(bhalf_t x)
{
    union
    {
        bhalf_t fp16;
        int int32;
    } u = {x};

    return u.int32;
}

template <>
inline __host__ __device__ constexpr bhalf_t type_convert_sp<bhalf_t, int>(int x)
{
    union
    {
        int int32;
        bhalf_t fp16;
    } u = {x};

    return u.fp16;
}

template <>
inline __host__ __device__ constexpr bhalf_t type_convert_sp<bhalf_t, float>(float x)
{
    return type_convert<bhalf_t>(x);
}

template <>
inline __host__ __device__ constexpr half_t type_convert_sp<half_t, float>(float x)
{
    return type_convert<half_t>(x);
}
// Declare a template function for fp8 conversion using SR
template <typename Y, typename X>
__host__ __device__ constexpr Y f8_convert_sr(X x);

// convert fp32 to fp8 with stochastic rounding
template <>
inline __host__ __device__ f8_fnuz_t f8_convert_sr<f8_fnuz_t, float>(float x)
{
#if defined(__gfx950__)
    // use HW clock for stochastic input multiply by incremented thread id
    uint32_t rng = __builtin_amdgcn_prng_b32(__builtin_amdgcn_s_memrealtime() *
                                             (get_thread_global_1d_id() + 1));
#else
    constexpr int seed = 1254739;
#ifndef CK_CODE_GEN_RTC
    uint32_t rng = prand_generator<float, seed>(reinterpret_cast<uintptr_t>(&x), x);
#else
    uint32_t rng = prand_generator<float, seed>(reinterpret_cast<size_t>(&x), x);
#endif // #ifndef CK_CODE_GEN_RTC
#endif // #if defined(__gfx950__)
#if defined(__gfx94__)
    union
    {
        float fval;
        uint32_t i32val;
        uint8_t i8val[4]; // not endian independent
    } val;
    val.fval            = x;
    uint32_t ival       = 0;
    const float max_fp8 = 240.0f;
    // if x is not +/- infinity or nan
    if((val.i32val & NumericUtils<float>::nan_mask) != NumericUtils<float>::Inf)
        // clip float value
        val.fval = __builtin_amdgcn_fmed3f(val.fval, max_fp8, -max_fp8);
    ival       = __builtin_amdgcn_cvt_sr_fp8_f32(val.fval, rng, ival, 0); // 0 pos
    val.i32val = ival;
    return f8_fnuz_t{val.i8val[0]}; // little endian
#else
    constexpr bool negative_zero_nan = true;
    constexpr bool clip              = true;
    constexpr f8_rounding_mode rm    = f8_rounding_mode::stochastic;
    return utils::
        cast_to_f8<float, f8_fnuz_t, negative_zero_nan, clip, (rm == f8_rounding_mode::stochastic)>(
            x, rng);
#endif
}

// convert fp16 to fp8 with stochastic rounding
template <>
inline __host__ __device__ f8_fnuz_t f8_convert_sr<f8_fnuz_t, half_t>(half_t x)
{
#if defined(__gfx94__)
    // convert to float and use native converion
    return f8_convert_sr<f8_fnuz_t>(type_convert<float>(x));
#else
    constexpr bool negative_zero_nan = true;
    constexpr bool clip              = true;
    constexpr f8_rounding_mode rm    = f8_rounding_mode::stochastic;
    constexpr int seed               = 1254739;
#ifndef CK_CODE_GEN_RTC
    uint32_t rng = prand_generator<half_t, seed>(reinterpret_cast<uintptr_t>(&x), x);
#else
    uint32_t rng = prand_generator<half_t, seed>(reinterpret_cast<size_t>(&x), x);
#endif
    return utils::cast_to_f8<half_t,
                             f8_fnuz_t,
                             negative_zero_nan,
                             clip,
                             (rm == f8_rounding_mode::stochastic)>(x, rng);
#endif
}

// convert fp32 to bf8 with stochastic rounding
template <>
inline __host__ __device__ bf8_fnuz_t f8_convert_sr<bf8_fnuz_t, float>(float x)
{
#if defined(__gfx950__)
    // use HW clock for stochastic input multiply by incremented thread id
    uint32_t rng = __builtin_amdgcn_prng_b32(__builtin_amdgcn_s_memrealtime() *
                                             (get_thread_global_1d_id() + 1));
#else
    constexpr int seed = 1254739;
#ifndef CK_CODE_GEN_RTC
    uint32_t rng = prand_generator<float, seed>(reinterpret_cast<uintptr_t>(&x), x);
#else
    uint32_t rng = prand_generator<float, seed>(reinterpret_cast<size_t>(&x), x);
#endif // #ifndef CK_CODE_GEN_RTC
#endif // #if defined(__gfx950__)
#if defined(__gfx94__)
    union
    {
        float fval;
        uint32_t i32val;
        uint8_t i8val[4]; // not endian independent
    } val;
    val.fval            = x;
    uint32_t ival       = 0;
    const float max_bf8 = 57344.0f;
    // if x is not +/- infinity or nan
    if((val.i32val & NumericUtils<float>::nan_mask) != NumericUtils<float>::Inf)
        // clip float value
        val.fval = __builtin_amdgcn_fmed3f(val.fval, max_bf8, -max_bf8);
    ival       = __builtin_amdgcn_cvt_sr_bf8_f32(val.fval, rng, ival, 0); // 0 pos
    val.i32val = ival;
    return bf8_fnuz_t{val.i8val[0]}; // little endian
#else
    constexpr bool negative_zero_nan = true;
    constexpr bool clip              = true;
    constexpr f8_rounding_mode rm    = f8_rounding_mode::stochastic;
    return utils::cast_to_f8<float,
                             bf8_fnuz_t,
                             negative_zero_nan,
                             clip,
                             (rm == f8_rounding_mode::stochastic)>(x, rng);
#endif
}

// convert fp16 to bf8 with stochastic rounding
template <>
inline __host__ __device__ bf8_fnuz_t f8_convert_sr<bf8_fnuz_t, half_t>(half_t x)
{
#if defined(__gfx94__)
    // convert to float and use native converion
    return f8_convert_sr<bf8_fnuz_t>(type_convert<float>(x));
#else
    constexpr bool negative_zero_nan = true;
    constexpr bool clip              = true;
    constexpr f8_rounding_mode rm    = f8_rounding_mode::stochastic;
    constexpr int seed               = 1254739;
#ifndef CK_CODE_GEN_RTC
    uint32_t rng = prand_generator<half_t, seed>(reinterpret_cast<uintptr_t>(&x), x);
#else
    uint32_t rng = prand_generator<half_t, seed>(reinterpret_cast<size_t>(&x), x);
#endif
    return utils::cast_to_f8<half_t,
                             bf8_fnuz_t,
                             negative_zero_nan,
                             clip,
                             (rm == f8_rounding_mode::stochastic)>(x, rng);
#endif
}

/**
 * @brief Converts a float to a 8-bit float type (f8_ocp_t) using stochastic rounding.
 *
 * @param x     The input float value.
 * @return      The converted f8_ocp_t value.
 */
template <>
inline __host__ __device__ f8_ocp_t f8_convert_sr<f8_ocp_t, float>(float x)
{
    return f8_ocp_t{
        fp8_impl::cvt_float_to_fp8<f8_ocp_t::default_interpret, f8_ocp_t::default_saturation, true>(
            x)};
}

/**
 * @brief Converts a vector of 2 floats to a vector of 2 8-bit float types (f8_ocp_t) using
 * stochastic rounding.
 *
 * @param x     The input vector of 2 floats.
 * @return      The converted vector of 2 f8_ocp_t.
 */
template <>
inline __host__ __device__ f8x2_ocp_t f8_convert_sr<f8x2_ocp_t, float2_t>(float2_t x)
{
    return f8x2_ocp_t{
        fp8_impl::cvt_float_to_fp8<f8_ocp_t::default_interpret, f8_ocp_t::default_saturation, true>(
            x)};
}

/**
 * @brief Converts a float to a 8-bit float type (bf8_ocp_t) using stochastic rounding.
 *
 * @param x     The input float value.
 * @return      The converted bf8_ocp_t value.
 */
template <>
inline __host__ __device__ bf8_ocp_t f8_convert_sr<bf8_ocp_t, float>(float x)
{
    return bf8_ocp_t{fp8_impl::cvt_float_to_fp8<bf8_ocp_t::default_interpret,
                                                bf8_ocp_t::default_saturation,
                                                true>(x)};
}

/**
 * @brief Converts a vector of 2 floats to a vector of 2 8-bit float types (bf8_ocp_t) using
 * stochastic rounding.
 *
 * @param x     The input vector of 2 floats.
 * @return      The converted vector of 2 bf8_ocp_t.
 */
template <>
inline __host__ __device__ bf8x2_ocp_t f8_convert_sr<bf8x2_ocp_t, float2_t>(float2_t x)
{
    return bf8x2_ocp_t{fp8_impl::cvt_float_to_fp8<bf8_ocp_t::default_interpret,
                                                  bf8_ocp_t::default_saturation,
                                                  true>(x)};
}

/**
 * @brief Converts a half_t to a 8-bit float type (f8_ocp_t) using stochastic rounding.
 *
 * @param x     The input half_t value.
 * @return      The converted f8_ocp_t value.
 */
template <>
inline __host__ __device__ f8_ocp_t f8_convert_sr<f8_ocp_t, half_t>(half_t x)
{
    return f8_ocp_t{fp8_impl::cvt_half_t_to_fp8<f8_ocp_t::default_interpret,
                                                f8_ocp_t::default_saturation,
                                                true>(x)};
}

/**
 * @brief Converts a vector of 2 half_t to a vector of 2 8-bit float types (f8_ocp_t) using
 * stochastic rounding.
 *
 * @param x     The input vector of 2 half_t.
 * @return      The converted vector of 2 f8_ocp_t.
 */
template <>
inline __host__ __device__ f8x2_ocp_t f8_convert_sr<f8x2_ocp_t, half2_t>(half2_t x)
{
    return f8x2_ocp_t{fp8_impl::cvt_half_t_to_fp8<f8_ocp_t::default_interpret,
                                                  f8_ocp_t::default_saturation,
                                                  true>(x)};
}

/**
 * @brief Converts a half_t to a 8-bit half_t type (bf8_ocp_t) using stochastic rounding.
 *
 * @param x     The input half_t value.
 * @return      The converted bf8_ocp_t value.
 */
template <>
inline __host__ __device__ bf8_ocp_t f8_convert_sr<bf8_ocp_t, half_t>(half_t x)
{
    return bf8_ocp_t{fp8_impl::cvt_half_t_to_fp8<bf8_ocp_t::default_interpret,
                                                 bf8_ocp_t::default_saturation,
                                                 true>(x)};
}

/**
 * @brief Converts a vector of 2 half_t to a vector of 2 8-bit float types (bf8_ocp_t) using
 * stochastic rounding.
 *
 * @param x     The input vector of 2 half_t.
 * @return      The converted vector of 2 bf8_ocp_t.
 */
template <>
inline __host__ __device__ bf8x2_ocp_t f8_convert_sr<bf8x2_ocp_t, half2_t>(half2_t x)
{
    return bf8x2_ocp_t{fp8_impl::cvt_half_t_to_fp8<bf8_ocp_t::default_interpret,
                                                   bf8_ocp_t::default_saturation,
                                                   true>(x)};
}

/**
 * @brief Converts a bhalf_t to a 8-bit float type (f8_ocp_t) using stochastic rounding.
 *
 * @param x     The input bhalf_t value.
 * @return      The converted f8_ocp_t value.
 */
template <>
inline __host__ __device__ f8_ocp_t f8_convert_sr<f8_ocp_t, bhalf_t>(bhalf_t x)
{
    return f8_ocp_t{fp8_impl::cvt_bhalf_t_to_fp8<f8_ocp_t::default_interpret,
                                                 f8_ocp_t::default_saturation,
                                                 true>(x)};
}

/**
 * @brief Converts a vector of 2 bhalf_t to a vector of 2 8-bit float types (f8_ocp_t) using
 * stochastic rounding.
 *
 * @param x     The input vector of 2 bhalf_t.
 * @return      The converted vector of 2 f8_ocp_t.
 */
template <>
inline __host__ __device__ f8x2_ocp_t f8_convert_sr<f8x2_ocp_t, bhalf2_t>(bhalf2_t x)
{
    return f8x2_ocp_t{fp8_impl::cvt_bhalf_t_to_fp8<f8_ocp_t::default_interpret,
                                                   f8_ocp_t::default_saturation,
                                                   true>(x)};
}

/**
 * @brief Converts a bhalf_t to a 8-bit half_t type (bf8_ocp_t) using stochastic rounding.
 *
 * @param x     The input bhalf_t value.
 * @return      The converted bf8_ocp_t value.
 */
template <>
inline __host__ __device__ bf8_ocp_t f8_convert_sr<bf8_ocp_t, bhalf_t>(bhalf_t x)
{
    return bf8_ocp_t{fp8_impl::cvt_bhalf_t_to_fp8<bf8_ocp_t::default_interpret,
                                                  bf8_ocp_t::default_saturation,
                                                  true>(x)};
}

/**
 * @brief Converts a vector of 2 bhalf_t to a vector of 2 8-bit float types (bf8_ocp_t) using
 * stochastic rounding.
 *
 * @param x     The input vector of 2 bhalf_t.
 * @return      The converted vector of 2 bf8_ocp_t.
 */
template <>
inline __host__ __device__ bf8x2_ocp_t f8_convert_sr<bf8x2_ocp_t, bhalf2_t>(bhalf2_t x)
{
    return bf8x2_ocp_t{fp8_impl::cvt_bhalf_t_to_fp8<bf8_ocp_t::default_interpret,
                                                    bf8_ocp_t::default_saturation,
                                                    true>(x)};
}

// Declare a template function for fp8 conversion using RNE
template <typename Y, typename X>
__host__ __device__ constexpr Y f8_convert_rne(X x);

// convert fp32 to fp8 with rounding to nearest even
template <>
inline __host__ __device__ f8_fnuz_t f8_convert_rne<f8_fnuz_t, float>(float x)
{
#if defined(__gfx94__)
    union
    {
        float fval;
        uint32_t i32val;
        uint8_t i8val[4]; // not endian independent
    } val;
    val.fval            = x;
    uint32_t ival       = 0;
    const float max_fp8 = 240.0f;
    // if x is not +/- infinity or nan
    if((val.i32val & NumericUtils<float>::nan_mask) != NumericUtils<float>::Inf)
        // clip float value
        val.fval = __builtin_amdgcn_fmed3f(val.fval, max_fp8, -max_fp8);
    ival       = __builtin_amdgcn_cvt_pk_fp8_f32(val.fval, val.fval, ival, false); // false -> WORD0
    val.i32val = ival;
    return f8_fnuz_t{val.i8val[0]};
#else
    constexpr bool negative_zero_nan = true;
    constexpr bool clip              = true;
    constexpr f8_rounding_mode rm    = f8_rounding_mode::standard;
    constexpr uint32_t rng           = 0;
    return utils::
        cast_to_f8<float, f8_fnuz_t, negative_zero_nan, clip, (rm == f8_rounding_mode::stochastic)>(
            x, rng);
#endif
}

// convert fp16 to fp8 with rounding to nearest even
template <>
inline __host__ __device__ f8_fnuz_t f8_convert_rne<f8_fnuz_t, half_t>(half_t x)
{
#if defined(__gfx94__)
    // convert to float and use native converion
    return f8_convert_rne<f8_fnuz_t>(type_convert<float>(x));
#else
    constexpr bool negative_zero_nan = true;
    constexpr bool clip              = true;
    constexpr f8_rounding_mode rm    = f8_rounding_mode::standard;
    constexpr uint32_t rng           = 0;
    return utils::cast_to_f8<half_t,
                             f8_fnuz_t,
                             negative_zero_nan,
                             clip,
                             (rm == f8_rounding_mode::stochastic)>(x, rng);
#endif
}

// convert fp32 to bf8 with rounding to nearest even
template <>
inline __host__ __device__ bf8_fnuz_t f8_convert_rne<bf8_fnuz_t, float>(float x)
{
#if defined(__gfx94__)
    union
    {
        float fval;
        uint32_t i32val;
        uint8_t i8val[4]; // not endian independent
    } val;
    val.fval            = x;
    uint32_t ival       = 0;
    const float max_bf8 = 57344.0f;
    // if x is not +/- infinity or nan
    if((val.i32val & NumericUtils<float>::nan_mask) != NumericUtils<float>::Inf)
        // clip float value
        val.fval = __builtin_amdgcn_fmed3f(val.fval, max_bf8, -max_bf8);
    ival       = __builtin_amdgcn_cvt_pk_bf8_f32(val.fval, val.fval, ival, false); // false -> WORD0
    val.i32val = ival;
    return bf8_fnuz_t{val.i8val[0]};
#else
    constexpr bool negative_zero_nan = true;
    constexpr bool clip              = true;
    constexpr f8_rounding_mode rm    = f8_rounding_mode::standard;
    constexpr uint32_t rng           = 0;
    return utils::cast_to_f8<float,
                             bf8_fnuz_t,
                             negative_zero_nan,
                             clip,
                             (rm == f8_rounding_mode::stochastic)>(x, rng);
#endif
}

// convert fp16 to bf8 with rounding to nearest even
template <>
inline __host__ __device__ bf8_fnuz_t f8_convert_rne<bf8_fnuz_t, half_t>(half_t x)
{
#if defined(__gfx94__)
    // convert to float and use native converion
    return f8_convert_rne<bf8_fnuz_t>(type_convert<float>(x));
#else
    constexpr bool negative_zero_nan = true;
    constexpr bool clip              = true;
    constexpr f8_rounding_mode rm    = f8_rounding_mode::standard;
    constexpr uint32_t rng           = 0;
    return utils::cast_to_f8<half_t,
                             bf8_fnuz_t,
                             negative_zero_nan,
                             clip,
                             (rm == f8_rounding_mode::stochastic)>(x, rng);
#endif
}

/**
 * @brief Converts a float to a 8-bit float type (f8_ocp_t) using rounding to nearest/even.
 *
 * @param x     The input float value.
 * @return      The converted f8_ocp_t value.
 */
template <>
inline __host__ __device__ f8_ocp_t f8_convert_rne<f8_ocp_t, float>(float x)
{
    return f8_ocp_t{
        fp8_impl::cvt_float_to_fp8<f8_ocp_t::default_interpret, f8_ocp_t::default_saturation>(x)};
}

/**
 * @brief Converts a vector of 2 floats to a vector of 2 8-bit float types (f8_ocp_t) using rounding
 * to nearest/even.
 *
 * @param x     The input vector of 2 floats.
 * @return      The converted vector of 2 f8_ocp_t.
 */
template <>
inline __host__ __device__ f8x2_ocp_t f8_convert_rne<f8x2_ocp_t, float2_t>(float2_t x)
{
    return f8x2_ocp_t{
        fp8_impl::cvt_float_to_fp8<f8_ocp_t::default_interpret, f8_ocp_t::default_saturation>(x)};
}

/**
 * @brief Converts a float to a 8-bit float type (bf8_ocp_t) using rounding to nearest/even.
 *
 * @param x     The input float value.
 * @return      The converted bf8_ocp_t value.
 */
template <>
inline __host__ __device__ bf8_ocp_t f8_convert_rne<bf8_ocp_t, float>(float x)
{
    return bf8_ocp_t{
        fp8_impl::cvt_float_to_fp8<bf8_ocp_t::default_interpret, bf8_ocp_t::default_saturation>(x)};
}

/**
 * @brief Converts a vector of 2 floats to a vector of 2 8-bit float types (bf8_ocp_t) using
 * rounding to nearest/even.
 *
 * @param x     The input vector of 2 floats.
 * @return      The converted vector of 2 bf8_ocp_t.
 */
template <>
inline __host__ __device__ bf8x2_ocp_t f8_convert_rne<bf8x2_ocp_t, float2_t>(float2_t x)
{
    return bf8x2_ocp_t{
        fp8_impl::cvt_float_to_fp8<bf8_ocp_t::default_interpret, bf8_ocp_t::default_saturation>(x)};
}

/**
 * @brief Converts a half_t to a 8-bit float type (f8_ocp_t) using rounding to nearest/even.
 *
 * @param x     The input half_t value.
 * @return      The converted f8_ocp_t value.
 */
template <>
inline __host__ __device__ f8_ocp_t f8_convert_rne<f8_ocp_t, half_t>(half_t x)
{
    return f8_ocp_t{
        fp8_impl::cvt_half_t_to_fp8<f8_ocp_t::default_interpret, f8_ocp_t::default_saturation>(x)};
}

/**
 * @brief Converts a vector of 2 half_t to a vector of 2 8-bit float types (f8_ocp_t) using rounding
 * to nearest/even.
 *
 * @param x     The input vector of 2 half_t.
 * @return      The converted vector of 2 f8_ocp_t.
 */
template <>
inline __host__ __device__ f8x2_ocp_t f8_convert_rne<f8x2_ocp_t, half2_t>(half2_t x)
{
    return f8x2_ocp_t{
        fp8_impl::cvt_half_t_to_fp8<f8_ocp_t::default_interpret, f8_ocp_t::default_saturation>(x)};
}

/**
 * @brief Converts a half_t to a 8-bit half_t type (bf8_ocp_t) using rounding to nearest/even.
 *
 * @param x     The input half_t value.
 * @return      The converted bf8_ocp_t value.
 */
template <>
inline __host__ __device__ bf8_ocp_t f8_convert_rne<bf8_ocp_t, half_t>(half_t x)
{
    return bf8_ocp_t{
        fp8_impl::cvt_half_t_to_fp8<bf8_ocp_t::default_interpret, bf8_ocp_t::default_saturation>(
            x)};
}

/**
 * @brief Converts a vector of 2 half_t to a vector of 2 8-bit float types (bf8_ocp_t) using
 * rounding to nearest/even.
 *
 * @param x     The input vector of 2 half_t.
 * @return      The converted vector of 2 bf8_ocp_t.
 */
template <>
inline __host__ __device__ bf8x2_ocp_t f8_convert_rne<bf8x2_ocp_t, half2_t>(half2_t x)
{
    return bf8x2_ocp_t{
        fp8_impl::cvt_half_t_to_fp8<bf8_ocp_t::default_interpret, bf8_ocp_t::default_saturation>(
            x)};
}

/**
 * @brief Converts a bhalf_t to a 8-bit float type (f8_ocp_t) using rounding to nearest/even.
 *
 * @param x     The input bhalf_t value.
 * @return      The converted f8_ocp_t value.
 */
template <>
inline __host__ __device__ f8_ocp_t f8_convert_rne<f8_ocp_t, bhalf_t>(bhalf_t x)
{
    return f8_ocp_t{
        fp8_impl::cvt_bhalf_t_to_fp8<f8_ocp_t::default_interpret, f8_ocp_t::default_saturation>(x)};
}

/**
 * @brief Converts a vector of 2 bhalf_t to a vector of 2 8-bit float types (f8_ocp_t) using
 * rounding to nearest/even.
 *
 * @param x     The input vector of 2 bhalf_t.
 * @return      The converted vector of 2 f8_ocp_t.
 */
template <>
inline __host__ __device__ f8x2_ocp_t f8_convert_rne<f8x2_ocp_t, bhalf2_t>(bhalf2_t x)
{
    return f8x2_ocp_t{
        fp8_impl::cvt_bhalf_t_to_fp8<f8_ocp_t::default_interpret, f8_ocp_t::default_saturation>(x)};
}

/**
 * @brief Converts a bhalf_t to a 8-bit half_t type (bf8_ocp_t) using rounding to nearest/even.
 *
 * @param x     The input bhalf_t value.
 * @return      The converted bf8_ocp_t value.
 */
template <>
inline __host__ __device__ bf8_ocp_t f8_convert_rne<bf8_ocp_t, bhalf_t>(bhalf_t x)
{
    return bf8_ocp_t{
        fp8_impl::cvt_bhalf_t_to_fp8<bf8_ocp_t::default_interpret, bf8_ocp_t::default_saturation>(
            x)};
}

/**
 * @brief Converts a vector of 2 bhalf_t to a vector of 2 8-bit float types (bf8_ocp_t) using
 * rounding to nearest/even.
 *
 * @param x     The input vector of 2 bhalf_t.
 * @return      The converted vector of 2 bf8_ocp_t.
 */
template <>
inline __host__ __device__ bf8x2_ocp_t f8_convert_rne<bf8x2_ocp_t, bhalf2_t>(bhalf2_t x)
{
    return bf8x2_ocp_t{
        fp8_impl::cvt_bhalf_t_to_fp8<bf8_ocp_t::default_interpret, bf8_ocp_t::default_saturation>(
            x)};
}

// convert fp32 to fp8
template <>
inline __host__ __device__ f8_fnuz_t type_convert<f8_fnuz_t, float>(float x)
{
#if CK_USE_SR_F8_CONVERSION
    return f8_convert_sr<f8_fnuz_t>(x);
#else
    return f8_convert_rne<f8_fnuz_t>(x);
#endif
}

// convert fp8 to fp32
template <>
inline __host__ __device__ float type_convert<float, f8_fnuz_t>(f8_fnuz_t x)
{
#if defined(__gfx94__)
    float fval;
    uint32_t i32val = static_cast<uint32_t>(static_cast<uint8_t>(x));
    fval            = __builtin_amdgcn_cvt_f32_fp8(i32val, 0);
    // asm volatile("v_cvt_f32_fp8 %0, %1 src0_sel:BYTE_0" : "=v"(fval) : "v"(i32val));
    return fval;
#else
    constexpr bool negative_zero_nan = true;
    return utils::cast_from_f8<f8_fnuz_t, float, negative_zero_nan>(x);
#endif
}

template <>
inline __host__ __device__ float2_t type_convert<float2_t, f8x2_fnuz_t>(f8x2_fnuz_t x)
{
#if defined(__gfx94__)
    const auto i16val = bit_cast<uint16_t>(x);
    return __builtin_amdgcn_cvt_pk_f32_fp8(i16val, 0);
#else
    constexpr bool negative_zero_nan = true;
    const auto f8x2_v                = vector_type<f8_fnuz_t, 2>(x);
    vector_type<float, 2> f32x2_v;
    f32x2_v.template AsType<float>()(Number<0>{}) =
        utils::cast_from_f8<f8_fnuz_t, float, negative_zero_nan>(
            f8x2_v.template AsType<f8_fnuz_t>()[Number<0>{}]);
    f32x2_v.template AsType<float>()(Number<1>{}) =
        utils::cast_from_f8<f8_fnuz_t, float, negative_zero_nan>(
            f8x2_v.template AsType<f8_fnuz_t>()[Number<1>{}]);
    return f32x2_v.template AsType<float2_t>()[Number<0>{}];
#endif
}

/**
 * @brief Converts a f8_ocp_t value to a float value.
 *
 * @param x     The input f8_ocp_t value.
 * @return      The converted float value.
 */
template <>
inline __host__ __device__ float type_convert<float, f8_ocp_t>(f8_ocp_t x)
{
#if CK_OCP_FP8_CVT_FAST_PATH
    union
    {
        unsigned int i32val;
        fp8_storage_t i8val[4];
    } val;
    val.i8val[0] = x.data;
    return __builtin_amdgcn_cvt_f32_fp8(val.i32val, 0);
#else
    return fp8_impl::cast_from_f8<float, f8_ocp_t::wm, f8_ocp_t::we, false>(x.data);
#endif
}

/**
 * @brief Converts a vector of 2 f8_ocp_t values to a vector of 2 float values.
 *
 * @param x     The input vector of 2 f8_ocp_t values.
 * @return      The converted vector of 2 float values.
 */
template <>
inline __host__ __device__ float2_t type_convert<float2_t, f8x2_ocp_t>(f8x2_ocp_t x)
{
#if CK_OCP_FP8_CVT_FAST_PATH
// __builtin_amdgcn_cvt_pk_f32_fp8 can produce incorrect results due to a compiler issue.
// TODO: Enable when SWDEV-532959 is fixed.
#if defined(__gfx12__)
    return float2_t{__builtin_amdgcn_cvt_f32_fp8(bit_cast<uint16_t>(x), 0),
                    __builtin_amdgcn_cvt_f32_fp8(bit_cast<uint16_t>(x), 1)};
#else
    return __builtin_amdgcn_cvt_pk_f32_fp8(bit_cast<uint16_t>(x), false);
#endif
#else
    return float2_t{fp8_impl::cast_from_f8<float, f8_ocp_t::wm, f8_ocp_t::we, false>(
                        x.AsType<fp8_storage_t>()[Number<0>{}]),
                    fp8_impl::cast_from_f8<float, f8_ocp_t::wm, f8_ocp_t::we, false>(
                        x.AsType<fp8_storage_t>()[Number<1>{}])};
#endif
}

/**
 * @brief Converts a f8_ocp_t value to a half_t value.
 *
 * @param x     The input f8_ocp_t value.
 * @return      The converted half_t value.
 */
template <>
inline __host__ __device__ half_t type_convert<half_t, f8_ocp_t>(f8_ocp_t x)
{
#if defined(__gfx950__)
    union
    {
        uint16_t i16val;
        fp8_storage_t i8val[2];
    } input;
    input.i8val[0] = x.data;

    union
    {
        half2_t half_vec;
        half_t half_arr[2];
    } output;
    output.half_vec = __builtin_amdgcn_cvt_scalef32_pk_f16_fp8(input.i16val, /*scale*/ 1.f, 0);

    return output.half_arr[0];
#else
    return fp8_impl::cast_from_f8<half_t, f8_ocp_t::wm, f8_ocp_t::we, false>(x.data);
#endif
}

/**
 * @brief Converts a vector of 2 f8_ocp_t values to a vector of 2 half_t values.
 *
 * @param x     The input vector of 2 f8_ocp_t values.
 * @return      The converted vector of 2 half_t values.
 */
template <>
inline __host__ __device__ half2_t type_convert<half2_t, f8x2_ocp_t>(f8x2_ocp_t x)
{
#if defined(__gfx950__)
    return __builtin_amdgcn_cvt_scalef32_pk_f16_fp8(bit_cast<uint16_t>(x), /*scale*/ 1.f, 0);
#else
    return half2_t{type_convert<half_t>(float(x.AsType<f8_ocp_t>()[Number<0>{}])),
                   type_convert<half_t>(float(x.AsType<f8_ocp_t>()[Number<1>{}]))};
#endif
}

/**
 * @brief Converts a f8_ocp_t value to a bhalf_t value.
 *
 * @param x     The input f8_ocp_t value.
 * @return      The converted bhalf_t value.
 */
template <>
inline __host__ __device__ bhalf_t type_convert<bhalf_t, f8_ocp_t>(f8_ocp_t x)
{
#if defined(__gfx950__)
    union
    {
        uint16_t i16val;
        fp8_storage_t i8val[2];
    } input;
    input.i8val[0] = x.data;

    union
    {
        bhalf2_t bhalf_vec;
        bhalf_t bhalf_arr[2];
    } output;
    output.bhalf_vec = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(input.i16val, /*scale*/ 1.f, 0);

    return output.bhalf_arr[0];
#else
    return type_convert<bhalf_t>(
        fp8_impl::cast_from_f8<float, f8_ocp_t::wm, f8_ocp_t::we, false>(x.data));
#endif
}

/**
 * @brief Converts a vector of 2 f8_ocp_t values to a vector of 2 bhalf_t values.
 *
 * @param x     The input vector of 2 f8_ocp_t values.
 * @return      The converted vector of 2 bhalf_t values.
 */
template <>
inline __host__ __device__ bhalf2_t type_convert<bhalf2_t, f8x2_ocp_t>(f8x2_ocp_t x)
{
#if defined(__gfx950__)
    return __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(bit_cast<uint16_t>(x), /*scale*/ 1.f, 0);
#else
    return bhalf2_t{type_convert<bhalf_t>(float(x.AsType<f8_ocp_t>()[Number<0>{}])),
                    type_convert<bhalf_t>(float(x.AsType<f8_ocp_t>()[Number<1>{}]))};
#endif
}

/**
 * @brief Converts a bf8_ocp_t value to a float value.
 *
 * @param x     The input bf8_ocp_t value.
 * @return      The converted float value.
 */
template <>
inline __host__ __device__ float type_convert<float, bf8_ocp_t>(bf8_ocp_t x)
{
#if CK_OCP_FP8_CVT_FAST_PATH
    union
    {
        unsigned int i32val;
        fp8_storage_t i8val[4];
    } val;
    val.i8val[0] = x.data;
    return __builtin_amdgcn_cvt_f32_bf8(val.i32val, 0);
#else
    return fp8_impl::cast_from_f8<float, bf8_ocp_t::wm, bf8_ocp_t::we, false>(x.data);
#endif
}

/**
 * @brief Converts a vector of 2 bf8_ocp_t values to a vector of 2 float values.
 *
 * @param x     The input vector of 2 bf8_ocp_t values.
 * @return      The converted vector of 2 float values.
 */
template <>
inline __host__ __device__ float2_t type_convert<float2_t, bf8x2_ocp_t>(bf8x2_ocp_t x)
{
#if CK_OCP_FP8_CVT_FAST_PATH
// __builtin_amdgcn_cvt_pk_f32_bf8 can produce incorrect results due to a compiler issue.
// TODO: Enable when SWDEV-532959 is fixed.
#if defined(__gfx12__)
    return float2_t{__builtin_amdgcn_cvt_f32_bf8(bit_cast<uint16_t>(x), 0),
                    __builtin_amdgcn_cvt_f32_bf8(bit_cast<uint16_t>(x), 1)};
#else
    return __builtin_amdgcn_cvt_pk_f32_bf8(bit_cast<uint16_t>(x), false);
#endif
#else
    return float2_t{fp8_impl::cast_from_f8<float, bf8_ocp_t::wm, bf8_ocp_t::we, false>(
                        x.AsType<fp8_storage_t>()[Number<0>{}]),
                    fp8_impl::cast_from_f8<float, bf8_ocp_t::wm, bf8_ocp_t::we, false>(
                        x.AsType<fp8_storage_t>()[Number<1>{}])};
#endif
}

/**
 * @brief Converts a bf8_ocp_t value to a half_t value.
 *
 * @param x     The input bf8_ocp_t value.
 * @return      The converted half_t value.
 */
template <>
inline __host__ __device__ half_t type_convert<half_t, bf8_ocp_t>(bf8_ocp_t x)
{
#if defined(__gfx950__)
    union
    {
        uint16_t i16val;
        fp8_storage_t i8val[2];
    } val;
    val.i8val[0] = x.data;
    return __builtin_amdgcn_cvt_scalef32_pk_f16_bf8(val.i16val, /*scale*/ 1.f, 0)[0];
#else
    return fp8_impl::cast_from_f8<half_t, bf8_ocp_t::wm, bf8_ocp_t::we, false>(x.data);
#endif
}

/**
 * @brief Converts a vector of 2 bf8_ocp_t values to a vector of 2 half_t values.
 *
 * @param x     The input vector of 2 bf8_ocp_t values.
 * @return      The converted vector of 2 half_t values.
 */
template <>
inline __host__ __device__ half2_t type_convert<half2_t, bf8x2_ocp_t>(bf8x2_ocp_t x)
{
#if defined(__gfx950__)
    return __builtin_amdgcn_cvt_scalef32_pk_f16_bf8(bit_cast<uint16_t>(x), /*scale*/ 1.f, 0);
#else
    return half2_t{type_convert<half_t>(float(x.AsType<bf8_ocp_t>()[Number<0>{}])),
                   type_convert<half_t>(float(x.AsType<bf8_ocp_t>()[Number<1>{}]))};
#endif
}

/**
 * @brief Converts a bf8_ocp_t value to a bhalf_t value.
 *
 * @param x     The input bf8_ocp_t value.
 * @return      The converted bhalf_t value.
 */
template <>
inline __host__ __device__ bhalf_t type_convert<bhalf_t, bf8_ocp_t>(bf8_ocp_t x)
{
#if defined(__gfx950__)
    union
    {
        uint16_t i16val;
        fp8_storage_t i8val[2];
    } input;
    input.i8val[0] = x.data;

    union
    {
        bhalf2_t bhalf_vec;
        bhalf_t bhalf_arr[2];
    } output;
    output.bhalf_vec = __builtin_amdgcn_cvt_scalef32_pk_bf16_bf8(input.i16val, /*scale*/ 1.f, 0);

    return output.bhalf_arr[0];
#else
    return type_convert<bhalf_t>(
        fp8_impl::cast_from_f8<float, bf8_ocp_t::wm, bf8_ocp_t::we, false>(x.data));
#endif
}

/**
 * @brief Converts a vector of 2 bf8_ocp_t values to a vector of 2 bhalf_t values.
 *
 * @param x     The input vector of 2 bf8_ocp_t values.
 * @return      The converted vector of 2 bhalf_t values.
 */
template <>
inline __host__ __device__ bhalf2_t type_convert<bhalf2_t, bf8x2_ocp_t>(bf8x2_ocp_t x)
{
#if defined(__gfx950__)
    return __builtin_amdgcn_cvt_scalef32_pk_bf16_bf8(bit_cast<uint16_t>(x), /*scale*/ 1.f, 0);
#else
    return bhalf2_t{type_convert<bhalf_t>(float(x.AsType<bf8_ocp_t>()[Number<0>{}])),
                    type_convert<bhalf_t>(float(x.AsType<bf8_ocp_t>()[Number<1>{}]))};
#endif
}

template <>
inline __host__ __device__ float2_t type_convert<float2_t, pk_i4_t>(pk_i4_t x)
{
    uint8_t x_u8 = ck::bit_cast<uint8_t>(x);

    float x_l = ((x_u8 & 0x0f) >> 0) - 8.f;
    float x_h = ((x_u8 & 0xf0) >> 4) - 8.f;

#ifdef CK_USE_PK4_LAYOUT_SHUFFLE
    float2_t res = {x_h, x_l};
#elif
    float2_t res = {x_l, x_h};
#endif
    return res;
}

template <>
inline __host__ __device__ half2_t type_convert<half2_t, pk_i4_t>(pk_i4_t x)
{
    uint8_t x_u8 = ck::bit_cast<uint8_t>(x);
#ifdef CK_USE_PK4_LAYOUT_SHUFFLE
    uint32_t i4s = ((x_u8 & 0x0f) << 16) | ((x_u8 & 0xf0) >> 4);
#else
    uint32_t i4s = ((x_u8 & 0xf0) << 12) | (x_u8 & 0xf);
#endif

    const int EX  = 0x64006400;
    const int SUB = 0xE408E408; //-8

    int lo = i4s | EX;

    return details::pk_add_f16(bit_cast<half2_t>(lo), bit_cast<half2_t>(SUB));
}

template <>
inline __host__ __device__ bhalf2_t type_convert<bhalf2_t, pk_i4_t>(pk_i4_t x)
{
    uint8_t x_u8 = ck::bit_cast<uint8_t>(x);

    float x_l = ((x_u8 & 0x0f) >> 0) - 8.f;
    float x_h = ((x_u8 & 0xf0) >> 4) - 8.f;

#ifdef CK_USE_PK4_LAYOUT_SHUFFLE
    bhalf2_t res = {type_convert<bhalf_t>(x_h), type_convert<bhalf_t>(x_l)};
#else
    bhalf2_t res = {type_convert<bhalf_t>(x_l), type_convert<bhalf_t>(x_h)};
#endif

    return res;
}

template <>
inline __host__ __device__ half2_t type_convert<half2_t, float2_t>(float2_t x)
{

    const vector_type<float, 2> f32x2_v(x);
    const auto y = __builtin_amdgcn_cvt_pkrtz(f32x2_v.template AsType<float>()[Number<0>{}],
                                              f32x2_v.template AsType<float>()[Number<1>{}]);
    return bit_cast<half2_t>(y);
}

// convert fp16 to fp8
template <>
inline __host__ __device__ f8_fnuz_t type_convert<f8_fnuz_t, half_t>(half_t x)
{
#if CK_USE_SR_F8_CONVERSION
    return f8_convert_sr<f8_fnuz_t>(x);
#else
    return f8_convert_rne<f8_fnuz_t>(x);
#endif
}

/**
 * @brief Converts a half_t value to a f8_ocp_t value with rounding determined by a flag.
 *
 * @param x     The input half_t value.
 * @return      The converted f8_ocp_t value.
 */
template <>
inline __host__ __device__ f8_ocp_t type_convert<f8_ocp_t, half_t>(half_t x)
{
#if CK_USE_SR_F8_CONVERSION
    return f8_convert_sr<f8_ocp_t>(x);
#else
    return f8_convert_rne<f8_ocp_t>(x);
#endif
}

/**
 * @brief Converts a half_t value to a bf8_ocp_t value with rounding determined by a flag.
 *
 * @param x     The input half_t value.
 * @return      The converted bf8_ocp_t value.
 */
template <>
inline __host__ __device__ bf8_ocp_t type_convert<bf8_ocp_t, half_t>(half_t x)
{
#if CK_USE_SR_F8_CONVERSION
    return f8_convert_sr<bf8_ocp_t>(x);
#else
    return f8_convert_rne<bf8_ocp_t>(x);
#endif
}

// convert fp8 to fp16
template <>
inline __host__ __device__ half_t type_convert<half_t, f8_fnuz_t>(f8_fnuz_t x)
{
#if defined(__gfx94__)
    // use native conversion to float and convert to fp16
    return type_convert<half_t>(type_convert<float>(x));
#else
    constexpr bool negative_zero_nan = true;
    return utils::cast_from_f8<f8_fnuz_t, half_t, negative_zero_nan>(x);
#endif
}

// convert fp32 to bf8
template <>
inline __host__ __device__ bf8_fnuz_t type_convert<bf8_fnuz_t, float>(float x)
{
#if CK_USE_SR_F8_CONVERSION
    return f8_convert_sr<bf8_fnuz_t>(x);
#else
    return f8_convert_rne<bf8_fnuz_t>(x);
#endif
}

/**
 * @brief Converts a float value to a f8_ocp_t value with rounding determined by a flag.
 *
 * @param x     The input float value.
 * @return      The converted f8_ocp_t value.
 */
template <>
inline __host__ __device__ f8_ocp_t type_convert<f8_ocp_t, float>(float x)
{
#if CK_USE_SR_F8_CONVERSION
    return f8_convert_sr<f8_ocp_t>(x);
#else
    return f8_convert_rne<f8_ocp_t>(x);
#endif
}

/**
 * @brief Converts a float value to a bf8_ocp_t value with rounding determined by a flag.
 *
 * @param x     The input float value.
 * @return      The converted bf8_ocp_t value.
 */
template <>
inline __host__ __device__ bf8_ocp_t type_convert<bf8_ocp_t, float>(float x)
{
#if CK_USE_SR_F8_CONVERSION
    return f8_convert_sr<bf8_ocp_t>(x);
#else
    return f8_convert_rne<bf8_ocp_t>(x);
#endif
}

/**
 * @brief Converts a bhalf_t value to a f8_ocp_t value with rounding determined by a flag.
 *
 * @param x     The input bhalf_t value.
 * @return      The converted f8_ocp_t value.
 */
template <>
inline __host__ __device__ f8_ocp_t type_convert<f8_ocp_t, bhalf_t>(bhalf_t x)
{
#if CK_USE_SR_F8_CONVERSION
    return f8_convert_sr<f8_ocp_t>(x);
#else
    return f8_convert_rne<f8_ocp_t>(x);
#endif
}

/**
 * @brief Converts a bhalf_t value to a bf8_ocp_t value with rounding determined by a flag.
 *
 * @param x     The input bhalf_t value.
 * @return      The converted bf8_ocp_t value.
 */
template <>
inline __host__ __device__ bf8_ocp_t type_convert<bf8_ocp_t, bhalf_t>(bhalf_t x)
{
#if CK_USE_SR_F8_CONVERSION
    return f8_convert_sr<bf8_ocp_t>(x);
#else
    return f8_convert_rne<bf8_ocp_t>(x);
#endif
}

// convert bf8 to fp32
template <>
inline __host__ __device__ float type_convert<float, bf8_fnuz_t>(bf8_fnuz_t x)
{
#if defined(__gfx94__)
    float fval;
    uint32_t i32val = static_cast<uint32_t>(static_cast<uint8_t>(x));
    fval            = __builtin_amdgcn_cvt_f32_bf8(i32val, 0);
    // asm volatile("v_cvt_f32_bf8 %0, %1 src0_sel:BYTE_0" : "=v"(fval) : "v"(i32val));
    return fval;
#else
    constexpr bool negative_zero_nan = true;
    return utils::cast_from_f8<bf8_fnuz_t, float, negative_zero_nan>(x);
#endif
}

// convert fp16 to bf8
template <>
inline __host__ __device__ bf8_fnuz_t type_convert<bf8_fnuz_t, half_t>(half_t x)
{
#if CK_USE_SR_F8_CONVERSION
    return f8_convert_sr<bf8_fnuz_t>(x);
#else
    return f8_convert_rne<bf8_fnuz_t>(x);
#endif
}

// convert bf8 to fp16
template <>
inline __host__ __device__ half_t type_convert<half_t, bf8_fnuz_t>(bf8_fnuz_t x)
{
#if defined(__gfx94__)
    // use native conversion to float and convert to fp16
    return type_convert<half_t>(type_convert<float>(x));
#else
    constexpr bool negative_zero_nan = true;
    return utils::cast_from_f8<bf8_fnuz_t, half_t, negative_zero_nan>(x);
#endif
}
#ifndef CK_CODE_GEN_RTC
// convert fp32 to fp4 with rounding to nearest even
inline __host__ __device__ f4_t f4_convert_rne(float x, float scale = 1.0f)
{
#if defined(__gfx950__)
    union
    {
        uint32_t bitwise;
        f4_t f4_array[4];
    } value{0};
    value.bitwise = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(value.bitwise, x, x, scale, 0);
    return value.f4_array[0];
#else
    return utils::sat_convert_to_type<f4_t>(x / scale);
#endif
}

// convert vector of 2 fp32 to vector of 2 fp4 with rne
inline __host__ __device__ f4x2_t f4_convert_rne(float2_t x, float scale = 1.0f)
{
#if defined(__gfx950__)
    union
    {
        uint32_t bitwise;
        f4x2_t f4x2_array[4];
    } value{0};
    value.bitwise = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(value.bitwise, x[0], x[1], scale, 0);
    return value.f4x2_array[0];
#else
    union
    {
        uint32_t bitwise;
        f4x2_t f4x2_array[4];
    } value{0};
    uint8_t l     = utils::sat_convert_to_type<f4_t>(x[0] / scale);
    uint8_t h     = utils::sat_convert_to_type<f4_t>(x[1] / scale);
    value.bitwise = (h << 4) | l;
    return value.f4x2_array[0];
#endif
}

// convert vector of 32 fp32 to vector of 32 fp4 with rne
inline __host__ __device__ f4x32_t f4_convert_rne(float32_t x, float scale = 1.0f)
{
#if defined(__gfx950__)
    union
    {
        __uint128_t bitwise;
        f4x2_t f4x2_array[16];
        f4x32_t f4x32_array;
    } f4_values{}, tmp_values{};

    ck::static_for<0, 32 / 2, 1>{}([&](auto idx) {
        tmp_values.bitwise = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(
            tmp_values.bitwise, x[2 * idx], x[2 * idx + 1], scale, 0);
        f4_values.f4x2_array[idx] = tmp_values.f4x2_array[0];
    });

    return f4_values.f4x32_array;
#else
    union
    {
        __uint128_t bitwise;
        f4x2_t f4x2_array[16];
        f4x32_t f4x32_array;
    } f4_values{};

    f4_t tmp;

    ck::static_for<0, 32, 1>{}([&](auto idx) {
        tmp = utils::sat_convert_to_type<f4_t>(x[static_cast<int>(idx)] / scale);
        f4_values.bitwise <<= 4;
        f4_values.bitwise |= tmp;
    });

    return f4_values.f4x32_array;
#endif
}

// convert fp32 to fp4 with stochastic rounding
inline __host__ __device__ f4_t f4_convert_sr(float x, float scale = 1.0f)
{
#if defined(__gfx950__)
    // use HW clock for stochastic input multiply by incremented thread id
    uint32_t rng = __builtin_amdgcn_prng_b32(__builtin_amdgcn_s_memrealtime() *
                                             (get_thread_global_1d_id() + 1));
    union
    {
        uint32_t bitwise;
        f4_t f4_array[4];
    } value{0};
    union
    {
        float float_array[2];
        float2_t float2_array;
    } float_values{{x}};

    value.bitwise = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(
        value.bitwise, float_values.float2_array, rng, scale, 0);
    return value.f4_array[0];
#else
    constexpr int seed = 1254739;
#ifndef CK_CODE_GEN_RTC
    uint32_t rng = prand_generator<float, seed>(reinterpret_cast<uintptr_t>(&x), x);
#else
    uint32_t rng = prand_generator<float, seed>(reinterpret_cast<size_t>(&x), x);
#endif
    return utils::sat_convert_to_type_sr<f4_t>(x / scale, rng);
#endif
}

// convert vector of 2 fp32 to vector of 2 fp4 with sr
inline __host__ __device__ f4x2_t f4_convert_sr(float2_t x, float scale = 1.0f)
{
#if defined(__gfx950__)
    // use HW clock for stochastic input multiply by incremented thread id
    uint32_t rng = __builtin_amdgcn_prng_b32(__builtin_amdgcn_s_memrealtime() *
                                             (get_thread_global_1d_id() + 1));
    union
    {
        uint32_t bitwise;
        f4x2_t f4x2_array[4];
    } value{0};
    value.bitwise = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(value.bitwise, x, rng, scale, 0);
    return value.f4x2_array[0];
#else
    constexpr int seed = 1254739;
#ifndef CK_CODE_GEN_RTC
    uint32_t rng = prand_generator<float, seed>(reinterpret_cast<uintptr_t>(&x), x[0]);
#else
    uint32_t rng = prand_generator<float, seed>(reinterpret_cast<size_t>(&x), x[0]);
#endif
    union
    {
        uint32_t bitwise;
        f4x2_t f4x2_array[4];
    } value{0};
    uint8_t l     = utils::sat_convert_to_type_sr<f4_t>(x[0] / scale, rng);
    uint8_t h     = utils::sat_convert_to_type_sr<f4_t>(x[1] / scale, rng);
    value.bitwise = (h << 4) | l;
    return value.f4x2_array[0];
#endif
}

// convert vector of 32 fp32 to vector of 32 fp4 with sr
inline __host__ __device__ f4x32_t f4_convert_sr(float32_t x, float scale = 1.0f)
{
#if defined(__gfx950__)
    // use HW clock for stochastic input multiply by incremented thread id
    uint32_t rng = __builtin_amdgcn_prng_b32(__builtin_amdgcn_s_memrealtime() *
                                             (get_thread_global_1d_id() + 1));
    union
    {
        __uint128_t bitwise;
        f4x2_t f4x2_array[16];
        f4x32_t f4x32_array;
    } f4_values{0};
    union
    {
        float2_t floatx2_array[16];
        float32_t floatx32_array;
    } float_values{{0}};
    float_values.floatx32_array = x;

    ck::static_for<0, 32 / 2, 1>{}([&](auto idx) {
        f4_values.f4x2_array[idx] = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(
            f4_values.bitwise, float_values.floatx2_array[idx], rng, scale, 0);
    });

    return f4_values.f4x32_array;
#else
    constexpr int seed = 1254739;
#ifndef CK_CODE_GEN_RTC
    uint32_t rng = prand_generator<float, seed>(reinterpret_cast<uintptr_t>(&x), x[0]);
#else
    uint32_t rng = prand_generator<float, seed>(reinterpret_cast<size_t>(&x), x[0]);
#endif
    union
    {
        __uint128_t bitwise;
        f4x2_t f4x2_array[16];
        f4x32_t f4x32_array;
    } f4_values{0};

    f4_t tmp;

    ck::static_for<0, 32, 1>{}([&](auto idx) {
        tmp = utils::sat_convert_to_type_sr<f4_t>(x[static_cast<int>(idx)] / scale, rng);
        f4_values.bitwise <<= 4;
        f4_values.bitwise |= tmp;
    });

    return f4_values.f4x32_array;
#endif
}

// convert fp32 to fp4
template <>
inline __host__ __device__ f4_t type_convert<f4_t, float>(float x)
{
#if CK_USE_SR_F4_CONVERSION
    return f4_convert_sr(x);
#else
    return f4_convert_rne(x);
#endif
}

// convert vector of 2 fp32 to vector of 2 fp4
template <>
inline __host__ __device__ f4x2_t type_convert<f4x2_t, float2_t>(float2_t x)
{
#if CK_USE_SR_F4_CONVERSION
    return f4_convert_sr(x);
#else
    return f4_convert_rne(x);
#endif
}
template <>
inline __host__ __device__ f4x2_pk_t type_convert<f4x2_pk_t, float2_t>(float2_t x)
{
    return static_cast<f4x2_pk_t>(type_convert<f4x2_t>(x));
}

// convert vector of 32 fp32 to vector of 32 fp4
template <>
inline __host__ __device__ f4x32_t type_convert<f4x32_t, float32_t>(float32_t x)
{
#if CK_USE_SR_F4_CONVERSION
    return f4_convert_sr(x);
#else
    return f4_convert_rne(x);
#endif
}

// convert fp4 to fp32
template <>
inline __host__ __device__ float type_convert<float, f4_t>(f4_t x)
{
#if defined(__gfx950__)
    union
    {
        float float_array[2];
        float2_t float2_array;
    } float_values{};
    float scale               = 1.0f;
    float_values.float2_array = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(x, scale, 0);
    return float_values.float_array[0];
#else
    return utils::to_float<f4_t>(NumericLimits<e8m0_bexp_t>::Binary_1(), x);
#endif
}

// convert vector of 2 fp4 to vector of 2 fp32
template <>
inline __host__ __device__ float2_t type_convert<float2_t, f4x2_t>(f4x2_t x)
{
#if defined(__gfx950__)
    union
    {
        uint32_t bitwise;
        f4x2_t f4x2_array[4];
    } value{};
    value.f4x2_array[0] = x;
    float scale         = 1.0f;
    return __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(value.bitwise, scale, 0);
#else
    float2_t ret{
        utils::to_float<f4_t>(NumericLimits<e8m0_bexp_t>::Binary_1(),
                              x.template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{})),
        utils::to_float<f4_t>(NumericLimits<e8m0_bexp_t>::Binary_1(),
                              x.template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{}))};
    return ret;
#endif
}

// convert vector of 32 fp4 to vector of 32 fp32
template <>
inline __host__ __device__ float32_t type_convert<float32_t, f4x32_t>(f4x32_t x)
{
#if defined(__gfx950__)
    union
    {
        f4x32_t f4x32_array;
        f4x2_t fp4x2[16];
    } value{x};
    float2_t op;
    float32_t ret;
    float scale = 1.0f;

    ck::static_for<0, 32 / 2, 1>{}([&](auto idx) {
        op               = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(value.fp4x2[idx], scale, 0);
        ret[2 * idx]     = op[0];
        ret[2 * idx + 1] = op[1];
    });

    return ret;
#else
    union
    {
        float32_t float32_array;
        float float_array[32];
    } float_values{};
    union
    {
        __uint128_t bitwise;
        f4x2_t f4x2_array[16];
        f4x32_t f4x32_array;
    } f4_values{bit_cast<__uint128_t>(x)};

    ck::static_for<0, 32 / 2, 1>{}([&](auto idx) {
        float_values.float_array[2 * idx] = utils::to_float<f4_t>(
            NumericLimits<e8m0_bexp_t>::Binary_1(),
            f4_values.f4x2_array[idx].template AsType<f4x2_pk_t>()[Number<0>{}].template unpack<>(
                Number<0>{}));

        float_values.float_array[2 * idx + 1] = utils::to_float<f4_t>(
            NumericLimits<e8m0_bexp_t>::Binary_1(),
            f4_values.f4x2_array[idx].template AsType<f4x2_pk_t>()[Number<0>{}].template unpack<>(
                Number<1>{}));
    });

    return float_values.float32_array;
#endif
}

/**
 * @brief Converts a float to a 6-bit float type (f6_t) using round-to-nearest-even.
 *
 * Divides the input by the specified scale, then saturates and converts it
 * to the 6-bit floating-point format (f6_t).
 *
 * @param x     The input float value.
 * @param scale A scaling factor applied to `x` before conversion.
 * @return      The converted f6_t value.
 */
inline __host__ __device__ f6_t f6_convert_rne(float x, float scale = 1.0f)
{
#if defined(__gfx950__)
    float16_t in1{x};
    float16_t in2{};

    union
    {
        f6x32_t f6_vector;
        f6_t f6_array[32];
    } out{};

    out.f6_vector = f6x32_t{__builtin_amdgcn_cvt_scalef32_2xpk16_fp6_f32(in1, in2, scale)};

    return out.f6_array[0];
#else
    return utils::sat_convert_to_type<f6_t>(x / scale);
#endif
}

/**
 * @brief Converts a 32-element single-precision float array into a packed 6-bit representation.
 *
 * This function divides each input float by the provided scale value, then performs conversion with
 * rounding to nearest / even to pack each element into 6 bits of precision.
 *
 * @param x     A vector of 32 floats stored in float32_t.
 * @param scale A scaling factor for each float before conversion.
 * @return An f6x32_t object storing the compressed 6-bit representation.
 */
inline __host__ __device__ f6x32_t f6_convert_rne(float32_t x, float scale = 1.0f)
{
#if defined(__gfx950__)
    float16_t* in1 = reinterpret_cast<float16_t*>(&x);
    float16_t* in2 = reinterpret_cast<float16_t*>(&x + 16);
    return f6x32_t{__builtin_amdgcn_cvt_scalef32_2xpk16_fp6_f32(*in1, *in2, scale)};
#else
    union
    {
        float32_t float_vector;
        float float_array[32];
    } in{x};

    using array_type = uint8_t __attribute__((ext_vector_type(32)));
    array_type uint8_array;

    // collect the 6-bit values into an array
    ck::static_for<0, 32, 1>{}([&](auto i) {
        uint8_array[static_cast<index_t>(i)] =
            utils::sat_convert_to_type<f6_t>(in.float_array[i] / scale);
    });
    return f6x32_t{f6x32_pk_t{uint8_array}};
#endif
}

/**
 * @brief Converts a float to the 6-bit floating-point type (f6_t) using stochastic rounding.
 *
 * Divides the input by the specified scale, then performs saturation and conversion
 * to f6_t based on a pseudo-randomly generated seed.
 *
 * @param x     The input float value.
 * @param scale A scaling factor applied to `x` before conversion.
 * @return      The converted f6_t value.
 */
inline __host__ __device__ f6_t f6_convert_sr(float x, float scale = 1.0f)
{
#if defined(__gfx950__)
    // use HW clock for stochastic input multiply by incremented thread id
    uint32_t rng = __builtin_amdgcn_prng_b32(__builtin_amdgcn_s_memrealtime() *
                                             (get_thread_global_1d_id() + 1));
    union
    {
        float32_t float_vector;
        float float_array[32];
    } in{x};

    union
    {
        f6x32_t f6_vector;
        f6_t f6_array[32];
    } out{};

    out.f6_vector =
        f6x32_t{__builtin_amdgcn_cvt_scalef32_sr_pk32_fp6_f32(in.float_vector, rng, scale)};

    return out.f6_array[0];
#else
    constexpr int seed = 1254739;
#ifndef CK_CODE_GEN_RTC
    uint32_t rng = prand_generator<float, seed>(reinterpret_cast<uintptr_t>(&x), x);
#else
    uint32_t rng = prand_generator<float, seed>(reinterpret_cast<size_t>(&x), x);
#endif
    return utils::sat_convert_to_type_sr<f6_t>(x / scale, rng);
#endif
}

/**
 * @brief Converts a 32-element single-precision float array into a packed 6-bit representation.
 *
 * This function divides each input float by the provided scale value, then performs conversion with
 * stochastic rounding to pack each element into 6 bits of precision.
 *
 * @param x     A vector of 32 floats stored in float32_t.
 * @param scale A scaling factor for each float before conversion.
 * @return An f6x32_t object storing the compressed 6-bit representation.
 */
inline __host__ __device__ f6x32_t f6_convert_sr(float32_t x, float scale = 1.0f)
{
#if defined(__gfx950__)
    // use HW clock for stochastic input multiply by incremented thread id
    uint32_t rng = __builtin_amdgcn_prng_b32(__builtin_amdgcn_s_memrealtime() *
                                             (get_thread_global_1d_id() + 1));
    return f6x32_t{__builtin_amdgcn_cvt_scalef32_sr_pk32_fp6_f32(x, rng, scale)};
#else
    constexpr int seed = 1254739;
    union
    {
        float32_t float_vector;
        float float_array[32];
    } float_values{x};
#ifndef CK_CODE_GEN_RTC
    uint32_t rng =
        prand_generator<float, seed>(reinterpret_cast<uintptr_t>(&x), float_values.float_array[0]);
#else
    uint32_t rng =
        prand_generator<float, seed>(reinterpret_cast<size_t>(&x), float_values.float_array[0]);
#endif

    union
    {
        float32_t float_vector;
        float float_array[32];
    } in{x};

    union
    {
        f6x32_t f6_vector;
        f6_t f6_array[32];
    } out{};

    ck::static_for<0, 32, 1>{}([&](auto i) {
        out.f6_array[i] = utils::sat_convert_to_type_sr<f6_t>(in.float_array[i] / scale, rng);
    });

    return out.f6_vector;
#endif
}

/**
 * @brief Specializes the type conversion template for converting a float into the 6-bit float type
 * (f6_t).
 *
 * Depending on the CK_USE_SR_F6_CONVERSION flag,
 * the conversion uses stochastic rounding
 * or round-to-nearest-even.
 *
 * @param x Input float value to be converted.
 * @return  The converted f6_t value.
 */
template <>
inline __host__ __device__ f6_t type_convert<f6_t, float>(float x)
{
#if CK_USE_SR_F6_CONVERSION
    return f6_convert_sr(x);
#else
    return f6_convert_rne(x);
#endif
}

/**
 * @brief Specializes the type conversion template for converting a vector of 32 floats into the
 * vector of 32 6-bit float types (f6x32_t).
 *
 * Depending on the CK_USE_SR_F6_CONVERSION flag,
 * the conversion uses stochastic rounding
 * or round-to-nearest-even.
 *
 * @param x Input float value to be converted.
 * @return  The converted f6x32_t vector.
 */
template <>
inline __host__ __device__ f6x32_t type_convert<f6x32_t, float32_t>(float32_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return f6_convert_sr(x);
#else
    return f6_convert_rne(x);
#endif
}

template <>
inline __host__ __device__ f6x32_pk_t type_convert<f6x32_pk_t, float32_t>(float32_t x)
{
    return static_cast<f6x32_pk_t>(type_convert<f6x32_t>(x));
}

template <>
inline __host__ __device__ f6x16_t type_convert<f6x16_t, float16_t>(float16_t x)
{

    union
    {
        float16_t v16x2[2];
        float32_t v32;
    } in{{x, x}};

    union
    {
        f6x32_t v32;
        f6x16_t v16x2[2];
    } out{};

#if CK_USE_SR_F6_CONVERSION
    out.v32 = f6_convert_sr(in.v32);
#else
    out.v32 = f6_convert_rne(in.v32);
#endif

    return out.v16x2[0];
}

template <>
inline __host__ __device__ f6x16_pk_t type_convert<f6x16_pk_t, float16_t>(float16_t x)
{
    return static_cast<f6x16_pk_t>(type_convert<f6x16_t>(x));
}

/**
 * @brief Specializes the type conversion template for converting the 6-bit float type (f6_t) to
 * float.
 *
 * Interprets an f6_t value as a float using the default scale factor of 1.
 *
 * @param x The 6-bit float (f6_t) value to be converted.
 * @return  The corresponding float representation.
 */
template <>
inline __host__ __device__ float type_convert<float, f6_t>(f6_t x)
{
#if defined(__gfx950__)
    union
    {
        f6_t f6_array[32];
        f6x32_t f6_vector;
    } in{{x}};

    union
    {
        float32_t float_vector;
        float float_array[32];
    } out{};

    out.float_vector = __builtin_amdgcn_cvt_scalef32_pk32_f32_fp6(
        in.f6_vector.template AsType<f6x32_t::data_t>()[Number<0>{}],
        type_convert<float>(NumericLimits<e8m0_bexp_t>::Binary_1()));
    return out.float_array[0];
#else
    return utils::to_float<f6_t>(NumericLimits<e8m0_bexp_t>::Binary_1(), x);
#endif
}

/**
 * @brief Specializes the type conversion template for converting the vector of 32 6-bit float
 * types (f6x32_t) to vector of 32 floats.
 *
 * Interprets an f6_t values as floats using the default scale factor of 1.
 *
 * @param x The vector of 32 6-bit float (f6x32_t) values to be converted.
 * @return  The corresponding float representation.
 */
template <>
inline __host__ __device__ float32_t type_convert<float32_t, f6x32_t>(f6x32_t x)
{
#if defined(__gfx950__)
    return __builtin_amdgcn_cvt_scalef32_pk32_f32_fp6(
        x.template AsType<f6x32_t::data_t>()[Number<0>{}],
        type_convert<float>(NumericLimits<e8m0_bexp_t>::Binary_1()));
#else
    union
    {
        f6x32_t f6_vector;
        f6_t f6_array[32];
    } in{x};

    union
    {
        float32_t float_vector;
        float float_array[32];
    } out{};

    ck::static_for<0, 32, 1>{}([&](auto i) {
        out.float_array[i] =
            utils::to_float<f6_t>(NumericLimits<e8m0_bexp_t>::Binary_1(), in.f6_array[i]);
    });

    return out.float_vector;
#endif
}

template <>
inline __host__ __device__ float16_t type_convert<float16_t, f6x16_t>(f6x16_t x)
{
    union
    {
        f6x16_t v16x2[2];
        f6x32_t v32;
    } in{{x, x}};

    union
    {
        float16_t v16x2[2];
        float32_t v32;
    } out{};

    out.v32 = type_convert<float32_t>(in.v32);
    return out.v16x2[0];
}

template <>
inline __host__ __device__ float16_t type_convert<float16_t, f6x16_pk_t>(f6x16_pk_t x)
{
    return type_convert<float16_t>(static_cast<f6x16_t>(x));
}

/**
 * @brief Converts a float to the 6-bit BF6 type using round-to-nearest-even.
 *
 * Divides the input by the specified scale, then saturates and converts
 * it to a 6-bit BF6 floating-point format.
 *
 * @param x     The float value to be converted.
 * @param scale The scaling factor applied to the input before conversion.
 * @return      The converted bf6_t value.
 */
inline __host__ __device__ bf6_t bf6_convert_rne(float x, float scale = 1.0f)
{
#if defined(__gfx950__)
    float16_t in1{x};
    float16_t in2{};

    union
    {
        bf6x32_t bf6_vector;
        bf6_t bf6_array[32];
    } out{};

    out.bf6_vector = bf6x32_t{__builtin_amdgcn_cvt_scalef32_2xpk16_bf6_f32(in1, in2, scale)};

    return out.bf6_array[0];
#else
    return utils::sat_convert_to_type<bf6_t>(x / scale);
#endif
}

/**
 * @brief Converts a vector of 32 floats to the vector of 32 6-bit BF6 types using
 * round-to-nearest-even.
 *
 * Divides the input by the specified scale, then saturates and converts
 * it to a 6-bit BF6 floating-point format.
 *
 * @param x     The float vector to be converted.
 * @param scale The scaling factor applied to the input before conversion.
 * @return      The converted bf6x32_t vector.
 */
inline __host__ __device__ bf6x32_t bf6_convert_rne(float32_t x, float scale = 1.0f)
{
#if defined(__gfx950__)
    float16_t* in1 = reinterpret_cast<float16_t*>(&x);
    float16_t* in2 = reinterpret_cast<float16_t*>(&x + 16);
    return bf6x32_t{__builtin_amdgcn_cvt_scalef32_2xpk16_bf6_f32(*in1, *in2, scale)};
#else
    union
    {
        float32_t float_vector;
        float float_array[32];
    } in{x};

    using array_type = uint8_t __attribute__((ext_vector_type(32)));
    array_type uint8_array;

    // collect the 6-bit values into an array
    ck::static_for<0, 32, 1>{}([&](auto i) {
        uint8_array[static_cast<index_t>(i)] =
            utils::sat_convert_to_type<bf6_t>(in.float_array[i] / scale);
    });
    return bf6x32_t{bf6x32_pk_t{uint8_array}};
#endif
}

/**
 * @brief Converts a float to the 6-bit BF6 type using stochastic rounding.
 *
 * Divides the input by the specified scale,
 * and converts the result to a 6-bit BF6 floating-point
 * format with stochastic rounding.
 *
 * @param x     The float value to be converted.
 * @param scale The scaling factor applied to the input before conversion.
 * @return      The converted bf6_t value.
 */
inline __host__ __device__ bf6_t bf6_convert_sr(float x, float scale = 1.0f)
{
#if defined(__gfx950__)
    // use HW clock for stochastic input multiply by incremented thread id
    uint32_t rng = __builtin_amdgcn_prng_b32(__builtin_amdgcn_s_memrealtime() *
                                             (get_thread_global_1d_id() + 1));
    union
    {
        float32_t float_vector;
        float float_array[32];
    } in{x};

    union
    {
        bf6x32_t bf6_vector;
        bf6_t bf6_array[32];
    } out{};

    out.bf6_vector =
        bf6x32_t{__builtin_amdgcn_cvt_scalef32_sr_pk32_bf6_f32(in.float_vector, rng, scale)};

    return out.bf6_array[0];
#else
    constexpr int seed = 1254739;
#ifndef CK_CODE_GEN_RTC
    uint32_t rng = prand_generator<float, seed>(reinterpret_cast<uintptr_t>(&x), x);
#else
    uint32_t rng = prand_generator<float, seed>(reinterpret_cast<size_t>(&x), x);
#endif
    return utils::sat_convert_to_type_sr<bf6_t>(x / scale, rng);
#endif
}

/**
 * @brief Converts a vector of 32 floats to the vector of 32 6-bit BF6 types using stochastic
 * rounding.
 *
 * Divides the input by the specified scale,
 * and converts the result to a 6-bit BF6 floating-point
 * format with stochastic rounding.
 *
 * @param x     The float vector to be converted.
 * @param scale The scaling factor applied to the input before conversion.
 * @return      The converted bf6x32_t vector.
 */
inline __host__ __device__ bf6x32_t bf6_convert_sr(float32_t x, float scale = 1.0f)
{
#if defined(__gfx950__)
    // use HW clock for stochastic input multiply by incremented thread id
    uint32_t rng = __builtin_amdgcn_prng_b32(__builtin_amdgcn_s_memrealtime() *
                                             (get_thread_global_1d_id() + 1));
    return bf6x32_t{__builtin_amdgcn_cvt_scalef32_sr_pk32_bf6_f32(x, rng, scale)};
#else
    constexpr int seed = 1254739;
    union
    {
        float32_t float_vector;
        float float_array[32];
    } float_values{x};
#ifndef CK_CODE_GEN_RTC
    uint32_t rng =
        prand_generator<float, seed>(reinterpret_cast<uintptr_t>(&x), float_values.float_array[0]);
#else
    uint32_t rng =
        prand_generator<float, seed>(reinterpret_cast<size_t>(&x), float_values.float_array[0]);
#endif
    union
    {
        float32_t float_vector;
        float float_array[32];
    } in{x};

    union
    {
        bf6x32_t bf6_vector;
        bf6_t bf6_array[32];
    } out{};

    ck::static_for<0, 32, 1>{}([&](auto i) {
        out.bf6_array[i] = utils::sat_convert_to_type_sr<bf6_t>(in.float_array[i] / scale, rng);
    });

    return out.bf6_vector;
#endif
}

/**
 * @brief Specializes float-to-bf6_t conversion.
 *
 * Uses stochastic rounding if CK_USE_SR_F6_CONVERSION is defined,
 * otherwise uses round-to-nearest-even.
 *
 * @param x Input float value to convert.
 * @return Converted bf6_t value.
 */
template <>
inline __host__ __device__ bf6_t type_convert<bf6_t, float>(float x)
{
#if CK_USE_SR_F6_CONVERSION
    return bf6_convert_sr(x);
#else
    return bf6_convert_rne(x);
#endif
}

/**
 * @brief Specializes vector of 32 float-to-bf6_t conversion.
 *
 * Uses stochastic rounding if CK_USE_SR_F6_CONVERSION is defined,
 * otherwise uses round-to-nearest-even.
 *
 * @param x Input float vector to convert.
 * @return Converted bf6x32_t vector.
 */
template <>
inline __host__ __device__ bf6x32_t type_convert<bf6x32_t, float32_t>(float32_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return bf6_convert_sr(x);
#else
    return bf6_convert_rne(x);
#endif
}

template <>
inline __host__ __device__ bf6x32_pk_t type_convert<bf6x32_pk_t, float32_t>(float32_t x)
{
    return static_cast<bf6x32_pk_t>(type_convert<bf6x32_t>(x));
}

template <>
inline __host__ __device__ bf6x16_t type_convert<bf6x16_t, float16_t>(float16_t x)
{

    union
    {
        float16_t v16x2[2];
        float32_t v32;
    } in{{x, x}};

    union
    {
        bf6x32_t v32;
        bf6x16_t v16x2[2];
    } out{};

#if CK_USE_SR_F6_CONVERSION
    out.v32 = bf6_convert_sr(in.v32);
#else
    out.v32 = bf6_convert_rne(in.v32);
#endif

    return out.v16x2[0];
}

template <>
inline __host__ __device__ bf6x16_pk_t type_convert<bf6x16_pk_t, float16_t>(float16_t x)
{
    return static_cast<bf6x16_pk_t>(type_convert<bf6x16_t>(x));
}

/**
 * @brief Specializes the type conversion template for converting a bf6_t value to float.
 *
 * Interprets the bf6_t value using the default scale factor of 1 and returns
 * its floating-point representation.
 *
 * @param x The bf6_t value to convert.
 * @return  The float representation of the given bf6_t value.
 */
template <>
inline __host__ __device__ float type_convert<float, bf6_t>(bf6_t x)
{
#if defined(__gfx950__)
    union
    {
        bf6_t bf6_array[32];
        bf6x32_t bf6_vector;
    } in{{x}};

    union
    {
        float32_t float_vector;
        float float_array[32];
    } out{};

    out.float_vector = __builtin_amdgcn_cvt_scalef32_pk32_f32_bf6(
        in.bf6_vector.template AsType<bf6x32_t::data_t>()[Number<0>{}],
        type_convert<float>(NumericLimits<e8m0_bexp_t>::Binary_1()));
    return out.float_array[0];
#else
    return utils::to_float<bf6_t>(NumericLimits<e8m0_bexp_t>::Binary_1(), x);
#endif
}

/**
 * @brief Specializes the type conversion template for converting a vector of 32 bf6_t values to
 * vector of 32 floats.
 *
 * Interprets the bf6x32_t value using the default scale factor of 1 and returns
 * its floating-point representation.
 *
 * @param x The bf6x32_t value to convert.
 * @return  The float representation of the given vector.
 */
template <>
inline __host__ __device__ float32_t type_convert<float32_t, bf6x32_t>(bf6x32_t x)
{
#if defined(__gfx950__)
    return __builtin_amdgcn_cvt_scalef32_pk32_f32_bf6(
        x.template AsType<bf6x32_t::data_t>()[Number<0>{}],
        type_convert<float>(NumericLimits<e8m0_bexp_t>::Binary_1()));
#else
    union
    {
        bf6x32_t bf6_vector;
        bf6_t bf6_array[32];
    } in{x};

    union
    {
        float32_t float_vector;
        float float_array[32];
    } out{};

    ck::static_for<0, 32, 1>{}([&](auto i) {
        out.float_array[i] =
            utils::to_float<bf6_t>(NumericLimits<e8m0_bexp_t>::Binary_1(), in.bf6_array[i]);
    });

    return out.float_vector;
#endif
}

template <>
inline __host__ __device__ float16_t type_convert<float16_t, bf6x16_t>(bf6x16_t x)
{
    union
    {
        bf6x16_t v16x2[2];
        bf6x32_t v32;
    } in{{x, x}};

    union
    {
        float16_t v16x2[2];
        float32_t v32;
    } out{};

    out.v32 = type_convert<float32_t>(in.v32);
    return out.v16x2[0];
}

template <>
inline __host__ __device__ float16_t type_convert<float16_t, bf6x16_pk_t>(bf6x16_pk_t x)
{
    return type_convert<float16_t>(static_cast<bf6x16_t>(x));
}

#endif
#if !defined(__HIPCC_RTC__) || !defined(CK_CODE_GEN_RTC)
template <typename Y, typename X, size_t NumElems>
inline __host__ __device__ void array_convert(std::array<Y, NumElems>& y,
                                              const std::array<X, NumElems>& x)
{
    for(size_t i = 0; i < NumElems; i++)
    {
        y[i] = type_convert<Y>(x[i]);
    }
}
#endif

template <typename Y, typename X, index_t NumElems>
inline __host__ __device__ void array_convert(Array<Y, NumElems>& y, const Array<X, NumElems>& x)
{
    for(size_t i = 0; i < NumElems; i++)
    {
        y[i] = type_convert<Y>(x[i]);
    }
}

} // namespace ck
