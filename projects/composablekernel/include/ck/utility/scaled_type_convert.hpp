// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/type_convert.hpp"
#include "ck/utility/mxf8_utils.hpp"

#ifdef CK_USE_NATIVE_MX_SUPPORT
#define CK_USE_NATIVE_MX_SUPPORT 1
#else
#define CK_USE_NATIVE_MX_SUPPORT 0
#endif

namespace ck {

// Declare a template function for scaled conversion
template <typename Y, typename X>
#if CK_USE_OCP_FP8
__host__ __device__ constexpr Y scaled_type_convert(e8m0_bexp_t scale, X x);
#else
__host__ constexpr Y scaled_type_convert(e8m0_bexp_t scale, X x);
#endif

// convert f8_ocp_t to fp32
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ float scaled_type_convert<float, f8_ocp_t>(e8m0_bexp_t scale, f8_ocp_t x)
#else
inline __host__ float scaled_type_convert<float, f8_ocp_t>(e8m0_bexp_t scale, f8_ocp_t x)
#endif
{

#if CK_MX_FP8_CVT_FAST_PATH
    return fp8_impl::cast_to_f32_from_f8_scaled<f8_ocp_t::default_interpret>(
        type_convert<float>(scale), x.data);
#else
    return type_convert<float>(scale) * type_convert<float>(x);
#endif
}

// convert bf8_ocp_t to fp32
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ float scaled_type_convert<float, bf8_ocp_t>(e8m0_bexp_t scale,
                                                                       bf8_ocp_t x)
#else
inline __host__ float scaled_type_convert<float, bf8_ocp_t>(e8m0_bexp_t scale, bf8_ocp_t x)
#endif
{

#if CK_MX_FP8_CVT_FAST_PATH
    return fp8_impl::cast_to_f32_from_f8_scaled<bf8_ocp_t::default_interpret>(
        type_convert<float>(scale), x.data);
#else
    return type_convert<float>(scale) * type_convert<float>(x);
#endif
}

// convert 2 x f8_ocp_t to 2 x fp32
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ float2_t scaled_type_convert<float2_t, f8x2_ocp_t>(e8m0_bexp_t scale,
                                                                              f8x2_ocp_t x)
#else
inline __host__ float2_t scaled_type_convert<float2_t, f8x2_ocp_t>(e8m0_bexp_t scale, f8x2_ocp_t x)
#endif
{
#if CK_MX_FP8_CVT_FAST_PATH
    return fp8_impl::cast_to_f32_from_f8_scaled<f8_ocp_t::default_interpret>(
        type_convert<float>(scale), x.AsType<fp8_impl::fp8x2_storage_t>()[Number<0>{}]);
#else
    return float2_t{scaled_type_convert<float>(scale, x.AsType<f8_ocp_t>()[Number<0>{}]),
                    scaled_type_convert<float>(scale, x.AsType<f8_ocp_t>()[Number<1>{}])};
#endif
}

// convert 2 x bf8_ocp_t to 2 x fp32
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ float2_t scaled_type_convert<float2_t, bf8x2_ocp_t>(e8m0_bexp_t scale,
                                                                               bf8x2_ocp_t x)
#else
inline __host__ float2_t scaled_type_convert<float2_t, bf8x2_ocp_t>(e8m0_bexp_t scale,
                                                                    bf8x2_ocp_t x)
#endif
{
#if CK_MX_FP8_CVT_FAST_PATH
    return fp8_impl::cast_to_f32_from_f8_scaled<bf8_ocp_t::default_interpret>(
        type_convert<float>(scale), x.AsType<fp8_impl::fp8x2_storage_t>()[Number<0>{}]);
#else
    return float2_t{scaled_type_convert<float>(scale, x.AsType<bf8_ocp_t>()[Number<0>{}]),
                    scaled_type_convert<float>(scale, x.AsType<bf8_ocp_t>()[Number<1>{}])};
#endif
}

// convert 16 x f8_ocp_t to 16 x fp32
// @note Host version gives compilation error. Requires extra compiler options.
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ float16_t scaled_type_convert<float16_t, f8x16_ocp_t>(e8m0_bexp_t scale,
                                                                                 f8x16_ocp_t x)
#else
inline __host__ float16_t scaled_type_convert<float16_t, f8x16_ocp_t>(e8m0_bexp_t scale,
                                                                      f8x16_ocp_t x)
#endif
{
    union
    {
        f8x16_ocp_t f8_1x16;
        f8x2_ocp_t f8_2x8[8];
    } in{x};
    union
    {
        float16_t float_1x16;
        float2_t float_2x8[8];
    } out{};

    ck::static_for<0, 8, 1>{}([&](auto i) {
        out.float_2x8[i] = scaled_type_convert<float2_t, f8x2_ocp_t>(scale, in.f8_2x8[i]);
    });

    return out.float_1x16;
}

// convert 16 x bf8_ocp_t to 16 x fp32
// @note Host version gives compilation error. Requires extra compiler options.
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ float16_t scaled_type_convert<float16_t, bf8x16_ocp_t>(e8m0_bexp_t scale,
                                                                                  bf8x16_ocp_t x)
#else
inline __host__ float16_t scaled_type_convert<float16_t, bf8x16_ocp_t>(e8m0_bexp_t scale,
                                                                       bf8x16_ocp_t x)
#endif
{
    union
    {
        bf8x16_ocp_t bf8_1x16;
        bf8x2_ocp_t bf8_2x8[8];
    } in{x};
    union
    {
        float16_t float_1x16;
        float2_t float_2x8[8];
    } out{};

    ck::static_for<0, 8, 1>{}([&](auto i) {
        out.float_2x8[i] = scaled_type_convert<float2_t, bf8x2_ocp_t>(scale, in.bf8_2x8[i]);
    });

    return out.float_1x16;
}

// convert 32 x f8_ocp_t to 32 x fp32
// @note Host version gives compilation error. Requires extra compiler options.
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ float32_t scaled_type_convert<float32_t, f8x32_ocp_t>(e8m0_bexp_t scale,
                                                                                 f8x32_ocp_t x)
#else
inline __host__ float32_t scaled_type_convert<float32_t, f8x32_ocp_t>(e8m0_bexp_t scale,
                                                                      f8x32_ocp_t x)
#endif
{
    union
    {
        f8x32_ocp_t f8_1x32;
        f8x16_ocp_t f8_16x2[2];
    } in{x};
    union
    {
        float32_t float_1x32;
        float16_t float_16x2[2];
    } out{};

    ck::static_for<0, 2, 1>{}([&](auto i) {
        out.float_16x2[i] = scaled_type_convert<float16_t, f8x16_ocp_t>(scale, in.f8_16x2[i]);
    });

    return out.float_1x32;
}

// convert 32 x bf8_ocp_t to 32 x fp32
// @note Host version gives compilation error. Requires extra compiler options.
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ float32_t scaled_type_convert<float32_t, bf8x32_ocp_t>(e8m0_bexp_t scale,
                                                                                  bf8x32_ocp_t x)
#else
inline __host__ float32_t scaled_type_convert<float32_t, bf8x32_ocp_t>(e8m0_bexp_t scale,
                                                                       bf8x32_ocp_t x)
#endif
{
    union
    {
        bf8x32_ocp_t bf8_1x32;
        bf8x16_ocp_t bf8_16x2[2];
    } in{x};
    union
    {
        float32_t float_1x32;
        float16_t float_16x2[2];
    } out{};

    ck::static_for<0, 2, 1>{}([&](auto i) {
        out.float_16x2[i] = scaled_type_convert<float16_t, bf8x16_ocp_t>(scale, in.bf8_16x2[i]);
    });

    return out.float_1x32;
}

// convert fp32 to fp8
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ f8_ocp_t scaled_type_convert<f8_ocp_t, float>(e8m0_bexp_t scale, float x)
#else
inline __host__ f8_ocp_t scaled_type_convert<f8_ocp_t, float>(e8m0_bexp_t scale, float x)
#endif
{
#if CK_USE_SR_F8_CONVERSION
    return mxf8_convert_sr<f8_ocp_t>(x, type_convert<float>(scale));
#else
    return mxf8_convert_rne<f8_ocp_t>(x, type_convert<float>(scale));
#endif
}

// convert fp32 to bf8
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ bf8_ocp_t scaled_type_convert<bf8_ocp_t, float>(e8m0_bexp_t scale,
                                                                           float x)
#else
inline __host__ bf8_ocp_t scaled_type_convert<bf8_ocp_t, float>(e8m0_bexp_t scale, float x)
#endif
{
#if CK_USE_SR_F8_CONVERSION
    return mxf8_convert_sr<bf8_ocp_t>(x, type_convert<float>(scale));
#else
    return mxf8_convert_rne<bf8_ocp_t>(x, type_convert<float>(scale));
#endif
}

// convert fp32x2 to fp8x2
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ f8x2_ocp_t scaled_type_convert<f8x2_ocp_t, float2_t>(e8m0_bexp_t scale,
                                                                                float2_t x)
#else
inline __host__ f8x2_ocp_t scaled_type_convert<f8x2_ocp_t, float2_t>(e8m0_bexp_t scale, float2_t x)
#endif
{
#if CK_USE_SR_F8_CONVERSION
    return mxf8_convert_sr<f8x2_ocp_t>(x, type_convert<float>(scale));
#else
    return mxf8_convert_rne<f8x2_ocp_t>(x, type_convert<float>(scale));
#endif
}
// convert fp32x2 to bf8x2
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ bf8x2_ocp_t scaled_type_convert<bf8x2_ocp_t, float2_t>(e8m0_bexp_t scale,
                                                                                  float2_t x)
#else
inline __host__ bf8x2_ocp_t scaled_type_convert<bf8x2_ocp_t, float2_t>(e8m0_bexp_t scale,
                                                                       float2_t x)
#endif
{
#if CK_USE_SR_F8_CONVERSION
    return mxf8_convert_sr<bf8x2_ocp_t>(x, type_convert<float>(scale));
#else
    return mxf8_convert_rne<bf8x2_ocp_t>(x, type_convert<float>(scale));
#endif
}

// convert fp32x16 to fp8x16
// @note Host version gives compilation error. Requires extra compiler options.
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ f8x16_ocp_t
scaled_type_convert<f8x16_ocp_t, float16_t>(e8m0_bexp_t scale, float16_t x)
#else
inline __host__ f8x16_ocp_t scaled_type_convert<f8x16_ocp_t, float16_t>(e8m0_bexp_t scale,
                                                                        float16_t x)
#endif
{
#if CK_USE_SR_F8_CONVERSION
    return mxf8_convert_sr<f8x16_ocp_t>(x, type_convert<float>(scale));
#else
    return mxf8_convert_rne<f8x16_ocp_t>(x, type_convert<float>(scale));
#endif
}

// convert fp32x16 to bf8x16
// @note Host version gives compilation error. Requires extra compiler options.
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ bf8x16_ocp_t
scaled_type_convert<bf8x16_ocp_t, float16_t>(e8m0_bexp_t scale, float16_t x)
#else
inline __host__ bf8x16_ocp_t scaled_type_convert<bf8x16_ocp_t, float16_t>(e8m0_bexp_t scale,
                                                                          float16_t x)
#endif
{
#if CK_USE_SR_F8_CONVERSION
    return mxf8_convert_sr<bf8x16_ocp_t>(x, type_convert<float>(scale));
#else
    return mxf8_convert_rne<bf8x16_ocp_t>(x, type_convert<float>(scale));
#endif
}

// convert fp32x32 to fp8x32
// @note Host version gives compilation error. Requires extra compiler options.
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ f8x32_ocp_t
scaled_type_convert<f8x32_ocp_t, float32_t>(e8m0_bexp_t scale, float32_t x)
#else
inline __host__ f8x32_ocp_t scaled_type_convert<f8x32_ocp_t, float32_t>(e8m0_bexp_t scale,
                                                                        float32_t x)
#endif
{
#if CK_USE_SR_F8_CONVERSION
    return mxf8_convert_sr<f8x32_ocp_t>(x, type_convert<float>(scale));
#else
    return mxf8_convert_rne<f8x32_ocp_t>(x, type_convert<float>(scale));
#endif
}

// convert fp32x32 to bf8x32
// @note Host version gives compilation error. Requires extra compiler options.
template <>
#if CK_USE_OCP_FP8
inline __host__ __device__ bf8x32_ocp_t
scaled_type_convert<bf8x32_ocp_t, float32_t>(e8m0_bexp_t scale, float32_t x)
#else
inline __host__ bf8x32_ocp_t scaled_type_convert<bf8x32_ocp_t, float32_t>(e8m0_bexp_t scale,
                                                                          float32_t x)
#endif
{
#if CK_USE_SR_F8_CONVERSION
    return mxf8_convert_sr<bf8x32_ocp_t>(x, type_convert<float>(scale));
#else
    return mxf8_convert_rne<bf8x32_ocp_t>(x, type_convert<float>(scale));
#endif
}

// activate for architectures with native MX support
#if CK_USE_NATIVE_MX_SUPPORT
// convert fp4 to fp32
template <>
inline __host__ __device__ float scaled_type_convert<float, f4_t>(e8m0_bexp_t scale, f4_t x)
{
#if defined(__gfx950__)
    union
    {
        float float_array[2];
        float2_t float2_array;
    } float_values{};
    float_values.float2_array =
        __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(x, type_convert<float>(scale), 0);
    return float_values.float_array[0];
#else
    return utils::to_float<f4_t>(scale, x);
#endif
}

// convert vector of 2 fp4 to vector of 2 fp32
template <>
inline __host__ __device__ float2_t scaled_type_convert<float2_t, f4x2_t>(e8m0_bexp_t scale,
                                                                          f4x2_t x)
{
#if defined(__gfx950__)
    union
    {
        uint32_t bitwise;
        f4x2_t f4x2_array[4];
    } value{};
    value.f4x2_array[0] = x;
    return __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(value.bitwise, type_convert<float>(scale), 0);
#else
    float2_t ret{utils::to_float<f4_t>(
                     scale, x.template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<0>{})),
                 utils::to_float<f4_t>(
                     scale, x.template AsType<f4x2_pk_t>()[Number<0>{}].unpack<>(Number<1>{}))};
    return ret;
#endif
}

// convert vector of 32 fp4 to vector of 32 fp32
template <>
inline __host__ __device__ float32_t scaled_type_convert<float32_t, f4x32_t>(e8m0_bexp_t scale,
                                                                             f4x32_t x)
{
#if defined(__gfx950__)
    union
    {
        f4x32_t f4x32_array;
        f4x2_t fp4x2[16];
    } value{x};
    float2_t op;
    float32_t ret;
    float f_scale = type_convert<float>(scale);

    ck::static_for<0, 32 / 2, 1>{}([&](auto idx) {
        op               = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(value.fp4x2[idx], f_scale, 0);
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
            scale,
            f4_values.f4x2_array[idx].template AsType<f4x2_pk_t>()[Number<0>{}].template unpack<>(
                Number<0>{}));

        float_values.float_array[2 * idx + 1] = utils::to_float<f4_t>(
            scale,
            f4_values.f4x2_array[idx].template AsType<f4x2_pk_t>()[Number<0>{}].template unpack<>(
                Number<1>{}));
    });

    return float_values.float32_array;
#endif
}

// convert fp32 to fp4
template <>
inline __host__ __device__ f4_t scaled_type_convert<f4_t, float>(e8m0_bexp_t scale, float x)
{
#if CK_USE_SR_F4_CONVERSION
    return f4_convert_sr(x, type_convert<float>(scale));
#else
    return f4_convert_rne(x, type_convert<float>(scale));
#endif
}

// convert vector of 2 fp32 to vector of 2 fp4
template <>
inline __host__ __device__ f4x2_t scaled_type_convert<f4x2_t, float2_t>(e8m0_bexp_t scale,
                                                                        float2_t x)
{
#if CK_USE_SR_F4_CONVERSION
    return f4_convert_sr(x, type_convert<float>(scale));
#else
    return f4_convert_rne(x, type_convert<float>(scale));
#endif
}

// convert vector of 32 fp32 to vector of 32 fp4
template <>
inline __host__ __device__ f4x32_t scaled_type_convert<f4x32_t, float32_t>(e8m0_bexp_t scale,
                                                                           float32_t x)
{
#if CK_USE_SR_F4_CONVERSION
    return f4_convert_sr(x, type_convert<float>(scale));
#else
    return f4_convert_rne(x, type_convert<float>(scale));
#endif
}

/**
 * @brief Converts a 6-bit floating-point value (f6_t) to a 32-bit float,
 *        applying the specified scaling factor.
 *
 * @param scale The exponent scale factor (e8m0_bexp_t) used for f6_t.
 * @param x     The f6_t value to be converted.
 * @return      The converted 32-bit float representation of the input.
 */
template <>
inline __host__ __device__ float scaled_type_convert<float, f6_t>(e8m0_bexp_t scale, f6_t x)
{
#if defined(__gfx950__)
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

    out.float_vector = __builtin_amdgcn_cvt_scalef32_pk32_f32_fp6(
        in.f6_vector.template AsType<f6x32_t::data_t>()[Number<0>{}], type_convert<float>(scale));
    return out.float_array[0];
#else
    return utils::to_float<f6_t>(scale, x);
#endif
}

/**
 * @brief Converts a vector of 32 6-bit floating-point values (f6x32_t) to a vector of 32 floats,
 *        applying the specified scaling factor.
 *
 * @param scale The exponent scale factor (e8m0_bexp_t).
 * @param x     The f6x32_t vector to be converted.
 * @return      The converted float vector representation of the input.
 */
template <>
inline __host__ __device__ float32_t scaled_type_convert<float32_t, f6x32_t>(e8m0_bexp_t scale,
                                                                             f6x32_t x)
{
#if defined(__gfx950__)
    return __builtin_amdgcn_cvt_scalef32_pk32_f32_fp6(
        x.template AsType<f6x32_t::data_t>()[Number<0>{}], type_convert<float>(scale));
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

    ck::static_for<0, 32, 1>{}(
        [&](auto i) { out.float_array[i] = utils::to_float<f6_t>(scale, in.f6_array[i]); });

    return out.float_vector;
#endif
}

/**
 * @brief Converts a 6-bit floating-point value (bf6_t) to a 32-bit float,
 *        applying the specified scaling factor.
 *
 * @param scale The exponent scale factor (e8m0_bexp_t) used for bf6_t.
 * @param x     The bf6_t value to be converted.
 * @return      The converted 32-bit float representation of the input.
 */
template <>
inline __host__ __device__ float scaled_type_convert<float, bf6_t>(e8m0_bexp_t scale, bf6_t x)
{
#if defined(__gfx950__)
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

    out.float_vector = __builtin_amdgcn_cvt_scalef32_pk32_f32_bf6(
        in.bf6_vector.template AsType<bf6x32_t::data_t>()[Number<0>{}], type_convert<float>(scale));
    return out.float_array[0];
#else
    return utils::to_float<bf6_t>(scale, x);
#endif
}

/**
 * @brief Converts a vector of 6-bit floating-point values (bf6x32_t) to a vector of 32 floats,
 *        applying the specified scaling factor.
 *
 * @param scale The exponent scale factor (e8m0_bexp_t).
 * @param x     The bf6x32_t vector to be converted.
 * @return      The converted vector of 32 float representation of the input.
 */
template <>
inline __host__ __device__ float32_t scaled_type_convert<float32_t, bf6x32_t>(e8m0_bexp_t scale,
                                                                              bf6x32_t x)
{
#if defined(__gfx950__)
    return __builtin_amdgcn_cvt_scalef32_pk32_f32_bf6(
        x.template AsType<bf6x32_t::data_t>()[Number<0>{}], type_convert<float>(scale));
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

    ck::static_for<0, 32, 1>{}(
        [&](auto i) { out.float_array[i] = utils::to_float<bf6_t>(scale, in.bf6_array[i]); });

    return out.float_vector;
#endif
}

/**
 * @brief Converts a 32-bit float to a 6-bit floating-point value (f6_t), applying the specified
 * scale.
 *
 * Depending on whether CK_USE_SR_F6_CONVERSION is defined, it uses either stochastic rounding
 * (f6_convert_sr) or round-to-nearest-even (f6_convert_rne).
 *
 * @param scale The exponent scale factor (e8m0_bexp_t) used for f6_t.
 * @param x     The float value to convert.
 * @return      The converted 6-bit floating-point value (f6_t).
 */
template <>
inline __host__ __device__ f6_t scaled_type_convert<f6_t, float>(e8m0_bexp_t scale, float x)
{
#if CK_USE_SR_F6_CONVERSION
    return f6_convert_sr(x, type_convert<float>(scale));
#else
    return f6_convert_rne(x, type_convert<float>(scale));
#endif
}

/**
 * @brief Converts a vector of 32 floats to a vector of 32 6-bit floating-point values (f6x32_t),
 * applying the specified scale.
 *
 * Depending on whether CK_USE_SR_F6_CONVERSION is defined, it uses either stochastic rounding
 * (f6_convert_sr) or round-to-nearest-even (f6_convert_rne).
 *
 * @param scale The exponent scale factor (e8m0_bexp_t).
 * @param x     The float vector to convert.
 * @return      The converted vector of 6-bit floating-point values (f6x32_t).
 */
template <>
inline __host__ __device__ f6x32_t scaled_type_convert<f6x32_t, float32_t>(e8m0_bexp_t scale,
                                                                           float32_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return f6_convert_sr(x, type_convert<float>(scale));
#else
    return f6_convert_rne(x, type_convert<float>(scale));
#endif
}

/**
 * @brief Converts a 32-bit float to a 6-bit floating-point value (bf6_t), applying the specified
 * scale.
 *
 * Depending on whether CK_USE_SR_F6_CONVERSION is defined, it uses either stochastic rounding
 * (bf6_convert_sr) or round-to-nearest-even (bf6_convert_rne).
 *
 * @param scale The exponent scale factor (e8m0_bexp_t) used for bf6_t.
 * @param x     The float value to convert.
 * @return      The converted 6-bit floating-point value (bf6_t).
 */
template <>
inline __host__ __device__ bf6_t scaled_type_convert<bf6_t, float>(e8m0_bexp_t scale, float x)
{
#if CK_USE_SR_F6_CONVERSION
    return bf6_convert_sr(x, type_convert<float>(scale));
#else
    return bf6_convert_rne(x, type_convert<float>(scale));
#endif
}

/**
 * @brief Converts a vector of 32 floats to a vector of 32 6-bit floating-point values (bf6x32_t),
 * applying the specified scale.
 *
 * Depending on whether CK_USE_SR_F6_CONVERSION is defined, it uses either stochastic rounding
 * (bf6_convert_sr) or round-to-nearest-even (bf6_convert_rne).
 *
 * @param scale The exponent scale factor (e8m0_bexp_t).
 * @param x     The float vector to convert.
 * @return      The converted 6-bit floating-point vector (bf6x32_t).
 */
template <>
inline __host__ __device__ bf6x32_t scaled_type_convert<bf6x32_t, float32_t>(e8m0_bexp_t scale,
                                                                             float32_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return bf6_convert_sr(x, type_convert<float>(scale));
#else
    return bf6_convert_rne(x, type_convert<float>(scale));
#endif
}
#endif // #if CK_USE_NATIVE_MX_SUPPORT

} // namespace ck
