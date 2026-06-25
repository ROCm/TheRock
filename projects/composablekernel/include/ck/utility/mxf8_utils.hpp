// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/utility/numeric_limits.hpp"
#include "ck/utility/mxfp_utils.hpp"

#if defined(__gfx950__) && __HIP_DEVICE_COMPILE__
#define CK_MX_FP8_CVT_FAST_PATH 1
#else
#define CK_MX_FP8_CVT_FAST_PATH 0
#endif

namespace ck {

namespace fp8_impl {
#if CK_MX_FP8_CVT_FAST_PATH
template <ck_fp8_interpretation_t interpret>
static __device__ float cast_to_f32_from_f8_scaled(float scale, fp8_storage_t v)
{
    union
    {
        unsigned int i32val;
        unsigned char i8val[4];
    } val;
    val.i8val[0] = v;

    static_assert(interpret == ck_fp8_interpretation_t::CK_E4M3_OCP ||
                      interpret == ck_fp8_interpretation_t::CK_E5M2_OCP,
                  "Only OCP interpretations are supported");

    if constexpr(interpret == ck_fp8_interpretation_t::CK_E4M3_OCP)
    {
        return __builtin_amdgcn_cvt_scalef32_f32_fp8(val.i32val, scale, 0);
    }
    else
    {
        return __builtin_amdgcn_cvt_scalef32_f32_bf8(val.i32val, scale, 0);
    }
}

template <ck_fp8_interpretation_t interpret>
static __device__ float2_t cast_to_f32_from_f8_scaled(float scale, fp8x2_storage_t v)
{
    const auto i16val = bit_cast<uint16_t>(v);

    static_assert(interpret == ck_fp8_interpretation_t::CK_E4M3_OCP ||
                      interpret == ck_fp8_interpretation_t::CK_E5M2_OCP,
                  "Only OCP interpretations are supported");

    if constexpr(interpret == ck_fp8_interpretation_t::CK_E4M3_OCP)
    {
        return __builtin_amdgcn_cvt_scalef32_pk_f32_fp8(i16val, scale, 0);
    }
    else
    {
        return __builtin_amdgcn_cvt_scalef32_pk_f32_bf8(i16val, scale, 0);
    }
}

template <ck_fp8_interpretation_t interpret, bool stochastic_rounding = false>
static __device__ fp8_storage_t cast_to_f8_from_f32_scaled(float v,
                                                           unsigned int rng = 0,
                                                           float scale      = 1.0f)
{
    fp8_storage_t i8data;
    union
    {
        float fval;
        unsigned int i32val;
    } val;

    union
    {
        uint32_t ival;
        vector_type<int16_t, 2>::type v2i16;
        fp8_storage_t v4i8[4];
    } ret{};

    // unsigned int ival = 0;
    val.fval = v;

    if constexpr(stochastic_rounding)
    {
        ret.ival =
            (interpret == ck_fp8_interpretation_t::CK_E4M3_OCP)
                ? __builtin_amdgcn_cvt_scalef32_sr_fp8_f32(ret.ival, val.fval, rng, scale, 0)
                : __builtin_amdgcn_cvt_scalef32_sr_bf8_f32(ret.ival, val.fval, rng, scale, 0);

        i8data = ret.v4i8[0];
    }
    else
    {
        // RNE CVT
        // llvm.amdgcn.cvt.scalef32.pk.fp8.f32
        // v2i16 old_vdst, float srcA, float srcB, float scale, bool dst_lo_hi_sel
        if constexpr(interpret == ck_fp8_interpretation_t::CK_E4M3_OCP)
        {
            // If fval / scale > max fp8, returns Nan
            ret.v2i16 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(/*old_vdst*/ ret.v2i16,
                                                                 val.fval,
                                                                 val.fval,
                                                                 scale,
                                                                 /*dst_lo_hi_sel*/ false);
        }
        else
        {
            // If fval / scale > max bf8, returns Inf
            ret.v2i16 = __builtin_amdgcn_cvt_scalef32_pk_bf8_f32(/*old_vdst*/ ret.v2i16,
                                                                 val.fval,
                                                                 val.fval,
                                                                 scale,
                                                                 /*dst_lo_hi_sel*/ false);
        }

        i8data = ret.v4i8[0];
    }
    return i8data;
}

template <ck_fp8_interpretation_t interpret, bool stochastic_rounding = false>
static __device__ fp8x2_storage_t cast_to_f8_from_f32_scaled(float2_t v,
                                                             unsigned int rng = 0,
                                                             float scale      = 1.0f)
{

    union
    {
        uint32_t ival;
        vector_type<int16_t, 2>::type v2i16;
        StaticallyIndexedArray<fp8x2_storage_t, 2> v2f8x2;
    } ret{};

    if constexpr(stochastic_rounding)
    {
        fp8x2_storage_t f8x2;
        if constexpr(interpret == ck_fp8_interpretation_t::CK_E4M3_OCP)
        {
            ret.ival = __builtin_amdgcn_cvt_scalef32_sr_fp8_f32(ret.ival, v[0], rng, scale, 0);
            f8x2[0]  = ret.v2f8x2(Number<0>{})[0];
            ret.ival = __builtin_amdgcn_cvt_scalef32_sr_fp8_f32(ret.ival, v[1], rng, scale, 0);
            f8x2[1]  = ret.v2f8x2(Number<0>{})[0];
        }
        else
        {
            ret.ival = __builtin_amdgcn_cvt_scalef32_sr_bf8_f32(ret.ival, v[0], rng, scale, 0);
            f8x2[0]  = ret.v2f8x2(Number<0>{})[0];
            ret.ival = __builtin_amdgcn_cvt_scalef32_sr_bf8_f32(ret.ival, v[1], rng, scale, 0);
            f8x2[1]  = ret.v2f8x2(Number<0>{})[0];
        }
        return f8x2;
    }
    else
    {
        // RNE CVT
        // llvm.amdgcn.cvt.scalef32.pk.fp8.f32
        // v2i16 old_vdst, float srcA, float srcB, float scale, bool dst_lo_hi_sel
        if constexpr(interpret == ck_fp8_interpretation_t::CK_E4M3_OCP)
        {
            // If fval / scale > max fp8, returns Nan
            ret.v2i16 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(/*old_vdst*/ ret.v2i16,
                                                                 v[0],
                                                                 v[1],
                                                                 scale,
                                                                 /*dst_lo_hi_sel*/ false);
        }
        else
        {
            // If fval / scale > max bf8, returns Inf
            ret.v2i16 = __builtin_amdgcn_cvt_scalef32_pk_bf8_f32(/*old_vdst*/ ret.v2i16,
                                                                 v[0],
                                                                 v[1],
                                                                 scale,
                                                                 /*dst_lo_hi_sel*/ false);
        }

        return ret.v2f8x2(Number<0>{});
    }
}

#endif // CK_MX_FP8_CVT_FAST_PATH

#if CK_MX_FP8_CVT_FAST_PATH
/**
 * \brief convert float to @p fp8_storage_t with scaling
 *
 * This version is used when the fast path (MX FP8 hardware) is available
 *
 * \tparam interp interpretation of fp8
 * \param f float number
 * \param scale scaling factor
 * \return fp8_storage_t
 */
template <ck_fp8_interpretation_t interp, bool stochastic_rounding = false>
__host__ __device__ static inline fp8_storage_t cvt_float_to_fp8_scaled(const float f, float scale)
{
    __is_interpret_supported(interp);
    uint32_t rng = 0;
    if constexpr(stochastic_rounding)
    {
        // use HW clock for stochastic input multiply by incremented thread id
        rng = __builtin_amdgcn_prng_b32(__builtin_amdgcn_s_memrealtime() *
                                        (get_thread_global_1d_id() + 1));
    }
    return cast_to_f8_from_f32_scaled<interp, stochastic_rounding>(f, rng, scale);
}

/**
 * \brief convert 2xfloat to @p 2xfp8_storage_t with scaling
 *
 * This version is used when the fast path (MX FP8 hardware) is available
 *
 * \tparam interp interpretation of fp8
 * \param f 2xfloat
 * \param scale scaling factor
 * \return 2xfp8_storage_t
 */
template <ck_fp8_interpretation_t interp, bool stochastic_rounding = false>
__host__ __device__ static inline fp8x2_storage_t cvt_float_to_fp8_scaled(const float2_t f,
                                                                          float scale)
{
    __is_interpret_supported(interp);
    uint32_t rng = 0;
    if constexpr(stochastic_rounding)
    {
        // use HW clock for stochastic input multiply by incremented thread id
        rng = __builtin_amdgcn_prng_b32(__builtin_amdgcn_s_memrealtime() *
                                        (get_thread_global_1d_id() + 1));
    }
    return cast_to_f8_from_f32_scaled<interp, stochastic_rounding>(f, rng, scale);
}

#else

/**
 * \brief convert float to @p fp8_storage_t with scaling
 *
 * This version is used when the fast path (MX FP8 hardware) is not available
 *
 * \tparam interp interpretation of fp8
 * \param f float number
 * \param scale scaling factor
 * \return fp8_storage_t
 */
template <ck_fp8_interpretation_t interp, bool stochastic_rounding = false>
__host__ __device__ static inline fp8_storage_t cvt_float_to_fp8_scaled(const float f, float scale)
{

    static_assert(interp == ck_fp8_interpretation_t::CK_E4M3_OCP ||
                      interp == ck_fp8_interpretation_t::CK_E5M2_OCP,
                  "Only OCP interpretations are supported");

    uint32_t rng = 0;
    if constexpr(stochastic_rounding)
    {
        constexpr int seed = 1254739;
        rng                = prand_generator<float, seed>(reinterpret_cast<uintptr_t>(&f), f);
    }

    if constexpr(interp == ck_fp8_interpretation_t::CK_E4M3_OCP)
    {
        return cast_to_f8<float, 3, 4, false, true, stochastic_rounding>(f / scale, rng);
    }
    else if constexpr(interp == ck_fp8_interpretation_t::CK_E5M2_OCP)
    {
        return cast_to_f8<float, 2, 5, false, true, stochastic_rounding>(f / scale, rng);
    }
    else
    {
        __hip_assert(false && "FP8 type is not supported by current target device");
        return 0;
    }
}

/**
 * \brief convert two float to @p 2xfp8_storage_t with scaling
 *
 * This version is used when the fast path (MX FP8 hardware) is not available
 *
 * \tparam interp interpretation of fp8
 * \param f 2xfloat
 * \param scale scaling factor
 * \return 2xfp8_storage_t
 */
template <ck_fp8_interpretation_t interp, bool stochastic_rounding = false>
__host__ __device__ static inline fp8x2_storage_t cvt_float_to_fp8_scaled(const float2_t f,
                                                                          float scale)
{

    static_assert(interp == ck_fp8_interpretation_t::CK_E4M3_OCP ||
                      interp == ck_fp8_interpretation_t::CK_E5M2_OCP,
                  "Only OCP interpretations are supported");

    uint32_t rng = 0;
    if constexpr(stochastic_rounding)
    {
        constexpr int seed = 1254739;
        rng                = prand_generator<float, seed>(reinterpret_cast<uintptr_t>(&f), f[0]);
    }

    if constexpr(interp == ck_fp8_interpretation_t::CK_E4M3_OCP)
    {
        return {cast_to_f8<float, 3, 4, false, true, stochastic_rounding>(f[0] / scale, rng),
                cast_to_f8<float, 3, 4, false, true, stochastic_rounding>(f[1] / scale, rng)};
    }
    else if constexpr(interp == ck_fp8_interpretation_t::CK_E5M2_OCP)
    {
        return {cast_to_f8<float, 2, 5, false, true, stochastic_rounding>(f[0] / scale, rng),
                cast_to_f8<float, 2, 5, false, true, stochastic_rounding>(f[1] / scale, rng)};
    }
    else
    {
        __hip_assert(false && "FP8 type is not supported by current target device");
        return 0;
    }
}

#endif // CK_MX_FP8_CVT_FAST_PATH

} // namespace fp8_impl

// Declare a template function for fp8 conversion using SR
template <typename Y, typename X>
__host__ __device__ constexpr Y mxf8_convert_sr(X x, float scale);

// Declare a template function for fp8 conversion using RNE
template <typename Y, typename X>
__host__ __device__ constexpr Y mxf8_convert_rne(X x, float scale);

// convert fp32 to fp8 with rounding to nearest even
template <>
inline __host__ __device__ f8_ocp_t mxf8_convert_rne<f8_ocp_t, float>(float x, float scale)
{
    return f8_ocp_t{fp8_impl::cvt_float_to_fp8_scaled<f8_ocp_t::default_interpret>(x, scale)};
}

// convert fp32 to bf8 with rounding to nearest even
template <>
inline __host__ __device__ bf8_ocp_t mxf8_convert_rne<bf8_ocp_t, float>(float x, float scale)
{
    return bf8_ocp_t{fp8_impl::cvt_float_to_fp8_scaled<bf8_ocp_t::default_interpret>(x, scale)};
}

// convert fp32x2 to fp8x2 with rounding to nearest even
template <>
inline __host__ __device__ f8x2_ocp_t mxf8_convert_rne<f8x2_ocp_t, float2_t>(float2_t x,
                                                                             float scale)
{
    return f8x2_ocp_t{fp8_impl::cvt_float_to_fp8_scaled<f8_ocp_t::default_interpret>(x, scale)};
}

// convert fp32x2 to bf8x2 with rounding to nearest even
template <>
inline __host__ __device__ bf8x2_ocp_t mxf8_convert_rne<bf8x2_ocp_t, float2_t>(float2_t x,
                                                                               float scale)
{
    return bf8x2_ocp_t{fp8_impl::cvt_float_to_fp8_scaled<bf8_ocp_t::default_interpret>(x, scale)};
}

// convert fp32x16 to fp8x16 with rounding to nearest even
template <>
inline __host__ __device__ f8x16_ocp_t mxf8_convert_rne<f8x16_ocp_t, float16_t>(float16_t x,
                                                                                float scale)
{
    union
    {
        float16_t float_1x16;
        float2_t float_2x8[8];
    } in{x};

    union
    {
        f8x16_ocp_t fp8_1x16;
        f8x2_ocp_t fp8_2x8[8];
    } out{};

    ck::static_for<0, 8, 1>{}(
        [&](auto i) { out.fp8_2x8[i] = mxf8_convert_rne<f8x2_ocp_t>(in.float_2x8[i], scale); });

    return out.fp8_1x16;
}

// convert fp32x16 to bf8x16 with rounding to nearest even
template <>
inline __host__ __device__ bf8x16_ocp_t mxf8_convert_rne<bf8x16_ocp_t, float16_t>(float16_t x,
                                                                                  float scale)
{
    union
    {
        float16_t float_1x16;
        float2_t float_2x8[8];
    } in{x};

    union
    {
        bf8x16_ocp_t bf8_1x16;
        bf8x2_ocp_t bf8_2x8[8];
    } out{};

    ck::static_for<0, 8, 1>{}(
        [&](auto i) { out.bf8_2x8[i] = mxf8_convert_rne<bf8x2_ocp_t>(in.float_2x8[i], scale); });

    return out.bf8_1x16;
}

// convert fp32x32 to fp8x32 with rounding to nearest even
template <>
inline __host__ __device__ f8x32_ocp_t mxf8_convert_rne<f8x32_ocp_t, float32_t>(float32_t x,
                                                                                float scale)
{
    union
    {
        float32_t float_1x32;
        float16_t float_16x2[2];
    } in{x};

    union
    {
        f8x32_ocp_t fp8_1x32;
        f8x16_ocp_t fp8_16x2[2];
    } out{};

    ck::static_for<0, 2, 1>{}(
        [&](auto i) { out.fp8_16x2[i] = mxf8_convert_rne<f8x16_ocp_t>(in.float_16x2[i], scale); });

    return out.fp8_1x32;
}

// convert fp32x32 to bf8x32 with rounding to nearest even
template <>
inline __host__ __device__ bf8x32_ocp_t mxf8_convert_rne<bf8x32_ocp_t, float32_t>(float32_t x,
                                                                                  float scale)
{
    union
    {
        float32_t float_1x32;
        float16_t float_16x2[2];
    } in{x};

    union
    {
        bf8x32_ocp_t bf8_1x32;
        bf8x16_ocp_t bf8_16x2[2];
    } out{};

    ck::static_for<0, 2, 1>{}(
        [&](auto i) { out.bf8_16x2[i] = mxf8_convert_rne<bf8x16_ocp_t>(in.float_16x2[i], scale); });

    return out.bf8_1x32;
}

// convert fp32 to fp8 with stochastic rounding
template <>
inline __host__ __device__ f8_ocp_t mxf8_convert_sr<f8_ocp_t, float>(float x, float scale)
{
    return f8_ocp_t{fp8_impl::cvt_float_to_fp8_scaled<f8_ocp_t::default_interpret, true>(x, scale)};
}

// convert fp32 to bf8 with stochastic rounding
template <>
inline __host__ __device__ bf8_ocp_t mxf8_convert_sr<bf8_ocp_t, float>(float x, float scale)
{
    return bf8_ocp_t{
        fp8_impl::cvt_float_to_fp8_scaled<bf8_ocp_t::default_interpret, true>(x, scale)};
}

// convert fp32x2 to fp8x2 with stochastic rounding
template <>
inline __host__ __device__ f8x2_ocp_t mxf8_convert_sr<f8x2_ocp_t, float2_t>(float2_t x, float scale)
{
    return f8x2_ocp_t{
        fp8_impl::cvt_float_to_fp8_scaled<f8_ocp_t::default_interpret, true>(x, scale)};
}

// convert fp32x2 to bf8x2 with stochastic rounding
template <>
inline __host__ __device__ bf8x2_ocp_t mxf8_convert_sr<bf8x2_ocp_t, float2_t>(float2_t x,
                                                                              float scale)
{
    return bf8x2_ocp_t{
        fp8_impl::cvt_float_to_fp8_scaled<bf8_ocp_t::default_interpret, true>(x, scale)};
}

// convert fp32x16 to fp8x16 with stochastic rounding
template <>
inline __host__ __device__ f8x16_ocp_t mxf8_convert_sr<f8x16_ocp_t, float16_t>(float16_t x,
                                                                               float scale)
{
    union
    {
        float16_t float_1x16;
        float2_t float_2x8[8];
    } in{x};

    union
    {
        f8x16_ocp_t fp8_1x16;
        f8x2_ocp_t fp8_2x8[8];
    } out{};

    ck::static_for<0, 8, 1>{}(
        [&](auto i) { out.fp8_2x8[i] = mxf8_convert_sr<f8x2_ocp_t>(in.float_2x8[i], scale); });

    return out.fp8_1x16;
}

// convert fp32x16 to bf8x16 with stochastic rounding
template <>
inline __host__ __device__ bf8x16_ocp_t mxf8_convert_sr<bf8x16_ocp_t, float16_t>(float16_t x,
                                                                                 float scale)
{
    union
    {
        float16_t float_1x16;
        float2_t float_2x8[8];
    } in{x};

    union
    {
        bf8x16_ocp_t bf8_1x16;
        bf8x2_ocp_t bf8_2x8[8];
    } out{};

    ck::static_for<0, 8, 1>{}(
        [&](auto i) { out.bf8_2x8[i] = mxf8_convert_sr<bf8x2_ocp_t>(in.float_2x8[i], scale); });

    return out.bf8_1x16;
}

// convert fp32x32 to fp8x32 with stochastic rounding
template <>
inline __host__ __device__ f8x32_ocp_t mxf8_convert_sr<f8x32_ocp_t, float32_t>(float32_t x,
                                                                               float scale)
{
    union
    {
        float32_t float_1x32;
        float16_t float_16x2[2];
    } in{x};

    union
    {
        f8x32_ocp_t fp8_1x32;
        f8x16_ocp_t fp8_16x2[2];
    } out{};

    ck::static_for<0, 2, 1>{}(
        [&](auto i) { out.fp8_16x2[i] = mxf8_convert_sr<f8x16_ocp_t>(in.float_16x2[i], scale); });

    return out.fp8_1x32;
}

// convert fp32x32 to bf8x32 with stochastic rounding
template <>
inline __host__ __device__ bf8x32_ocp_t mxf8_convert_sr<bf8x32_ocp_t, float32_t>(float32_t x,
                                                                                 float scale)
{
    union
    {
        float32_t float_1x32;
        float16_t float_16x2[2];
    } in{x};

    union
    {
        bf8x32_ocp_t bf8_1x32;
        bf8x16_ocp_t bf8_16x2[2];
    } out{};

    ck::static_for<0, 2, 1>{}(
        [&](auto i) { out.bf8_16x2[i] = mxf8_convert_sr<bf8x16_ocp_t>(in.float_16x2[i], scale); });

    return out.bf8_1x32;
}

} // namespace ck
