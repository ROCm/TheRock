// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifndef __HIP_DEVICE_COMPILE__
#include <cmath>
#endif

#include "ck/utility/data_type.hpp"
#include "ck/utility/type.hpp"
#include "ck/utility/type_convert.hpp"

namespace ck {
namespace math {

#if CK_WORKAROUND_SWDEV_383542
extern "C" __device__ float __ocml_native_recip_f32(float);
#endif

// math functions for the host,  some are implemented by calling C++ std functions
#if !defined(__HIPCC_RTC__) || !defined(CK_CODE_GEN_RTC)
static inline __host__ float abs(float x) { return std::abs(x); };

static inline __host__ double abs(double x) { return std::abs(x); };

static inline __host__ int8_t abs(int8_t x)
{
    int8_t sgn = x >> (8 - 1);

    return (x ^ sgn) - sgn;
};

static inline __host__ int32_t abs(int32_t x)
{
    int32_t sgn = x >> (32 - 1);

    return (x ^ sgn) - sgn;
};

static inline __host__ half_t abs(half_t x)
{
    uint16_t xx = ck::bit_cast<uint16_t>(x);

    uint16_t abs_xx = xx & 0x7fff;

    half_t abs_x = ck::bit_cast<half_t>(abs_xx);

    return abs_x;
};

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
static inline __host__ int4_t abs(int4_t x)
{
    int4_t sgn = x >> (4 - 1);
    return (x ^ sgn) - sgn;
}
#endif

static inline __host__ bool isnan(float x) { return std::isnan(x); };

static inline __host__ bool isnan(double x) { return std::isnan(x); };

static inline __host__ bool isnan(int8_t x)
{
    (void)x;
    return false;
};

static inline __host__ bool isnan(int32_t x)
{
    (void)x;
    return false;
};

static inline __host__ bool isnan(half_t x)
{
    uint16_t xx = ck::bit_cast<uint16_t>(x);

    return (xx & 0x7FFF) > 0x7C00;
};

static inline __host__ bool isnan(f8_t x) { return ck::fp8_is_nan(x); };

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
static inline __host__ bool isnan(int4_t x)
{
    (void)x;
    return false;
};
#endif

static inline __host__ half_t sqrt(half_t x)
{
    return static_cast<half_t>(std::sqrt(static_cast<float>(x)));
};

static inline __host__ float sqrt(float x) { return std::sqrt(x); };

static inline __host__ double sqrt(double x) { return std::sqrt(x); };

template <typename T>
inline __host__ T tanh(T x)
{
    return ck::type_convert<T>(std::tanhf(ck::type_convert<float>(x)));
};

template <>
inline __host__ float tanh<float>(float x)
{
    return std::tanhf(x);
};

template <>
inline __host__ double tanh<double>(double x)
{
    return std::tanh(x);
};

template <typename T>
inline __host__ T acos(T x)
{
    return ck::type_convert<T>(std::acosf(ck::type_convert<float>(x)));
};

template <>
inline __host__ float acos<float>(float x)
{
    return std::acosf(x);
};

template <>
inline __host__ double acos<double>(double x)
{
    return std::acos(x);
};

template <typename T>
inline __host__ T neg(T x)
{
    return ck::type_convert<T>(-(ck::type_convert<float>(x)));
};

template <>
inline __host__ float neg<float>(float x)
{
    return -x;
};

template <>
inline __host__ double neg<double>(double x)
{
    return -x;
};

template <>
inline __host__ int32_t neg<int32_t>(int32_t x)
{
    return -x;
};

template <>
inline __host__ int8_t neg<int8_t>(int8_t x)
{
    return -x;
};

template <typename T>
inline __host__ T atan(T x)
{
    return ck::type_convert<T>(std::atanf(ck::type_convert<float>(x)));
};

template <>
inline __host__ float atan<float>(float x)
{
    return std::atanf(x);
};

template <>
inline __host__ double atan<double>(double x)
{
    return std::atan(x);
};

template <typename T>
inline __host__ T sin(T x)
{
    return ck::type_convert<T>(std::sinf(ck::type_convert<float>(x)));
};

template <>
inline __host__ float sin<float>(float x)
{
    return std::sinf(x);
};

template <>
inline __host__ double sin<double>(double x)
{
    return std::sin(x);
};

template <typename T>
inline __host__ T asin(T x)
{
    return ck::type_convert<T>(std::asinf(ck::type_convert<float>(x)));
};

template <>
inline __host__ float asin<float>(float x)
{
    return std::asinf(x);
};

template <>
inline __host__ double asin<double>(double x)
{
    return std::asin(x);
};

template <typename T>
inline __host__ T asinh(T x)
{
    return ck::type_convert<T>(std::asinhf(ck::type_convert<float>(x)));
};

template <>
inline __host__ float asinh<float>(float x)
{
    return std::asinhf(x);
};

template <>
inline __host__ double asinh<double>(double x)
{
    return std::asinh(x);
};

template <typename T>
inline __host__ T cos(T x)
{
    return ck::type_convert<T>(std::cosf(ck::type_convert<float>(x)));
};

template <>
inline __host__ float cos<float>(float x)
{
    return std::cosf(x);
};

template <>
inline __host__ double cos<double>(double x)
{
    return std::cos(x);
};

template <typename T>
inline __host__ T acosh(T x)
{
    return ck::type_convert<T>(std::acoshf(ck::type_convert<float>(x)));
};

template <>
inline __host__ float acosh<float>(float x)
{
    return std::acoshf(x);
};

template <>
inline __host__ double acosh<double>(double x)
{
    return std::acosh(x);
};

template <typename T>
inline __host__ T tan(T x)
{
    return ck::type_convert<T>(std::tanf(ck::type_convert<float>(x)));
};

template <>
inline __host__ float tan<float>(float x)
{
    return std::tanf(x);
};

template <>
inline __host__ double tan<double>(double x)
{
    return std::tan(x);
};

template <typename T>
inline __host__ T atanh(T x)
{
    return ck::type_convert<T>(std::atanhf(ck::type_convert<float>(x)));
};

template <>
inline __host__ float atanh<float>(float x)
{
    return std::atanhf(x);
};

template <>
inline __host__ double atanh<double>(double x)
{
    return std::atanh(x);
};

template <typename T>
inline __host__ T sinh(T x)
{
    return ck::type_convert<T>(std::sinhf(ck::type_convert<float>(x)));
};

template <>
inline __host__ float sinh<float>(float x)
{
    return std::sinhf(x);
};

template <>
inline __host__ double sinh<double>(double x)
{
    return std::sinh(x);
};

template <typename T>
inline __host__ T ceil(T x)
{
    return ck::type_convert<T>(std::ceilf(ck::type_convert<float>(x)));
};

template <>
inline __host__ float ceil<float>(float x)
{
    return std::ceilf(x);
};

template <>
inline __host__ double ceil<double>(double x)
{
    return std::ceil(x);
};

template <typename T>
inline __host__ T cosh(T x)
{
    return ck::type_convert<T>(std::coshf(ck::type_convert<float>(x)));
};

template <>
inline __host__ float cosh<float>(float x)
{
    return std::coshf(x);
};

template <>
inline __host__ double cosh<double>(double x)
{
    return std::cosh(x);
};

template <typename T>
inline __host__ T floor(T x)
{
    return ck::type_convert<T>(std::floorf(ck::type_convert<float>(x)));
};

template <>
inline __host__ float floor<float>(float x)
{
    return std::floorf(x);
};

template <>
inline __host__ double floor<double>(double x)
{
    return std::floor(x);
};

template <typename T>
inline __host__ T rcp(T x)
{
    return ck::type_convert<T>(1.f / ck::type_convert<float>(x));
};

template <typename T>
inline __host__ T exp(T x)
{
    return ck::type_convert<T>(std::expf(ck::type_convert<float>(x)));
}

template <>
inline __host__ float exp<float>(float x)
{
    return std::expf(x);
}

template <>
inline __host__ double exp<double>(double x)
{
    return std::exp(x);
}

template <typename T>
inline __host__ T log(T x)
{
    return ck::type_convert<T>(std::logf(ck::type_convert<float>(x)));
}

template <>
inline __host__ float log<float>(float x)
{
    return std::logf(x);
}

template <>
inline __host__ double log<double>(double x)
{
    return std::log(x);
}

template <typename T>
inline __host__ T pow(T x, T gamma)
{
    return ck::type_convert<T>(
        std::powf(ck::type_convert<float>(x), ck::type_convert<float>(gamma)));
}

template <>
inline __host__ float pow<float>(float x, float gamma)
{
    return std::powf(x, gamma);
}

template <>
inline __host__ double pow<double>(double x, double gamma)
{
    return std::pow(x, gamma);
}

template <typename T>
inline __host__ T expm1(T x)
{
    return ck::type_convert<T>(std::expm1f(ck::type_convert<float>(x)));
}

template <>
inline __host__ float expm1<float>(float x)
{
    return std::expm1f(x);
}

template <>
inline __host__ double expm1<double>(double x)
{
    return std::expm1(x);
}
#endif
// math functions for the HIP kernel,  some are implemented by calling hip builtin functions

static inline __device__ float abs(float x) { return ::abs(x); };

static inline __device__ double abs(double x) { return ::abs(x); };

static inline __device__ int8_t abs(int8_t x)
{
    int8_t sgn = x >> (8 - 1);

    return (x ^ sgn) - sgn;
};

static inline __device__ int32_t abs(int32_t x)
{
    int32_t sgn = x >> (32 - 1);

    return (x ^ sgn) - sgn;
};

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
static inline __device__ int4_t abs(int4_t x)
{
    int4_t sgn = x >> (4 - 1);

    return (x ^ sgn) - sgn;
};
#endif

static inline __device__ half_t abs(half_t x)
{
    uint16_t xx = ck::bit_cast<uint16_t>(x);

    uint16_t abs_xx = xx & 0x7fff;

    half_t abs_x = ck::bit_cast<half_t>(abs_xx);

    return abs_x;
};

static inline __device__ bool isnan(float x) { return ::isnan(x); };

static inline __device__ bool isnan(double x) { return ::isnan(x); };

static inline __device__ bool isnan(int8_t x)
{
    (void)x;
    return false;
};

static inline __device__ bool isnan(int32_t x)
{
    (void)x;
    return false;
};

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
static inline __device__ bool isnan(int4_t x)
{
    (void)x;
    return false;
};
#endif

static inline __device__ bool isnan(half_t x)
{
    uint16_t xx = ck::bit_cast<uint16_t>(x);

    return (xx & 0x7FFF) > 0x7C00;
};

static inline __device__ bool isnan(f8_t x) { return ck::fp8_is_nan(x); };

static inline __device__ half_t sqrt(half_t x)
{
    return static_cast<half_t>(__builtin_amdgcn_sqrtf(static_cast<float>(x)));
};

static inline __device__ float sqrt(float x) { return __builtin_amdgcn_sqrtf(x); };

static inline __device__ double sqrt(double x) { return __builtin_amdgcn_sqrt(x); };

template <typename T>
inline __device__ T tanh(T x)
{
    return ck::type_convert<T>(::tanhf(ck::type_convert<float>(x)));
};

template <>
inline __device__ float tanh<float>(float x)
{
    return ::tanhf(x);
};

template <>
inline __device__ double tanh<double>(double x)
{
    return ::tanh(x);
};

template <typename T>
inline __device__ T acos(T x)
{
    return ck::type_convert<T>(::acosf(ck::type_convert<float>(x)));
};

template <>
inline __device__ float acos<float>(float x)
{
    return ::acosf(x);
};

template <>
inline __device__ double acos<double>(double x)
{
    return ::acos(x);
};

template <typename T>
inline __device__ T neg(T x)
{
    return ck::type_convert<T>(-(ck::type_convert<float>(x)));
};

template <>
inline __device__ float neg<float>(float x)
{
    return -x;
};

template <>
inline __device__ double neg<double>(double x)
{
    return -x;
};

template <>
inline __device__ int32_t neg<int32_t>(int32_t x)
{
    return -x;
};

template <>
inline __device__ int8_t neg<int8_t>(int8_t x)
{
    return -x;
};

template <>
inline __device__ half_t neg<half_t>(half_t x)
{
    return __hneg(static_cast<__half>(x));
};

template <typename T>
inline __device__ T atan(T x)
{
    return ck::type_convert<T>(::atanf(ck::type_convert<float>(x)));
};

template <>
inline __device__ float atan<float>(float x)
{
    return ::atanf(x);
};

template <>
inline __device__ double atan<double>(double x)
{
    return ::atan(x);
};

template <typename T>
inline __device__ T sin(T x)
{
    return ck::type_convert<T>(::sinf(ck::type_convert<float>(x)));
};

template <>
inline __device__ float sin<float>(float x)
{
    return ::sinf(x);
};

template <>
inline __device__ double sin<double>(double x)
{
    return ::sin(x);
};

template <>
inline __device__ half_t sin<half_t>(half_t x)
{
    return hsin(static_cast<__half>(x));
};

template <typename T>
inline __device__ T asin(T x)
{
    return ck::type_convert<T>(::asinf(ck::type_convert<float>(x)));
};

template <>
inline __device__ float asin<float>(float x)
{
    return ::asinf(x);
};

template <>
inline __device__ double asin<double>(double x)
{
    return ::asin(x);
};

template <typename T>
inline __device__ T asinh(T x)
{
    return ck::type_convert<T>(::asinhf(ck::type_convert<float>(x)));
};

template <>
inline __device__ float asinh<float>(float x)
{
    return ::asinhf(x);
};

template <>
inline __device__ double asinh<double>(double x)
{
    return ::asinh(x);
};

template <typename T>
inline __device__ T acosh(T x)
{
    return ck::type_convert<T>(::acoshf(ck::type_convert<float>(x)));
};

template <>
inline __device__ float acosh<float>(float x)
{
    return ::acoshf(x);
};

template <>
inline __device__ double acosh<double>(double x)
{
    return ::acosh(x);
};

template <typename T>
inline __device__ T tan(T x)
{
    return ck::type_convert<T>(::tanf(ck::type_convert<float>(x)));
};

template <>
inline __device__ float tan<float>(float x)
{
    return ::tanf(x);
};

template <>
inline __device__ double tan<double>(double x)
{
    return ::tan(x);
};

template <typename T>
inline __device__ T atanh(T x)
{
    return ck::type_convert<T>(::atanhf(ck::type_convert<float>(x)));
};

template <>
inline __device__ float atanh<float>(float x)
{
    return ::atanhf(x);
};

template <>
inline __device__ double atanh<double>(double x)
{
    return ::atanh(x);
};

template <typename T>
inline __device__ T sinh(T x)
{
    return ck::type_convert<T>(::sinhf(ck::type_convert<float>(x)));
};

template <>
inline __device__ float sinh<float>(float x)
{
    return ::sinhf(x);
};

template <>
inline __device__ double sinh<double>(double x)
{
    return ::sinh(x);
};

template <typename T>
inline __device__ T ceil(T x)
{
    return ck::type_convert<T>(::ceilf(ck::type_convert<float>(x)));
};

template <>
inline __device__ float ceil<float>(float x)
{
    return ::ceilf(x);
};

template <>
inline __device__ double ceil<double>(double x)
{
    return ::ceil(x);
};

template <>
inline __device__ half_t ceil<half_t>(half_t x)
{
    return hceil(static_cast<__half>(x));
};

template <typename T>
inline __device__ T cosh(T x)
{
    return ck::type_convert<T>(::coshf(ck::type_convert<float>(x)));
};

template <>
inline __device__ float cosh<float>(float x)
{
    return ::coshf(x);
};

template <>
inline __device__ double cosh<double>(double x)
{
    return ::cosh(x);
};

template <typename T>
inline __device__ T floor(T x)
{
    return ck::type_convert<T>(::floorf(ck::type_convert<float>(x)));
};

template <>
inline __device__ float floor<float>(float x)
{
    return ::floorf(x);
};

template <>
inline __device__ double floor<double>(double x)
{
    return ::floor(x);
};

template <>
inline __device__ half_t floor<half_t>(half_t x)
{
    return hfloor(static_cast<__half>(x));
};

template <typename T>
inline __device__ T rcp(T x)
{
#if !CK_WORKAROUND_SWDEV_383542
    return __frcp_rn(x);
#else
    return __ocml_native_recip_f32(x);
#endif
};

template <typename T>
inline __device__ T exp(T x)
{
    return ck::type_convert<T>(__ocml_exp_f32(ck::type_convert<float>(x)));
};

template <>
inline __device__ half_t exp<half_t>(half_t x)
{
    return hexp(static_cast<__half>(x));
};

template <>
inline __device__ float exp<float>(float x)
{
    return __ocml_exp_f32(x);
};

template <>
inline __device__ double exp<double>(double x)
{
    return exp(x);
};

template <typename T>
inline __device__ T log(T x)
{
    return ck::type_convert<T>(__logf(ck::type_convert<float>(x)));
};

template <>
inline __device__ half_t log<half_t>(half_t x)
{
    return hlog(static_cast<__half>(x));
};

template <>
inline __device__ float log<float>(float x)
{
    return __logf(x);
};

template <>
inline __device__ double log<double>(double x)
{
    return log(x);
};

template <typename T>
inline __device__ T pow(T x, T gamma)
{
    return ck::type_convert<T>(powf(ck::type_convert<float>(x), ck::type_convert<float>(gamma)));
};

template <>
inline __device__ float pow<float>(float x, float gamma)
{
    return powf(x, gamma);
};

template <>
inline __device__ double pow<double>(double x, double gamma)
{
    return pow(x, gamma);
};

template <typename T>
inline __device__ T expm1(T x)
{
    return ck::type_convert<T>(expm1f(ck::type_convert<float>(x)));
};

template <>
inline __device__ float expm1<float>(float x)
{
    return expm1f(x);
};

template <>
inline __device__ double expm1<double>(double x)
{
    return expm1(x);
};

template <typename T>
inline __device__ T cos(T x)
{
    return ck::type_convert<T>(cosf(ck::type_convert<float>(x)));
};

template <>
inline __device__ float cos<float>(float x)
{
    return cosf(x);
};

template <>
inline __device__ double cos<double>(double x)
{
    return cos(x);
};

} // namespace math
} // namespace ck
