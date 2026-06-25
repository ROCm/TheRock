/* ************************************************************************
 * Copyright (C) 2016-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#include <hipblas/hipblas.h>
#include <stdbool.h>

#ifdef __cplusplus
#include <cmath>
#include <complex>
#include <cstdio>
#include <immintrin.h>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>
#endif

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus

// Return true if value is NaN
template <typename T>
inline bool hipblas_isnan(T)
{
    return false;
}
inline bool hipblas_isnan(double arg)
{
    return std::isnan(arg);
}
inline bool hipblas_isnan(float arg)
{
    return std::isnan(arg);
}

inline bool hipblas_isnan(hipblasHalf arg)
{
    auto half_data = static_cast<unsigned short>(arg);
    return (~(half_data)&0x7c00) == 0 && (half_data & 0x3ff) != 0;
}
inline bool hipblas_isnan(std::complex<float> arg)
{
    return std::isnan(arg.real()) || std::isnan(arg.imag());
}
inline bool hipblas_isnan(std::complex<double> arg)
{
    return std::isnan(arg.real()) || std::isnan(arg.imag());
}

inline hipblasHalf float_to_half(float val)
{
#ifdef HIPBLAS_USE_HIP_HALF
    return __float2half(val);
#else
    uint16_t a = _cvtss_sh(val, 0);
    return a;
#endif
}

inline float bfloat16_to_float(hipblasBfloat16 bf)
{
    union
    {
        uint32_t int32;
        float    fp32;
    } u = {uint32_t(bf.data) << 16};
    return u.fp32;
}

inline hipblasBfloat16 float_to_bfloat16(float f)
{
    hipblasBfloat16 rv;
    union
    {
        float    fp32;
        uint32_t int32;
    } u = {f};
    if(~u.int32 & 0x7f800000)
    {
        u.int32 += 0x7fff + ((u.int32 >> 16) & 1); // Round to nearest, round to even
    }
    else if(u.int32 & 0xffff)
    {
        u.int32 |= 0x10000; // Preserve signaling NaN
    }
    rv.data = uint16_t(u.int32 >> 16);
    return rv;
}

inline float half_to_float(hipblasHalf val)
{
#ifdef HIPBLAS_USE_HIP_HALF
    return __half2float(val);
#else
    return _cvtsh_ss(val);
#endif
}

/* =============================================================================================== */
/* Absolute values                                                                                 */
// template <typename T>
// inline T hipblas_abs(const T& x)
// {
//     return x < 0 ? -x : x;
// }

// template <>
// inline double hipblas_abs(const hipblasHalf& x)
// {
//     return std::abs(half_to_float(x));
// }

// template <>
// inline double hipblas_abs(const hipblasBfloat16& x)
// {
//     return std::abs(bfloat16_to_float(x));
// }

// rocblas_bfloat16 is handled specially
inline hipblasBfloat16 hipblas_abs(hipblasBfloat16 x)
{
    x.data &= 0x7fff;
    return x;
}

// rocblas_half
inline hipblasHalf hipblas_abs(hipblasHalf x)
{
    union
    {
        hipblasHalf x;
        uint16_t    data;
    } t = {x};
    t.data &= 0x7fff;
    return t.x;
}

inline double hipblas_abs(const double& x)
{
    return x < 0 ? -x : x;
}
inline float hipblas_abs(const float& x)
{
    return x < 0 ? -x : x;
}

inline double hipblas_abs(const std::complex<float>& x)
{
    return std::abs(reinterpret_cast<const std::complex<float>&>(x));
}

inline double hipblas_abs(const std::complex<double>& x)
{
    return std::abs(reinterpret_cast<const std::complex<double>&>(x));
}

inline int hipblas_abs(const int& x)
{
    return x < 0 ? -x : x;
}

inline int64_t hipblas_abs(const int64_t& x)
{
    return x < 0 ? -x : x;
}

/* =============================================================================================== */
/* Complex / real helpers.                                                                         */
template <typename T>
inline constexpr bool is_complex = false;

template <>
inline constexpr bool is_complex<std::complex<float>> = true;

template <>
inline constexpr bool is_complex<std::complex<double>> = true;

// Get base types from complex types.
template <typename T>
struct real_t_impl
{
    using type = T;
};

template <>
struct real_t_impl<std::complex<float>>
{
    using type = float;
};

template <>
struct real_t_impl<std::complex<double>>
{
    using type = double;
};

template <typename T>
using real_t = typename real_t_impl<T>::type;

// Using std::complex for host-side on the client, but need hipComplex for calls to hipblas
template <typename T>
struct hipblas_internal_type_impl
{
    using type = T;
};

template <>
struct hipblas_internal_type_impl<std::complex<float>>
{
    using type = hipComplex;
};

template <>
struct hipblas_internal_type_impl<std::complex<double>>
{
    using type = hipDoubleComplex;
};

template <typename T>
using hipblas_internal_type = typename hipblas_internal_type_impl<T>::type;

// Conjugate a value. For most types, simply return argument; for
// std::complex<float> and std::complex<double>, return std::conj(z)
template <typename T, std::enable_if_t<!is_complex<T>, int> = 0>
__device__ __host__ inline T hipblas_conjugate(const T& z)
{
    return z;
}

template <typename T, std::enable_if_t<is_complex<T>, int> = 0>
__device__ __host__ inline T hipblas_conjugate(const T& z)
{
    return std::conj(z);
}

// std::complex<float> and std::complex<double>, return real
template <typename T, std::enable_if_t<!is_complex<T>, int> = 0>
__device__ __host__ inline T hipblas_real(const T& z)
{
    return z;
}

template <typename T, std::enable_if_t<is_complex<T>, int> = 0>
__device__ __host__ inline T hipblas_real(const T& z)
{
    return std::real(z);
}

#endif // __cplusplus
