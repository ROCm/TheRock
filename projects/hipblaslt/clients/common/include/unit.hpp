/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
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
 *******************************************************************************/

/* ========================================Gtest Unit Check
 * ==================================================== */

/*! \brief gtest unit compare two matrices float/double/complex */

#pragma once

#include "hipblaslt_math.hpp"
#include "hipblaslt_ostream.hpp"
#include "hipblaslt_test.hpp"
#include "hipblaslt_vector.hpp"
#include <hipblaslt/hipblaslt.h>

#ifndef GOOGLE_TEST
#define UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, UNIT_ASSERT_EQ)
#define UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, UNIT_ASSERT_EQ)
#else
#define UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, UNIT_ASSERT_EQ)                \
    do                                                                                         \
    {                                                                                          \
        for(size_t k = 0; k < batch_count; k++)                                                \
            for(size_t j = 0; j < N; j++)                                                      \
                for(size_t i = 0; i < M; i++)                                                  \
                    if(hipblaslt_isnan(hCPU[i + j * size_t(lda) + k * strideA]))               \
                    {                                                                          \
                        ASSERT_TRUE(hipblaslt_isnan(hGPU[i + j * size_t(lda) + k * strideA])); \
                    }                                                                          \
                    else                                                                       \
                    {                                                                          \
                        UNIT_ASSERT_EQ(hCPU[i + j * size_t(lda) + k * strideA],                \
                                       hGPU[i + j * size_t(lda) + k * strideA]);               \
                    }                                                                          \
    } while(0)

#define UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, UNIT_ASSERT_EQ)            \
    do                                                                              \
    {                                                                               \
        for(size_t k = 0; k < batch_count; k++)                                     \
            for(size_t j = 0; j < N; j++)                                           \
                for(size_t i = 0; i < M; i++)                                       \
                    if(hipblaslt_isnan(hCPU[k][i + j * size_t(lda)]))               \
                    {                                                               \
                        ASSERT_TRUE(hipblaslt_isnan(hGPU[k][i + j * size_t(lda)])); \
                    }                                                               \
                    else                                                            \
                    {                                                               \
                        UNIT_ASSERT_EQ(hCPU[k][i + j * size_t(lda)],                \
                                       hGPU[k][i + j * size_t(lda)]);               \
                    }                                                               \
    } while(0)

#define ASSERT_HALF_EQ(a, b) ASSERT_FLOAT_EQ(float(a), float(b))
#define ASSERT_BF16_EQ(a, b) ASSERT_FLOAT_EQ(float(a), float(b))
#define ASSERT_F8_EQ(a, b) ASSERT_FLOAT_EQ(float(a), float(b))
#define ASSERT_BF8_EQ(a, b) ASSERT_FLOAT_EQ(float(a), float(b))

// Compare float to hip_bfloat16
// Allow the hip_bfloat16 to match the rounded or truncated value of float
// Only call ASSERT_FLOAT_EQ with the rounded value if the truncated value does not match
#include <gtest/internal/gtest-internal.h>
#define ASSERT_FLOAT_BF16_EQ(a, b)                                   \
    do                                                               \
    {                                                                \
        using testing::internal::FloatingPoint;                      \
        if(!FloatingPoint<float>(b).AlmostEquals(                    \
               FloatingPoint<float>(float_to_bfloat16_truncate(a)))) \
            ASSERT_FLOAT_EQ(b, hip_bfloat16(a));                     \
    } while(0)

#define ASSERT_FLOAT_COMPLEX_EQ(a, b)                  \
    do                                                 \
    {                                                  \
        auto ta = (a), tb = (b);                       \
        ASSERT_FLOAT_EQ(std::real(ta), std::real(tb)); \
        ASSERT_FLOAT_EQ(std::imag(ta), std::imag(tb)); \
    } while(0)

#define ASSERT_DOUBLE_COMPLEX_EQ(a, b)                  \
    do                                                  \
    {                                                   \
        auto ta = (a), tb = (b);                        \
        ASSERT_DOUBLE_EQ(std::real(ta), std::real(tb)); \
        ASSERT_DOUBLE_EQ(std::imag(ta), std::imag(tb)); \
    } while(0)

#endif // GOOGLE_TEST

// TODO: Replace std::remove_cv_t with std::type_identity_t in C++20
// It is only used to make T_hpa non-deduced
template <typename T, typename T_hpa = T>
inline void unit_check_general(
    int64_t M, int64_t N, int64_t lda, const std::remove_cv_t<T_hpa>* hCPU, const T* hGPU);

template <>
inline void unit_check_general(
    int64_t M, int64_t N, int64_t lda, const hipblaslt_f8_fnuz* hCPU, const hipblaslt_f8_fnuz* hGPU)
{
    UNIT_CHECK(M, N, lda, 0, hCPU, hGPU, 1, ASSERT_F8_EQ);
}

template <>
inline void unit_check_general(int64_t                   M,
                               int64_t                   N,
                               int64_t                   lda,
                               const hipblaslt_bf8_fnuz* hCPU,
                               const hipblaslt_bf8_fnuz* hGPU)
{
    UNIT_CHECK(M, N, lda, 0, hCPU, hGPU, 1, ASSERT_BF8_EQ);
}

template <>
inline void unit_check_general(
    int64_t M, int64_t N, int64_t lda, const hipblaslt_f8* hCPU, const hipblaslt_f8* hGPU)
{
    UNIT_CHECK(M, N, lda, 0, hCPU, hGPU, 1, ASSERT_F8_EQ);
}

template <>
inline void unit_check_general(
    int64_t M, int64_t N, int64_t lda, const hipblaslt_bf8* hCPU, const hipblaslt_bf8* hGPU)
{
    UNIT_CHECK(M, N, lda, 0, hCPU, hGPU, 1, ASSERT_BF8_EQ);
}

template <>
inline void unit_check_general(
    int64_t M, int64_t N, int64_t lda, const hip_bfloat16* hCPU, const hip_bfloat16* hGPU)
{
    UNIT_CHECK(M, N, lda, 0, hCPU, hGPU, 1, ASSERT_BF16_EQ);
}

template <>
inline void unit_check_general<hip_bfloat16, float>(
    int64_t M, int64_t N, int64_t lda, const float* hCPU, const hip_bfloat16* hGPU)
{
    UNIT_CHECK(M, N, lda, 0, hCPU, hGPU, 1, ASSERT_FLOAT_BF16_EQ);
}

template <>
inline void unit_check_general(
    int64_t M, int64_t N, int64_t lda, const hipblasLtHalf* hCPU, const hipblasLtHalf* hGPU)
{
    UNIT_CHECK(M, N, lda, 0, hCPU, hGPU, 1, ASSERT_HALF_EQ);
}

template <>
inline void
    unit_check_general(int64_t M, int64_t N, int64_t lda, const float* hCPU, const float* hGPU)
{
    UNIT_CHECK(M, N, lda, 0, hCPU, hGPU, 1, ASSERT_FLOAT_EQ);
}

template <>
inline void
    unit_check_general(int64_t M, int64_t N, int64_t lda, const double* hCPU, const double* hGPU)
{
    UNIT_CHECK(M, N, lda, 0, hCPU, hGPU, 1, ASSERT_DOUBLE_EQ);
}

template <>
inline void
    unit_check_general(int64_t M, int64_t N, int64_t lda, const int64_t* hCPU, const int64_t* hGPU)
{
    UNIT_CHECK(M, N, lda, 0, hCPU, hGPU, 1, ASSERT_EQ);
}

template <>
inline void
    unit_check_general(int64_t M, int64_t N, int64_t lda, const int8_t* hCPU, const int8_t* hGPU)
{
    UNIT_CHECK(M, N, lda, 0, hCPU, hGPU, 1, ASSERT_EQ);
}

template <typename T, typename T_hpa = T>
inline void unit_check_general(int64_t                        M,
                               int64_t                        N,
                               int64_t                        lda,
                               int64_t                        strideA,
                               const std::remove_cv_t<T_hpa>* hCPU,
                               const T*                       hGPU,
                               int64_t                        batch_count);

template <>
inline void unit_check_general(int64_t             M,
                               int64_t             N,
                               int64_t             lda,
                               int64_t             strideA,
                               const hip_bfloat16* hCPU,
                               const hip_bfloat16* hGPU,
                               int64_t             batch_count)
{
    UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, ASSERT_BF16_EQ);
}

template <>
inline void unit_check_general(int64_t                  M,
                               int64_t                  N,
                               int64_t                  lda,
                               int64_t                  strideA,
                               const hipblaslt_f8_fnuz* hCPU,
                               const hipblaslt_f8_fnuz* hGPU,
                               int64_t                  batch_count)
{
    UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, ASSERT_F8_EQ);
}

template <>
inline void unit_check_general(int64_t                   M,
                               int64_t                   N,
                               int64_t                   lda,
                               int64_t                   strideA,
                               const hipblaslt_bf8_fnuz* hCPU,
                               const hipblaslt_bf8_fnuz* hGPU,
                               int64_t                   batch_count)
{
    UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, ASSERT_BF8_EQ);
}

template <>
inline void unit_check_general(int64_t                 M,
                               int64_t                 N,
                               int64_t                 lda,
                               int64_t                 strideA,
                               const hipblaslt_f8* hCPU,
                               const hipblaslt_f8* hGPU,
                               int64_t                 batch_count)
{
    UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, ASSERT_F8_EQ);
}

template <>
inline void unit_check_general(int64_t                  M,
                               int64_t                  N,
                               int64_t                  lda,
                               int64_t                  strideA,
                               const hipblaslt_bf8* hCPU,
                               const hipblaslt_bf8* hGPU,
                               int64_t                  batch_count)
{
    UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, ASSERT_BF8_EQ);
}

template <>
inline void unit_check_general<hip_bfloat16, float>(int64_t             M,
                                                    int64_t             N,
                                                    int64_t             lda,
                                                    int64_t             strideA,
                                                    const float*        hCPU,
                                                    const hip_bfloat16* hGPU,
                                                    int64_t             batch_count)
{
    UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, ASSERT_FLOAT_BF16_EQ);
}

template <>
inline void unit_check_general(int64_t              M,
                               int64_t              N,
                               int64_t              lda,
                               int64_t              strideA,
                               const hipblasLtHalf* hCPU,
                               const hipblasLtHalf* hGPU,
                               int64_t              batch_count)
{
    UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, ASSERT_HALF_EQ);
}

template <>
inline void unit_check_general(int64_t      M,
                               int64_t      N,
                               int64_t      lda,
                               int64_t      strideA,
                               const float* hCPU,
                               const float* hGPU,
                               int64_t      batch_count)
{
    UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, ASSERT_FLOAT_EQ);
}

template <>
inline void unit_check_general(int64_t       M,
                               int64_t       N,
                               int64_t       lda,
                               int64_t       strideA,
                               const double* hCPU,
                               const double* hGPU,
                               int64_t       batch_count)
{
    UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, ASSERT_DOUBLE_EQ);
}

template <>
inline void unit_check_general(int64_t        M,
                               int64_t        N,
                               int64_t        lda,
                               int64_t        strideA,
                               const int64_t* hCPU,
                               const int64_t* hGPU,
                               int64_t        batch_count)
{
    UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, ASSERT_EQ);
}

template <>
inline void unit_check_general(int64_t       M,
                               int64_t       N,
                               int64_t       lda,
                               int64_t       strideA,
                               const int8_t* hCPU,
                               const int8_t* hGPU,
                               int64_t       batch_count)
{
    UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, ASSERT_EQ);
}

template <>
inline void unit_check_general(int64_t        M,
                               int64_t        N,
                               int64_t        lda,
                               int64_t        strideA,
                               const int32_t* hCPU,
                               const int32_t* hGPU,
                               int64_t        batch_count)
{
    UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, ASSERT_EQ);
}

template <typename T, typename T_hpa = T>
inline void unit_check_general(int64_t                                    M,
                               int64_t                                    N,
                               int64_t                                    lda,
                               const host_vector<std::remove_cv_t<T_hpa>> hCPU[],
                               const host_vector<T>                       hGPU[],
                               int64_t                                    batch_count);

template <>
inline void unit_check_general(int64_t                         M,
                               int64_t                         N,
                               int64_t                         lda,
                               const host_vector<hip_bfloat16> hCPU[],
                               const host_vector<hip_bfloat16> hGPU[],
                               int64_t                         batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_BF16_EQ);
}

template <>
inline void unit_check_general<hip_bfloat16, float>(int64_t                         M,
                                                    int64_t                         N,
                                                    int64_t                         lda,
                                                    const host_vector<float>        hCPU[],
                                                    const host_vector<hip_bfloat16> hGPU[],
                                                    int64_t                         batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_FLOAT_BF16_EQ);
}

template <>
inline void unit_check_general(int64_t                          M,
                               int64_t                          N,
                               int64_t                          lda,
                               const host_vector<hipblasLtHalf> hCPU[],
                               const host_vector<hipblasLtHalf> hGPU[],
                               int64_t                          batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_HALF_EQ);
}

template <>
inline void unit_check_general(int64_t                M,
                               int64_t                N,
                               int64_t                lda,
                               const host_vector<int> hCPU[],
                               const host_vector<int> hGPU[],
                               int64_t                batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_EQ);
}

template <>
inline void unit_check_general(int64_t                   M,
                               int64_t                   N,
                               int64_t                   lda,
                               const host_vector<int8_t> hCPU[],
                               const host_vector<int8_t> hGPU[],
                               int64_t                   batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_EQ);
}

template <>
inline void unit_check_general(int64_t                  M,
                               int64_t                  N,
                               int64_t                  lda,
                               const host_vector<float> hCPU[],
                               const host_vector<float> hGPU[],
                               int64_t                  batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_FLOAT_EQ);
}

template <>
inline void unit_check_general(int64_t                   M,
                               int64_t                   N,
                               int64_t                   lda,
                               const host_vector<double> hCPU[],
                               const host_vector<double> hGPU[],
                               int64_t                   batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_DOUBLE_EQ);
}

template <typename T, typename T_hpa = T>
inline void unit_check_general(int64_t                              M,
                               int64_t                              N,
                               int64_t                              lda,
                               const std::remove_cv_t<T_hpa>* const hCPU[],
                               const T* const                       hGPU[],
                               int64_t                              batch_count);

template <>
inline void unit_check_general(int64_t                   M,
                               int64_t                   N,
                               int64_t                   lda,
                               const hip_bfloat16* const hCPU[],
                               const hip_bfloat16* const hGPU[],
                               int64_t                   batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_BF16_EQ);
}

template <>
inline void unit_check_general<hip_bfloat16, float>(int64_t                   M,
                                                    int64_t                   N,
                                                    int64_t                   lda,
                                                    const float* const        hCPU[],
                                                    const hip_bfloat16* const hGPU[],
                                                    int64_t                   batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_FLOAT_BF16_EQ);
}

template <>
inline void unit_check_general(int64_t                    M,
                               int64_t                    N,
                               int64_t                    lda,
                               const hipblasLtHalf* const hCPU[],
                               const hipblasLtHalf* const hGPU[],
                               int64_t                    batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_HALF_EQ);
}

template <>
inline void unit_check_general(int64_t          M,
                               int64_t          N,
                               int64_t          lda,
                               const int* const hCPU[],
                               const int* const hGPU[],
                               int64_t          batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_EQ);
}

template <>
inline void unit_check_general(int64_t             M,
                               int64_t             N,
                               int64_t             lda,
                               const int8_t* const hCPU[],
                               const int8_t* const hGPU[],
                               int64_t             batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_EQ);
}

template <>
inline void unit_check_general(int64_t            M,
                               int64_t            N,
                               int64_t            lda,
                               const float* const hCPU[],
                               const float* const hGPU[],
                               int64_t            batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_FLOAT_EQ);
}

template <>
inline void unit_check_general(int64_t             M,
                               int64_t             N,
                               int64_t             lda,
                               const double* const hCPU[],
                               const double* const hGPU[],
                               int64_t             batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_DOUBLE_EQ);
}

template <typename T>
constexpr double get_epsilon()
{
    return std::numeric_limits<T>::epsilon();
}

template <typename T>
inline int64_t unit_check_diff(
    int64_t M, int64_t N, int64_t lda, int64_t stride, T* hCPU, T* hGPU, int64_t batch_count)
{
    using c_type  = std::conditional_t<std::is_same<hipblasLtHalf, T>::value, float, T>;
    int64_t error = 0;
    do
    {
        for(size_t k = 0; k < batch_count; k++)
            for(size_t j = 0; j < N; j++)
                for(size_t i = 0; i < M; i++)
                    if(hipblaslt_isnan(hCPU[i + j * size_t(lda) + k * stride]))
                    {
                        error += hipblaslt_isnan(hGPU[i + j * size_t(lda) + k * stride]) ? 0 : 1;
                    }
                    else
                    {
                        error += static_cast<c_type>(hCPU[i + j * size_t(lda) + k * stride])
                                         == static_cast<c_type>(
                                             hGPU[i + j * size_t(lda) + k * stride])
                                     ? 0
                                     : 1;
                    }
    } while(0);
    return error;
}

inline void unit_check_general(int64_t     M,
                               int64_t     N,
                               int64_t     lda,
                               int64_t     strideA,
                               void*       hCPU,
                               void*       hGPU,
                               int64_t     batch_count,
                               hipDataType type)
{
    switch(type)
    {
    case HIP_R_32F:
        unit_check_general(
            M, N, lda, strideA, static_cast<float*>(hCPU), static_cast<float*>(hGPU), batch_count);
        break;
    case HIP_R_64F:
        unit_check_general(M,
                           N,
                           lda,
                           strideA,
                           static_cast<double*>(hCPU),
                           static_cast<double*>(hGPU),
                           batch_count);
        break;
    case HIP_R_16F:
        unit_check_general(M,
                           N,
                           lda,
                           strideA,
                           static_cast<hipblasLtHalf*>(hCPU),
                           static_cast<hipblasLtHalf*>(hGPU),
                           batch_count);
        break;
    case HIP_R_16BF:
        unit_check_general(M,
                           N,
                           lda,
                           strideA,
                           static_cast<hip_bfloat16*>(hCPU),
                           static_cast<hip_bfloat16*>(hGPU),
                           batch_count);
        break;
    case HIP_R_8F_E4M3_FNUZ:
        unit_check_general(M,
                           N,
                           lda,
                           strideA,
                           static_cast<hipblaslt_f8_fnuz*>(hCPU),
                           static_cast<hipblaslt_f8_fnuz*>(hGPU),
                           batch_count);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        unit_check_general(M,
                           N,
                           lda,
                           strideA,
                           static_cast<hipblaslt_bf8_fnuz*>(hCPU),
                           static_cast<hipblaslt_bf8_fnuz*>(hGPU),
                           batch_count);
        break;
    case HIP_R_8F_E4M3:
        unit_check_general(M,
                           N,
                           lda,
                           strideA,
                           static_cast<hipblaslt_f8*>(hCPU),
                           static_cast<hipblaslt_f8*>(hGPU),
                           batch_count);
        break;
    case HIP_R_8F_E5M2:
        unit_check_general(M,
                           N,
                           lda,
                           strideA,
                           static_cast<hipblaslt_bf8*>(hCPU),
                           static_cast<hipblaslt_bf8*>(hGPU),
                           batch_count);
        break;
    case HIP_R_32I:
        unit_check_general(M,
                           N,
                           lda,
                           strideA,
                           static_cast<int32_t*>(hCPU),
                           static_cast<int32_t*>(hGPU),
                           batch_count);
        break;
    case HIP_R_8I:
        unit_check_general(M,
                           N,
                           lda,
                           strideA,
                           static_cast<hipblasLtInt8*>(hCPU),
                           static_cast<hipblasLtInt8*>(hGPU),
                           batch_count);
        break;
    default:
        hipblaslt_cerr << "Error type in unit_check_general" << std::endl;
        break;
    }
}
