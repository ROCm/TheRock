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

/* =====================================================================
    Google Near check: ASSERT_NEAR( elementof(A), elementof(B))
   =================================================================== */

/*!\file
 * \brief compares two results (usually, CPU and GPU results); provides Google Near check.
 */

#pragma once

#include "hipBuffer.hpp"
#include "hipblaslt_math.hpp"
#include "hipblaslt_ostream.hpp"
#include "hipblaslt_test.hpp"
#include "hipblaslt_vector.hpp"
#include <hipblaslt/hipblaslt.h>

template <class Tc, class Ti, class To>
static constexpr double sum_error_tolerance_for_gfx11 = std::numeric_limits<Tc>::epsilon();

template <>
constexpr double sum_error_tolerance_for_gfx11<float, hip_bfloat16, float> = 1 / 10.0;

template <>
constexpr double sum_error_tolerance_for_gfx11<float, hip_bfloat16, hip_bfloat16> = 1 / 10.0;

template <>
constexpr double sum_error_tolerance_for_gfx11<float, hipblasLtHalf, float> = 1 / 100.0;

template <>
constexpr double
    sum_error_tolerance_for_gfx11<float, hipblasLtHalf, hipblasLtHalf> = 1 / 100.0;

template <>
constexpr double
    sum_error_tolerance_for_gfx11<hipblasLtHalf, hipblasLtHalf, hipblasLtHalf> = 1 / 100.0;

double sum_error_tolerance_for_gfx11_type(hipDataType Tc, hipDataType Ti, hipDataType To)
{
    if(Tc == HIP_R_32F && Ti == HIP_R_16BF && (To == HIP_R_32F || To == HIP_R_16BF))
        return 1 / 10.0;
    else if(Tc == HIP_R_32F && Ti == HIP_R_16F && (To == HIP_R_32F || To == HIP_R_16F))
        return 1 / 100.0;
    else if(Tc == HIP_R_16F && Ti == HIP_R_16F && To == HIP_R_16F)
        return 1 / 100.0;
    else
    {
        switch(Tc)
        {
        case HIP_R_32F:
            return std::numeric_limits<float>::epsilon();
        case HIP_R_64F:
            return std::numeric_limits<double>::epsilon();
        case HIP_R_16F:
            return std::numeric_limits<hipblasLtHalf>::epsilon();
        case HIP_R_32I:
            return std::numeric_limits<int32_t>::epsilon();
        default:
            hipblaslt_cerr << "Error type in sum_error_tolerance_for_gfx11_type" << std::endl;
            return 0.0;
        }
    }
};

#ifndef GOOGLE_TEST
#define NEAR_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, err, NEAR_ASSERT)
#define NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, err, NEAR_ASSERT)
#else

#define NEAR_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, err, NEAR_ASSERT) \
    do                                                                            \
    {                                                                             \
        for(size_t k = 0; k < batch_count; k++)                                   \
            for(size_t j = 0; j < N; j++)                                         \
                for(size_t i = 0; i < M; i++)                                     \
                    NEAR_ASSERT(hCPU[i + j * size_t(lda) + k * strideA],          \
                                hGPU[i + j * size_t(lda) + k * strideA],          \
                                err);                                             \
    } while(0)

#define NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, err, NEAR_ASSERT)                    \
    do                                                                                        \
    {                                                                                         \
        for(size_t k = 0; k < batch_count; k++)                                               \
            for(size_t j = 0; j < N; j++)                                                     \
                for(size_t i = 0; i < M; i++)                                                 \
                    if(hipblaslt_isnan(hCPU[k][i + j * size_t(lda)]))                         \
                    {                                                                         \
                        ASSERT_TRUE(hipblaslt_isnan(hGPU[k][i + j * size_t(lda)]));           \
                    }                                                                         \
                    else                                                                      \
                    {                                                                         \
                        NEAR_ASSERT(                                                          \
                            hCPU[k][i + j * size_t(lda)], hGPU[k][i + j * size_t(lda)], err); \
                    }                                                                         \
    } while(0)

#endif

#define NEAR_ASSERT_HALF(a, b, err) ASSERT_NEAR(double(a), double(b), err)

#define NEAR_ASSERT_BF16(a, b, err) ASSERT_NEAR(double(a), double(b), err)

#define NEAR_ASSERT_FP8(a, b, err) ASSERT_NEAR(double(float(a)), double(float(b)), err)

#define NEAR_ASSERT_COMPLEX(a, b, err)                  \
    do                                                  \
    {                                                   \
        auto ta = (a), tb = (b);                        \
        ASSERT_NEAR(std::real(ta), std::real(tb), err); \
        ASSERT_NEAR(std::imag(ta), std::imag(tb), err); \
    } while(0)

// TODO: Replace std::remove_cv_t with std::type_identity_t in C++20
// It is only used to make T_hpa non-deduced
template <typename T, typename T_hpa = T>
inline void near_check_general(int64_t                        M,
                               int64_t                        N,
                               int64_t                        lda,
                               const std::remove_cv_t<T_hpa>* hCPU,
                               const T*                       hGPU,
                               double                         abs_error)
{
    NEAR_CHECK(M, N, lda, 0, hCPU, hGPU, 1, abs_error, ASSERT_NEAR);
}

template <>
inline void near_check_general(int64_t              M,
                               int64_t              N,
                               int64_t              lda,
                               const hipblasLtHalf* hCPU,
                               const hipblasLtHalf* hGPU,
                               double               abs_error)
{
    NEAR_CHECK(M, N, lda, 0, hCPU, hGPU, 1, abs_error, NEAR_ASSERT_HALF);
}

template <>
inline void near_check_general<hip_bfloat16, float>(int64_t             M,
                                                    int64_t             N,
                                                    int64_t             lda,
                                                    const float*        hCPU,
                                                    const hip_bfloat16* hGPU,
                                                    double              abs_error)
{
    NEAR_CHECK(M, N, lda, 0, hCPU, hGPU, 1, abs_error, NEAR_ASSERT_BF16);
}

template <>
inline void near_check_general(int64_t                  M,
                               int64_t                  N,
                               int64_t                  lda,
                               const hipblaslt_f8_fnuz* hCPU,
                               const hipblaslt_f8_fnuz* hGPU,
                               double                   abs_error)
{
    NEAR_CHECK(M, N, lda, 0, hCPU, hGPU, 1, abs_error, NEAR_ASSERT_FP8);
}

template <>
inline void near_check_general(int64_t                   M,
                               int64_t                   N,
                               int64_t                   lda,
                               const hipblaslt_bf8_fnuz* hCPU,
                               const hipblaslt_bf8_fnuz* hGPU,
                               double                    abs_error)
{
    NEAR_CHECK(M, N, lda, 0, hCPU, hGPU, 1, abs_error, NEAR_ASSERT_FP8);
}

template <>
inline void near_check_general(int64_t                 M,
                               int64_t                 N,
                               int64_t                 lda,
                               const hipblaslt_f8* hCPU,
                               const hipblaslt_f8* hGPU,
                               double                  abs_error)
{
    NEAR_CHECK(M, N, lda, 0, hCPU, hGPU, 1, abs_error, NEAR_ASSERT_FP8);
}

template <>
inline void near_check_general(int64_t                  M,
                               int64_t                  N,
                               int64_t                  lda,
                               const hipblaslt_bf8* hCPU,
                               const hipblaslt_bf8* hGPU,
                               double                   abs_error)
{
    NEAR_CHECK(M, N, lda, 0, hCPU, hGPU, 1, abs_error, NEAR_ASSERT_FP8);
}

template <typename T, typename T_hpa = T>
inline void near_check_general(int64_t                        M,
                               int64_t                        N,
                               int64_t                        lda,
                               int64_t                        strideA,
                               const std::remove_cv_t<T_hpa>* hCPU,
                               const T*                       hGPU,
                               int64_t                        batch_count,
                               double                         abs_error)
{
    NEAR_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, abs_error, ASSERT_NEAR);
}

template <>
inline void near_check_general(int64_t              M,
                               int64_t              N,
                               int64_t              lda,
                               int64_t              strideA,
                               const hipblasLtHalf* hCPU,
                               const hipblasLtHalf* hGPU,
                               int64_t              batch_count,
                               double               abs_error)
{
    NEAR_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_HALF);
}

template <>
inline void near_check_general<hip_bfloat16, float>(int64_t             M,
                                                    int64_t             N,
                                                    int64_t             lda,
                                                    int64_t             strideA,
                                                    const float*        hCPU,
                                                    const hip_bfloat16* hGPU,
                                                    int64_t             batch_count,
                                                    double              abs_error)
{
    NEAR_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_BF16);
}

template <>
inline void near_check_general(int64_t                  M,
                               int64_t                  N,
                               int64_t                  lda,
                               int64_t                  strideA,
                               const hipblaslt_f8_fnuz* hCPU,
                               const hipblaslt_f8_fnuz* hGPU,
                               int64_t                  batch_count,
                               double                   abs_error)
{
    NEAR_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_FP8);
}

template <>
inline void near_check_general(int64_t                   M,
                               int64_t                   N,
                               int64_t                   lda,
                               int64_t                   strideA,
                               const hipblaslt_bf8_fnuz* hCPU,
                               const hipblaslt_bf8_fnuz* hGPU,
                               int64_t                   batch_count,
                               double                    abs_error)
{
    NEAR_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_FP8);
}

template <>
inline void near_check_general(int64_t                 M,
                               int64_t                 N,
                               int64_t                 lda,
                               int64_t                 strideA,
                               const hipblaslt_f8* hCPU,
                               const hipblaslt_f8* hGPU,
                               int64_t                 batch_count,
                               double                  abs_error)
{
    NEAR_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_FP8);
}

template <>
inline void near_check_general(int64_t                  M,
                               int64_t                  N,
                               int64_t                  lda,
                               int64_t                  strideA,
                               const hipblaslt_bf8* hCPU,
                               const hipblaslt_bf8* hGPU,
                               int64_t                  batch_count,
                               double                   abs_error)
{
    NEAR_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_FP8);
}

template <typename T, typename T_hpa = T>
void near_check_general(int64_t                                    M,
                        int64_t                                    N,
                        int64_t                                    lda,
                        const host_vector<std::remove_cv_t<T_hpa>> hCPU[],
                        const host_vector<T>                       hGPU[],
                        int64_t                                    batch_count,
                        double                                     abs_error)
{
    NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, abs_error, ASSERT_NEAR);
}

template <>
inline void near_check_general(int64_t                          M,
                               int64_t                          N,
                               int64_t                          lda,
                               const host_vector<hipblasLtHalf> hCPU[],
                               const host_vector<hipblasLtHalf> hGPU[],
                               int64_t                          batch_count,
                               double                           abs_error)
{
    NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_HALF);
}
template <>
inline void near_check_general<hip_bfloat16, float>(int64_t                         M,
                                                    int64_t                         N,
                                                    int64_t                         lda,
                                                    const host_vector<float>        hCPU[],
                                                    const host_vector<hip_bfloat16> hGPU[],
                                                    int64_t                         batch_count,
                                                    double                          abs_error)
{
    NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_BF16);
}

template <>
inline void near_check_general(int64_t                              M,
                               int64_t                              N,
                               int64_t                              lda,
                               const host_vector<hipblaslt_f8_fnuz> hCPU[],
                               const host_vector<hipblaslt_f8_fnuz> hGPU[],
                               int64_t                              batch_count,
                               double                               abs_error)
{
    NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_FP8);
}

template <>
inline void near_check_general(int64_t                               M,
                               int64_t                               N,
                               int64_t                               lda,
                               const host_vector<hipblaslt_bf8_fnuz> hCPU[],
                               const host_vector<hipblaslt_bf8_fnuz> hGPU[],
                               int64_t                               batch_count,
                               double                                abs_error)
{
    NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_FP8);
}

template <>
inline void near_check_general(int64_t                             M,
                               int64_t                             N,
                               int64_t                             lda,
                               const host_vector<hipblaslt_f8> hCPU[],
                               const host_vector<hipblaslt_f8> hGPU[],
                               int64_t                             batch_count,
                               double                              abs_error)
{
    NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_FP8);
}

template <>
inline void near_check_general(int64_t                              M,
                               int64_t                              N,
                               int64_t                              lda,
                               const host_vector<hipblaslt_bf8> hCPU[],
                               const host_vector<hipblaslt_bf8> hGPU[],
                               int64_t                              batch_count,
                               double                               abs_error)
{
    NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_FP8);
}

template <typename T, typename T_hpa = T>
inline void near_check_general(int64_t                              M,
                               int64_t                              N,
                               int64_t                              lda,
                               const std::remove_cv_t<T_hpa>* const hCPU[],
                               const T* const                       hGPU[],
                               int64_t                              batch_count,
                               double                               abs_error)
{
    NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, abs_error, ASSERT_NEAR);
}

template <>
inline void near_check_general(int64_t                    M,
                               int64_t                    N,
                               int64_t                    lda,
                               const hipblasLtHalf* const hCPU[],
                               const hipblasLtHalf* const hGPU[],
                               int64_t                    batch_count,
                               double                     abs_error)
{
    NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_HALF);
}

template <>
inline void near_check_general<hip_bfloat16, float>(int64_t                   M,
                                                    int64_t                   N,
                                                    int64_t                   lda,
                                                    const float* const        hCPU[],
                                                    const hip_bfloat16* const hGPU[],
                                                    int64_t                   batch_count,
                                                    double                    abs_error)
{
    NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_BF16);
}

template <>
inline void near_check_general(int64_t                        M,
                               int64_t                        N,
                               int64_t                        lda,
                               const hipblaslt_f8_fnuz* const hCPU[],
                               const hipblaslt_f8_fnuz* const hGPU[],
                               int64_t                        batch_count,
                               double                         abs_error)
{
    NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_FP8);
}

template <>
inline void near_check_general(int64_t                         M,
                               int64_t                         N,
                               int64_t                         lda,
                               const hipblaslt_bf8_fnuz* const hCPU[],
                               const hipblaslt_bf8_fnuz* const hGPU[],
                               int64_t                         batch_count,
                               double                          abs_error)
{
    NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_FP8);
}

template <>
inline void near_check_general(int64_t                       M,
                               int64_t                       N,
                               int64_t                       lda,
                               const hipblaslt_f8* const hCPU[],
                               const hipblaslt_f8* const hGPU[],
                               int64_t                       batch_count,
                               double                        abs_error)
{
    NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_FP8);
}

template <>
inline void near_check_general(int64_t                        M,
                               int64_t                        N,
                               int64_t                        lda,
                               const hipblaslt_bf8* const hCPU[],
                               const hipblaslt_bf8* const hGPU[],
                               int64_t                        batch_count,
                               double                         abs_error)
{
    NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_FP8);
}

inline void near_check_general(int64_t     M,
                               int64_t     N,
                               int64_t     lda,
                               int64_t     strideA,
                               void*       hCPU,
                               void*       hGPU,
                               int64_t     batch_count,
                               double      abs_error,
                               hipDataType type)
{
    switch(type)
    {
    case HIP_R_32F:
        near_check_general(M,
                           N,
                           lda,
                           strideA,
                           static_cast<float*>(hCPU),
                           static_cast<float*>(hGPU),
                           batch_count,
                           abs_error);
        break;
    case HIP_R_64F:
        near_check_general(M,
                           N,
                           lda,
                           strideA,
                           static_cast<double*>(hCPU),
                           static_cast<double*>(hGPU),
                           batch_count,
                           abs_error);
        break;
    case HIP_R_16F:
        near_check_general(M,
                           N,
                           lda,
                           strideA,
                           static_cast<hipblasLtHalf*>(hCPU),
                           static_cast<hipblasLtHalf*>(hGPU),
                           batch_count,
                           abs_error);
        break;
    case HIP_R_16BF:
        near_check_general(M,
                           N,
                           lda,
                           strideA,
                           static_cast<hip_bfloat16*>(hCPU),
                           static_cast<hip_bfloat16*>(hGPU),
                           batch_count,
                           abs_error);
        break;
    case HIP_R_8F_E4M3_FNUZ:
        near_check_general(M,
                           N,
                           lda,
                           strideA,
                           static_cast<hipblaslt_f8_fnuz*>(hCPU),
                           static_cast<hipblaslt_f8_fnuz*>(hGPU),
                           batch_count,
                           abs_error);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        near_check_general(M,
                           N,
                           lda,
                           strideA,
                           static_cast<hipblaslt_bf8_fnuz*>(hCPU),
                           static_cast<hipblaslt_bf8_fnuz*>(hGPU),
                           batch_count,
                           abs_error);
        break;
    case HIP_R_8F_E4M3:
        near_check_general(M,
                           N,
                           lda,
                           strideA,
                           static_cast<hipblaslt_f8*>(hCPU),
                           static_cast<hipblaslt_f8*>(hGPU),
                           batch_count,
                           abs_error);
        break;
    case HIP_R_8F_E5M2:
        near_check_general(M,
                           N,
                           lda,
                           strideA,
                           static_cast<hipblaslt_bf8*>(hCPU),
                           static_cast<hipblaslt_bf8*>(hGPU),
                           batch_count,
                           abs_error);
        break;
    case HIP_R_32I:
        near_check_general(M,
                           N,
                           lda,
                           strideA,
                           static_cast<int32_t*>(hCPU),
                           static_cast<int32_t*>(hGPU),
                           batch_count,
                           abs_error);
        break;
    case HIP_R_8I:
        near_check_general(M,
                           N,
                           lda,
                           strideA,
                           static_cast<hipblasLtInt8*>(hCPU),
                           static_cast<hipblasLtInt8*>(hGPU),
                           batch_count,
                           abs_error);
        break;
    default:
        hipblaslt_cerr << "Error type in near_check_general" << std::endl;
        break;
    }
}
