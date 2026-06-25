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
 *
 * ************************************************************************ */

#include "near.h"
#include "hipblas.h"
#include "host_vector.hpp"
#include "type_utils.h"
#include "utility.h"

/* ========================================Gtest Unit Check
 * ==================================================== */

/*! \brief Template: gtest unit compare two matrices float/double/complex */
// Do not put a wrapper over ASSERT_FLOAT_EQ, sincer assert exit the current function NOT the test
// case
// a wrapper will cause the loop keep going

#ifndef GOOGLE_TEST
#define NEAR_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, err, NEAR_ASSERT)
#define NEAR_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, err, NEAR_ASSERT)
#else

#define NEAR_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, err, NEAR_ASSERT)    \
    do                                                                               \
    {                                                                                \
        for(size_t k = 0; k < batch_count; k++)                                      \
            for(size_t j = 0; j < N; j++)                                            \
                for(size_t i = 0; i < M; i++)                                        \
                    if(hipblas_isnan(hCPU[i + j * lda + k * strideA]))               \
                    {                                                                \
                        ASSERT_TRUE(hipblas_isnan(hGPU[i + j * lda + k * strideA])); \
                    }                                                                \
                    else                                                             \
                    {                                                                \
                        NEAR_ASSERT(hCPU[i + j * lda + k * strideA],                 \
                                    hGPU[i + j * lda + k * strideA],                 \
                                    err);                                            \
                    }                                                                \
    } while(0)

#define NEAR_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, err, NEAR_ASSERT)            \
    do                                                                                \
    {                                                                                 \
        for(size_t k = 0; k < batch_count; k++)                                       \
            for(size_t j = 0; j < N; j++)                                             \
                for(size_t i = 0; i < M; i++)                                         \
                    if(hipblas_isnan(hCPU[k][i + j * lda]))                           \
                    {                                                                 \
                        ASSERT_TRUE(hipblas_isnan(hGPU[k][i + j * lda]));             \
                    }                                                                 \
                    else                                                              \
                    {                                                                 \
                        NEAR_ASSERT(hCPU[k][i + j * lda], hGPU[k][i + j * lda], err); \
                    }                                                                 \
    } while(0)

#endif

#define NEAR_ASSERT_HALF(a, b, err) ASSERT_NEAR(half_to_float(a), half_to_float(b), err)
#define NEAR_ASSERT_BF16(a, b, err) ASSERT_NEAR(bfloat16_to_float(a), bfloat16_to_float(b), err)
#define NEAR_ASSERT_FLOAT_BF16(a, b, err) ASSERT_NEAR((a), bfloat16_to_float(b), err)

#define NEAR_ASSERT_COMPLEX(a, b, err)          \
    do                                          \
    {                                           \
        auto ta = (a), tb = (b);                \
        ASSERT_NEAR(ta.real(), tb.real(), err); \
        ASSERT_NEAR(ta.imag(), tb.imag(), err); \
    } while(0)

template <>
void near_check_general(
    int64_t M, int64_t N, int64_t lda, int32_t* hCPU, int32_t* hGPU, double abs_error)
{
    NEAR_CHECK(M, N, 1, lda, 0, hCPU, hGPU, abs_error, ASSERT_NEAR);
}

template <>
void near_check_general(
    int64_t M, int64_t N, int64_t lda, float* hCPU, float* hGPU, double abs_error)
{
    NEAR_CHECK(M, N, 1, lda, 0, hCPU, hGPU, abs_error, ASSERT_NEAR);
}

template <>
void near_check_general(
    int64_t M, int64_t N, int64_t lda, double* hCPU, double* hGPU, double abs_error)
{
    NEAR_CHECK(M, N, 1, lda, 0, hCPU, hGPU, abs_error, ASSERT_NEAR);
}

template <>
void near_check_general(
    int64_t M, int64_t N, int64_t lda, hipblasHalf* hCPU, hipblasHalf* hGPU, double abs_error)
{
    NEAR_CHECK(M, N, 1, lda, 0, hCPU, hGPU, abs_error, NEAR_ASSERT_HALF);
}

template <>
void near_check_general(int64_t          M,
                        int64_t          N,
                        int64_t          lda,
                        hipblasBfloat16* hCPU,
                        hipblasBfloat16* hGPU,
                        double           abs_error)
{
    NEAR_CHECK(M, N, 1, lda, 0, hCPU, hGPU, abs_error, NEAR_ASSERT_BF16);
}

template <>
void near_check_general(int64_t              M,
                        int64_t              N,
                        int64_t              lda,
                        std::complex<float>* hCPU,
                        std::complex<float>* hGPU,
                        double               abs_error)
{
    abs_error *= sqrthalf;
    NEAR_CHECK(M, N, 1, lda, 0, hCPU, hGPU, abs_error, NEAR_ASSERT_COMPLEX);
}

template <>
void near_check_general(int64_t               M,
                        int64_t               N,
                        int64_t               lda,
                        std::complex<double>* hCPU,
                        std::complex<double>* hGPU,
                        double                abs_error)
{
    abs_error *= sqrthalf;
    NEAR_CHECK(M, N, 1, lda, 0, hCPU, hGPU, abs_error, NEAR_ASSERT_COMPLEX);
}

template <>
void near_check_general(int64_t       M,
                        int64_t       N,
                        int64_t       batch_count,
                        int64_t       lda,
                        hipblasStride strideA,
                        int32_t*      hCPU,
                        int32_t*      hGPU,
                        double        abs_error)
{
    NEAR_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, abs_error, ASSERT_NEAR);
}

template <>
void near_check_general(int64_t       M,
                        int64_t       N,
                        int64_t       batch_count,
                        int64_t       lda,
                        hipblasStride strideA,
                        float*        hCPU,
                        float*        hGPU,
                        double        abs_error)
{
    NEAR_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, abs_error, ASSERT_NEAR);
}

template <>
void near_check_general(int64_t       M,
                        int64_t       N,
                        int64_t       batch_count,
                        int64_t       lda,
                        hipblasStride strideA,
                        double*       hCPU,
                        double*       hGPU,
                        double        abs_error)
{
    NEAR_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, abs_error, ASSERT_NEAR);
}

template <>
void near_check_general(int64_t       M,
                        int64_t       N,
                        int64_t       batch_count,
                        int64_t       lda,
                        hipblasStride strideA,
                        hipblasHalf*  hCPU,
                        hipblasHalf*  hGPU,
                        double        abs_error)
{
    NEAR_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, abs_error, NEAR_ASSERT_HALF);
}

template <>
void near_check_general(int64_t          M,
                        int64_t          N,
                        int64_t          batch_count,
                        int64_t          lda,
                        hipblasStride    strideA,
                        hipblasBfloat16* hCPU,
                        hipblasBfloat16* hGPU,
                        double           abs_error)
{
    NEAR_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, abs_error, NEAR_ASSERT_BF16);
}

template <>
void near_check_general(int64_t              M,
                        int64_t              N,
                        int64_t              batch_count,
                        int64_t              lda,
                        hipblasStride        strideA,
                        std::complex<float>* hCPU,
                        std::complex<float>* hGPU,
                        double               abs_error)
{
    abs_error *= sqrthalf;
    NEAR_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, abs_error, NEAR_ASSERT_COMPLEX);
}

template <>
void near_check_general(int64_t               M,
                        int64_t               N,
                        int64_t               batch_count,
                        int64_t               lda,
                        hipblasStride         strideA,
                        std::complex<double>* hCPU,
                        std::complex<double>* hGPU,
                        double                abs_error)
{
    abs_error *= sqrthalf;
    NEAR_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, abs_error, NEAR_ASSERT_COMPLEX);
}

template <>
void near_check_general(int64_t                  M,
                        int64_t                  N,
                        int64_t                  batch_count,
                        int64_t                  lda,
                        host_vector<hipblasHalf> hCPU[],
                        host_vector<hipblasHalf> hGPU[],
                        double                   abs_error)
{
    NEAR_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, abs_error, NEAR_ASSERT_HALF);
}

template <>
void near_check_general(int64_t                      M,
                        int64_t                      N,
                        int64_t                      batch_count,
                        int64_t                      lda,
                        host_vector<hipblasBfloat16> hCPU[],
                        host_vector<hipblasBfloat16> hGPU[],
                        double                       abs_error)
{
    NEAR_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, abs_error, NEAR_ASSERT_BF16);
}

template <>
void near_check_general(int64_t              M,
                        int64_t              N,
                        int64_t              batch_count,
                        int64_t              lda,
                        host_vector<int32_t> hCPU[],
                        host_vector<int32_t> hGPU[],
                        double               abs_error)
{
    NEAR_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, abs_error, ASSERT_NEAR);
}

void near_check_general(int64_t            M,
                        int64_t            N,
                        int64_t            batch_count,
                        int64_t            lda,
                        host_vector<float> hCPU[],
                        host_vector<float> hGPU[],
                        double             abs_error)
{
    NEAR_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, abs_error, ASSERT_NEAR);
}

template <>
void near_check_general(int64_t             M,
                        int64_t             N,
                        int64_t             batch_count,
                        int64_t             lda,
                        host_vector<double> hCPU[],
                        host_vector<double> hGPU[],
                        double              abs_error)
{
    NEAR_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, abs_error, ASSERT_NEAR);
}

template <>
void near_check_general(int64_t                          M,
                        int64_t                          N,
                        int64_t                          batch_count,
                        int64_t                          lda,
                        host_vector<std::complex<float>> hCPU[],
                        host_vector<std::complex<float>> hGPU[],
                        double                           abs_error)
{
    abs_error *= sqrthalf;
    NEAR_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, abs_error, NEAR_ASSERT_COMPLEX);
}

template <>
void near_check_general(int64_t                           M,
                        int64_t                           N,
                        int64_t                           batch_count,
                        int64_t                           lda,
                        host_vector<std::complex<double>> hCPU[],
                        host_vector<std::complex<double>> hGPU[],
                        double                            abs_error)
{
    abs_error *= sqrthalf;
    NEAR_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, abs_error, NEAR_ASSERT_COMPLEX);
}

template <>
void near_check_general(int64_t      M,
                        int64_t      N,
                        int64_t      batch_count,
                        int64_t      lda,
                        hipblasHalf* hCPU[],
                        hipblasHalf* hGPU[],
                        double       abs_error)
{
    NEAR_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, abs_error, NEAR_ASSERT_HALF);
}

template <>
void near_check_general(int64_t          M,
                        int64_t          N,
                        int64_t          batch_count,
                        int64_t          lda,
                        hipblasBfloat16* hCPU[],
                        hipblasBfloat16* hGPU[],
                        double           abs_error)
{
    NEAR_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, abs_error, NEAR_ASSERT_BF16);
}

template <>
void near_check_general(int64_t  M,
                        int64_t  N,
                        int64_t  batch_count,
                        int64_t  lda,
                        int32_t* hCPU[],
                        int32_t* hGPU[],
                        double   abs_error)
{
    NEAR_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, abs_error, ASSERT_NEAR);
}

template <>
void near_check_general(int64_t M,
                        int64_t N,
                        int64_t batch_count,
                        int64_t lda,
                        float*  hCPU[],
                        float*  hGPU[],
                        double  abs_error)
{
    NEAR_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, abs_error, ASSERT_NEAR);
}

template <>
void near_check_general(int64_t M,
                        int64_t N,
                        int64_t batch_count,
                        int64_t lda,
                        double* hCPU[],
                        double* hGPU[],
                        double  abs_error)
{
    NEAR_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, abs_error, ASSERT_NEAR);
}

template <>
void near_check_general(int64_t              M,
                        int64_t              N,
                        int64_t              batch_count,
                        int64_t              lda,
                        std::complex<float>* hCPU[],
                        std::complex<float>* hGPU[],
                        double               abs_error)
{
    abs_error *= sqrthalf;
    NEAR_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, abs_error, NEAR_ASSERT_COMPLEX);
}

template <>
void near_check_general(int64_t               M,
                        int64_t               N,
                        int64_t               batch_count,
                        int64_t               lda,
                        std::complex<double>* hCPU[],
                        std::complex<double>* hGPU[],
                        double                abs_error)
{
    abs_error *= sqrthalf;
    NEAR_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, abs_error, NEAR_ASSERT_COMPLEX);
}

// mixed precision

template <typename T, typename T_hpa>
void near_check_mixed(int64_t                        M,
                      int64_t                        N,
                      int64_t                        lda,
                      const std::remove_cv_t<T_hpa>* hCPU,
                      const T*                       hGPU,
                      double                         abs_error)
{
    NEAR_CHECK(M, N, 1, lda, 0, hCPU, hGPU, abs_error, ASSERT_NEAR);
}

template <>
void near_check_mixed<hipblasHalf, hipblasHalf>(int64_t            M,
                                                int64_t            N,
                                                int64_t            lda,
                                                const hipblasHalf* hCPU,
                                                const hipblasHalf* hGPU,
                                                double             abs_error)
{
    NEAR_CHECK(M, N, 1, lda, 0, hCPU, hGPU, abs_error, NEAR_ASSERT_HALF);
}

template <>
void near_check_mixed<hipblasBfloat16, float>(int64_t                M,
                                              int64_t                N,
                                              int64_t                lda,
                                              const float*           hCPU,
                                              const hipblasBfloat16* hGPU,
                                              double                 abs_error)
{
    NEAR_CHECK(M, N, 1, lda, 0, hCPU, hGPU, abs_error, NEAR_ASSERT_FLOAT_BF16);
}

#define INST(Ta_, Tb_)                        \
    template void near_check_mixed<Ta_, Tb_>( \
        int64_t M, int64_t N, int64_t lda, const Ta_* hCPU, const Tb_* hGPU, double abs_error);

INST(float, float)
INST(double, double)
