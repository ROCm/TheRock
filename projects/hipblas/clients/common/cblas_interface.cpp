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
 * ************************************************************************/
#if !defined(WIN32) && defined(FLA_ENABLE_ILP64)
#include <lapacke.h>
#endif

#include "cblas_interface.h"
#include "hipblas.h"
#include "lapack_utilities.hpp"
#include "utility.h"
#include <cmath>
#include <memory>
#include <typeinfo>

/*!\file
 * \brief provide template functions interfaces to CBLAS C89 interfaces, it is only used for testing
 * not part of the GPU library
*/

#ifndef FLA_ENABLE_ILP64

#ifdef __cplusplus
extern "C" {
#endif

void spotrf_(char* uplo, int64_t* m, float* A, int64_t* lda, int64_t* info);
void dpotrf_(char* uplo, int64_t* m, double* A, int64_t* lda, int64_t* info);
void cpotrf_(char* uplo, int64_t* m, std::complex<float>* A, int64_t* lda, int64_t* info);
void zpotrf_(char* uplo, int64_t* m, std::complex<double>* A, int64_t* lda, int64_t* info);

void sgetrf_(int64_t* m, int64_t* n, float* A, int64_t* lda, int64_t* ipiv, int64_t* info);
void dgetrf_(int64_t* m, int64_t* n, double* A, int64_t* lda, int64_t* ipiv, int64_t* info);
void cgetrf_(
    int64_t* m, int64_t* n, std::complex<float>* A, int64_t* lda, int64_t* ipiv, int64_t* info);
void zgetrf_(
    int64_t* m, int64_t* n, std::complex<double>* A, int64_t* lda, int64_t* ipiv, int64_t* info);

void sgetrs_(char*    trans,
             int64_t* n,
             int64_t* nrhs,
             float*   A,
             int64_t* lda,
             int64_t* ipiv,
             float*   B,
             int64_t* ldb,
             int64_t* info);
void dgetrs_(char*    trans,
             int64_t* n,
             int64_t* nrhs,
             double*  A,
             int64_t* lda,
             int64_t* ipiv,
             double*  B,
             int64_t* ldb,
             int64_t* info);
void cgetrs_(char*                trans,
             int64_t*             n,
             int64_t*             nrhs,
             std::complex<float>* A,
             int64_t*             lda,
             int64_t*             ipiv,
             std::complex<float>* B,
             int64_t*             ldb,
             int64_t*             info);
void zgetrs_(char*                 trans,
             int64_t*              n,
             int64_t*              nrhs,
             std::complex<double>* A,
             int64_t*              lda,
             int64_t*              ipiv,
             std::complex<double>* B,
             int64_t*              ldb,
             int64_t*              info);

void sgetri_(
    int64_t* n, float* A, int64_t* lda, int64_t* ipiv, float* work, int64_t* lwork, int64_t* info);
void dgetri_(int64_t* n,
             double*  A,
             int64_t* lda,
             int64_t* ipiv,
             double*  work,
             int64_t* lwork,
             int64_t* info);
void cgetri_(int64_t*             n,
             std::complex<float>* A,
             int64_t*             lda,
             int64_t*             ipiv,
             std::complex<float>* work,
             int64_t*             lwork,
             int64_t*             info);
void zgetri_(int64_t*              n,
             std::complex<double>* A,
             int64_t*              lda,
             int64_t*              ipiv,
             std::complex<double>* work,
             int64_t*              lwork,
             int64_t*              info);

void sgeqrf_(int64_t* m,
             int64_t* n,
             float*   A,
             int64_t* lda,
             float*   tau,
             float*   work,
             int64_t* lwork,
             int64_t* info);
void dgeqrf_(int64_t* m,
             int64_t* n,
             double*  A,
             int64_t* lda,
             double*  tau,
             double*  work,
             int64_t* lwork,
             int64_t* info);
void cgeqrf_(int64_t*             m,
             int64_t*             n,
             std::complex<float>* A,
             int64_t*             lda,
             std::complex<float>* tau,
             std::complex<float>* work,
             int64_t*             lwork,
             int64_t*             info);
void zgeqrf_(int64_t*              m,
             int64_t*              n,
             std::complex<double>* A,
             int64_t*              lda,
             std::complex<double>* tau,
             std::complex<double>* work,
             int64_t*              lwork,
             int64_t*              info);

void sgels_(char*    trans,
            int64_t* m,
            int64_t* n,
            int64_t* nrhs,
            float*   A,
            int64_t* lda,
            float*   B,
            int64_t* ldb,
            float*   work,
            int64_t* lwork,
            int64_t* info);
void dgels_(char*    trans,
            int64_t* m,
            int64_t* n,
            int64_t* nrhs,
            double*  A,
            int64_t* lda,
            double*  B,
            int64_t* ldb,
            double*  work,
            int64_t* lwork,
            int64_t* info);
void cgels_(char*                trans,
            int64_t*             m,
            int64_t*             n,
            int64_t*             nrhs,
            std::complex<float>* A,
            int64_t*             lda,
            std::complex<float>* B,
            int64_t*             ldb,
            std::complex<float>* work,
            int64_t*             lwork,
            int64_t*             info);
void zgels_(char*                 trans,
            int64_t*              m,
            int64_t*              n,
            int64_t*              nrhs,
            std::complex<double>* A,
            int64_t*              lda,
            std::complex<double>* B,
            int64_t*              ldb,
            std::complex<double>* work,
            int64_t*              lwork,
            int64_t*              info);

/*
void strtri_(char* uplo, char* diag, int64_t* n, float* A, int64_t* lda, int64_t* info);
void dtrtri_(char* uplo, char* diag, int64_t* n, double* A, int64_t* lda, int64_t* info);
void ctrtri_(char* uplo, char* diag, int64_t* n, std::complex<float>* A, int64_t* lda, int64_t* info);
void ztrtri_(char* uplo, char* diag, int64_t* n, std::complex<double>* A, int64_t* lda, int64_t* info);

void cspr_(
    char* uplo, int64_t* n, std::complex<float>* alpha, std::complex<float>* x, int64_t* incx, std::complex<float>* A);

void zspr_(char*                 uplo,
           int64_t*                  n,
           std::complex<double>* alpha,
           std::complex<double>* x,
           int64_t*                  incx,
           std::complex<double>* A);

void csyr_(char*           uplo,
           int64_t*            n,
           std::complex<float>* alpha,
           std::complex<float>* x,
           int64_t*            incx,
           std::complex<float>* a,
           int64_t*            lda);
void zsyr_(char*                 uplo,
           int64_t*                  n,
           std::complex<double>* alpha,
           std::complex<double>* x,
           int64_t*                  incx,
           std::complex<double>* a,
           int64_t*                  lda);

void csymv_(char*           uplo,
            int64_t*            n,
            std::complex<float>* alpha,
            std::complex<float>* A,
            int64_t*            lda,
            std::complex<float>* x,
            int64_t*            incx,
            std::complex<float>* beta,
            std::complex<float>* y,
            int64_t*            incy);

void zsymv_(char*                 uplo,
            int64_t*                  n,
            std::complex<double>* alpha,
            std::complex<double>* A,
            int64_t*                  lda,
            std::complex<double>* x,
            int64_t*                  incx,
            std::complex<double>* beta,
            std::complex<double>* y,
            int64_t*                  incy);
*/

#ifdef __cplusplus
}
#endif

#endif

// helpers

template <typename T, typename U>
void cast_to_buffer(
    hipblasOperation_t transA, int64_t m, int64_t k, int64_t lda, const T* A_t, host_vector<U>& A_u)
{
    size_t colsA = (transA == HIPBLAS_OP_N ? k : m);
    size_t rowsA = (transA == HIPBLAS_OP_N ? m : k);

    size_t sizeA = colsA * lda;

    A_u.resize(sizeA);

    for(size_t i = 0; i < colsA; i++)
    {
        size_t   offset = i * lda;
        const T* src    = A_t + offset;
        U*       dst    = A_u + offset;
        for(size_t j = 0; j < rowsA; j++)
        {
            if constexpr(std::is_same_v<T, hipblasBfloat16> && std::is_same_v<U, float>)
                *dst++ = bfloat16_to_float(*src++);
            else
                *dst++ = static_cast<U>(*src++);
        }
    }
}

template <typename T, typename U>
void cast_from_buffer(int64_t m, int64_t n, int64_t ldc, const host_vector<T>& C_t, U* C_u)
{
    for(size_t i = 0; i < n; i++)
    {
        size_t offset = i * ldc;
        for(size_t j = 0; j < m; j++)
        {
            if constexpr(std::is_same_v<U, hipblasBfloat16> && std::is_same_v<T, float>)
                C_u[j + offset] = float_to_bfloat16(C_t[j + offset]);
            else
                C_u[j + offset] = static_cast<U>(C_t[j + offset]);
        }
    }
}

/*
 * ===========================================================================
 *    level 1 BLAS
 * ===========================================================================
 */

// axpy
template <>
void ref_axpy<hipblasBfloat16, hipblasBfloat16>(int64_t                n,
                                                const hipblasBfloat16  alpha,
                                                const hipblasBfloat16* x,
                                                int64_t                incx,
                                                hipblasBfloat16*       y,
                                                int64_t                incy)
{
    size_t             abs_incx = incx >= 0 ? incx : -incx;
    size_t             abs_incy = incy >= 0 ? incy : -incy;
    std::vector<float> x_float(n * abs_incx);
    std::vector<float> y_float(n * abs_incy);

    for(size_t i = 0; i < n; i++)
    {
        x_float[i * abs_incx] = bfloat16_to_float(x[i * abs_incx]);
        y_float[i * abs_incy] = bfloat16_to_float(y[i * abs_incy]);
    }

    cblas_saxpy(n, bfloat16_to_float(alpha), x_float.data(), incx, y_float.data(), incy);

    for(size_t i = 0; i < n; i++)
    {
        y[i * abs_incy] = float_to_bfloat16(y_float[i * abs_incy]);
    }
}

template <>
void ref_axpy<float, hipblasBfloat16>(int64_t                n,
                                      const float            alpha,
                                      const hipblasBfloat16* x,
                                      int64_t                incx,
                                      hipblasBfloat16*       y,
                                      int64_t                incy)
{
    size_t             abs_incx = incx >= 0 ? incx : -incx;
    size_t             abs_incy = incy >= 0 ? incy : -incy;
    std::vector<float> x_float(n * abs_incx);
    std::vector<float> y_float(n * abs_incy);

    for(size_t i = 0; i < n; i++)
    {
        x_float[i * abs_incx] = bfloat16_to_float(x[i * abs_incx]);
        y_float[i * abs_incy] = bfloat16_to_float(y[i * abs_incy]);
    }

    cblas_saxpy(n, alpha, x_float.data(), incx, y_float.data(), incy);

    for(size_t i = 0; i < n; i++)
    {
        y[i * abs_incy] = float_to_bfloat16(y_float[i * abs_incy]);
    }
}

template <>
void ref_axpy<hipblasHalf, hipblasHalf>(int64_t            n,
                                        const hipblasHalf  alpha,
                                        const hipblasHalf* x,
                                        int64_t            incx,
                                        hipblasHalf*       y,
                                        int64_t            incy)
{
    size_t             abs_incx = incx >= 0 ? incx : -incx;
    size_t             abs_incy = incy >= 0 ? incy : -incy;
    std::vector<float> x_float(n * abs_incx);
    std::vector<float> y_float(n * abs_incy);

    for(size_t i = 0; i < n; i++)
    {
        x_float[i * abs_incx] = half_to_float(x[i * abs_incx]);
        y_float[i * abs_incy] = half_to_float(y[i * abs_incy]);
    }

    cblas_saxpy(n, half_to_float(alpha), x_float.data(), incx, y_float.data(), incy);

    for(size_t i = 0; i < n; i++)
    {
        y[i * abs_incy] = float_to_half(y_float[i * abs_incy]);
    }
}

template <>
void ref_axpy<float, hipblasHalf>(
    int64_t n, const float alpha, const hipblasHalf* x, int64_t incx, hipblasHalf* y, int64_t incy)
{
    size_t             abs_incx = incx >= 0 ? incx : -incx;
    size_t             abs_incy = incy >= 0 ? incy : -incy;
    std::vector<float> x_float(n * abs_incx);
    std::vector<float> y_float(n * abs_incy);

    for(size_t i = 0; i < n; i++)
    {
        x_float[i * abs_incx] = half_to_float(x[i * abs_incx]);
        y_float[i * abs_incy] = half_to_float(y[i * abs_incy]);
    }

    cblas_saxpy(n, alpha, x_float.data(), incx, y_float.data(), incy);

    for(size_t i = 0; i < n; i++)
    {
        y[i * abs_incy] = float_to_half(y_float[i * abs_incy]);
    }
}

template <>
void ref_axpy<float, float>(
    int64_t n, const float alpha, const float* x, int64_t incx, float* y, int64_t incy)
{
    cblas_saxpy(n, alpha, x, incx, y, incy);
}

template <>
void ref_axpy<double, double>(
    int64_t n, const double alpha, const double* x, int64_t incx, double* y, int64_t incy)
{
    cblas_daxpy(n, alpha, x, incx, y, incy);
}

template <>
void ref_axpy<std::complex<float>, std::complex<float>>(int64_t                    n,
                                                        const std::complex<float>  alpha,
                                                        const std::complex<float>* x,
                                                        int64_t                    incx,
                                                        std::complex<float>*       y,
                                                        int64_t                    incy)
{
    cblas_caxpy(n, &alpha, x, incx, y, incy);
}

template <>
void ref_axpy<std::complex<double>, std::complex<double>>(int64_t                     n,
                                                          const std::complex<double>  alpha,
                                                          const std::complex<double>* x,
                                                          int64_t                     incx,
                                                          std::complex<double>*       y,
                                                          int64_t                     incy)
{
    cblas_zaxpy(n, &alpha, x, incx, y, incy);
}

// asum
template <>
void ref_asum<float>(int64_t n, const float* x, int64_t incx, float* result)
{
    if(n <= 0 || incx <= 0)
        return;

    float sum = 0;

    // using partial sums to reduce rounding errors for 64-bit n
    int64_t block_size = 1024 * 512;
    int64_t blocks     = (n - 1) / block_size + 1;
    for(int64_t b = 0; b < blocks; b++)
    {
        float partial_sum = 0;
        for(int64_t i = 0; i < block_size; i++)
        {
            int64_t idx = i + b * block_size;
            if(idx < n)
                partial_sum += std::abs(x[idx * incx]);
        }
        sum += partial_sum;
    }
    *result = sum;
}

// scal
template <>
void ref_scal<hipblasHalf>(int64_t n, const hipblasHalf alpha, hipblasHalf* x, int64_t incx)
{
    if(n <= 0 || incx <= 0)
        return;

    std::vector<float> x_float(n * incx);

    for(size_t i = 0; i < n; i++)
        x_float[i * incx] = half_to_float(x[i * incx]);

    cblas_sscal(n, half_to_float(alpha), x_float.data(), incx);

    for(size_t i = 0; i < n; i++)
        x[i * incx] = float_to_half(x_float[i * incx]);
}

template <>
void ref_scal<hipblasBfloat16>(int64_t               n,
                               const hipblasBfloat16 alpha,
                               hipblasBfloat16*      x,
                               int64_t               incx)
{
    if(n <= 0 || incx <= 0)
        return;

    std::vector<float> x_float(n * incx);

    for(size_t i = 0; i < n; i++)
        x_float[i * incx] = bfloat16_to_float(x[i * incx]);

    cblas_sscal(n, bfloat16_to_float(alpha), x_float.data(), incx);

    for(size_t i = 0; i < n; i++)
        x[i * incx] = float_to_bfloat16(x_float[i * incx]);
}

template <>
void ref_scal<hipblasHalf, float>(int64_t n, const float alpha, hipblasHalf* x, int64_t incx)
{
    if(n <= 0 || incx <= 0)
        return;

    std::vector<float> x_float(n * incx);

    for(size_t i = 0; i < n; i++)
        x_float[i * incx] = half_to_float(x[i * incx]);

    cblas_sscal(n, alpha, x_float.data(), incx);

    for(size_t i = 0; i < n; i++)
        x[i * incx] = float_to_half(x_float[i * incx]);
}

template <>
void ref_scal<hipblasBfloat16, float>(int64_t          n,
                                      const float      alpha,
                                      hipblasBfloat16* x,
                                      int64_t          incx)
{
    if(n <= 0 || incx <= 0)
        return;

    std::vector<float> x_float(n * incx);

    for(size_t i = 0; i < n; i++)
        x_float[i * incx] = bfloat16_to_float(x[i * incx]);

    cblas_sscal(n, alpha, x_float.data(), incx);

    for(size_t i = 0; i < n; i++)
        x[i * incx] = float_to_bfloat16(x_float[i * incx]);
}

template <>
void ref_scal<float>(int64_t n, const float alpha, float* x, int64_t incx)
{
    cblas_sscal(n, alpha, x, incx);
}

template <>
void ref_scal<double>(int64_t n, const double alpha, double* x, int64_t incx)
{
    cblas_dscal(n, alpha, x, incx);
}

template <>
void ref_scal<std::complex<float>>(int64_t                   n,
                                   const std::complex<float> alpha,
                                   std::complex<float>*      x,
                                   int64_t                   incx)
{
    cblas_cscal(n, &alpha, x, incx);
}

template <>
void ref_scal<std::complex<float>, float>(int64_t              n,
                                          const float          alpha,
                                          std::complex<float>* x,
                                          int64_t              incx)
{
    cblas_csscal(n, alpha, x, incx);
}

template <>
void ref_scal<std::complex<double>>(int64_t                    n,
                                    const std::complex<double> alpha,
                                    std::complex<double>*      x,
                                    int64_t                    incx)
{
    cblas_zscal(n, &alpha, x, incx);
}

template <>
void ref_scal<std::complex<double>, double>(int64_t               n,
                                            const double          alpha,
                                            std::complex<double>* x,
                                            int64_t               incx)
{
    cblas_zdscal(n, alpha, x, incx);
}

// copy
template <>
void ref_copy<float>(int64_t n, float* x, int64_t incx, float* y, int64_t incy)
{
    cblas_scopy(n, x, incx, y, incy);
}

template <>
void ref_copy<double>(int64_t n, double* x, int64_t incx, double* y, int64_t incy)
{
    cblas_dcopy(n, x, incx, y, incy);
}

template <>
void ref_copy<std::complex<float>>(
    int64_t n, std::complex<float>* x, int64_t incx, std::complex<float>* y, int64_t incy)
{
    cblas_ccopy(n, x, incx, y, incy);
}

template <>
void ref_copy<std::complex<double>>(
    int64_t n, std::complex<double>* x, int64_t incx, std::complex<double>* y, int64_t incy)
{
    cblas_zcopy(n, x, incx, y, incy);
}

// swap
template <>
void ref_swap<float>(int64_t n, float* x, int64_t incx, float* y, int64_t incy)
{
    cblas_sswap(n, x, incx, y, incy);
}

template <>
void ref_swap<double>(int64_t n, double* x, int64_t incx, double* y, int64_t incy)
{
    cblas_dswap(n, x, incx, y, incy);
}

template <>
void ref_swap<std::complex<float>>(
    int64_t n, std::complex<float>* x, int64_t incx, std::complex<float>* y, int64_t incy)
{
    cblas_cswap(n, x, incx, y, incy);
}

template <>
void ref_swap<std::complex<double>>(
    int64_t n, std::complex<double>* x, int64_t incx, std::complex<double>* y, int64_t incy)
{
    cblas_zswap(n, x, incx, y, incy);
}

// dot
template <>
void ref_dot<hipblasHalf>(int64_t            n,
                          const hipblasHalf* x,
                          int64_t            incx,
                          const hipblasHalf* y,
                          int64_t            incy,
                          hipblasHalf*       result)
{
    size_t             abs_incx = incx >= 0 ? incx : -incx;
    size_t             abs_incy = incy >= 0 ? incy : -incy;
    std::vector<float> x_float(n * abs_incx);
    std::vector<float> y_float(n * abs_incy);

    for(size_t i = 0; i < n; i++)
    {
        x_float[i * abs_incx] = half_to_float(x[i * abs_incx]);
        y_float[i * abs_incy] = half_to_float(y[i * abs_incy]);
    }
    *result = float_to_half(cblas_sdot(n, x_float.data(), incx, y_float.data(), incy));
}

template <>
void ref_dot<hipblasBfloat16>(int64_t                n,
                              const hipblasBfloat16* x,
                              int64_t                incx,
                              const hipblasBfloat16* y,
                              int64_t                incy,
                              hipblasBfloat16*       result)
{
    size_t             abs_incx = incx >= 0 ? incx : -incx;
    size_t             abs_incy = incy >= 0 ? incy : -incy;
    std::vector<float> x_float(n * abs_incx);
    std::vector<float> y_float(n * abs_incy);

    for(size_t i = 0; i < n; i++)
    {
        x_float[i * abs_incx] = bfloat16_to_float(x[i * abs_incx]);
        y_float[i * abs_incy] = bfloat16_to_float(y[i * abs_incy]);
    }
    *result = float_to_bfloat16(cblas_sdot(n, x_float.data(), incx, y_float.data(), incy));
}

template <>
void ref_dot<float>(
    int64_t n, const float* x, int64_t incx, const float* y, int64_t incy, float* result)
{
    *result = cblas_sdot(n, x, incx, y, incy);
}

template <>
void ref_dot<double>(
    int64_t n, const double* x, int64_t incx, const double* y, int64_t incy, double* result)
{
    *result = cblas_ddot(n, x, incx, y, incy);
}

template <>
void ref_dot<std::complex<float>>(int64_t                    n,
                                  const std::complex<float>* x,
                                  int64_t                    incx,
                                  const std::complex<float>* y,
                                  int64_t                    incy,
                                  std::complex<float>*       result)
{
    cblas_cdotu_sub(n, x, incx, y, incy, result);
}

template <>
void ref_dot<std::complex<double>>(int64_t                     n,
                                   const std::complex<double>* x,
                                   int64_t                     incx,
                                   const std::complex<double>* y,
                                   int64_t                     incy,
                                   std::complex<double>*       result)
{
    cblas_zdotu_sub(n, x, incx, y, incy, result);
}

template <>
void ref_dotc<hipblasHalf>(int64_t            n,
                           const hipblasHalf* x,
                           int64_t            incx,
                           const hipblasHalf* y,
                           int64_t            incy,
                           hipblasHalf*       result)
{
    // Not complex - call regular dot.
    ref_dot(n, x, incx, y, incy, result);
}

template <>
void ref_dotc<hipblasBfloat16>(int64_t                n,
                               const hipblasBfloat16* x,
                               int64_t                incx,
                               const hipblasBfloat16* y,
                               int64_t                incy,
                               hipblasBfloat16*       result)
{
    // Not complex - call regular dot.
    ref_dot(n, x, incx, y, incy, result);
}

template <>
void ref_dotc<float>(
    int64_t n, const float* x, int64_t incx, const float* y, int64_t incy, float* result)
{
    // Not complex - call regular dot.
    ref_dot(n, x, incx, y, incy, result);
}

template <>
void ref_dotc<double>(
    int64_t n, const double* x, int64_t incx, const double* y, int64_t incy, double* result)
{
    // Not complex - call regular dot.
    ref_dot(n, x, incx, y, incy, result);
}

template <>
void ref_dotc<std::complex<float>>(int64_t                    n,
                                   const std::complex<float>* x,
                                   int64_t                    incx,
                                   const std::complex<float>* y,
                                   int64_t                    incy,
                                   std::complex<float>*       result)
{
    cblas_cdotc_sub(n, x, incx, y, incy, result);
}

template <>
void ref_dotc<std::complex<double>>(int64_t                     n,
                                    const std::complex<double>* x,
                                    int64_t                     incx,
                                    const std::complex<double>* y,
                                    int64_t                     incy,
                                    std::complex<double>*       result)
{
    cblas_zdotc_sub(n, x, incx, y, incy, result);
}

// nrm2
template <>
void ref_nrm2<hipblasHalf, hipblasHalf>(int64_t            n,
                                        const hipblasHalf* x,
                                        int64_t            incx,
                                        hipblasHalf*       result)
{
    if(n <= 0 || incx <= 0)
        return;

    std::vector<float> x_float(n * incx);

    for(size_t i = 0; i < n; i++)
        x_float[i * incx] = half_to_float(x[i * incx]);

    *result = float_to_half(cblas_snrm2(n, x_float.data(), incx));
}

template <>
void ref_nrm2<hipblasBfloat16, hipblasBfloat16>(int64_t                n,
                                                const hipblasBfloat16* x,
                                                int64_t                incx,
                                                hipblasBfloat16*       result)
{
    if(n <= 0 || incx <= 0)
        return;

    std::vector<float> x_float(n * incx);

    for(size_t i = 0; i < n; i++)
        x_float[i * incx] = bfloat16_to_float(x[i * incx]);

    *result = float_to_bfloat16(cblas_snrm2(n, x_float.data(), incx));
}

template <>
void ref_nrm2<float, float>(int64_t n, const float* x, int64_t incx, float* result)
{
    *result = cblas_snrm2(n, x, incx);
}

template <>
void ref_nrm2<double, double>(int64_t n, const double* x, int64_t incx, double* result)
{
    *result = cblas_dnrm2(n, x, incx);
}

template <>
void ref_nrm2<std::complex<float>, float>(int64_t                    n,
                                          const std::complex<float>* x,
                                          int64_t                    incx,
                                          float*                     result)
{
    *result = cblas_scnrm2(n, x, incx);
}

template <>
void ref_nrm2<std::complex<double>, double>(int64_t                     n,
                                            const std::complex<double>* x,
                                            int64_t                     incx,
                                            double*                     result)
{
    *result = cblas_dznrm2(n, x, incx);
}

///////////////////
// rot functions //
///////////////////
// LAPACK fortran library functionality
extern "C" {
void crot_(const int64_t*             n,
           std::complex<float>*       cx,
           const int64_t*             incx,
           std::complex<float>*       cy,
           const int64_t*             incy,
           const float*               c,
           const std::complex<float>* s);
void csrot_(const int64_t*       n,
            std::complex<float>* cx,
            const int64_t*       incx,
            std::complex<float>* cy,
            const int64_t*       incy,
            const float*         c,
            const float*         s);
void zrot_(const int64_t*              n,
           std::complex<double>*       cx,
           const int64_t*              incx,
           std::complex<double>*       cy,
           const int64_t*              incy,
           const double*               c,
           const std::complex<double>* s);
void zdrot_(const int64_t*        n,
            std::complex<double>* cx,
            const int64_t*        incx,
            std::complex<double>* cy,
            const int64_t*        incy,
            const double*         c,
            const double*         s);

void crotg_(std::complex<float>* a, std::complex<float>* b, float* c, std::complex<float>* s);
void zrotg_(std::complex<double>* a, std::complex<double>* b, double* c, std::complex<double>* s);
}

// rot
template <>
void ref_rot<hipblasHalf>(int64_t      n,
                          hipblasHalf* x,
                          int64_t      incx,
                          hipblasHalf* y,
                          int64_t      incy,
                          hipblasHalf  c,
                          hipblasHalf  s)
{
    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t abs_incy = incy >= 0 ? incy : -incy;
    size_t size_x   = n * abs_incx;
    size_t size_y   = n * abs_incy;
    if(!size_x)
        size_x = 1;
    if(!size_y)
        size_y = 1;
    std::vector<float> x_float(size_x);
    std::vector<float> y_float(size_y);

    for(size_t i = 0; i < n; i++)
    {
        x_float[i * abs_incx] = half_to_float(x[i * abs_incx]);
        y_float[i * abs_incy] = half_to_float(y[i * abs_incy]);
    }

    const float c_float = half_to_float(c);
    const float s_float = half_to_float(s);

    cblas_srot(n, x_float.data(), incx, y_float.data(), incy, c_float, s_float);

    for(size_t i = 0; i < n; i++)
    {
        x[i * abs_incx] = float_to_half(x_float[i * abs_incx]);
        y[i * abs_incy] = float_to_half(y_float[i * abs_incy]);
    }
}

template <>
void ref_rot<hipblasBfloat16>(int64_t          n,
                              hipblasBfloat16* x,
                              int64_t          incx,
                              hipblasBfloat16* y,
                              int64_t          incy,
                              hipblasBfloat16  c,
                              hipblasBfloat16  s)
{
    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t abs_incy = incy >= 0 ? incy : -incy;
    size_t size_x   = n * abs_incx;
    size_t size_y   = n * abs_incy;
    if(!size_x)
        size_x = 1;
    if(!size_y)
        size_y = 1;
    std::vector<float> x_float(size_x);
    std::vector<float> y_float(size_y);

    for(size_t i = 0; i < n; i++)
    {
        x_float[i * abs_incx] = bfloat16_to_float(x[i * abs_incx]);
        y_float[i * abs_incy] = bfloat16_to_float(y[i * abs_incy]);
    }

    const float c_float = bfloat16_to_float(c);
    const float s_float = bfloat16_to_float(s);

    cblas_srot(n, x_float.data(), incx, y_float.data(), incy, c_float, s_float);

    for(size_t i = 0; i < n; i++)
    {
        x[i * abs_incx] = float_to_bfloat16(x_float[i * abs_incx]);
        y[i * abs_incy] = float_to_bfloat16(y_float[i * abs_incy]);
    }
}

template <>
void ref_rot<float>(int64_t n, float* x, int64_t incx, float* y, int64_t incy, float c, float s)
{
    cblas_srot(n, x, incx, y, incy, c, s);
}

template <>
void ref_rot<double>(
    int64_t n, double* x, int64_t incx, double* y, int64_t incy, double c, double s)
{
    cblas_drot(n, x, incx, y, incy, c, s);
}

template <>
void ref_rot<std::complex<float>>(int64_t              n,
                                  std::complex<float>* x,
                                  int64_t              incx,
                                  std::complex<float>* y,
                                  int64_t              incy,
                                  std::complex<float>  c,
                                  std::complex<float>  s)
{
    float c_real = std::real(c);
    lapack_xrot(n, x, incx, y, incy, c_real, s);
}

template <>
void ref_rot<std::complex<float>, float>(int64_t              n,
                                         std::complex<float>* x,
                                         int64_t              incx,
                                         std::complex<float>* y,
                                         int64_t              incy,
                                         float                c,
                                         std::complex<float>  s)
{
    lapack_xrot(n, x, incx, y, incy, c, s);
}

template <>
void ref_rot<std::complex<float>, float, float>(int64_t              n,
                                                std::complex<float>* x,
                                                int64_t              incx,
                                                std::complex<float>* y,
                                                int64_t              incy,
                                                float                c,
                                                float                s)
{
    lapack_xrot(n, x, incx, y, incy, c, s);
}

template <>
void ref_rot<std::complex<double>>(int64_t               n,
                                   std::complex<double>* x,
                                   int64_t               incx,
                                   std::complex<double>* y,
                                   int64_t               incy,
                                   std::complex<double>  c,
                                   std::complex<double>  s)
{
    double c_real = std::real(c);
    lapack_xrot(n, x, incx, y, incy, c_real, s);
}

template <>
void ref_rot<std::complex<double>, double>(int64_t               n,
                                           std::complex<double>* x,
                                           int64_t               incx,
                                           std::complex<double>* y,
                                           int64_t               incy,
                                           double                c,
                                           std::complex<double>  s)
{
    lapack_xrot(n, x, incx, y, incy, c, s);
}

template <>
void ref_rot<std::complex<double>, double, double>(int64_t               n,
                                                   std::complex<double>* x,
                                                   int64_t               incx,
                                                   std::complex<double>* y,
                                                   int64_t               incy,
                                                   double                c,
                                                   double                s)
{
    lapack_xrot(n, x, incx, y, incy, c, s);
}

// rotg
template <>
void ref_rotg<float>(float* a, float* b, float* c, float* s)
{
    cblas_srotg(a, b, c, s);
}

template <>
void ref_rotg<double>(double* a, double* b, double* c, double* s)
{
    cblas_drotg(a, b, c, s);
}

template <>
void ref_rotg<std::complex<float>, float>(std::complex<float>* a,
                                          std::complex<float>* b,
                                          float*               c,
                                          std::complex<float>* s)
{
    lapack_xrotg(*a, *b, *c, *s);
}

template <>
void ref_rotg<std::complex<double>, double>(std::complex<double>* a,
                                            std::complex<double>* b,
                                            double*               c,
                                            std::complex<double>* s)
{
    lapack_xrotg(*a, *b, *c, *s);
}

// asum
/*
template <>
void ref_asum<float, float>(int64_t n, const float* x, int64_t incx, float* result)
{
    *result = ref_sasum(n, x, incx);
}

template <>
void ref_asum<double, double>(int64_t n, const double* x, int64_t incx, double* result)
{
    *result = ref_dasum(n, x, incx);
}

template <>
void ref_asum<std::complex<float>, float>(int64_t               n,
                                       const std::complex<float>* x,
                                       int64_t               incx,
                                       float*                result)
{
    *result = ref_scasum(n, x, incx);
}

template <>
void ref_asum<std::complex<double>, double>(int64_t                     n,
                                              const std::complex<double>* x,
                                              int64_t                     incx,
                                              double*                     result)
{
    *result = ref_dzasum(n, x, incx);
}
*/

// amax

/* local versions of amax and amin for minimum index in case of ties.
See hipblas_iamax_imin_fef.hpp

template <>
void ref_iamax<float>(int64_t n, const float* x, int64_t incx, int64_t* result)
{
    *result = (int64_t)cblas_isamax(n, x, incx);
}

template <>
void ref_iamax<double>(int64_t n, const double* x, int64_t incx, int64_t* result)
{
    *result = (int64_t)cblas_idamax(n, x, incx);
}

template <>
void ref_iamax<std::complex<float>>(int64_t n, const std::complex<float>* x, int64_t incx, int64_t* result)
{
    *result = (int64_t)cblas_icamax(n, x, incx);
}

template <>
void ref_iamax<std::complex<double>>(int64_t                     n,
                                     const std::complex<double>* x,
                                     int64_t                     incx,
                                     int64_t*                    result)
{
    *result = (int64_t)cblas_izamax(n, x, incx);
}

// amin
// amin is not implemented in cblas, make local version

template <typename T>
double hipblas_magnitude(T val)
{
    return val < 0 ? -val : val;
}

template <>
double hipblas_magnitude(std::complex<float> val)
{
    return std::abs(val.real()) + std::abs(val.imag());
}

template <>
double hipblas_magnitude(std::complex<double> val)
{
    return std::abs(val.real()) + std::abs(val.imag());
}

template <typename T>
int64_t ref_iamin_helper(int64_t N, const T* X, int64_t incx)
{
    int64_t minpos = -1;
    if(N > 0 && incx > 0)
    {
        auto min = hipblas_magnitude(X[0]);
        minpos   = 0;
        for(size_t i = 1; i < N; ++i)
        {
            auto a = hipblas_magnitude(X[i * incx]);
            if(a < min)
            {
                min    = a;
                minpos = i;
            }
        }
    }
    return minpos;
}

template <>
void ref_iamin<float>(int64_t n, const float* x, int64_t incx, int64_t* result)
{
    *result = (int64_t)ref_iamin_helper(n, x, incx);
}

template <>
void ref_iamin<double>(int64_t n, const double* x, int64_t incx, int64_t* result)
{
    *result = (int64_t)ref_iamin_helper(n, x, incx);
}

template <>
void ref_iamin<std::complex<float>>(int64_t n, const std::complex<float>* x, int64_t incx, int64_t* result)
{
    *result = (int64_t)ref_iamin_helper(n, x, incx);
}

template <>
void ref_iamin<std::complex<double>>(int64_t                     n,
                                     const std::complex<double>* x,
                                     int64_t                     incx,
                                     int64_t*                    result)
{
    *result = (int64_t)ref_iamin_helper(n, x, incx);
}
*/

/*
 * ===========================================================================
 *    level 2 BLAS
 * ===========================================================================
 */

// gbmv
template <>
void ref_gbmv<float>(hipblasOperation_t transA,
                     int64_t            m,
                     int64_t            n,
                     int64_t            kl,
                     int64_t            ku,
                     float              alpha,
                     float*             A,
                     int64_t            lda,
                     float*             x,
                     int64_t            incx,
                     float              beta,
                     float*             y,
                     int64_t            incy)
{
    cblas_sgbmv(CblasColMajor,
                (CBLAS_TRANSPOSE)transA,
                m,
                n,
                kl,
                ku,
                alpha,
                A,
                lda,
                x,
                incx,
                beta,
                y,
                incy);
}

template <>
void ref_gbmv<double>(hipblasOperation_t transA,
                      int64_t            m,
                      int64_t            n,
                      int64_t            kl,
                      int64_t            ku,
                      double             alpha,
                      double*            A,
                      int64_t            lda,
                      double*            x,
                      int64_t            incx,
                      double             beta,
                      double*            y,
                      int64_t            incy)
{
    cblas_dgbmv(CblasColMajor,
                (CBLAS_TRANSPOSE)transA,
                m,
                n,
                kl,
                ku,
                alpha,
                A,
                lda,
                x,
                incx,
                beta,
                y,
                incy);
}

template <>
void ref_gbmv<std::complex<float>>(hipblasOperation_t   transA,
                                   int64_t              m,
                                   int64_t              n,
                                   int64_t              kl,
                                   int64_t              ku,
                                   std::complex<float>  alpha,
                                   std::complex<float>* A,
                                   int64_t              lda,
                                   std::complex<float>* x,
                                   int64_t              incx,
                                   std::complex<float>  beta,
                                   std::complex<float>* y,
                                   int64_t              incy)
{
    cblas_cgbmv(CblasColMajor,
                (CBLAS_TRANSPOSE)transA,
                m,
                n,
                kl,
                ku,
                &alpha,
                A,
                lda,
                x,
                incx,
                &beta,
                y,
                incy);
}

template <>
void ref_gbmv<std::complex<double>>(hipblasOperation_t    transA,
                                    int64_t               m,
                                    int64_t               n,
                                    int64_t               kl,
                                    int64_t               ku,
                                    std::complex<double>  alpha,
                                    std::complex<double>* A,
                                    int64_t               lda,
                                    std::complex<double>* x,
                                    int64_t               incx,
                                    std::complex<double>  beta,
                                    std::complex<double>* y,
                                    int64_t               incy)
{
    cblas_zgbmv(CblasColMajor,
                (CBLAS_TRANSPOSE)transA,
                m,
                n,
                kl,
                ku,
                &alpha,
                A,
                lda,
                x,
                incx,
                &beta,
                y,
                incy);
}

// gemv
template <>
void ref_gemv<float>(hipblasOperation_t transA,
                     int64_t            m,
                     int64_t            n,
                     float              alpha,
                     float*             A,
                     int64_t            lda,
                     float*             x,
                     int64_t            incx,
                     float              beta,
                     float*             y,
                     int64_t            incy)
{
    cblas_sgemv(
        CblasColMajor, (CBLAS_TRANSPOSE)transA, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
void ref_gemv<double>(hipblasOperation_t transA,
                      int64_t            m,
                      int64_t            n,
                      double             alpha,
                      double*            A,
                      int64_t            lda,
                      double*            x,
                      int64_t            incx,
                      double             beta,
                      double*            y,
                      int64_t            incy)
{
    cblas_dgemv(
        CblasColMajor, (CBLAS_TRANSPOSE)transA, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
void ref_gemv<std::complex<float>>(hipblasOperation_t   transA,
                                   int64_t              m,
                                   int64_t              n,
                                   std::complex<float>  alpha,
                                   std::complex<float>* A,
                                   int64_t              lda,
                                   std::complex<float>* x,
                                   int64_t              incx,
                                   std::complex<float>  beta,
                                   std::complex<float>* y,
                                   int64_t              incy)
{
    cblas_cgemv(
        CblasColMajor, (CBLAS_TRANSPOSE)transA, m, n, &alpha, A, lda, x, incx, &beta, y, incy);
}

template <>
void ref_gemv<std::complex<double>>(hipblasOperation_t    transA,
                                    int64_t               m,
                                    int64_t               n,
                                    std::complex<double>  alpha,
                                    std::complex<double>* A,
                                    int64_t               lda,
                                    std::complex<double>* x,
                                    int64_t               incx,
                                    std::complex<double>  beta,
                                    std::complex<double>* y,
                                    int64_t               incy)
{
    cblas_zgemv(
        CblasColMajor, (CBLAS_TRANSPOSE)transA, m, n, &alpha, A, lda, x, incx, &beta, y, incy);
}

// ger
template <>
void ref_ger<float, false>(int64_t m,
                           int64_t n,
                           float   alpha,
                           float*  x,
                           int64_t incx,
                           float*  y,
                           int64_t incy,
                           float*  A,
                           int64_t lda)
{
    cblas_sger(CblasColMajor, m, n, alpha, x, incx, y, incy, A, lda);
}

template <>
void ref_ger<double, false>(int64_t m,
                            int64_t n,
                            double  alpha,
                            double* x,
                            int64_t incx,
                            double* y,
                            int64_t incy,
                            double* A,
                            int64_t lda)
{
    cblas_dger(CblasColMajor, m, n, alpha, x, incx, y, incy, A, lda);
}

template <>
void ref_ger<std::complex<float>, false>(int64_t              m,
                                         int64_t              n,
                                         std::complex<float>  alpha,
                                         std::complex<float>* x,
                                         int64_t              incx,
                                         std::complex<float>* y,
                                         int64_t              incy,
                                         std::complex<float>* A,
                                         int64_t              lda)
{
    cblas_cgeru(CblasColMajor, m, n, &alpha, x, incx, y, incy, A, lda);
}

template <>
void ref_ger<std::complex<float>, true>(int64_t              m,
                                        int64_t              n,
                                        std::complex<float>  alpha,
                                        std::complex<float>* x,
                                        int64_t              incx,
                                        std::complex<float>* y,
                                        int64_t              incy,
                                        std::complex<float>* A,
                                        int64_t              lda)
{
    cblas_cgerc(CblasColMajor, m, n, &alpha, x, incx, y, incy, A, lda);
}

template <>
void ref_ger<std::complex<double>, false>(int64_t               m,
                                          int64_t               n,
                                          std::complex<double>  alpha,
                                          std::complex<double>* x,
                                          int64_t               incx,
                                          std::complex<double>* y,
                                          int64_t               incy,
                                          std::complex<double>* A,
                                          int64_t               lda)
{
    cblas_zgeru(CblasColMajor, m, n, &alpha, x, incx, y, incy, A, lda);
}

template <>
void ref_ger<std::complex<double>, true>(int64_t               m,
                                         int64_t               n,
                                         std::complex<double>  alpha,
                                         std::complex<double>* x,
                                         int64_t               incx,
                                         std::complex<double>* y,
                                         int64_t               incy,
                                         std::complex<double>* A,
                                         int64_t               lda)
{
    cblas_zgerc(CblasColMajor, m, n, &alpha, x, incx, y, incy, A, lda);
}

// hbmv
template <>
void ref_hbmv<std::complex<float>>(hipblasFillMode_t    uplo,
                                   int64_t              n,
                                   int64_t              k,
                                   std::complex<float>  alpha,
                                   std::complex<float>* A,
                                   int64_t              lda,
                                   std::complex<float>* x,
                                   int64_t              incx,
                                   std::complex<float>  beta,
                                   std::complex<float>* y,
                                   int64_t              incy)
{
    cblas_chbmv(CblasColMajor, (CBLAS_UPLO)uplo, n, k, &alpha, A, lda, x, incx, &beta, y, incy);
}

template <>
void ref_hbmv<std::complex<double>>(hipblasFillMode_t     uplo,
                                    int64_t               n,
                                    int64_t               k,
                                    std::complex<double>  alpha,
                                    std::complex<double>* A,
                                    int64_t               lda,
                                    std::complex<double>* x,
                                    int64_t               incx,
                                    std::complex<double>  beta,
                                    std::complex<double>* y,
                                    int64_t               incy)
{
    cblas_zhbmv(CblasColMajor, (CBLAS_UPLO)uplo, n, k, &alpha, A, lda, x, incx, &beta, y, incy);
}

// hemv
template <>
void ref_hemv<std::complex<float>>(hipblasFillMode_t    uplo,
                                   int64_t              n,
                                   std::complex<float>  alpha,
                                   std::complex<float>* A,
                                   int64_t              lda,
                                   std::complex<float>* x,
                                   int64_t              incx,
                                   std::complex<float>  beta,
                                   std::complex<float>* y,
                                   int64_t              incy)
{
    cblas_chemv(CblasColMajor, (CBLAS_UPLO)uplo, n, &alpha, A, lda, x, incx, &beta, y, incy);
}

template <>
void ref_hemv<std::complex<double>>(hipblasFillMode_t     uplo,
                                    int64_t               n,
                                    std::complex<double>  alpha,
                                    std::complex<double>* A,
                                    int64_t               lda,
                                    std::complex<double>* x,
                                    int64_t               incx,
                                    std::complex<double>  beta,
                                    std::complex<double>* y,
                                    int64_t               incy)
{
    cblas_zhemv(CblasColMajor, (CBLAS_UPLO)uplo, n, &alpha, A, lda, x, incx, &beta, y, incy);
}

// her
template <>
void ref_her<std::complex<float>, float>(hipblasFillMode_t    uplo,
                                         int64_t              n,
                                         float                alpha,
                                         std::complex<float>* x,
                                         int64_t              incx,
                                         std::complex<float>* A,
                                         int64_t              lda)
{
    cblas_cher(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, x, incx, A, lda);
}

template <>
void ref_her<std::complex<double>, double>(hipblasFillMode_t     uplo,
                                           int64_t               n,
                                           double                alpha,
                                           std::complex<double>* x,
                                           int64_t               incx,
                                           std::complex<double>* A,
                                           int64_t               lda)
{
    cblas_zher(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, x, incx, A, lda);
}

// her2
template <>
void ref_her2<std::complex<float>>(hipblasFillMode_t    uplo,
                                   int64_t              n,
                                   std::complex<float>  alpha,
                                   std::complex<float>* x,
                                   int64_t              incx,
                                   std::complex<float>* y,
                                   int64_t              incy,
                                   std::complex<float>* A,
                                   int64_t              lda)
{
    cblas_cher2(CblasColMajor, (CBLAS_UPLO)uplo, n, &alpha, x, incx, y, incy, A, lda);
}

template <>
void ref_her2<std::complex<double>>(hipblasFillMode_t     uplo,
                                    int64_t               n,
                                    std::complex<double>  alpha,
                                    std::complex<double>* x,
                                    int64_t               incx,
                                    std::complex<double>* y,
                                    int64_t               incy,
                                    std::complex<double>* A,
                                    int64_t               lda)
{
    cblas_zher2(CblasColMajor, (CBLAS_UPLO)uplo, n, &alpha, x, incx, y, incy, A, lda);
}

// hpmv
template <>
void ref_hpmv<std::complex<float>>(hipblasFillMode_t    uplo,
                                   int64_t              n,
                                   std::complex<float>  alpha,
                                   std::complex<float>* AP,
                                   std::complex<float>* x,
                                   int64_t              incx,
                                   std::complex<float>  beta,
                                   std::complex<float>* y,
                                   int64_t              incy)
{
    cblas_chpmv(CblasColMajor, (CBLAS_UPLO)uplo, n, &alpha, AP, x, incx, &beta, y, incy);
}

template <>
void ref_hpmv<std::complex<double>>(hipblasFillMode_t     uplo,
                                    int64_t               n,
                                    std::complex<double>  alpha,
                                    std::complex<double>* AP,
                                    std::complex<double>* x,
                                    int64_t               incx,
                                    std::complex<double>  beta,
                                    std::complex<double>* y,
                                    int64_t               incy)
{
    cblas_zhpmv(CblasColMajor, (CBLAS_UPLO)uplo, n, &alpha, AP, x, incx, &beta, y, incy);
}

// hpr
template <>
void ref_hpr(hipblasFillMode_t    uplo,
             int64_t              n,
             float                alpha,
             std::complex<float>* x,
             int64_t              incx,
             std::complex<float>* AP)
{
    cblas_chpr(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, x, incx, AP);
}

template <>
void ref_hpr(hipblasFillMode_t     uplo,
             int64_t               n,
             double                alpha,
             std::complex<double>* x,
             int64_t               incx,
             std::complex<double>* AP)
{
    cblas_zhpr(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, x, incx, AP);
}

// hpr2
template <>
void ref_hpr2(hipblasFillMode_t    uplo,
              int64_t              n,
              std::complex<float>  alpha,
              std::complex<float>* x,
              int64_t              incx,
              std::complex<float>* y,
              int64_t              incy,
              std::complex<float>* AP)
{
    cblas_chpr2(CblasColMajor, (CBLAS_UPLO)uplo, n, &alpha, x, incx, y, incy, AP);
}

template <>
void ref_hpr2(hipblasFillMode_t     uplo,
              int64_t               n,
              std::complex<double>  alpha,
              std::complex<double>* x,
              int64_t               incx,
              std::complex<double>* y,
              int64_t               incy,
              std::complex<double>* AP)
{
    cblas_zhpr2(CblasColMajor, (CBLAS_UPLO)uplo, n, &alpha, x, incx, y, incy, AP);
}

// sbmv
template <>
void ref_sbmv(hipblasFillMode_t uplo,
              int64_t           n,
              int64_t           k,
              float             alpha,
              float*            A,
              int64_t           lda,
              float*            x,
              int64_t           incx,
              float             beta,
              float*            y,
              int64_t           incy)
{
    cblas_ssbmv(CblasColMajor, (CBLAS_UPLO)uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
void ref_sbmv(hipblasFillMode_t uplo,
              int64_t           n,
              int64_t           k,
              double            alpha,
              double*           A,
              int64_t           lda,
              double*           x,
              int64_t           incx,
              double            beta,
              double*           y,
              int64_t           incy)
{
    cblas_dsbmv(CblasColMajor, (CBLAS_UPLO)uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
}

// spmv
template <>
void ref_spmv(hipblasFillMode_t uplo,
              int64_t           n,
              float             alpha,
              float*            AP,
              float*            x,
              int64_t           incx,
              float             beta,
              float*            y,
              int64_t           incy)
{
    cblas_sspmv(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, AP, x, incx, beta, y, incy);
}

template <>
void ref_spmv(hipblasFillMode_t uplo,
              int64_t           n,
              double            alpha,
              double*           AP,
              double*           x,
              int64_t           incx,
              double            beta,
              double*           y,
              int64_t           incy)
{
    cblas_dspmv(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, AP, x, incx, beta, y, incy);
}

// spr
template <>
void ref_spr(hipblasFillMode_t uplo, int64_t n, float alpha, float* x, int64_t incx, float* AP)
{
    cblas_sspr(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, x, incx, AP);
}

template <>
void ref_spr(hipblasFillMode_t uplo, int64_t n, double alpha, double* x, int64_t incx, double* AP)
{
    cblas_dspr(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, x, incx, AP);
}

template <>
void ref_spr(hipblasFillMode_t    uplo,
             int64_t              n,
             std::complex<float>  alpha,
             std::complex<float>* x,
             int64_t              incx,
             std::complex<float>* AP)
{
    lapack_xspr(uplo, n, alpha, x, incx, AP);
}

template <>
void ref_spr(hipblasFillMode_t     uplo,
             int64_t               n,
             std::complex<double>  alpha,
             std::complex<double>* x,
             int64_t               incx,
             std::complex<double>* AP)
{
    lapack_xspr(uplo, n, alpha, x, incx, AP);
}

// spr2
template <>
void ref_spr2(hipblasFillMode_t uplo,
              int64_t           n,
              float             alpha,
              float*            x,
              int64_t           incx,
              float*            y,
              int64_t           incy,
              float*            AP)
{
    cblas_sspr2(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, x, incx, y, incy, AP);
}

template <>
void ref_spr2(hipblasFillMode_t uplo,
              int64_t           n,
              double            alpha,
              double*           x,
              int64_t           incx,
              double*           y,
              int64_t           incy,
              double*           AP)
{
    cblas_dspr2(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, x, incx, y, incy, AP);
}

// symv
template <>
void ref_symv(hipblasFillMode_t uplo,
              int64_t           n,
              float             alpha,
              float*            A,
              int64_t           lda,
              float*            x,
              int64_t           incx,
              float             beta,
              float*            y,
              int64_t           incy)
{
    cblas_ssymv(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
void ref_symv(hipblasFillMode_t uplo,
              int64_t           n,
              double            alpha,
              double*           A,
              int64_t           lda,
              double*           x,
              int64_t           incx,
              double            beta,
              double*           y,
              int64_t           incy)
{
    cblas_dsymv(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
void ref_symv(hipblasFillMode_t    uplo,
              int64_t              n,
              std::complex<float>  alpha,
              std::complex<float>* A,
              int64_t              lda,
              std::complex<float>* x,
              int64_t              incx,
              std::complex<float>  beta,
              std::complex<float>* y,
              int64_t              incy)
{
    lapack_xsymv(uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
void ref_symv(hipblasFillMode_t     uplo,
              int64_t               n,
              std::complex<double>  alpha,
              std::complex<double>* A,
              int64_t               lda,
              std::complex<double>* x,
              int64_t               incx,
              std::complex<double>  beta,
              std::complex<double>* y,
              int64_t               incy)
{
    lapack_xsymv(uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

// syr
template <>
void ref_syr<float>(
    hipblasFillMode_t uplo, int64_t n, float alpha, float* x, int64_t incx, float* A, int64_t lda)
{
    cblas_ssyr(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, x, incx, A, lda);
}

template <>
void ref_syr<double>(hipblasFillMode_t uplo,
                     int64_t           n,
                     double            alpha,
                     double*           x,
                     int64_t           incx,
                     double*           A,
                     int64_t           lda)
{
    cblas_dsyr(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, x, incx, A, lda);
}

template <>
void ref_syr(hipblasFillMode_t    uplo,
             int64_t              n,
             std::complex<float>  alpha,
             std::complex<float>* xa,
             int64_t              incx,
             std::complex<float>* A,
             int64_t              lda)
{
    lapack_xsyr(uplo, n, alpha, xa, incx, A, lda);
}

template <>
void ref_syr(hipblasFillMode_t     uplo,
             int64_t               n,
             std::complex<double>  alpha,
             std::complex<double>* xa,
             int64_t               incx,
             std::complex<double>* A,
             int64_t               lda)
{
    lapack_xsyr(uplo, n, alpha, xa, incx, A, lda);
}

// syr2

template <>
void ref_syr2(hipblasFillMode_t uplo,
              int64_t           n,
              float             alpha,
              float*            x,
              int64_t           incx,
              float*            y,
              int64_t           incy,
              float*            A,
              int64_t           lda)
{
    cblas_ssyr2(CblasColMajor, CBLAS_UPLO(uplo), n, alpha, x, incx, y, incy, A, lda);
}

template <>
void ref_syr2(hipblasFillMode_t uplo,
              int64_t           n,
              double            alpha,
              double*           x,
              int64_t           incx,
              double*           y,
              int64_t           incy,
              double*           A,
              int64_t           lda)
{
    cblas_dsyr2(CblasColMajor, CBLAS_UPLO(uplo), n, alpha, x, incx, y, incy, A, lda);
}

template <>
void ref_syr2(hipblasFillMode_t    uplo,
              int64_t              n,
              std::complex<float>  alpha,
              std::complex<float>* x,
              int64_t              incx,
              std::complex<float>* y,
              int64_t              incy,
              std::complex<float>* A,
              int64_t              lda)
{
    lapack_xsyr2(uplo, n, alpha, x, incx, y, incy, A, lda);
}

template <>
void ref_syr2(hipblasFillMode_t     uplo,
              int64_t               n,
              std::complex<double>  alpha,
              std::complex<double>* x,
              int64_t               incx,
              std::complex<double>* y,
              int64_t               incy,
              std::complex<double>* A,
              int64_t               lda)
{
    lapack_xsyr2(uplo, n, alpha, x, incx, y, incy, A, lda);
}

// tbmv
template <>
void ref_tbmv<float>(hipblasFillMode_t  uplo,
                     hipblasOperation_t transA,
                     hipblasDiagType_t  diag,
                     int64_t            m,
                     int64_t            k,
                     const float*       A,
                     int64_t            lda,
                     float*             x,
                     int64_t            incx)
{
    cblas_stbmv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                k,
                A,
                lda,
                x,
                incx);
}

template <>
void ref_tbmv<double>(hipblasFillMode_t  uplo,
                      hipblasOperation_t transA,
                      hipblasDiagType_t  diag,
                      int64_t            m,
                      int64_t            k,
                      const double*      A,
                      int64_t            lda,
                      double*            x,
                      int64_t            incx)
{
    cblas_dtbmv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                k,
                A,
                lda,
                x,
                incx);
}

template <>
void ref_tbmv<std::complex<float>>(hipblasFillMode_t          uplo,
                                   hipblasOperation_t         transA,
                                   hipblasDiagType_t          diag,
                                   int64_t                    m,
                                   int64_t                    k,
                                   const std::complex<float>* A,
                                   int64_t                    lda,
                                   std::complex<float>*       x,
                                   int64_t                    incx)
{
    cblas_ctbmv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                k,
                A,
                lda,
                x,
                incx);
}

template <>
void ref_tbmv<std::complex<double>>(hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int64_t                     m,
                                    int64_t                     k,
                                    const std::complex<double>* A,
                                    int64_t                     lda,
                                    std::complex<double>*       x,
                                    int64_t                     incx)
{
    cblas_ztbmv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                k,
                A,
                lda,
                x,
                incx);
}

// tbsv
template <>
void ref_tbsv<float>(hipblasFillMode_t  uplo,
                     hipblasOperation_t transA,
                     hipblasDiagType_t  diag,
                     int64_t            m,
                     int64_t            k,
                     const float*       A,
                     int64_t            lda,
                     float*             x,
                     int64_t            incx)
{
    cblas_stbsv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                k,
                A,
                lda,
                x,
                incx);
}

template <>
void ref_tbsv<double>(hipblasFillMode_t  uplo,
                      hipblasOperation_t transA,
                      hipblasDiagType_t  diag,
                      int64_t            m,
                      int64_t            k,
                      const double*      A,
                      int64_t            lda,
                      double*            x,
                      int64_t            incx)
{
    cblas_dtbsv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                k,
                A,
                lda,
                x,
                incx);
}

template <>
void ref_tbsv<std::complex<float>>(hipblasFillMode_t          uplo,
                                   hipblasOperation_t         transA,
                                   hipblasDiagType_t          diag,
                                   int64_t                    m,
                                   int64_t                    k,
                                   const std::complex<float>* A,
                                   int64_t                    lda,
                                   std::complex<float>*       x,
                                   int64_t                    incx)
{
    cblas_ctbsv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                k,
                A,
                lda,
                x,
                incx);
}

template <>
void ref_tbsv<std::complex<double>>(hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int64_t                     m,
                                    int64_t                     k,
                                    const std::complex<double>* A,
                                    int64_t                     lda,
                                    std::complex<double>*       x,
                                    int64_t                     incx)
{
    cblas_ztbsv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                k,
                A,
                lda,
                x,
                incx);
}

// tpmv
template <>
void ref_tpmv(hipblasFillMode_t  uplo,
              hipblasOperation_t transA,
              hipblasDiagType_t  diag,
              int64_t            m,
              const float*       A,
              float*             x,
              int64_t            incx)
{
    cblas_stpmv(
        CblasColMajor, CBLAS_UPLO(uplo), CBLAS_TRANSPOSE(transA), CBLAS_DIAG(diag), m, A, x, incx);
}

template <>
void ref_tpmv(hipblasFillMode_t  uplo,
              hipblasOperation_t transA,
              hipblasDiagType_t  diag,
              int64_t            m,
              const double*      A,
              double*            x,
              int64_t            incx)
{
    cblas_dtpmv(
        CblasColMajor, CBLAS_UPLO(uplo), CBLAS_TRANSPOSE(transA), CBLAS_DIAG(diag), m, A, x, incx);
}

template <>
void ref_tpmv(hipblasFillMode_t          uplo,
              hipblasOperation_t         transA,
              hipblasDiagType_t          diag,
              int64_t                    m,
              const std::complex<float>* A,
              std::complex<float>*       x,
              int64_t                    incx)
{
    cblas_ctpmv(
        CblasColMajor, CBLAS_UPLO(uplo), CBLAS_TRANSPOSE(transA), CBLAS_DIAG(diag), m, A, x, incx);
}

template <>
void ref_tpmv(hipblasFillMode_t           uplo,
              hipblasOperation_t          transA,
              hipblasDiagType_t           diag,
              int64_t                     m,
              const std::complex<double>* A,
              std::complex<double>*       x,
              int64_t                     incx)
{
    cblas_ztpmv(
        CblasColMajor, CBLAS_UPLO(uplo), CBLAS_TRANSPOSE(transA), CBLAS_DIAG(diag), m, A, x, incx);
}

// tpsv
template <>
void ref_tpsv(hipblasFillMode_t  uplo,
              hipblasOperation_t transA,
              hipblasDiagType_t  diag,
              int64_t            n,
              const float*       AP,
              float*             x,
              int64_t            incx)
{
    cblas_stpsv(
        CblasColMajor, CBLAS_UPLO(uplo), CBLAS_TRANSPOSE(transA), CBLAS_DIAG(diag), n, AP, x, incx);
}

template <>
void ref_tpsv(hipblasFillMode_t  uplo,
              hipblasOperation_t transA,
              hipblasDiagType_t  diag,
              int64_t            n,
              const double*      AP,
              double*            x,
              int64_t            incx)
{
    cblas_dtpsv(
        CblasColMajor, CBLAS_UPLO(uplo), CBLAS_TRANSPOSE(transA), CBLAS_DIAG(diag), n, AP, x, incx);
}

template <>
void ref_tpsv(hipblasFillMode_t          uplo,
              hipblasOperation_t         transA,
              hipblasDiagType_t          diag,
              int64_t                    n,
              const std::complex<float>* AP,
              std::complex<float>*       x,
              int64_t                    incx)
{
    cblas_ctpsv(
        CblasColMajor, CBLAS_UPLO(uplo), CBLAS_TRANSPOSE(transA), CBLAS_DIAG(diag), n, AP, x, incx);
}

template <>
void ref_tpsv(hipblasFillMode_t           uplo,
              hipblasOperation_t          transA,
              hipblasDiagType_t           diag,
              int64_t                     n,
              const std::complex<double>* AP,
              std::complex<double>*       x,
              int64_t                     incx)
{
    cblas_ztpsv(
        CblasColMajor, CBLAS_UPLO(uplo), CBLAS_TRANSPOSE(transA), CBLAS_DIAG(diag), n, AP, x, incx);
}

// trmv
template <>
void ref_trmv<float>(hipblasFillMode_t  uplo,
                     hipblasOperation_t transA,
                     hipblasDiagType_t  diag,
                     int64_t            m,
                     const float*       A,
                     int64_t            lda,
                     float*             x,
                     int64_t            incx)
{
    cblas_strmv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                A,
                lda,
                x,
                incx);
}

template <>
void ref_trmv<double>(hipblasFillMode_t  uplo,
                      hipblasOperation_t transA,
                      hipblasDiagType_t  diag,
                      int64_t            m,
                      const double*      A,
                      int64_t            lda,
                      double*            x,
                      int64_t            incx)
{
    cblas_dtrmv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                A,
                lda,
                x,
                incx);
}

template <>
void ref_trmv<std::complex<float>>(hipblasFillMode_t          uplo,
                                   hipblasOperation_t         transA,
                                   hipblasDiagType_t          diag,
                                   int64_t                    m,
                                   const std::complex<float>* A,
                                   int64_t                    lda,
                                   std::complex<float>*       x,
                                   int64_t                    incx)
{
    cblas_ctrmv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                A,
                lda,
                x,
                incx);
}

template <>
void ref_trmv<std::complex<double>>(hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int64_t                     m,
                                    const std::complex<double>* A,
                                    int64_t                     lda,
                                    std::complex<double>*       x,
                                    int64_t                     incx)
{
    cblas_ztrmv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                A,
                lda,
                x,
                incx);
}

// trsv
template <>
void ref_trsv<float>(hipblasHandle_t    handle,
                     hipblasFillMode_t  uplo,
                     hipblasOperation_t transA,
                     hipblasDiagType_t  diag,
                     int64_t            m,
                     const float*       A,
                     int64_t            lda,
                     float*             x,
                     int64_t            incx)
{
    cblas_strsv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                A,
                lda,
                x,
                incx);
}

template <>
void ref_trsv<double>(hipblasHandle_t    handle,
                      hipblasFillMode_t  uplo,
                      hipblasOperation_t transA,
                      hipblasDiagType_t  diag,
                      int64_t            m,
                      const double*      A,
                      int64_t            lda,
                      double*            x,
                      int64_t            incx)
{
    cblas_dtrsv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                A,
                lda,
                x,
                incx);
}

template <>
void ref_trsv<std::complex<float>>(hipblasHandle_t            handle,
                                   hipblasFillMode_t          uplo,
                                   hipblasOperation_t         transA,
                                   hipblasDiagType_t          diag,
                                   int64_t                    m,
                                   const std::complex<float>* A,
                                   int64_t                    lda,
                                   std::complex<float>*       x,
                                   int64_t                    incx)
{
    cblas_ctrsv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                A,
                lda,
                x,
                incx);
}

template <>
void ref_trsv<std::complex<double>>(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int64_t                     m,
                                    const std::complex<double>* A,
                                    int64_t                     lda,
                                    std::complex<double>*       x,
                                    int64_t                     incx)
{
    cblas_ztrsv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                A,
                lda,
                x,
                incx);
}

/*
 * ===========================================================================
 *    level 3 BLAS
 * ===========================================================================
 */

template <typename T>
void ref_geam_helper(hipblasOperation_t transA,
                     hipblasOperation_t transB,
                     int64_t            M,
                     int64_t            N,
                     T                  alpha,
                     T*                 A,
                     int64_t            lda,
                     T                  beta,
                     T*                 B,
                     int64_t            ldb,
                     T*                 C,
                     int64_t            ldc)
{
    int64_t inc1_A = transA == HIPBLAS_OP_N ? 1 : lda;
    int64_t inc2_A = transA == HIPBLAS_OP_N ? lda : 1;
    int64_t inc1_B = transB == HIPBLAS_OP_N ? 1 : ldb;
    int64_t inc2_B = transB == HIPBLAS_OP_N ? ldb : 1;

    for(int64_t i = 0; i < M; i++)
    {
        for(int64_t j = 0; j < N; j++)
        {
            T a_val = A[i * inc1_A + j * inc2_A];
            T b_val = B[i * inc1_B + j * inc2_B];
            if constexpr(is_complex<T>)
            {
                if(transA == HIPBLAS_OP_C)
                    a_val = std::conj(a_val);
                if(transB == HIPBLAS_OP_C)
                    b_val = std::conj(b_val);
            }
            C[i + j * ldc] = alpha * a_val + beta * b_val;
        }
    }
}

template <typename T>
void ref_dgmm_helper(hipblasSideMode_t side,
                     int64_t           M,
                     int64_t           N,
                     const T*          A,
                     int64_t           lda,
                     const T*          x,
                     int64_t           incx,
                     T*                C,
                     int64_t           ldc)
{
    ptrdiff_t shift_x = incx < 0 ? -ptrdiff_t(incx) * (N - 1) : 0;
    for(size_t i1 = 0; i1 < M; i1++)
    {
        for(size_t i2 = 0; i2 < N; i2++)
        {
            if(HIPBLAS_SIDE_RIGHT == side)
            {
                C[i1 + i2 * ldc] = A[i1 + i2 * lda] * x[shift_x + i2 * incx];
            }
            else
            {
                C[i1 + i2 * ldc] = A[i1 + i2 * lda] * x[shift_x + i1 * incx];
            }
        }
    }
}

// dgmm
template <>
void ref_dgmm(hipblasSideMode_t side,
              int64_t           M,
              int64_t           N,
              const float*      A,
              int64_t           lda,
              const float*      x,
              int64_t           incx,
              float*            C,
              int64_t           ldc)
{
    ref_dgmm_helper(side, M, N, A, lda, x, incx, C, ldc);
}

template <>
void ref_dgmm(hipblasSideMode_t side,
              int64_t           M,
              int64_t           N,
              const double*     A,
              int64_t           lda,
              const double*     x,
              int64_t           incx,
              double*           C,
              int64_t           ldc)
{
    ref_dgmm_helper(side, M, N, A, lda, x, incx, C, ldc);
}

template <>
void ref_dgmm(hipblasSideMode_t          side,
              int64_t                    M,
              int64_t                    N,
              const std::complex<float>* A,
              int64_t                    lda,
              const std::complex<float>* x,
              int64_t                    incx,
              std::complex<float>*       C,
              int64_t                    ldc)
{
    ref_dgmm_helper(side, M, N, A, lda, x, incx, C, ldc);
}

template <>
void ref_dgmm(hipblasSideMode_t           side,
              int64_t                     M,
              int64_t                     N,
              const std::complex<double>* A,
              int64_t                     lda,
              const std::complex<double>* x,
              int64_t                     incx,
              std::complex<double>*       C,
              int64_t                     ldc)
{
    ref_dgmm_helper(side, M, N, A, lda, x, incx, C, ldc);
}

// geam
template <>
void ref_geam(hipblasOperation_t transa,
              hipblasOperation_t transb,
              int64_t            m,
              int64_t            n,
              float*             alpha,
              float*             A,
              int64_t            lda,
              float*             beta,
              float*             B,
              int64_t            ldb,
              float*             C,
              int64_t            ldc)
{
    return ref_geam_helper(transa, transb, m, n, *alpha, A, lda, *beta, B, ldb, C, ldc);
}

template <>
void ref_geam(hipblasOperation_t transa,
              hipblasOperation_t transb,
              int64_t            m,
              int64_t            n,
              double*            alpha,
              double*            A,
              int64_t            lda,
              double*            beta,
              double*            B,
              int64_t            ldb,
              double*            C,
              int64_t            ldc)
{
    return ref_geam_helper(transa, transb, m, n, *alpha, A, lda, *beta, B, ldb, C, ldc);
}

template <>
void ref_geam(hipblasOperation_t   transa,
              hipblasOperation_t   transb,
              int64_t              m,
              int64_t              n,
              std::complex<float>* alpha,
              std::complex<float>* A,
              int64_t              lda,
              std::complex<float>* beta,
              std::complex<float>* B,
              int64_t              ldb,
              std::complex<float>* C,
              int64_t              ldc)
{
    return ref_geam_helper(transa, transb, m, n, *alpha, A, lda, *beta, B, ldb, C, ldc);
}

template <>
void ref_geam(hipblasOperation_t    transa,
              hipblasOperation_t    transb,
              int64_t               m,
              int64_t               n,
              std::complex<double>* alpha,
              std::complex<double>* A,
              int64_t               lda,
              std::complex<double>* beta,
              std::complex<double>* B,
              int64_t               ldb,
              std::complex<double>* C,
              int64_t               ldc)
{
    return ref_geam_helper(transa, transb, m, n, *alpha, A, lda, *beta, B, ldb, C, ldc);
}

// gemm
template <>
void ref_gemm<hipblasHalf>(hipblasOperation_t transA,
                           hipblasOperation_t transB,
                           int64_t            m,
                           int64_t            n,
                           int64_t            k,
                           hipblasHalf        alpha,
                           hipblasHalf*       A,
                           int64_t            lda,
                           hipblasHalf*       B,
                           int64_t            ldb,
                           hipblasHalf        beta,
                           hipblasHalf*       C,
                           int64_t            ldc)
{
    // cblas does not support hipblasHalf, so convert to higher precision float
    // This will give more precise result which is acceptable for testing
    float alpha_float = half_to_float(alpha);
    float beta_float  = half_to_float(beta);

    size_t sizeA = transA == HIPBLAS_OP_N ? size_t(k) * lda : size_t(m) * lda;
    size_t sizeB = transB == HIPBLAS_OP_N ? size_t(n) * ldb : size_t(k) * ldb;
    size_t sizeC = size_t(n) * ldc;

    std::unique_ptr<float[]> A_float(new float[sizeA]());
    std::unique_ptr<float[]> B_float(new float[sizeB]());
    std::unique_ptr<float[]> C_float(new float[sizeC]());

    for(size_t i = 0; i < sizeA; i++)
    {
        A_float[i] = half_to_float(A[i]);
    }
    for(size_t i = 0; i < sizeB; i++)
    {
        B_float[i] = half_to_float(B[i]);
    }
    for(size_t i = 0; i < sizeC; i++)
    {
        C_float[i] = half_to_float(C[i]);
    }

    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: rocblas =%d, cblas=%d\n", transA, (CBLAS_TRANSPOSE)transA );
    cblas_sgemm(CblasColMajor,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_TRANSPOSE)transB,
                m,
                n,
                k,
                alpha_float,
                const_cast<const float*>(A_float.get()),
                lda,
                const_cast<const float*>(B_float.get()),
                ldb,
                beta_float,
                static_cast<float*>(C_float.get()),
                ldc);

    for(size_t i = 0; i < sizeC; i++)
    {
        C[i] = float_to_half(C_float[i]);
    }
}

template <>
void ref_gemm<hipblasHalf, hipblasHalf, float>(hipblasOperation_t transA,
                                               hipblasOperation_t transB,
                                               int64_t            m,
                                               int64_t            n,
                                               int64_t            k,
                                               float              alpha,
                                               hipblasHalf*       A,
                                               int64_t            lda,
                                               hipblasHalf*       B,
                                               int64_t            ldb,
                                               float              beta,
                                               hipblasHalf*       C,
                                               int64_t            ldc)
{
    // cblas does not support hipblasHalf, so convert to higher precision float
    // This will give more precise result which is acceptable for testing

    size_t sizeA = transA == HIPBLAS_OP_N ? size_t(k) * lda : size_t(m) * lda;
    size_t sizeB = transB == HIPBLAS_OP_N ? size_t(n) * ldb : size_t(k) * ldb;
    size_t sizeC = n * ldc;

    std::unique_ptr<float[]> A_float(new float[sizeA]());
    std::unique_ptr<float[]> B_float(new float[sizeB]());
    std::unique_ptr<float[]> C_float(new float[sizeC]());

    for(size_t i = 0; i < sizeA; i++)
    {
        A_float[i] = half_to_float(A[i]);
    }
    for(size_t i = 0; i < sizeB; i++)
    {
        B_float[i] = half_to_float(B[i]);
    }
    for(size_t i = 0; i < sizeC; i++)
    {
        C_float[i] = half_to_float(C[i]);
    }

    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: rocblas =%d, cblas=%d\n", transA, (CBLAS_TRANSPOSE)transA );
    cblas_sgemm(CblasColMajor,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_TRANSPOSE)transB,
                m,
                n,
                k,
                alpha,
                const_cast<const float*>(A_float.get()),
                lda,
                const_cast<const float*>(B_float.get()),
                ldb,
                beta,
                static_cast<float*>(C_float.get()),
                ldc);

    for(size_t i = 0; i < sizeC; i++)
    {
        C[i] = float_to_half(C_float[i]);
    }
}

template <>
void ref_gemm<hipblasHalf, float, float>(hipblasOperation_t transA,
                                         hipblasOperation_t transB,
                                         int64_t            m,
                                         int64_t            n,
                                         int64_t            k,
                                         float              alpha,
                                         hipblasHalf*       A,
                                         int64_t            lda,
                                         hipblasHalf*       B,
                                         int64_t            ldb,
                                         float              beta,
                                         float*             C,
                                         int64_t            ldc)
{
    // cblas does not support hipblasHalf, so convert to higher precision float
    // This will give more precise result which is acceptable for testing

    size_t sizeA = transA == HIPBLAS_OP_N ? size_t(k) * lda : size_t(m) * lda;
    size_t sizeB = transB == HIPBLAS_OP_N ? size_t(n) * ldb : size_t(k) * ldb;

    std::unique_ptr<float[]> A_float(new float[sizeA]());
    std::unique_ptr<float[]> B_float(new float[sizeB]());

    for(size_t i = 0; i < sizeA; i++)
    {
        A_float[i] = half_to_float(A[i]);
    }
    for(size_t i = 0; i < sizeB; i++)
    {
        B_float[i] = half_to_float(B[i]);
    }

    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: rocblas =%d, cblas=%d\n", transA, (CBLAS_TRANSPOSE)transA );
    cblas_sgemm(CblasColMajor,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_TRANSPOSE)transB,
                m,
                n,
                k,
                alpha,
                const_cast<const float*>(A_float.get()),
                lda,
                const_cast<const float*>(B_float.get()),
                ldb,
                beta,
                C,
                ldc);
}

template <>
void ref_gemm<hipblasBfloat16, hipblasBfloat16, float>(hipblasOperation_t transA,
                                                       hipblasOperation_t transB,
                                                       int64_t            m,
                                                       int64_t            n,
                                                       int64_t            k,
                                                       float              alpha,
                                                       hipblasBfloat16*   A,
                                                       int64_t            lda,
                                                       hipblasBfloat16*   B,
                                                       int64_t            ldb,
                                                       float              beta,
                                                       hipblasBfloat16*   C,
                                                       int64_t            ldc)
{
    // cblas does not support hipblasBfloat16, so convert to higher precision float
    // This will give more precise result which is acceptable for testing

    size_t sizeA = transA == HIPBLAS_OP_N ? size_t(k) * lda : size_t(m) * lda;
    size_t sizeB = transB == HIPBLAS_OP_N ? size_t(n) * ldb : size_t(k) * ldb;
    size_t sizeC = size_t(n) * ldc;

    std::unique_ptr<float[]> A_float(new float[sizeA]());
    std::unique_ptr<float[]> B_float(new float[sizeB]());
    std::unique_ptr<float[]> C_float(new float[sizeC]());

    for(size_t i = 0; i < sizeA; i++)
    {
        A_float[i] = bfloat16_to_float(A[i]);
    }
    for(size_t i = 0; i < sizeB; i++)
    {
        B_float[i] = bfloat16_to_float(B[i]);
    }
    for(size_t i = 0; i < sizeC; i++)
    {
        C_float[i] = bfloat16_to_float(C[i]);
    }

    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: rocblas =%d, cblas=%d\n", transA, (CBLAS_TRANSPOSE)transA );
    cblas_sgemm(CblasColMajor,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_TRANSPOSE)transB,
                m,
                n,
                k,
                alpha,
                const_cast<const float*>(A_float.get()),
                lda,
                const_cast<const float*>(B_float.get()),
                ldb,
                beta,
                static_cast<float*>(C_float.get()),
                ldc);

    for(size_t i = 0; i < sizeC; i++)
    {
        C[i] = float_to_bfloat16(C_float[i]);
    }
}

template <>
void ref_gemm<hipblasBfloat16, float, float>(hipblasOperation_t transA,
                                             hipblasOperation_t transB,
                                             int64_t            m,
                                             int64_t            n,
                                             int64_t            k,
                                             float              alpha,
                                             hipblasBfloat16*   A,
                                             int64_t            lda,
                                             hipblasBfloat16*   B,
                                             int64_t            ldb,
                                             float              beta,
                                             float*             C,
                                             int64_t            ldc)
{
    // cblas does not support hipblasBfloat16, so convert to higher precision float
    // This will give more precise result which is acceptable for testing

    size_t sizeA = transA == HIPBLAS_OP_N ? size_t(k) * lda : size_t(m) * lda;
    size_t sizeB = transB == HIPBLAS_OP_N ? size_t(n) * ldb : size_t(k) * ldb;

    std::unique_ptr<float[]> A_float(new float[sizeA]());
    std::unique_ptr<float[]> B_float(new float[sizeB]());

    for(size_t i = 0; i < sizeA; i++)
    {
        A_float[i] = bfloat16_to_float(A[i]);
    }
    for(size_t i = 0; i < sizeB; i++)
    {
        B_float[i] = bfloat16_to_float(B[i]);
    }

    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: rocblas =%d, cblas=%d\n", transA, (CBLAS_TRANSPOSE)transA );
    cblas_sgemm(CblasColMajor,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_TRANSPOSE)transB,
                m,
                n,
                k,
                alpha,
                const_cast<const float*>(A_float.get()),
                lda,
                const_cast<const float*>(B_float.get()),
                ldb,
                beta,
                C,
                ldc);
}

template <>
void ref_gemm<float>(hipblasOperation_t transA,
                     hipblasOperation_t transB,
                     int64_t            m,
                     int64_t            n,
                     int64_t            k,
                     float              alpha,
                     float*             A,
                     int64_t            lda,
                     float*             B,
                     int64_t            ldb,
                     float              beta,
                     float*             C,
                     int64_t            ldc)
{
    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: hipblas =%d, cblas=%d\n", transA, (CBLAS_TRANSPOSE)transA );
    cblas_sgemm(CblasColMajor,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_TRANSPOSE)transB,
                m,
                n,
                k,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc);
}

template <>
void ref_gemm<double>(hipblasOperation_t transA,
                      hipblasOperation_t transB,
                      int64_t            m,
                      int64_t            n,
                      int64_t            k,
                      double             alpha,
                      double*            A,
                      int64_t            lda,
                      double*            B,
                      int64_t            ldb,
                      double             beta,
                      double*            C,
                      int64_t            ldc)
{
    cblas_dgemm(CblasColMajor,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_TRANSPOSE)transB,
                m,
                n,
                k,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc);
}

template <>
void ref_gemm<std::complex<float>>(hipblasOperation_t   transA,
                                   hipblasOperation_t   transB,
                                   int64_t              m,
                                   int64_t              n,
                                   int64_t              k,
                                   std::complex<float>  alpha,
                                   std::complex<float>* A,
                                   int64_t              lda,
                                   std::complex<float>* B,
                                   int64_t              ldb,
                                   std::complex<float>  beta,
                                   std::complex<float>* C,
                                   int64_t              ldc)
{
    //just directly cast, since transA, transB are integers in the enum
    cblas_cgemm(CblasColMajor,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_TRANSPOSE)transB,
                m,
                n,
                k,
                &alpha,
                A,
                lda,
                B,
                ldb,
                &beta,
                C,
                ldc);
}

template <>
void ref_gemm<std::complex<double>>(hipblasOperation_t    transA,
                                    hipblasOperation_t    transB,
                                    int64_t               m,
                                    int64_t               n,
                                    int64_t               k,
                                    std::complex<double>  alpha,
                                    std::complex<double>* A,
                                    int64_t               lda,
                                    std::complex<double>* B,
                                    int64_t               ldb,
                                    std::complex<double>  beta,
                                    std::complex<double>* C,
                                    int64_t               ldc)
{
    cblas_zgemm(CblasColMajor,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_TRANSPOSE)transB,
                m,
                n,
                k,
                &alpha,
                A,
                lda,
                B,
                ldb,
                &beta,
                C,
                ldc);
}

template <>
void ref_gemm<int8_t, int32_t, int32_t>(hipblasOperation_t transA,
                                        hipblasOperation_t transB,
                                        int64_t            m,
                                        int64_t            n,
                                        int64_t            k,
                                        int32_t            alpha,
                                        int8_t*            A,
                                        int64_t            lda,
                                        int8_t*            B,
                                        int64_t            ldb,
                                        int32_t            beta,
                                        int32_t*           C,
                                        int64_t            ldc)
{
    double alpha_double = static_cast<double>(alpha);
    double beta_double  = static_cast<double>(beta);

    size_t const sizeA = ((transA == HIPBLAS_OP_N) ? k : m) * size_t(lda);
    size_t const sizeB = ((transB == HIPBLAS_OP_N) ? n : k) * size_t(ldb);
    size_t const sizeC = n * size_t(ldc);

    std::unique_ptr<double[]> A_double(new double[sizeA]());
    std::unique_ptr<double[]> B_double(new double[sizeB]());
    std::unique_ptr<double[]> C_double(new double[sizeC]());

    for(int64_t i = 0; i < sizeA; i++)
    {
        A_double[i] = static_cast<double>(A[i]);
    }
    for(int64_t i = 0; i < sizeB; i++)
    {
        B_double[i] = static_cast<double>(B[i]);
    }
    for(int64_t i = 0; i < sizeC; i++)
    {
        C_double[i] = static_cast<double>(C[i]);
    }

    cblas_dgemm(CblasColMajor,
                static_cast<CBLAS_TRANSPOSE>(transA),
                static_cast<CBLAS_TRANSPOSE>(transB),
                m,
                n,
                k,
                alpha_double,
                const_cast<const double*>(A_double.get()),
                lda,
                const_cast<const double*>(B_double.get()),
                ldb,
                beta_double,
                static_cast<double*>(C_double.get()),
                ldc);

    for(size_t i = 0; i < sizeC; i++)
        C[i] = static_cast<int32_t>(C_double[i]);
}

// hemm
template <>
void ref_hemm(hipblasSideMode_t    side,
              hipblasFillMode_t    uplo,
              int64_t              m,
              int64_t              n,
              std::complex<float>  alpha,
              std::complex<float>* A,
              int64_t              lda,
              std::complex<float>* B,
              int64_t              ldb,
              std::complex<float>  beta,
              std::complex<float>* C,
              int64_t              ldc)
{
    cblas_chemm(CblasColMajor,
                (CBLAS_SIDE)side,
                (CBLAS_UPLO)uplo,
                m,
                n,
                &alpha,
                A,
                lda,
                B,
                ldb,
                &beta,
                C,
                ldc);
}

template <>
void ref_hemm(hipblasSideMode_t     side,
              hipblasFillMode_t     uplo,
              int64_t               m,
              int64_t               n,
              std::complex<double>  alpha,
              std::complex<double>* A,
              int64_t               lda,
              std::complex<double>* B,
              int64_t               ldb,
              std::complex<double>  beta,
              std::complex<double>* C,
              int64_t               ldc)
{
    cblas_zhemm(CblasColMajor,
                (CBLAS_SIDE)side,
                (CBLAS_UPLO)uplo,
                m,
                n,
                &alpha,
                A,
                lda,
                B,
                ldb,
                &beta,
                C,
                ldc);
}

// herk
template <>
void ref_herk(hipblasFillMode_t    uplo,
              hipblasOperation_t   transA,
              int64_t              n,
              int64_t              k,
              float                alpha,
              std::complex<float>* A,
              int64_t              lda,
              float                beta,
              std::complex<float>* C,
              int64_t              ldc)
{
    cblas_cherk(CblasColMajor,
                (CBLAS_UPLO)uplo,
                (CBLAS_TRANSPOSE)transA,
                n,
                k,
                alpha,
                A,
                lda,
                beta,
                C,
                ldc);
}

template <>
void ref_herk(hipblasFillMode_t     uplo,
              hipblasOperation_t    transA,
              int64_t               n,
              int64_t               k,
              double                alpha,
              std::complex<double>* A,
              int64_t               lda,
              double                beta,
              std::complex<double>* C,
              int64_t               ldc)
{
    cblas_zherk(CblasColMajor,
                (CBLAS_UPLO)uplo,
                (CBLAS_TRANSPOSE)transA,
                n,
                k,
                alpha,
                A,
                lda,
                beta,
                C,
                ldc);
}

// herkx
template <typename T, typename U>
void ref_herkx_local(hipblasFillMode_t  uplo,
                     hipblasOperation_t transA,
                     int64_t            n,
                     int64_t            k,
                     T                  alpha,
                     T*                 A,
                     int64_t            lda,
                     T*                 B,
                     int64_t            ldb,
                     U                  beta,
                     T*                 C,
                     int64_t            ldc)
{

    if(n <= 0 || (beta == 1 && (k == 0 || alpha == T(0.0))))
        return;

    if(transA == HIPBLAS_OP_N)
    {
        if(uplo == HIPBLAS_FILL_MODE_UPPER)
        {
            for(int64_t j = 0; j < n; ++j)
            {
                for(int64_t i = 0; i <= j; i++)
                    C[i + j * ldc] *= T(beta);

                for(int64_t l = 0; l < k; l++)
                {
                    T temp = alpha * std::conj(B[j + l * ldb]);
                    for(int64_t i = 0; i <= j; ++i)
                        C[i + j * ldc] += temp * A[i + l * lda];
                }
            }
        }
        else // lower
        {
            for(int64_t j = 0; j < n; ++j)
            {
                for(int64_t i = j; i < n; i++)
                    C[i + j * ldc] *= T(beta);

                for(int64_t l = 0; l < k; l++)
                {
                    T temp = alpha * std::conj(B[j + l * ldb]);
                    for(int64_t i = j; i < n; ++i)
                        C[i + j * ldc] += temp * A[i + l * lda];
                }
            }
        }
    }
    else // conjugate transpose
    {
        if(uplo == HIPBLAS_FILL_MODE_UPPER)
        {
            for(int64_t j = 0; j < n; ++j)
                for(int64_t i = 0; i <= j; i++)
                {
                    C[i + j * ldc] *= T(beta);
                    T temp(0);
                    for(int64_t l = 0; l < k; l++)
                        temp += std::conj(A[l + i * lda]) * B[l + j * ldb];
                    C[i + j * ldc] += alpha * temp;
                }
        }
        else // lower
        {
            for(int64_t j = 0; j < n; ++j)
                for(int64_t i = j; i < n; i++)
                {
                    C[i + j * ldc] *= T(beta);
                    T temp(0);
                    for(int64_t l = 0; l < k; l++)
                        temp += std::conj(A[l + i * lda]) * B[l + j * ldb];
                    C[i + j * ldc] += alpha * temp;
                }
        }
    }

    for(int64_t i = 0; i < n; i++)
        C[i + i * ldc].imag(0);
}

template <>
void ref_herkx(hipblasFillMode_t    uplo,
               hipblasOperation_t   transA,
               int64_t              n,
               int64_t              k,
               std::complex<float>  alpha,
               std::complex<float>* A,
               int64_t              lda,
               std::complex<float>* B,
               int64_t              ldb,
               float                beta,
               std::complex<float>* C,
               int64_t              ldc)
{
    ref_herkx_local(uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
void ref_herkx(hipblasFillMode_t     uplo,
               hipblasOperation_t    transA,
               int64_t               n,
               int64_t               k,
               std::complex<double>  alpha,
               std::complex<double>* A,
               int64_t               lda,
               std::complex<double>* B,
               int64_t               ldb,
               double                beta,
               std::complex<double>* C,
               int64_t               ldc)
{
    ref_herkx_local(uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

// her2k
template <>
void ref_her2k(hipblasFillMode_t    uplo,
               hipblasOperation_t   transA,
               int64_t              n,
               int64_t              k,
               std::complex<float>  alpha,
               std::complex<float>* A,
               int64_t              lda,
               std::complex<float>* B,
               int64_t              ldb,
               float                beta,
               std::complex<float>* C,
               int64_t              ldc)
{
    cblas_cher2k(CblasColMajor,
                 (CBLAS_UPLO)uplo,
                 (CBLAS_TRANSPOSE)transA,
                 n,
                 k,
                 &alpha,
                 A,
                 lda,
                 B,
                 ldb,
                 beta,
                 C,
                 ldc);
}

template <>
void ref_her2k(hipblasFillMode_t     uplo,
               hipblasOperation_t    transA,
               int64_t               n,
               int64_t               k,
               std::complex<double>  alpha,
               std::complex<double>* A,
               int64_t               lda,
               std::complex<double>* B,
               int64_t               ldb,
               double                beta,
               std::complex<double>* C,
               int64_t               ldc)
{
    cblas_zher2k(CblasColMajor,
                 (CBLAS_UPLO)uplo,
                 (CBLAS_TRANSPOSE)transA,
                 n,
                 k,
                 &alpha,
                 A,
                 lda,
                 B,
                 ldb,
                 beta,
                 C,
                 ldc);
}

// symm
template <>
void ref_symm(hipblasSideMode_t side,
              hipblasFillMode_t uplo,
              int64_t           m,
              int64_t           n,
              float             alpha,
              float*            A,
              int64_t           lda,
              float*            B,
              int64_t           ldb,
              float             beta,
              float*            C,
              int64_t           ldc)
{
    cblas_ssymm(CblasColMajor,
                (CBLAS_SIDE)side,
                (CBLAS_UPLO)uplo,
                m,
                n,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc);
}

template <>
void ref_symm(hipblasSideMode_t side,
              hipblasFillMode_t uplo,
              int64_t           m,
              int64_t           n,
              double            alpha,
              double*           A,
              int64_t           lda,
              double*           B,
              int64_t           ldb,
              double            beta,
              double*           C,
              int64_t           ldc)
{
    cblas_dsymm(CblasColMajor,
                (CBLAS_SIDE)side,
                (CBLAS_UPLO)uplo,
                m,
                n,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc);
}

template <>
void ref_symm(hipblasSideMode_t    side,
              hipblasFillMode_t    uplo,
              int64_t              m,
              int64_t              n,
              std::complex<float>  alpha,
              std::complex<float>* A,
              int64_t              lda,
              std::complex<float>* B,
              int64_t              ldb,
              std::complex<float>  beta,
              std::complex<float>* C,
              int64_t              ldc)
{
    cblas_csymm(CblasColMajor,
                (CBLAS_SIDE)side,
                (CBLAS_UPLO)uplo,
                m,
                n,
                &alpha,
                A,
                lda,
                B,
                ldb,
                &beta,
                C,
                ldc);
}

template <>
void ref_symm(hipblasSideMode_t     side,
              hipblasFillMode_t     uplo,
              int64_t               m,
              int64_t               n,
              std::complex<double>  alpha,
              std::complex<double>* A,
              int64_t               lda,
              std::complex<double>* B,
              int64_t               ldb,
              std::complex<double>  beta,
              std::complex<double>* C,
              int64_t               ldc)
{
    cblas_zsymm(CblasColMajor,
                (CBLAS_SIDE)side,
                (CBLAS_UPLO)uplo,
                m,
                n,
                &alpha,
                A,
                lda,
                B,
                ldb,
                &beta,
                C,
                ldc);
}

// syrk
template <>
void ref_syrk(hipblasFillMode_t  uplo,
              hipblasOperation_t transA,
              int64_t            n,
              int64_t            k,
              float              alpha,
              float*             A,
              int64_t            lda,
              float              beta,
              float*             C,
              int64_t            ldc)
{
    cblas_ssyrk(CblasColMajor,
                (CBLAS_UPLO)uplo,
                (CBLAS_TRANSPOSE)transA,
                n,
                k,
                alpha,
                A,
                lda,
                beta,
                C,
                ldc);
}

template <>
void ref_syrk(hipblasFillMode_t  uplo,
              hipblasOperation_t transA,
              int64_t            n,
              int64_t            k,
              double             alpha,
              double*            A,
              int64_t            lda,
              double             beta,
              double*            C,
              int64_t            ldc)
{
    cblas_dsyrk(CblasColMajor,
                (CBLAS_UPLO)uplo,
                (CBLAS_TRANSPOSE)transA,
                n,
                k,
                alpha,
                A,
                lda,
                beta,
                C,
                ldc);
}

template <>
void ref_syrk(hipblasFillMode_t    uplo,
              hipblasOperation_t   transA,
              int64_t              n,
              int64_t              k,
              std::complex<float>  alpha,
              std::complex<float>* A,
              int64_t              lda,
              std::complex<float>  beta,
              std::complex<float>* C,
              int64_t              ldc)
{
    cblas_csyrk(CblasColMajor,
                (CBLAS_UPLO)uplo,
                (CBLAS_TRANSPOSE)transA,
                n,
                k,
                &alpha,
                A,
                lda,
                &beta,
                C,
                ldc);
}

template <>
void ref_syrk(hipblasFillMode_t     uplo,
              hipblasOperation_t    transA,
              int64_t               n,
              int64_t               k,
              std::complex<double>  alpha,
              std::complex<double>* A,
              int64_t               lda,
              std::complex<double>  beta,
              std::complex<double>* C,
              int64_t               ldc)
{
    cblas_zsyrk(CblasColMajor,
                (CBLAS_UPLO)uplo,
                (CBLAS_TRANSPOSE)transA,
                n,
                k,
                &alpha,
                A,
                lda,
                &beta,
                C,
                ldc);
}

// syrk_ex
template <typename T, typename U, typename Tc>
void ref_syrk_ex(hipblasFillMode_t  uplo,
                 hipblasOperation_t transA,
                 int64_t            n,
                 int64_t            k,
                 Tc                 alpha,
                 const T*           A,
                 int64_t            lda,
                 Tc                 beta,
                 U*                 C,
                 int64_t            ldc)
{
    if constexpr(!std::is_same_v<Tc, double>)
    {
        float alpha_float = alpha;
        float beta_float  = beta;

        host_vector<float> A_float, C_float;

        cast_to_buffer(transA, n, k, lda, A, A_float);
        cast_to_buffer(HIPBLAS_OP_N, n, n, ldc, C, C_float);

        ref_syrk(
            uplo, transA, n, k, alpha_float, A_float.data(), lda, beta_float, C_float.data(), ldc);

        cast_from_buffer(n, n, ldc, C_float, C);
    }
    else
    {
        double alpha_double = alpha;
        double beta_double  = beta;

        host_vector<double> A_double, C_double;

        cast_to_buffer(transA, n, k, lda, A, A_double);
        cast_to_buffer(HIPBLAS_OP_N, n, n, ldc, C, C_double);

        ref_syrk(uplo,
                 transA,
                 n,
                 k,
                 alpha_double,
                 A_double.data(),
                 lda,
                 beta_double,
                 C_double.data(),
                 ldc);

        cast_from_buffer(n, n, ldc, C_double, C);
    }
}

#define INSTANTIATE_SYRK_EX_TEMPLATE(T_, U_, Tc_)                     \
    template void ref_syrk_ex<T_, U_, Tc_>(hipblasFillMode_t  uplo,   \
                                           hipblasOperation_t transA, \
                                           int64_t            n,      \
                                           int64_t            k,      \
                                           Tc_                alpha,  \
                                           const T_*          A,      \
                                           int64_t            lda,    \
                                           Tc_                beta,   \
                                           U_*                C,      \
                                           int64_t            ldc);

INSTANTIATE_SYRK_EX_TEMPLATE(hipblasHalf, hipblasHalf, float)
INSTANTIATE_SYRK_EX_TEMPLATE(hipblasHalf, float, float)
INSTANTIATE_SYRK_EX_TEMPLATE(hipblasBfloat16, hipblasBfloat16, float)
INSTANTIATE_SYRK_EX_TEMPLATE(hipblasBfloat16, float, float)
INSTANTIATE_SYRK_EX_TEMPLATE(float, float, double)
INSTANTIATE_SYRK_EX_TEMPLATE(float, double, double)

// herk_ex
template <typename T, typename U, typename Tc>
void ref_herk_ex(hipblasFillMode_t  uplo,
                 hipblasOperation_t transA,
                 int64_t            n,
                 int64_t            k,
                 Tc                 alpha,
                 const T*           A,
                 int64_t            lda,
                 Tc                 beta,
                 U*                 C,
                 int64_t            ldc)
{
    double alpha_double = alpha;
    double beta_double  = beta;

    host_vector<std::complex<double>> A_double, C_double;

    cast_to_buffer(transA, n, k, lda, A, A_double);
    cast_to_buffer(HIPBLAS_OP_N, n, n, ldc, C, C_double);

    ref_herk(uplo,
             transA,
             n,
             k,
             alpha_double,
             (std::complex<double>*)A_double.data(),
             lda,
             beta_double,
             (std::complex<double>*)C_double.data(),
             ldc);

    cast_from_buffer(n, n, ldc, C_double, C);
}

#define INSTANTIATE_HERK_EX_TEMPLATE(T_, U_, Tc_)                     \
    template void ref_herk_ex<T_, U_, Tc_>(hipblasFillMode_t  uplo,   \
                                           hipblasOperation_t transA, \
                                           int64_t            n,      \
                                           int64_t            k,      \
                                           Tc_                alpha,  \
                                           const T_*          A,      \
                                           int64_t            lda,    \
                                           Tc_                beta,   \
                                           U_*                C,      \
                                           int64_t            ldc);

INSTANTIATE_HERK_EX_TEMPLATE(std::complex<float>, std::complex<float>, double)
INSTANTIATE_HERK_EX_TEMPLATE(std::complex<float>, std::complex<double>, double)

// syr2k
template <>
void ref_syr2k(hipblasFillMode_t  uplo,
               hipblasOperation_t transA,
               int64_t            n,
               int64_t            k,
               float              alpha,
               float*             A,
               int64_t            lda,
               float*             B,
               int64_t            ldb,
               float              beta,
               float*             C,
               int64_t            ldc)
{
    cblas_ssyr2k(CblasColMajor,
                 (CBLAS_UPLO)uplo,
                 (CBLAS_TRANSPOSE)transA,
                 n,
                 k,
                 alpha,
                 A,
                 lda,
                 B,
                 ldb,
                 beta,
                 C,
                 ldc);
}

template <>
void ref_syr2k(hipblasFillMode_t  uplo,
               hipblasOperation_t transA,
               int64_t            n,
               int64_t            k,
               double             alpha,
               double*            A,
               int64_t            lda,
               double*            B,
               int64_t            ldb,
               double             beta,
               double*            C,
               int64_t            ldc)
{
    cblas_dsyr2k(CblasColMajor,
                 (CBLAS_UPLO)uplo,
                 (CBLAS_TRANSPOSE)transA,
                 n,
                 k,
                 alpha,
                 A,
                 lda,
                 B,
                 ldb,
                 beta,
                 C,
                 ldc);
}

template <>
void ref_syr2k(hipblasFillMode_t    uplo,
               hipblasOperation_t   transA,
               int64_t              n,
               int64_t              k,
               std::complex<float>  alpha,
               std::complex<float>* A,
               int64_t              lda,
               std::complex<float>* B,
               int64_t              ldb,
               std::complex<float>  beta,
               std::complex<float>* C,
               int64_t              ldc)
{
    cblas_csyr2k(CblasColMajor,
                 (CBLAS_UPLO)uplo,
                 (CBLAS_TRANSPOSE)transA,
                 n,
                 k,
                 &alpha,
                 A,
                 lda,
                 B,
                 ldb,
                 &beta,
                 C,
                 ldc);
}

template <>
void ref_syr2k(hipblasFillMode_t     uplo,
               hipblasOperation_t    transA,
               int64_t               n,
               int64_t               k,
               std::complex<double>  alpha,
               std::complex<double>* A,
               int64_t               lda,
               std::complex<double>* B,
               int64_t               ldb,
               std::complex<double>  beta,
               std::complex<double>* C,
               int64_t               ldc)
{
    cblas_zsyr2k(CblasColMajor,
                 (CBLAS_UPLO)uplo,
                 (CBLAS_TRANSPOSE)transA,
                 n,
                 k,
                 &alpha,
                 A,
                 lda,
                 B,
                 ldb,
                 &beta,
                 C,
                 ldc);
}

// syrkx
// Use syrk with A == B for now.

/*
// trsm
template <>
void ref_trsm<float>(hipblasSideMode_t  side,
                     hipblasFillMode_t  uplo,
                     hipblasOperation_t transA,
                     hipblasDiagType_t  diag,
                     int64_t            m,
                     int64_t            n,
                     float              alpha,
                     const float*       A,
                     int64_t            lda,
                     float*             B,
                     int64_t            ldb)
{
    // just directly cast, since transA, transB are integers in the enum
    cblas_strsm(CblasColMajor,
                (CBLAS_SIDE)side,
                (CBLAS_UPLO)uplo,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_DIAG)diag,
                m,
                n,
                alpha,
                A,
                lda,
                B,
                ldb);
}

template <>
void ref_trsm<double>(hipblasSideMode_t  side,
                      hipblasFillMode_t  uplo,
                      hipblasOperation_t transA,
                      hipblasDiagType_t  diag,
                      int64_t            m,
                      int64_t            n,
                      double             alpha,
                      const double*      A,
                      int64_t            lda,
                      double*            B,
                      int64_t            ldb)
{
    // just directly cast, since transA, transB are integers in the enum
    cblas_dtrsm(CblasColMajor,
                (CBLAS_SIDE)side,
                (CBLAS_UPLO)uplo,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_DIAG)diag,
                m,
                n,
                alpha,
                A,
                lda,
                B,
                ldb);
}

template <>
void ref_trsm<std::complex<float>>(hipblasSideMode_t     side,
                              hipblasFillMode_t     uplo,
                              hipblasOperation_t    transA,
                              hipblasDiagType_t     diag,
                              int64_t               m,
                              int64_t               n,
                              std::complex<float>        alpha,
                              const std::complex<float>* A,
                              int64_t               lda,
                              std::complex<float>*       B,
                              int64_t               ldb)
{
    cblas_ctrsm(CblasColMajor,
                (CBLAS_SIDE)side,
                (CBLAS_UPLO)uplo,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_DIAG)diag,
                m,
                n,
                &alpha,
                A,
                lda,
                B,
                ldb);
}

template <>
void ref_trsm<std::complex<double>>(hipblasSideMode_t           side,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int64_t                     m,
                                    int64_t                     n,
                                    std::complex<double>        alpha,
                                    const std::complex<double>* A,
                                    int64_t                     lda,
                                    std::complex<double>*       B,
                                    int64_t                     ldb)
{
    cblas_ztrsm(CblasColMajor,
                (CBLAS_SIDE)side,
                (CBLAS_UPLO)uplo,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_DIAG)diag,
                m,
                n,
                &alpha,
                A,
                lda,
                B,
                ldb);
}

*/

// trtri
template <>
void ref_trtri<float>(char uplo, char diag, int64_t n, float* A, int64_t lda)
{
    lapack_xtrtri(uplo, diag, n, A, lda);
}

template <>
void ref_trtri<double>(char uplo, char diag, int64_t n, double* A, int64_t lda)
{
    lapack_xtrtri(uplo, diag, n, A, lda);
}

template <>
void ref_trtri<std::complex<float>>(
    char uplo, char diag, int64_t n, std::complex<float>* A, int64_t lda)
{
    lapack_xtrtri(uplo, diag, n, A, lda);
}

template <>
void ref_trtri<std::complex<double>>(
    char uplo, char diag, int64_t n, std::complex<double>* A, int64_t lda)
{
    lapack_xtrtri(uplo, diag, n, A, lda);
}

// trmm
template <>
void ref_trmm<float>(hipblasSideMode_t  side,
                     hipblasFillMode_t  uplo,
                     hipblasOperation_t transA,
                     hipblasDiagType_t  diag,
                     int64_t            m,
                     int64_t            n,
                     float              alpha,
                     const float*       A,
                     int64_t            lda,
                     float*             B,
                     int64_t            ldb)
{
    // just directly cast, since transA, transB are integers in the enum
    cblas_strmm(CblasColMajor,
                (CBLAS_SIDE)side,
                (CBLAS_UPLO)uplo,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_DIAG)diag,
                m,
                n,
                alpha,
                A,
                lda,
                B,
                ldb);
}

template <>
void ref_trmm<double>(hipblasSideMode_t  side,
                      hipblasFillMode_t  uplo,
                      hipblasOperation_t transA,
                      hipblasDiagType_t  diag,
                      int64_t            m,
                      int64_t            n,
                      double             alpha,
                      const double*      A,
                      int64_t            lda,
                      double*            B,
                      int64_t            ldb)
{
    // just directly cast, since transA, transB are integers in the enum
    cblas_dtrmm(CblasColMajor,
                (CBLAS_SIDE)side,
                (CBLAS_UPLO)uplo,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_DIAG)diag,
                m,
                n,
                alpha,
                A,
                lda,
                B,
                ldb);
}

template <>
void ref_trmm<std::complex<float>>(hipblasSideMode_t          side,
                                   hipblasFillMode_t          uplo,
                                   hipblasOperation_t         transA,
                                   hipblasDiagType_t          diag,
                                   int64_t                    m,
                                   int64_t                    n,
                                   std::complex<float>        alpha,
                                   const std::complex<float>* A,
                                   int64_t                    lda,
                                   std::complex<float>*       B,
                                   int64_t                    ldb)
{
    cblas_ctrmm(CblasColMajor,
                (CBLAS_SIDE)side,
                (CBLAS_UPLO)uplo,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_DIAG)diag,
                m,
                n,
                &alpha,
                A,
                lda,
                B,
                ldb);
}

template <>
void ref_trmm<std::complex<double>>(hipblasSideMode_t           side,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int64_t                     m,
                                    int64_t                     n,
                                    std::complex<double>        alpha,
                                    const std::complex<double>* A,
                                    int64_t                     lda,
                                    std::complex<double>*       B,
                                    int64_t                     ldb)
{
    cblas_ztrmm(CblasColMajor,
                (CBLAS_SIDE)side,
                (CBLAS_UPLO)uplo,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_DIAG)diag,
                m,
                n,
                &alpha,
                A,
                lda,
                B,
                ldb);
}

/*
 * ===========================================================================
 *    LAPACK OR OTHER
 * ===========================================================================
 */

#ifdef __HIP_PLATFORM_SOLVER__

// potrf
template <>
int64_t ref_potrf(char uplo, int64_t m, float* A, int64_t lda)
{
    int64_t info;

#ifdef FLA_ENABLE_ILP64
    info = LAPACKE_spotrf(LAPACK_COL_MAJOR, uplo, m, A, lda);
#else
    spotrf_(&uplo, &m, A, &lda, &info);
#endif

    return info;
}

template <>
int64_t ref_potrf(char uplo, int64_t m, double* A, int64_t lda)
{
    int64_t info;

#ifdef FLA_ENABLE_ILP64
    info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, uplo, m, A, lda);
#else
    dpotrf_(&uplo, &m, A, &lda, &info);
#endif

    return info;
}

template <>
int64_t ref_potrf(char uplo, int64_t m, std::complex<float>* A, int64_t lda)
{
    int64_t info;

#ifdef FLA_ENABLE_ILP64
    info = LAPACKE_cpotrf(LAPACK_COL_MAJOR, uplo, m, (lapack_complex_float*)A, lda);
#else
    cpotrf_(&uplo, &m, A, &lda, &info);
#endif

    return info;
}

template <>
int64_t ref_potrf(char uplo, int64_t m, std::complex<double>* A, int64_t lda)
{
    int64_t info;

#ifdef FLA_ENABLE_ILP64
    info = LAPACKE_zpotrf(LAPACK_COL_MAJOR, uplo, m, (lapack_complex_double*)A, lda);
#else
    zpotrf_(&uplo, &m, A, &lda, &info);
#endif

    return info;
}

// getrf
template <>
int64_t ref_getrf<float>(int64_t m, int64_t n, float* A, int64_t lda, int64_t* ipiv)
{
    int64_t info;

#ifdef FLA_ENABLE_ILP64
    info = LAPACKE_sgetrf(LAPACK_COL_MAJOR, m, n, A, lda, ipiv);
#else
    sgetrf_(&m, &n, A, &lda, ipiv, &info);
#endif

    return info;
}

template <>
int64_t ref_getrf<double>(int64_t m, int64_t n, double* A, int64_t lda, int64_t* ipiv)
{
    int64_t info;

#ifdef FLA_ENABLE_ILP64
    info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, m, n, A, lda, ipiv);
#else
    dgetrf_(&m, &n, A, &lda, ipiv, &info);
#endif

    return info;
}

template <>
int64_t ref_getrf<std::complex<float>>(
    int64_t m, int64_t n, std::complex<float>* A, int64_t lda, int64_t* ipiv)
{
    int64_t info;

#ifdef FLA_ENABLE_ILP64
    info = LAPACKE_cgetrf(LAPACK_COL_MAJOR, m, n, (lapack_complex_float*)A, lda, ipiv);
#else
    cgetrf_(&m, &n, (std::complex<float>*)A, &lda, ipiv, &info);
#endif

    return info;
}

template <>
int64_t ref_getrf<std::complex<double>>(
    int64_t m, int64_t n, std::complex<double>* A, int64_t lda, int64_t* ipiv)
{
    int64_t info;

#ifdef FLA_ENABLE_ILP64
    info = LAPACKE_zgetrf(LAPACK_COL_MAJOR, m, n, (lapack_complex_double*)A, lda, ipiv);
#else
    zgetrf_(&m, &n, (std::complex<double>*)A, &lda, ipiv, &info);
#endif

    return info;
}

// getrs
template <>
int64_t ref_getrs<float>(char     trans,
                         int64_t  n,
                         int64_t  nrhs,
                         float*   A,
                         int64_t  lda,
                         int64_t* ipiv,
                         float*   B,
                         int64_t  ldb)
{
    int64_t info;

#ifdef FLA_ENABLE_ILP64
    info = LAPACKE_sgetrs(LAPACK_COL_MAJOR, trans, n, nrhs, A, lda, ipiv, B, ldb);
#else
    sgetrs_(&trans, &n, &nrhs, A, &lda, ipiv, B, &ldb, &info);
#endif

    return info;
}

template <>
int64_t ref_getrs<double>(char     trans,
                          int64_t  n,
                          int64_t  nrhs,
                          double*  A,
                          int64_t  lda,
                          int64_t* ipiv,
                          double*  B,
                          int64_t  ldb)
{
    int64_t info;

#ifdef FLA_ENABLE_ILP64
    info = LAPACKE_dgetrs(LAPACK_COL_MAJOR, trans, n, nrhs, A, lda, ipiv, B, ldb);
#else
    dgetrs_(&trans, &n, &nrhs, A, &lda, ipiv, B, &ldb, &info);
#endif

    return info;
}

template <>
int64_t ref_getrs<std::complex<float>>(char                 trans,
                                       int64_t              n,
                                       int64_t              nrhs,
                                       std::complex<float>* A,
                                       int64_t              lda,
                                       int64_t*             ipiv,
                                       std::complex<float>* B,
                                       int64_t              ldb)
{
    int64_t info;

#ifdef FLA_ENABLE_ILP64
    info = LAPACKE_cgetrs(LAPACK_COL_MAJOR,
                          trans,
                          n,
                          nrhs,
                          (lapack_complex_float*)A,
                          lda,
                          ipiv,
                          (lapack_complex_float*)B,
                          ldb);
#else
    cgetrs_(&trans,
            &n,
            &nrhs,
            (std::complex<float>*)A,
            &lda,
            ipiv,
            (std::complex<float>*)B,
            &ldb,
            &info);
#endif

    return info;
}

template <>
int64_t ref_getrs<std::complex<double>>(char                  trans,
                                        int64_t               n,
                                        int64_t               nrhs,
                                        std::complex<double>* A,
                                        int64_t               lda,
                                        int64_t*              ipiv,
                                        std::complex<double>* B,
                                        int64_t               ldb)
{

    int64_t info;

#ifdef FLA_ENABLE_ILP64
    info = LAPACKE_zgetrs(LAPACK_COL_MAJOR,
                          trans,
                          n,
                          nrhs,
                          (lapack_complex_double*)A,
                          lda,
                          ipiv,
                          (lapack_complex_double*)B,
                          ldb);
#else
    zgetrs_(&trans,
            &n,
            &nrhs,
            (std::complex<double>*)A,
            &lda,
            ipiv,
            (std::complex<double>*)B,
            &ldb,
            &info);
#endif

    return info;
}

// getri
template <>
int64_t
    ref_getri<float>(int64_t n, float* A, int64_t lda, int64_t* ipiv, float* work, int64_t lwork)
{
    int64_t info;

#ifdef FLA_ENABLE_ILP64
    info = LAPACKE_sgetri_work(LAPACK_COL_MAJOR, n, A, lda, ipiv, work, lwork);
#else
    sgetri_(&n, A, &lda, ipiv, work, &lwork, &info);
#endif

    return info;
}

template <>
int64_t
    ref_getri<double>(int64_t n, double* A, int64_t lda, int64_t* ipiv, double* work, int64_t lwork)
{
    int64_t info;

#ifdef FLA_ENABLE_ILP64
    info = LAPACKE_dgetri_work(LAPACK_COL_MAJOR, n, A, lda, ipiv, work, lwork);
#else
    dgetri_(&n, A, &lda, ipiv, work, &lwork, &info);
#endif

    return info;
}

template <>
int64_t ref_getri<std::complex<float>>(int64_t              n,
                                       std::complex<float>* A,
                                       int64_t              lda,
                                       int64_t*             ipiv,
                                       std::complex<float>* work,
                                       int64_t              lwork)
{
    int64_t info;

#ifdef FLA_ENABLE_ILP64
    info = LAPACKE_cgetri_work(LAPACK_COL_MAJOR,
                               n,
                               (lapack_complex_float*)A,
                               lda,
                               ipiv,
                               (lapack_complex_float*)work,
                               lwork);
#else
    cgetri_(&n, A, &lda, ipiv, work, &lwork, &info);
#endif

    return info;
}

template <>
int64_t ref_getri<std::complex<double>>(int64_t               n,
                                        std::complex<double>* A,
                                        int64_t               lda,
                                        int64_t*              ipiv,
                                        std::complex<double>* work,
                                        int64_t               lwork)
{
    int64_t info;

#ifdef FLA_ENABLE_ILP64
    info = LAPACKE_zgetri_work(LAPACK_COL_MAJOR,
                               n,
                               (lapack_complex_double*)A,
                               lda,
                               ipiv,
                               (lapack_complex_double*)work,
                               lwork);
#else
    zgetri_(&n, A, &lda, ipiv, work, &lwork, &info);
#endif

    return info;
}

// geqrf
template <>
int64_t ref_geqrf<float>(
    int64_t m, int64_t n, float* A, int64_t lda, float* tau, float* work, int64_t lwork)
{
    int64_t info;

#ifdef FLA_ENABLE_ILP64
    info = LAPACKE_sgeqrf_work(LAPACK_COL_MAJOR, m, n, A, lda, tau, work, lwork);
#else
    sgeqrf_(&m, &n, A, &lda, tau, work, &lwork, &info);
#endif

    return info;
}

template <>
int64_t ref_geqrf<double>(
    int64_t m, int64_t n, double* A, int64_t lda, double* tau, double* work, int64_t lwork)
{
    int64_t info;

#ifdef FLA_ENABLE_ILP64
    info = LAPACKE_dgeqrf_work(LAPACK_COL_MAJOR, m, n, A, lda, tau, work, lwork);
#else
    dgeqrf_(&m, &n, A, &lda, tau, work, &lwork, &info);
#endif

    return info;
}
template <>
int64_t ref_geqrf<std::complex<float>>(int64_t              m,
                                       int64_t              n,
                                       std::complex<float>* A,
                                       int64_t              lda,
                                       std::complex<float>* tau,
                                       std::complex<float>* work,
                                       int64_t              lwork)
{
    int64_t info;

#ifdef FLA_ENABLE_ILP64
    info = LAPACKE_cgeqrf_work(LAPACK_COL_MAJOR,
                               m,
                               n,
                               (lapack_complex_float*)A,
                               lda,
                               (lapack_complex_float*)tau,
                               (lapack_complex_float*)work,
                               lwork);
#else
    cgeqrf_(&m, &n, A, &lda, tau, work, &lwork, &info);
#endif

    return info;
}

template <>
int64_t ref_geqrf<std::complex<double>>(int64_t               m,
                                        int64_t               n,
                                        std::complex<double>* A,
                                        int64_t               lda,
                                        std::complex<double>* tau,
                                        std::complex<double>* work,
                                        int64_t               lwork)
{
    int64_t info;

#ifdef FLA_ENABLE_ILP64
    info = LAPACKE_zgeqrf_work(LAPACK_COL_MAJOR,
                               m,
                               n,
                               (lapack_complex_double*)A,
                               lda,
                               (lapack_complex_double*)tau,
                               (lapack_complex_double*)work,
                               lwork);
#else
    zgeqrf_(&m, &n, A, &lda, tau, work, &lwork, &info);
#endif

    return info;
}

// gels
template <>
int64_t ref_gels<float>(char    trans,
                        int64_t m,
                        int64_t n,
                        int64_t nrhs,
                        float*  A,
                        int64_t lda,
                        float*  B,
                        int64_t ldb,
                        float*  work,
                        int64_t lwork)
{
    int64_t info;

#ifdef FLA_ENABLE_ILP64
    info = LAPACKE_sgels_work(LAPACK_COL_MAJOR, trans, m, n, nrhs, A, lda, B, ldb, work, lwork);
#else
    sgels_(&trans, &m, &n, &nrhs, A, &lda, B, &ldb, work, &lwork, &info);
#endif

    return info;
}

template <>
int64_t ref_gels<double>(char    trans,
                         int64_t m,
                         int64_t n,
                         int64_t nrhs,
                         double* A,
                         int64_t lda,
                         double* B,
                         int64_t ldb,
                         double* work,
                         int64_t lwork)
{
    int64_t info;

#ifdef FLA_ENABLE_ILP64
    info = LAPACKE_dgels_work(LAPACK_COL_MAJOR, trans, m, n, nrhs, A, lda, B, ldb, work, lwork);
#else
    dgels_(&trans, &m, &n, &nrhs, A, &lda, B, &ldb, work, &lwork, &info);
#endif

    return info;
}

template <>
int64_t ref_gels<std::complex<float>>(char                 trans,
                                      int64_t              m,
                                      int64_t              n,
                                      int64_t              nrhs,
                                      std::complex<float>* A,
                                      int64_t              lda,
                                      std::complex<float>* B,
                                      int64_t              ldb,
                                      std::complex<float>* work,
                                      int64_t              lwork)
{
    int64_t info;
#ifdef FLA_ENABLE_ILP64
    info = LAPACKE_cgels_work(LAPACK_COL_MAJOR,
                              trans,
                              m,
                              n,
                              nrhs,
                              (lapack_complex_float*)A,
                              lda,
                              (lapack_complex_float*)B,
                              ldb,
                              (lapack_complex_float*)work,
                              lwork);
#else
    cgels_(&trans, &m, &n, &nrhs, A, &lda, B, &ldb, work, &lwork, &info);
#endif

    return info;
}

template <>
int64_t ref_gels<std::complex<double>>(char                  trans,
                                       int64_t               m,
                                       int64_t               n,
                                       int64_t               nrhs,
                                       std::complex<double>* A,
                                       int64_t               lda,
                                       std::complex<double>* B,
                                       int64_t               ldb,
                                       std::complex<double>* work,
                                       int64_t               lwork)
{
    int64_t info;

#ifdef FLA_ENABLE_ILP64
    info = LAPACKE_zgels_work(LAPACK_COL_MAJOR,
                              trans,
                              m,
                              n,
                              nrhs,
                              (lapack_complex_double*)A,
                              lda,
                              (lapack_complex_double*)B,
                              ldb,
                              (lapack_complex_double*)work,
                              lwork);
#else
    zgels_(&trans, &m, &n, &nrhs, A, &lda, B, &ldb, work, &lwork, &info);
#endif

    return info;
}

#endif
