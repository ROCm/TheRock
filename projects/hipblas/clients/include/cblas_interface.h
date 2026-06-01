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

#include "cblas.h"
#include "hipblas.h"
#include "type_utils.h"

/*!\file
 * \brief provide template functions interfaces to CBLAS C89 interfaces, it is only used for testing
 * not part of the GPU library
*/

/*
 * ===========================================================================
 *    level 1 BLAS
 * ===========================================================================
 */

template <typename Ta, typename Tx = Ta>
void ref_axpy(int64_t n, const Ta alpha, const Tx* x, int64_t incx, Tx* y, int64_t incy);

template <typename T, typename U = T>
void ref_scal(int64_t n, const U alpha, T* x, int64_t incx);

template <typename T>
void ref_copy(int64_t n, T* x, int64_t incx, T* y, int64_t incy);

template <typename T>
void ref_swap(int64_t n, T* x, int64_t incx, T* y, int64_t incy);

template <typename T>
void ref_dot(int64_t n, const T* x, int64_t incx, const T* y, int64_t incy, T* result);

template <typename T>
void ref_dotc(int64_t n, const T* x, int64_t incx, const T* y, int64_t incy, T* result);

template <typename T1, typename T2>
void ref_nrm2(int64_t n, const T1* x, int64_t incx, T2* result);

// template <typename T>
// void ref_iamax(int64_t n, const T* x, int64_t incx, int64_t* result);

// template <typename T>
// void ref_iamin(int64_t n, const T* x, int64_t incx, int64_t* result);

template <typename T1, typename T2 = T1, typename T3 = T1>
void ref_rot(int64_t n, T1* x, int64_t incx, T1* y, int64_t incy, T2 c, T3 s);

template <typename T1, typename T2 = T1>
void ref_rotg(T1* a, T1* b, T2* c, T1* s);

// asum
template <typename T>
void ref_asum(int64_t n, const T* x, int64_t incx, real_t<T>* result);

template <>
inline void ref_asum(int64_t n, const double* x, int64_t incx, double* result)
{
    *result = cblas_dasum(n, x, incx);
}

template <>
inline void ref_asum(int64_t n, const std::complex<float>* x, int64_t incx, float* result)
{
    *result = cblas_scasum(n, x, incx);
}

template <>
inline void ref_asum(int64_t n, const std::complex<double>* x, int64_t incx, double* result)
{
    *result = cblas_dzasum(n, x, incx);
}

// rotm
template <typename T1>
void ref_rotm(int64_t n, T1* x, int64_t incx, T1* y, int64_t incy, T1* param);

template <>
inline void ref_rotm<float>(int64_t n, float* x, int64_t incx, float* y, int64_t incy, float* param)
{
    cblas_srotm(n, x, incx, y, incy, param);
}

template <>
inline void
    ref_rotm<double>(int64_t n, double* x, int64_t incx, double* y, int64_t incy, double* param)
{
    cblas_drotm(n, x, incx, y, incy, param);
}

// rotmg
template <typename T1>
void ref_rotmg(T1* d1, T1* d2, T1* x1, T1* y1, T1* param);

template <>
inline void ref_rotmg<float>(float* d1, float* d2, float* x1, float* y1, float* param)
{
    cblas_srotmg(d1, d2, x1, *y1, param);
}

template <>
inline void ref_rotmg<double>(double* d1, double* d2, double* x1, double* y1, double* param)
{
    cblas_drotmg(d1, d2, x1, *y1, param);
}

/*
 * ===========================================================================
 *    BLAS 2
 * ===========================================================================
 */

template <typename T>
void ref_gbmv(hipblasOperation_t transA,
              int64_t            m,
              int64_t            n,
              int64_t            kl,
              int64_t            ku,
              T                  alpha,
              T*                 A,
              int64_t            lda,
              T*                 x,
              int64_t            incx,
              T                  beta,
              T*                 y,
              int64_t            incy);

template <typename T>
void ref_gemv(hipblasOperation_t transA,
              int64_t            m,
              int64_t            n,
              T                  alpha,
              T*                 A,
              int64_t            lda,
              T*                 x,
              int64_t            incx,
              T                  beta,
              T*                 y,
              int64_t            incy);

template <typename T>
void ref_symv(hipblasFillMode_t uplo,
              int64_t           n,
              T                 alpha,
              T*                A,
              int64_t           lda,
              T*                x,
              int64_t           incx,
              T                 beta,
              T*                y,
              int64_t           incy);

// ger (ger, geru, gerc)
template <typename T, bool CONJ>
void ref_ger(
    int64_t m, int64_t n, T alpha, T* x, int64_t incx, T* y, int64_t incy, T* A, int64_t lda);

// hbmv
template <typename T>
void ref_hbmv(hipblasFillMode_t uplo,
              int64_t           n,
              int64_t           k,
              T                 alpha,
              T*                A,
              int64_t           lda,
              T*                x,
              int64_t           incx,
              T                 beta,
              T*                y,
              int64_t           incy);

// hemv
template <typename T, typename U>
void ref_hemv(hipblasFillMode_t uplo,
              int64_t           n,
              U                 alpha,
              T*                A,
              int64_t           lda,
              T*                x,
              int64_t           incx,
              T                 beta,
              T*                y,
              int64_t           incy);

// spr
template <typename T>
void ref_spr(hipblasFillMode_t uplo, int64_t n, T alpha, T* x, int64_t incx, T* AP);

// spr2
template <typename T>
void ref_spr2(
    hipblasFillMode_t uplo, int64_t n, T alpha, T* x, int64_t incx, T* y, int64_t incy, T* AP);

// syr
template <typename T>
void ref_syr(hipblasFillMode_t uplo, int64_t n, T alpha, T* x, int64_t incx, T* A, int64_t lda);

// syr2
template <typename T>
void ref_syr2(hipblasFillMode_t uplo,
              int64_t           n,
              T                 alpha,
              T*                x,
              int64_t           incx,
              T*                y,
              int64_t           incy,
              T*                A,
              int64_t           lda);

// her
template <typename T, typename U>
void ref_her(hipblasFillMode_t uplo, int64_t n, U alpha, T* x, int64_t incx, T* A, int64_t lda);

// her2
template <typename T>
void ref_her2(hipblasFillMode_t uplo,
              int64_t           n,
              T                 alpha,
              T*                x,
              int64_t           incx,
              T*                y,
              int64_t           incy,
              T*                A,
              int64_t           lda);

// hpmv
template <typename T>
void ref_hpmv(hipblasFillMode_t uplo,
              int64_t           n,
              T                 alpha,
              T*                AP,
              T*                x,
              int64_t           incx,
              T                 beta,
              T*                y,
              int64_t           incy);

// hpr
template <typename T, typename U>
void ref_hpr(hipblasFillMode_t uplo, int64_t n, U alpha, T* x, int64_t incx, T* AP);

// hpr2
template <typename T>
void ref_hpr2(
    hipblasFillMode_t uplo, int64_t n, T alpha, T* x, int64_t incx, T* y, int64_t incy, T* AP);

// hemv
template <typename T, typename U>
void ref_hemv(hipblasFillMode_t uplo,
              int64_t           n,
              U                 alpha,
              T*                A,
              int64_t           lda,
              T*                x,
              int64_t           incx,
              T                 beta,
              T*                y,
              int64_t           incy);

// spr
template <typename T>
void ref_spr(hipblasFillMode_t uplo, int64_t n, T alpha, T* x, int64_t incx, T* AP);

// spr2
template <typename T>
void ref_spr2(
    hipblasFillMode_t uplo, int64_t n, T alpha, T* x, int64_t incx, T* y, int64_t incy, T* AP);

// syr
template <typename T>
void ref_syr(hipblasFillMode_t uplo, int64_t n, T alpha, T* x, int64_t incx, T* A, int64_t lda);

// syr2
template <typename T>
void ref_syr2(hipblasFillMode_t uplo,
              int64_t           n,
              T                 alpha,
              T*                x,
              int64_t           incx,
              T*                y,
              int64_t           incy,
              T*                A,
              int64_t           lda);

// her
template <typename T, typename U>
void ref_her(hipblasFillMode_t uplo, int64_t n, U alpha, T* x, int64_t incx, T* A, int64_t lda);

// her2
template <typename T>
void ref_her2(hipblasFillMode_t uplo,
              int64_t           n,
              T                 alpha,
              T*                x,
              int64_t           incx,
              T*                y,
              int64_t           incy,
              T*                A,
              int64_t           lda);

// hemv
template <typename T>
void ref_hemv(hipblasFillMode_t uplo,
              int64_t           n,
              T                 alpha,
              T*                A,
              int64_t           lda,
              T*                x,
              int64_t           incx,
              T                 beta,
              T*                y,
              int64_t           incy);

// hpmv
template <typename T>
void ref_hpmv(hipblasFillMode_t uplo,
              int64_t           n,
              T                 alpha,
              T*                AP,
              T*                x,
              int64_t           incx,
              T                 beta,
              T*                y,
              int64_t           incy);

// hpr
template <typename T, typename U>
void ref_hpr(hipblasFillMode_t uplo, int64_t n, U alpha, T* x, int64_t incx, T* AP);

// hpr2
template <typename T>
void ref_hpr2(
    hipblasFillMode_t uplo, int64_t n, T alpha, T* x, int64_t incx, T* y, int64_t incy, T* AP);

// sbmv
template <typename T>
void ref_sbmv(hipblasFillMode_t uplo,
              int64_t           n,
              int64_t           k,
              T                 alpha,
              T*                A,
              int64_t           lda,
              T*                x,
              int64_t           incx,
              T                 beta,
              T*                y,
              int64_t           incy);

// spmv
template <typename T>
void ref_spmv(hipblasFillMode_t uplo,
              int64_t           n,
              T                 alpha,
              T*                AP,
              T*                x,
              int64_t           incx,
              T                 beta,
              T*                y,
              int64_t           incy);

// symv
template <typename T>
void ref_symv(hipblasFillMode_t uplo,
              int64_t           n,
              T                 alpha,
              T*                A,
              int64_t           lda,
              T*                x,
              int64_t           incx,
              T                 beta,
              T*                y,
              int64_t           incy);

// spmv
template <typename T>
void ref_spmv(hipblasFillMode_t uplo,
              int64_t           n,
              T                 alpha,
              T*                AP,
              T*                x,
              int64_t           incx,
              T                 beta,
              T*                y,
              int64_t           incy);

// symv
template <typename T>
void ref_symv(hipblasFillMode_t uplo,
              int64_t           n,
              T                 alpha,
              T*                A,
              int64_t           lda,
              T*                x,
              int64_t           incx,
              T                 beta,
              T*                y,
              int64_t           incy);

// tbmv
template <typename T>
void ref_tbmv(hipblasFillMode_t  uplo,
              hipblasOperation_t transA,
              hipblasDiagType_t  diag,
              int64_t            m,
              int64_t            k,
              const T*           A,
              int64_t            lda,
              T*                 x,
              int64_t            incx);

// tbsv
template <typename T>
void ref_tbsv(hipblasFillMode_t  uplo,
              hipblasOperation_t transA,
              hipblasDiagType_t  diag,
              int64_t            m,
              int64_t            k,
              const T*           A,
              int64_t            lda,
              T*                 x,
              int64_t            incx);

// tpmv
template <typename T>
void ref_tpmv(hipblasFillMode_t  uplo,
              hipblasOperation_t transA,
              hipblasDiagType_t  diag,
              int64_t            m,
              const T*           A,
              T*                 x,
              int64_t            incx);

// tpsv
template <typename T>
void ref_tpsv(hipblasFillMode_t  uplo,
              hipblasOperation_t transA,
              hipblasDiagType_t  diag,
              int64_t            n,
              const T*           AP,
              T*                 x,
              int64_t            incx);

// trmv
template <typename T>
void ref_trmv(hipblasFillMode_t  uplo,
              hipblasOperation_t transA,
              hipblasDiagType_t  diag,
              int64_t            m,
              const T*           A,
              int64_t            lda,
              T*                 x,
              int64_t            incx);

// trsv
template <typename T>
void ref_trsv(hipblasHandle_t    handle,
              hipblasFillMode_t  uplo,
              hipblasOperation_t transA,
              hipblasDiagType_t  diag,
              int64_t            m,
              const T*           A,
              int64_t            lda,
              T*                 x,
              int64_t            incx);

/*
 * ===========================================================================
 *    BLAS 3
 * ===========================================================================
 */

// herk
template <typename T, typename U>
void ref_herk(hipblasFillMode_t  uplo,
              hipblasOperation_t transA,
              int64_t            n,
              int64_t            k,
              U                  alpha,
              T*                 A,
              int64_t            lda,
              U                  beta,
              T*                 C,
              int64_t            ldc);

// herkx
template <typename T, typename U>
void ref_herkx(hipblasFillMode_t  uplo,
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
               int64_t            ldc);

// her2k
template <typename T, typename U>
void ref_her2k(hipblasFillMode_t  uplo,
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
               int64_t            ldc);

// geam
template <typename T>
void ref_geam(hipblasOperation_t transa,
              hipblasOperation_t transb,
              int64_t            m,
              int64_t            n,
              T*                 alpha,
              T*                 A,
              int64_t            lda,
              T*                 beta,
              T*                 B,
              int64_t            ldb,
              T*                 C,
              int64_t            ldc);

// gemm
template <typename Ti, typename To = Ti, typename Tc = To>
void ref_gemm(hipblasOperation_t transA,
              hipblasOperation_t transB,
              int64_t            m,
              int64_t            n,
              int64_t            k,
              Tc                 alpha,
              Ti*                A,
              int64_t            lda,
              Ti*                B,
              int64_t            ldb,
              Tc                 beta,
              To*                C,
              int64_t            ldc);

// dgmm
template <typename T>
void ref_dgmm(hipblasSideMode_t side,
              int64_t           M,
              int64_t           N,
              const T*          dA,
              int64_t           lda,
              const T*          dx,
              int64_t           incx,
              T*                dC,
              int64_t           ldc);

// hemm
template <typename T>
void ref_hemm(hipblasSideMode_t side,
              hipblasFillMode_t uplo,
              int64_t           m,
              int64_t           n,
              T                 alpha,
              T*                A,
              int64_t           lda,
              T*                B,
              int64_t           ldb,
              T                 beta,
              T*                C,
              int64_t           ldc);

// symm
template <typename T>
void ref_symm(hipblasSideMode_t side,
              hipblasFillMode_t uplo,
              int64_t           m,
              int64_t           n,
              T                 alpha,
              T*                A,
              int64_t           lda,
              T*                B,
              int64_t           ldb,
              T                 beta,
              T*                C,
              int64_t           ldc);

// syrk
template <typename T>
void ref_syrk(hipblasFillMode_t  uplo,
              hipblasOperation_t transA,
              int64_t            n,
              int64_t            k,
              T                  alpha,
              T*                 A,
              int64_t            lda,
              T                  beta,
              T*                 C,
              int64_t            ldc);

// syrk_ex
template <typename T, typename U = T, typename Tc = U>
void ref_syrk_ex(hipblasFillMode_t  uplo,
                 hipblasOperation_t transA,
                 int64_t            n,
                 int64_t            k,
                 Tc                 alpha,
                 const T*           A,
                 int64_t            lda,
                 Tc                 beta,
                 U*                 C,
                 int64_t            ldc);

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
                 int64_t            ldc);

// syr2k
template <typename T>
void ref_syr2k(hipblasFillMode_t  uplo,
               hipblasOperation_t transA,
               int64_t            n,
               int64_t            k,
               T                  alpha,
               T*                 A,
               int64_t            lda,
               T*                 B,
               int64_t            ldb,
               T                  beta,
               T*                 C,
               int64_t            ldc);

// syrkx
template <typename T>
void ref_syrkx(hipblasFillMode_t  uplo,
               hipblasOperation_t transA,
               int64_t            n,
               int64_t            k,
               T                  alpha,
               T*                 A,
               int64_t            lda,
               T*                 B,
               int64_t            ldb,
               T                  beta,
               T*                 C,
               int64_t            ldc);

// trsm
template <typename T>
inline void ref_trsm(hipblasSideMode_t  side,
                     hipblasFillMode_t  uplo,
                     hipblasOperation_t transA,
                     hipblasDiagType_t  diag,
                     int64_t            m,
                     int64_t            n,
                     T                  alpha,
                     const T*           A,
                     int64_t            lda,
                     T*                 B,
                     int64_t            ldb);

// trsm
template <>
inline void ref_trsm<float>(hipblasSideMode_t  side,
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
inline void ref_trsm<double>(hipblasSideMode_t  side,
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
inline void ref_trsm<std::complex<float>>(hipblasSideMode_t          side,
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
inline void ref_trsm<std::complex<double>>(hipblasSideMode_t           side,
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

// trtri
template <typename T>
void ref_trtri(char uplo, char diag, int64_t n, T* A, int64_t lda);

// trmm
template <typename T>
void ref_trmm(hipblasSideMode_t  side,
              hipblasFillMode_t  uplo,
              hipblasOperation_t transA,
              hipblasDiagType_t  diag,
              int64_t            m,
              int64_t            n,
              T                  alpha,
              const T*           A,
              int64_t            lda,
              T*                 B,
              int64_t            ldb);

/*
 * ===========================================================================
 *    LAPACK OR OTHER
 * ===========================================================================
 */

#ifdef __HIP_PLATFORM_SOLVER__

// potrf
template <typename T>
int64_t ref_potrf(char uplo, int64_t m, T* A, int64_t lda);

template <typename T>
int64_t ref_getrf(int64_t m, int64_t n, T* A, int64_t lda, int64_t* ipiv);

template <typename T>
int64_t ref_getrs(
    char trans, int64_t n, int64_t nrhs, T* A, int64_t lda, int64_t* ipiv, T* B, int64_t ldb);

template <typename T>
int64_t ref_getri(int64_t n, T* A, int64_t lda, int64_t* ipiv, T* work, int64_t lwork);

template <typename T>
int64_t ref_geqrf(int64_t m, int64_t n, T* A, int64_t lda, T* tau, T* work, int64_t lwork);

template <typename T>
int64_t ref_gels(char    trans,
                 int64_t m,
                 int64_t n,
                 int64_t nrhs,
                 T*      A,
                 int64_t lda,
                 T*      B,
                 int64_t ldb,
                 T*      work,
                 int64_t lwork);

#endif

/* ============================================================================================ */
