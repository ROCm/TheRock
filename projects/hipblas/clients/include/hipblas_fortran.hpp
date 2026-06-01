/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef _HIPBLAS_FORTRAN_HPP
#define _HIPBLAS_FORTRAN_HPP

/*!\file
 *  This file interfaces with our Fortran BLAS interface.
 */

/*
 * ============================================================================
 *     Fortran functions
 * ============================================================================
 */

extern "C" {

#include "hipblas_fortran.h.in"
#define HIPBLAS_INTERNAL_ILP64 1
#include "hipblas_fortran.h.in"
#undef HIPBLAS_INTERNAL_ILP64

/* ==========
 *    Aux
 * ========== */
hipblasStatus_t
    hipblasSetVectorFortran(int n, int elemSize, const void* x, int incx, void* y, int incy);

hipblasStatus_t
    hipblasGetVectorFortran(int n, int elemSize, const void* x, int incx, void* y, int incy);

hipblasStatus_t hipblasSetMatrixFortran(
    int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb);

hipblasStatus_t hipblasGetMatrixFortran(
    int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb);

hipblasStatus_t hipblasSetVectorAsyncFortran(
    int n, int elemSize, const void* x, int incx, void* y, int incy, hipStream_t stream);

hipblasStatus_t hipblasGetVectorAsyncFortran(
    int n, int elemSize, const void* x, int incx, void* y, int incy, hipStream_t stream);

hipblasStatus_t hipblasSetMatrixAsyncFortran(
    int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, hipStream_t stream);

hipblasStatus_t hipblasGetMatrixAsyncFortran(
    int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, hipStream_t stream);

hipblasStatus_t hipblasSetAtomicsModeFortran(hipblasHandle_t      handle,
                                             hipblasAtomicsMode_t atomics_mode);

hipblasStatus_t hipblasGetAtomicsModeFortran(hipblasHandle_t       handle,
                                             hipblasAtomicsMode_t* atomics_mode);

/* ==========
 *    L3
 * ========== */

// trtri
hipblasStatus_t hipblasStrtriFortran(hipblasHandle_t   handle,
                                     hipblasFillMode_t uplo,
                                     hipblasDiagType_t diag,
                                     int               n,
                                     const float*      A,
                                     int               lda,
                                     float*            invA,
                                     int               ldinvA);

hipblasStatus_t hipblasDtrtriFortran(hipblasHandle_t   handle,
                                     hipblasFillMode_t uplo,
                                     hipblasDiagType_t diag,
                                     int               n,
                                     const double*     A,
                                     int               lda,
                                     double*           invA,
                                     int               ldinvA);

hipblasStatus_t hipblasCtrtriFortran(hipblasHandle_t   handle,
                                     hipblasFillMode_t uplo,
                                     hipblasDiagType_t diag,
                                     int               n,
                                     const hipComplex* A,
                                     int               lda,
                                     hipComplex*       invA,
                                     int               ldinvA);

hipblasStatus_t hipblasZtrtriFortran(hipblasHandle_t         handle,
                                     hipblasFillMode_t       uplo,
                                     hipblasDiagType_t       diag,
                                     int                     n,
                                     const hipDoubleComplex* A,
                                     int                     lda,
                                     hipDoubleComplex*       invA,
                                     int                     ldinvA);

// trtri_batched
hipblasStatus_t hipblasStrtriBatchedFortran(hipblasHandle_t    handle,
                                            hipblasFillMode_t  uplo,
                                            hipblasDiagType_t  diag,
                                            int                n,
                                            const float* const A[],
                                            int                lda,
                                            float*             invA[],
                                            int                ldinvA,
                                            int                batch_count);

hipblasStatus_t hipblasDtrtriBatchedFortran(hipblasHandle_t     handle,
                                            hipblasFillMode_t   uplo,
                                            hipblasDiagType_t   diag,
                                            int                 n,
                                            const double* const A[],
                                            int                 lda,
                                            double*             invA[],
                                            int                 ldinvA,
                                            int                 batch_count);

hipblasStatus_t hipblasCtrtriBatchedFortran(hipblasHandle_t         handle,
                                            hipblasFillMode_t       uplo,
                                            hipblasDiagType_t       diag,
                                            int                     n,
                                            const hipComplex* const A[],
                                            int                     lda,
                                            hipComplex*             invA[],
                                            int                     ldinvA,
                                            int                     batch_count);

hipblasStatus_t hipblasZtrtriBatchedFortran(hipblasHandle_t               handle,
                                            hipblasFillMode_t             uplo,
                                            hipblasDiagType_t             diag,
                                            int                           n,
                                            const hipDoubleComplex* const A[],
                                            int                           lda,
                                            hipDoubleComplex*             invA[],
                                            int                           ldinvA,
                                            int                           batch_count);

// trtri_strided_batched
hipblasStatus_t hipblasStrtriStridedBatchedFortran(hipblasHandle_t   handle,
                                                   hipblasFillMode_t uplo,
                                                   hipblasDiagType_t diag,
                                                   int               n,
                                                   const float*      A,
                                                   int               lda,
                                                   hipblasStride     stride_A,
                                                   float*            invA,
                                                   int               ldinvA,
                                                   hipblasStride     stride_invA,
                                                   int               batch_count);

hipblasStatus_t hipblasDtrtriStridedBatchedFortran(hipblasHandle_t   handle,
                                                   hipblasFillMode_t uplo,
                                                   hipblasDiagType_t diag,
                                                   int               n,
                                                   const double*     A,
                                                   int               lda,
                                                   hipblasStride     stride_A,
                                                   double*           invA,
                                                   int               ldinvA,
                                                   hipblasStride     stride_invA,
                                                   int               batch_count);

hipblasStatus_t hipblasCtrtriStridedBatchedFortran(hipblasHandle_t   handle,
                                                   hipblasFillMode_t uplo,
                                                   hipblasDiagType_t diag,
                                                   int               n,
                                                   const hipComplex* A,
                                                   int               lda,
                                                   hipblasStride     stride_A,
                                                   hipComplex*       invA,
                                                   int               ldinvA,
                                                   hipblasStride     stride_invA,
                                                   int               batch_count);

hipblasStatus_t hipblasZtrtriStridedBatchedFortran(hipblasHandle_t         handle,
                                                   hipblasFillMode_t       uplo,
                                                   hipblasDiagType_t       diag,
                                                   int                     n,
                                                   const hipDoubleComplex* A,
                                                   int                     lda,
                                                   hipblasStride           stride_A,
                                                   hipDoubleComplex*       invA,
                                                   int                     ldinvA,
                                                   hipblasStride           stride_invA,
                                                   int                     batch_count);

// trsm_ex
hipblasStatus_t hipblasTrsmExFortran(hipblasHandle_t    handle,
                                     hipblasSideMode_t  side,
                                     hipblasFillMode_t  uplo,
                                     hipblasOperation_t transA,
                                     hipblasDiagType_t  diag,
                                     int                m,
                                     int                n,
                                     const void*        alpha,
                                     void*              A,
                                     int                lda,
                                     void*              B,
                                     int                ldb,
                                     const void*        invA,
                                     int                invA_size,
                                     hipDataType        compute_type);

hipblasStatus_t hipblasTrsmBatchedExFortran(hipblasHandle_t    handle,
                                            hipblasSideMode_t  side,
                                            hipblasFillMode_t  uplo,
                                            hipblasOperation_t transA,
                                            hipblasDiagType_t  diag,
                                            int                m,
                                            int                n,
                                            const void*        alpha,
                                            void*              A,
                                            int                lda,
                                            void*              B,
                                            int                ldb,
                                            int                batch_count,
                                            const void*        invA,
                                            int                invA_size,
                                            hipDataType        compute_type);

hipblasStatus_t hipblasTrsmStridedBatchedExFortran(hipblasHandle_t    handle,
                                                   hipblasSideMode_t  side,
                                                   hipblasFillMode_t  uplo,
                                                   hipblasOperation_t transA,
                                                   hipblasDiagType_t  diag,
                                                   int                m,
                                                   int                n,
                                                   const void*        alpha,
                                                   void*              A,
                                                   int                lda,
                                                   hipblasStride      stride_A,
                                                   void*              B,
                                                   int                ldb,
                                                   hipblasStride      stride_B,
                                                   int                batch_count,
                                                   const void*        invA,
                                                   int                invA_size,
                                                   hipblasStride      stride_invA,
                                                   hipDataType        compute_type);

// // syrk_ex
// hipblasStatus_t hipblasCsyrkExFortran(hipblasHandle_t       handle,
//                                       hipblasFillMode_t     uplo,
//                                       hipblasOperation_t    trans,
//                                       int                   n,
//                                       int                   k,
//                                       const hipComplex* alpha,
//                                       const void*           A,
//                                       hipDataType     Atype,
//                                       int                   lda,
//                                       const hipComplex* beta,
//                                       hipComplex*       C,
//                                       hipDataType     Ctype,
//                                       int                   ldc);

// // herk_ex
// hipblasStatus_t hipblasCherkExFortran(hipblasHandle_t    handle,
//                                       hipblasFillMode_t  uplo,
//                                       hipblasOperation_t trans,
//                                       int                n,
//                                       int                k,
//                                       const float*       alpha,
//                                       const void*        A,
//                                       hipDataType  Atype,
//                                       int                lda,
//                                       const float*       beta,
//                                       hipComplex*    C,
//                                       hipDataType  Ctype,
//                                       int                ldc);

/* ==========
 *    Solver
 * ========== */

// getrf
hipblasStatus_t hipblasSgetrfFortran(
    hipblasHandle_t handle, const int n, float* A, const int lda, int* ipiv, int* info);

hipblasStatus_t hipblasDgetrfFortran(
    hipblasHandle_t handle, const int n, double* A, const int lda, int* ipiv, int* info);

hipblasStatus_t hipblasCgetrfFortran(
    hipblasHandle_t handle, const int n, hipComplex* A, const int lda, int* ipiv, int* info);

hipblasStatus_t hipblasZgetrfFortran(
    hipblasHandle_t handle, const int n, hipDoubleComplex* A, const int lda, int* ipiv, int* info);

// getrf_batched
hipblasStatus_t hipblasSgetrfBatchedFortran(hipblasHandle_t handle,
                                            const int       n,
                                            float* const    A[],
                                            const int       lda,
                                            int*            ipiv,
                                            int*            info,
                                            const int       batch_count);

hipblasStatus_t hipblasDgetrfBatchedFortran(hipblasHandle_t handle,
                                            const int       n,
                                            double* const   A[],
                                            const int       lda,
                                            int*            ipiv,
                                            int*            info,
                                            const int       batch_count);

hipblasStatus_t hipblasCgetrfBatchedFortran(hipblasHandle_t   handle,
                                            const int         n,
                                            hipComplex* const A[],
                                            const int         lda,
                                            int*              ipiv,
                                            int*              info,
                                            const int         batch_count);

hipblasStatus_t hipblasZgetrfBatchedFortran(hipblasHandle_t         handle,
                                            const int               n,
                                            hipDoubleComplex* const A[],
                                            const int               lda,
                                            int*                    ipiv,
                                            int*                    info,
                                            const int               batch_count);

// getrf_strided_batched
hipblasStatus_t hipblasSgetrfStridedBatchedFortran(hipblasHandle_t     handle,
                                                   const int           n,
                                                   float*              A,
                                                   const int           lda,
                                                   const hipblasStride stride_A,
                                                   int*                ipiv,
                                                   const hipblasStride stride_P,
                                                   int*                info,
                                                   const int           batch_count);

hipblasStatus_t hipblasDgetrfStridedBatchedFortran(hipblasHandle_t     handle,
                                                   const int           n,
                                                   double*             A,
                                                   const int           lda,
                                                   const hipblasStride stride_A,
                                                   int*                ipiv,
                                                   const hipblasStride stride_P,
                                                   int*                info,
                                                   const int           batch_count);

hipblasStatus_t hipblasCgetrfStridedBatchedFortran(hipblasHandle_t     handle,
                                                   const int           n,
                                                   hipComplex*         A,
                                                   const int           lda,
                                                   const hipblasStride stride_A,
                                                   int*                ipiv,
                                                   const hipblasStride stride_P,
                                                   int*                info,
                                                   const int           batch_count);

hipblasStatus_t hipblasZgetrfStridedBatchedFortran(hipblasHandle_t     handle,
                                                   const int           n,
                                                   hipDoubleComplex*   A,
                                                   const int           lda,
                                                   const hipblasStride stride_A,
                                                   int*                ipiv,
                                                   const hipblasStride stride_P,
                                                   int*                info,
                                                   const int           batch_count);

// getrs
hipblasStatus_t hipblasSgetrsFortran(hipblasHandle_t          handle,
                                     const hipblasOperation_t trans,
                                     const int                n,
                                     const int                nrhs,
                                     float*                   A,
                                     const int                lda,
                                     const int*               ipiv,
                                     float*                   B,
                                     const int                ldb,
                                     int*                     info);

hipblasStatus_t hipblasDgetrsFortran(hipblasHandle_t          handle,
                                     const hipblasOperation_t trans,
                                     const int                n,
                                     const int                nrhs,
                                     double*                  A,
                                     const int                lda,
                                     const int*               ipiv,
                                     double*                  B,
                                     const int                ldb,
                                     int*                     info);

hipblasStatus_t hipblasCgetrsFortran(hipblasHandle_t          handle,
                                     const hipblasOperation_t trans,
                                     const int                n,
                                     const int                nrhs,
                                     hipComplex*              A,
                                     const int                lda,
                                     const int*               ipiv,
                                     hipComplex*              B,
                                     const int                ldb,
                                     int*                     info);

hipblasStatus_t hipblasZgetrsFortran(hipblasHandle_t          handle,
                                     const hipblasOperation_t trans,
                                     const int                n,
                                     const int                nrhs,
                                     hipDoubleComplex*        A,
                                     const int                lda,
                                     const int*               ipiv,
                                     hipDoubleComplex*        B,
                                     const int                ldb,
                                     int*                     info);

// getrs_batched
hipblasStatus_t hipblasSgetrsBatchedFortran(hipblasHandle_t          handle,
                                            const hipblasOperation_t trans,
                                            const int                n,
                                            const int                nrhs,
                                            float* const             A[],
                                            const int                lda,
                                            const int*               ipiv,
                                            float* const             B[],
                                            const int                ldb,
                                            int*                     info,
                                            const int                batch_count);

hipblasStatus_t hipblasDgetrsBatchedFortran(hipblasHandle_t          handle,
                                            const hipblasOperation_t trans,
                                            const int                n,
                                            const int                nrhs,
                                            double* const            A[],
                                            const int                lda,
                                            const int*               ipiv,
                                            double* const            B[],
                                            const int                ldb,
                                            int*                     info,
                                            const int                batch_count);

hipblasStatus_t hipblasCgetrsBatchedFortran(hipblasHandle_t          handle,
                                            const hipblasOperation_t trans,
                                            const int                n,
                                            const int                nrhs,
                                            hipComplex* const        A[],
                                            const int                lda,
                                            const int*               ipiv,
                                            hipComplex* const        B[],
                                            const int                ldb,
                                            int*                     info,
                                            const int                batch_count);

hipblasStatus_t hipblasZgetrsBatchedFortran(hipblasHandle_t          handle,
                                            const hipblasOperation_t trans,
                                            const int                n,
                                            const int                nrhs,
                                            hipDoubleComplex* const  A[],
                                            const int                lda,
                                            const int*               ipiv,
                                            hipDoubleComplex* const  B[],
                                            const int                ldb,
                                            int*                     info,
                                            const int                batch_count);

// getrs_strided_batched
hipblasStatus_t hipblasSgetrsStridedBatchedFortran(hipblasHandle_t          handle,
                                                   const hipblasOperation_t trans,
                                                   const int                n,
                                                   const int                nrhs,
                                                   float*                   A,
                                                   const int                lda,
                                                   const hipblasStride      stride_A,
                                                   const int*               ipiv,
                                                   const hipblasStride      stride_P,
                                                   float*                   B,
                                                   const int                ldb,
                                                   const hipblasStride      stride_B,
                                                   int*                     info,
                                                   const int                batch_count);

hipblasStatus_t hipblasDgetrsStridedBatchedFortran(hipblasHandle_t          handle,
                                                   const hipblasOperation_t trans,
                                                   const int                n,
                                                   const int                nrhs,
                                                   double*                  A,
                                                   const int                lda,
                                                   const hipblasStride      stride_A,
                                                   const int*               ipiv,
                                                   const hipblasStride      stride_P,
                                                   double*                  B,
                                                   const int                ldb,
                                                   const hipblasStride      stride_B,
                                                   int*                     info,
                                                   const int                batch_count);

hipblasStatus_t hipblasCgetrsStridedBatchedFortran(hipblasHandle_t          handle,
                                                   const hipblasOperation_t trans,
                                                   const int                n,
                                                   const int                nrhs,
                                                   hipComplex*              A,
                                                   const int                lda,
                                                   const hipblasStride      stride_A,
                                                   const int*               ipiv,
                                                   const hipblasStride      stride_P,
                                                   hipComplex*              B,
                                                   const int                ldb,
                                                   const hipblasStride      stride_B,
                                                   int*                     info,
                                                   const int                batch_count);

hipblasStatus_t hipblasZgetrsStridedBatchedFortran(hipblasHandle_t          handle,
                                                   const hipblasOperation_t trans,
                                                   const int                n,
                                                   const int                nrhs,
                                                   hipDoubleComplex*        A,
                                                   const int                lda,
                                                   const hipblasStride      stride_A,
                                                   const int*               ipiv,
                                                   const hipblasStride      stride_P,
                                                   hipDoubleComplex*        B,
                                                   const int                ldb,
                                                   const hipblasStride      stride_B,
                                                   int*                     info,
                                                   const int                batch_count);

// getri_batched
hipblasStatus_t hipblasSgetriBatchedFortran(hipblasHandle_t handle,
                                            const int       n,
                                            float* const    A[],
                                            const int       lda,
                                            int*            ipiv,
                                            float* const    C[],
                                            const int       ldc,
                                            int*            info,
                                            const int       batch_count);

hipblasStatus_t hipblasDgetriBatchedFortran(hipblasHandle_t handle,
                                            const int       n,
                                            double* const   A[],
                                            const int       lda,
                                            int*            ipiv,
                                            double* const   C[],
                                            const int       ldc,
                                            int*            info,
                                            const int       batch_count);

hipblasStatus_t hipblasCgetriBatchedFortran(hipblasHandle_t   handle,
                                            const int         n,
                                            hipComplex* const A[],
                                            const int         lda,
                                            int*              ipiv,
                                            hipComplex* const C[],
                                            const int         ldc,
                                            int*              info,
                                            const int         batch_count);

hipblasStatus_t hipblasZgetriBatchedFortran(hipblasHandle_t         handle,
                                            const int               n,
                                            hipDoubleComplex* const A[],
                                            const int               lda,
                                            int*                    ipiv,
                                            hipDoubleComplex* const C[],
                                            const int               ldc,
                                            int*                    info,
                                            const int               batch_count);

// geqrf
hipblasStatus_t hipblasSgeqrfFortran(hipblasHandle_t handle,
                                     const int       m,
                                     const int       n,
                                     float*          A,
                                     const int       lda,
                                     float*          tau,
                                     int*            info);

hipblasStatus_t hipblasDgeqrfFortran(hipblasHandle_t handle,
                                     const int       m,
                                     const int       n,
                                     double*         A,
                                     const int       lda,
                                     double*         tau,
                                     int*            info);

hipblasStatus_t hipblasCgeqrfFortran(hipblasHandle_t handle,
                                     const int       m,
                                     const int       n,
                                     hipComplex*     A,
                                     const int       lda,
                                     hipComplex*     tau,
                                     int*            info);

hipblasStatus_t hipblasZgeqrfFortran(hipblasHandle_t   handle,
                                     const int         m,
                                     const int         n,
                                     hipDoubleComplex* A,
                                     const int         lda,
                                     hipDoubleComplex* tau,
                                     int*              info);

// geqrf_batched
hipblasStatus_t hipblasSgeqrfBatchedFortran(hipblasHandle_t handle,
                                            const int       m,
                                            const int       n,
                                            float* const    A[],
                                            const int       lda,
                                            float* const    tau[],
                                            int*            info,
                                            const int       batch_count);

hipblasStatus_t hipblasDgeqrfBatchedFortran(hipblasHandle_t handle,
                                            const int       m,
                                            const int       n,
                                            double* const   A[],
                                            const int       lda,
                                            double* const   tau[],
                                            int*            info,
                                            const int       batch_count);

hipblasStatus_t hipblasCgeqrfBatchedFortran(hipblasHandle_t   handle,
                                            const int         m,
                                            const int         n,
                                            hipComplex* const A[],
                                            const int         lda,
                                            hipComplex* const tau[],
                                            int*              info,
                                            const int         batch_count);

hipblasStatus_t hipblasZgeqrfBatchedFortran(hipblasHandle_t         handle,
                                            const int               m,
                                            const int               n,
                                            hipDoubleComplex* const A[],
                                            const int               lda,
                                            hipDoubleComplex* const tau[],
                                            int*                    info,
                                            const int               batch_count);

// geqrf_strided_batched
hipblasStatus_t hipblasSgeqrfStridedBatchedFortran(hipblasHandle_t     handle,
                                                   const int           m,
                                                   const int           n,
                                                   float*              A,
                                                   const int           lda,
                                                   const hipblasStride stride_A,
                                                   float*              tau,
                                                   const hipblasStride stride_T,
                                                   int*                info,
                                                   const int           batch_count);

hipblasStatus_t hipblasDgeqrfStridedBatchedFortran(hipblasHandle_t     handle,
                                                   const int           m,
                                                   const int           n,
                                                   double*             A,
                                                   const int           lda,
                                                   const hipblasStride stride_A,
                                                   double*             tau,
                                                   const hipblasStride stride_T,
                                                   int*                info,
                                                   const int           batch_count);

hipblasStatus_t hipblasCgeqrfStridedBatchedFortran(hipblasHandle_t     handle,
                                                   const int           m,
                                                   const int           n,
                                                   hipComplex*         A,
                                                   const int           lda,
                                                   const hipblasStride stride_A,
                                                   hipComplex*         tau,
                                                   const hipblasStride stride_T,
                                                   int*                info,
                                                   const int           batch_count);

hipblasStatus_t hipblasZgeqrfStridedBatchedFortran(hipblasHandle_t     handle,
                                                   const int           m,
                                                   const int           n,
                                                   hipDoubleComplex*   A,
                                                   const int           lda,
                                                   const hipblasStride stride_A,
                                                   hipDoubleComplex*   tau,
                                                   const hipblasStride stride_T,
                                                   int*                info,
                                                   const int           batch_count);

// gels
hipblasStatus_t hipblasSgelsFortran(hipblasHandle_t    handle,
                                    hipblasOperation_t trans,
                                    const int          m,
                                    const int          n,
                                    const int          nrhs,
                                    float*             A,
                                    const int          lda,
                                    float*             B,
                                    const int          ldb,
                                    int*               info,
                                    int*               deviceInfo);

hipblasStatus_t hipblasDgelsFortran(hipblasHandle_t    handle,
                                    hipblasOperation_t trans,
                                    const int          m,
                                    const int          n,
                                    const int          nrhs,
                                    double*            A,
                                    const int          lda,
                                    double*            B,
                                    const int          ldb,
                                    int*               info,
                                    int*               deviceInfo);

hipblasStatus_t hipblasCgelsFortran(hipblasHandle_t    handle,
                                    hipblasOperation_t trans,
                                    const int          m,
                                    const int          n,
                                    const int          nrhs,
                                    hipComplex*        A,
                                    const int          lda,
                                    hipComplex*        B,
                                    const int          ldb,
                                    int*               info,
                                    int*               deviceInfo);

hipblasStatus_t hipblasZgelsFortran(hipblasHandle_t    handle,
                                    hipblasOperation_t trans,
                                    const int          m,
                                    const int          n,
                                    const int          nrhs,
                                    hipDoubleComplex*  A,
                                    const int          lda,
                                    hipDoubleComplex*  B,
                                    const int          ldb,
                                    int*               info,
                                    int*               deviceInfo);

// gelsBatched
hipblasStatus_t hipblasSgelsBatchedFortran(hipblasHandle_t    handle,
                                           hipblasOperation_t trans,
                                           const int          m,
                                           const int          n,
                                           const int          nrhs,
                                           float* const       A[],
                                           const int          lda,
                                           float* const       B[],
                                           const int          ldb,
                                           int*               info,
                                           int*               deviceInfo,
                                           const int          batchCount);

hipblasStatus_t hipblasDgelsBatchedFortran(hipblasHandle_t    handle,
                                           hipblasOperation_t trans,
                                           const int          m,
                                           const int          n,
                                           const int          nrhs,
                                           double* const      A[],
                                           const int          lda,
                                           double* const      B[],
                                           const int          ldb,
                                           int*               info,
                                           int*               deviceInfo,
                                           const int          batchCount);

hipblasStatus_t hipblasCgelsBatchedFortran(hipblasHandle_t    handle,
                                           hipblasOperation_t trans,
                                           const int          m,
                                           const int          n,
                                           const int          nrhs,
                                           hipComplex* const  A[],
                                           const int          lda,
                                           hipComplex* const  B[],
                                           const int          ldb,
                                           int*               info,
                                           int*               deviceInfo,
                                           const int          batchCount);

hipblasStatus_t hipblasZgelsBatchedFortran(hipblasHandle_t         handle,
                                           hipblasOperation_t      trans,
                                           const int               m,
                                           const int               n,
                                           const int               nrhs,
                                           hipDoubleComplex* const A[],
                                           const int               lda,
                                           hipDoubleComplex* const B[],
                                           const int               ldb,
                                           int*                    info,
                                           int*                    deviceInfo,
                                           const int               batchCount);

// gelsStridedBatched
hipblasStatus_t hipblasSgelsStridedBatchedFortran(hipblasHandle_t     handle,
                                                  hipblasOperation_t  trans,
                                                  const int           m,
                                                  const int           n,
                                                  const int           nrhs,
                                                  float*              A,
                                                  const int           lda,
                                                  const hipblasStride strideA,
                                                  float*              B,
                                                  const int           ldb,
                                                  const hipblasStride strideB,
                                                  int*                info,
                                                  int*                deviceInfo,
                                                  const int           batchCount);

hipblasStatus_t hipblasDgelsStridedBatchedFortran(hipblasHandle_t     handle,
                                                  hipblasOperation_t  trans,
                                                  const int           m,
                                                  const int           n,
                                                  const int           nrhs,
                                                  double*             A,
                                                  const int           lda,
                                                  const hipblasStride strideA,
                                                  double*             B,
                                                  const int           ldb,
                                                  const hipblasStride strideB,
                                                  int*                info,
                                                  int*                deviceInfo,
                                                  const int           batchCount);

hipblasStatus_t hipblasCgelsStridedBatchedFortran(hipblasHandle_t     handle,
                                                  hipblasOperation_t  trans,
                                                  const int           m,
                                                  const int           n,
                                                  const int           nrhs,
                                                  hipComplex*         A,
                                                  const int           lda,
                                                  const hipblasStride strideA,
                                                  hipComplex*         B,
                                                  const int           ldb,
                                                  const hipblasStride strideB,
                                                  int*                info,
                                                  int*                deviceInfo,
                                                  const int           batchCount);

hipblasStatus_t hipblasZgelsStridedBatchedFortran(hipblasHandle_t     handle,
                                                  hipblasOperation_t  trans,
                                                  const int           m,
                                                  const int           n,
                                                  const int           nrhs,
                                                  hipDoubleComplex*   A,
                                                  const int           lda,
                                                  const hipblasStride strideA,
                                                  hipDoubleComplex*   B,
                                                  const int           ldb,
                                                  const hipblasStride strideB,
                                                  int*                info,
                                                  int*                deviceInfo,
                                                  const int           batchCount);
}

#endif
