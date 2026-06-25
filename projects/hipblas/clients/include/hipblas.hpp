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
#ifndef _HIPBLAS_HPP_
#define _HIPBLAS_HPP_

/* library headers */
#include "hipblas.h"
#include "type_utils.h"

#ifdef HIPBLAS_FORTRAN_CLIENTS
#include "hipblas_fortran.hpp"
#else
#include "hipblas_no_fortran.hpp"
#endif

#if not defined(__clang_major__)
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

#define GET_MACRO(_1, _2, _3, _4, _5, NAME, ...) NAME

#define MAP2CF(...) GET_MACRO(__VA_ARGS__, MAP2CF5, MAP2CF4, MAP2CF3, dum2, dum1)(__VA_ARGS__)
// dual API C and FORTRAN
#define MAP2CF_D64(...) \
    GET_MACRO(__VA_ARGS__, MAP2DCF5, MAP2DCF4, MAP2DCF3, dum2, dum1)(__VA_ARGS__)

#if defined(HIPBLAS_FORTRAN_CLIENTS)
#define MAP2CF3(FN, A, PFN)  \
    template <>              \
    auto FN<A, false> = PFN; \
    template <>              \
    auto FN<A, true> = PFN##Fortran
#define MAP2CF4(FN, A, B, PFN)  \
    template <>                 \
    auto FN<A, B, false> = PFN; \
    template <>                 \
    auto FN<A, B, true> = PFN##Fortran
#define MAP2CF5(FN, A, B, C, PFN)  \
    template <>                    \
    auto FN<A, B, C, false> = PFN; \
    template <>                    \
    auto FN<A, B, C, true> = PFN##Fortran
// dual API C and FORTRAN
#define MAP2DCF3(FN, A, PFN)           \
    template <>                        \
    auto FN<A, false> = PFN;           \
    template <>                        \
    auto FN<A, true> = PFN##Fortran;   \
    template <>                        \
    auto FN##_64<A, false> = PFN##_64; \
    template <>                        \
    auto FN##_64<A, true> = PFN##_64Fortran
#define MAP2DCF4(FN, A, B, PFN)           \
    template <>                           \
    auto FN<A, B, false> = PFN;           \
    template <>                           \
    auto FN<A, B, true> = PFN##Fortran;   \
    template <>                           \
    auto FN##_64<A, B, false> = PFN##_64; \
    template <>                           \
    auto FN##_64<A, B, true> = PFN##_64Fortran
#define MAP2DCF5(FN, A, B, C, PFN)           \
    template <>                              \
    auto FN<A, B, C, false> = PFN;           \
    template <>                              \
    auto FN<A, B, C, true> = PFN##Fortran;   \
    template <>                              \
    auto FN##_64<A, B, C, false> = PFN##_64; \
    template <>                              \
    auto FN##_64<A, B, C, true> = PFN##_64Fortran
#else
// mapping fortran and C to C API so we keep tests the same even without FORTRAN clients
#define MAP2CF3(FN, A, PFN)  \
    template <>              \
    auto FN<A, false> = PFN; \
    template <>              \
    auto FN<A, true> = PFN
#define MAP2CF4(FN, A, B, PFN)  \
    template <>                 \
    auto FN<A, B, false> = PFN; \
    template <>                 \
    auto FN<A, B, true> = PFN
#define MAP2CF5(FN, A, B, C, PFN)  \
    template <>                    \
    auto FN<A, B, C, false> = PFN; \
    template <>                    \
    auto FN<A, B, C, true> = PFN
// dual API C and FORTRAN
#define MAP2DCF3(FN, A, PFN)           \
    template <>                        \
    auto FN<A, false> = PFN;           \
    template <>                        \
    auto FN<A, true> = PFN;            \
    template <>                        \
    auto FN##_64<A, false> = PFN##_64; \
    template <>                        \
    auto FN##_64<A, true> = PFN##_64
#define MAP2DCF4(FN, A, B, PFN)           \
    template <>                           \
    auto FN<A, B, false> = PFN;           \
    template <>                           \
    auto FN<A, B, true> = PFN;            \
    template <>                           \
    auto FN##_64<A, B, false> = PFN##_64; \
    template <>                           \
    auto FN##_64<A, B, true> = PFN##_64
#define MAP2DCF5(FN, A, B, C, PFN)           \
    template <>                              \
    auto FN<A, B, C, false> = PFN;           \
    template <>                              \
    auto FN<A, B, C, true> = PFN;            \
    template <>                              \
    auto FN##_64<A, B, C, false> = PFN##_64; \
    template <>                              \
    auto FN##_64<A, B, C, true> = PFN##_64
#endif

namespace
{
    // Scal
    template <typename T, typename U = T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasScal)(hipblasHandle_t                 handle,
                                   int                             n,
                                   const hipblas_internal_type<U>* alpha,
                                   hipblas_internal_type<T>*       x,
                                   int                             incx);

    template <typename T, typename U = T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasScalBatched)(hipblasHandle_t                 handle,
                                          int                             n,
                                          const hipblas_internal_type<U>* alpha,
                                          hipblas_internal_type<T>* const x[],
                                          int                             incx,
                                          int                             batch_count);

    template <typename T, typename U = T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasScalStridedBatched)(hipblasHandle_t                 handle,
                                                 int                             n,
                                                 const hipblas_internal_type<U>* alpha,
                                                 hipblas_internal_type<T>*       x,
                                                 int                             incx,
                                                 hipblasStride                   stridex,
                                                 int                             batch_count);

    template <typename T, typename U = T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasScal_64)(hipblasHandle_t                 handle,
                                      int64_t                         n,
                                      const hipblas_internal_type<U>* alpha,
                                      hipblas_internal_type<T>*       x,
                                      int64_t                         incx);

    template <typename T, typename U = T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasScalBatched_64)(hipblasHandle_t                 handle,
                                             int64_t                         n,
                                             const hipblas_internal_type<U>* alpha,
                                             hipblas_internal_type<T>* const x[],
                                             int64_t                         incx,
                                             int64_t                         batch_count);

    template <typename T, typename U = T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasScalStridedBatched_64)(hipblasHandle_t                 handle,
                                                    int64_t                         n,
                                                    const hipblas_internal_type<U>* alpha,
                                                    hipblas_internal_type<T>*       x,
                                                    int64_t                         incx,
                                                    hipblasStride                   stridex,
                                                    int64_t                         batch_count);

    MAP2CF_D64(hipblasScal, float, float, hipblasSscal);
    MAP2CF_D64(hipblasScal, double, double, hipblasDscal);
    MAP2CF_D64(hipblasScal, std::complex<float>, std::complex<float>, hipblasCscal);
    MAP2CF_D64(hipblasScal, std::complex<double>, std::complex<double>, hipblasZscal);
    MAP2CF_D64(hipblasScal, std::complex<float>, float, hipblasCsscal);
    MAP2CF_D64(hipblasScal, std::complex<double>, double, hipblasZdscal);

    MAP2CF_D64(hipblasScalBatched, float, float, hipblasSscalBatched);
    MAP2CF_D64(hipblasScalBatched, double, double, hipblasDscalBatched);
    MAP2CF_D64(hipblasScalBatched, std::complex<float>, std::complex<float>, hipblasCscalBatched);
    MAP2CF_D64(hipblasScalBatched, std::complex<double>, std::complex<double>, hipblasZscalBatched);
    MAP2CF_D64(hipblasScalBatched, std::complex<float>, float, hipblasCsscalBatched);
    MAP2CF_D64(hipblasScalBatched, std::complex<double>, double, hipblasZdscalBatched);

    MAP2CF_D64(hipblasScalStridedBatched, float, float, hipblasSscalStridedBatched);
    MAP2CF_D64(hipblasScalStridedBatched, double, double, hipblasDscalStridedBatched);
    MAP2CF_D64(hipblasScalStridedBatched,
               std::complex<float>,
               std::complex<float>,
               hipblasCscalStridedBatched);
    MAP2CF_D64(hipblasScalStridedBatched,
               std::complex<double>,
               std::complex<double>,
               hipblasZscalStridedBatched);
    MAP2CF_D64(hipblasScalStridedBatched, std::complex<float>, float, hipblasCsscalStridedBatched);
    MAP2CF_D64(hipblasScalStridedBatched,
               std::complex<double>,
               double,
               hipblasZdscalStridedBatched);

    // Copy
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasCopy)(hipblasHandle_t                 handle,
                                   int                             n,
                                   const hipblas_internal_type<T>* x,
                                   int                             incx,
                                   hipblas_internal_type<T>*       y,
                                   int                             incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasCopyBatched)(hipblasHandle_t                       handle,
                                          int                                   n,
                                          const hipblas_internal_type<T>* const x[],
                                          int                                   incx,
                                          hipblas_internal_type<T>* const       y[],
                                          int                                   incy,
                                          int                                   batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasCopyStridedBatched)(hipblasHandle_t                 handle,
                                                 int                             n,
                                                 const hipblas_internal_type<T>* x,
                                                 int                             incx,
                                                 hipblasStride                   stridex,
                                                 hipblas_internal_type<T>*       y,
                                                 int                             incy,
                                                 hipblasStride                   stridey,
                                                 int                             batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasCopy_64)(hipblasHandle_t                 handle,
                                      int64_t                         n,
                                      const hipblas_internal_type<T>* x,
                                      int64_t                         incx,
                                      hipblas_internal_type<T>*       y,
                                      int64_t                         incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasCopyBatched_64)(hipblasHandle_t                       handle,
                                             int64_t                               n,
                                             const hipblas_internal_type<T>* const x[],
                                             int64_t                               incx,
                                             hipblas_internal_type<T>* const       y[],
                                             int64_t                               incy,
                                             int64_t                               batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasCopyStridedBatched_64)(hipblasHandle_t                 handle,
                                                    int64_t                         n,
                                                    const hipblas_internal_type<T>* x,
                                                    int64_t                         incx,
                                                    hipblasStride                   stridex,
                                                    hipblas_internal_type<T>*       y,
                                                    int64_t                         incy,
                                                    hipblasStride                   stridey,
                                                    int64_t                         batch_count);

    MAP2CF_D64(hipblasCopy, float, hipblasScopy);
    MAP2CF_D64(hipblasCopy, double, hipblasDcopy);
    MAP2CF_D64(hipblasCopy, std::complex<float>, hipblasCcopy);
    MAP2CF_D64(hipblasCopy, std::complex<double>, hipblasZcopy);

    MAP2CF_D64(hipblasCopyBatched, float, hipblasScopyBatched);
    MAP2CF_D64(hipblasCopyBatched, double, hipblasDcopyBatched);
    MAP2CF_D64(hipblasCopyBatched, std::complex<float>, hipblasCcopyBatched);
    MAP2CF_D64(hipblasCopyBatched, std::complex<double>, hipblasZcopyBatched);

    MAP2CF_D64(hipblasCopyStridedBatched, float, hipblasScopyStridedBatched);
    MAP2CF_D64(hipblasCopyStridedBatched, double, hipblasDcopyStridedBatched);
    MAP2CF_D64(hipblasCopyStridedBatched, std::complex<float>, hipblasCcopyStridedBatched);
    MAP2CF_D64(hipblasCopyStridedBatched, std::complex<double>, hipblasZcopyStridedBatched);

    // Swap
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSwap)(hipblasHandle_t           handle,
                                   int                       n,
                                   hipblas_internal_type<T>* x,
                                   int                       incx,
                                   hipblas_internal_type<T>* y,
                                   int                       incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSwapBatched)(hipblasHandle_t                 handle,
                                          int                             n,
                                          hipblas_internal_type<T>* const x[],
                                          int                             incx,
                                          hipblas_internal_type<T>* const y[],
                                          int                             incy,
                                          int                             batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSwapStridedBatched)(hipblasHandle_t           handle,
                                                 int                       n,
                                                 hipblas_internal_type<T>* x,
                                                 int                       incx,
                                                 hipblasStride             stridex,
                                                 hipblas_internal_type<T>* y,
                                                 int                       incy,
                                                 hipblasStride             stridey,
                                                 int                       batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSwap_64)(hipblasHandle_t           handle,
                                      int64_t                   n,
                                      hipblas_internal_type<T>* x,
                                      int64_t                   incx,
                                      hipblas_internal_type<T>* y,
                                      int64_t                   incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSwapBatched_64)(hipblasHandle_t                 handle,
                                             int64_t                         n,
                                             hipblas_internal_type<T>* const x[],
                                             int64_t                         incx,
                                             hipblas_internal_type<T>* const y[],
                                             int64_t                         incy,
                                             int64_t                         batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSwapStridedBatched_64)(hipblasHandle_t           handle,
                                                    int64_t                   n,
                                                    hipblas_internal_type<T>* x,
                                                    int64_t                   incx,
                                                    hipblasStride             stridex,
                                                    hipblas_internal_type<T>* y,
                                                    int64_t                   incy,
                                                    hipblasStride             stridey,
                                                    int64_t                   batch_count);

    MAP2CF_D64(hipblasSwap, float, hipblasSswap);
    MAP2CF_D64(hipblasSwap, double, hipblasDswap);
    MAP2CF_D64(hipblasSwap, std::complex<float>, hipblasCswap);
    MAP2CF_D64(hipblasSwap, std::complex<double>, hipblasZswap);

    MAP2CF_D64(hipblasSwapBatched, float, hipblasSswapBatched);
    MAP2CF_D64(hipblasSwapBatched, double, hipblasDswapBatched);
    MAP2CF_D64(hipblasSwapBatched, std::complex<float>, hipblasCswapBatched);
    MAP2CF_D64(hipblasSwapBatched, std::complex<double>, hipblasZswapBatched);

    MAP2CF_D64(hipblasSwapStridedBatched, float, hipblasSswapStridedBatched);
    MAP2CF_D64(hipblasSwapStridedBatched, double, hipblasDswapStridedBatched);
    MAP2CF_D64(hipblasSwapStridedBatched, std::complex<float>, hipblasCswapStridedBatched);
    MAP2CF_D64(hipblasSwapStridedBatched, std::complex<double>, hipblasZswapStridedBatched);

    // Dot
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDot)(hipblasHandle_t                 handle,
                                  int                             n,
                                  const hipblas_internal_type<T>* x,
                                  int                             incx,
                                  const hipblas_internal_type<T>* y,
                                  int                             incy,
                                  hipblas_internal_type<T>*       result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDotc)(hipblasHandle_t                 handle,
                                   int                             n,
                                   const hipblas_internal_type<T>* x,
                                   int                             incx,
                                   const hipblas_internal_type<T>* y,
                                   int                             incy,
                                   hipblas_internal_type<T>*       result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDotBatched)(hipblasHandle_t                       handle,
                                         int                                   n,
                                         const hipblas_internal_type<T>* const x[],
                                         int                                   incx,
                                         const hipblas_internal_type<T>* const y[],
                                         int                                   incy,
                                         int                                   batch_count,
                                         hipblas_internal_type<T>*             result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDotcBatched)(hipblasHandle_t                       handle,
                                          int                                   n,
                                          const hipblas_internal_type<T>* const x[],
                                          int                                   incx,
                                          const hipblas_internal_type<T>* const y[],
                                          int                                   incy,
                                          int                                   batch_count,
                                          hipblas_internal_type<T>*             result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDotStridedBatched)(hipblasHandle_t                 handle,
                                                int                             n,
                                                const hipblas_internal_type<T>* x,
                                                int                             incx,
                                                hipblasStride                   stridex,
                                                const hipblas_internal_type<T>* y,
                                                int                             incy,
                                                hipblasStride                   stridey,
                                                int                             batch_count,
                                                hipblas_internal_type<T>*       result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDotcStridedBatched)(hipblasHandle_t                 handle,
                                                 int                             n,
                                                 const hipblas_internal_type<T>* x,
                                                 int                             incx,
                                                 hipblasStride                   stridex,
                                                 const hipblas_internal_type<T>* y,
                                                 int                             incy,
                                                 hipblasStride                   stridey,
                                                 int                             batch_count,
                                                 hipblas_internal_type<T>*       result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDot_64)(hipblasHandle_t                 handle,
                                     int64_t                         n,
                                     const hipblas_internal_type<T>* x,
                                     int64_t                         incx,
                                     const hipblas_internal_type<T>* y,
                                     int64_t                         incy,
                                     hipblas_internal_type<T>*       result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDotc_64)(hipblasHandle_t                 handle,
                                      int64_t                         n,
                                      const hipblas_internal_type<T>* x,
                                      int64_t                         incx,
                                      const hipblas_internal_type<T>* y,
                                      int64_t                         incy,
                                      hipblas_internal_type<T>*       result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDotBatched_64)(hipblasHandle_t                       handle,
                                            int64_t                               n,
                                            const hipblas_internal_type<T>* const x[],
                                            int64_t                               incx,
                                            const hipblas_internal_type<T>* const y[],
                                            int64_t                               incy,
                                            int64_t                               batch_count,
                                            hipblas_internal_type<T>*             result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDotcBatched_64)(hipblasHandle_t                       handle,
                                             int64_t                               n,
                                             const hipblas_internal_type<T>* const x[],
                                             int64_t                               incx,
                                             const hipblas_internal_type<T>* const y[],
                                             int64_t                               incy,
                                             int64_t                               batch_count,
                                             hipblas_internal_type<T>*             result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDotStridedBatched_64)(hipblasHandle_t                 handle,
                                                   int64_t                         n,
                                                   const hipblas_internal_type<T>* x,
                                                   int64_t                         incx,
                                                   hipblasStride                   stridex,
                                                   const hipblas_internal_type<T>* y,
                                                   int64_t                         incy,
                                                   hipblasStride                   stridey,
                                                   int64_t                         batch_count,
                                                   hipblas_internal_type<T>*       result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDotcStridedBatched_64)(hipblasHandle_t                 handle,
                                                    int64_t                         n,
                                                    const hipblas_internal_type<T>* x,
                                                    int64_t                         incx,
                                                    hipblasStride                   stridex,
                                                    const hipblas_internal_type<T>* y,
                                                    int64_t                         incy,
                                                    hipblasStride                   stridey,
                                                    hipblas_internal_type<int64_t>  batch_count,
                                                    hipblas_internal_type<T>*       result);

    MAP2CF_D64(hipblasDot, hipblasHalf, hipblasHdot);
    MAP2CF_D64(hipblasDot, hipblasBfloat16, hipblasBfdot);
    MAP2CF_D64(hipblasDot, float, hipblasSdot);
    MAP2CF_D64(hipblasDot, double, hipblasDdot);
    MAP2CF_D64(hipblasDot, std::complex<float>, hipblasCdotu);
    MAP2CF_D64(hipblasDot, std::complex<double>, hipblasZdotu);
    MAP2CF_D64(hipblasDotc, std::complex<float>, hipblasCdotc);
    MAP2CF_D64(hipblasDotc, std::complex<double>, hipblasZdotc);

    MAP2CF_D64(hipblasDotBatched, hipblasHalf, hipblasHdotBatched);
    MAP2CF_D64(hipblasDotBatched, hipblasBfloat16, hipblasBfdotBatched);
    MAP2CF_D64(hipblasDotBatched, float, hipblasSdotBatched);
    MAP2CF_D64(hipblasDotBatched, double, hipblasDdotBatched);
    MAP2CF_D64(hipblasDotBatched, std::complex<float>, hipblasCdotuBatched);
    MAP2CF_D64(hipblasDotBatched, std::complex<double>, hipblasZdotuBatched);
    MAP2CF_D64(hipblasDotcBatched, std::complex<float>, hipblasCdotcBatched);
    MAP2CF_D64(hipblasDotcBatched, std::complex<double>, hipblasZdotcBatched);

    MAP2CF_D64(hipblasDotStridedBatched, hipblasHalf, hipblasHdotStridedBatched);
    MAP2CF_D64(hipblasDotStridedBatched, hipblasBfloat16, hipblasBfdotStridedBatched);
    MAP2CF_D64(hipblasDotStridedBatched, float, hipblasSdotStridedBatched);
    MAP2CF_D64(hipblasDotStridedBatched, double, hipblasDdotStridedBatched);
    MAP2CF_D64(hipblasDotStridedBatched, std::complex<float>, hipblasCdotuStridedBatched);
    MAP2CF_D64(hipblasDotStridedBatched, std::complex<double>, hipblasZdotuStridedBatched);
    MAP2CF_D64(hipblasDotcStridedBatched, std::complex<float>, hipblasCdotcStridedBatched);
    MAP2CF_D64(hipblasDotcStridedBatched, std::complex<double>, hipblasZdotcStridedBatched);

    // Asum
    template <typename T1, typename T2, bool FORTRAN = false>
    hipblasStatus_t (*hipblasAsum)(hipblasHandle_t                  handle,
                                   int                              n,
                                   const hipblas_internal_type<T1>* x,
                                   int                              incx,
                                   hipblas_internal_type<T2>*       result);

    template <typename T1, typename T2, bool FORTRAN = false>
    hipblasStatus_t (*hipblasAsumBatched)(hipblasHandle_t                        handle,
                                          int                                    n,
                                          const hipblas_internal_type<T1>* const x[],
                                          int                                    incx,
                                          int                                    batch_count,
                                          hipblas_internal_type<T2>*             result);

    template <typename T1, typename T2, bool FORTRAN = false>
    hipblasStatus_t (*hipblasAsumStridedBatched)(hipblasHandle_t                  handle,
                                                 int                              n,
                                                 const hipblas_internal_type<T1>* x,
                                                 int                              incx,
                                                 hipblasStride                    stridex,
                                                 int                              batch_count,
                                                 hipblas_internal_type<T2>*       result);

    template <typename T1, typename T2, bool FORTRAN = false>
    hipblasStatus_t (*hipblasAsum_64)(hipblasHandle_t                  handle,
                                      int64_t                          n,
                                      const hipblas_internal_type<T1>* x,
                                      int64_t                          incx,
                                      hipblas_internal_type<T2>*       result);

    template <typename T1, typename T2, bool FORTRAN = false>
    hipblasStatus_t (*hipblasAsumBatched_64)(hipblasHandle_t                        handle,
                                             int64_t                                n,
                                             const hipblas_internal_type<T1>* const x[],
                                             int64_t                                incx,
                                             int64_t                                batch_count,
                                             hipblas_internal_type<T2>*             result);

    template <typename T1, typename T2, bool FORTRAN = false>
    hipblasStatus_t (*hipblasAsumStridedBatched_64)(hipblasHandle_t                  handle,
                                                    int64_t                          n,
                                                    const hipblas_internal_type<T1>* x,
                                                    int64_t                          incx,
                                                    hipblasStride                    stridex,
                                                    int64_t                          batch_count,
                                                    hipblas_internal_type<T2>*       result);

    MAP2CF_D64(hipblasAsum, float, float, hipblasSasum);
    MAP2CF_D64(hipblasAsum, double, double, hipblasDasum);
    MAP2CF_D64(hipblasAsum, std::complex<float>, float, hipblasScasum);
    MAP2CF_D64(hipblasAsum, std::complex<double>, double, hipblasDzasum);

    MAP2CF_D64(hipblasAsumBatched, float, float, hipblasSasumBatched);
    MAP2CF_D64(hipblasAsumBatched, double, double, hipblasDasumBatched);
    MAP2CF_D64(hipblasAsumBatched, std::complex<float>, float, hipblasScasumBatched);
    MAP2CF_D64(hipblasAsumBatched, std::complex<double>, double, hipblasDzasumBatched);

    MAP2CF_D64(hipblasAsumStridedBatched, float, float, hipblasSasumStridedBatched);
    MAP2CF_D64(hipblasAsumStridedBatched, double, double, hipblasDasumStridedBatched);
    MAP2CF_D64(hipblasAsumStridedBatched, std::complex<float>, float, hipblasScasumStridedBatched);
    MAP2CF_D64(hipblasAsumStridedBatched,
               std::complex<double>,
               double,
               hipblasDzasumStridedBatched);

    // nrm2
    template <typename T1, typename T2, bool FORTRAN = false>
    hipblasStatus_t (*hipblasNrm2)(hipblasHandle_t                  handle,
                                   int                              n,
                                   const hipblas_internal_type<T1>* x,
                                   int                              incx,
                                   hipblas_internal_type<T2>*       result);

    template <typename T1, typename T2, bool FORTRAN = false>
    hipblasStatus_t (*hipblasNrm2Batched)(hipblasHandle_t                        handle,
                                          int                                    n,
                                          const hipblas_internal_type<T1>* const x[],
                                          int                                    incx,
                                          int                                    batch_count,
                                          hipblas_internal_type<T2>*             result);

    template <typename T1, typename T2, bool FORTRAN = false>
    hipblasStatus_t (*hipblasNrm2StridedBatched)(hipblasHandle_t                  handle,
                                                 int                              n,
                                                 const hipblas_internal_type<T1>* x,
                                                 int                              incx,
                                                 hipblasStride                    stridex,
                                                 int                              batch_count,
                                                 hipblas_internal_type<T2>*       result);

    template <typename T1, typename T2, bool FORTRAN = false>
    hipblasStatus_t (*hipblasNrm2_64)(hipblasHandle_t                  handle,
                                      int64_t                          n,
                                      const hipblas_internal_type<T1>* x,
                                      int64_t                          incx,
                                      hipblas_internal_type<T2>*       result);

    template <typename T1, typename T2, bool FORTRAN = false>
    hipblasStatus_t (*hipblasNrm2Batched_64)(hipblasHandle_t                        handle,
                                             int64_t                                n,
                                             const hipblas_internal_type<T1>* const x[],
                                             int64_t                                incx,
                                             int64_t                                batch_count,
                                             hipblas_internal_type<T2>*             result);

    template <typename T1, typename T2, bool FORTRAN = false>
    hipblasStatus_t (*hipblasNrm2StridedBatched_64)(hipblasHandle_t                  handle,
                                                    int64_t                          n,
                                                    const hipblas_internal_type<T1>* x,
                                                    int64_t                          incx,
                                                    hipblasStride                    stridex,
                                                    int64_t                          batch_count,
                                                    hipblas_internal_type<T2>*       result);

    MAP2CF_D64(hipblasNrm2, float, float, hipblasSnrm2);
    MAP2CF_D64(hipblasNrm2, double, double, hipblasDnrm2);
    MAP2CF_D64(hipblasNrm2, std::complex<float>, float, hipblasScnrm2);
    MAP2CF_D64(hipblasNrm2, std::complex<double>, double, hipblasDznrm2);

    MAP2CF_D64(hipblasNrm2Batched, float, float, hipblasSnrm2Batched);
    MAP2CF_D64(hipblasNrm2Batched, double, double, hipblasDnrm2Batched);
    MAP2CF_D64(hipblasNrm2Batched, std::complex<float>, float, hipblasScnrm2Batched);
    MAP2CF_D64(hipblasNrm2Batched, std::complex<double>, double, hipblasDznrm2Batched);

    MAP2CF_D64(hipblasNrm2StridedBatched, float, float, hipblasSnrm2StridedBatched);
    MAP2CF_D64(hipblasNrm2StridedBatched, double, double, hipblasDnrm2StridedBatched);
    MAP2CF_D64(hipblasNrm2StridedBatched, std::complex<float>, float, hipblasScnrm2StridedBatched);
    MAP2CF_D64(hipblasNrm2StridedBatched,
               std::complex<double>,
               double,
               hipblasDznrm2StridedBatched);

    // Rot
    template <typename T1, typename T2, typename T3 = T1, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRot)(hipblasHandle_t                  handle,
                                  int                              n,
                                  hipblas_internal_type<T1>*       x,
                                  int                              incx,
                                  hipblas_internal_type<T1>*       y,
                                  int                              incy,
                                  const hipblas_internal_type<T2>* c,
                                  const hipblas_internal_type<T3>* s);

    template <typename T1, typename T2 = T1, typename T3 = T1, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotBatched)(hipblasHandle_t                  handle,
                                         int                              n,
                                         hipblas_internal_type<T1>* const x[],
                                         int                              incx,
                                         hipblas_internal_type<T1>* const y[],
                                         int                              incy,
                                         const hipblas_internal_type<T2>* c,
                                         const hipblas_internal_type<T3>* s,
                                         int                              batch_count);

    template <typename T1, typename T2 = T1, typename T3 = T1, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotStridedBatched)(hipblasHandle_t                  handle,
                                                int                              n,
                                                hipblas_internal_type<T1>*       x,
                                                int                              incx,
                                                hipblasStride                    stridex,
                                                hipblas_internal_type<T1>*       y,
                                                int                              incy,
                                                hipblasStride                    stridey,
                                                const hipblas_internal_type<T2>* c,
                                                const hipblas_internal_type<T3>* s,
                                                int                              batch_count);

    template <typename T1, typename T2, typename T3 = T1, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRot_64)(hipblasHandle_t                  handle,
                                     int64_t                          n,
                                     hipblas_internal_type<T1>*       x,
                                     int64_t                          incx,
                                     hipblas_internal_type<T1>*       y,
                                     int64_t                          incy,
                                     const hipblas_internal_type<T2>* c,
                                     const hipblas_internal_type<T3>* s);

    template <typename T1, typename T2 = T1, typename T3 = T1, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotBatched_64)(hipblasHandle_t                  handle,
                                            int64_t                          n,
                                            hipblas_internal_type<T1>* const x[],
                                            int64_t                          incx,
                                            hipblas_internal_type<T1>* const y[],
                                            int64_t                          incy,
                                            const hipblas_internal_type<T2>* c,
                                            const hipblas_internal_type<T3>* s,
                                            int64_t                          batch_count);

    template <typename T1, typename T2 = T1, typename T3 = T1, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotStridedBatched_64)(hipblasHandle_t                  handle,
                                                   int64_t                          n,
                                                   hipblas_internal_type<T1>*       x,
                                                   int64_t                          incx,
                                                   hipblasStride                    stridex,
                                                   hipblas_internal_type<T1>*       y,
                                                   int64_t                          incy,
                                                   hipblasStride                    stridey,
                                                   const hipblas_internal_type<T2>* c,
                                                   const hipblas_internal_type<T3>* s,
                                                   int64_t                          batch_count);

    MAP2CF_D64(hipblasRot, float, float, float, hipblasSrot);
    MAP2CF_D64(hipblasRot, double, double, double, hipblasDrot);
    MAP2CF_D64(hipblasRot, std::complex<float>, float, std::complex<float>, hipblasCrot);
    MAP2CF_D64(hipblasRot, std::complex<double>, double, std::complex<double>, hipblasZrot);
    MAP2CF_D64(hipblasRot, std::complex<float>, float, float, hipblasCsrot);
    MAP2CF_D64(hipblasRot, std::complex<double>, double, double, hipblasZdrot);

    MAP2CF_D64(hipblasRotBatched, float, float, float, hipblasSrotBatched);
    MAP2CF_D64(hipblasRotBatched, double, double, double, hipblasDrotBatched);
    MAP2CF_D64(
        hipblasRotBatched, std::complex<float>, float, std::complex<float>, hipblasCrotBatched);
    MAP2CF_D64(
        hipblasRotBatched, std::complex<double>, double, std::complex<double>, hipblasZrotBatched);
    MAP2CF_D64(hipblasRotBatched, std::complex<float>, float, float, hipblasCsrotBatched);
    MAP2CF_D64(hipblasRotBatched, std::complex<double>, double, double, hipblasZdrotBatched);

    MAP2CF_D64(hipblasRotStridedBatched, float, float, float, hipblasSrotStridedBatched);
    MAP2CF_D64(hipblasRotStridedBatched, double, double, double, hipblasDrotStridedBatched);
    MAP2CF_D64(hipblasRotStridedBatched,
               std::complex<float>,
               float,
               std::complex<float>,
               hipblasCrotStridedBatched);
    MAP2CF_D64(hipblasRotStridedBatched,
               std::complex<double>,
               double,
               std::complex<double>,
               hipblasZrotStridedBatched);
    MAP2CF_D64(
        hipblasRotStridedBatched, std::complex<float>, float, float, hipblasCsrotStridedBatched);
    MAP2CF_D64(
        hipblasRotStridedBatched, std::complex<double>, double, double, hipblasZdrotStridedBatched);

    // Rotg
    template <typename T1, typename T2 = T1, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotg)(hipblasHandle_t            handle,
                                   hipblas_internal_type<T1>* a,
                                   hipblas_internal_type<T1>* b,
                                   T2*                        c,
                                   hipblas_internal_type<T1>* s);

    template <typename T1, typename T2 = T1, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotgBatched)(hipblasHandle_t                  handle,
                                          hipblas_internal_type<T1>* const a[],
                                          hipblas_internal_type<T1>* const b[],
                                          hipblas_internal_type<T2>* const c[],
                                          hipblas_internal_type<T1>* const s[],
                                          int                              batch_count);

    template <typename T1, typename T2 = T1, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotgStridedBatched)(hipblasHandle_t            handle,
                                                 hipblas_internal_type<T1>* a,
                                                 hipblasStride              stridea,
                                                 hipblas_internal_type<T1>* b,
                                                 hipblasStride              strideb,
                                                 hipblas_internal_type<T2>* c,
                                                 hipblasStride              stridec,
                                                 hipblas_internal_type<T1>* s,
                                                 hipblasStride              strides,
                                                 int                        batch_count);

    template <typename T1, typename T2 = T1, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotg_64)(hipblasHandle_t            handle,
                                      hipblas_internal_type<T1>* a,
                                      hipblas_internal_type<T1>* b,
                                      hipblas_internal_type<T2>* c,
                                      hipblas_internal_type<T1>* s);

    template <typename T1, typename T2 = T1, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotgBatched_64)(hipblasHandle_t                  handle,
                                             hipblas_internal_type<T1>* const a[],
                                             hipblas_internal_type<T1>* const b[],
                                             hipblas_internal_type<T2>* const c[],
                                             hipblas_internal_type<T1>* const s[],
                                             int64_t                          batch_count);

    template <typename T1, typename T2 = T1, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotgStridedBatched_64)(hipblasHandle_t            handle,
                                                    hipblas_internal_type<T1>* a,
                                                    hipblasStride              stridea,
                                                    hipblas_internal_type<T1>* b,
                                                    hipblasStride              strideb,
                                                    hipblas_internal_type<T2>* c,
                                                    hipblasStride              stridec,
                                                    hipblas_internal_type<T1>* s,
                                                    hipblasStride              strides,
                                                    int64_t                    batch_count);

    MAP2CF_D64(hipblasRotg, float, float, hipblasSrotg);
    MAP2CF_D64(hipblasRotg, double, double, hipblasDrotg);
    MAP2CF_D64(hipblasRotg, std::complex<float>, float, hipblasCrotg);
    MAP2CF_D64(hipblasRotg, std::complex<double>, double, hipblasZrotg);

    MAP2CF_D64(hipblasRotgBatched, float, float, hipblasSrotgBatched);
    MAP2CF_D64(hipblasRotgBatched, double, double, hipblasDrotgBatched);
    MAP2CF_D64(hipblasRotgBatched, std::complex<float>, float, hipblasCrotgBatched);
    MAP2CF_D64(hipblasRotgBatched, std::complex<double>, double, hipblasZrotgBatched);

    MAP2CF_D64(hipblasRotgStridedBatched, float, float, hipblasSrotgStridedBatched);
    MAP2CF_D64(hipblasRotgStridedBatched, double, double, hipblasDrotgStridedBatched);
    MAP2CF_D64(hipblasRotgStridedBatched, std::complex<float>, float, hipblasCrotgStridedBatched);
    MAP2CF_D64(hipblasRotgStridedBatched, std::complex<double>, double, hipblasZrotgStridedBatched);

    // rotm
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotm)(hipblasHandle_t                 handle,
                                   int                             n,
                                   hipblas_internal_type<T>*       x,
                                   int                             incx,
                                   hipblas_internal_type<T>*       y,
                                   int                             incy,
                                   const hipblas_internal_type<T>* param);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotmBatched)(hipblasHandle_t                       handle,
                                          int                                   n,
                                          hipblas_internal_type<T>* const       x[],
                                          int                                   incx,
                                          hipblas_internal_type<T>* const       y[],
                                          int                                   incy,
                                          const hipblas_internal_type<T>* const param[],
                                          int                                   batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotmStridedBatched)(hipblasHandle_t                 handle,
                                                 int                             n,
                                                 hipblas_internal_type<T>*       x,
                                                 int                             incx,
                                                 hipblasStride                   stridex,
                                                 hipblas_internal_type<T>*       y,
                                                 int                             incy,
                                                 hipblasStride                   stridey,
                                                 const hipblas_internal_type<T>* param,
                                                 hipblasStride                   strideparam,
                                                 int                             batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotm_64)(hipblasHandle_t                 handle,
                                      int64_t                         n,
                                      hipblas_internal_type<T>*       x,
                                      int64_t                         incx,
                                      hipblas_internal_type<T>*       y,
                                      int64_t                         incy,
                                      const hipblas_internal_type<T>* param);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotmBatched_64)(hipblasHandle_t                       handle,
                                             int64_t                               n,
                                             hipblas_internal_type<T>* const       x[],
                                             int64_t                               incx,
                                             hipblas_internal_type<T>* const       y[],
                                             int64_t                               incy,
                                             const hipblas_internal_type<T>* const param[],
                                             int64_t                               batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotmStridedBatched_64)(hipblasHandle_t                 handle,
                                                    int64_t                         n,
                                                    hipblas_internal_type<T>*       x,
                                                    int64_t                         incx,
                                                    hipblasStride                   stridex,
                                                    hipblas_internal_type<T>*       y,
                                                    int64_t                         incy,
                                                    hipblasStride                   stridey,
                                                    const hipblas_internal_type<T>* param,
                                                    hipblasStride                   strideparam,
                                                    int64_t                         batch_count);

    MAP2CF_D64(hipblasRotm, float, hipblasSrotm);
    MAP2CF_D64(hipblasRotm, double, hipblasDrotm);

    MAP2CF_D64(hipblasRotmBatched, float, hipblasSrotmBatched);
    MAP2CF_D64(hipblasRotmBatched, double, hipblasDrotmBatched);

    MAP2CF_D64(hipblasRotmStridedBatched, float, hipblasSrotmStridedBatched);
    MAP2CF_D64(hipblasRotmStridedBatched, double, hipblasDrotmStridedBatched);

    // rotmg
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotmg)(hipblasHandle_t                 handle,
                                    hipblas_internal_type<T>*       d1,
                                    hipblas_internal_type<T>*       d2,
                                    hipblas_internal_type<T>*       x1,
                                    const hipblas_internal_type<T>* y1,
                                    hipblas_internal_type<T>*       param);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotmgBatched)(hipblasHandle_t                       handle,
                                           hipblas_internal_type<T>* const       d1[],
                                           hipblas_internal_type<T>* const       d2[],
                                           hipblas_internal_type<T>* const       x1[],
                                           const hipblas_internal_type<T>* const y1[],
                                           hipblas_internal_type<T>* const       param[],
                                           int                                   batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotmgStridedBatched)(hipblasHandle_t                 handle,
                                                  hipblas_internal_type<T>*       d1,
                                                  hipblasStride                   stride_d1,
                                                  hipblas_internal_type<T>*       d2,
                                                  hipblasStride                   stride_d2,
                                                  hipblas_internal_type<T>*       x1,
                                                  hipblasStride                   stride_x1,
                                                  const hipblas_internal_type<T>* y1,
                                                  hipblasStride                   stride_y1,
                                                  hipblas_internal_type<T>*       param,
                                                  hipblasStride                   strideparam,
                                                  int                             batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotmg_64)(hipblasHandle_t                 handle,
                                       hipblas_internal_type<T>*       d1,
                                       hipblas_internal_type<T>*       d2,
                                       hipblas_internal_type<T>*       x1,
                                       const hipblas_internal_type<T>* y1,
                                       hipblas_internal_type<T>*       param);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotmgBatched_64)(hipblasHandle_t                       handle,
                                              hipblas_internal_type<T>* const       d1[],
                                              hipblas_internal_type<T>* const       d2[],
                                              hipblas_internal_type<T>* const       x1[],
                                              const hipblas_internal_type<T>* const y1[],
                                              hipblas_internal_type<T>* const       param[],
                                              int64_t                               batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotmgStridedBatched_64)(hipblasHandle_t                 handle,
                                                     hipblas_internal_type<T>*       d1,
                                                     hipblasStride                   stride_d1,
                                                     hipblas_internal_type<T>*       d2,
                                                     hipblasStride                   stride_d2,
                                                     hipblas_internal_type<T>*       x1,
                                                     hipblasStride                   stride_x1,
                                                     const hipblas_internal_type<T>* y1,
                                                     hipblasStride                   stride_y1,
                                                     hipblas_internal_type<T>*       param,
                                                     hipblasStride                   strideparam,
                                                     int64_t                         batch_count);

    MAP2CF_D64(hipblasRotmg, float, hipblasSrotmg);
    MAP2CF_D64(hipblasRotmg, double, hipblasDrotmg);

    MAP2CF_D64(hipblasRotmgBatched, float, hipblasSrotmgBatched);
    MAP2CF_D64(hipblasRotmgBatched, double, hipblasDrotmgBatched);

    MAP2CF_D64(hipblasRotmgStridedBatched, float, hipblasSrotmgStridedBatched);
    MAP2CF_D64(hipblasRotmgStridedBatched, double, hipblasDrotmgStridedBatched);

    // amax
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasIamax)(
        hipblasHandle_t handle, int n, const hipblas_internal_type<T>* x, int incx, int* result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasIamaxBatched)(hipblasHandle_t                       handle,
                                           int                                   n,
                                           const hipblas_internal_type<T>* const x[],
                                           int                                   incx,
                                           int                                   batch_count,
                                           int*                                  result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasIamaxStridedBatched)(hipblasHandle_t                 handle,
                                                  int                             n,
                                                  const hipblas_internal_type<T>* x,
                                                  int                             incx,
                                                  hipblasStride                   stridex,
                                                  int                             batch_count,
                                                  int*                            result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasIamax_64)(hipblasHandle_t                 handle,
                                       int64_t                         n,
                                       const hipblas_internal_type<T>* x,
                                       int64_t                         incx,
                                       int64_t*                        result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasIamaxBatched_64)(hipblasHandle_t                       handle,
                                              int64_t                               n,
                                              const hipblas_internal_type<T>* const x[],
                                              int64_t                               incx,
                                              int64_t                               batch_count,
                                              int64_t*                              result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasIamaxStridedBatched_64)(hipblasHandle_t                 handle,
                                                     int64_t                         n,
                                                     const hipblas_internal_type<T>* x,
                                                     int64_t                         incx,
                                                     hipblasStride                   stridex,
                                                     int64_t                         batch_count,
                                                     int64_t*                        result);

    MAP2CF_D64(hipblasIamax, float, hipblasIsamax);
    MAP2CF_D64(hipblasIamax, double, hipblasIdamax);
    MAP2CF_D64(hipblasIamax, std::complex<float>, hipblasIcamax);
    MAP2CF_D64(hipblasIamax, std::complex<double>, hipblasIzamax);

    MAP2CF_D64(hipblasIamaxBatched, float, hipblasIsamaxBatched);
    MAP2CF_D64(hipblasIamaxBatched, double, hipblasIdamaxBatched);
    MAP2CF_D64(hipblasIamaxBatched, std::complex<float>, hipblasIcamaxBatched);
    MAP2CF_D64(hipblasIamaxBatched, std::complex<double>, hipblasIzamaxBatched);

    MAP2CF_D64(hipblasIamaxStridedBatched, float, hipblasIsamaxStridedBatched);
    MAP2CF_D64(hipblasIamaxStridedBatched, double, hipblasIdamaxStridedBatched);
    MAP2CF_D64(hipblasIamaxStridedBatched, std::complex<float>, hipblasIcamaxStridedBatched);
    MAP2CF_D64(hipblasIamaxStridedBatched, std::complex<double>, hipblasIzamaxStridedBatched);

    // amin
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasIamin)(
        hipblasHandle_t handle, int n, const hipblas_internal_type<T>* x, int incx, int* result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasIaminBatched)(hipblasHandle_t                       handle,
                                           int                                   n,
                                           const hipblas_internal_type<T>* const x[],
                                           int                                   incx,
                                           int                                   batch_count,
                                           int*                                  result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasIaminStridedBatched)(hipblasHandle_t                 handle,
                                                  int                             n,
                                                  const hipblas_internal_type<T>* x,
                                                  int                             incx,
                                                  hipblasStride                   stridex,
                                                  int                             batch_count,
                                                  int*                            result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasIamin_64)(hipblasHandle_t                 handle,
                                       int64_t                         n,
                                       const hipblas_internal_type<T>* x,
                                       int64_t                         incx,
                                       int64_t*                        result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasIaminBatched_64)(hipblasHandle_t                       handle,
                                              int64_t                               n,
                                              const hipblas_internal_type<T>* const x[],
                                              int64_t                               incx,
                                              int64_t                               batch_count,
                                              int64_t*                              result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasIaminStridedBatched_64)(hipblasHandle_t                 handle,
                                                     int64_t                         n,
                                                     const hipblas_internal_type<T>* x,
                                                     int64_t                         incx,
                                                     hipblasStride                   stridex,
                                                     int64_t                         batch_count,
                                                     int64_t*                        result);

    MAP2CF_D64(hipblasIamin, float, hipblasIsamin);
    MAP2CF_D64(hipblasIamin, double, hipblasIdamin);
    MAP2CF_D64(hipblasIamin, std::complex<float>, hipblasIcamin);
    MAP2CF_D64(hipblasIamin, std::complex<double>, hipblasIzamin);

    MAP2CF_D64(hipblasIaminBatched, float, hipblasIsaminBatched);
    MAP2CF_D64(hipblasIaminBatched, double, hipblasIdaminBatched);
    MAP2CF_D64(hipblasIaminBatched, std::complex<float>, hipblasIcaminBatched);
    MAP2CF_D64(hipblasIaminBatched, std::complex<double>, hipblasIzaminBatched);

    MAP2CF_D64(hipblasIaminStridedBatched, float, hipblasIsaminStridedBatched);
    MAP2CF_D64(hipblasIaminStridedBatched, double, hipblasIdaminStridedBatched);
    MAP2CF_D64(hipblasIaminStridedBatched, std::complex<float>, hipblasIcaminStridedBatched);
    MAP2CF_D64(hipblasIaminStridedBatched, std::complex<double>, hipblasIzaminStridedBatched);

    // axpy
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasAxpy)(hipblasHandle_t                 handle,
                                   int                             n,
                                   const hipblas_internal_type<T>* alpha,
                                   const hipblas_internal_type<T>* x,
                                   int                             incx,
                                   hipblas_internal_type<T>*       y,
                                   int                             incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasAxpyBatched)(hipblasHandle_t                       handle,
                                          int                                   n,
                                          const hipblas_internal_type<T>*       alpha,
                                          const hipblas_internal_type<T>* const x[],
                                          int                                   incx,
                                          hipblas_internal_type<T>* const       y[],
                                          int                                   incy,
                                          int                                   batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasAxpyStridedBatched)(hipblasHandle_t                 handle,
                                                 int                             n,
                                                 const hipblas_internal_type<T>* alpha,
                                                 const hipblas_internal_type<T>* x,
                                                 int                             incx,
                                                 hipblasStride                   stridex,
                                                 hipblas_internal_type<T>*       y,
                                                 int                             incy,
                                                 hipblasStride                   stridey,
                                                 int                             batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasAxpy_64)(hipblasHandle_t                 handle,
                                      int64_t                         n,
                                      const hipblas_internal_type<T>* alpha,
                                      const hipblas_internal_type<T>* x,
                                      int64_t                         incx,
                                      hipblas_internal_type<T>*       y,
                                      int64_t                         incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasAxpyBatched_64)(hipblasHandle_t                       handle,
                                             int64_t                               n,
                                             const hipblas_internal_type<T>*       alpha,
                                             const hipblas_internal_type<T>* const x[],
                                             int64_t                               incx,
                                             hipblas_internal_type<T>* const       y[],
                                             int64_t                               incy,
                                             int64_t                               batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasAxpyStridedBatched_64)(hipblasHandle_t                 handle,
                                                    int64_t                         n,
                                                    const hipblas_internal_type<T>* alpha,
                                                    const hipblas_internal_type<T>* x,
                                                    int64_t                         incx,
                                                    hipblasStride                   stridex,
                                                    hipblas_internal_type<T>*       y,
                                                    int64_t                         incy,
                                                    hipblasStride                   stridey,
                                                    int64_t                         batch_count);

    MAP2CF_D64(hipblasAxpy, hipblasHalf, hipblasHaxpy);
    MAP2CF_D64(hipblasAxpy, float, hipblasSaxpy);
    MAP2CF_D64(hipblasAxpy, double, hipblasDaxpy);
    MAP2CF_D64(hipblasAxpy, std::complex<float>, hipblasCaxpy);
    MAP2CF_D64(hipblasAxpy, std::complex<double>, hipblasZaxpy);

    MAP2CF_D64(hipblasAxpyBatched, hipblasHalf, hipblasHaxpyBatched);
    MAP2CF_D64(hipblasAxpyBatched, float, hipblasSaxpyBatched);
    MAP2CF_D64(hipblasAxpyBatched, double, hipblasDaxpyBatched);
    MAP2CF_D64(hipblasAxpyBatched, std::complex<float>, hipblasCaxpyBatched);
    MAP2CF_D64(hipblasAxpyBatched, std::complex<double>, hipblasZaxpyBatched);

    MAP2CF_D64(hipblasAxpyStridedBatched, hipblasHalf, hipblasHaxpyStridedBatched);
    MAP2CF_D64(hipblasAxpyStridedBatched, float, hipblasSaxpyStridedBatched);
    MAP2CF_D64(hipblasAxpyStridedBatched, double, hipblasDaxpyStridedBatched);
    MAP2CF_D64(hipblasAxpyStridedBatched, std::complex<float>, hipblasCaxpyStridedBatched);
    MAP2CF_D64(hipblasAxpyStridedBatched, std::complex<double>, hipblasZaxpyStridedBatched);

    // ger
    template <typename T, bool CONJ, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGer)(hipblasHandle_t                 handle,
                                  int                             m,
                                  int                             n,
                                  const hipblas_internal_type<T>* alpha,
                                  const hipblas_internal_type<T>* x,
                                  int                             incx,
                                  const hipblas_internal_type<T>* y,
                                  int                             incy,
                                  hipblas_internal_type<T>*       A,
                                  int                             lda);

    template <typename T, bool CONJ, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGerBatched)(hipblasHandle_t                       handle,
                                         int                                   m,
                                         int                                   n,
                                         const hipblas_internal_type<T>*       alpha,
                                         const hipblas_internal_type<T>* const x[],
                                         int                                   incx,
                                         const hipblas_internal_type<T>* const y[],
                                         int                                   incy,
                                         hipblas_internal_type<T>* const       A[],
                                         int                                   lda,
                                         int                                   batch_count);

    template <typename T, bool CONJ, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGerStridedBatched)(hipblasHandle_t                 handle,
                                                int                             m,
                                                int                             n,
                                                const hipblas_internal_type<T>* alpha,
                                                const hipblas_internal_type<T>* x,
                                                int                             incx,
                                                hipblasStride                   stridex,
                                                const hipblas_internal_type<T>* y,
                                                int                             incy,
                                                hipblasStride                   stridey,
                                                hipblas_internal_type<T>*       A,
                                                int                             lda,
                                                hipblasStride                   strideA,
                                                int                             batch_count);

    // ger_64
    template <typename T, bool CONJ, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGer_64)(hipblasHandle_t                 handle,
                                     int64_t                         m,
                                     int64_t                         n,
                                     const hipblas_internal_type<T>* alpha,
                                     const hipblas_internal_type<T>* x,
                                     int64_t                         incx,
                                     const hipblas_internal_type<T>* y,
                                     int64_t                         incy,
                                     hipblas_internal_type<T>*       A,
                                     int64_t                         lda);

    template <typename T, bool CONJ, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGerBatched_64)(hipblasHandle_t                       handle,
                                            int64_t                               m,
                                            int64_t                               n,
                                            const hipblas_internal_type<T>*       alpha,
                                            const hipblas_internal_type<T>* const x[],
                                            int64_t                               incx,
                                            const hipblas_internal_type<T>* const y[],
                                            int64_t                               incy,
                                            hipblas_internal_type<T>* const       A[],
                                            int64_t                               lda,
                                            int64_t                               batch_count);

    template <typename T, bool CONJ, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGerStridedBatched_64)(hipblasHandle_t                 handle,
                                                   int64_t                         m,
                                                   int64_t                         n,
                                                   const hipblas_internal_type<T>* alpha,
                                                   const hipblas_internal_type<T>* x,
                                                   int64_t                         incx,
                                                   hipblasStride                   stridex,
                                                   const hipblas_internal_type<T>* y,
                                                   int64_t                         incy,
                                                   hipblasStride                   stridey,
                                                   hipblas_internal_type<T>*       A,
                                                   int64_t                         lda,
                                                   hipblasStride                   strideA,
                                                   int64_t                         batch_count);

    MAP2CF_D64(hipblasGer, float, false, hipblasSger);
    MAP2CF_D64(hipblasGer, double, false, hipblasDger);
    MAP2CF_D64(hipblasGer, std::complex<float>, false, hipblasCgeru);
    MAP2CF_D64(hipblasGer, std::complex<double>, false, hipblasZgeru);
    MAP2CF_D64(hipblasGer, std::complex<float>, true, hipblasCgerc);
    MAP2CF_D64(hipblasGer, std::complex<double>, true, hipblasZgerc);

    MAP2CF_D64(hipblasGerBatched, float, false, hipblasSgerBatched);
    MAP2CF_D64(hipblasGerBatched, double, false, hipblasDgerBatched);
    MAP2CF_D64(hipblasGerBatched, std::complex<float>, false, hipblasCgeruBatched);
    MAP2CF_D64(hipblasGerBatched, std::complex<double>, false, hipblasZgeruBatched);
    MAP2CF_D64(hipblasGerBatched, std::complex<float>, true, hipblasCgercBatched);
    MAP2CF_D64(hipblasGerBatched, std::complex<double>, true, hipblasZgercBatched);

    MAP2CF_D64(hipblasGerStridedBatched, float, false, hipblasSgerStridedBatched);
    MAP2CF_D64(hipblasGerStridedBatched, double, false, hipblasDgerStridedBatched);
    MAP2CF_D64(hipblasGerStridedBatched, std::complex<float>, false, hipblasCgeruStridedBatched);
    MAP2CF_D64(hipblasGerStridedBatched, std::complex<double>, false, hipblasZgeruStridedBatched);
    MAP2CF_D64(hipblasGerStridedBatched, std::complex<float>, true, hipblasCgercStridedBatched);
    MAP2CF_D64(hipblasGerStridedBatched, std::complex<double>, true, hipblasZgercStridedBatched);

    // hbmv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHbmv)(hipblasHandle_t                 handle,
                                   hipblasFillMode_t               uplo,
                                   int                             n,
                                   int                             k,
                                   const hipblas_internal_type<T>* alpha,
                                   const hipblas_internal_type<T>* A,
                                   int                             lda,
                                   const hipblas_internal_type<T>* x,
                                   int                             incx,
                                   const hipblas_internal_type<T>* beta,
                                   hipblas_internal_type<T>*       y,
                                   int                             incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHbmv_64)(hipblasHandle_t                 handle,
                                      hipblasFillMode_t               uplo,
                                      int64_t                         n,
                                      int64_t                         k,
                                      const hipblas_internal_type<T>* alpha,
                                      const hipblas_internal_type<T>* A,
                                      int64_t                         lda,
                                      const hipblas_internal_type<T>* x,
                                      int64_t                         incx,
                                      const hipblas_internal_type<T>* beta,
                                      hipblas_internal_type<T>*       y,
                                      int64_t                         incy);

    MAP2CF_D64(hipblasHbmv, std::complex<float>, hipblasChbmv);
    MAP2CF_D64(hipblasHbmv, std::complex<double>, hipblasZhbmv);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHbmvBatched)(hipblasHandle_t                       handle,
                                          hipblasFillMode_t                     uplo,
                                          int                                   n,
                                          int                                   k,
                                          const hipblas_internal_type<T>*       alpha,
                                          const hipblas_internal_type<T>* const A[],
                                          int                                   lda,
                                          const hipblas_internal_type<T>* const x[],
                                          int                                   incx,
                                          const hipblas_internal_type<T>*       beta,
                                          hipblas_internal_type<T>* const       y[],
                                          int                                   incy,
                                          int                                   batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHbmvBatched_64)(hipblasHandle_t                       handle,
                                             hipblasFillMode_t                     uplo,
                                             int64_t                               n,
                                             int64_t                               k,
                                             const hipblas_internal_type<T>*       alpha,
                                             const hipblas_internal_type<T>* const A[],
                                             int64_t                               lda,
                                             const hipblas_internal_type<T>* const x[],
                                             int64_t                               incx,
                                             const hipblas_internal_type<T>*       beta,
                                             hipblas_internal_type<T>* const       y[],
                                             int64_t                               incy,
                                             int64_t                               batchCount);

    MAP2CF_D64(hipblasHbmvBatched, std::complex<float>, hipblasChbmvBatched);
    MAP2CF_D64(hipblasHbmvBatched, std::complex<double>, hipblasZhbmvBatched);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHbmvStridedBatched)(hipblasHandle_t                 handle,
                                                 hipblasFillMode_t               uplo,
                                                 int                             n,
                                                 int                             k,
                                                 const hipblas_internal_type<T>* alpha,
                                                 const hipblas_internal_type<T>* A,
                                                 int                             lda,
                                                 hipblasStride                   strideA,
                                                 const hipblas_internal_type<T>* x,
                                                 int                             incx,
                                                 hipblasStride                   stridex,
                                                 const hipblas_internal_type<T>* beta,
                                                 hipblas_internal_type<T>*       y,
                                                 int                             incy,
                                                 hipblasStride                   stridey,
                                                 int                             batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHbmvStridedBatched_64)(hipblasHandle_t                 handle,
                                                    hipblasFillMode_t               uplo,
                                                    int64_t                         n,
                                                    int64_t                         k,
                                                    const hipblas_internal_type<T>* alpha,
                                                    const hipblas_internal_type<T>* A,
                                                    int64_t                         lda,
                                                    hipblasStride                   strideA,
                                                    const hipblas_internal_type<T>* x,
                                                    int64_t                         incx,
                                                    hipblasStride                   stridex,
                                                    const hipblas_internal_type<T>* beta,
                                                    hipblas_internal_type<T>*       y,
                                                    int64_t                         incy,
                                                    hipblasStride                   stridey,
                                                    int64_t                         batchCount);

    MAP2CF_D64(hipblasHbmvStridedBatched, std::complex<float>, hipblasChbmvStridedBatched);
    MAP2CF_D64(hipblasHbmvStridedBatched, std::complex<double>, hipblasZhbmvStridedBatched);

    // hemv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHemv)(hipblasHandle_t                 handle,
                                   hipblasFillMode_t               uplo,
                                   int                             n,
                                   const hipblas_internal_type<T>* alpha,
                                   const hipblas_internal_type<T>* A,
                                   int                             lda,
                                   const hipblas_internal_type<T>* x,
                                   int                             incx,
                                   const hipblas_internal_type<T>* beta,
                                   hipblas_internal_type<T>*       y,
                                   int                             incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHemv_64)(hipblasHandle_t                 handle,
                                      hipblasFillMode_t               uplo,
                                      int64_t                         n,
                                      const hipblas_internal_type<T>* alpha,
                                      const hipblas_internal_type<T>* A,
                                      int64_t                         lda,
                                      const hipblas_internal_type<T>* x,
                                      int64_t                         incx,
                                      const hipblas_internal_type<T>* beta,
                                      hipblas_internal_type<T>*       y,
                                      int64_t                         incy);

    MAP2CF_D64(hipblasHemv, std::complex<float>, hipblasChemv);
    MAP2CF_D64(hipblasHemv, std::complex<double>, hipblasZhemv);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHemvBatched)(hipblasHandle_t                       handle,
                                          hipblasFillMode_t                     uplo,
                                          int                                   n,
                                          const hipblas_internal_type<T>*       alpha,
                                          const hipblas_internal_type<T>* const A[],
                                          int                                   lda,
                                          const hipblas_internal_type<T>* const x[],
                                          int                                   incx,
                                          const hipblas_internal_type<T>*       beta,
                                          hipblas_internal_type<T>* const       y[],
                                          int                                   incy,
                                          int                                   batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHemvBatched_64)(hipblasHandle_t                       handle,
                                             hipblasFillMode_t                     uplo,
                                             int64_t                               n,
                                             const hipblas_internal_type<T>*       alpha,
                                             const hipblas_internal_type<T>* const A[],
                                             int64_t                               lda,
                                             const hipblas_internal_type<T>* const x[],
                                             int64_t                               incx,
                                             const hipblas_internal_type<T>*       beta,
                                             hipblas_internal_type<T>* const       y[],
                                             int64_t                               incy,
                                             int64_t                               batch_count);

    MAP2CF_D64(hipblasHemvBatched, std::complex<float>, hipblasChemvBatched);
    MAP2CF_D64(hipblasHemvBatched, std::complex<double>, hipblasZhemvBatched);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHemvStridedBatched)(hipblasHandle_t                 handle,
                                                 hipblasFillMode_t               uplo,
                                                 int                             n,
                                                 const hipblas_internal_type<T>* alpha,
                                                 const hipblas_internal_type<T>* A,
                                                 int                             lda,
                                                 hipblasStride                   stride_a,
                                                 const hipblas_internal_type<T>* x,
                                                 int                             incx,
                                                 hipblasStride                   stride_x,
                                                 const hipblas_internal_type<T>* beta,
                                                 hipblas_internal_type<T>*       y,
                                                 int                             incy,
                                                 hipblasStride                   stride_y,
                                                 int                             batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHemvStridedBatched_64)(hipblasHandle_t                 handle,
                                                    hipblasFillMode_t               uplo,
                                                    int64_t                         n,
                                                    const hipblas_internal_type<T>* alpha,
                                                    const hipblas_internal_type<T>* A,
                                                    int64_t                         lda,
                                                    hipblasStride                   stride_a,
                                                    const hipblas_internal_type<T>* x,
                                                    int64_t                         incx,
                                                    hipblasStride                   stride_x,
                                                    const hipblas_internal_type<T>* beta,
                                                    hipblas_internal_type<T>*       y,
                                                    int64_t                         incy,
                                                    hipblasStride                   stride_y,
                                                    int64_t                         batch_count);

    MAP2CF_D64(hipblasHemvStridedBatched, std::complex<float>, hipblasChemvStridedBatched);
    MAP2CF_D64(hipblasHemvStridedBatched, std::complex<double>, hipblasZhemvStridedBatched);

    // her
    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHer)(hipblasHandle_t                 handle,
                                  hipblasFillMode_t               uplo,
                                  int                             n,
                                  const hipblas_internal_type<U>* alpha,
                                  const hipblas_internal_type<T>* x,
                                  int                             incx,
                                  hipblas_internal_type<T>*       A,
                                  int                             lda);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHer_64)(hipblasHandle_t                 handle,
                                     hipblasFillMode_t               uplo,
                                     int64_t                         n,
                                     const hipblas_internal_type<U>* alpha,
                                     const hipblas_internal_type<T>* x,
                                     int64_t                         incx,
                                     hipblas_internal_type<T>*       A,
                                     int64_t                         lda);

    MAP2CF_D64(hipblasHer, std::complex<float>, float, hipblasCher);
    MAP2CF_D64(hipblasHer, std::complex<double>, double, hipblasZher);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerBatched)(hipblasHandle_t                       handle,
                                         hipblasFillMode_t                     uplo,
                                         int                                   n,
                                         const hipblas_internal_type<U>*       alpha,
                                         const hipblas_internal_type<T>* const x[],
                                         int                                   incx,
                                         hipblas_internal_type<T>* const       A[],
                                         int                                   lda,
                                         int                                   batchCount);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerBatched_64)(hipblasHandle_t                       handle,
                                            hipblasFillMode_t                     uplo,
                                            int64_t                               n,
                                            const hipblas_internal_type<U>*       alpha,
                                            const hipblas_internal_type<T>* const x[],
                                            int64_t                               incx,
                                            hipblas_internal_type<T>* const       A[],
                                            int64_t                               lda,
                                            int64_t                               batchCount);

    MAP2CF_D64(hipblasHerBatched, std::complex<float>, float, hipblasCherBatched);
    MAP2CF_D64(hipblasHerBatched, std::complex<double>, double, hipblasZherBatched);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerStridedBatched)(hipblasHandle_t                 handle,
                                                hipblasFillMode_t               uplo,
                                                int                             n,
                                                const hipblas_internal_type<U>* alpha,
                                                const hipblas_internal_type<T>* x,
                                                int                             incx,
                                                hipblasStride                   stridex,
                                                hipblas_internal_type<T>*       A,
                                                int                             lda,
                                                hipblasStride                   strideA,
                                                int                             batchCount);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerStridedBatched_64)(hipblasHandle_t                 handle,
                                                   hipblasFillMode_t               uplo,
                                                   int64_t                         n,
                                                   const hipblas_internal_type<U>* alpha,
                                                   const hipblas_internal_type<T>* x,
                                                   int64_t                         incx,
                                                   hipblasStride                   stridex,
                                                   hipblas_internal_type<T>*       A,
                                                   int64_t                         lda,
                                                   hipblasStride                   strideA,
                                                   int64_t                         batchCount);

    MAP2CF_D64(hipblasHerStridedBatched, std::complex<float>, float, hipblasCherStridedBatched);
    MAP2CF_D64(hipblasHerStridedBatched, std::complex<double>, double, hipblasZherStridedBatched);

    // her2
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHer2)(hipblasHandle_t                 handle,
                                   hipblasFillMode_t               uplo,
                                   int                             n,
                                   const hipblas_internal_type<T>* alpha,
                                   const hipblas_internal_type<T>* x,
                                   int                             incx,
                                   const hipblas_internal_type<T>* y,
                                   int                             incy,
                                   hipblas_internal_type<T>*       A,
                                   int                             lda);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHer2_64)(hipblasHandle_t                 handle,
                                      hipblasFillMode_t               uplo,
                                      int64_t                         n,
                                      const hipblas_internal_type<T>* alpha,
                                      const hipblas_internal_type<T>* x,
                                      int64_t                         incx,
                                      const hipblas_internal_type<T>* y,
                                      int64_t                         incy,
                                      hipblas_internal_type<T>*       A,
                                      int64_t                         lda);

    MAP2CF_D64(hipblasHer2, std::complex<float>, hipblasCher2);
    MAP2CF_D64(hipblasHer2, std::complex<double>, hipblasZher2);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHer2Batched)(hipblasHandle_t                       handle,
                                          hipblasFillMode_t                     uplo,
                                          int                                   n,
                                          const hipblas_internal_type<T>*       alpha,
                                          const hipblas_internal_type<T>* const x[],
                                          int                                   incx,
                                          const hipblas_internal_type<T>* const y[],
                                          int                                   incy,
                                          hipblas_internal_type<T>* const       A[],
                                          int                                   lda,
                                          int                                   batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHer2Batched_64)(hipblasHandle_t                       handle,
                                             hipblasFillMode_t                     uplo,
                                             int64_t                               n,
                                             const hipblas_internal_type<T>*       alpha,
                                             const hipblas_internal_type<T>* const x[],
                                             int64_t                               incx,
                                             const hipblas_internal_type<T>* const y[],
                                             int64_t                               incy,
                                             hipblas_internal_type<T>* const       A[],
                                             int64_t                               lda,
                                             int64_t                               batchCount);

    MAP2CF_D64(hipblasHer2Batched, std::complex<float>, hipblasCher2Batched);
    MAP2CF_D64(hipblasHer2Batched, std::complex<double>, hipblasZher2Batched);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHer2StridedBatched)(hipblasHandle_t                 handle,
                                                 hipblasFillMode_t               uplo,
                                                 int                             n,
                                                 const hipblas_internal_type<T>* alpha,
                                                 const hipblas_internal_type<T>* x,
                                                 int                             incx,
                                                 hipblasStride                   stridex,
                                                 const hipblas_internal_type<T>* y,
                                                 int                             incy,
                                                 hipblasStride                   stridey,
                                                 hipblas_internal_type<T>*       A,
                                                 int                             lda,
                                                 hipblasStride                   strideA,
                                                 int                             batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHer2StridedBatched_64)(hipblasHandle_t                 handle,
                                                    hipblasFillMode_t               uplo,
                                                    int64_t                         n,
                                                    const hipblas_internal_type<T>* alpha,
                                                    const hipblas_internal_type<T>* x,
                                                    int64_t                         incx,
                                                    hipblasStride                   stridex,
                                                    const hipblas_internal_type<T>* y,
                                                    int64_t                         incy,
                                                    hipblasStride                   stridey,
                                                    hipblas_internal_type<T>*       A,
                                                    int64_t                         lda,
                                                    hipblasStride                   strideA,
                                                    int64_t                         batchCount);

    MAP2CF_D64(hipblasHer2StridedBatched, std::complex<float>, hipblasCher2StridedBatched);
    MAP2CF_D64(hipblasHer2StridedBatched, std::complex<double>, hipblasZher2StridedBatched);

    // hpmv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHpmv)(hipblasHandle_t                 handle,
                                   hipblasFillMode_t               uplo,
                                   int                             n,
                                   const hipblas_internal_type<T>* alpha,
                                   const hipblas_internal_type<T>* AP,
                                   const hipblas_internal_type<T>* x,
                                   int                             incx,
                                   const hipblas_internal_type<T>* beta,
                                   hipblas_internal_type<T>*       y,
                                   int                             incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHpmv_64)(hipblasHandle_t                 handle,
                                      hipblasFillMode_t               uplo,
                                      int64_t                         n,
                                      const hipblas_internal_type<T>* alpha,
                                      const hipblas_internal_type<T>* AP,
                                      const hipblas_internal_type<T>* x,
                                      int64_t                         incx,
                                      const hipblas_internal_type<T>* beta,
                                      hipblas_internal_type<T>*       y,
                                      int64_t                         incy);

    MAP2CF_D64(hipblasHpmv, std::complex<float>, hipblasChpmv);
    MAP2CF_D64(hipblasHpmv, std::complex<double>, hipblasZhpmv);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHpmvBatched)(hipblasHandle_t                       handle,
                                          hipblasFillMode_t                     uplo,
                                          int                                   n,
                                          const hipblas_internal_type<T>*       alpha,
                                          const hipblas_internal_type<T>* const AP[],
                                          const hipblas_internal_type<T>* const x[],
                                          int                                   incx,
                                          const hipblas_internal_type<T>*       beta,
                                          hipblas_internal_type<T>* const       y[],
                                          int                                   incy,
                                          int                                   batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHpmvBatched_64)(hipblasHandle_t                       handle,
                                             hipblasFillMode_t                     uplo,
                                             int64_t                               n,
                                             const hipblas_internal_type<T>*       alpha,
                                             const hipblas_internal_type<T>* const AP[],
                                             const hipblas_internal_type<T>* const x[],
                                             int64_t                               incx,
                                             const hipblas_internal_type<T>*       beta,
                                             hipblas_internal_type<T>* const       y[],
                                             int64_t                               incy,
                                             int64_t                               batchCount);

    MAP2CF_D64(hipblasHpmvBatched, std::complex<float>, hipblasChpmvBatched);
    MAP2CF_D64(hipblasHpmvBatched, std::complex<double>, hipblasZhpmvBatched);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHpmvStridedBatched)(hipblasHandle_t                 handle,
                                                 hipblasFillMode_t               uplo,
                                                 int                             n,
                                                 const hipblas_internal_type<T>* alpha,
                                                 const hipblas_internal_type<T>* AP,
                                                 hipblasStride                   strideAP,
                                                 const hipblas_internal_type<T>* x,
                                                 int                             incx,
                                                 hipblasStride                   stridex,
                                                 const hipblas_internal_type<T>* beta,
                                                 hipblas_internal_type<T>*       y,
                                                 int                             incy,
                                                 hipblasStride                   stridey,
                                                 int                             batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHpmvStridedBatched_64)(hipblasHandle_t                 handle,
                                                    hipblasFillMode_t               uplo,
                                                    int64_t                         n,
                                                    const hipblas_internal_type<T>* alpha,
                                                    const hipblas_internal_type<T>* AP,
                                                    hipblasStride                   strideAP,
                                                    const hipblas_internal_type<T>* x,
                                                    int64_t                         incx,
                                                    hipblasStride                   stridex,
                                                    const hipblas_internal_type<T>* beta,
                                                    hipblas_internal_type<T>*       y,
                                                    int64_t                         incy,
                                                    hipblasStride                   stridey,
                                                    int64_t                         batchCount);

    MAP2CF_D64(hipblasHpmvStridedBatched, std::complex<float>, hipblasChpmvStridedBatched);
    MAP2CF_D64(hipblasHpmvStridedBatched, std::complex<double>, hipblasZhpmvStridedBatched);

    // hpr
    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHpr)(hipblasHandle_t                 handle,
                                  hipblasFillMode_t               uplo,
                                  int                             n,
                                  const hipblas_internal_type<U>* alpha,
                                  const hipblas_internal_type<T>* x,
                                  int                             incx,
                                  hipblas_internal_type<T>*       AP);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHpr_64)(hipblasHandle_t                 handle,
                                     hipblasFillMode_t               uplo,
                                     int64_t                         n,
                                     const hipblas_internal_type<U>* alpha,
                                     const hipblas_internal_type<T>* x,
                                     int64_t                         incx,
                                     hipblas_internal_type<T>*       AP);

    MAP2CF_D64(hipblasHpr, std::complex<float>, float, hipblasChpr);
    MAP2CF_D64(hipblasHpr, std::complex<double>, double, hipblasZhpr);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHprBatched)(hipblasHandle_t                       handle,
                                         hipblasFillMode_t                     uplo,
                                         int                                   n,
                                         const hipblas_internal_type<U>*       alpha,
                                         const hipblas_internal_type<T>* const x[],
                                         int                                   incx,
                                         hipblas_internal_type<T>* const       AP[],
                                         int                                   batchCount);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHprBatched_64)(hipblasHandle_t                       handle,
                                            hipblasFillMode_t                     uplo,
                                            int64_t                               n,
                                            const hipblas_internal_type<U>*       alpha,
                                            const hipblas_internal_type<T>* const x[],
                                            int64_t                               incx,
                                            hipblas_internal_type<T>* const       AP[],
                                            int64_t                               batchCount);

    MAP2CF_D64(hipblasHprBatched, std::complex<float>, float, hipblasChprBatched);
    MAP2CF_D64(hipblasHprBatched, std::complex<double>, double, hipblasZhprBatched);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHprStridedBatched)(hipblasHandle_t                 handle,
                                                hipblasFillMode_t               uplo,
                                                int                             n,
                                                const hipblas_internal_type<U>* alpha,
                                                const hipblas_internal_type<T>* x,
                                                int                             incx,
                                                hipblasStride                   stridex,
                                                hipblas_internal_type<T>*       AP,
                                                hipblasStride                   strideAP,
                                                int                             batchCount);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHprStridedBatched_64)(hipblasHandle_t                 handle,
                                                   hipblasFillMode_t               uplo,
                                                   int64_t                         n,
                                                   const hipblas_internal_type<U>* alpha,
                                                   const hipblas_internal_type<T>* x,
                                                   int64_t                         incx,
                                                   hipblasStride                   stridex,
                                                   hipblas_internal_type<T>*       AP,
                                                   hipblasStride                   strideAP,
                                                   int64_t                         batchCount);

    MAP2CF_D64(hipblasHprStridedBatched, std::complex<float>, float, hipblasChprStridedBatched);
    MAP2CF_D64(hipblasHprStridedBatched, std::complex<double>, double, hipblasZhprStridedBatched);

    // hpr2
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHpr2)(hipblasHandle_t                 handle,
                                   hipblasFillMode_t               uplo,
                                   int                             n,
                                   const hipblas_internal_type<T>* alpha,
                                   const hipblas_internal_type<T>* x,
                                   int                             incx,
                                   const hipblas_internal_type<T>* y,
                                   int                             incy,
                                   hipblas_internal_type<T>*       AP);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHpr2_64)(hipblasHandle_t                 handle,
                                      hipblasFillMode_t               uplo,
                                      int64_t                         n,
                                      const hipblas_internal_type<T>* alpha,
                                      const hipblas_internal_type<T>* x,
                                      int64_t                         incx,
                                      const hipblas_internal_type<T>* y,
                                      int64_t                         incy,
                                      hipblas_internal_type<T>*       AP);

    MAP2CF_D64(hipblasHpr2, std::complex<float>, hipblasChpr2);
    MAP2CF_D64(hipblasHpr2, std::complex<double>, hipblasZhpr2);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHpr2Batched)(hipblasHandle_t                       handle,
                                          hipblasFillMode_t                     uplo,
                                          int                                   n,
                                          const hipblas_internal_type<T>*       alpha,
                                          const hipblas_internal_type<T>* const x[],
                                          int                                   incx,
                                          const hipblas_internal_type<T>* const y[],
                                          int                                   incy,
                                          hipblas_internal_type<T>* const       AP[],
                                          int                                   batchCount);
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHpr2Batched_64)(hipblasHandle_t                       handle,
                                             hipblasFillMode_t                     uplo,
                                             int64_t                               n,
                                             const hipblas_internal_type<T>*       alpha,
                                             const hipblas_internal_type<T>* const x[],
                                             int64_t                               incx,
                                             const hipblas_internal_type<T>* const y[],
                                             int64_t                               incy,
                                             hipblas_internal_type<T>* const       AP[],
                                             int64_t                               batchCount);

    MAP2CF_D64(hipblasHpr2Batched, std::complex<float>, hipblasChpr2Batched);
    MAP2CF_D64(hipblasHpr2Batched, std::complex<double>, hipblasZhpr2Batched);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHpr2StridedBatched)(hipblasHandle_t                 handle,
                                                 hipblasFillMode_t               uplo,
                                                 int                             n,
                                                 const hipblas_internal_type<T>* alpha,
                                                 const hipblas_internal_type<T>* x,
                                                 int                             incx,
                                                 hipblasStride                   stridex,
                                                 const hipblas_internal_type<T>* y,
                                                 int                             incy,
                                                 hipblasStride                   stridey,
                                                 hipblas_internal_type<T>*       AP,
                                                 hipblasStride                   strideAP,
                                                 int                             batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHpr2StridedBatched_64)(hipblasHandle_t                 handle,
                                                    hipblasFillMode_t               uplo,
                                                    int64_t                         n,
                                                    const hipblas_internal_type<T>* alpha,
                                                    const hipblas_internal_type<T>* x,
                                                    int64_t                         incx,
                                                    hipblasStride                   stridex,
                                                    const hipblas_internal_type<T>* y,
                                                    int64_t                         incy,
                                                    hipblasStride                   stridey,
                                                    hipblas_internal_type<T>*       AP,
                                                    hipblasStride                   strideAP,
                                                    int64_t                         batchCount);

    MAP2CF_D64(hipblasHpr2StridedBatched, std::complex<float>, hipblasChpr2StridedBatched);
    MAP2CF_D64(hipblasHpr2StridedBatched, std::complex<double>, hipblasZhpr2StridedBatched);

    // sbmv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSbmv)(hipblasHandle_t                 handle,
                                   hipblasFillMode_t               uplo,
                                   int                             n,
                                   int                             k,
                                   const hipblas_internal_type<T>* alpha,
                                   const hipblas_internal_type<T>* A,
                                   int                             lda,
                                   const hipblas_internal_type<T>* x,
                                   int                             incx,
                                   const hipblas_internal_type<T>* beta,
                                   hipblas_internal_type<T>*       y,
                                   int                             incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSbmvBatched)(hipblasHandle_t                       handle,
                                          hipblasFillMode_t                     uplo,
                                          int                                   n,
                                          int                                   k,
                                          const hipblas_internal_type<T>*       alpha,
                                          const hipblas_internal_type<T>* const A[],
                                          int                                   lda,
                                          const hipblas_internal_type<T>* const x[],
                                          int                                   incx,
                                          const hipblas_internal_type<T>*       beta,
                                          hipblas_internal_type<T>* const       y[],
                                          int                                   incy,
                                          int                                   batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSbmvStridedBatched)(hipblasHandle_t                 handle,
                                                 hipblasFillMode_t               uplo,
                                                 int                             n,
                                                 int                             k,
                                                 const hipblas_internal_type<T>* alpha,
                                                 const hipblas_internal_type<T>* A,
                                                 int                             lda,
                                                 hipblasStride                   strideA,
                                                 const hipblas_internal_type<T>* x,
                                                 int                             incx,
                                                 hipblasStride                   stridex,
                                                 const hipblas_internal_type<T>* beta,
                                                 hipblas_internal_type<T>*       y,
                                                 int                             incy,
                                                 hipblasStride                   stridey,
                                                 int                             batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSbmv_64)(hipblasHandle_t                 handle,
                                      hipblasFillMode_t               uplo,
                                      int64_t                         n,
                                      int64_t                         k,
                                      const hipblas_internal_type<T>* alpha,
                                      const hipblas_internal_type<T>* A,
                                      int64_t                         lda,
                                      const hipblas_internal_type<T>* x,
                                      int64_t                         incx,
                                      const hipblas_internal_type<T>* beta,
                                      hipblas_internal_type<T>*       y,
                                      int64_t                         incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSbmvBatched_64)(hipblasHandle_t                       handle,
                                             hipblasFillMode_t                     uplo,
                                             int64_t                               n,
                                             int64_t                               k,
                                             const hipblas_internal_type<T>*       alpha,
                                             const hipblas_internal_type<T>* const A[],
                                             int64_t                               lda,
                                             const hipblas_internal_type<T>* const x[],
                                             int64_t                               incx,
                                             const hipblas_internal_type<T>*       beta,
                                             hipblas_internal_type<T>* const       y[],
                                             int64_t                               incy,
                                             int64_t                               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSbmvStridedBatched_64)(hipblasHandle_t                 handle,
                                                    hipblasFillMode_t               uplo,
                                                    int64_t                         n,
                                                    int64_t                         k,
                                                    const hipblas_internal_type<T>* alpha,
                                                    const hipblas_internal_type<T>* A,
                                                    int64_t                         lda,
                                                    hipblasStride                   strideA,
                                                    const hipblas_internal_type<T>* x,
                                                    int64_t                         incx,
                                                    hipblasStride                   stridex,
                                                    const hipblas_internal_type<T>* beta,
                                                    hipblas_internal_type<T>*       y,
                                                    int64_t                         incy,
                                                    hipblasStride                   stridey,
                                                    int64_t                         batchCount);

    MAP2CF_D64(hipblasSbmv, float, hipblasSsbmv);
    MAP2CF_D64(hipblasSbmv, double, hipblasDsbmv);

    MAP2CF_D64(hipblasSbmvBatched, float, hipblasSsbmvBatched);
    MAP2CF_D64(hipblasSbmvBatched, double, hipblasDsbmvBatched);

    MAP2CF_D64(hipblasSbmvStridedBatched, float, hipblasSsbmvStridedBatched);
    MAP2CF_D64(hipblasSbmvStridedBatched, double, hipblasDsbmvStridedBatched);

    // spmv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSpmv)(hipblasHandle_t                 handle,
                                   hipblasFillMode_t               uplo,
                                   int                             n,
                                   const hipblas_internal_type<T>* alpha,
                                   const hipblas_internal_type<T>* AP,
                                   const hipblas_internal_type<T>* x,
                                   int                             incx,
                                   const hipblas_internal_type<T>* beta,
                                   hipblas_internal_type<T>*       y,
                                   int                             incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSpmvBatched)(hipblasHandle_t                       handle,
                                          hipblasFillMode_t                     uplo,
                                          int                                   n,
                                          const hipblas_internal_type<T>*       alpha,
                                          const hipblas_internal_type<T>* const AP[],
                                          const hipblas_internal_type<T>* const x[],
                                          int                                   incx,
                                          const hipblas_internal_type<T>*       beta,
                                          hipblas_internal_type<T>* const       y[],
                                          int                                   incy,
                                          int                                   batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSpmvStridedBatched)(hipblasHandle_t                 handle,
                                                 hipblasFillMode_t               uplo,
                                                 int                             n,
                                                 const hipblas_internal_type<T>* alpha,
                                                 const hipblas_internal_type<T>* AP,
                                                 hipblasStride                   strideAP,
                                                 const hipblas_internal_type<T>* x,
                                                 int                             incx,
                                                 hipblasStride                   stridex,
                                                 const hipblas_internal_type<T>* beta,
                                                 hipblas_internal_type<T>*       y,
                                                 int                             incy,
                                                 hipblasStride                   stridey,
                                                 int                             batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSpmv_64)(hipblasHandle_t                 handle,
                                      hipblasFillMode_t               uplo,
                                      int64_t                         n,
                                      const hipblas_internal_type<T>* alpha,
                                      const hipblas_internal_type<T>* AP,
                                      const hipblas_internal_type<T>* x,
                                      int64_t                         incx,
                                      const hipblas_internal_type<T>* beta,
                                      hipblas_internal_type<T>*       y,
                                      int64_t                         incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSpmvBatched_64)(hipblasHandle_t                       handle,
                                             hipblasFillMode_t                     uplo,
                                             int64_t                               n,
                                             const hipblas_internal_type<T>*       alpha,
                                             const hipblas_internal_type<T>* const AP[],
                                             const hipblas_internal_type<T>* const x[],
                                             int64_t                               incx,
                                             const hipblas_internal_type<T>*       beta,
                                             hipblas_internal_type<T>* const       y[],
                                             int64_t                               incy,
                                             int64_t                               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSpmvStridedBatched_64)(hipblasHandle_t                 handle,
                                                    hipblasFillMode_t               uplo,
                                                    int64_t                         n,
                                                    const hipblas_internal_type<T>* alpha,
                                                    const hipblas_internal_type<T>* AP,
                                                    hipblasStride                   strideAP,
                                                    const hipblas_internal_type<T>* x,
                                                    int64_t                         incx,
                                                    hipblasStride                   stridex,
                                                    const hipblas_internal_type<T>* beta,
                                                    hipblas_internal_type<T>*       y,
                                                    int64_t                         incy,
                                                    hipblasStride                   stridey,
                                                    int64_t                         batchCount);

    MAP2CF_D64(hipblasSpmv, float, hipblasSspmv);
    MAP2CF_D64(hipblasSpmv, double, hipblasDspmv);

    MAP2CF_D64(hipblasSpmvBatched, float, hipblasSspmvBatched);
    MAP2CF_D64(hipblasSpmvBatched, double, hipblasDspmvBatched);

    MAP2CF_D64(hipblasSpmvStridedBatched, float, hipblasSspmvStridedBatched);
    MAP2CF_D64(hipblasSpmvStridedBatched, double, hipblasDspmvStridedBatched);

    // spr
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSpr)(hipblasHandle_t                 handle,
                                  hipblasFillMode_t               uplo,
                                  int                             n,
                                  const hipblas_internal_type<T>* alpha,
                                  const hipblas_internal_type<T>* x,
                                  int                             incx,
                                  hipblas_internal_type<T>*       AP);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSprBatched)(hipblasHandle_t                       handle,
                                         hipblasFillMode_t                     uplo,
                                         int                                   n,
                                         const hipblas_internal_type<T>*       alpha,
                                         const hipblas_internal_type<T>* const x[],
                                         int                                   incx,
                                         hipblas_internal_type<T>* const       AP[],
                                         int                                   batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSprStridedBatched)(hipblasHandle_t                 handle,
                                                hipblasFillMode_t               uplo,
                                                int                             n,
                                                const hipblas_internal_type<T>* alpha,
                                                const hipblas_internal_type<T>* x,
                                                int                             incx,
                                                hipblasStride                   stridex,
                                                hipblas_internal_type<T>*       AP,
                                                hipblasStride                   strideAP,
                                                int                             batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSpr_64)(hipblasHandle_t                 handle,
                                     hipblasFillMode_t               uplo,
                                     int64_t                         n,
                                     const hipblas_internal_type<T>* alpha,
                                     const hipblas_internal_type<T>* x,
                                     int64_t                         incx,
                                     hipblas_internal_type<T>*       AP);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSprBatched_64)(hipblasHandle_t                       handle,
                                            hipblasFillMode_t                     uplo,
                                            int64_t                               n,
                                            const hipblas_internal_type<T>*       alpha,
                                            const hipblas_internal_type<T>* const x[],
                                            int64_t                               incx,
                                            hipblas_internal_type<T>* const       AP[],
                                            int64_t                               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSprStridedBatched_64)(hipblasHandle_t                 handle,
                                                   hipblasFillMode_t               uplo,
                                                   int64_t                         n,
                                                   const hipblas_internal_type<T>* alpha,
                                                   const hipblas_internal_type<T>* x,
                                                   int64_t                         incx,
                                                   hipblasStride                   stridex,
                                                   hipblas_internal_type<T>*       AP,
                                                   hipblasStride                   strideAP,
                                                   int64_t                         batchCount);

    MAP2CF_D64(hipblasSpr, float, hipblasSspr);
    MAP2CF_D64(hipblasSpr, double, hipblasDspr);
    MAP2CF_D64(hipblasSpr, std::complex<float>, hipblasCspr);
    MAP2CF_D64(hipblasSpr, std::complex<double>, hipblasZspr);

    MAP2CF_D64(hipblasSprBatched, float, hipblasSsprBatched);
    MAP2CF_D64(hipblasSprBatched, double, hipblasDsprBatched);
    MAP2CF_D64(hipblasSprBatched, std::complex<float>, hipblasCsprBatched);
    MAP2CF_D64(hipblasSprBatched, std::complex<double>, hipblasZsprBatched);

    MAP2CF_D64(hipblasSprStridedBatched, float, hipblasSsprStridedBatched);
    MAP2CF_D64(hipblasSprStridedBatched, double, hipblasDsprStridedBatched);
    MAP2CF_D64(hipblasSprStridedBatched, std::complex<float>, hipblasCsprStridedBatched);
    MAP2CF_D64(hipblasSprStridedBatched, std::complex<double>, hipblasZsprStridedBatched);

    // spr2
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSpr2)(hipblasHandle_t                 handle,
                                   hipblasFillMode_t               uplo,
                                   int                             n,
                                   const hipblas_internal_type<T>* alpha,
                                   const hipblas_internal_type<T>* x,
                                   int                             incx,
                                   const hipblas_internal_type<T>* y,
                                   int                             incy,
                                   hipblas_internal_type<T>*       AP);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSpr2Batched)(hipblasHandle_t                       handle,
                                          hipblasFillMode_t                     uplo,
                                          int                                   n,
                                          const hipblas_internal_type<T>*       alpha,
                                          const hipblas_internal_type<T>* const x[],
                                          int                                   incx,
                                          const hipblas_internal_type<T>* const y[],
                                          int                                   incy,
                                          hipblas_internal_type<T>* const       AP[],
                                          int                                   batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSpr2StridedBatched)(hipblasHandle_t                 handle,
                                                 hipblasFillMode_t               uplo,
                                                 int                             n,
                                                 const hipblas_internal_type<T>* alpha,
                                                 const hipblas_internal_type<T>* x,
                                                 int                             incx,
                                                 hipblasStride                   stridex,
                                                 const hipblas_internal_type<T>* y,
                                                 int                             incy,
                                                 hipblasStride                   stridey,
                                                 hipblas_internal_type<T>*       AP,
                                                 hipblasStride                   strideAP,
                                                 int                             batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSpr2_64)(hipblasHandle_t                 handle,
                                      hipblasFillMode_t               uplo,
                                      int64_t                         n,
                                      const hipblas_internal_type<T>* alpha,
                                      const hipblas_internal_type<T>* x,
                                      int64_t                         incx,
                                      const hipblas_internal_type<T>* y,
                                      int64_t                         incy,
                                      hipblas_internal_type<T>*       AP);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSpr2Batched_64)(hipblasHandle_t                       handle,
                                             hipblasFillMode_t                     uplo,
                                             int64_t                               n,
                                             const hipblas_internal_type<T>*       alpha,
                                             const hipblas_internal_type<T>* const x[],
                                             int64_t                               incx,
                                             const hipblas_internal_type<T>* const y[],
                                             int64_t                               incy,
                                             hipblas_internal_type<T>* const       AP[],
                                             int64_t                               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSpr2StridedBatched_64)(hipblasHandle_t                 handle,
                                                    hipblasFillMode_t               uplo,
                                                    int64_t                         n,
                                                    const hipblas_internal_type<T>* alpha,
                                                    const hipblas_internal_type<T>* x,
                                                    int64_t                         incx,
                                                    hipblasStride                   stridex,
                                                    const hipblas_internal_type<T>* y,
                                                    int64_t                         incy,
                                                    hipblasStride                   stridey,
                                                    hipblas_internal_type<T>*       AP,
                                                    hipblasStride                   strideAP,
                                                    int64_t                         batchCount);

    MAP2CF_D64(hipblasSpr2, float, hipblasSspr2);
    MAP2CF_D64(hipblasSpr2, double, hipblasDspr2);

    MAP2CF_D64(hipblasSpr2Batched, float, hipblasSspr2Batched);
    MAP2CF_D64(hipblasSpr2Batched, double, hipblasDspr2Batched);

    MAP2CF_D64(hipblasSpr2StridedBatched, float, hipblasSspr2StridedBatched);
    MAP2CF_D64(hipblasSpr2StridedBatched, double, hipblasDspr2StridedBatched);

    // symv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSymv)(hipblasHandle_t                 handle,
                                   hipblasFillMode_t               uplo,
                                   int                             n,
                                   const hipblas_internal_type<T>* alpha,
                                   const hipblas_internal_type<T>* A,
                                   int                             lda,
                                   const hipblas_internal_type<T>* x,
                                   int                             incx,
                                   const hipblas_internal_type<T>* beta,
                                   hipblas_internal_type<T>*       y,
                                   int                             incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSymvBatched)(hipblasHandle_t                       handle,
                                          hipblasFillMode_t                     uplo,
                                          int                                   n,
                                          const hipblas_internal_type<T>*       alpha,
                                          const hipblas_internal_type<T>* const A[],
                                          int                                   lda,
                                          const hipblas_internal_type<T>* const x[],
                                          int                                   incx,
                                          const hipblas_internal_type<T>*       beta,
                                          hipblas_internal_type<T>* const       y[],
                                          int                                   incy,
                                          int                                   batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSymvStridedBatched)(hipblasHandle_t                 handle,
                                                 hipblasFillMode_t               uplo,
                                                 int                             n,
                                                 const hipblas_internal_type<T>* alpha,
                                                 const hipblas_internal_type<T>* A,
                                                 int                             lda,
                                                 hipblasStride                   strideA,
                                                 const hipblas_internal_type<T>* x,
                                                 int                             incx,
                                                 hipblasStride                   stridex,
                                                 const hipblas_internal_type<T>* beta,
                                                 hipblas_internal_type<T>*       y,
                                                 int                             incy,
                                                 hipblasStride                   stridey,
                                                 int                             batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSymv_64)(hipblasHandle_t                 handle,
                                      hipblasFillMode_t               uplo,
                                      int64_t                         n,
                                      const hipblas_internal_type<T>* alpha,
                                      const hipblas_internal_type<T>* A,
                                      int64_t                         lda,
                                      const hipblas_internal_type<T>* x,
                                      int64_t                         incx,
                                      const hipblas_internal_type<T>* beta,
                                      hipblas_internal_type<T>*       y,
                                      int64_t                         incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSymvBatched_64)(hipblasHandle_t                       handle,
                                             hipblasFillMode_t                     uplo,
                                             int64_t                               n,
                                             const hipblas_internal_type<T>*       alpha,
                                             const hipblas_internal_type<T>* const A[],
                                             int64_t                               lda,
                                             const hipblas_internal_type<T>* const x[],
                                             int64_t                               incx,
                                             const hipblas_internal_type<T>*       beta,
                                             hipblas_internal_type<T>* const       y[],
                                             int64_t                               incy,
                                             int64_t                               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSymvStridedBatched_64)(hipblasHandle_t                 handle,
                                                    hipblasFillMode_t               uplo,
                                                    int64_t                         n,
                                                    const hipblas_internal_type<T>* alpha,
                                                    const hipblas_internal_type<T>* A,
                                                    int64_t                         lda,
                                                    hipblasStride                   strideA,
                                                    const hipblas_internal_type<T>* x,
                                                    int64_t                         incx,
                                                    hipblasStride                   stridex,
                                                    const hipblas_internal_type<T>* beta,
                                                    hipblas_internal_type<T>*       y,
                                                    int64_t                         incy,
                                                    hipblasStride                   stridey,
                                                    int64_t                         batchCount);

    MAP2CF_D64(hipblasSymv, float, hipblasSsymv);
    MAP2CF_D64(hipblasSymv, double, hipblasDsymv);
    MAP2CF_D64(hipblasSymv, std::complex<float>, hipblasCsymv);
    MAP2CF_D64(hipblasSymv, std::complex<double>, hipblasZsymv);

    MAP2CF_D64(hipblasSymvBatched, float, hipblasSsymvBatched);
    MAP2CF_D64(hipblasSymvBatched, double, hipblasDsymvBatched);
    MAP2CF_D64(hipblasSymvBatched, std::complex<float>, hipblasCsymvBatched);
    MAP2CF_D64(hipblasSymvBatched, std::complex<double>, hipblasZsymvBatched);

    MAP2CF_D64(hipblasSymvStridedBatched, float, hipblasSsymvStridedBatched);
    MAP2CF_D64(hipblasSymvStridedBatched, double, hipblasDsymvStridedBatched);
    MAP2CF_D64(hipblasSymvStridedBatched, std::complex<float>, hipblasCsymvStridedBatched);
    MAP2CF_D64(hipblasSymvStridedBatched, std::complex<double>, hipblasZsymvStridedBatched);

    // syr
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyr)(hipblasHandle_t                 handle,
                                  hipblasFillMode_t               uplo,
                                  int                             n,
                                  const hipblas_internal_type<T>* alpha,
                                  const hipblas_internal_type<T>* x,
                                  int                             incx,
                                  hipblas_internal_type<T>*       A,
                                  int                             lda);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrBatched)(hipblasHandle_t                       handle,
                                         hipblasFillMode_t                     uplo,
                                         int                                   n,
                                         const hipblas_internal_type<T>*       alpha,
                                         const hipblas_internal_type<T>* const x[],
                                         int                                   incx,
                                         hipblas_internal_type<T>* const       A[],
                                         int                                   lda,
                                         int                                   batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrStridedBatched)(hipblasHandle_t                 handle,
                                                hipblasFillMode_t               uplo,
                                                int                             n,
                                                const hipblas_internal_type<T>* alpha,
                                                const hipblas_internal_type<T>* x,
                                                int                             incx,
                                                hipblasStride                   stridex,
                                                hipblas_internal_type<T>*       A,
                                                int                             lda,
                                                hipblasStride                   strideA,
                                                int                             batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyr_64)(hipblasHandle_t                 handle,
                                     hipblasFillMode_t               uplo,
                                     int64_t                         n,
                                     const hipblas_internal_type<T>* alpha,
                                     const hipblas_internal_type<T>* x,
                                     int64_t                         incx,
                                     hipblas_internal_type<T>*       A,
                                     int64_t                         lda);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrBatched_64)(hipblasHandle_t                       handle,
                                            hipblasFillMode_t                     uplo,
                                            int64_t                               n,
                                            const hipblas_internal_type<T>*       alpha,
                                            const hipblas_internal_type<T>* const x[],
                                            int64_t                               incx,
                                            hipblas_internal_type<T>* const       A[],
                                            int64_t                               lda,
                                            int64_t                               batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrStridedBatched_64)(hipblasHandle_t                 handle,
                                                   hipblasFillMode_t               uplo,
                                                   int64_t                         n,
                                                   const hipblas_internal_type<T>* alpha,
                                                   const hipblas_internal_type<T>* x,
                                                   int64_t                         incx,
                                                   hipblasStride                   stridex,
                                                   hipblas_internal_type<T>*       A,
                                                   int64_t                         lda,
                                                   hipblasStride                   strideA,
                                                   int64_t                         batch_count);

    MAP2CF_D64(hipblasSyr, float, hipblasSsyr);
    MAP2CF_D64(hipblasSyr, double, hipblasDsyr);
    MAP2CF_D64(hipblasSyr, std::complex<float>, hipblasCsyr);
    MAP2CF_D64(hipblasSyr, std::complex<double>, hipblasZsyr);

    MAP2CF_D64(hipblasSyrBatched, float, hipblasSsyrBatched);
    MAP2CF_D64(hipblasSyrBatched, double, hipblasDsyrBatched);
    MAP2CF_D64(hipblasSyrBatched, std::complex<float>, hipblasCsyrBatched);
    MAP2CF_D64(hipblasSyrBatched, std::complex<double>, hipblasZsyrBatched);

    MAP2CF_D64(hipblasSyrStridedBatched, float, hipblasSsyrStridedBatched);
    MAP2CF_D64(hipblasSyrStridedBatched, double, hipblasDsyrStridedBatched);
    MAP2CF_D64(hipblasSyrStridedBatched, std::complex<float>, hipblasCsyrStridedBatched);
    MAP2CF_D64(hipblasSyrStridedBatched, std::complex<double>, hipblasZsyrStridedBatched);

    // syr2
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyr2)(hipblasHandle_t                 handle,
                                   hipblasFillMode_t               uplo,
                                   int                             n,
                                   const hipblas_internal_type<T>* alpha,
                                   const hipblas_internal_type<T>* x,
                                   int                             incx,
                                   const hipblas_internal_type<T>* y,
                                   int                             incy,
                                   hipblas_internal_type<T>*       A,
                                   int                             lda);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyr2Batched)(hipblasHandle_t                       handle,
                                          hipblasFillMode_t                     uplo,
                                          int                                   n,
                                          const hipblas_internal_type<T>*       alpha,
                                          const hipblas_internal_type<T>* const x[],
                                          int                                   incx,
                                          const hipblas_internal_type<T>* const y[],
                                          int                                   incy,
                                          hipblas_internal_type<T>* const       A[],
                                          int                                   lda,
                                          int                                   batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyr2StridedBatched)(hipblasHandle_t                 handle,
                                                 hipblasFillMode_t               uplo,
                                                 int                             n,
                                                 const hipblas_internal_type<T>* alpha,
                                                 const hipblas_internal_type<T>* x,
                                                 int                             incx,
                                                 hipblasStride                   stridex,
                                                 const hipblas_internal_type<T>* y,
                                                 int                             incy,
                                                 hipblasStride                   stridey,
                                                 hipblas_internal_type<T>*       A,
                                                 int                             lda,
                                                 hipblasStride                   strideA,
                                                 int                             batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyr2_64)(hipblasHandle_t                 handle,
                                      hipblasFillMode_t               uplo,
                                      int64_t                         n,
                                      const hipblas_internal_type<T>* alpha,
                                      const hipblas_internal_type<T>* x,
                                      int64_t                         incx,
                                      const hipblas_internal_type<T>* y,
                                      int64_t                         incy,
                                      hipblas_internal_type<T>*       A,
                                      int64_t                         lda);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyr2Batched_64)(hipblasHandle_t                       handle,
                                             hipblasFillMode_t                     uplo,
                                             int64_t                               n,
                                             const hipblas_internal_type<T>*       alpha,
                                             const hipblas_internal_type<T>* const x[],
                                             int64_t                               incx,
                                             const hipblas_internal_type<T>* const y[],
                                             int64_t                               incy,
                                             hipblas_internal_type<T>* const       A[],
                                             int64_t                               lda,
                                             int64_t                               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyr2StridedBatched_64)(hipblasHandle_t                 handle,
                                                    hipblasFillMode_t               uplo,
                                                    int64_t                         n,
                                                    const hipblas_internal_type<T>* alpha,
                                                    const hipblas_internal_type<T>* x,
                                                    int64_t                         incx,
                                                    hipblasStride                   stridex,
                                                    const hipblas_internal_type<T>* y,
                                                    int64_t                         incy,
                                                    hipblasStride                   stridey,
                                                    hipblas_internal_type<T>*       A,
                                                    int64_t                         lda,
                                                    hipblasStride                   strideA,
                                                    int64_t                         batchCount);

    MAP2CF_D64(hipblasSyr2, float, hipblasSsyr2);
    MAP2CF_D64(hipblasSyr2, double, hipblasDsyr2);
    MAP2CF_D64(hipblasSyr2, std::complex<float>, hipblasCsyr2);
    MAP2CF_D64(hipblasSyr2, std::complex<double>, hipblasZsyr2);

    MAP2CF_D64(hipblasSyr2Batched, float, hipblasSsyr2Batched);
    MAP2CF_D64(hipblasSyr2Batched, double, hipblasDsyr2Batched);
    MAP2CF_D64(hipblasSyr2Batched, std::complex<float>, hipblasCsyr2Batched);
    MAP2CF_D64(hipblasSyr2Batched, std::complex<double>, hipblasZsyr2Batched);

    MAP2CF_D64(hipblasSyr2StridedBatched, float, hipblasSsyr2StridedBatched);
    MAP2CF_D64(hipblasSyr2StridedBatched, double, hipblasDsyr2StridedBatched);
    MAP2CF_D64(hipblasSyr2StridedBatched, std::complex<float>, hipblasCsyr2StridedBatched);
    MAP2CF_D64(hipblasSyr2StridedBatched, std::complex<double>, hipblasZsyr2StridedBatched);

    // tbmv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTbmv)(hipblasHandle_t                 handle,
                                   hipblasFillMode_t               uplo,
                                   hipblasOperation_t              transA,
                                   hipblasDiagType_t               diag,
                                   int                             m,
                                   int                             k,
                                   const hipblas_internal_type<T>* A,
                                   int                             lda,
                                   hipblas_internal_type<T>*       x,
                                   int                             incx);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTbmvBatched)(hipblasHandle_t                       handle,
                                          hipblasFillMode_t                     uplo,
                                          hipblasOperation_t                    transA,
                                          hipblasDiagType_t                     diag,
                                          int                                   m,
                                          int                                   k,
                                          const hipblas_internal_type<T>* const A[],
                                          int                                   lda,
                                          hipblas_internal_type<T>* const       x[],
                                          int                                   incx,
                                          int                                   batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTbmvStridedBatched)(hipblasHandle_t                 handle,
                                                 hipblasFillMode_t               uplo,
                                                 hipblasOperation_t              transA,
                                                 hipblasDiagType_t               diag,
                                                 int                             m,
                                                 int                             k,
                                                 const hipblas_internal_type<T>* A,
                                                 int                             lda,
                                                 hipblasStride                   stride_a,
                                                 hipblas_internal_type<T>*       x,
                                                 int                             incx,
                                                 hipblasStride                   stride_x,
                                                 int                             batch_count);
    //tbmv_64
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTbmv_64)(hipblasHandle_t                 handle,
                                      hipblasFillMode_t               uplo,
                                      hipblasOperation_t              transA,
                                      hipblasDiagType_t               diag,
                                      int64_t                         m,
                                      int64_t                         k,
                                      const hipblas_internal_type<T>* A,
                                      int64_t                         lda,
                                      hipblas_internal_type<T>*       x,
                                      int64_t                         incx);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTbmvBatched_64)(hipblasHandle_t                       handle,
                                             hipblasFillMode_t                     uplo,
                                             hipblasOperation_t                    transA,
                                             hipblasDiagType_t                     diag,
                                             int64_t                               m,
                                             int64_t                               k,
                                             const hipblas_internal_type<T>* const A[],
                                             int64_t                               lda,
                                             hipblas_internal_type<T>* const       x[],
                                             int64_t                               incx,
                                             int64_t                               batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTbmvStridedBatched_64)(hipblasHandle_t                 handle,
                                                    hipblasFillMode_t               uplo,
                                                    hipblasOperation_t              transA,
                                                    hipblasDiagType_t               diag,
                                                    int64_t                         m,
                                                    int64_t                         k,
                                                    const hipblas_internal_type<T>* A,
                                                    int64_t                         lda,
                                                    hipblasStride                   stride_a,
                                                    hipblas_internal_type<T>*       x,
                                                    int64_t                         incx,
                                                    hipblasStride                   stride_x,
                                                    int64_t                         batch_count);

    MAP2CF_D64(hipblasTbmv, float, hipblasStbmv);
    MAP2CF_D64(hipblasTbmv, double, hipblasDtbmv);
    MAP2CF_D64(hipblasTbmv, std::complex<float>, hipblasCtbmv);
    MAP2CF_D64(hipblasTbmv, std::complex<double>, hipblasZtbmv);

    MAP2CF_D64(hipblasTbmvBatched, float, hipblasStbmvBatched);
    MAP2CF_D64(hipblasTbmvBatched, double, hipblasDtbmvBatched);
    MAP2CF_D64(hipblasTbmvBatched, std::complex<float>, hipblasCtbmvBatched);
    MAP2CF_D64(hipblasTbmvBatched, std::complex<double>, hipblasZtbmvBatched);

    MAP2CF_D64(hipblasTbmvStridedBatched, float, hipblasStbmvStridedBatched);
    MAP2CF_D64(hipblasTbmvStridedBatched, double, hipblasDtbmvStridedBatched);
    MAP2CF_D64(hipblasTbmvStridedBatched, std::complex<float>, hipblasCtbmvStridedBatched);
    MAP2CF_D64(hipblasTbmvStridedBatched, std::complex<double>, hipblasZtbmvStridedBatched);

    // tbsv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTbsv)(hipblasHandle_t                 handle,
                                   hipblasFillMode_t               uplo,
                                   hipblasOperation_t              transA,
                                   hipblasDiagType_t               diag,
                                   int                             m,
                                   int                             k,
                                   const hipblas_internal_type<T>* A,
                                   int                             lda,
                                   hipblas_internal_type<T>*       x,
                                   int                             incx);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTbsvBatched)(hipblasHandle_t                       handle,
                                          hipblasFillMode_t                     uplo,
                                          hipblasOperation_t                    transA,
                                          hipblasDiagType_t                     diag,
                                          int                                   m,
                                          int                                   k,
                                          const hipblas_internal_type<T>* const A[],
                                          int                                   lda,
                                          hipblas_internal_type<T>* const       x[],
                                          int                                   incx,
                                          int                                   batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTbsvStridedBatched)(hipblasHandle_t                 handle,
                                                 hipblasFillMode_t               uplo,
                                                 hipblasOperation_t              transA,
                                                 hipblasDiagType_t               diag,
                                                 int                             m,
                                                 int                             k,
                                                 const hipblas_internal_type<T>* A,
                                                 int                             lda,
                                                 hipblasStride                   strideA,
                                                 hipblas_internal_type<T>*       x,
                                                 int                             incx,
                                                 hipblasStride                   stridex,
                                                 int                             batchCount);

    // tbsv_64
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTbsv_64)(hipblasHandle_t                 handle,
                                      hipblasFillMode_t               uplo,
                                      hipblasOperation_t              transA,
                                      hipblasDiagType_t               diag,
                                      int64_t                         m,
                                      int64_t                         k,
                                      const hipblas_internal_type<T>* A,
                                      int64_t                         lda,
                                      hipblas_internal_type<T>*       x,
                                      int64_t                         incx);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTbsvBatched_64)(hipblasHandle_t                       handle,
                                             hipblasFillMode_t                     uplo,
                                             hipblasOperation_t                    transA,
                                             hipblasDiagType_t                     diag,
                                             int64_t                               m,
                                             int64_t                               k,
                                             const hipblas_internal_type<T>* const A[],
                                             int64_t                               lda,
                                             hipblas_internal_type<T>* const       x[],
                                             int64_t                               incx,
                                             int64_t                               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTbsvStridedBatched_64)(hipblasHandle_t                 handle,
                                                    hipblasFillMode_t               uplo,
                                                    hipblasOperation_t              transA,
                                                    hipblasDiagType_t               diag,
                                                    int64_t                         m,
                                                    int64_t                         k,
                                                    const hipblas_internal_type<T>* A,
                                                    int64_t                         lda,
                                                    hipblasStride                   strideA,
                                                    hipblas_internal_type<T>*       x,
                                                    int64_t                         incx,
                                                    hipblasStride                   stridex,
                                                    int64_t                         batchCount);

    MAP2CF_D64(hipblasTbsv, float, hipblasStbsv);
    MAP2CF_D64(hipblasTbsv, double, hipblasDtbsv);
    MAP2CF_D64(hipblasTbsv, std::complex<float>, hipblasCtbsv);
    MAP2CF_D64(hipblasTbsv, std::complex<double>, hipblasZtbsv);

    MAP2CF_D64(hipblasTbsvBatched, float, hipblasStbsvBatched);
    MAP2CF_D64(hipblasTbsvBatched, double, hipblasDtbsvBatched);
    MAP2CF_D64(hipblasTbsvBatched, std::complex<float>, hipblasCtbsvBatched);
    MAP2CF_D64(hipblasTbsvBatched, std::complex<double>, hipblasZtbsvBatched);

    MAP2CF_D64(hipblasTbsvStridedBatched, float, hipblasStbsvStridedBatched);
    MAP2CF_D64(hipblasTbsvStridedBatched, double, hipblasDtbsvStridedBatched);
    MAP2CF_D64(hipblasTbsvStridedBatched, std::complex<float>, hipblasCtbsvStridedBatched);
    MAP2CF_D64(hipblasTbsvStridedBatched, std::complex<double>, hipblasZtbsvStridedBatched);

    // tpmv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTpmv)(hipblasHandle_t                 handle,
                                   hipblasFillMode_t               uplo,
                                   hipblasOperation_t              transA,
                                   hipblasDiagType_t               diag,
                                   int                             m,
                                   const hipblas_internal_type<T>* AP,
                                   hipblas_internal_type<T>*       x,
                                   int                             incx);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTpmvBatched)(hipblasHandle_t                       handle,
                                          hipblasFillMode_t                     uplo,
                                          hipblasOperation_t                    transA,
                                          hipblasDiagType_t                     diag,
                                          int                                   m,
                                          const hipblas_internal_type<T>* const AP[],
                                          hipblas_internal_type<T>* const       x[],
                                          int                                   incx,
                                          int                                   batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTpmvStridedBatched)(hipblasHandle_t                 handle,
                                                 hipblasFillMode_t               uplo,
                                                 hipblasOperation_t              transA,
                                                 hipblasDiagType_t               diag,
                                                 int                             m,
                                                 const hipblas_internal_type<T>* AP,
                                                 hipblasStride                   strideAP,
                                                 hipblas_internal_type<T>*       x,
                                                 int                             incx,
                                                 hipblasStride                   stridex,
                                                 int                             batchCount);

    // tpmv_64
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTpmv_64)(hipblasHandle_t                 handle,
                                      hipblasFillMode_t               uplo,
                                      hipblasOperation_t              transA,
                                      hipblasDiagType_t               diag,
                                      int64_t                         m,
                                      const hipblas_internal_type<T>* AP,
                                      hipblas_internal_type<T>*       x,
                                      int64_t                         incx);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTpmvBatched_64)(hipblasHandle_t                       handle,
                                             hipblasFillMode_t                     uplo,
                                             hipblasOperation_t                    transA,
                                             hipblasDiagType_t                     diag,
                                             int64_t                               m,
                                             const hipblas_internal_type<T>* const AP[],
                                             hipblas_internal_type<T>* const       x[],
                                             int64_t                               incx,
                                             int64_t                               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTpmvStridedBatched_64)(hipblasHandle_t                 handle,
                                                    hipblasFillMode_t               uplo,
                                                    hipblasOperation_t              transA,
                                                    hipblasDiagType_t               diag,
                                                    int64_t                         m,
                                                    const hipblas_internal_type<T>* AP,
                                                    hipblasStride                   strideAP,
                                                    hipblas_internal_type<T>*       x,
                                                    int64_t                         incx,
                                                    hipblasStride                   stridex,
                                                    int64_t                         batchCount);

    MAP2CF_D64(hipblasTpmv, float, hipblasStpmv);
    MAP2CF_D64(hipblasTpmv, double, hipblasDtpmv);
    MAP2CF_D64(hipblasTpmv, std::complex<float>, hipblasCtpmv);
    MAP2CF_D64(hipblasTpmv, std::complex<double>, hipblasZtpmv);

    MAP2CF_D64(hipblasTpmvBatched, float, hipblasStpmvBatched);
    MAP2CF_D64(hipblasTpmvBatched, double, hipblasDtpmvBatched);
    MAP2CF_D64(hipblasTpmvBatched, std::complex<float>, hipblasCtpmvBatched);
    MAP2CF_D64(hipblasTpmvBatched, std::complex<double>, hipblasZtpmvBatched);

    MAP2CF_D64(hipblasTpmvStridedBatched, float, hipblasStpmvStridedBatched);
    MAP2CF_D64(hipblasTpmvStridedBatched, double, hipblasDtpmvStridedBatched);
    MAP2CF_D64(hipblasTpmvStridedBatched, std::complex<float>, hipblasCtpmvStridedBatched);
    MAP2CF_D64(hipblasTpmvStridedBatched, std::complex<double>, hipblasZtpmvStridedBatched);

    // tpsv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTpsv)(hipblasHandle_t                 handle,
                                   hipblasFillMode_t               uplo,
                                   hipblasOperation_t              transA,
                                   hipblasDiagType_t               diag,
                                   int                             m,
                                   const hipblas_internal_type<T>* AP,
                                   hipblas_internal_type<T>*       x,
                                   int                             incx);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTpsvBatched)(hipblasHandle_t                       handle,
                                          hipblasFillMode_t                     uplo,
                                          hipblasOperation_t                    transA,
                                          hipblasDiagType_t                     diag,
                                          int                                   m,
                                          const hipblas_internal_type<T>* const AP[],
                                          hipblas_internal_type<T>* const       x[],
                                          int                                   incx,
                                          int                                   batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTpsvStridedBatched)(hipblasHandle_t                 handle,
                                                 hipblasFillMode_t               uplo,
                                                 hipblasOperation_t              transA,
                                                 hipblasDiagType_t               diag,
                                                 int                             m,
                                                 const hipblas_internal_type<T>* AP,
                                                 hipblasStride                   strideAP,
                                                 hipblas_internal_type<T>*       x,
                                                 int                             incx,
                                                 hipblasStride                   stridex,
                                                 int                             batchCount);

    // tpsv_64
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTpsv_64)(hipblasHandle_t                 handle,
                                      hipblasFillMode_t               uplo,
                                      hipblasOperation_t              transA,
                                      hipblasDiagType_t               diag,
                                      int64_t                         m,
                                      const hipblas_internal_type<T>* AP,
                                      hipblas_internal_type<T>*       x,
                                      int64_t                         incx);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTpsvBatched_64)(hipblasHandle_t                       handle,
                                             hipblasFillMode_t                     uplo,
                                             hipblasOperation_t                    transA,
                                             hipblasDiagType_t                     diag,
                                             int64_t                               m,
                                             const hipblas_internal_type<T>* const AP[],
                                             hipblas_internal_type<T>* const       x[],
                                             int64_t                               incx,
                                             int64_t                               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTpsvStridedBatched_64)(hipblasHandle_t                 handle,
                                                    hipblasFillMode_t               uplo,
                                                    hipblasOperation_t              transA,
                                                    hipblasDiagType_t               diag,
                                                    int64_t                         m,
                                                    const hipblas_internal_type<T>* AP,
                                                    hipblasStride                   strideAP,
                                                    hipblas_internal_type<T>*       x,
                                                    int64_t                         incx,
                                                    hipblasStride                   stridex,
                                                    int64_t                         batchCount);

    MAP2CF_D64(hipblasTpsv, float, hipblasStpsv);
    MAP2CF_D64(hipblasTpsv, double, hipblasDtpsv);
    MAP2CF_D64(hipblasTpsv, std::complex<float>, hipblasCtpsv);
    MAP2CF_D64(hipblasTpsv, std::complex<double>, hipblasZtpsv);

    MAP2CF_D64(hipblasTpsvBatched, float, hipblasStpsvBatched);
    MAP2CF_D64(hipblasTpsvBatched, double, hipblasDtpsvBatched);
    MAP2CF_D64(hipblasTpsvBatched, std::complex<float>, hipblasCtpsvBatched);
    MAP2CF_D64(hipblasTpsvBatched, std::complex<double>, hipblasZtpsvBatched);

    MAP2CF_D64(hipblasTpsvStridedBatched, float, hipblasStpsvStridedBatched);
    MAP2CF_D64(hipblasTpsvStridedBatched, double, hipblasDtpsvStridedBatched);
    MAP2CF_D64(hipblasTpsvStridedBatched, std::complex<float>, hipblasCtpsvStridedBatched);
    MAP2CF_D64(hipblasTpsvStridedBatched, std::complex<double>, hipblasZtpsvStridedBatched);

    // trmv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrmv)(hipblasHandle_t                 handle,
                                   hipblasFillMode_t               uplo,
                                   hipblasOperation_t              transA,
                                   hipblasDiagType_t               diag,
                                   int                             m,
                                   const hipblas_internal_type<T>* A,
                                   int                             lda,
                                   hipblas_internal_type<T>*       x,
                                   int                             incx);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrmvBatched)(hipblasHandle_t                       handle,
                                          hipblasFillMode_t                     uplo,
                                          hipblasOperation_t                    transA,
                                          hipblasDiagType_t                     diag,
                                          int                                   m,
                                          const hipblas_internal_type<T>* const A[],
                                          int                                   lda,
                                          hipblas_internal_type<T>* const       x[],
                                          int                                   incx,
                                          int                                   batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrmvStridedBatched)(hipblasHandle_t                 handle,
                                                 hipblasFillMode_t               uplo,
                                                 hipblasOperation_t              transA,
                                                 hipblasDiagType_t               diag,
                                                 int                             m,
                                                 const hipblas_internal_type<T>* A,
                                                 int                             lda,
                                                 hipblasStride                   stride_a,
                                                 hipblas_internal_type<T>*       x,
                                                 int                             incx,
                                                 hipblasStride                   stride_x,
                                                 int                             batch_count);

    // trmv_64
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrmv_64)(hipblasHandle_t                 handle,
                                      hipblasFillMode_t               uplo,
                                      hipblasOperation_t              transA,
                                      hipblasDiagType_t               diag,
                                      int64_t                         m,
                                      const hipblas_internal_type<T>* A,
                                      int64_t                         lda,
                                      hipblas_internal_type<T>*       x,
                                      int64_t                         incx);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrmvBatched_64)(hipblasHandle_t                       handle,
                                             hipblasFillMode_t                     uplo,
                                             hipblasOperation_t                    transA,
                                             hipblasDiagType_t                     diag,
                                             int64_t                               m,
                                             const hipblas_internal_type<T>* const A[],
                                             int64_t                               lda,
                                             hipblas_internal_type<T>* const       x[],
                                             int64_t                               incx,
                                             int64_t                               batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrmvStridedBatched_64)(hipblasHandle_t                 handle,
                                                    hipblasFillMode_t               uplo,
                                                    hipblasOperation_t              transA,
                                                    hipblasDiagType_t               diag,
                                                    int64_t                         m,
                                                    const hipblas_internal_type<T>* A,
                                                    int64_t                         lda,
                                                    hipblasStride                   stride_a,
                                                    hipblas_internal_type<T>*       x,
                                                    int64_t                         incx,
                                                    hipblasStride                   stride_x,
                                                    int64_t                         batch_count);

    MAP2CF_D64(hipblasTrmv, float, hipblasStrmv);
    MAP2CF_D64(hipblasTrmv, double, hipblasDtrmv);
    MAP2CF_D64(hipblasTrmv, std::complex<float>, hipblasCtrmv);
    MAP2CF_D64(hipblasTrmv, std::complex<double>, hipblasZtrmv);

    MAP2CF_D64(hipblasTrmvBatched, float, hipblasStrmvBatched);
    MAP2CF_D64(hipblasTrmvBatched, double, hipblasDtrmvBatched);
    MAP2CF_D64(hipblasTrmvBatched, std::complex<float>, hipblasCtrmvBatched);
    MAP2CF_D64(hipblasTrmvBatched, std::complex<double>, hipblasZtrmvBatched);

    MAP2CF_D64(hipblasTrmvStridedBatched, float, hipblasStrmvStridedBatched);
    MAP2CF_D64(hipblasTrmvStridedBatched, double, hipblasDtrmvStridedBatched);
    MAP2CF_D64(hipblasTrmvStridedBatched, std::complex<float>, hipblasCtrmvStridedBatched);
    MAP2CF_D64(hipblasTrmvStridedBatched, std::complex<double>, hipblasZtrmvStridedBatched);

    // trsv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrsv)(hipblasHandle_t                 handle,
                                   hipblasFillMode_t               uplo,
                                   hipblasOperation_t              transA,
                                   hipblasDiagType_t               diag,
                                   int                             m,
                                   const hipblas_internal_type<T>* A,
                                   int                             lda,
                                   hipblas_internal_type<T>*       x,
                                   int                             incx);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrsvBatched)(hipblasHandle_t                       handle,
                                          hipblasFillMode_t                     uplo,
                                          hipblasOperation_t                    transA,
                                          hipblasDiagType_t                     diag,
                                          int                                   m,
                                          const hipblas_internal_type<T>* const A[],
                                          int                                   lda,
                                          hipblas_internal_type<T>* const       x[],
                                          int                                   incx,
                                          int                                   batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrsvStridedBatched)(hipblasHandle_t                 handle,
                                                 hipblasFillMode_t               uplo,
                                                 hipblasOperation_t              transA,
                                                 hipblasDiagType_t               diag,
                                                 int                             m,
                                                 const hipblas_internal_type<T>* A,
                                                 int                             lda,
                                                 hipblasStride                   strideA,
                                                 hipblas_internal_type<T>*       x,
                                                 int                             incx,
                                                 hipblasStride                   stridex,
                                                 int                             batch_count);

    // trsv_64
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrsv_64)(hipblasHandle_t                 handle,
                                      hipblasFillMode_t               uplo,
                                      hipblasOperation_t              transA,
                                      hipblasDiagType_t               diag,
                                      int64_t                         m,
                                      const hipblas_internal_type<T>* A,
                                      int64_t                         lda,
                                      hipblas_internal_type<T>*       x,
                                      int64_t                         incx);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrsvBatched_64)(hipblasHandle_t                       handle,
                                             hipblasFillMode_t                     uplo,
                                             hipblasOperation_t                    transA,
                                             hipblasDiagType_t                     diag,
                                             int64_t                               m,
                                             const hipblas_internal_type<T>* const A[],
                                             int64_t                               lda,
                                             hipblas_internal_type<T>* const       x[],
                                             int64_t                               incx,
                                             int64_t                               batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrsvStridedBatched_64)(hipblasHandle_t                 handle,
                                                    hipblasFillMode_t               uplo,
                                                    hipblasOperation_t              transA,
                                                    hipblasDiagType_t               diag,
                                                    int64_t                         m,
                                                    const hipblas_internal_type<T>* A,
                                                    int64_t                         lda,
                                                    hipblasStride                   strideA,
                                                    hipblas_internal_type<T>*       x,
                                                    int64_t                         incx,
                                                    hipblasStride                   stridex,
                                                    int64_t                         batch_count);

    MAP2CF_D64(hipblasTrsv, float, hipblasStrsv);
    MAP2CF_D64(hipblasTrsv, double, hipblasDtrsv);
    MAP2CF_D64(hipblasTrsv, std::complex<float>, hipblasCtrsv);
    MAP2CF_D64(hipblasTrsv, std::complex<double>, hipblasZtrsv);

    MAP2CF_D64(hipblasTrsvBatched, float, hipblasStrsvBatched);
    MAP2CF_D64(hipblasTrsvBatched, double, hipblasDtrsvBatched);
    MAP2CF_D64(hipblasTrsvBatched, std::complex<float>, hipblasCtrsvBatched);
    MAP2CF_D64(hipblasTrsvBatched, std::complex<double>, hipblasZtrsvBatched);

    MAP2CF_D64(hipblasTrsvStridedBatched, float, hipblasStrsvStridedBatched);
    MAP2CF_D64(hipblasTrsvStridedBatched, double, hipblasDtrsvStridedBatched);
    MAP2CF_D64(hipblasTrsvStridedBatched, std::complex<float>, hipblasCtrsvStridedBatched);
    MAP2CF_D64(hipblasTrsvStridedBatched, std::complex<double>, hipblasZtrsvStridedBatched);

    // gbmv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGbmv)(hipblasHandle_t                 handle,
                                   hipblasOperation_t              transA,
                                   int                             m,
                                   int                             n,
                                   int                             kl,
                                   int                             ku,
                                   const hipblas_internal_type<T>* alpha,
                                   const hipblas_internal_type<T>* A,
                                   int                             lda,
                                   const hipblas_internal_type<T>* x,
                                   int                             incx,
                                   const hipblas_internal_type<T>* beta,
                                   hipblas_internal_type<T>*       y,
                                   int                             incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGbmvBatched)(hipblasHandle_t                       handle,
                                          hipblasOperation_t                    transA,
                                          int                                   m,
                                          int                                   n,
                                          int                                   kl,
                                          int                                   ku,
                                          const hipblas_internal_type<T>*       alpha,
                                          const hipblas_internal_type<T>* const A[],
                                          int                                   lda,
                                          const hipblas_internal_type<T>* const x[],
                                          int                                   incx,
                                          const hipblas_internal_type<T>*       beta,
                                          hipblas_internal_type<T>* const       y[],
                                          int                                   incy,
                                          int                                   batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGbmvStridedBatched)(hipblasHandle_t                 handle,
                                                 hipblasOperation_t              transA,
                                                 int                             m,
                                                 int                             n,
                                                 int                             kl,
                                                 int                             ku,
                                                 const hipblas_internal_type<T>* alpha,
                                                 const hipblas_internal_type<T>* A,
                                                 int                             lda,
                                                 hipblasStride                   stride_a,
                                                 const hipblas_internal_type<T>* x,
                                                 int                             incx,
                                                 hipblasStride                   stride_x,
                                                 const hipblas_internal_type<T>* beta,
                                                 hipblas_internal_type<T>*       y,
                                                 int                             incy,
                                                 hipblasStride                   stride_y,
                                                 int                             batch_count);

    // gbmv_64
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGbmv_64)(hipblasHandle_t                 handle,
                                      hipblasOperation_t              transA,
                                      int64_t                         m,
                                      int64_t                         n,
                                      int64_t                         kl,
                                      int64_t                         ku,
                                      const hipblas_internal_type<T>* alpha,
                                      const hipblas_internal_type<T>* A,
                                      int64_t                         lda,
                                      const hipblas_internal_type<T>* x,
                                      int64_t                         incx,
                                      const hipblas_internal_type<T>* beta,
                                      hipblas_internal_type<T>*       y,
                                      int64_t                         incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGbmvBatched_64)(hipblasHandle_t                       handle,
                                             hipblasOperation_t                    transA,
                                             int64_t                               m,
                                             int64_t                               n,
                                             int64_t                               kl,
                                             int64_t                               ku,
                                             const hipblas_internal_type<T>*       alpha,
                                             const hipblas_internal_type<T>* const A[],
                                             int64_t                               lda,
                                             const hipblas_internal_type<T>* const x[],
                                             int64_t                               incx,
                                             const hipblas_internal_type<T>*       beta,
                                             hipblas_internal_type<T>* const       y[],
                                             int64_t                               incy,
                                             int64_t                               batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGbmvStridedBatched_64)(hipblasHandle_t                 handle,
                                                    hipblasOperation_t              transA,
                                                    int64_t                         m,
                                                    int64_t                         n,
                                                    int64_t                         kl,
                                                    int64_t                         ku,
                                                    const hipblas_internal_type<T>* alpha,
                                                    const hipblas_internal_type<T>* A,
                                                    int64_t                         lda,
                                                    hipblasStride                   stride_a,
                                                    const hipblas_internal_type<T>* x,
                                                    int64_t                         incx,
                                                    hipblasStride                   stride_x,
                                                    const hipblas_internal_type<T>* beta,
                                                    hipblas_internal_type<T>*       y,
                                                    int64_t                         incy,
                                                    hipblasStride                   stride_y,
                                                    int64_t                         batch_count);

    MAP2CF_D64(hipblasGbmv, float, hipblasSgbmv);
    MAP2CF_D64(hipblasGbmv, double, hipblasDgbmv);
    MAP2CF_D64(hipblasGbmv, std::complex<float>, hipblasCgbmv);
    MAP2CF_D64(hipblasGbmv, std::complex<double>, hipblasZgbmv);

    MAP2CF_D64(hipblasGbmvBatched, float, hipblasSgbmvBatched);
    MAP2CF_D64(hipblasGbmvBatched, double, hipblasDgbmvBatched);
    MAP2CF_D64(hipblasGbmvBatched, std::complex<float>, hipblasCgbmvBatched);
    MAP2CF_D64(hipblasGbmvBatched, std::complex<double>, hipblasZgbmvBatched);

    MAP2CF_D64(hipblasGbmvStridedBatched, float, hipblasSgbmvStridedBatched);
    MAP2CF_D64(hipblasGbmvStridedBatched, double, hipblasDgbmvStridedBatched);
    MAP2CF_D64(hipblasGbmvStridedBatched, std::complex<float>, hipblasCgbmvStridedBatched);
    MAP2CF_D64(hipblasGbmvStridedBatched, std::complex<double>, hipblasZgbmvStridedBatched);

    // gemv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGemv)(hipblasHandle_t                 handle,
                                   hipblasOperation_t              transA,
                                   int                             m,
                                   int                             n,
                                   const hipblas_internal_type<T>* alpha,
                                   const hipblas_internal_type<T>* A,
                                   int                             lda,
                                   const hipblas_internal_type<T>* x,
                                   int                             incx,
                                   const hipblas_internal_type<T>* beta,
                                   hipblas_internal_type<T>*       y,
                                   int                             incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGemvBatched)(hipblasHandle_t                       handle,
                                          hipblasOperation_t                    transA,
                                          int                                   m,
                                          int                                   n,
                                          const hipblas_internal_type<T>*       alpha,
                                          const hipblas_internal_type<T>* const A[],
                                          int                                   lda,
                                          const hipblas_internal_type<T>* const x[],
                                          int                                   incx,
                                          const hipblas_internal_type<T>*       beta,
                                          hipblas_internal_type<T>* const       y[],
                                          int                                   incy,
                                          int                                   batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGemvStridedBatched)(hipblasHandle_t                 handle,
                                                 hipblasOperation_t              transA,
                                                 int                             m,
                                                 int                             n,
                                                 const hipblas_internal_type<T>* alpha,
                                                 const hipblas_internal_type<T>* A,
                                                 int                             lda,
                                                 hipblasStride                   stride_a,
                                                 const hipblas_internal_type<T>* x,
                                                 int                             incx,
                                                 hipblasStride                   stride_x,
                                                 const hipblas_internal_type<T>* beta,
                                                 hipblas_internal_type<T>*       y,
                                                 int                             incy,
                                                 hipblasStride                   stride_y,
                                                 int                             batch_count);

    // gemv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGemv_64)(hipblasHandle_t                 handle,
                                      hipblasOperation_t              transA,
                                      int64_t                         m,
                                      int64_t                         n,
                                      const hipblas_internal_type<T>* alpha,
                                      const hipblas_internal_type<T>* A,
                                      int64_t                         lda,
                                      const hipblas_internal_type<T>* x,
                                      int64_t                         incx,
                                      const hipblas_internal_type<T>* beta,
                                      hipblas_internal_type<T>*       y,
                                      int64_t                         incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGemvBatched_64)(hipblasHandle_t                       handle,
                                             hipblasOperation_t                    transA,
                                             int64_t                               m,
                                             int64_t                               n,
                                             const hipblas_internal_type<T>*       alpha,
                                             const hipblas_internal_type<T>* const A[],
                                             int64_t                               lda,
                                             const hipblas_internal_type<T>* const x[],
                                             int64_t                               incx,
                                             const hipblas_internal_type<T>*       beta,
                                             hipblas_internal_type<T>* const       y[],
                                             int64_t                               incy,
                                             int64_t                               batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGemvStridedBatched_64)(hipblasHandle_t                 handle,
                                                    hipblasOperation_t              transA,
                                                    int64_t                         m,
                                                    int64_t                         n,
                                                    const hipblas_internal_type<T>* alpha,
                                                    const hipblas_internal_type<T>* A,
                                                    int64_t                         lda,
                                                    hipblasStride                   stride_a,
                                                    const hipblas_internal_type<T>* x,
                                                    int64_t                         incx,
                                                    hipblasStride                   stride_x,
                                                    const hipblas_internal_type<T>* beta,
                                                    hipblas_internal_type<T>*       y,
                                                    int64_t                         incy,
                                                    hipblasStride                   stride_y,
                                                    int64_t                         batch_count);

    MAP2CF_D64(hipblasGemv, float, hipblasSgemv);
    MAP2CF_D64(hipblasGemv, double, hipblasDgemv);
    MAP2CF_D64(hipblasGemv, std::complex<float>, hipblasCgemv);
    MAP2CF_D64(hipblasGemv, std::complex<double>, hipblasZgemv);

    MAP2CF_D64(hipblasGemvBatched, float, hipblasSgemvBatched);
    MAP2CF_D64(hipblasGemvBatched, double, hipblasDgemvBatched);
    MAP2CF_D64(hipblasGemvBatched, std::complex<float>, hipblasCgemvBatched);
    MAP2CF_D64(hipblasGemvBatched, std::complex<double>, hipblasZgemvBatched);

    MAP2CF_D64(hipblasGemvStridedBatched, float, hipblasSgemvStridedBatched);
    MAP2CF_D64(hipblasGemvStridedBatched, double, hipblasDgemvStridedBatched);
    MAP2CF_D64(hipblasGemvStridedBatched, std::complex<float>, hipblasCgemvStridedBatched);
    MAP2CF_D64(hipblasGemvStridedBatched, std::complex<double>, hipblasZgemvStridedBatched);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGemm)(hipblasHandle_t                 handle,
                                   hipblasOperation_t              transA,
                                   hipblasOperation_t              transB,
                                   int                             m,
                                   int                             n,
                                   int                             k,
                                   const hipblas_internal_type<T>* alpha,
                                   const hipblas_internal_type<T>* A,
                                   int                             lda,
                                   const hipblas_internal_type<T>* B,
                                   int                             ldb,
                                   const hipblas_internal_type<T>* beta,
                                   hipblas_internal_type<T>*       C,
                                   int                             ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGemmStridedBatched)(hipblasHandle_t                 handle,
                                                 hipblasOperation_t              transA,
                                                 hipblasOperation_t              transB,
                                                 int                             m,
                                                 int                             n,
                                                 int                             k,
                                                 const hipblas_internal_type<T>* alpha,
                                                 const hipblas_internal_type<T>* A,
                                                 int                             lda,
                                                 int                             bsa,
                                                 const hipblas_internal_type<T>* B,
                                                 int                             ldb,
                                                 int                             bsb,
                                                 const hipblas_internal_type<T>* beta,
                                                 hipblas_internal_type<T>*       C,
                                                 int                             ldc,
                                                 int                             bsc,
                                                 int                             batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGemmBatched)(hipblasHandle_t                       handle,
                                          hipblasOperation_t                    transA,
                                          hipblasOperation_t                    transB,
                                          int                                   m,
                                          int                                   n,
                                          int                                   k,
                                          const hipblas_internal_type<T>*       alpha,
                                          const hipblas_internal_type<T>* const A[],
                                          int                                   lda,
                                          const hipblas_internal_type<T>* const B[],
                                          int                                   ldb,
                                          const hipblas_internal_type<T>*       beta,
                                          hipblas_internal_type<T>* const       C[],
                                          int                                   ldc,
                                          int                                   batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGemm_64)(hipblasHandle_t                 handle,
                                      hipblasOperation_t              transA,
                                      hipblasOperation_t              transB,
                                      int64_t                         m,
                                      int64_t                         n,
                                      int64_t                         k,
                                      const hipblas_internal_type<T>* alpha,
                                      const hipblas_internal_type<T>* A,
                                      int64_t                         lda,
                                      const hipblas_internal_type<T>* B,
                                      int64_t                         ldb,
                                      const hipblas_internal_type<T>* beta,
                                      hipblas_internal_type<T>*       C,
                                      int64_t                         ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGemmStridedBatched_64)(hipblasHandle_t                 handle,
                                                    hipblasOperation_t              transA,
                                                    hipblasOperation_t              transB,
                                                    int64_t                         m,
                                                    int64_t                         n,
                                                    int64_t                         k,
                                                    const hipblas_internal_type<T>* alpha,
                                                    const hipblas_internal_type<T>* A,
                                                    int64_t                         lda,
                                                    int64_t                         bsa,
                                                    const hipblas_internal_type<T>* B,
                                                    int64_t                         ldb,
                                                    int64_t                         bsb,
                                                    const hipblas_internal_type<T>* beta,
                                                    hipblas_internal_type<T>*       C,
                                                    int64_t                         ldc,
                                                    int64_t                         bsc,
                                                    int64_t                         batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGemmBatched_64)(hipblasHandle_t                       handle,
                                             hipblasOperation_t                    transA,
                                             hipblasOperation_t                    transB,
                                             int64_t                               m,
                                             int64_t                               n,
                                             int64_t                               k,
                                             const hipblas_internal_type<T>*       alpha,
                                             const hipblas_internal_type<T>* const A[],
                                             int64_t                               lda,
                                             const hipblas_internal_type<T>* const B[],
                                             int64_t                               ldb,
                                             const hipblas_internal_type<T>*       beta,
                                             hipblas_internal_type<T>* const       C[],
                                             int64_t                               ldc,
                                             int64_t                               batch_count);

    MAP2CF_D64(hipblasGemm, hipblasHalf, hipblasHgemm);
    MAP2CF_D64(hipblasGemm, float, hipblasSgemm);
    MAP2CF_D64(hipblasGemm, double, hipblasDgemm);
    MAP2CF_D64(hipblasGemm, std::complex<float>, hipblasCgemm);
    MAP2CF_D64(hipblasGemm, std::complex<double>, hipblasZgemm);

    MAP2CF_D64(hipblasGemmBatched, hipblasHalf, hipblasHgemmBatched);
    MAP2CF_D64(hipblasGemmBatched, float, hipblasSgemmBatched);
    MAP2CF_D64(hipblasGemmBatched, double, hipblasDgemmBatched);
    MAP2CF_D64(hipblasGemmBatched, std::complex<float>, hipblasCgemmBatched);
    MAP2CF_D64(hipblasGemmBatched, std::complex<double>, hipblasZgemmBatched);

    MAP2CF_D64(hipblasGemmStridedBatched, hipblasHalf, hipblasHgemmStridedBatched);
    MAP2CF_D64(hipblasGemmStridedBatched, float, hipblasSgemmStridedBatched);
    MAP2CF_D64(hipblasGemmStridedBatched, double, hipblasDgemmStridedBatched);
    MAP2CF_D64(hipblasGemmStridedBatched, std::complex<float>, hipblasCgemmStridedBatched);
    MAP2CF_D64(hipblasGemmStridedBatched, std::complex<double>, hipblasZgemmStridedBatched);

    // herk
    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerk)(hipblasHandle_t                 handle,
                                   hipblasFillMode_t               uplo,
                                   hipblasOperation_t              transA,
                                   int                             n,
                                   int                             k,
                                   const hipblas_internal_type<U>* alpha,
                                   const hipblas_internal_type<T>* A,
                                   int                             lda,
                                   const hipblas_internal_type<U>* beta,
                                   hipblas_internal_type<T>*       C,
                                   int                             ldc);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerkBatched)(hipblasHandle_t                       handle,
                                          hipblasFillMode_t                     uplo,
                                          hipblasOperation_t                    transA,
                                          int                                   n,
                                          int                                   k,
                                          const hipblas_internal_type<U>*       alpha,
                                          const hipblas_internal_type<T>* const A[],
                                          int                                   lda,
                                          const hipblas_internal_type<U>*       beta,
                                          hipblas_internal_type<T>* const       C[],
                                          int                                   ldc,
                                          int                                   batchCount);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerkStridedBatched)(hipblasHandle_t                 handle,
                                                 hipblasFillMode_t               uplo,
                                                 hipblasOperation_t              transA,
                                                 int                             n,
                                                 int                             k,
                                                 const hipblas_internal_type<U>* alpha,
                                                 const hipblas_internal_type<T>* A,
                                                 int                             lda,
                                                 hipblasStride                   strideA,
                                                 const hipblas_internal_type<U>* beta,
                                                 hipblas_internal_type<T>*       C,
                                                 int                             ldc,
                                                 hipblasStride                   strideC,
                                                 int                             batchCount);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerk_64)(hipblasHandle_t                 handle,
                                      hipblasFillMode_t               uplo,
                                      hipblasOperation_t              transA,
                                      int64_t                         n,
                                      int64_t                         k,
                                      const hipblas_internal_type<U>* alpha,
                                      const hipblas_internal_type<T>* A,
                                      int64_t                         lda,
                                      const hipblas_internal_type<U>* beta,
                                      hipblas_internal_type<T>*       C,
                                      int64_t                         ldc);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerkBatched_64)(hipblasHandle_t                       handle,
                                             hipblasFillMode_t                     uplo,
                                             hipblasOperation_t                    transA,
                                             int64_t                               n,
                                             int64_t                               k,
                                             const hipblas_internal_type<U>*       alpha,
                                             const hipblas_internal_type<T>* const A[],
                                             int64_t                               lda,
                                             const hipblas_internal_type<U>*       beta,
                                             hipblas_internal_type<T>* const       C[],
                                             int64_t                               ldc,
                                             int64_t                               batchCount);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerkStridedBatched_64)(hipblasHandle_t                 handle,
                                                    hipblasFillMode_t               uplo,
                                                    hipblasOperation_t              transA,
                                                    int64_t                         n,
                                                    int64_t                         k,
                                                    const hipblas_internal_type<U>* alpha,
                                                    const hipblas_internal_type<T>* A,
                                                    int64_t                         lda,
                                                    hipblasStride                   strideA,
                                                    const hipblas_internal_type<U>* beta,
                                                    hipblas_internal_type<T>*       C,
                                                    int64_t                         ldc,
                                                    hipblasStride                   strideC,
                                                    int64_t                         batchCount);

    MAP2CF_D64(hipblasHerk, std::complex<float>, float, hipblasCherk);
    MAP2CF_D64(hipblasHerk, std::complex<double>, double, hipblasZherk);

    MAP2CF_D64(hipblasHerkBatched, std::complex<float>, float, hipblasCherkBatched);
    MAP2CF_D64(hipblasHerkBatched, std::complex<double>, double, hipblasZherkBatched);

    MAP2CF_D64(hipblasHerkStridedBatched, std::complex<float>, float, hipblasCherkStridedBatched);
    MAP2CF_D64(hipblasHerkStridedBatched, std::complex<double>, double, hipblasZherkStridedBatched);

    // her2k
    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHer2k)(hipblasHandle_t                 handle,
                                    hipblasFillMode_t               uplo,
                                    hipblasOperation_t              transA,
                                    int                             n,
                                    int                             k,
                                    const hipblas_internal_type<T>* alpha,
                                    const hipblas_internal_type<T>* A,
                                    int                             lda,
                                    const hipblas_internal_type<T>* B,
                                    int                             ldb,
                                    const hipblas_internal_type<U>* beta,
                                    hipblas_internal_type<T>*       C,
                                    int                             ldc);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHer2kBatched)(hipblasHandle_t                       handle,
                                           hipblasFillMode_t                     uplo,
                                           hipblasOperation_t                    transA,
                                           int                                   n,
                                           int                                   k,
                                           const hipblas_internal_type<T>*       alpha,
                                           const hipblas_internal_type<T>* const A[],
                                           int                                   lda,
                                           const hipblas_internal_type<T>* const B[],
                                           int                                   ldb,
                                           const hipblas_internal_type<U>*       beta,
                                           hipblas_internal_type<T>* const       C[],
                                           int                                   ldc,
                                           int                                   batchCount);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHer2kStridedBatched)(hipblasHandle_t                 handle,
                                                  hipblasFillMode_t               uplo,
                                                  hipblasOperation_t              transA,
                                                  int                             n,
                                                  int                             k,
                                                  const hipblas_internal_type<T>* alpha,
                                                  const hipblas_internal_type<T>* A,
                                                  int                             lda,
                                                  hipblasStride                   strideA,
                                                  const hipblas_internal_type<T>* B,
                                                  int                             ldb,
                                                  hipblasStride                   strideB,
                                                  const hipblas_internal_type<U>* beta,
                                                  hipblas_internal_type<T>*       C,
                                                  int                             ldc,
                                                  hipblasStride                   strideC,
                                                  int                             batchCount);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHer2k_64)(hipblasHandle_t                 handle,
                                       hipblasFillMode_t               uplo,
                                       hipblasOperation_t              transA,
                                       int64_t                         n,
                                       int64_t                         k,
                                       const hipblas_internal_type<T>* alpha,
                                       const hipblas_internal_type<T>* A,
                                       int64_t                         lda,
                                       const hipblas_internal_type<T>* B,
                                       int64_t                         ldb,
                                       const hipblas_internal_type<U>* beta,
                                       hipblas_internal_type<T>*       C,
                                       int64_t                         ldc);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHer2kBatched_64)(hipblasHandle_t                       handle,
                                              hipblasFillMode_t                     uplo,
                                              hipblasOperation_t                    transA,
                                              int64_t                               n,
                                              int64_t                               k,
                                              const hipblas_internal_type<T>*       alpha,
                                              const hipblas_internal_type<T>* const A[],
                                              int64_t                               lda,
                                              const hipblas_internal_type<T>* const B[],
                                              int64_t                               ldb,
                                              const hipblas_internal_type<U>*       beta,
                                              hipblas_internal_type<T>* const       C[],
                                              int64_t                               ldc,
                                              int64_t                               batchCount);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHer2kStridedBatched_64)(hipblasHandle_t                 handle,
                                                     hipblasFillMode_t               uplo,
                                                     hipblasOperation_t              transA,
                                                     int64_t                         n,
                                                     int64_t                         k,
                                                     const hipblas_internal_type<T>* alpha,
                                                     const hipblas_internal_type<T>* A,
                                                     int64_t                         lda,
                                                     hipblasStride                   strideA,
                                                     const hipblas_internal_type<T>* B,
                                                     int64_t                         ldb,
                                                     hipblasStride                   strideB,
                                                     const hipblas_internal_type<U>* beta,
                                                     hipblas_internal_type<T>*       C,
                                                     int64_t                         ldc,
                                                     hipblasStride                   strideC,
                                                     int64_t                         batchCount);

    MAP2CF_D64(hipblasHer2k, std::complex<float>, float, hipblasCher2k);
    MAP2CF_D64(hipblasHer2k, std::complex<double>, double, hipblasZher2k);

    MAP2CF_D64(hipblasHer2kBatched, std::complex<float>, float, hipblasCher2kBatched);
    MAP2CF_D64(hipblasHer2kBatched, std::complex<double>, double, hipblasZher2kBatched);

    MAP2CF_D64(hipblasHer2kStridedBatched, std::complex<float>, float, hipblasCher2kStridedBatched);
    MAP2CF_D64(hipblasHer2kStridedBatched,
               std::complex<double>,
               double,
               hipblasZher2kStridedBatched);

    // herkx
    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerkx)(hipblasHandle_t                 handle,
                                    hipblasFillMode_t               uplo,
                                    hipblasOperation_t              transA,
                                    int                             n,
                                    int                             k,
                                    const hipblas_internal_type<T>* alpha,
                                    const hipblas_internal_type<T>* A,
                                    int                             lda,
                                    const hipblas_internal_type<T>* B,
                                    int                             ldb,
                                    const hipblas_internal_type<U>* beta,
                                    hipblas_internal_type<T>*       C,
                                    int                             ldc);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerkxBatched)(hipblasHandle_t                       handle,
                                           hipblasFillMode_t                     uplo,
                                           hipblasOperation_t                    transA,
                                           int                                   n,
                                           int                                   k,
                                           const hipblas_internal_type<T>*       alpha,
                                           const hipblas_internal_type<T>* const A[],
                                           int                                   lda,
                                           const hipblas_internal_type<T>* const B[],
                                           int                                   ldb,
                                           const hipblas_internal_type<U>*       beta,
                                           hipblas_internal_type<T>* const       C[],
                                           int                                   ldc,
                                           int                                   batchCount);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerkxStridedBatched)(hipblasHandle_t                 handle,
                                                  hipblasFillMode_t               uplo,
                                                  hipblasOperation_t              transA,
                                                  int                             n,
                                                  int                             k,
                                                  const hipblas_internal_type<T>* alpha,
                                                  const hipblas_internal_type<T>* A,
                                                  int                             lda,
                                                  hipblasStride                   strideA,
                                                  const hipblas_internal_type<T>* B,
                                                  int                             ldb,
                                                  hipblasStride                   strideB,
                                                  const hipblas_internal_type<U>* beta,
                                                  hipblas_internal_type<T>*       C,
                                                  int                             ldc,
                                                  hipblasStride                   strideC,
                                                  int                             batchCount);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerkx_64)(hipblasHandle_t                 handle,
                                       hipblasFillMode_t               uplo,
                                       hipblasOperation_t              transA,
                                       int64_t                         n,
                                       int64_t                         k,
                                       const hipblas_internal_type<T>* alpha,
                                       const hipblas_internal_type<T>* A,
                                       int64_t                         lda,
                                       const hipblas_internal_type<T>* B,
                                       int64_t                         ldb,
                                       const hipblas_internal_type<U>* beta,
                                       hipblas_internal_type<T>*       C,
                                       int64_t                         ldc);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerkxBatched_64)(hipblasHandle_t                       handle,
                                              hipblasFillMode_t                     uplo,
                                              hipblasOperation_t                    transA,
                                              int64_t                               n,
                                              int64_t                               k,
                                              const hipblas_internal_type<T>*       alpha,
                                              const hipblas_internal_type<T>* const A[],
                                              int64_t                               lda,
                                              const hipblas_internal_type<T>* const B[],
                                              int64_t                               ldb,
                                              const hipblas_internal_type<U>*       beta,
                                              hipblas_internal_type<T>* const       C[],
                                              int64_t                               ldc,
                                              int64_t                               batchCount);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerkxStridedBatched_64)(hipblasHandle_t                 handle,
                                                     hipblasFillMode_t               uplo,
                                                     hipblasOperation_t              transA,
                                                     int64_t                         n,
                                                     int64_t                         k,
                                                     const hipblas_internal_type<T>* alpha,
                                                     const hipblas_internal_type<T>* A,
                                                     int64_t                         lda,
                                                     hipblasStride                   strideA,
                                                     const hipblas_internal_type<T>* B,
                                                     int64_t                         ldb,
                                                     hipblasStride                   strideB,
                                                     const hipblas_internal_type<U>* beta,
                                                     hipblas_internal_type<T>*       C,
                                                     int64_t                         ldc,
                                                     hipblasStride                   strideC,
                                                     int64_t                         batchCount);

    MAP2CF_D64(hipblasHerkx, std::complex<float>, float, hipblasCherkx);
    MAP2CF_D64(hipblasHerkx, std::complex<double>, double, hipblasZherkx);

    MAP2CF_D64(hipblasHerkxBatched, std::complex<float>, float, hipblasCherkxBatched);
    MAP2CF_D64(hipblasHerkxBatched, std::complex<double>, double, hipblasZherkxBatched);

    MAP2CF_D64(hipblasHerkxStridedBatched, std::complex<float>, float, hipblasCherkxStridedBatched);
    MAP2CF_D64(hipblasHerkxStridedBatched,
               std::complex<double>,
               double,
               hipblasZherkxStridedBatched);

    // symm
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSymm)(hipblasHandle_t                 handle,
                                   hipblasSideMode_t               side,
                                   hipblasFillMode_t               uplo,
                                   int                             m,
                                   int                             n,
                                   const hipblas_internal_type<T>* alpha,
                                   const hipblas_internal_type<T>* A,
                                   int                             lda,
                                   const hipblas_internal_type<T>* B,
                                   int                             ldb,
                                   const hipblas_internal_type<T>* beta,
                                   hipblas_internal_type<T>*       C,
                                   int                             ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSymmBatched)(hipblasHandle_t                       handle,
                                          hipblasSideMode_t                     side,
                                          hipblasFillMode_t                     uplo,
                                          int                                   m,
                                          int                                   n,
                                          const hipblas_internal_type<T>*       alpha,
                                          const hipblas_internal_type<T>* const A[],
                                          int                                   lda,
                                          const hipblas_internal_type<T>* const B[],
                                          int                                   ldb,
                                          const hipblas_internal_type<T>*       beta,
                                          hipblas_internal_type<T>* const       C[],
                                          int                                   ldc,
                                          int                                   batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSymmStridedBatched)(hipblasHandle_t                 handle,
                                                 hipblasSideMode_t               side,
                                                 hipblasFillMode_t               uplo,
                                                 int                             m,
                                                 int                             n,
                                                 const hipblas_internal_type<T>* alpha,
                                                 const hipblas_internal_type<T>* A,
                                                 int                             lda,
                                                 hipblasStride                   strideA,
                                                 const hipblas_internal_type<T>* B,
                                                 int                             ldb,
                                                 hipblasStride                   strideB,
                                                 const hipblas_internal_type<T>* beta,
                                                 hipblas_internal_type<T>*       C,
                                                 int                             ldc,
                                                 hipblasStride                   strideC,
                                                 int                             batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSymm_64)(hipblasHandle_t                 handle,
                                      hipblasSideMode_t               side,
                                      hipblasFillMode_t               uplo,
                                      int64_t                         m,
                                      int64_t                         n,
                                      const hipblas_internal_type<T>* alpha,
                                      const hipblas_internal_type<T>* A,
                                      int64_t                         lda,
                                      const hipblas_internal_type<T>* B,
                                      int64_t                         ldb,
                                      const hipblas_internal_type<T>* beta,
                                      hipblas_internal_type<T>*       C,
                                      int64_t                         ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSymmBatched_64)(hipblasHandle_t                       handle,
                                             hipblasSideMode_t                     side,
                                             hipblasFillMode_t                     uplo,
                                             int64_t                               m,
                                             int64_t                               n,
                                             const hipblas_internal_type<T>*       alpha,
                                             const hipblas_internal_type<T>* const A[],
                                             int64_t                               lda,
                                             const hipblas_internal_type<T>* const B[],
                                             int64_t                               ldb,
                                             const hipblas_internal_type<T>*       beta,
                                             hipblas_internal_type<T>* const       C[],
                                             int64_t                               ldc,
                                             int64_t                               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSymmStridedBatched_64)(hipblasHandle_t                 handle,
                                                    hipblasSideMode_t               side,
                                                    hipblasFillMode_t               uplo,
                                                    int64_t                         m,
                                                    int64_t                         n,
                                                    const hipblas_internal_type<T>* alpha,
                                                    const hipblas_internal_type<T>* A,
                                                    int64_t                         lda,
                                                    hipblasStride                   strideA,
                                                    const hipblas_internal_type<T>* B,
                                                    int64_t                         ldb,
                                                    hipblasStride                   strideB,
                                                    const hipblas_internal_type<T>* beta,
                                                    hipblas_internal_type<T>*       C,
                                                    int64_t                         ldc,
                                                    hipblasStride                   strideC,
                                                    int64_t                         batchCount);

    MAP2CF_D64(hipblasSymm, float, hipblasSsymm);
    MAP2CF_D64(hipblasSymm, double, hipblasDsymm);
    MAP2CF_D64(hipblasSymm, std::complex<float>, hipblasCsymm);
    MAP2CF_D64(hipblasSymm, std::complex<double>, hipblasZsymm);

    MAP2CF_D64(hipblasSymmBatched, float, hipblasSsymmBatched);
    MAP2CF_D64(hipblasSymmBatched, double, hipblasDsymmBatched);
    MAP2CF_D64(hipblasSymmBatched, std::complex<float>, hipblasCsymmBatched);
    MAP2CF_D64(hipblasSymmBatched, std::complex<double>, hipblasZsymmBatched);

    MAP2CF_D64(hipblasSymmStridedBatched, float, hipblasSsymmStridedBatched);
    MAP2CF_D64(hipblasSymmStridedBatched, double, hipblasDsymmStridedBatched);
    MAP2CF_D64(hipblasSymmStridedBatched, std::complex<float>, hipblasCsymmStridedBatched);
    MAP2CF_D64(hipblasSymmStridedBatched, std::complex<double>, hipblasZsymmStridedBatched);

    // syrk
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrk)(hipblasHandle_t                 handle,
                                   hipblasFillMode_t               uplo,
                                   hipblasOperation_t              transA,
                                   int                             n,
                                   int                             k,
                                   const hipblas_internal_type<T>* alpha,
                                   const hipblas_internal_type<T>* A,
                                   int                             lda,
                                   const hipblas_internal_type<T>* beta,
                                   hipblas_internal_type<T>*       C,
                                   int                             ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrkBatched)(hipblasHandle_t                       handle,
                                          hipblasFillMode_t                     uplo,
                                          hipblasOperation_t                    transA,
                                          int                                   n,
                                          int                                   k,
                                          const hipblas_internal_type<T>*       alpha,
                                          const hipblas_internal_type<T>* const A[],
                                          int                                   lda,
                                          const hipblas_internal_type<T>*       beta,
                                          hipblas_internal_type<T>* const       C[],
                                          int                                   ldc,
                                          int                                   batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrkStridedBatched)(hipblasHandle_t                 handle,
                                                 hipblasFillMode_t               uplo,
                                                 hipblasOperation_t              transA,
                                                 int                             n,
                                                 int                             k,
                                                 const hipblas_internal_type<T>* alpha,
                                                 const hipblas_internal_type<T>* A,
                                                 int                             lda,
                                                 hipblasStride                   strideA,
                                                 const hipblas_internal_type<T>* beta,
                                                 hipblas_internal_type<T>*       C,
                                                 int                             ldc,
                                                 hipblasStride                   strideC,
                                                 int                             batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrk_64)(hipblasHandle_t                 handle,
                                      hipblasFillMode_t               uplo,
                                      hipblasOperation_t              transA,
                                      int64_t                         n,
                                      int64_t                         k,
                                      const hipblas_internal_type<T>* alpha,
                                      const hipblas_internal_type<T>* A,
                                      int64_t                         lda,
                                      const hipblas_internal_type<T>* beta,
                                      hipblas_internal_type<T>*       C,
                                      int64_t                         ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrkBatched_64)(hipblasHandle_t                       handle,
                                             hipblasFillMode_t                     uplo,
                                             hipblasOperation_t                    transA,
                                             int64_t                               n,
                                             int64_t                               k,
                                             const hipblas_internal_type<T>*       alpha,
                                             const hipblas_internal_type<T>* const A[],
                                             int64_t                               lda,
                                             const hipblas_internal_type<T>*       beta,
                                             hipblas_internal_type<T>* const       C[],
                                             int64_t                               ldc,
                                             int64_t                               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrkStridedBatched_64)(hipblasHandle_t                 handle,
                                                    hipblasFillMode_t               uplo,
                                                    hipblasOperation_t              transA,
                                                    int64_t                         n,
                                                    int64_t                         k,
                                                    const hipblas_internal_type<T>* alpha,
                                                    const hipblas_internal_type<T>* A,
                                                    int64_t                         lda,
                                                    hipblasStride                   strideA,
                                                    const hipblas_internal_type<T>* beta,
                                                    hipblas_internal_type<T>*       C,
                                                    int64_t                         ldc,
                                                    hipblasStride                   strideC,
                                                    int64_t                         batchCount);

    MAP2CF_D64(hipblasSyrk, float, hipblasSsyrk);
    MAP2CF_D64(hipblasSyrk, double, hipblasDsyrk);
    MAP2CF_D64(hipblasSyrk, std::complex<float>, hipblasCsyrk);
    MAP2CF_D64(hipblasSyrk, std::complex<double>, hipblasZsyrk);

    MAP2CF_D64(hipblasSyrkBatched, float, hipblasSsyrkBatched);
    MAP2CF_D64(hipblasSyrkBatched, double, hipblasDsyrkBatched);
    MAP2CF_D64(hipblasSyrkBatched, std::complex<float>, hipblasCsyrkBatched);
    MAP2CF_D64(hipblasSyrkBatched, std::complex<double>, hipblasZsyrkBatched);

    MAP2CF_D64(hipblasSyrkStridedBatched, float, hipblasSsyrkStridedBatched);
    MAP2CF_D64(hipblasSyrkStridedBatched, double, hipblasDsyrkStridedBatched);
    MAP2CF_D64(hipblasSyrkStridedBatched, std::complex<float>, hipblasCsyrkStridedBatched);
    MAP2CF_D64(hipblasSyrkStridedBatched, std::complex<double>, hipblasZsyrkStridedBatched);

    // syr2k
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyr2k)(hipblasHandle_t                 handle,
                                    hipblasFillMode_t               uplo,
                                    hipblasOperation_t              transA,
                                    int                             n,
                                    int                             k,
                                    const hipblas_internal_type<T>* alpha,
                                    const hipblas_internal_type<T>* A,
                                    int                             lda,
                                    const hipblas_internal_type<T>* B,
                                    int                             ldb,
                                    const hipblas_internal_type<T>* beta,
                                    hipblas_internal_type<T>*       C,
                                    int                             ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyr2kBatched)(hipblasHandle_t                       handle,
                                           hipblasFillMode_t                     uplo,
                                           hipblasOperation_t                    transA,
                                           int                                   n,
                                           int                                   k,
                                           const hipblas_internal_type<T>*       alpha,
                                           const hipblas_internal_type<T>* const A[],
                                           int                                   lda,
                                           const hipblas_internal_type<T>* const B[],
                                           int                                   ldb,
                                           const hipblas_internal_type<T>*       beta,
                                           hipblas_internal_type<T>* const       C[],
                                           int                                   ldc,
                                           int                                   batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyr2kStridedBatched)(hipblasHandle_t                 handle,
                                                  hipblasFillMode_t               uplo,
                                                  hipblasOperation_t              transA,
                                                  int                             n,
                                                  int                             k,
                                                  const hipblas_internal_type<T>* alpha,
                                                  const hipblas_internal_type<T>* A,
                                                  int                             lda,
                                                  hipblasStride                   strideA,
                                                  const hipblas_internal_type<T>* B,
                                                  int                             ldb,
                                                  hipblasStride                   strideB,
                                                  const hipblas_internal_type<T>* beta,
                                                  hipblas_internal_type<T>*       C,
                                                  int                             ldc,
                                                  hipblasStride                   strideC,
                                                  int                             batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyr2k_64)(hipblasHandle_t                 handle,
                                       hipblasFillMode_t               uplo,
                                       hipblasOperation_t              transA,
                                       int64_t                         n,
                                       int64_t                         k,
                                       const hipblas_internal_type<T>* alpha,
                                       const hipblas_internal_type<T>* A,
                                       int64_t                         lda,
                                       const hipblas_internal_type<T>* B,
                                       int64_t                         ldb,
                                       const hipblas_internal_type<T>* beta,
                                       hipblas_internal_type<T>*       C,
                                       int64_t                         ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyr2kBatched_64)(hipblasHandle_t                       handle,
                                              hipblasFillMode_t                     uplo,
                                              hipblasOperation_t                    transA,
                                              int64_t                               n,
                                              int64_t                               k,
                                              const hipblas_internal_type<T>*       alpha,
                                              const hipblas_internal_type<T>* const A[],
                                              int64_t                               lda,
                                              const hipblas_internal_type<T>* const B[],
                                              int64_t                               ldb,
                                              const hipblas_internal_type<T>*       beta,
                                              hipblas_internal_type<T>* const       C[],
                                              int64_t                               ldc,
                                              int64_t                               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyr2kStridedBatched_64)(hipblasHandle_t                 handle,
                                                     hipblasFillMode_t               uplo,
                                                     hipblasOperation_t              transA,
                                                     int64_t                         n,
                                                     int64_t                         k,
                                                     const hipblas_internal_type<T>* alpha,
                                                     const hipblas_internal_type<T>* A,
                                                     int64_t                         lda,
                                                     hipblasStride                   strideA,
                                                     const hipblas_internal_type<T>* B,
                                                     int64_t                         ldb,
                                                     hipblasStride                   strideB,
                                                     const hipblas_internal_type<T>* beta,
                                                     hipblas_internal_type<T>*       C,
                                                     int64_t                         ldc,
                                                     hipblasStride                   strideC,
                                                     int64_t                         batchCount);

    MAP2CF_D64(hipblasSyr2k, float, hipblasSsyr2k);
    MAP2CF_D64(hipblasSyr2k, double, hipblasDsyr2k);
    MAP2CF_D64(hipblasSyr2k, std::complex<float>, hipblasCsyr2k);
    MAP2CF_D64(hipblasSyr2k, std::complex<double>, hipblasZsyr2k);

    MAP2CF_D64(hipblasSyr2kBatched, float, hipblasSsyr2kBatched);
    MAP2CF_D64(hipblasSyr2kBatched, double, hipblasDsyr2kBatched);
    MAP2CF_D64(hipblasSyr2kBatched, std::complex<float>, hipblasCsyr2kBatched);
    MAP2CF_D64(hipblasSyr2kBatched, std::complex<double>, hipblasZsyr2kBatched);

    MAP2CF_D64(hipblasSyr2kStridedBatched, float, hipblasSsyr2kStridedBatched);
    MAP2CF_D64(hipblasSyr2kStridedBatched, double, hipblasDsyr2kStridedBatched);
    MAP2CF_D64(hipblasSyr2kStridedBatched, std::complex<float>, hipblasCsyr2kStridedBatched);
    MAP2CF_D64(hipblasSyr2kStridedBatched, std::complex<double>, hipblasZsyr2kStridedBatched);

    // syrkx
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrkx)(hipblasHandle_t                 handle,
                                    hipblasFillMode_t               uplo,
                                    hipblasOperation_t              transA,
                                    int                             n,
                                    int                             k,
                                    const hipblas_internal_type<T>* alpha,
                                    const hipblas_internal_type<T>* A,
                                    int                             lda,
                                    const hipblas_internal_type<T>* B,
                                    int                             ldb,
                                    const hipblas_internal_type<T>* beta,
                                    hipblas_internal_type<T>*       C,
                                    int                             ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrkxBatched)(hipblasHandle_t                       handle,
                                           hipblasFillMode_t                     uplo,
                                           hipblasOperation_t                    transA,
                                           int                                   n,
                                           int                                   k,
                                           const hipblas_internal_type<T>*       alpha,
                                           const hipblas_internal_type<T>* const A[],
                                           int                                   lda,
                                           const hipblas_internal_type<T>* const B[],
                                           int                                   ldb,
                                           const hipblas_internal_type<T>*       beta,
                                           hipblas_internal_type<T>* const       C[],
                                           int                                   ldc,
                                           int                                   batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrkxStridedBatched)(hipblasHandle_t                 handle,
                                                  hipblasFillMode_t               uplo,
                                                  hipblasOperation_t              transA,
                                                  int                             n,
                                                  int                             k,
                                                  const hipblas_internal_type<T>* alpha,
                                                  const hipblas_internal_type<T>* A,
                                                  int                             lda,
                                                  hipblasStride                   strideA,
                                                  const hipblas_internal_type<T>* B,
                                                  int                             ldb,
                                                  hipblasStride                   strideB,
                                                  const hipblas_internal_type<T>* beta,
                                                  hipblas_internal_type<T>*       C,
                                                  int                             ldc,
                                                  hipblasStride                   strideC,
                                                  int                             batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrkx_64)(hipblasHandle_t                 handle,
                                       hipblasFillMode_t               uplo,
                                       hipblasOperation_t              transA,
                                       int64_t                         n,
                                       int64_t                         k,
                                       const hipblas_internal_type<T>* alpha,
                                       const hipblas_internal_type<T>* A,
                                       int64_t                         lda,
                                       const hipblas_internal_type<T>* B,
                                       int64_t                         ldb,
                                       const hipblas_internal_type<T>* beta,
                                       hipblas_internal_type<T>*       C,
                                       int64_t                         ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrkxBatched_64)(hipblasHandle_t                       handle,
                                              hipblasFillMode_t                     uplo,
                                              hipblasOperation_t                    transA,
                                              int64_t                               n,
                                              int64_t                               k,
                                              const hipblas_internal_type<T>*       alpha,
                                              const hipblas_internal_type<T>* const A[],
                                              int64_t                               lda,
                                              const hipblas_internal_type<T>* const B[],
                                              int64_t                               ldb,
                                              const hipblas_internal_type<T>*       beta,
                                              hipblas_internal_type<T>* const       C[],
                                              int64_t                               ldc,
                                              int64_t                               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrkxStridedBatched_64)(hipblasHandle_t                 handle,
                                                     hipblasFillMode_t               uplo,
                                                     hipblasOperation_t              transA,
                                                     int64_t                         n,
                                                     int64_t                         k,
                                                     const hipblas_internal_type<T>* alpha,
                                                     const hipblas_internal_type<T>* A,
                                                     int64_t                         lda,
                                                     hipblasStride                   strideA,
                                                     const hipblas_internal_type<T>* B,
                                                     int64_t                         ldb,
                                                     hipblasStride                   strideB,
                                                     const hipblas_internal_type<T>* beta,
                                                     hipblas_internal_type<T>*       C,
                                                     int64_t                         ldc,
                                                     hipblasStride                   strideC,
                                                     int64_t                         batchCount);

    MAP2CF_D64(hipblasSyrkx, float, hipblasSsyrkx);
    MAP2CF_D64(hipblasSyrkx, double, hipblasDsyrkx);
    MAP2CF_D64(hipblasSyrkx, std::complex<float>, hipblasCsyrkx);
    MAP2CF_D64(hipblasSyrkx, std::complex<double>, hipblasZsyrkx);

    MAP2CF_D64(hipblasSyrkxBatched, float, hipblasSsyrkxBatched);
    MAP2CF_D64(hipblasSyrkxBatched, double, hipblasDsyrkxBatched);
    MAP2CF_D64(hipblasSyrkxBatched, std::complex<float>, hipblasCsyrkxBatched);
    MAP2CF_D64(hipblasSyrkxBatched, std::complex<double>, hipblasZsyrkxBatched);

    MAP2CF_D64(hipblasSyrkxStridedBatched, float, hipblasSsyrkxStridedBatched);
    MAP2CF_D64(hipblasSyrkxStridedBatched, double, hipblasDsyrkxStridedBatched);
    MAP2CF_D64(hipblasSyrkxStridedBatched, std::complex<float>, hipblasCsyrkxStridedBatched);
    MAP2CF_D64(hipblasSyrkxStridedBatched, std::complex<double>, hipblasZsyrkxStridedBatched);

    // geam
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGeam)(hipblasHandle_t                 handle,
                                   hipblasOperation_t              transA,
                                   hipblasOperation_t              transB,
                                   int                             m,
                                   int                             n,
                                   const hipblas_internal_type<T>* alpha,
                                   const hipblas_internal_type<T>* A,
                                   int                             lda,
                                   const hipblas_internal_type<T>* beta,
                                   const hipblas_internal_type<T>* B,
                                   int                             ldb,
                                   hipblas_internal_type<T>*       C,
                                   int                             ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGeamBatched)(hipblasHandle_t                       handle,
                                          hipblasOperation_t                    transA,
                                          hipblasOperation_t                    transB,
                                          int                                   m,
                                          int                                   n,
                                          const hipblas_internal_type<T>*       alpha,
                                          const hipblas_internal_type<T>* const A[],
                                          int                                   lda,
                                          const hipblas_internal_type<T>*       beta,
                                          const hipblas_internal_type<T>* const B[],
                                          int                                   ldb,
                                          hipblas_internal_type<T>* const       C[],
                                          int                                   ldc,
                                          int                                   batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGeamStridedBatched)(hipblasHandle_t                 handle,
                                                 hipblasOperation_t              transA,
                                                 hipblasOperation_t              transB,
                                                 int                             m,
                                                 int                             n,
                                                 const hipblas_internal_type<T>* alpha,
                                                 const hipblas_internal_type<T>* A,
                                                 int                             lda,
                                                 hipblasStride                   strideA,
                                                 const hipblas_internal_type<T>* beta,
                                                 const hipblas_internal_type<T>* B,
                                                 int                             ldb,
                                                 hipblasStride                   strideB,
                                                 hipblas_internal_type<T>*       C,
                                                 int                             ldc,
                                                 hipblasStride                   strideC,
                                                 int                             batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGeam_64)(hipblasHandle_t                 handle,
                                      hipblasOperation_t              transA,
                                      hipblasOperation_t              transB,
                                      int64_t                         m,
                                      int64_t                         n,
                                      const hipblas_internal_type<T>* alpha,
                                      const hipblas_internal_type<T>* A,
                                      int64_t                         lda,
                                      const hipblas_internal_type<T>* beta,
                                      const hipblas_internal_type<T>* B,
                                      int64_t                         ldb,
                                      hipblas_internal_type<T>*       C,
                                      int64_t                         ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGeamBatched_64)(hipblasHandle_t                       handle,
                                             hipblasOperation_t                    transA,
                                             hipblasOperation_t                    transB,
                                             int64_t                               m,
                                             int64_t                               n,
                                             const hipblas_internal_type<T>*       alpha,
                                             const hipblas_internal_type<T>* const A[],
                                             int64_t                               lda,
                                             const hipblas_internal_type<T>*       beta,
                                             const hipblas_internal_type<T>* const B[],
                                             int64_t                               ldb,
                                             hipblas_internal_type<T>* const       C[],
                                             int64_t                               ldc,
                                             int64_t                               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGeamStridedBatched_64)(hipblasHandle_t                 handle,
                                                    hipblasOperation_t              transA,
                                                    hipblasOperation_t              transB,
                                                    int64_t                         m,
                                                    int64_t                         n,
                                                    const hipblas_internal_type<T>* alpha,
                                                    const hipblas_internal_type<T>* A,
                                                    int64_t                         lda,
                                                    hipblasStride                   strideA,
                                                    const hipblas_internal_type<T>* beta,
                                                    const hipblas_internal_type<T>* B,
                                                    int64_t                         ldb,
                                                    hipblasStride                   strideB,
                                                    hipblas_internal_type<T>*       C,
                                                    int64_t                         ldc,
                                                    hipblasStride                   strideC,
                                                    int64_t                         batchCount);

    MAP2CF_D64(hipblasGeam, float, hipblasSgeam);
    MAP2CF_D64(hipblasGeam, double, hipblasDgeam);
    MAP2CF_D64(hipblasGeam, std::complex<float>, hipblasCgeam);
    MAP2CF_D64(hipblasGeam, std::complex<double>, hipblasZgeam);

    MAP2CF_D64(hipblasGeamBatched, float, hipblasSgeamBatched);
    MAP2CF_D64(hipblasGeamBatched, double, hipblasDgeamBatched);
    MAP2CF_D64(hipblasGeamBatched, std::complex<float>, hipblasCgeamBatched);
    MAP2CF_D64(hipblasGeamBatched, std::complex<double>, hipblasZgeamBatched);

    MAP2CF_D64(hipblasGeamStridedBatched, float, hipblasSgeamStridedBatched);
    MAP2CF_D64(hipblasGeamStridedBatched, double, hipblasDgeamStridedBatched);
    MAP2CF_D64(hipblasGeamStridedBatched, std::complex<float>, hipblasCgeamStridedBatched);
    MAP2CF_D64(hipblasGeamStridedBatched, std::complex<double>, hipblasZgeamStridedBatched);

    // hemm
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHemm)(hipblasHandle_t                 handle,
                                   hipblasSideMode_t               side,
                                   hipblasFillMode_t               uplo,
                                   int                             n,
                                   int                             k,
                                   const hipblas_internal_type<T>* alpha,
                                   const hipblas_internal_type<T>* A,
                                   int                             lda,
                                   const hipblas_internal_type<T>* B,
                                   int                             ldb,
                                   const hipblas_internal_type<T>* beta,
                                   hipblas_internal_type<T>*       C,
                                   int                             ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHemmBatched)(hipblasHandle_t                       handle,
                                          hipblasSideMode_t                     side,
                                          hipblasFillMode_t                     uplo,
                                          int                                   n,
                                          int                                   k,
                                          const hipblas_internal_type<T>*       alpha,
                                          const hipblas_internal_type<T>* const A[],
                                          int                                   lda,
                                          const hipblas_internal_type<T>* const B[],
                                          int                                   ldb,
                                          const hipblas_internal_type<T>*       beta,
                                          hipblas_internal_type<T>* const       C[],
                                          int                                   ldc,
                                          int                                   batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHemmStridedBatched)(hipblasHandle_t                 handle,
                                                 hipblasSideMode_t               side,
                                                 hipblasFillMode_t               uplo,
                                                 int                             n,
                                                 int                             k,
                                                 const hipblas_internal_type<T>* alpha,
                                                 const hipblas_internal_type<T>* A,
                                                 int                             lda,
                                                 hipblasStride                   strideA,
                                                 const hipblas_internal_type<T>* B,
                                                 int                             ldb,
                                                 hipblasStride                   strideB,
                                                 const hipblas_internal_type<T>* beta,
                                                 hipblas_internal_type<T>*       C,
                                                 int                             ldc,
                                                 hipblasStride                   strideC,
                                                 int                             batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHemm_64)(hipblasHandle_t                 handle,
                                      hipblasSideMode_t               side,
                                      hipblasFillMode_t               uplo,
                                      int64_t                         n,
                                      int64_t                         k,
                                      const hipblas_internal_type<T>* alpha,
                                      const hipblas_internal_type<T>* A,
                                      int64_t                         lda,
                                      const hipblas_internal_type<T>* B,
                                      int64_t                         ldb,
                                      const hipblas_internal_type<T>* beta,
                                      hipblas_internal_type<T>*       C,
                                      int64_t                         ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHemmBatched_64)(hipblasHandle_t                       handle,
                                             hipblasSideMode_t                     side,
                                             hipblasFillMode_t                     uplo,
                                             int64_t                               n,
                                             int64_t                               k,
                                             const hipblas_internal_type<T>*       alpha,
                                             const hipblas_internal_type<T>* const A[],
                                             int64_t                               lda,
                                             const hipblas_internal_type<T>* const B[],
                                             int64_t                               ldb,
                                             const hipblas_internal_type<T>*       beta,
                                             hipblas_internal_type<T>* const       C[],
                                             int64_t                               ldc,
                                             int64_t                               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHemmStridedBatched_64)(hipblasHandle_t                 handle,
                                                    hipblasSideMode_t               side,
                                                    hipblasFillMode_t               uplo,
                                                    int64_t                         n,
                                                    int64_t                         k,
                                                    const hipblas_internal_type<T>* alpha,
                                                    const hipblas_internal_type<T>* A,
                                                    int64_t                         lda,
                                                    hipblasStride                   strideA,
                                                    const hipblas_internal_type<T>* B,
                                                    int64_t                         ldb,
                                                    hipblasStride                   strideB,
                                                    const hipblas_internal_type<T>* beta,
                                                    hipblas_internal_type<T>*       C,
                                                    int64_t                         ldc,
                                                    hipblasStride                   strideC,
                                                    int64_t                         batchCount);

    MAP2CF_D64(hipblasHemm, std::complex<float>, hipblasChemm);
    MAP2CF_D64(hipblasHemm, std::complex<double>, hipblasZhemm);

    MAP2CF_D64(hipblasHemmBatched, std::complex<float>, hipblasChemmBatched);
    MAP2CF_D64(hipblasHemmBatched, std::complex<double>, hipblasZhemmBatched);

    MAP2CF_D64(hipblasHemmStridedBatched, std::complex<float>, hipblasChemmStridedBatched);
    MAP2CF_D64(hipblasHemmStridedBatched, std::complex<double>, hipblasZhemmStridedBatched);

    // trmm
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrmm)(hipblasHandle_t                 handle,
                                   hipblasSideMode_t               side,
                                   hipblasFillMode_t               uplo,
                                   hipblasOperation_t              transA,
                                   hipblasDiagType_t               diag,
                                   int                             m,
                                   int                             n,
                                   const hipblas_internal_type<T>* alpha,
                                   const hipblas_internal_type<T>* A,
                                   int                             lda,
                                   const hipblas_internal_type<T>* B,
                                   int                             ldb,
                                   hipblas_internal_type<T>*       C,
                                   int                             ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrmmBatched)(hipblasHandle_t                       handle,
                                          hipblasSideMode_t                     side,
                                          hipblasFillMode_t                     uplo,
                                          hipblasOperation_t                    transA,
                                          hipblasDiagType_t                     diag,
                                          int                                   m,
                                          int                                   n,
                                          const hipblas_internal_type<T>*       alpha,
                                          const hipblas_internal_type<T>* const A[],
                                          int                                   lda,
                                          const hipblas_internal_type<T>* const B[],
                                          int                                   ldb,
                                          hipblas_internal_type<T>* const       C[],
                                          int                                   ldc,
                                          int                                   batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrmmStridedBatched)(hipblasHandle_t                 handle,
                                                 hipblasSideMode_t               side,
                                                 hipblasFillMode_t               uplo,
                                                 hipblasOperation_t              transA,
                                                 hipblasDiagType_t               diag,
                                                 int                             m,
                                                 int                             n,
                                                 const hipblas_internal_type<T>* alpha,
                                                 const hipblas_internal_type<T>* A,
                                                 int                             lda,
                                                 hipblasStride                   strideA,
                                                 const hipblas_internal_type<T>* B,
                                                 int                             ldb,
                                                 hipblasStride                   strideB,
                                                 hipblas_internal_type<T>*       C,
                                                 int                             ldc,
                                                 hipblasStride                   strideC,
                                                 int                             batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrmm_64)(hipblasHandle_t                 handle,
                                      hipblasSideMode_t               side,
                                      hipblasFillMode_t               uplo,
                                      hipblasOperation_t              transA,
                                      hipblasDiagType_t               diag,
                                      int64_t                         m,
                                      int64_t                         n,
                                      const hipblas_internal_type<T>* alpha,
                                      const hipblas_internal_type<T>* A,
                                      int64_t                         lda,
                                      const hipblas_internal_type<T>* B,
                                      int64_t                         ldb,
                                      hipblas_internal_type<T>*       C,
                                      int64_t                         ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrmmBatched_64)(hipblasHandle_t                       handle,
                                             hipblasSideMode_t                     side,
                                             hipblasFillMode_t                     uplo,
                                             hipblasOperation_t                    transA,
                                             hipblasDiagType_t                     diag,
                                             int64_t                               m,
                                             int64_t                               n,
                                             const hipblas_internal_type<T>*       alpha,
                                             const hipblas_internal_type<T>* const A[],
                                             int64_t                               lda,
                                             const hipblas_internal_type<T>* const B[],
                                             int64_t                               ldb,
                                             hipblas_internal_type<T>* const       C[],
                                             int64_t                               ldc,
                                             int64_t                               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrmmStridedBatched_64)(hipblasHandle_t                 handle,
                                                    hipblasSideMode_t               side,
                                                    hipblasFillMode_t               uplo,
                                                    hipblasOperation_t              transA,
                                                    hipblasDiagType_t               diag,
                                                    int64_t                         m,
                                                    int64_t                         n,
                                                    const hipblas_internal_type<T>* alpha,
                                                    const hipblas_internal_type<T>* A,
                                                    int64_t                         lda,
                                                    hipblasStride                   strideA,
                                                    const hipblas_internal_type<T>* B,
                                                    int64_t                         ldb,
                                                    hipblasStride                   strideB,
                                                    hipblas_internal_type<T>*       C,
                                                    int64_t                         ldc,
                                                    hipblasStride                   strideC,
                                                    int64_t                         batchCount);

    MAP2CF_D64(hipblasTrmm, float, hipblasStrmm);
    MAP2CF_D64(hipblasTrmm, double, hipblasDtrmm);
    MAP2CF_D64(hipblasTrmm, std::complex<float>, hipblasCtrmm);
    MAP2CF_D64(hipblasTrmm, std::complex<double>, hipblasZtrmm);

    MAP2CF_D64(hipblasTrmmBatched, float, hipblasStrmmBatched);
    MAP2CF_D64(hipblasTrmmBatched, double, hipblasDtrmmBatched);
    MAP2CF_D64(hipblasTrmmBatched, std::complex<float>, hipblasCtrmmBatched);
    MAP2CF_D64(hipblasTrmmBatched, std::complex<double>, hipblasZtrmmBatched);

    MAP2CF_D64(hipblasTrmmStridedBatched, float, hipblasStrmmStridedBatched);
    MAP2CF_D64(hipblasTrmmStridedBatched, double, hipblasDtrmmStridedBatched);
    MAP2CF_D64(hipblasTrmmStridedBatched, std::complex<float>, hipblasCtrmmStridedBatched);
    MAP2CF_D64(hipblasTrmmStridedBatched, std::complex<double>, hipblasZtrmmStridedBatched);

    // trsm
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrsm)(hipblasHandle_t                 handle,
                                   hipblasSideMode_t               side,
                                   hipblasFillMode_t               uplo,
                                   hipblasOperation_t              transA,
                                   hipblasDiagType_t               diag,
                                   int                             m,
                                   int                             n,
                                   const hipblas_internal_type<T>* alpha,
                                   const hipblas_internal_type<T>* A,
                                   int                             lda,
                                   hipblas_internal_type<T>*       B,
                                   int                             ldb);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrsmBatched)(hipblasHandle_t                       handle,
                                          hipblasSideMode_t                     side,
                                          hipblasFillMode_t                     uplo,
                                          hipblasOperation_t                    transA,
                                          hipblasDiagType_t                     diag,
                                          int                                   m,
                                          int                                   n,
                                          const hipblas_internal_type<T>*       alpha,
                                          const hipblas_internal_type<T>* const A[],
                                          int                                   lda,
                                          hipblas_internal_type<T>* const       B[],
                                          int                                   ldb,
                                          int                                   batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrsmStridedBatched)(hipblasHandle_t                 handle,
                                                 hipblasSideMode_t               side,
                                                 hipblasFillMode_t               uplo,
                                                 hipblasOperation_t              transA,
                                                 hipblasDiagType_t               diag,
                                                 int                             m,
                                                 int                             n,
                                                 const hipblas_internal_type<T>* alpha,
                                                 const hipblas_internal_type<T>* A,
                                                 int                             lda,
                                                 hipblasStride                   strideA,
                                                 hipblas_internal_type<T>*       B,
                                                 int                             ldb,
                                                 hipblasStride                   strideB,
                                                 int                             batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrsm_64)(hipblasHandle_t                 handle,
                                      hipblasSideMode_t               side,
                                      hipblasFillMode_t               uplo,
                                      hipblasOperation_t              transA,
                                      hipblasDiagType_t               diag,
                                      int64_t                         m,
                                      int64_t                         n,
                                      const hipblas_internal_type<T>* alpha,
                                      const hipblas_internal_type<T>* A,
                                      int64_t                         lda,
                                      hipblas_internal_type<T>*       B,
                                      int64_t                         ldb);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrsmBatched_64)(hipblasHandle_t                       handle,
                                             hipblasSideMode_t                     side,
                                             hipblasFillMode_t                     uplo,
                                             hipblasOperation_t                    transA,
                                             hipblasDiagType_t                     diag,
                                             int64_t                               m,
                                             int64_t                               n,
                                             const hipblas_internal_type<T>*       alpha,
                                             const hipblas_internal_type<T>* const A[],
                                             int64_t                               lda,
                                             hipblas_internal_type<T>* const       B[],
                                             int64_t                               ldb,
                                             int64_t                               batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrsmStridedBatched_64)(hipblasHandle_t                 handle,
                                                    hipblasSideMode_t               side,
                                                    hipblasFillMode_t               uplo,
                                                    hipblasOperation_t              transA,
                                                    hipblasDiagType_t               diag,
                                                    int64_t                         m,
                                                    int64_t                         n,
                                                    const hipblas_internal_type<T>* alpha,
                                                    const hipblas_internal_type<T>* A,
                                                    int64_t                         lda,
                                                    hipblasStride                   strideA,
                                                    hipblas_internal_type<T>*       B,
                                                    int64_t                         ldb,
                                                    hipblasStride                   strideB,
                                                    int64_t                         batch_count);

    MAP2CF_D64(hipblasTrsm, float, hipblasStrsm);
    MAP2CF_D64(hipblasTrsm, double, hipblasDtrsm);
    MAP2CF_D64(hipblasTrsm, std::complex<float>, hipblasCtrsm);
    MAP2CF_D64(hipblasTrsm, std::complex<double>, hipblasZtrsm);

    MAP2CF_D64(hipblasTrsmBatched, float, hipblasStrsmBatched);
    MAP2CF_D64(hipblasTrsmBatched, double, hipblasDtrsmBatched);
    MAP2CF_D64(hipblasTrsmBatched, std::complex<float>, hipblasCtrsmBatched);
    MAP2CF_D64(hipblasTrsmBatched, std::complex<double>, hipblasZtrsmBatched);

    MAP2CF_D64(hipblasTrsmStridedBatched, float, hipblasStrsmStridedBatched);
    MAP2CF_D64(hipblasTrsmStridedBatched, double, hipblasDtrsmStridedBatched);
    MAP2CF_D64(hipblasTrsmStridedBatched, std::complex<float>, hipblasCtrsmStridedBatched);
    MAP2CF_D64(hipblasTrsmStridedBatched, std::complex<double>, hipblasZtrsmStridedBatched);

    // dgmm
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDgmm)(hipblasHandle_t                 handle,
                                   hipblasSideMode_t               side,
                                   int                             m,
                                   int                             n,
                                   const hipblas_internal_type<T>* A,
                                   int                             lda,
                                   const hipblas_internal_type<T>* x,
                                   int                             incx,
                                   hipblas_internal_type<T>*       C,
                                   int                             ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDgmmBatched)(hipblasHandle_t                       handle,
                                          hipblasSideMode_t                     side,
                                          int                                   m,
                                          int                                   n,
                                          const hipblas_internal_type<T>* const A[],
                                          int                                   lda,
                                          const hipblas_internal_type<T>* const x[],
                                          int                                   incx,
                                          hipblas_internal_type<T>* const       C[],
                                          int                                   ldc,
                                          int                                   batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDgmmStridedBatched)(hipblasHandle_t                 handle,
                                                 hipblasSideMode_t               side,
                                                 int                             m,
                                                 int                             n,
                                                 const hipblas_internal_type<T>* A,
                                                 int                             lda,
                                                 hipblasStride                   stride_A,
                                                 const hipblas_internal_type<T>* x,
                                                 int                             incx,
                                                 hipblasStride                   stride_x,
                                                 hipblas_internal_type<T>*       C,
                                                 int                             ldc,
                                                 hipblasStride                   stride_C,
                                                 int                             batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDgmm_64)(hipblasHandle_t                 handle,
                                      hipblasSideMode_t               side,
                                      int64_t                         m,
                                      int64_t                         n,
                                      const hipblas_internal_type<T>* A,
                                      int64_t                         lda,
                                      const hipblas_internal_type<T>* x,
                                      int64_t                         incx,
                                      hipblas_internal_type<T>*       C,
                                      int64_t                         ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDgmmBatched_64)(hipblasHandle_t                       handle,
                                             hipblasSideMode_t                     side,
                                             int64_t                               m,
                                             int64_t                               n,
                                             const hipblas_internal_type<T>* const A[],
                                             int64_t                               lda,
                                             const hipblas_internal_type<T>* const x[],
                                             int64_t                               incx,
                                             hipblas_internal_type<T>* const       C[],
                                             int64_t                               ldc,
                                             int64_t                               batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDgmmStridedBatched_64)(hipblasHandle_t                 handle,
                                                    hipblasSideMode_t               side,
                                                    int64_t                         m,
                                                    int64_t                         n,
                                                    const hipblas_internal_type<T>* A,
                                                    int64_t                         lda,
                                                    hipblasStride                   stride_A,
                                                    const hipblas_internal_type<T>* x,
                                                    int64_t                         incx,
                                                    hipblasStride                   stride_x,
                                                    hipblas_internal_type<T>*       C,
                                                    int64_t                         ldc,
                                                    hipblasStride                   stride_C,
                                                    int64_t                         batch_count);

    MAP2CF_D64(hipblasDgmm, float, hipblasSdgmm);
    MAP2CF_D64(hipblasDgmm, double, hipblasDdgmm);
    MAP2CF_D64(hipblasDgmm, std::complex<float>, hipblasCdgmm);
    MAP2CF_D64(hipblasDgmm, std::complex<double>, hipblasZdgmm);

    MAP2CF_D64(hipblasDgmmBatched, float, hipblasSdgmmBatched);
    MAP2CF_D64(hipblasDgmmBatched, double, hipblasDdgmmBatched);
    MAP2CF_D64(hipblasDgmmBatched, std::complex<float>, hipblasCdgmmBatched);
    MAP2CF_D64(hipblasDgmmBatched, std::complex<double>, hipblasZdgmmBatched);

    MAP2CF_D64(hipblasDgmmStridedBatched, float, hipblasSdgmmStridedBatched);
    MAP2CF_D64(hipblasDgmmStridedBatched, double, hipblasDdgmmStridedBatched);
    MAP2CF_D64(hipblasDgmmStridedBatched, std::complex<float>, hipblasCdgmmStridedBatched);
    MAP2CF_D64(hipblasDgmmStridedBatched, std::complex<double>, hipblasZdgmmStridedBatched);

    // trtri
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrtri)(hipblasHandle_t                 handle,
                                    hipblasFillMode_t               uplo,
                                    hipblasDiagType_t               diag,
                                    int                             n,
                                    const hipblas_internal_type<T>* A,
                                    int                             lda,
                                    hipblas_internal_type<T>*       invA,
                                    int                             ldinvA);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrtriBatched)(hipblasHandle_t                       handle,
                                           hipblasFillMode_t                     uplo,
                                           hipblasDiagType_t                     diag,
                                           int                                   n,
                                           const hipblas_internal_type<T>* const A[],
                                           int                                   lda,
                                           hipblas_internal_type<T>*             invA[],
                                           int                                   ldinvA,
                                           int                                   batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrtriStridedBatched)(hipblasHandle_t                 handle,
                                                  hipblasFillMode_t               uplo,
                                                  hipblasDiagType_t               diag,
                                                  int                             n,
                                                  const hipblas_internal_type<T>* A,
                                                  int                             lda,
                                                  hipblasStride                   stride_A,
                                                  hipblas_internal_type<T>*       invA,
                                                  int                             ldinvA,
                                                  hipblasStride                   stride_invA,
                                                  int                             batch_count);

    MAP2CF(hipblasTrtri, float, hipblasStrtri);
    MAP2CF(hipblasTrtri, double, hipblasDtrtri);
    MAP2CF(hipblasTrtri, std::complex<float>, hipblasCtrtri);
    MAP2CF(hipblasTrtri, std::complex<double>, hipblasZtrtri);

    MAP2CF(hipblasTrtriBatched, float, hipblasStrtriBatched);
    MAP2CF(hipblasTrtriBatched, double, hipblasDtrtriBatched);
    MAP2CF(hipblasTrtriBatched, std::complex<float>, hipblasCtrtriBatched);
    MAP2CF(hipblasTrtriBatched, std::complex<double>, hipblasZtrtriBatched);

    MAP2CF(hipblasTrtriStridedBatched, float, hipblasStrtriStridedBatched);
    MAP2CF(hipblasTrtriStridedBatched, double, hipblasDtrtriStridedBatched);
    MAP2CF(hipblasTrtriStridedBatched, std::complex<float>, hipblasCtrtriStridedBatched);
    MAP2CF(hipblasTrtriStridedBatched, std::complex<double>, hipblasZtrtriStridedBatched);

#ifdef __HIP_PLATFORM_SOLVER__

    // getrf
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGetrf)(hipblasHandle_t           handle,
                                    const int                 n,
                                    hipblas_internal_type<T>* A,
                                    const int                 lda,
                                    int*                      ipiv,
                                    int*                      info);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGetrfBatched)(hipblasHandle_t                 handle,
                                           const int                       n,
                                           hipblas_internal_type<T>* const A[],
                                           const int                       lda,
                                           int*                            ipiv,
                                           int*                            info,
                                           const int                       batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGetrfStridedBatched)(hipblasHandle_t           handle,
                                                  const int                 n,
                                                  hipblas_internal_type<T>* A,
                                                  const int                 lda,
                                                  const hipblasStride       strideA,
                                                  int*                      ipiv,
                                                  const hipblasStride       strideP,
                                                  int*                      info,
                                                  const int                 batchCount);

    MAP2CF(hipblasGetrf, float, hipblasSgetrf);
    MAP2CF(hipblasGetrf, double, hipblasDgetrf);
    MAP2CF(hipblasGetrf, std::complex<float>, hipblasCgetrf);
    MAP2CF(hipblasGetrf, std::complex<double>, hipblasZgetrf);

    MAP2CF(hipblasGetrfBatched, float, hipblasSgetrfBatched);
    MAP2CF(hipblasGetrfBatched, double, hipblasDgetrfBatched);
    MAP2CF(hipblasGetrfBatched, std::complex<float>, hipblasCgetrfBatched);
    MAP2CF(hipblasGetrfBatched, std::complex<double>, hipblasZgetrfBatched);

    MAP2CF(hipblasGetrfStridedBatched, float, hipblasSgetrfStridedBatched);
    MAP2CF(hipblasGetrfStridedBatched, double, hipblasDgetrfStridedBatched);
    MAP2CF(hipblasGetrfStridedBatched, std::complex<float>, hipblasCgetrfStridedBatched);
    MAP2CF(hipblasGetrfStridedBatched, std::complex<double>, hipblasZgetrfStridedBatched);

    // getrs
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGetrs)(hipblasHandle_t           handle,
                                    const hipblasOperation_t  trans,
                                    const int                 n,
                                    const int                 nrhs,
                                    hipblas_internal_type<T>* A,
                                    const int                 lda,
                                    const int*                ipiv,
                                    hipblas_internal_type<T>* B,
                                    const int                 ldb,
                                    int*                      info);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGetrsBatched)(hipblasHandle_t                 handle,
                                           const hipblasOperation_t        trans,
                                           const int                       n,
                                           const int                       nrhs,
                                           hipblas_internal_type<T>* const A[],
                                           const int                       lda,
                                           const int*                      ipiv,
                                           hipblas_internal_type<T>* const B[],
                                           const int                       ldb,
                                           int*                            info,
                                           const int                       batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGetrsStridedBatched)(hipblasHandle_t           handle,
                                                  const hipblasOperation_t  trans,
                                                  const int                 n,
                                                  const int                 nrhs,
                                                  hipblas_internal_type<T>* A,
                                                  const int                 lda,
                                                  const hipblasStride       strideA,
                                                  const int*                ipiv,
                                                  const hipblasStride       strideP,
                                                  hipblas_internal_type<T>* B,
                                                  const int                 ldb,
                                                  const hipblasStride       strideB,
                                                  int*                      info,
                                                  const int                 batchCount);

    MAP2CF(hipblasGetrs, float, hipblasSgetrs);
    MAP2CF(hipblasGetrs, double, hipblasDgetrs);
    MAP2CF(hipblasGetrs, std::complex<float>, hipblasCgetrs);
    MAP2CF(hipblasGetrs, std::complex<double>, hipblasZgetrs);

    MAP2CF(hipblasGetrsBatched, float, hipblasSgetrsBatched);
    MAP2CF(hipblasGetrsBatched, double, hipblasDgetrsBatched);
    MAP2CF(hipblasGetrsBatched, std::complex<float>, hipblasCgetrsBatched);
    MAP2CF(hipblasGetrsBatched, std::complex<double>, hipblasZgetrsBatched);

    MAP2CF(hipblasGetrsStridedBatched, float, hipblasSgetrsStridedBatched);
    MAP2CF(hipblasGetrsStridedBatched, double, hipblasDgetrsStridedBatched);
    MAP2CF(hipblasGetrsStridedBatched, std::complex<float>, hipblasCgetrsStridedBatched);
    MAP2CF(hipblasGetrsStridedBatched, std::complex<double>, hipblasZgetrsStridedBatched);

    // getri
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGetriBatched)(hipblasHandle_t                 handle,
                                           const int                       n,
                                           hipblas_internal_type<T>* const A[],
                                           const int                       lda,
                                           int*                            ipiv,
                                           hipblas_internal_type<T>* const C[],
                                           const int                       ldc,
                                           int*                            info,
                                           const int                       batchCount);

    MAP2CF(hipblasGetriBatched, float, hipblasSgetriBatched);
    MAP2CF(hipblasGetriBatched, double, hipblasDgetriBatched);
    MAP2CF(hipblasGetriBatched, std::complex<float>, hipblasCgetriBatched);
    MAP2CF(hipblasGetriBatched, std::complex<double>, hipblasZgetriBatched);

    // geqrf
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGeqrf)(hipblasHandle_t           handle,
                                    const int                 m,
                                    const int                 n,
                                    hipblas_internal_type<T>* A,
                                    const int                 lda,
                                    hipblas_internal_type<T>* ipiv,
                                    int*                      info);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGeqrfBatched)(hipblasHandle_t                 handle,
                                           const int                       m,
                                           const int                       n,
                                           hipblas_internal_type<T>* const A[],
                                           const int                       lda,
                                           hipblas_internal_type<T>* const ipiv[],
                                           int*                            info,
                                           const int                       batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGeqrfStridedBatched)(hipblasHandle_t           handle,
                                                  const int                 m,
                                                  const int                 n,
                                                  hipblas_internal_type<T>* A,
                                                  const int                 lda,
                                                  const hipblasStride       strideA,
                                                  hipblas_internal_type<T>* ipiv,
                                                  const hipblasStride       strideP,
                                                  int*                      info,
                                                  const int                 batchCount);

    MAP2CF(hipblasGeqrf, float, hipblasSgeqrf);
    MAP2CF(hipblasGeqrf, double, hipblasDgeqrf);
    MAP2CF(hipblasGeqrf, std::complex<float>, hipblasCgeqrf);
    MAP2CF(hipblasGeqrf, std::complex<double>, hipblasZgeqrf);

    MAP2CF(hipblasGeqrfBatched, float, hipblasSgeqrfBatched);
    MAP2CF(hipblasGeqrfBatched, double, hipblasDgeqrfBatched);
    MAP2CF(hipblasGeqrfBatched, std::complex<float>, hipblasCgeqrfBatched);
    MAP2CF(hipblasGeqrfBatched, std::complex<double>, hipblasZgeqrfBatched);

    MAP2CF(hipblasGeqrfStridedBatched, float, hipblasSgeqrfStridedBatched);
    MAP2CF(hipblasGeqrfStridedBatched, double, hipblasDgeqrfStridedBatched);
    MAP2CF(hipblasGeqrfStridedBatched, std::complex<float>, hipblasCgeqrfStridedBatched);
    MAP2CF(hipblasGeqrfStridedBatched, std::complex<double>, hipblasZgeqrfStridedBatched);

    // gels
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGels)(hipblasHandle_t           handle,
                                   hipblasOperation_t        trans,
                                   const int                 m,
                                   const int                 n,
                                   const int                 nrhs,
                                   hipblas_internal_type<T>* A,
                                   const int                 lda,
                                   hipblas_internal_type<T>* B,
                                   const int                 ldb,
                                   int*                      info,
                                   int*                      deviceInfo);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGelsBatched)(hipblasHandle_t                 handle,
                                          hipblasOperation_t              trans,
                                          const int                       m,
                                          const int                       n,
                                          const int                       nrhs,
                                          hipblas_internal_type<T>* const A[],
                                          const int                       lda,
                                          hipblas_internal_type<T>* const B[],
                                          const int                       ldb,
                                          int*                            info,
                                          int*                            deviceInfo,
                                          const int                       batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGelsStridedBatched)(hipblasHandle_t           handle,
                                                 hipblasOperation_t        trans,
                                                 const int                 m,
                                                 const int                 n,
                                                 const int                 nrhs,
                                                 hipblas_internal_type<T>* A,
                                                 const int                 lda,
                                                 const hipblasStride       strideA,
                                                 hipblas_internal_type<T>* B,
                                                 const int                 ldb,
                                                 const hipblasStride       strideB,
                                                 int*                      info,
                                                 int*                      deviceInfo,
                                                 const int                 batchCount);

    MAP2CF(hipblasGels, float, hipblasSgels);
    MAP2CF(hipblasGels, double, hipblasDgels);
    MAP2CF(hipblasGels, std::complex<float>, hipblasCgels);
    MAP2CF(hipblasGels, std::complex<double>, hipblasZgels);

    MAP2CF(hipblasGelsBatched, float, hipblasSgelsBatched);
    MAP2CF(hipblasGelsBatched, double, hipblasDgelsBatched);
    MAP2CF(hipblasGelsBatched, std::complex<float>, hipblasCgelsBatched);
    MAP2CF(hipblasGelsBatched, std::complex<double>, hipblasZgelsBatched);

    MAP2CF(hipblasGelsStridedBatched, float, hipblasSgelsStridedBatched);
    MAP2CF(hipblasGelsStridedBatched, double, hipblasDgelsStridedBatched);
    MAP2CF(hipblasGelsStridedBatched, std::complex<float>, hipblasCgelsStridedBatched);
    MAP2CF(hipblasGelsStridedBatched, std::complex<double>, hipblasZgelsStridedBatched);

#endif
}

#endif // _ROCBLAS_HPP_
