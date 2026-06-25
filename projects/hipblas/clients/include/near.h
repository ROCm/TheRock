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
#ifndef _NEAR_H
#define _NEAR_H

#include "hipblas.h"
#include "host_vector.hpp"
#include "unit.h"

#ifdef GOOGLE_TEST
#include "gtest/gtest.h"
#endif

/* =====================================================================

    Google Unit check: ASSERT_EQ( elementof(A), elementof(B))

   =================================================================== */

/*!\file
 * \brief compares two results (usually, CPU and GPU results); provides Google Unit check.
 */

/* ========================================Gtest Unit Check
 * ==================================================== */

// sqrt(0.5) factor for complex cutoff calculations
constexpr double sqrthalf = 0.7071067811865475244;

/*! \brief Template: gtest near compare two matrices float/double/complex */
template <typename T>
void near_check_general(int64_t M, int64_t N, int64_t lda, T* hCPU, T* hGPU, double abs_error);

template <typename T>
void near_check_general(
    int64_t M, int64_t N, int64_t lda, host_vector<T> hCPU, host_vector<T> hGPU, double abs_error);

template <typename T>
void near_check_general(int64_t       M,
                        int64_t       N,
                        int64_t       batch_count,
                        int64_t       lda,
                        hipblasStride stride_A,
                        T*            hCPU,
                        T*            hGPU,
                        double        abs_error);

template <typename T>
void near_check_general(
    int64_t M, int64_t N, int64_t batch_count, int64_t lda, T** hCPU, T** hGPU, double abs_error);

template <typename T>
void near_check_general(int64_t        M,
                        int64_t        N,
                        int64_t        batch_count,
                        int64_t        lda,
                        host_vector<T> hCPU[],
                        host_vector<T> hGPU[],
                        double         abs_error);

template <typename T, typename T_hpa = T>
void near_check_mixed(int64_t                        M,
                      int64_t                        N,
                      int64_t                        lda,
                      const std::remove_cv_t<T_hpa>* hCPU,
                      const T*                       hGPU,
                      double                         abs_error);

// mixed precision or hpl
template <class T>
inline constexpr double error_tolerance = get_epsilon<T>();

template <>
inline constexpr double error_tolerance<hipblasBfloat16> = 1 / 100.0;

// 2 ^ -14, smallest positive normal number for IEEE16
template <>
inline constexpr double error_tolerance<hipblasHalf> = 0.000061035;

template <>
inline constexpr double error_tolerance<std::complex<float>> = 1 / 10000.0;

template <>
inline constexpr double error_tolerance<std::complex<double>> = 1 / 1000000.0;

// mixed precision or hpl
template <class Tc, class Ti, class To>
static constexpr double sum_error_tolerance_for_gfx11 = get_epsilon<Tc>();

template <>
inline constexpr double sum_error_tolerance_for_gfx11<float, hipblasBfloat16, float> = 1 / 10.0;

template <>
inline constexpr double
    sum_error_tolerance_for_gfx11<float, hipblasBfloat16, hipblasBfloat16> = 1 / 10.0;

template <>
inline constexpr double sum_error_tolerance_for_gfx11<float, hipblasHalf, float> = 1 / 100.0;

template <>
inline constexpr double sum_error_tolerance_for_gfx11<float, hipblasHalf, hipblasHalf> = 1 / 100.0;

template <>
inline constexpr double
    sum_error_tolerance_for_gfx11<hipblasHalf, hipblasHalf, hipblasHalf> = 1 / 100.0;

template <>
inline constexpr double sum_error_tolerance_for_gfx11<std::complex<float>,
                                                      std::complex<float>,
                                                      std::complex<float>> = 1 / 10000.0;

template <>
inline constexpr double sum_error_tolerance_for_gfx11<std::complex<double>,
                                                      std::complex<double>,
                                                      std::complex<double>> = 1 / 1000000.0;

#endif
