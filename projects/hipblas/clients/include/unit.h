/* ************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef _UNIT_H
#define _UNIT_H

#include "hipblas.h"
#include "host_vector.hpp"

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

/*! \brief Template: gtest unit compare two matrices float/double/complex */
// Do not put a wrapper over ASSERT_FLOAT_EQ, sincer assert exit the current function NOT the test
// case
// a wrapper will cause the loop keep going
template <typename T>
void unit_check_general(int64_t M, int64_t N, int64_t lda, T* hCPU, T* hGPU);

template <typename T>
void unit_check_general(int64_t       M,
                        int64_t       N,
                        int64_t       batch_count,
                        int64_t       lda,
                        hipblasStride stride_A,
                        T*            hCPU,
                        T*            hGPU);

template <typename T>
void unit_check_general(int64_t M, int64_t N, int64_t batch_count, int64_t lda, T** hCPU, T** hGPU);

template <typename T>
void unit_check_general(int64_t        M,
                        int64_t        N,
                        int64_t        batch_count,
                        int64_t        lda,
                        host_vector<T> hCPU[],
                        host_vector<T> hGPU[]);

template <typename T, typename T_hpa = T>
void unit_check_mixed(
    int64_t M, int64_t N, int64_t lda, const std::remove_cv_t<T_hpa>* hCPU, const T* hGPU);

template <typename T>
void unit_check_error(T error, T tolerance)
{
#ifdef GOOGLE_TEST
    ASSERT_LE(error, tolerance);
#endif
}

template <typename T, typename Tex = T>
void unit_check_nrm2(T cpu_result, T gpu_result, int64_t vector_length)
{
    T allowable_error = vector_length * hipblas_type_epsilon<Tex> * cpu_result;
    if(allowable_error == 0)
        allowable_error = vector_length * hipblas_type_epsilon<Tex>;
#ifdef GOOGLE_TEST
    ASSERT_NEAR(cpu_result, gpu_result, allowable_error);
#endif
}

template <typename T, typename Tex = T>
void unit_check_nrm2(int64_t        batch_count,
                     host_vector<T> cpu_result,
                     host_vector<T> gpu_result,
                     int64_t        vector_length)
{
    for(int64_t b = 0; b < batch_count; b++)
    {
        T allowable_error = vector_length * hipblas_type_epsilon<Tex> * cpu_result[b];
        if(allowable_error == 0)
            allowable_error = vector_length * hipblas_type_epsilon<Tex>;
#ifdef GOOGLE_TEST
        ASSERT_NEAR(cpu_result[b], gpu_result[b], allowable_error);
#endif
    }
}

template <typename T, std::enable_if_t<!is_complex<T>, int> = 0>
constexpr double get_epsilon()
{
    return std::numeric_limits<T>::epsilon();
}

template <typename T, std::enable_if_t<+is_complex<T>, int> = 0>
constexpr auto get_epsilon()
{
    return get_epsilon<decltype(std::real(T{}))>();
}

#endif
