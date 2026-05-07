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
#ifndef _NORM_H
#define _NORM_H

#include "hipblas.h"
#include "host_batch_matrix.hpp"
#include "host_batch_vector.hpp"
#include "host_vector.hpp"
#include "type_utils.h"
#include "utility.h"

/* =====================================================================
        Norm check: norm(A-B)/norm(A), evaluate relative error
    =================================================================== */

/*!\file
 * \brief compares two results (usually, CPU and GPU results); provides Norm check
 */

/* ========================================Norm Check
 * ==================================================== */

/*! \brief  Template: norm check for general Matrix: float/doubel/complex  */

// see check_norm.cpp for template speciliazation
// use auto as the return type is only allowed in c++14
// convert float/float to double
template <typename T>
double norm_check_general(char norm_type, int64_t M, int64_t N, int64_t lda, T* hCPU, T* hGPU);

/*! \brief  Template: norm check for hermitian/symmetric Matrix: float/double/complex */

template <typename T>
double norm_check_symmetric(char norm_type, char uplo, int64_t N, int64_t lda, T* hCPU, T* hGPU);

template <typename T>
double norm_check_general(char           norm_type,
                          int64_t        M,
                          int64_t        N,
                          int64_t        lda,
                          host_vector<T> hCPU[],
                          host_vector<T> hGPU[],
                          int64_t        batch_count)
{
    // norm type can be O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries
    //
    // use triangle inequality ||a+b|| <= ||a|| + ||b|| to calculate upper limit for Frobenius norm
    // of strided batched matrix

    double cumulative_error = 0.0;

    for(int64_t i = 0; i < batch_count; i++)
    {
        auto index = i;

        auto error = norm_check_general<T>(norm_type, M, N, lda, hCPU[index], hGPU[index]);

        if(norm_type == 'F' || norm_type == 'f')
        {
            cumulative_error += error;
        }
        else if(norm_type == 'O' || norm_type == 'o' || norm_type == 'I' || norm_type == 'i')
        {
            cumulative_error = cumulative_error > error ? cumulative_error : error;
        }
    }

    return cumulative_error;
}

/* ============== Norm Check for strided_batched case ============= */
template <typename T>
double norm_check_general(char    norm_type,
                          int64_t M,
                          int64_t N,
                          int64_t lda,
                          int64_t stride_a,
                          T*      hCPU,
                          T*      hGPU,
                          int64_t batch_count)
{
    // norm type can be O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries
    //
    // use triangle inequality ||a+b|| <= ||a|| + ||b|| to calculate upper limit for Frobenius norm
    // of strided batched matrix

    double cumulative_error = 0.0;

    for(size_t i = 0; i < batch_count; i++)
    {
        auto index = i * stride_a;

        auto error = norm_check_general(norm_type, M, N, lda, hCPU + index, hGPU + index);

        if(norm_type == 'F' || norm_type == 'f')
        {
            cumulative_error += error;
        }
        else if(norm_type == 'O' || norm_type == 'o' || norm_type == 'I' || norm_type == 'i')
        {
            cumulative_error = cumulative_error > error ? cumulative_error : error;
        }
    }

    return cumulative_error;
}

template <typename T, typename T_hpa>
double norm_check_general(char                      norm_type,
                          int64_t                   M,
                          int64_t                   N,
                          int64_t                   lda,
                          host_batch_vector<T_hpa>& hCPU,
                          host_batch_vector<T>&     hGPU,
                          int64_t                   batch_count)
{
    // norm type can be O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries
    //
    // use triangle inequality ||a+b|| <= ||a|| + ||b|| to calculate upper limit for Frobenius norm
    // of strided batched matrix

    double cumulative_error = 0.0;

    for(int64_t i = 0; i < batch_count; i++)
    {
        auto index = i;

        auto error = norm_check_general<T>(norm_type, M, N, lda, hCPU[index], hGPU[index]);

        if(norm_type == 'F' || norm_type == 'f')
        {
            cumulative_error += error;
        }
        else if(norm_type == 'O' || norm_type == 'o' || norm_type == 'I' || norm_type == 'i')
        {
            cumulative_error = cumulative_error > error ? cumulative_error : error;
        }
    }

    return cumulative_error;
}

template <typename T, typename T_hpa>
double norm_check_general(char                      norm_type,
                          int64_t                   M,
                          int64_t                   N,
                          int64_t                   lda,
                          host_batch_matrix<T_hpa>& hCPU,
                          host_batch_matrix<T>&     hGPU,
                          int64_t                   batch_count)
{
    // norm type can be O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries
    //
    // use triangle inequality ||a+b|| <= ||a|| + ||b|| to calculate upper limit for Frobenius norm
    // of strided batched matrix

    double cumulative_error = 0.0;

    for(int64_t i = 0; i < batch_count; i++)
    {
        auto index = i;

        auto error = norm_check_general<T>(norm_type, M, N, lda, hCPU[index], hGPU[index]);

        if(norm_type == 'F' || norm_type == 'f')
        {
            cumulative_error += error;
        }
        else if(norm_type == 'O' || norm_type == 'o' || norm_type == 'I' || norm_type == 'i')
        {
            cumulative_error = cumulative_error > error ? cumulative_error : error;
        }
    }

    return cumulative_error;
}

template <typename T, typename U>
double norm_check_mixed(char norm_type, int64_t M, int64_t N, int64_t lda, T* hCPU, U* hGPU)
{
    if constexpr(is_complex<T>)
    {
        return norm_check_general(norm_type, M, N, lda, hCPU, hGPU);
    }
    else
    {
        size_t              size = N * (size_t)lda;
        host_vector<double> hCPU_double(size);
        host_vector<double> hGPU_double(size);

        for(int64_t i = 0; i < N; i++)
        {
            for(int64_t j = 0; j < M; j++)
            {
                size_t idx       = j + i * (size_t)lda;
                hCPU_double[idx] = (double)(hCPU[idx]);
                if constexpr(std::is_same_v<U, hipblasHalf>)
                {
                    hGPU_double[idx] = half_to_float(hGPU[idx]);
                }
                else if constexpr(std::is_same_v<U, hipblasBfloat16>)
                {
                    hGPU_double[idx] = bfloat16_to_float(hGPU[idx]);
                }
                else
                {
                    hGPU_double[idx] = (double)hGPU[idx];
                }
            }
        }

        return norm_check_general<double>(norm_type, M, N, lda, hCPU_double, hGPU_double);
    }
}

template <typename T>
double vector_norm_1(int64_t M, int64_t incx, T* hx_gold, T* hx)
{
    double max_err_scal = 0.0;
    double max_err      = 0.0;
    for(int64_t i = 0; i < M; i++)
    {
        if constexpr(std::is_same_v<T, hipblasHalf>)
        {
            double hx_value;
            double gold_value;
            hx_value   = half_to_float(hx[i * incx]);
            gold_value = half_to_float(hx_gold[i * incx]);
            max_err += hipblas_abs(gold_value - hx_value);
            max_err_scal += hipblas_abs(gold_value);
        }
        else
        {
            max_err += hipblas_abs(hx_gold[i * incx] - hx[i * incx]);
            max_err_scal += hipblas_abs(hx_gold[i * incx]);
        }
    }

    if(hipblas_abs(max_err_scal) < 1e6)
        max_err_scal = 1;

    return max_err / max_err_scal;
}

#endif
