/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
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

#pragma once

#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_math.hpp"
#include "hipblaslt_ostream.hpp"
#include "hipblaslt_random.hpp"
#include <cinttypes>
#include <hipblaslt/hipblaslt.h>
#include <iostream>
#include <omp.h>
#include <vector>

enum class ABC_dims
{
    A,
    B,
    C
};

void hipblaslt_init_device(ABC_dims                 ABC_dims,
                           hipblaslt_initialization init,
                           bool                     is_nan,
                           void*                    A,
                           size_t                   M,
                           size_t                   N,
                           size_t                   lda,
                           hipDataType              type,
                           size_t                   stride,
                           size_t                   batch_count);

/* ============================================================================================ */
/*! \brief  matrix/vector initialization: */
// for vector x (M=1, N=lengthX, lda=incx);
// for complex number, the real/imag part would be initialized with the same value

// Initialize matrices with random values
template <typename T>
inline void
    hipblaslt_init(T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
    {
        size_t b_idx = i_batch * stride;
        for(size_t j = 0; j < N; ++j)
        {
            size_t col_idx = b_idx + j * lda;
            if(M > 4)
                random_run_generator<T>(A + col_idx, M);
            else
            {
                for(size_t i = 0; i < M; ++i)
                    A[col_idx + i] = random_generator<T>();
            }
        }
    }
}

// Initialize matrices with random values
template <typename T>
inline void hipblaslt_init_small(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
    {
        size_t b_idx = i_batch * stride;
        for(size_t j = 0; j < N; ++j)
        {
            size_t col_idx = b_idx + j * lda;
            if(M > 4)
                random_run_generator_small<T>(A + col_idx, M);
            else
            {
                for(size_t i = 0; i < M; ++i)
                    A[col_idx + i] = random_generator_small<T>();
            }
        }
    }
}

// Initialize matrices with random values
inline void hipblaslt_init(void*       A,
                           size_t      M,
                           size_t      N,
                           size_t      lda,
                           hipDataType type,
                           size_t      stride      = 0,
                           size_t      batch_count = 1)
{
    switch(type)
    {
    case HIP_R_32F:
        hipblaslt_init<float>(static_cast<float*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_64F:
        hipblaslt_init<double>(static_cast<double*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16F:
        hipblaslt_init<hipblasLtHalf>(
            static_cast<hipblasLtHalf*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16BF:
        hipblaslt_init<hip_bfloat16>(static_cast<hip_bfloat16*>(A), M, N, lda, stride, batch_count);
        break;
#if HIP_FP8_TYPE_FNUZ
    case HIP_R_8F_E4M3_FNUZ:
        hipblaslt_init<hipblaslt_f8_fnuz>(
            static_cast<hipblaslt_f8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        hipblaslt_init<hipblaslt_bf8_fnuz>(
            static_cast<hipblaslt_bf8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
#endif
#if HIP_FP8_TYPE_OCP
    case HIP_R_8F_E4M3:
        hipblaslt_init<hipblaslt_f8>(static_cast<hipblaslt_f8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2:
        hipblaslt_init<hipblaslt_bf8>(
            static_cast<hipblaslt_bf8*>(A), M, N, lda, stride, batch_count);
        break;
#endif
    case HIP_R_32I:
        hipblaslt_init<int32_t>(static_cast<int32_t*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8I:
        hipblaslt_init<hipblasLtInt8>(
            static_cast<hipblasLtInt8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_6F_E2M3_EXT:
        hipblaslt_cerr << "hipblaslt_init not supports FP6" << std::endl;
        break;
    case HIP_R_6F_E3M2_EXT:
        hipblaslt_cerr << "hipblaslt_init not supports BF6" << std::endl;
        break;
    case HIP_R_4F_E2M1_EXT:
        hipblaslt_cerr << "hipblaslt_init not supports FP4" << std::endl;
        break;
    default:
        hipblaslt_cerr << "Error type in hipblaslt_init" << std::endl;
        break;
    }
}

inline void hipblaslt_init_small(void*       A,
                                 size_t      M,
                                 size_t      N,
                                 size_t      lda,
                                 hipDataType type,
                                 size_t      stride      = 0,
                                 size_t      batch_count = 1)
{
    switch(type)
    {
    case HIP_R_32F:
        hipblaslt_init_small<float>(static_cast<float*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_64F:
        hipblaslt_init_small<double>(static_cast<double*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16F:
        hipblaslt_init_small<hipblasLtHalf>(
            static_cast<hipblasLtHalf*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_32I:
        hipblaslt_init_small<int32_t>(static_cast<int32_t*>(A), M, N, lda, stride, batch_count);
        break;
    default:
        hipblaslt_cerr << "Error type in hipblaslt_init_small" << std::endl;
        break;
    }
}

template <typename T>
inline void hipblaslt_init_sin(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
#pragma omp parallel for
        for(size_t j = 0; j < N; ++j)
        {
            size_t offset      = j * lda + i_batch * stride;
            size_t offsetValue = j * M + i_batch * M * N;
            for(size_t i = 0; i < M; ++i)
                A[i + offset] = static_cast<T>(sin(double(i + offsetValue))); //force cast to double
        }
}

inline void hipblaslt_init_sin(void*       A,
                               size_t      M,
                               size_t      N,
                               size_t      lda,
                               hipDataType type,
                               size_t      stride      = 0,
                               size_t      batch_count = 1)
{
    switch(type)
    {
    case HIP_R_32F:
        hipblaslt_init_sin<float>(static_cast<float*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_64F:
        hipblaslt_init_sin<double>(static_cast<double*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16F:
        hipblaslt_init_sin<hipblasLtHalf>(
            static_cast<hipblasLtHalf*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16BF:
        hipblaslt_init_sin<hip_bfloat16>(
            static_cast<hip_bfloat16*>(A), M, N, lda, stride, batch_count);
        break;
#if HIP_FP8_TYPE_FNUZ
    case HIP_R_8F_E4M3_FNUZ:
        hipblaslt_init_sin<hipblaslt_f8_fnuz>(
            static_cast<hipblaslt_f8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        hipblaslt_init_sin<hipblaslt_bf8_fnuz>(
            static_cast<hipblaslt_bf8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
#endif
#if HIP_FP8_TYPE_OCP
    case HIP_R_8F_E4M3:
        hipblaslt_init_sin<hipblaslt_f8>(
            static_cast<hipblaslt_f8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2:
        hipblaslt_init_sin<hipblaslt_bf8>(
            static_cast<hipblaslt_bf8*>(A), M, N, lda, stride, batch_count);
        break;
#endif
    case HIP_R_32I:
        hipblaslt_init_sin<int32_t>(static_cast<int32_t*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8I:
        hipblaslt_init_sin<hipblasLtInt8>(
            static_cast<hipblasLtInt8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_6F_E2M3_EXT:
        hipblaslt_cerr << "hipblaslt_init_sin not supports FP6" << std::endl;
        break;
    case HIP_R_6F_E3M2_EXT:
        hipblaslt_cerr << "hipblaslt_init_sin not supports BF6" << std::endl;
        break;
    case HIP_R_4F_E2M1_EXT:
        hipblaslt_cerr << "hipblaslt_init_sin not supports FP4" << std::endl;
        break;
    default:
        hipblaslt_cerr << "Error type in hipblaslt_init_sin" << std::endl;
        break;
    }
}

// Initialize matrix so adjacent entries have alternating sign.
// In gemm if either A or B are initialized with alternating
// sign the reduction sum will be summing positive
// and negative numbers, so it should not get too large.
// This helps reduce floating point inaccuracies for 16bit
// arithmetic where the exponent has only 5 bits, and the
// mantissa 10 bits.
template <typename T>
inline void hipblaslt_init_alternating_sign(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
#pragma omp parallel for
        for(size_t j = 0; j < N; ++j)
        {
            size_t offset = j * lda + i_batch * stride;
            for(size_t i = 0; i < M; ++i)
            {
                auto value    = random_generator<T>();
                A[i + offset] = (i ^ j) & 1 ? value : negate(value);
            }
        }
}

inline void hipblaslt_init_alternating_sign(void*       A,
                                            size_t      M,
                                            size_t      N,
                                            size_t      lda,
                                            hipDataType type,
                                            size_t      stride      = 0,
                                            size_t      batch_count = 1)
{
    switch(type)
    {
    case HIP_R_32F:
        hipblaslt_init_alternating_sign<float>(
            static_cast<float*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_64F:
        hipblaslt_init_alternating_sign<double>(
            static_cast<double*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16F:
        hipblaslt_init_alternating_sign<hipblasLtHalf>(
            static_cast<hipblasLtHalf*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16BF:
        hipblaslt_init_alternating_sign<hip_bfloat16>(
            static_cast<hip_bfloat16*>(A), M, N, lda, stride, batch_count);
        break;
#if HIP_FP8_TYPE_FNUZ
    case HIP_R_8F_E4M3_FNUZ:
        hipblaslt_init_alternating_sign<hipblaslt_f8_fnuz>(
            static_cast<hipblaslt_f8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        hipblaslt_init_alternating_sign<hipblaslt_bf8_fnuz>(
            static_cast<hipblaslt_bf8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
#endif
#if HIP_FP8_TYPE_OCP
    case HIP_R_8F_E4M3:
        hipblaslt_init_alternating_sign<hipblaslt_f8>(
            static_cast<hipblaslt_f8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2:
        hipblaslt_init_alternating_sign<hipblaslt_bf8>(
            static_cast<hipblaslt_bf8*>(A), M, N, lda, stride, batch_count);
        break;
#endif
    case HIP_R_32I:
        hipblaslt_init_alternating_sign<int32_t>(
            static_cast<int32_t*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8I:
        hipblaslt_init_alternating_sign<hipblasLtInt8>(
            static_cast<hipblasLtInt8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_6F_E2M3_EXT:
        hipblaslt_cerr << "hipblaslt_init_alternating_sign not supports FP6" << std::endl;
        break;
    case HIP_R_6F_E3M2_EXT:
        hipblaslt_cerr << "hipblaslt_init_alternating_sign not supports BF6" << std::endl;
        break;
    case HIP_R_4F_E2M1_EXT:
        hipblaslt_cerr << "hipblaslt_init_alternating_sign not supports FP4" << std::endl;
        break;
    default:
        hipblaslt_cerr << "Error type in hipblaslt_init_alternating_sign" << std::endl;
        break;
    }
}

// Initialize matrix so adjacent entries have alternating sign.
template <typename T>
inline void hipblaslt_init_hpl_alternating_sign(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
#pragma omp parallel for
        for(size_t j = 0; j < N; ++j)
        {
            size_t offset = j * lda + i_batch * stride;
            for(size_t i = 0; i < M; ++i)
            {
                auto value    = random_hpl_generator<T>();
                A[i + offset] = (i ^ j) & 1 ? value : negate(value);
            }
        }
}

inline void hipblaslt_init_hpl_alternating_sign(void*       A,
                                                size_t      M,
                                                size_t      N,
                                                size_t      lda,
                                                hipDataType type,
                                                size_t      stride      = 0,
                                                size_t      batch_count = 1)
{
    switch(type)
    {
    case HIP_R_32F:
        hipblaslt_init_hpl_alternating_sign<float>(
            static_cast<float*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_64F:
        hipblaslt_init_hpl_alternating_sign<double>(
            static_cast<double*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16F:
        hipblaslt_init_hpl_alternating_sign<hipblasLtHalf>(
            static_cast<hipblasLtHalf*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16BF:
        hipblaslt_init_hpl_alternating_sign<hip_bfloat16>(
            static_cast<hip_bfloat16*>(A), M, N, lda, stride, batch_count);
        break;
#if HIP_FP8_TYPE_FNUZ
    case HIP_R_8F_E4M3_FNUZ:
        hipblaslt_init_hpl_alternating_sign<hipblaslt_f8_fnuz>(
            static_cast<hipblaslt_f8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        hipblaslt_init_hpl_alternating_sign<hipblaslt_bf8_fnuz>(
            static_cast<hipblaslt_bf8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
#endif
#if HIP_FP8_TYPE_OCP
    case HIP_R_8F_E4M3:
        hipblaslt_init_hpl_alternating_sign<hipblaslt_f8>(
            static_cast<hipblaslt_f8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2:
        hipblaslt_init_hpl_alternating_sign<hipblaslt_bf8>(
            static_cast<hipblaslt_bf8*>(A), M, N, lda, stride, batch_count);
        break;
#endif
    case HIP_R_32I:
        hipblaslt_init_hpl_alternating_sign<int32_t>(
            static_cast<int32_t*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8I:
        hipblaslt_init_hpl_alternating_sign<hipblasLtInt8>(
            static_cast<hipblasLtInt8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_6F_E2M3_EXT:
        hipblaslt_cerr << "hipblaslt_init_hpl_alternating_sign not supports FP6" << std::endl;
        break;
    case HIP_R_6F_E3M2_EXT:
        hipblaslt_cerr << "hipblaslt_init_hpl_alternating_sign not supports BF6" << std::endl;
        break;
    case HIP_R_4F_E2M1_EXT:
        hipblaslt_cerr << "hipblaslt_init_hpl_alternating_sign not supports FP4" << std::endl;
        break;
    default:
        hipblaslt_cerr << "Error type in hipblaslt_init_hpl_alternating_sign" << std::endl;
        break;
    }
}

template <typename T>
inline void hipblaslt_init_cos(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
#pragma omp parallel for
        for(size_t j = 0; j < N; ++j)
        {
            size_t offset      = j * lda + i_batch * stride;
            size_t offsetValue = j * M + i_batch * M * N;
            for(size_t i = 0; i < M; ++i)
                A[i + offset] = T(cos(double(i + offsetValue))); //force cast to double
        }
}

inline void hipblaslt_init_cos(void*       A,
                               size_t      M,
                               size_t      N,
                               size_t      lda,
                               hipDataType type,
                               size_t      stride      = 0,
                               size_t      batch_count = 1)
{
    switch(type)
    {
    case HIP_R_32F:
        hipblaslt_init_cos<float>(static_cast<float*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_64F:
        hipblaslt_init_cos<double>(static_cast<double*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16F:
        hipblaslt_init_cos<hipblasLtHalf>(
            static_cast<hipblasLtHalf*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16BF:
        hipblaslt_init_cos<hip_bfloat16>(
            static_cast<hip_bfloat16*>(A), M, N, lda, stride, batch_count);
        break;
#if HIP_FP8_TYPE_FNUZ
    case HIP_R_8F_E4M3_FNUZ:
        hipblaslt_init_cos<hipblaslt_f8_fnuz>(
            static_cast<hipblaslt_f8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        hipblaslt_init_cos<hipblaslt_bf8_fnuz>(
            static_cast<hipblaslt_bf8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
#endif
#if HIP_FP8_TYPE_OCP
    case HIP_R_8F_E4M3:
        hipblaslt_init_cos<hipblaslt_f8>(
            static_cast<hipblaslt_f8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2:
        hipblaslt_init_cos<hipblaslt_bf8>(
            static_cast<hipblaslt_bf8*>(A), M, N, lda, stride, batch_count);
        break;
#endif
    case HIP_R_32I:
        hipblaslt_init_cos<int32_t>(static_cast<int32_t*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8I:
        hipblaslt_init_cos<hipblasLtInt8>(
            static_cast<hipblasLtInt8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_6F_E2M3_EXT:
        hipblaslt_cerr << "hipblaslt_init_cos not supports FP6" << std::endl;
        break;
    case HIP_R_6F_E3M2_EXT:
        hipblaslt_cerr << "hipblaslt_init_cos not supports BF6" << std::endl;
        break;
    case HIP_R_4F_E2M1_EXT:
        hipblaslt_cerr << "hipblaslt_init_cos not supports FP4" << std::endl;
        break;
    default:
        hipblaslt_cerr << "Error type in hipblaslt_init_cos" << std::endl;
        break;
    }
}

// Initialize vector with HPL-like random values
template <typename T>
inline void hipblaslt_init_hpl(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = random_hpl_generator<T>();
}

template <typename T>
inline void hipblaslt_init_hpl(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = random_hpl_generator<T>();
}

inline void hipblaslt_init_hpl(void*       A,
                               size_t      M,
                               size_t      N,
                               size_t      lda,
                               hipDataType type,
                               size_t      stride      = 0,
                               size_t      batch_count = 1)
{
    switch(type)
    {
    case HIP_R_32F:
        hipblaslt_init_hpl<float>(static_cast<float*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_64F:
        hipblaslt_init_hpl<double>(static_cast<double*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16F:
        hipblaslt_init_hpl<hipblasLtHalf>(
            static_cast<hipblasLtHalf*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16BF:
        hipblaslt_init_hpl<hip_bfloat16>(
            static_cast<hip_bfloat16*>(A), M, N, lda, stride, batch_count);
        break;
#if HIP_FP8_TYPE_FNUZ
    case HIP_R_8F_E4M3_FNUZ:
        hipblaslt_init_hpl<hipblaslt_f8_fnuz>(
            static_cast<hipblaslt_f8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        hipblaslt_init_hpl<hipblaslt_bf8_fnuz>(
            static_cast<hipblaslt_bf8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
#endif
#if HIP_FP8_TYPE_OCP
    case HIP_R_8F_E4M3:
        hipblaslt_init_hpl<hipblaslt_f8>(
            static_cast<hipblaslt_f8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2:
        hipblaslt_init_hpl<hipblaslt_bf8>(
            static_cast<hipblaslt_bf8*>(A), M, N, lda, stride, batch_count);
        break;
#endif
    case HIP_R_32I:
        hipblaslt_init_hpl<int32_t>(static_cast<int32_t*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8I:
        hipblaslt_init_hpl<hipblasLtInt8>(
            static_cast<hipblasLtInt8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_6F_E2M3_EXT:
        hipblaslt_cerr << "hipblaslt_init_hpl not supports FP6" << std::endl;
        break;
    case HIP_R_6F_E3M2_EXT:
        hipblaslt_cerr << "hipblaslt_init_hpl not supports BF6" << std::endl;
        break;
    case HIP_R_4F_E2M1_EXT:
        hipblaslt_cerr << "hipblaslt_init_hpl not supports FP4" << std::endl;
        break;
    default:
        hipblaslt_cerr << "Error type in hipblaslt_init_hpl" << std::endl;
        break;
    }
}

/* ============================================================================================ */
/*! \brief  Initialize an array with random data, with NaN where appropriate */

template <typename T>
inline void hipblaslt_init_nan(T* A, size_t N)
{
    for(size_t i = 0; i < N; ++i)
        A[i] = T(hipblaslt_nan_rng());
}

template <typename T>
inline void hipblaslt_init_nan(T* A, size_t start_offset, size_t end_offset)
{
    for(size_t i = start_offset; i < end_offset; ++i)
        A[i] = T(hipblaslt_nan_rng());
}

inline void hipblaslt_init_nan(void* A, size_t N, hipDataType type)
{
    switch(type)
    {
    case HIP_R_32F:
        hipblaslt_init_nan<float>(static_cast<float*>(A), N);
        break;
    case HIP_R_64F:
        hipblaslt_init_nan<double>(static_cast<double*>(A), N);
        break;
    case HIP_R_16F:
        hipblaslt_init_nan<hipblasLtHalf>(static_cast<hipblasLtHalf*>(A), N);
        break;
    case HIP_R_16BF:
        hipblaslt_init_nan<hip_bfloat16>(static_cast<hip_bfloat16*>(A), N);
        break;
#if HIP_FP8_TYPE_FNUZ
    case HIP_R_8F_E4M3_FNUZ:
        hipblaslt_init_nan<hipblaslt_f8_fnuz>(static_cast<hipblaslt_f8_fnuz*>(A), N);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        hipblaslt_init_nan<hipblaslt_bf8_fnuz>(static_cast<hipblaslt_bf8_fnuz*>(A), N);
        break;
#endif
#if HIP_FP8_TYPE_OCP
    case HIP_R_8F_E4M3:
        hipblaslt_init_nan<hipblaslt_f8>(static_cast<hipblaslt_f8*>(A), N);
        break;
    case HIP_R_8F_E5M2:
        hipblaslt_init_nan<hipblaslt_bf8>(static_cast<hipblaslt_bf8*>(A), N);
        break;
#endif
    case HIP_R_32I:
        hipblaslt_init_nan<int32_t>(static_cast<int32_t*>(A), N);
        break;
    case HIP_R_8I:
        hipblaslt_init_nan<hipblasLtInt8>(static_cast<hipblasLtInt8*>(A), N);
        break;
    case HIP_R_6F_E2M3_EXT:
        hipblaslt_cerr << "hipblaslt_init_nan not supports FP6" << std::endl;
        break;
    case HIP_R_6F_E3M2_EXT:
        hipblaslt_cerr << "hipblaslt_init_nan not supports BF6" << std::endl;
        break;
    case HIP_R_4F_E2M1_EXT:
        hipblaslt_cerr << "hipblaslt_init_nan not supports FP4" << std::endl;
        break;
    default:
        hipblaslt_cerr << "Error type in hipblaslt_init_nan" << std::endl;
        break;
    }
}

inline void hipblaslt_init_nan(void* A, size_t start_offset, size_t end_offset, hipDataType type)
{
    switch(type)
    {
    case HIP_R_32F:
        hipblaslt_init_nan<float>(static_cast<float*>(A), start_offset, end_offset);
        break;
    case HIP_R_64F:
        hipblaslt_init_nan<double>(static_cast<double*>(A), start_offset, end_offset);
        break;
    case HIP_R_16F:
        hipblaslt_init_nan<hipblasLtHalf>(static_cast<hipblasLtHalf*>(A), start_offset, end_offset);
        break;
    case HIP_R_16BF:
        hipblaslt_init_nan<hip_bfloat16>(static_cast<hip_bfloat16*>(A), start_offset, end_offset);
        break;
#if HIP_FP8_TYPE_FNUZ
    case HIP_R_8F_E4M3_FNUZ:
        hipblaslt_init_nan<hipblaslt_f8_fnuz>(
            static_cast<hipblaslt_f8_fnuz*>(A), start_offset, end_offset);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        hipblaslt_init_nan<hipblaslt_bf8_fnuz>(
            static_cast<hipblaslt_bf8_fnuz*>(A), start_offset, end_offset);
        break;
#endif
#if HIP_FP8_TYPE_OCP
    case HIP_R_8F_E4M3:
        hipblaslt_init_nan<hipblaslt_f8>(static_cast<hipblaslt_f8*>(A), start_offset, end_offset);
        break;
    case HIP_R_8F_E5M2:
        hipblaslt_init_nan<hipblaslt_bf8>(static_cast<hipblaslt_bf8*>(A), start_offset, end_offset);
        break;
#endif
    case HIP_R_32I:
        hipblaslt_init_nan<int32_t>(static_cast<int32_t*>(A), start_offset, end_offset);
        break;
    case HIP_R_8I:
        hipblaslt_init_nan<hipblasLtInt8>(static_cast<hipblasLtInt8*>(A), start_offset, end_offset);
        break;
    case HIP_R_6F_E2M3_EXT:
        hipblaslt_cerr << "hipblaslt_init_nan not supports FP6" << std::endl;
        break;
    case HIP_R_6F_E3M2_EXT:
        hipblaslt_cerr << "hipblaslt_init_nan not supports BF6" << std::endl;
        break;
    case HIP_R_4F_E2M1_EXT:
        hipblaslt_cerr << "hipblaslt_init_nan not supports FP4" << std::endl;
        break;
    default:
        hipblaslt_cerr << "Error type in hipblaslt_init_nan" << std::endl;
        break;
    }
}

template <typename T>
inline void hipblaslt_init_nan_tri(
    bool upper, T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
            {
                T val = upper ? (j >= i ? T(hipblaslt_nan_rng()) : static_cast<T>(0))
                              : (j <= i ? T(hipblaslt_nan_rng()) : static_cast<T>(0));
                A[i + j * lda + i_batch * stride] = val;
            }
}

inline void hipblaslt_init_nan_tri(bool        upper,
                                   void*       A,
                                   size_t      M,
                                   size_t      N,
                                   size_t      lda,
                                   hipDataType type,
                                   size_t      stride      = 0,
                                   size_t      batch_count = 1)
{
    switch(type)
    {
    case HIP_R_32F:
        hipblaslt_init_nan_tri(upper, static_cast<float*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_64F:
        hipblaslt_init_nan_tri(upper, static_cast<double*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16F:
        hipblaslt_init_nan_tri(
            upper, static_cast<hipblasLtHalf*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16BF:
        hipblaslt_init_nan_tri(
            upper, static_cast<hip_bfloat16*>(A), M, N, lda, stride, batch_count);
        break;
#if HIP_FP8_TYPE_FNUZ
    case HIP_R_8F_E4M3_FNUZ:
        hipblaslt_init_nan_tri(
            upper, static_cast<hipblaslt_f8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        hipblaslt_init_nan_tri(
            upper, static_cast<hipblaslt_bf8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
#endif
#if HIP_FP8_TYPE_OCP
    case HIP_R_8F_E4M3:
        hipblaslt_init_nan_tri(
            upper, static_cast<hipblaslt_f8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2:
        hipblaslt_init_nan_tri(
            upper, static_cast<hipblaslt_bf8*>(A), M, N, lda, stride, batch_count);
        break;
#endif
    case HIP_R_32I:
        hipblaslt_init_nan_tri(upper, static_cast<int32_t*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8I:
        hipblaslt_init_nan_tri(
            upper, static_cast<hipblasLtInt8*>(A), M, N, lda, stride, batch_count);
        break;
    default:
        hipblaslt_cerr << "Error type in hipblaslt_init_nan_tri" << std::endl;
        break;
    }
}

template <typename T>
inline void hipblaslt_init_nan(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(hipblaslt_nan_rng());
}

inline void hipblaslt_init_nan(void*       A,
                               size_t      M,
                               size_t      N,
                               size_t      lda,
                               hipDataType type,
                               size_t      stride      = 0,
                               size_t      batch_count = 1)
{
    switch(type)
    {
    case HIP_R_32F:
        hipblaslt_init_nan<float>(static_cast<float*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_64F:
        hipblaslt_init_nan<double>(static_cast<double*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16F:
        hipblaslt_init_nan<hipblasLtHalf>(
            static_cast<hipblasLtHalf*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16BF:
        hipblaslt_init_nan<hip_bfloat16>(
            static_cast<hip_bfloat16*>(A), M, N, lda, stride, batch_count);
        break;
#if HIP_FP8_TYPE_FNUZ
    case HIP_R_8F_E4M3_FNUZ:
        hipblaslt_init_nan<hipblaslt_f8_fnuz>(
            static_cast<hipblaslt_f8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        hipblaslt_init_nan<hipblaslt_bf8_fnuz>(
            static_cast<hipblaslt_bf8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
#endif
#if HIP_FP8_TYPE_OCP
    case HIP_R_8F_E4M3:
        hipblaslt_init_nan<hipblaslt_f8>(
            static_cast<hipblaslt_f8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2:
        hipblaslt_init_nan<hipblaslt_bf8>(
            static_cast<hipblaslt_bf8*>(A), M, N, lda, stride, batch_count);
        break;
#endif
    case HIP_R_32I:
        hipblaslt_init_nan<int32_t>(static_cast<int32_t*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8I:
        hipblaslt_init_nan<hipblasLtInt8>(
            static_cast<hipblasLtInt8*>(A), M, N, lda, stride, batch_count);
        break;
    default:
        hipblaslt_cerr << "Error type in hipblaslt_init_nan" << std::endl;
        break;
    }
}

/* ============================================================================================ */
/*! \brief  Initialize an array with random data, with Inf where appropriate */

template <typename T>
inline void hipblaslt_init_inf(T* A, size_t N)
{
    for(size_t i = 0; i < N; ++i)
        A[i] = static_cast<T>(hipblaslt_inf_rng());
}

template <typename T>
inline void hipblaslt_init_inf(T* A, size_t start_offset, size_t end_offset)
{
    for(size_t i = start_offset; i < end_offset; ++i)
        A[i] = static_cast<T>(hipblaslt_inf_rng());
}

inline void hipblaslt_init_inf(void* A, size_t N, hipDataType type)
{
    switch(type)
    {
    case HIP_R_32F:
        hipblaslt_init_inf<float>(static_cast<float*>(A), N);
        break;
    case HIP_R_64F:
        hipblaslt_init_inf<double>(static_cast<double*>(A), N);
        break;
    case HIP_R_16F:
        hipblaslt_init_inf<hipblasLtHalf>(static_cast<hipblasLtHalf*>(A), N);
        break;
    case HIP_R_16BF:
        hipblaslt_init_inf<hip_bfloat16>(static_cast<hip_bfloat16*>(A), N);
        break;
#if HIP_FP8_TYPE_FNUZ
    case HIP_R_8F_E4M3_FNUZ:
        hipblaslt_init_inf<hipblaslt_f8_fnuz>(static_cast<hipblaslt_f8_fnuz*>(A), N);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        hipblaslt_init_inf<hipblaslt_bf8_fnuz>(static_cast<hipblaslt_bf8_fnuz*>(A), N);
        break;
#endif
#if HIP_FP8_TYPE_OCP
    case HIP_R_8F_E4M3:
        hipblaslt_init_inf<hipblaslt_f8>(static_cast<hipblaslt_f8*>(A), N);
        break;
    case HIP_R_8F_E5M2:
        hipblaslt_init_inf<hipblaslt_bf8>(static_cast<hipblaslt_bf8*>(A), N);
        break;
#endif
    case HIP_R_32I:
        hipblaslt_init_inf<int32_t>(static_cast<int32_t*>(A), N);
        break;
    case HIP_R_8I:
        hipblaslt_init_inf<hipblasLtInt8>(static_cast<hipblasLtInt8*>(A), N);
        break;
    default:
        hipblaslt_cerr << "Error type in hipblaslt_init_inf" << std::endl;
        break;
    }
}

template <typename T>
inline void hipblaslt_init_inf(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(hipblaslt_inf_rng());
}

inline void hipblaslt_init_inf(void* A, size_t start_offset, size_t end_offset, hipDataType type)
{
    switch(type)
    {
    case HIP_R_32F:
        hipblaslt_init_inf<float>(static_cast<float*>(A), start_offset, end_offset);
        break;
    case HIP_R_64F:
        hipblaslt_init_inf<double>(static_cast<double*>(A), start_offset, end_offset);
        break;
    case HIP_R_16F:
        hipblaslt_init_inf<hipblasLtHalf>(static_cast<hipblasLtHalf*>(A), start_offset, end_offset);
        break;
    case HIP_R_16BF:
        hipblaslt_init_inf<hip_bfloat16>(static_cast<hip_bfloat16*>(A), start_offset, end_offset);
        break;
#if HIP_FP8_TYPE_FNUZ
    case HIP_R_8F_E4M3_FNUZ:
        hipblaslt_init_inf<hipblaslt_f8_fnuz>(
            static_cast<hipblaslt_f8_fnuz*>(A), start_offset, end_offset);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        hipblaslt_init_inf<hipblaslt_bf8_fnuz>(
            static_cast<hipblaslt_bf8_fnuz*>(A), start_offset, end_offset);
        break;
#endif
#if HIP_FP8_TYPE_OCP
    case HIP_R_8F_E4M3:
        hipblaslt_init_inf<hipblaslt_f8>(static_cast<hipblaslt_f8*>(A), start_offset, end_offset);
        break;
    case HIP_R_8F_E5M2:
        hipblaslt_init_inf<hipblaslt_bf8>(static_cast<hipblaslt_bf8*>(A), start_offset, end_offset);
        break;
#endif
    case HIP_R_32I:
        hipblaslt_init_inf<int32_t>(static_cast<int32_t*>(A), start_offset, end_offset);
        break;
    case HIP_R_8I:
        hipblaslt_init_inf<hipblasLtInt8>(static_cast<hipblasLtInt8*>(A), start_offset, end_offset);
        break;
    default:
        hipblaslt_cerr << "Error type in hipblaslt_init_inf" << std::endl;
        break;
    }
}

inline void hipblaslt_init_inf(void*       A,
                               size_t      M,
                               size_t      N,
                               size_t      lda,
                               hipDataType type,
                               size_t      stride      = 0,
                               size_t      batch_count = 1)
{
    switch(type)
    {
    case HIP_R_32F:
        hipblaslt_init_inf<float>(static_cast<float*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_64F:
        hipblaslt_init_inf<double>(static_cast<double*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16F:
        hipblaslt_init_inf<hipblasLtHalf>(
            static_cast<hipblasLtHalf*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16BF:
        hipblaslt_init_inf<hip_bfloat16>(
            static_cast<hip_bfloat16*>(A), M, N, lda, stride, batch_count);
        break;
#if HIP_FP8_TYPE_FNUZ
    case HIP_R_8F_E4M3_FNUZ:
        hipblaslt_init_inf<hipblaslt_f8_fnuz>(
            static_cast<hipblaslt_f8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        hipblaslt_init_inf<hipblaslt_bf8_fnuz>(
            static_cast<hipblaslt_bf8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
#endif
#if HIP_FP8_TYPE_OCP
    case HIP_R_8F_E4M3:
        hipblaslt_init_inf<hipblaslt_f8>(
            static_cast<hipblaslt_f8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2:
        hipblaslt_init_inf<hipblaslt_bf8>(
            static_cast<hipblaslt_bf8*>(A), M, N, lda, stride, batch_count);
        break;
#endif
    case HIP_R_32I:
        hipblaslt_init_inf<int32_t>(static_cast<int32_t*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8I:
        hipblaslt_init_inf<hipblasLtInt8>(
            static_cast<hipblasLtInt8*>(A), M, N, lda, stride, batch_count);
        break;
    default:
        hipblaslt_cerr << "Error type in hipblaslt_init_inf" << std::endl;
        break;
    }
}

/* ============================================================================================ */
/*! \brief  Initialize an array with random data, with zero */

template <typename T>
inline void hipblaslt_init_zero(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(0);
}

template <typename T>
inline void hipblaslt_init_zero(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(0);
}

template <typename T>
inline void hipblaslt_init_zero(T* A, size_t start_offset, size_t end_offset)
{
    for(size_t i = start_offset; i < end_offset; ++i)
        A[i] = T(hipblaslt_zero_rng());
}

inline void hipblaslt_init_zero(void*       A,
                                size_t      M,
                                size_t      N,
                                size_t      lda,
                                hipDataType type,
                                size_t      stride      = 0,
                                size_t      batch_count = 1)
{
    switch(type)
    {
    case HIP_R_32F:
        hipblaslt_init_zero<float>(static_cast<float*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_64F:
        hipblaslt_init_zero<double>(static_cast<double*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16F:
        hipblaslt_init_zero<hipblasLtHalf>(
            static_cast<hipblasLtHalf*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16BF:
        hipblaslt_init_zero<hip_bfloat16>(
            static_cast<hip_bfloat16*>(A), M, N, lda, stride, batch_count);
        break;
#if HIP_FP8_TYPE_FNUZ
    case HIP_R_8F_E4M3_FNUZ:
        hipblaslt_init_zero<hipblaslt_f8_fnuz>(
            static_cast<hipblaslt_f8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        hipblaslt_init_zero<hipblaslt_bf8_fnuz>(
            static_cast<hipblaslt_bf8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
#endif
#if HIP_FP8_TYPE_OCP
    case HIP_R_8F_E4M3:
        hipblaslt_init_zero<hipblaslt_f8>(
            static_cast<hipblaslt_f8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2:
        hipblaslt_init_zero<hipblaslt_bf8>(
            static_cast<hipblaslt_bf8*>(A), M, N, lda, stride, batch_count);
        break;
#endif
    case HIP_R_32I:
        hipblaslt_init_zero<int32_t>(static_cast<int32_t*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8I:
        hipblaslt_init_zero<hipblasLtInt8>(
            static_cast<hipblasLtInt8*>(A), M, N, lda, stride, batch_count);
        break;
    default:
        hipblaslt_cerr << "Error type in hipblaslt_init_zero" << std::endl;
        break;
    }
}

inline void hipblaslt_init_zero(void* A, size_t start_offset, size_t end_offset, hipDataType type)
{
    switch(type)
    {
    case HIP_R_32F:
        hipblaslt_init_zero<float>(static_cast<float*>(A), start_offset, end_offset);
        break;
    case HIP_R_64F:
        hipblaslt_init_zero<double>(static_cast<double*>(A), start_offset, end_offset);
        break;
    case HIP_R_16F:
        hipblaslt_init_zero<hipblasLtHalf>(
            static_cast<hipblasLtHalf*>(A), start_offset, end_offset);
        break;
    case HIP_R_16BF:
        hipblaslt_init_zero<hip_bfloat16>(static_cast<hip_bfloat16*>(A), start_offset, end_offset);
        break;
#if HIP_FP8_TYPE_FNUZ
    case HIP_R_8F_E4M3_FNUZ:
        hipblaslt_init_zero<hipblaslt_f8_fnuz>(
            static_cast<hipblaslt_f8_fnuz*>(A), start_offset, end_offset);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        hipblaslt_init_zero<hipblaslt_bf8_fnuz>(
            static_cast<hipblaslt_bf8_fnuz*>(A), start_offset, end_offset);
        break;
#endif
#if HIP_FP8_TYPE_OCP
    case HIP_R_8F_E4M3:
        hipblaslt_init_zero<hipblaslt_f8>(static_cast<hipblaslt_f8*>(A), start_offset, end_offset);
        break;
    case HIP_R_8F_E5M2:
        hipblaslt_init_zero<hipblaslt_bf8>(
            static_cast<hipblaslt_bf8*>(A), start_offset, end_offset);
        break;
#endif
    case HIP_R_32I:
        hipblaslt_init_zero<int32_t>(static_cast<int32_t*>(A), start_offset, end_offset);
        break;
    case HIP_R_8I:
        hipblaslt_init_zero<hipblasLtInt8>(
            static_cast<hipblasLtInt8*>(A), start_offset, end_offset);
        break;
    default:
        hipblaslt_cerr << "Error type in hipblaslt_init_zero" << std::endl;
        break;
    }
}

template <typename T>
inline void hipblaslt_init_alt_impl_big(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    const hipblasLtHalf ieee_half_max(65280.0);
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(ieee_half_max);
}

template <typename T>
inline void hipblaslt_init_alt_impl_big(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    const hipblasLtHalf ieee_half_max(65280.0);
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(ieee_half_max);
}

inline void hipblaslt_init_alt_impl_big(void*       A,
                                        size_t      M,
                                        size_t      N,
                                        size_t      lda,
                                        hipDataType type,
                                        size_t      stride      = 0,
                                        size_t      batch_count = 1)
{
    switch(type)
    {
    case HIP_R_32F:
        hipblaslt_init_alt_impl_big<float>(static_cast<float*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_64F:
        hipblaslt_init_alt_impl_big<double>(
            static_cast<double*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16F:
        hipblaslt_init_alt_impl_big<hipblasLtHalf>(
            static_cast<hipblasLtHalf*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16BF:
        hipblaslt_init_alt_impl_big<hip_bfloat16>(
            static_cast<hip_bfloat16*>(A), M, N, lda, stride, batch_count);
        break;
#if HIP_FP8_TYPE_FNUZ
    case HIP_R_8F_E4M3_FNUZ:
        hipblaslt_init_alt_impl_big<hipblaslt_f8_fnuz>(
            static_cast<hipblaslt_f8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        hipblaslt_init_alt_impl_big<hipblaslt_bf8_fnuz>(
            static_cast<hipblaslt_bf8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
#endif
#if HIP_FP8_TYPE_OCP
    case HIP_R_8F_E4M3:
        hipblaslt_init_alt_impl_big<hipblaslt_f8>(
            static_cast<hipblaslt_f8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2:
        hipblaslt_init_alt_impl_big<hipblaslt_bf8>(
            static_cast<hipblaslt_bf8*>(A), M, N, lda, stride, batch_count);
        break;
#endif
    case HIP_R_32I:
        hipblaslt_init_alt_impl_big<int32_t>(
            static_cast<int32_t*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8I:
        hipblaslt_init_alt_impl_big<hipblasLtInt8>(
            static_cast<hipblasLtInt8*>(A), M, N, lda, stride, batch_count);
        break;
    default:
        hipblaslt_cerr << "Error type in hipblaslt_init_alt_impl_big" << std::endl;
        break;
    }
}

template <typename T>
inline void hipblaslt_init_alt_impl_small(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    const hipblasLtHalf ieee_half_small(0.0000607967376708984375);
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(ieee_half_small);
}

template <typename T>
inline void hipblaslt_init_alt_impl_small(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    const hipblasLtHalf ieee_half_small(0.0000607967376708984375);
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(ieee_half_small);
}

inline void hipblaslt_init_alt_impl_small(void*       A,
                                          size_t      M,
                                          size_t      N,
                                          size_t      lda,
                                          hipDataType type,
                                          size_t      stride      = 0,
                                          size_t      batch_count = 1)
{
    switch(type)
    {
    case HIP_R_32F:
        hipblaslt_init_alt_impl_small<float>(
            static_cast<float*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_64F:
        hipblaslt_init_alt_impl_small<double>(
            static_cast<double*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16F:
        hipblaslt_init_alt_impl_small<hipblasLtHalf>(
            static_cast<hipblasLtHalf*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16BF:
        hipblaslt_init_alt_impl_small<hip_bfloat16>(
            static_cast<hip_bfloat16*>(A), M, N, lda, stride, batch_count);
        break;
#if HIP_FP8_TYPE_FNUZ
    case HIP_R_8F_E4M3_FNUZ:
        hipblaslt_init_alt_impl_small<hipblaslt_f8_fnuz>(
            static_cast<hipblaslt_f8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        hipblaslt_init_alt_impl_small<hipblaslt_bf8_fnuz>(
            static_cast<hipblaslt_bf8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
#endif
#if HIP_FP8_TYPE_OCP
    case HIP_R_8F_E4M3:
        hipblaslt_init_alt_impl_small<hipblaslt_f8>(
            static_cast<hipblaslt_f8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2:
        hipblaslt_init_alt_impl_small<hipblaslt_bf8>(
            static_cast<hipblaslt_bf8*>(A), M, N, lda, stride, batch_count);
        break;
#endif
    case HIP_R_32I:
        hipblaslt_init_alt_impl_small<int32_t>(
            static_cast<int32_t*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8I:
        hipblaslt_init_alt_impl_small<hipblasLtInt8>(
            static_cast<hipblasLtInt8*>(A), M, N, lda, stride, batch_count);
        break;
    default:
        hipblaslt_cerr << "Error type in hipblaslt_init_alt_impl_small" << std::endl;
        break;
    }
}

/* ============================================================================================ */
/*! \brief  matrix matrix initialization: copies from A into same position in B */
template <typename T>
void hipblaslt_copy_matrix(const T* A,
                           T*       B,
                           size_t   M,
                           size_t   N,
                           size_t   lda,
                           size_t   ldb,
                           size_t   stridea     = 0,
                           size_t   strideb     = 0,
                           size_t   batch_count = 1)
{

    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
    {
        size_t stride_offset_a = i_batch * stridea;
        size_t stride_offset_b = i_batch * strideb;
#pragma omp parallel for
        for(size_t j = 0; j < N; ++j)
        {
            size_t offset_a = stride_offset_a + j * lda;
            size_t offset_b = stride_offset_b + j * ldb;
            memcpy(B + offset_b, A + offset_a, M * sizeof(T));
        }
    }
}
