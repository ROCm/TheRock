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

#include "allclose.hpp"
#include "cblas.h"
#include "hipblaslt_ostream.hpp"
#include "hipblaslt_vector.hpp"
#include "utility.hpp"
#include <cstdio>
#include <hipblaslt/hipblaslt.h>
#include <limits>
#include <memory>

/* =====================================================================
        allclose check: abs(A-B) <= atol + abs(rtol * B)
    =================================================================== */

/*!\file
 * \brief compares two results (usually, CPU and GPU results); provides allclose check
 */

/* ============== Allclose Check for General Matrix ============= */
/*! \brief compare the allclose error of two matrices hCPU & hGPU */

template <typename T>
bool allclose(size_t* N, T* a, T* b, double atol, double rtol, bool equal_nan = false)
{
    for(size_t i = 0; i < *N; i++)
    {
        //Returning ture immediately if 2 elements are identical.
        //This also can handle inf == inf case
        if(a[i] == b[i])
            continue;
        //NaNs compare equal when equal_nan is true
        if(equal_nan && (std::isnan(a[i]) && std::isnan(b[i])))
            continue;
        if(equal_nan && (std::isnan(a[i]) ^ std::isnan(b[i])))
            return false;

        double error     = std::abs(a[i] - b[i]);
        double tolerance = atol + std::abs(rtol * b[i]);
        if(!(error <= tolerance))
            return false;
    }
    return true;
}

template <
    typename T,
    std::enable_if_t<!(std::is_same<T, hipblaslt_f8_fnuz>{} || std::is_same<T, hipblaslt_bf8_fnuz>{}
                       || std::is_same<T, hipblaslt_f8>{} || std::is_same<T, hipblaslt_bf8>{}
                       ),
                     int> = 0>
bool allclose_check_general(char    allclose_type,
                            int64_t M,
                            int64_t N,
                            int64_t lda,
                            T*      hCPU,
                            T*      hGPU,
                            double& hipblaslt_atol,
                            double& hipblaslt_rtol)
{
    if(M * N == 0)
        return 0;
    size_t              size = N * (size_t)lda;
    host_vector<double> hCPU_double(size);
    host_vector<double> hGPU_double(size);

    for(int64_t i = 0; i < N; i++)
    {
        for(int64_t j = 0; j < M; j++)
        {
            size_t idx       = j + i * (size_t)lda;
            hCPU_double[idx] = double(hCPU[idx]);
            hGPU_double[idx] = double(hGPU[idx]);
        }
    }

    std::vector<double> atols{1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1};
    std::vector<double> rtols{1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1};
    for(auto& atol : atols)
    {
        for(auto& rtol : rtols)
        {
            if(allclose(&size, hCPU_double.data(), hGPU_double.data(), atol, rtol, false))
            {
                hipblaslt_atol = atol;
                hipblaslt_rtol = rtol;
                //early termination for accending rtols
                break;
            }
        }
        //early termination for accending atols
        if(hipblaslt_atol != 1)
            break;
    }

    if(hipblaslt_atol == 1)
    {
        return false;
    }

    return true;
}

template <typename T,
          std::enable_if_t<(std::is_same<T, hipblaslt_f8_fnuz>{}
                            || std::is_same<T, hipblaslt_bf8_fnuz>{}),
                           int> = 0>
bool allclose_check_general(char    allclose_type,
                            int64_t M,
                            int64_t N,
                            int64_t lda,
                            T*      hCPU,
                            T*      hGPU,
                            double& hipblaslt_atol,
                            double& hipblaslt_rtol)
{
    if(M * N == 0)
        return 0;
    size_t              size = N * (size_t)lda;
    host_vector<double> hCPU_double(size);
    host_vector<double> hGPU_double(size);

    for(int64_t i = 0; i < N; i++)
    {
        for(int64_t j = 0; j < M; j++)
        {
            size_t idx       = j + i * (size_t)lda;
            hCPU_double[idx] = double(float(hCPU[idx]));
            hGPU_double[idx] = double(float(hGPU[idx]));
        }
    }

    std::vector<double> atols{1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1};
    std::vector<double> rtols{1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1};
    for(auto& atol : atols)
    {
        for(auto& rtol : rtols)
        {
            if(allclose(&size, hCPU_double.data(), hGPU_double.data(), atol, rtol, false))
            {
                hipblaslt_atol = atol;
                hipblaslt_rtol = rtol;
                //early termination for accending rtols
                break;
            }
        }
        //early termination for accending atols
        if(hipblaslt_atol != 1)
            break;
    }

    if(hipblaslt_atol == 1)
    {
        return false;
    }

    return true;
}

template <typename T,
          std::enable_if_t<(std::is_same<T, hipblaslt_f8>{} || std::is_same<T, hipblaslt_bf8>{}),
                           int> = 0>
bool allclose_check_general(char    allclose_type,
                            int64_t M,
                            int64_t N,
                            int64_t lda,
                            T*      hCPU,
                            T*      hGPU,
                            double& hipblaslt_atol,
                            double& hipblaslt_rtol)
{
    if(M * N == 0)
        return 0;
    size_t              size = N * (size_t)lda;
    host_vector<double> hCPU_double(size);
    host_vector<double> hGPU_double(size);

    for(int64_t i = 0; i < N; i++)
    {
        for(int64_t j = 0; j < M; j++)
        {
            size_t idx       = j + i * (size_t)lda;
            hCPU_double[idx] = double(float(hCPU[idx]));
            hGPU_double[idx] = double(float(hGPU[idx]));
        }
    }

    std::vector<double> atols{1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1};
    std::vector<double> rtols{1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1};
    for(auto& atol : atols)
    {
        for(auto& rtol : rtols)
        {
            if(allclose(&size, hCPU_double.data(), hGPU_double.data(), atol, rtol, false))
            {
                hipblaslt_atol = atol;
                hipblaslt_rtol = rtol;
                //early termination for accending rtols
                break;
            }
        }
        //early termination for accending atols
        if(hipblaslt_atol != 1)
            break;
    }

    if(hipblaslt_atol == 1)
    {
        return false;
    }

    return true;
}
// For BF16 and half, we convert the results to double first
template <
    typename T,
    typename VEC,
    std::enable_if_t<std::is_same<T, hipblasLtHalf>{} || std::is_same<T, hip_bfloat16>{}, int> = 0>
bool allclose_check_general(char    allclose_type,
                            int64_t M,
                            int64_t N,
                            int64_t lda,
                            VEC&&   hCPU,
                            T*      hGPU,
                            double& hipblaslt_atol,
                            double& hipblaslt_rtol)
{
    if(M * N == 0)
        return 0;
    size_t              size = N * (size_t)lda;
    host_vector<double> hCPU_double(size);
    host_vector<double> hGPU_double(size);

    for(int64_t i = 0; i < N; i++)
    {
        for(int64_t j = 0; j < M; j++)
        {
            size_t idx       = j + i * (size_t)lda;
            hCPU_double[idx] = hCPU[idx];
            hGPU_double[idx] = hGPU[idx];
        }
    }

    return allclose_check_general<double>(
        allclose_type, M, N, lda, hCPU_double, hGPU_double, hipblaslt_atol, hipblaslt_rtol);
}

// For int8, we convert the results to int first
template <typename T, typename VEC, std::enable_if_t<std::is_same<T, int8_t>{}, int> = 0>
bool allclose_check_general(char    allclose_type,
                            int64_t M,
                            int64_t N,
                            int64_t lda,
                            VEC&&   hCPU,
                            T*      hGPU,
                            double& hipblaslt_atol,
                            double& hipblaslt_rtol)
{
    if(M * N == 0)
        return 0;
    size_t           size = N * (size_t)lda;
    host_vector<int> hCPU_int(size);
    host_vector<int> hGPU_int(size);

    for(int64_t i = 0; i < N; i++)
    {
        for(int64_t j = 0; j < M; j++)
        {
            size_t idx    = j + i * (size_t)lda;
            hCPU_int[idx] = hCPU[idx];
            hGPU_int[idx] = hGPU[idx];
        }
    }

    return allclose_check_general<int>(
        allclose_type, M, N, lda, hCPU_int, hGPU_int, hipblaslt_atol, hipblaslt_rtol);
}

/* ============== allclose check for strided_batched case ============= */
template <typename T, typename T_hpa>
bool allclose_check_general(char    allclose_type,
                            int64_t M,
                            int64_t N,
                            int64_t lda,
                            int64_t stride_a,
                            T_hpa*  hCPU,
                            T*      hGPU,
                            int64_t batch_count,
                            double& hipblaslt_atol,
                            double& hipblaslt_rtol)
{
    if(M * N == 0)
        return 0;

    for(size_t i = 0; i < batch_count; i++)
    {
        auto index = i * stride_a;
        bool close = allclose_check_general(
            allclose_type, M, N, lda, hCPU + index, hGPU + index, hipblaslt_atol, hipblaslt_rtol);
        if(!close)
            return false;
    }

    return true;
}

/* ============== allclose check for batched case ============= */
template <typename T>
bool allclose_check_general(char    allclose_type,
                            int64_t M,
                            int64_t N,
                            int64_t lda,
                            T*      hCPU[],
                            T*      hGPU[],
                            int64_t batch_count,
                            double& hipblaslt_atol,
                            double& hipblaslt_rtol)
{
    if(M * N == 0)
        return 0;

    for(int64_t i = 0; i < batch_count; i++)
    {
        auto index = i;
        bool close = allclose_check_general<T>(
            allclose_type, M, N, lda, hCPU[index], hGPU[index], hipblaslt_atol, hipblaslt_rtol);
        if(!close)
            return false;
    }

    return true;
}

bool allclose_check_general(char        allclose_type,
                            int64_t     M,
                            int64_t     N,
                            int64_t     lda,
                            int64_t     stride_a,
                            void*       hCPU,
                            void*       hGPU,
                            int64_t     batch_count,
                            double&     hipblaslt_atol,
                            double&     hipblaslt_rtol,
                            hipDataType type)
{
    switch(type)
    {
    case HIP_R_32F:
        return allclose_check_general<float>(allclose_type,
                                             M,
                                             N,
                                             lda,
                                             stride_a,
                                             static_cast<float*>(hCPU),
                                             static_cast<float*>(hGPU),
                                             batch_count,
                                             hipblaslt_atol,
                                             hipblaslt_rtol);
    case HIP_R_64F:
        return allclose_check_general<double>(allclose_type,
                                              M,
                                              N,
                                              lda,
                                              stride_a,
                                              static_cast<double*>(hCPU),
                                              static_cast<double*>(hGPU),
                                              batch_count,
                                              hipblaslt_atol,
                                              hipblaslt_rtol);
    case HIP_R_16F:
        return allclose_check_general<hipblasLtHalf>(allclose_type,
                                                     M,
                                                     N,
                                                     lda,
                                                     stride_a,
                                                     static_cast<hipblasLtHalf*>(hCPU),
                                                     static_cast<hipblasLtHalf*>(hGPU),
                                                     batch_count,
                                                     hipblaslt_atol,
                                                     hipblaslt_rtol);
    case HIP_R_16BF:
        return allclose_check_general<hip_bfloat16>(allclose_type,
                                                    M,
                                                    N,
                                                    lda,
                                                    stride_a,
                                                    static_cast<hip_bfloat16*>(hCPU),
                                                    static_cast<hip_bfloat16*>(hGPU),
                                                    batch_count,
                                                    hipblaslt_atol,
                                                    hipblaslt_rtol);
    case HIP_R_8F_E4M3_FNUZ:
        return allclose_check_general<hipblaslt_f8_fnuz>(allclose_type,
                                                         M,
                                                         N,
                                                         lda,
                                                         stride_a,
                                                         static_cast<hipblaslt_f8_fnuz*>(hCPU),
                                                         static_cast<hipblaslt_f8_fnuz*>(hGPU),
                                                         batch_count,
                                                         hipblaslt_atol,
                                                         hipblaslt_rtol);
    case HIP_R_8F_E5M2_FNUZ:
        return allclose_check_general<hipblaslt_bf8_fnuz>(allclose_type,
                                                          M,
                                                          N,
                                                          lda,
                                                          stride_a,
                                                          static_cast<hipblaslt_bf8_fnuz*>(hCPU),
                                                          static_cast<hipblaslt_bf8_fnuz*>(hGPU),
                                                          batch_count,
                                                          hipblaslt_atol,
                                                          hipblaslt_rtol);
    case HIP_R_8F_E4M3:
        return allclose_check_general<hipblaslt_f8>(allclose_type,
                                                    M,
                                                    N,
                                                    lda,
                                                    stride_a,
                                                    static_cast<hipblaslt_f8*>(hCPU),
                                                    static_cast<hipblaslt_f8*>(hGPU),
                                                    batch_count,
                                                    hipblaslt_atol,
                                                    hipblaslt_rtol);
    case HIP_R_8F_E5M2:
        return allclose_check_general<hipblaslt_bf8>(allclose_type,
                                                     M,
                                                     N,
                                                     lda,
                                                     stride_a,
                                                     static_cast<hipblaslt_bf8*>(hCPU),
                                                     static_cast<hipblaslt_bf8*>(hGPU),
                                                     batch_count,
                                                     hipblaslt_atol,
                                                     hipblaslt_rtol);
    case HIP_R_32I:
        return allclose_check_general<int32_t>(allclose_type,
                                               M,
                                               N,
                                               lda,
                                               stride_a,
                                               static_cast<int32_t*>(hCPU),
                                               static_cast<int32_t*>(hGPU),
                                               batch_count,
                                               hipblaslt_atol,
                                               hipblaslt_rtol);
    case HIP_R_8I:
        return allclose_check_general<hipblasLtInt8>(allclose_type,
                                                     M,
                                                     N,
                                                     lda,
                                                     stride_a,
                                                     static_cast<hipblasLtInt8*>(hCPU),
                                                     static_cast<hipblasLtInt8*>(hGPU),
                                                     batch_count,
                                                     hipblaslt_atol,
                                                     hipblaslt_rtol);
    default:
        hipblaslt_cerr << "Error type in allclose_check_general" << std::endl;
        return false;
    }
}
