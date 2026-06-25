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

#include "cblas.h"
#include "hipblaslt_ostream.hpp"
#include "hipblaslt_vector.hpp"
#include "norm.hpp"
#include "utility.hpp"
#include <cstdio>
#include <hipblaslt/hipblaslt.h>
#include <limits>
#include <memory>

/* =====================================================================
        Norm check: norm(A-B)/norm(A), evaluate relative error
    =================================================================== */

/*!\file
 * \brief compares two results (usually, CPU and GPU results); provides Norm check
 */

/* ========================================Norm Check
 * ==================================================== */

/* LAPACK fortran library functionality */
extern "C" {
float  slange_(char* norm_type, int* m, int* n, float* A, int* lda, float* work);
double dlange_(char* norm_type, int* m, int* n, double* A, int* lda, double* work);

float  slansy_(char* norm_type, char* uplo, int* n, float* A, int* lda, float* work);
double dlansy_(char* norm_type, char* uplo, int* n, double* A, int* lda, double* work);

void saxpy_(int* n, float* alpha, float* x, int* incx, float* y, int* incy);
void daxpy_(int* n, double* alpha, double* x, int* incx, double* y, int* incy);
}

/*! \brief  Overloading: norm check for general Matrix: half/float/doubel/complex */
inline float xlange(char* norm_type, int* m, int* n, float* A, int* lda, float* work)
{
    return slange_(norm_type, m, n, A, lda, work);
}

inline double xlange(char* norm_type, int* m, int* n, double* A, int* lda, double* work)
{
    return dlange_(norm_type, m, n, A, lda, work);
}

inline float xlanhe(char* norm_type, char* uplo, int* n, float* A, int* lda, float* work)
{
    return slansy_(norm_type, uplo, n, A, lda, work);
}

inline double xlanhe(char* norm_type, char* uplo, int* n, double* A, int* lda, double* work)
{
    return dlansy_(norm_type, uplo, n, A, lda, work);
}

inline void xaxpy(int* n, float* alpha, float* x, int* incx, float* y, int* incy)
{
    return saxpy_(n, alpha, x, incx, y, incy);
}

inline void xaxpy(int* n, double* alpha, double* x, int* incx, double* y, int* incy)
{
    return daxpy_(n, alpha, x, incx, y, incy);
}

template <typename T>
void m_axpy(size_t* N, T* alpha, T* x, int* incx, T* y, int* incy)
{
    for(size_t i = 0; i < *N; i++)
    {
        y[i * (*incy)] = (*alpha) * x[i * (*incx)] + y[i * (*incy)];
    }
}

/* ============== Norm Check for General Matrix ============= */
/*! \brief compare the norm error of two matrices hCPU & hGPU */

// Real
template <
    typename T,
    std::enable_if_t<!(std::is_same<T, hipblaslt_f8_fnuz>{} || std::is_same<T, hipblaslt_bf8_fnuz>{}
                       || std::is_same<T, hipblaslt_f8>{} || std::is_same<T, hipblaslt_bf8>{}
                       ),
                     int>
    = 0>
double norm_check_general(char norm_type, int64_t M, int64_t N, int64_t lda, T* hCPU, T* hGPU)
{
    if(M * N == 0)
        return 0;
    // norm type can be 'O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries
    size_t size = N * (size_t)lda;

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

    double work[1];
    int    incx  = 1;
    double alpha = -1.0;
    int    m     = static_cast<int>(M);
    int    n     = static_cast<int>(N);
    int    l     = static_cast<int>(lda);

    double cpu_norm = xlange(&norm_type, &m, &n, hCPU_double.data(), &l, work);
    m_axpy(&size, &alpha, hCPU_double.data(), &incx, hGPU_double.data(), &incx);
    double error = xlange(&norm_type, &m, &n, hGPU_double.data(), &l, work) / cpu_norm;
    return error;
}

template <typename T,
          std::enable_if_t<(std::is_same<T, hipblaslt_f8_fnuz>{}
                            || std::is_same<T, hipblaslt_bf8_fnuz>{}),
                           int>
          = 0>
double norm_check_general(char norm_type, int64_t M, int64_t N, int64_t lda, T* hCPU, T* hGPU)
{
    if(M * N == 0)
        return 0;
    size_t size = N * (size_t)lda;

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

    double work[1];
    int    incx  = 1;
    double alpha = -1.0;
    int    m     = static_cast<int>(M);
    int    n     = static_cast<int>(N);
    int    l     = static_cast<int>(lda);

    double cpu_norm = xlange(&norm_type, &m, &n, hCPU_double.data(), &l, work);
    m_axpy(&size, &alpha, hCPU_double.data(), &incx, hGPU_double.data(), &incx);
    double error = xlange(&norm_type, &m, &n, hGPU_double.data(), &l, work) / cpu_norm;

    return error;
}

template <
    typename T,
    std::enable_if_t<(std::is_same<T, hipblaslt_f8>{} || std::is_same<T, hipblaslt_bf8>{}), int>
    = 0>
double norm_check_general(char norm_type, int64_t M, int64_t N, int64_t lda, T* hCPU, T* hGPU)
{
    if(M * N == 0)
        return 0;
    size_t size = N * (size_t)lda;

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

    double work[1];
    int    incx  = 1;
    double alpha = -1.0;
    int    m     = static_cast<int>(M);
    int    n     = static_cast<int>(N);
    int    l     = static_cast<int>(lda);

    double cpu_norm = xlange(&norm_type, &m, &n, hCPU_double.data(), &l, work);
    m_axpy(&size, &alpha, hCPU_double.data(), &incx, hGPU_double.data(), &incx);
    double error = xlange(&norm_type, &m, &n, hGPU_double.data(), &l, work) / cpu_norm;

    return error;
}

/* ============== Norm Check for strided_batched case ============= */
template <typename T, template <typename> class VEC, typename T_hpa>
double norm_check_general(char        norm_type,
                          int64_t     M,
                          int64_t     N,
                          int64_t     lda,
                          int64_t     stride_a,
                          VEC<T_hpa>& hCPU,
                          T*          hGPU,
                          int64_t     batch_count)
{
    if(M * N == 0)
        return 0;
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

        auto error = norm_check_general(norm_type, M, N, lda, (T_hpa*)hCPU + index, hGPU + index);

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
template <typename T, typename T_hpa>
double norm_check_general(char    norm_type,
                          int64_t M,
                          int64_t N,
                          int64_t lda,
                          int64_t stride_a,
                          T_hpa*  hCPU,
                          T*      hGPU,
                          int64_t batch_count)
{
    if(M * N == 0)
        return 0;
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

/* ============== Norm Check for batched case ============= */
template <typename T>
double norm_check_general(
    char norm_type, int64_t M, int64_t N, int64_t lda, T* hCPU[], T* hGPU[], int64_t batch_count)
{
    if(M * N == 0)
        return 0;
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

double norm_check_general(
    char norm_type, int64_t M, int64_t N, int64_t lda, void* hCPU, void* hGPU, hipDataType type)
{
    switch(type)
    {
    case HIP_R_32F:
        return norm_check_general<float>(
            norm_type, M, N, lda, static_cast<float*>(hCPU), static_cast<float*>(hGPU));
    case HIP_R_64F:
        return norm_check_general<double>(
            norm_type, M, N, lda, static_cast<double*>(hCPU), static_cast<double*>(hGPU));
    case HIP_R_16F:
        return norm_check_general<hipblasLtHalf>(norm_type,
                                                 M,
                                                 N,
                                                 lda,
                                                 static_cast<hipblasLtHalf*>(hCPU),
                                                 static_cast<hipblasLtHalf*>(hGPU));
    case HIP_R_16BF:
        return norm_check_general<hip_bfloat16>(norm_type,
                                                M,
                                                N,
                                                lda,
                                                static_cast<hip_bfloat16*>(hCPU),
                                                static_cast<hip_bfloat16*>(hGPU));
    case HIP_R_8F_E4M3_FNUZ:
        return norm_check_general<hipblaslt_f8_fnuz>(norm_type,
                                                     M,
                                                     N,
                                                     lda,
                                                     static_cast<hipblaslt_f8_fnuz*>(hCPU),
                                                     static_cast<hipblaslt_f8_fnuz*>(hGPU));
    case HIP_R_8F_E5M2_FNUZ:
        return norm_check_general<hipblaslt_bf8_fnuz>(norm_type,
                                                      M,
                                                      N,
                                                      lda,
                                                      static_cast<hipblaslt_bf8_fnuz*>(hCPU),
                                                      static_cast<hipblaslt_bf8_fnuz*>(hGPU));
    case HIP_R_8F_E4M3:
        return norm_check_general<hipblaslt_f8>(norm_type,
                                                M,
                                                N,
                                                lda,
                                                static_cast<hipblaslt_f8*>(hCPU),
                                                static_cast<hipblaslt_f8*>(hGPU));
    case HIP_R_8F_E5M2:
        return norm_check_general<hipblaslt_bf8>(norm_type,
                                                 M,
                                                 N,
                                                 lda,
                                                 static_cast<hipblaslt_bf8*>(hCPU),
                                                 static_cast<hipblaslt_bf8*>(hGPU));
    case HIP_R_32I:
        return norm_check_general<int32_t>(
            norm_type, M, N, lda, static_cast<int32_t*>(hCPU), static_cast<int32_t*>(hGPU));
    case HIP_R_8I:
        return norm_check_general<hipblasLtInt8>(norm_type,
                                                 M,
                                                 N,
                                                 lda,
                                                 static_cast<hipblasLtInt8*>(hCPU),
                                                 static_cast<hipblasLtInt8*>(hGPU));
    default:
        hipblaslt_cerr << "Error type in norm_check_general" << std::endl;
        return 0;
    }
}

double norm_check_general(char        norm_type,
                          int64_t     M,
                          int64_t     N,
                          int64_t     lda,
                          int64_t     stride_a,
                          void*       hCPU,
                          void*       hGPU,
                          int64_t     batch_count,
                          hipDataType type)
{
    switch(type)
    {
    case HIP_R_32F:
        return norm_check_general<float>(norm_type,
                                         M,
                                         N,
                                         lda,
                                         stride_a,
                                         static_cast<float*>(hCPU),
                                         static_cast<float*>(hGPU),
                                         batch_count);
    case HIP_R_64F:
        return norm_check_general<double>(norm_type,
                                          M,
                                          N,
                                          lda,
                                          stride_a,
                                          static_cast<double*>(hCPU),
                                          static_cast<double*>(hGPU),
                                          batch_count);
    case HIP_R_16F:
        return norm_check_general<hipblasLtHalf>(norm_type,
                                                 M,
                                                 N,
                                                 lda,
                                                 stride_a,
                                                 static_cast<hipblasLtHalf*>(hCPU),
                                                 static_cast<hipblasLtHalf*>(hGPU),
                                                 batch_count);
    case HIP_R_16BF:
        return norm_check_general<hip_bfloat16>(norm_type,
                                                M,
                                                N,
                                                lda,
                                                stride_a,
                                                static_cast<hip_bfloat16*>(hCPU),
                                                static_cast<hip_bfloat16*>(hGPU),
                                                batch_count);
    case HIP_R_8F_E4M3_FNUZ:
        return norm_check_general<hipblaslt_f8_fnuz>(norm_type,
                                                     M,
                                                     N,
                                                     lda,
                                                     stride_a,
                                                     static_cast<hipblaslt_f8_fnuz*>(hCPU),
                                                     static_cast<hipblaslt_f8_fnuz*>(hGPU),
                                                     batch_count);
    case HIP_R_8F_E5M2_FNUZ:
        return norm_check_general<hipblaslt_bf8_fnuz>(norm_type,
                                                      M,
                                                      N,
                                                      lda,
                                                      stride_a,
                                                      static_cast<hipblaslt_bf8_fnuz*>(hCPU),
                                                      static_cast<hipblaslt_bf8_fnuz*>(hGPU),
                                                      batch_count);
    case HIP_R_8F_E4M3:
        return norm_check_general<hipblaslt_f8>(norm_type,
                                                M,
                                                N,
                                                lda,
                                                stride_a,
                                                static_cast<hipblaslt_f8*>(hCPU),
                                                static_cast<hipblaslt_f8*>(hGPU),
                                                batch_count);
    case HIP_R_8F_E5M2:
        return norm_check_general<hipblaslt_bf8>(norm_type,
                                                 M,
                                                 N,
                                                 lda,
                                                 stride_a,
                                                 static_cast<hipblaslt_bf8*>(hCPU),
                                                 static_cast<hipblaslt_bf8*>(hGPU),
                                                 batch_count);
    case HIP_R_32I:
        return norm_check_general<int32_t>(norm_type,
                                           M,
                                           N,
                                           lda,
                                           stride_a,
                                           static_cast<int32_t*>(hCPU),
                                           static_cast<int32_t*>(hGPU),
                                           batch_count);
    case HIP_R_8I:
        return norm_check_general<hipblasLtInt8>(norm_type,
                                                 M,
                                                 N,
                                                 lda,
                                                 stride_a,
                                                 static_cast<hipblasLtInt8*>(hCPU),
                                                 static_cast<hipblasLtInt8*>(hGPU),
                                                 batch_count);
    default:
        hipblaslt_cerr << "Error type in norm_check_general" << std::endl;
        return 0;
    }
}

template <typename T>
bool norm_check(double norm_error)
{
    if(std::is_same<T, float>{})
        return norm_error < 0.00001;
    if(std::is_same<T, double>{})
        return norm_error < 0.000000000001;
    if(std::is_same<T, hipblasLtHalf>{})
        return norm_error < 0.01;
    if(std::is_same<T, hip_bfloat16>{})
        return norm_error < 0.1;
    if(std::is_same<T, hipblasLtInt8>{})
        return norm_error < 0.01;
    if(std::is_same<T, hipblasLtInt32>{})
        return norm_error < 0.0001;
    if(std::is_same<T, hipblaslt_f8_fnuz>{})
        return norm_error < 0.125;
    if(std::is_same<T, hipblaslt_bf8_fnuz>{})
        return norm_error < 0.25;
    if(std::is_same<T, hipblaslt_f8>{})
        return norm_error < 0.125;
    if(std::is_same<T, hipblaslt_bf8>{})
        return norm_error < 0.25;
    return false;
}

// TODO: norm_check determines the required norm solely based on
//       To (type). This might cause tests to fail when the input
//       matrices are MX types (F4/F8/F6). A better way is
//       needed to determine the required norm for MX types.
bool norm_check(double norm_error, hipDataType type)
{
    switch(type)
    {
    case HIP_R_32F:
        return norm_error < 0.00001;
    case HIP_R_64F:
        return norm_error < 0.000000000001;
    case HIP_R_16F:
        return norm_error < 0.01;
    case HIP_R_16BF:
        return norm_error < 0.1;
    case HIP_R_8F_E4M3_FNUZ:
    case HIP_R_8F_E4M3:
        return norm_error < 0.125;
    case HIP_R_8F_E5M2_FNUZ:
    case HIP_R_8F_E5M2:
        return norm_error < 0.25;
    case HIP_R_32I:
        return norm_error < 0.0001;
    case HIP_R_8I:
        return norm_error < 0.01;
    // TODO: find a suitable rnom value for f6 and f4
    case HIP_R_6F_E2M3_EXT:
    case HIP_R_6F_E3M2_EXT:
    case HIP_R_4F_E2M1_EXT:
        return norm_error < 0.5;
    default:
        return false;
    }
}

bool norm_check(double norm_error, hipDataType type, hipblasComputeType_t compute_type)
{
    if(compute_type == HIPBLAS_COMPUTE_32F_FAST_16BF && type == HIP_R_32F)
        return norm_error < 0.5;
    return norm_check(norm_error, type);
}
