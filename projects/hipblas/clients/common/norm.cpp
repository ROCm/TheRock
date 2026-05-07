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
 * ************************************************************************ */

#include "norm.h"
#include "cblas.h"
#include "hipblas.h"
#include "lapack_utilities.hpp"

#include <stdio.h>

/* =====================================================================
     README: Norm check: norm(A-B)/norm(A), evaluate relative error
             Numerically, it is recommended by lapack.

    Call lapack fortran routines that do not exsit in cblas library.
    No special header is required. But need to declare
    function prototype

    All the functions are fortran and should append underscore (_) while declaring prototype and
   calling.
    xlange and xaxpy prototype are like following
    =================================================================== */

#ifdef __cplusplus
extern "C" {
#endif

// float  slange_(char* norm_type, int* m, int* n, float* A, int* lda, float* work);
// double dlange_(char* norm_type, int* m, int* n, double* A, int* lda, double* work);
// float  clange_(char* norm_type, int* m, int* n, std::complex<float>* A, int* lda, float* work);
// double zlange_(char* norm_type, int* m, int* n, std::complex<double>* A, int* lda, double* work);

// float  slansy_(char* norm_type, char* uplo, int* n, float* A, int* lda, float* work);
// double dlansy_(char* norm_type, char* uplo, int* n, double* A, int* lda, double* work);
//  float  clanhe_(char* norm_type, char* uplo, int* n, std::complex<float>* A, int* lda, float* work);
//  double zlanhe_(char* norm_type, char* uplo, int* n, std::complex<double>* A, int* lda, double*
//  work);

// void m_axpy_64(int* n, float* alpha, float* x, int* incx, float* y, int* incy);
// void m_axpy_64(int* n, double* alpha, double* x, int* incx, double* y, int* incy);
// void m_axpy_64(
//     int* n, std::complex<float>* alpha, std::complex<float>* x, int* incx, std::complex<float>* y, int* incy);
// void m_axpy_64(int*                  n,
//             std::complex<double>* alpha,
//             std::complex<double>* x,
//             int*                  incx,
//             std::complex<double>* y,
//             int*                  incy);

#ifdef __cplusplus
}
#endif

template <typename T>
void m_axpy_64(int64_t N, T* alpha, T* x, int64_t incx, T* y, int64_t incy)
{
    int64_t x_offset = incx >= 0 ? 0 : incx * (1 - N);
    int64_t y_offset = incy >= 0 ? 0 : incy * (1 - N);
    for(int64_t i = 0; i < N; i++)
    {
        y[y_offset + i * incy] = (*alpha) * x[x_offset + i * incx] + y[y_offset + i * incy];
    }
}

/* ============================Norm Check for General Matrix: float/double/complex template
 * speciliazation ======================================= */

/*! \brief compare the norm error of two matrices hCPU & hGPU */
template <>
double norm_check_general<float>(
    char norm_type, int64_t M, int64_t N, int64_t lda, float* hCPU, float* hGPU)
{
    // norm type can be M', 'I', 'F', 'l': 'F' (Frobenius norm) is used mostly

    host_vector<double> work(std::max(int64_t(1), M));
    int64_t             incx  = 1;
    float               alpha = -1.0f;
    int64_t             size  = lda * N;

    double cpu_norm = lapack_xlange(norm_type, M, N, hCPU, lda, work.data());
    m_axpy_64(size, &alpha, hCPU, incx, hGPU, incx);
    double error = lapack_xlange(norm_type, M, N, hGPU, lda, work.data()) / cpu_norm;

    return error;
}

template <>
double norm_check_general<double>(
    char norm_type, int64_t M, int64_t N, int64_t lda, double* hCPU, double* hGPU)
{
    // norm type can be M', 'I', 'F', 'l': 'F' (Frobenius norm) is used mostly

    host_vector<double> work(std::max(int64_t(1), M));
    int64_t             incx  = 1;
    double              alpha = -1.0;
    int64_t             size  = lda * N;

    double cpu_norm = lapack_xlange(norm_type, M, N, hCPU, lda, work.data());
    m_axpy_64(size, &alpha, hCPU, incx, hGPU, incx);
    double error = lapack_xlange(norm_type, M, N, hGPU, lda, work.data()) / cpu_norm;

    return error;
}

template <>
double norm_check_general<std::complex<float>>(char                 norm_type,
                                               int64_t              M,
                                               int64_t              N,
                                               int64_t              lda,
                                               std::complex<float>* hCPU,
                                               std::complex<float>* hGPU)
{
    //norm type can be M', 'I', 'F', 'l': 'F' (Frobenius norm) is used mostly

    host_vector<double> work(std::max(int64_t(1), M));
    int64_t             incx  = 1;
    std::complex<float> alpha = -1.0f;
    int64_t             size  = lda * N;

    double cpu_norm = lapack_xlange(norm_type, M, N, hCPU, lda, work.data());
    m_axpy_64(size, &alpha, hCPU, incx, hGPU, incx);
    double error = lapack_xlange(norm_type, M, N, hGPU, lda, work.data()) / cpu_norm;

    return error;
}

template <>
double norm_check_general<std::complex<double>>(char                  norm_type,
                                                int64_t               M,
                                                int64_t               N,
                                                int64_t               lda,
                                                std::complex<double>* hCPU,
                                                std::complex<double>* hGPU)
{
    //norm type can be M', 'I', 'F', 'l': 'F' (Frobenius norm) is used mostly

    host_vector<double>  work(std::max(int64_t(1), M));
    int64_t              incx  = 1;
    std::complex<double> alpha = -1.0;
    int64_t              size  = lda * N;

    double cpu_norm = lapack_xlange(norm_type, M, N, hCPU, lda, work.data());
    m_axpy_64(size, &alpha, hCPU, incx, hGPU, incx);
    double error = lapack_xlange(norm_type, M, N, hGPU, lda, work.data()) / cpu_norm;

    return error;
}

template <>
double norm_check_general<hipblasHalf>(
    char norm_type, int64_t M, int64_t N, int64_t lda, hipblasHalf* hCPU, hipblasHalf* hGPU)
{
    // norm type can be 'M', 'I', 'F', 'l': 'F' (Frobenius norm) is used mostly

    host_vector<double> hCPU_double(N * lda);
    host_vector<double> hGPU_double(N * lda);

    for(int64_t i = 0; i < M; i++)
    {
        for(int64_t j = 0; j < N; j++)
        {
            hCPU_double[i + j * lda] = hCPU[i + j * lda];
            hGPU_double[i + j * lda] = hGPU[i + j * lda];
        }
    }

    return norm_check_general<double>(norm_type, M, N, lda, hCPU_double, hGPU_double);
}

template <>
double norm_check_general<hipblasBfloat16>(
    char norm_type, int64_t M, int64_t N, int64_t lda, hipblasBfloat16* hCPU, hipblasBfloat16* hGPU)
{
    // norm type can be 'M', 'I', 'F', 'l': 'F' (Frobenius norm) is used mostly

    host_vector<float> hCPU_double(N * lda);
    host_vector<float> hGPU_double(N * lda);

    for(int64_t i = 0; i < M; i++)
    {
        for(int64_t j = 0; j < N; j++)
        {
            hCPU_double[i + j * lda] = bfloat16_to_float(hCPU[i + j * lda]);
            hGPU_double[i + j * lda] = bfloat16_to_float(hGPU[i + j * lda]);
        }
    }

    return norm_check_general<float>(norm_type, M, N, lda, hCPU_double, hGPU_double);
}

template <>
double norm_check_general<int32_t>(
    char norm_type, int64_t M, int64_t N, int64_t lda, int32_t* hCPU, int32_t* hGPU)
{
    // norm type can be 'M', 'I', 'F', 'l': 'F' (Frobenius norm) is used mostly

    host_vector<float> hCPU_float(N * lda);
    host_vector<float> hGPU_float(N * lda);

    for(int64_t i = 0; i < M; i++)
    {
        for(int64_t j = 0; j < N; j++)
        {
            hCPU_float[i + j * lda] = (hCPU[i + j * lda]);
            hGPU_float[i + j * lda] = (hGPU[i + j * lda]);
        }
    }

    return norm_check_general<float>(norm_type, M, N, lda, hCPU_float, hGPU_float);
}

/* ============================Norm Check for Symmetric Matrix: float/double/complex template
 * speciliazation ======================================= */

/*! \brief compare the norm error of two hermitian/symmetric matrices hCPU & hGPU */

template <typename T, std::enable_if_t<!is_complex<T>, int> = 0>
double norm_check_symmetric(char norm_type, char uplo, int64_t N, int64_t lda, T* hCPU, T* hGPU)
{
    // norm type can be M', 'I', 'F', 'l': 'F' (Frobenius norm) is used mostly

    host_vector<double> work(std::max(int64_t(1), N));
    int64_t             incx  = 1;
    double              alpha = -1.0;
    size_t              size  = N * (size_t)lda;

    host_vector<double> hCPU_double(size);
    host_vector<double> hGPU_double(size);

    for(int64_t i = 0; i < N; i++)
    {
        for(int64_t j = 0; j < N; j++)
        {
            size_t idx       = j + i * (size_t)lda;
            hCPU_double[idx] = double(hCPU[idx]);
            hGPU_double[idx] = double(hGPU[idx]);
        }
    }
    constexpr bool HERM = false;
    double cpu_norm = lapack_xlansy<HERM>(norm_type, uplo, N, hCPU_double.data(), lda, work.data());
    m_axpy_64(size, &alpha, hCPU_double.data(), incx, hGPU_double.data(), incx);
    double error
        = lapack_xlansy<HERM>(norm_type, uplo, N, hGPU_double.data(), lda, work.data()) / cpu_norm;

    return error;
}

template <typename T, std::enable_if_t<is_complex<T>, int> = 0>
double norm_check_symmetric(char norm_type, char uplo, int64_t N, int64_t lda, T* hCPU, T* hGPU)
{
    // norm type can be M', 'I', 'F', 'l': 'F' (Frobenius norm) is used mostly
    host_vector<double> work(std::max(int64_t(1), N));
    int64_t             incx  = 1;
    T                   alpha = -1.0;
    size_t              size  = (size_t)lda * N;

    constexpr bool HERM     = true;
    double         cpu_norm = lapack_xlansy<HERM>(norm_type, uplo, N, hCPU, lda, work.data());
    m_axpy_64(size, &alpha, hCPU, incx, hGPU, incx);
    double error = lapack_xlansy<HERM>(norm_type, uplo, N, hGPU, lda, work.data()) / cpu_norm;

    return error;
}

/*
template <>
double norm_check_symmetric<float>(
    char norm_type, char uplo, int64_t N, int64_t lda, float* hCPU, float* hGPU)
{
    // norm type can be M', 'I', 'F', 'l': 'F' (Frobenius norm) is used mostly

    float   work[1];
    int64_t incx  = 1;
    float   alpha = -1.0f;
    int64_t size  = lda * N;

    float cpu_norm = slansy_(&norm_type, &uplo, &N, hCPU, &lda, work);
    m_axpy_64(size, &alpha, hCPU, incx, hGPU, incx);

    float error = slansy_(&norm_type, &uplo, &N, hGPU, &lda, work) / cpu_norm;

    return (double)error;
}

template <>
double norm_check_symmetric<double>(
    char norm_type, char uplo, int64_t N, int64_t lda, double* hCPU, double* hGPU)
{
    // norm type can be M', 'I', 'F', 'l': 'F' (Frobenius norm) is used mostly

    double  work[1];
    int64_t incx  = 1;
    double  alpha = -1.0;
    int64_t size  = lda * N;

    double cpu_norm = dlansy_(&norm_type, &uplo, &N, hCPU, &lda, work);
    m_axpy_64(size, &alpha, hCPU, incx, hGPU, incx);

    double error = dlansy_(&norm_type, &uplo, &N, hGPU, &lda, work) / cpu_norm;

    return error;
}

*/

// template<>
// double norm_check_symmetric<std::complex<float>>(char norm_type, char uplo, int64_t N, int64_t lda, std::complex<float>
// *hCPU, std::complex<float> *hGPU)
//{
////norm type can be M', 'I', 'F', 'l': 'F' (Frobenius norm) is used mostly
//
//    float work[1];
//    int64_t incx = 1;
//    std::complex<float> alpha = -1.0f;
//    int64_t size = lda * N;
//
//    float cpu_norm = clanhe_(&norm_type, &uplo, &N, hCPU, &lda, work);
//    m_axpy_64(size, &alpha, hCPU, incx, hGPU, incx);
//
//     float error = clanhe_(&norm_type, &uplo, &N, hGPU, &lda, work)/cpu_norm;
//
//    return (double)error;
//}
//
// template<>
// double norm_check_symmetric<std::complex<double>>(char norm_type, char uplo, int64_t N, int64_t lda,
// std::complex<double> *hCPU, std::complex<double> *hGPU)
//{
////norm type can be M', 'I', 'F', 'l': 'F' (Frobenius norm) is used mostly
//
//    double work[1];
//    int64_t incx = 1;
//    std::complex<double> alpha = -1.0;
//    int64_t size = lda * N;
//
//    double cpu_norm = zlanhe_(&norm_type, &uplo, &N, hCPU, &lda, work);
//    m_axpy_64(size, &alpha, hCPU, incx, hGPU, incx);
//
//     double error = zlanhe_(&norm_type, &uplo, &N, hGPU, &lda, work)/cpu_norm;
//
//    return error;
//}
