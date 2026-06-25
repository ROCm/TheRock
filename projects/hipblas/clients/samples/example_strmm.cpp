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

#include <hipblas/hipblas.h>
#include <iostream>
#include <limits>
#include <math.h> // isnan
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(error)                    \
    if(error != hipSuccess)                       \
    {                                             \
        fprintf(stderr,                           \
                "Hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error),         \
                error,                            \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif

#ifndef CHECK_HIPBLAS_ERROR
#define CHECK_HIPBLAS_ERROR(error)                              \
    if(error != HIPBLAS_STATUS_SUCCESS)                         \
    {                                                           \
        fprintf(stderr, "hipBLAS error: ");                     \
        if(error == HIPBLAS_STATUS_NOT_INITIALIZED)             \
            fprintf(stderr, "HIPBLAS_STATUS_NOT_INITIALIZED");  \
        if(error == HIPBLAS_STATUS_ALLOC_FAILED)                \
            fprintf(stderr, "HIPBLAS_STATUS_ALLOC_FAILED");     \
        if(error == HIPBLAS_STATUS_INVALID_VALUE)               \
            fprintf(stderr, "HIPBLAS_STATUS_INVALID_VALUE");    \
        if(error == HIPBLAS_STATUS_MAPPING_ERROR)               \
            fprintf(stderr, "HIPBLAS_STATUS_MAPPING_ERROR");    \
        if(error == HIPBLAS_STATUS_EXECUTION_FAILED)            \
            fprintf(stderr, "HIPBLAS_STATUS_EXECUTION_FAILED"); \
        if(error == HIPBLAS_STATUS_INTERNAL_ERROR)              \
            fprintf(stderr, "HIPBLAS_STATUS_INTERNAL_ERROR");   \
        if(error == HIPBLAS_STATUS_NOT_SUPPORTED)               \
            fprintf(stderr, "HIPBLAS_STATUS_NOT_SUPPORTED");    \
        if(error == HIPBLAS_STATUS_INVALID_ENUM)                \
            fprintf(stderr, "HIPBLAS_STATUS_INVALID_ENUM");     \
        if(error == HIPBLAS_STATUS_UNKNOWN)                     \
            fprintf(stderr, "HIPBLAS_STATUS_UNKNOWN");          \
        fprintf(stderr, "\n");                                  \
        exit(EXIT_FAILURE);                                     \
    }
#endif

#define DIM1 4
#define DIM2 4

// reference code for trmm (triangle matrix matrix multiplication)
template <typename T>
void trmm_reference(hipblasSideMode_t  side,
                    hipblasFillMode_t  uplo,
                    hipblasOperation_t trans,
                    hipblasDiagType_t  diag,
                    int                M,
                    int                N,
                    T                  alpha,
                    const T*           A,
                    int                lda,
                    const T*           B,
                    int                ldb,
                    T*                 C,
                    int                ldc)
{
    int As1 = HIPBLAS_OP_N == trans ? 1 : lda;
    int As2 = HIPBLAS_OP_N == trans ? lda : 1;

    // this is 3 loop gemm algorithm with non-relevant triangle part masked
    if(HIPBLAS_SIDE_LEFT == side)
    {
        for(int i1 = 0; i1 < M; i1++)
        {
            for(int i2 = 0; i2 < N; i2++)
            {
                T t = 0.0;
                for(int i3 = 0; i3 < M; i3++)
                {
                    if((i1 == i3) && (HIPBLAS_DIAG_UNIT == diag))
                    {
                        t += B[i3 + i2 * ldb];
                    }
                    else if(((i3 > i1) && (HIPBLAS_FILL_MODE_UPPER == uplo))
                            || ((i1 > i3) && (HIPBLAS_FILL_MODE_LOWER == uplo))
                            || ((i1 == i3) && (HIPBLAS_DIAG_NON_UNIT == diag)))
                    {
                        t += A[i1 * As1 + i3 * As2] * B[i3 + i2 * ldb];
                    }
                }
                C[i1 + i2 * ldc] = alpha * t;
            }
        }
    }
    else if(HIPBLAS_SIDE_RIGHT == side)
    {
        for(int i1 = 0; i1 < M; i1++)
        {
            for(int i2 = 0; i2 < N; i2++)
            {
                T t = 0.0;
                for(int i3 = 0; i3 < N; i3++)
                {
                    if((i3 == i2) && (HIPBLAS_DIAG_UNIT == diag))
                    {
                        t += B[i1 + i3 * ldb];
                    }
                    else if(((i2 > i3) && (HIPBLAS_FILL_MODE_UPPER == uplo))
                            || ((i3 > i2) && (HIPBLAS_FILL_MODE_LOWER == uplo))
                            || ((i3 == i2) && (HIPBLAS_DIAG_NON_UNIT == diag)))
                    {
                        t += B[i1 + i3 * ldb] * A[i3 * As1 + i2 * As2];
                    }
                }
                C[i1 + i2 * ldc] = alpha * t;
            }
        }
    }
}

int main()
{
    hipblasSideMode_t  side   = HIPBLAS_SIDE_LEFT;
    hipblasFillMode_t  uplo   = HIPBLAS_FILL_MODE_UPPER;
    hipblasOperation_t transa = HIPBLAS_OP_N;
    hipblasDiagType_t  diag   = HIPBLAS_DIAG_NON_UNIT;
    float              alpha  = 1.0;

    int m = DIM1, n = DIM2;
    int lda, ldb, ldc, size_a, size_b, size_c;
    std::cout << "strmm V3 example" << std::endl;

    if(HIPBLAS_SIDE_LEFT == side)
    {
        lda    = m;
        size_a = m * lda;
        std::cout << "left";
    }
    else if(HIPBLAS_SIDE_RIGHT == side)
    {
        lda    = n;
        size_a = n * lda;
        std::cout << "right";
    }
    HIPBLAS_FILL_MODE_UPPER == uplo ? std::cout << ",upper" : std::cout << ",lower";
    HIPBLAS_OP_N == transa ? std::cout << ",N" : std::cout << ",T";
    HIPBLAS_DIAG_NON_UNIT == diag ? std::cout << ",non_unit_diag:" : std::cout << ",unit_diag:";

    ldb    = m;
    size_b = n * ldb;

    ldc    = m;
    size_c = n * ldc;

    // Naming: da is in GPU (device) memory. ha is in CPU (host) memory
    std::vector<float> ha(size_a);
    std::vector<float> hb(size_b);
    std::vector<float> hc(size_c);
    std::vector<float> hc_gold(size_c);

    // initial data on host
    srand(1);
    for(int i = 0; i < size_a; ++i)
    {
        //      ha[i] = 1.0;
        ha[i] = rand() % 17;
    }
    for(int i = 0; i < size_b; ++i)
    {
        //      hb[i] = 1.0;
        hb[i] = rand() % 17;
    }
    for(int i = 0; i < size_c; ++i)
    {
        //      hc[i] = 1.0;
        hc[i] = rand() % 17;
    }
    hc_gold = hc;

    // allocate memory on device
    float *da, *db, *dc;
    CHECK_HIP_ERROR(hipMalloc(&da, size_a * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&db, size_b * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&dc, size_c * sizeof(float)));

    // copy matrices from host to device
    CHECK_HIP_ERROR(hipMemcpy(da, ha.data(), sizeof(float) * size_a, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(db, hb.data(), sizeof(float) * size_b, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dc, hc.data(), sizeof(float) * size_c, hipMemcpyHostToDevice));

    hipblasHandle_t handle;
    CHECK_HIPBLAS_ERROR(hipblasCreate(&handle));

    CHECK_HIPBLAS_ERROR(
        hipblasStrmm(handle, side, uplo, transa, diag, m, n, &alpha, da, lda, db, ldb, dc, ldc));

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hc.data(), dc, sizeof(float) * size_c, hipMemcpyDeviceToHost));

    std::cout << "m, n, lda, ldb, ldc = " << m << ", " << n << ", " << lda << ", " << ldb << ", "
              << ldc << std::endl;

    // calculate golden or correct result
    trmm_reference<float>(
        side, uplo, transa, diag, m, n, alpha, ha.data(), lda, hb.data(), ldb, hc_gold.data(), ldc);

    float max_relative_error = 0;
    for(int i = 0; i < size_c; i++)
    {
        float relative_error
            = hc_gold[i] == 0 ? (hc_gold[i] - hc[i]) : (hc_gold[i] - hc[i]) / hc_gold[i];
        relative_error = relative_error > 0 ? relative_error : -relative_error;
        max_relative_error
            = relative_error < max_relative_error ? max_relative_error : relative_error;
    }
    float eps       = std::numeric_limits<float>::epsilon();
    float tolerance = 10;
    if(isnan(max_relative_error) || max_relative_error > eps * tolerance)
    {
        std::cout << "FAIL: max_relative_error = " << max_relative_error << std::endl;
    }
    else
    {
        std::cout << "PASS: max_relative_error = " << max_relative_error << std::endl;
    }

    CHECK_HIP_ERROR(hipFree(da));
    CHECK_HIP_ERROR(hipFree(db));
    CHECK_HIP_ERROR(hipFree(dc));
    CHECK_HIPBLAS_ERROR(hipblasDestroy(handle));
    return EXIT_SUCCESS;
}
