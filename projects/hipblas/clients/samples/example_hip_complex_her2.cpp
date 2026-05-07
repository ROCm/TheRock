/* ************************************************************************
 * Copyright (C) 2021-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <assert.h>
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <hipblas/hipblas.h>

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

int main(int argc, char** argv)
{
    int N = 267;
    if(argc > 1)
        N = atoi(argv[1]);

    size_t lda;
    size_t rows, cols;
    int    incx, incy;

    rows = N;
    cols = N;
    lda  = N;
    incx = incy = 1;

    size_t sizeA = size_t(cols) * lda;
    size_t sizeX = size_t(N) * incx;
    size_t sizeY = size_t(N) * incy;

    hipblasHandle_t handle;
    hipblasStatus_t rstatus = hipblasCreate(&handle);
    CHECK_HIPBLAS_ERROR(rstatus);

    std::vector<hipFloatComplex> hA(sizeA);
    std::vector<hipFloatComplex> hResult(sizeA);
    std::vector<hipFloatComplex> hX(sizeX);
    std::vector<hipFloatComplex> hY(sizeY);

    for(int i1 = 0; i1 < N; i1++)
    {
        hX[i1 * incx] = make_hipFloatComplex(1.0f, 0.0f);
        hY[i1 * incy] = make_hipFloatComplex(1.0f, 0.0f);
    }

    for(int i1 = 0; i1 < rows; i1++)
        for(int i2 = 0; i2 < cols; i2++)
            hA[i1 + i2 * lda] = make_hipFloatComplex((float)(rand() % 10), 0.0f);

    hipFloatComplex* dA = nullptr;
    hipFloatComplex* dX = nullptr;
    hipFloatComplex* dY = nullptr;
    CHECK_HIP_ERROR(hipMalloc((void**)&dA, sizeof(hipFloatComplex) * sizeA));
    CHECK_HIP_ERROR(hipMalloc((void**)&dX, sizeof(hipFloatComplex) * sizeX));
    CHECK_HIP_ERROR(hipMalloc((void**)&dY, sizeof(hipFloatComplex) * sizeY));

    // scalar arguments will be from host memory
    rstatus = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);
    CHECK_HIPBLAS_ERROR(rstatus);

    const hipblasFillMode_t uplo   = HIPBLAS_FILL_MODE_UPPER;
    hipFloatComplex         hAlpha = make_hipFloatComplex(2.0f, 0.0f);

    // copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dX, hX.data(), sizeof(hipFloatComplex) * sizeX, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dY, hY.data(), sizeof(hipFloatComplex) * sizeY, hipMemcpyHostToDevice));

    rstatus = hipblasSetMatrix(rows, cols, sizeof(hipFloatComplex), hA.data(), lda, dA, lda);
    CHECK_HIPBLAS_ERROR(rstatus);

    // asynchronous calculation on device, returns before finished calculations
    // API is defined as using hipFloatComplex types so no casting required
    rstatus = hipblasCher2(handle, uplo, N, &hAlpha, dX, incx, dY, incy, dA, lda);
    CHECK_HIPBLAS_ERROR(rstatus);

    // fetch results
    rstatus = hipblasGetMatrix(rows, cols, sizeof(hipFloatComplex), dA, lda, hResult.data(), lda);
    CHECK_HIPBLAS_ERROR(rstatus);

    // check against expected results for upper and numeric inputs
    bool fail = false;
    for(size_t i1 = 0; i1 < rows; i1++)
        for(size_t i2 = 0; i2 < cols; i2++)
        {
            hipFloatComplex tmp
                = hipCaddf(hA[i1 + i2 * lda], make_hipFloatComplex(4.0 * hX[i1 * incx].x, 0.0f));
            if(i1 <= i2 && (hResult[i1 + i2 * lda].x != tmp.x || hResult[i1 + i2 * lda].y != tmp.y))
                fail = true;
        }
    CHECK_HIP_ERROR(hipFree(dA));
    CHECK_HIP_ERROR(hipFree(dX));
    CHECK_HIP_ERROR(hipFree(dY));

    rstatus = hipblasDestroy(handle);
    CHECK_HIPBLAS_ERROR(rstatus);

    fprintf(stdout, "%s\n", fail ? "FAIL" : "PASS");

    return 0;
}
