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

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "hipblas_init.hpp"
#include "utility.h"
#include <hip/hip_bfloat16.h>
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

/* ============================================================================================ */

template <typename T>
float bfloat16_to_float_helper(T bf)
{
    union
    {
        uint32_t int32;
        float    fp32;
    } u = {uint32_t(bf.data) << 16};
    return u.fp32;
}

// if just using host compiler with hip_bfloat16, hip gives minimum definition of bfloat16, so doing
// most of the work in fp32
void reference_bfdot(
    int n, hipblasBfloat16* hx, int incx, hipblasBfloat16* hy, int incy, hipblasBfloat16* res)
{
    float tmp = 0;
    for(int i = 0; i < n; i++)
        tmp += bfloat16_to_float(hx[i * size_t(incx)]) * bfloat16_to_float(hy[i * size_t(incy)]);
    *res = float_to_bfloat16(tmp);
}

int main()
{
    int             N    = 1024;
    int             incx = 1;
    int             incy = 1;
    hipblasStatus_t status;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    std::vector<hip_bfloat16> hx(N);
    std::vector<hip_bfloat16> hy(N);
    hip_bfloat16*             dx;
    hip_bfloat16*             dy;
    size_t                    x_size = N * size_t(incx);
    size_t                    y_size = N * size_t(incy);

    double gpu_time_used = 0.0;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dx, x_size * sizeof(hip_bfloat16)));
    CHECK_HIP_ERROR(hipMalloc(&dy, y_size * sizeof(hip_bfloat16)));

    // Initial Data on CPU
    // Initializes in range of [1, 3]
    // hipblasBfloat16 and hip_bfloat16 have same format, can be casted to use our helper functions
    srand(1);
    hipblas_init<hipblasBfloat16>((hipblasBfloat16*)hx.data(), 1, N, incx);
    hipblas_init<hipblasBfloat16>((hipblasBfloat16*)hy.data(), 1, N, incy);

    hip_bfloat16 hipblas_result, gold_result;

    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), x_size * sizeof(hip_bfloat16), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), y_size * sizeof(hip_bfloat16), hipMemcpyHostToDevice));

    printf("N        hipblas(us)     \n");

    gpu_time_used = get_time_us(); // in microseconds

    /* =====================================================================
         HIPBLAS C interface
    =================================================================== */

    status = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        CHECK_HIP_ERROR(hipFree(dx));
        CHECK_HIP_ERROR(hipFree(dy));
        hipblasDestroy(handle);
        return status;
    }

#ifdef HIPBLAS_USE_HIP_BFLOAT16
    status = hipblasBfdot(handle, N, dx, incx, dy, incy, &hipblas_result);
#else
    // without exposing hip_bfloat16 interface, we can cast to hipblasBfloat16 pointers
    status = hipblasBfdot(handle,
                          N,
                          (hipblasBfloat16*)dx,
                          incx,
                          (hipblasBfloat16*)dy,
                          incy,
                          (hipblasBfloat16*)(&hipblas_result));
#endif

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        CHECK_HIP_ERROR(hipFree(dx));
        CHECK_HIP_ERROR(hipFree(dy));
        hipblasDestroy(handle);
        return status;
    }

    gpu_time_used = get_time_us() - gpu_time_used;

    // verify hipblas_bfdot result
    reference_bfdot(N,
                    (hipblasBfloat16*)hx.data(),
                    incx,
                    (hipblasBfloat16*)hy.data(),
                    incy,
                    (hipblasBfloat16*)(&gold_result));
    float gold_resf = bfloat16_to_float_helper((gold_result));
    float hip_resf  = bfloat16_to_float_helper((hipblas_result));
    float diff      = std::abs(gold_resf - hip_resf);

    printf("%d    %8.2f        \n", (int)N, gpu_time_used);

    if(diff)
    {
        printf("CPU RESULT: %f, GPU_RESULT: %f\n", gold_resf, hip_resf);
        printf("BFDOT TEST FAILS\n");
    }
    else
    {
        printf("BFDOT TEST PASSES\n");
    }

    CHECK_HIP_ERROR(hipFree(dx));
    CHECK_HIP_ERROR(hipFree(dy));
    hipblasDestroy(handle);
    return EXIT_SUCCESS;
}
