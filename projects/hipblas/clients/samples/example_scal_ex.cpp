/* ************************************************************************
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip/library_types.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "hipblas.h"
#include "hipblas_init.hpp"
#include "utility.h"

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

int main()
{
    // Testing scalEx with alpha_type == x_type == f16_r; execution_type = f32_r
    int             N = 10240;
    hipblasStatus_t status;
    hipblasHalf     alpha = float_to_half(10.0f);

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    std::vector<hipblasHalf> hx(N);
    hipblasHalf*             dx = nullptr;

    double gpu_time_used = 0.0;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dx, N * sizeof(hipblasHalf)));

    // Initial Data on CPU
    srand(1);
    hipblas_init<hipblasHalf>(hx.data(), 1, N, 1);

    // copy vector is easy in STL; hz(hx): save a copy in hz which will be output of CPU BLAS
    std::vector<hipblasHalf> hz(hx);

    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(hipblasHalf) * N, hipMemcpyHostToDevice));

    printf("N        hipblas(us)     \n");

    gpu_time_used = get_time_us(); // in microseconds

    /* =====================================================================
         ROCBLAS  C interface
    =================================================================== */
    status = hipblasScalEx(handle, N, &alpha, HIP_R_16F, dx, HIP_R_16F, 1, HIP_R_32F);

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        printf("Bad return status from hipblasScalEx: %d\n", status);
        CHECK_HIP_ERROR(hipFree(dx));
        hipblasDestroy(handle);
        return status;
    }

    gpu_time_used = get_time_us() - gpu_time_used;

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hx.data(), dx, sizeof(hipblasHalf) * N, hipMemcpyDeviceToHost));

    // verify hipblas_scal result
    bool error_in_element = false;
    for(int i = 0; i < N; i++)
    {
        hipblasHalf cpu_res = float_to_half(half_to_float(hz[i]) * half_to_float(alpha));
        if(cpu_res != hx[i])
        {
            printf("error in element %d: CPU=%f, GPU=%f ",
                   i,
                   half_to_float(cpu_res),
                   half_to_float(hx[i]));
            error_in_element = true;
            break;
        }
    }

    printf("%d    %8.2f        \n", (int)N, gpu_time_used);

    if(error_in_element)
    {
        printf("SCALEX TEST FAILS\n");
    }
    else
    {
        printf("SCALEX TEST PASSES\n");
    }

    CHECK_HIP_ERROR(hipFree(dx));
    hipblasDestroy(handle);
    return EXIT_SUCCESS;
}
