/* ************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "utility.h"
#include <hipblas/hipblas.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================================ */

int main()
{
    int             N      = 10240;
    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;
    float           alpha  = 10.0;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    float* hx = (float*)calloc(N, sizeof(*hx));
    float* hz = (float*)calloc(N, sizeof(*hz));
    float* dx;

    double gpu_time_used = 0.0;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // allocate memory on device
    hipMalloc((void**)&dx, N * sizeof(float));

    // Initial Data on CPU
    srand(1);

    for(int i = 0; i < N; ++i)
        hx[i] = rand();

    memcpy(hz, hx, sizeof(*hz) * N);

    hipMemcpy(dx, hx, sizeof(*dx) * N, hipMemcpyHostToDevice);

    printf("N        hipblas(us)     \n");

    gpu_time_used = get_time_us(); // in microseconds

    status = hipblasSscal(handle, N, &alpha, dx, 1);
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        hipFree(dx);
        hipblasDestroy(handle);
        free(hx);
        free(hz);
        return status;
    }

    gpu_time_used = get_time_us() - gpu_time_used;

    // copy output from device to CPU
    hipMemcpy(hx, dx, sizeof(*hx) * N, hipMemcpyDeviceToHost);

    // verify hipblas_scal result
    bool error_in_element = false;
    for(int i = 0; i < N; i++)
    {
        if(hz[i] * alpha != hx[i])
        {
            printf("error in element %d: CPU=%f, GPU=%f ", i, hz[i] * alpha, hx[i]);
            error_in_element = true;
            break;
        }
    }

    printf("%d    %8.2f        \n", (int)N, gpu_time_used);

    if(error_in_element)
        printf("SSCAL TEST FAILS\n");
    else
        printf("SSCAL TEST PASSES\n");

    hipFree(dx);
    hipblasDestroy(handle);
    free(hx);
    free(hz);
    return EXIT_SUCCESS;
}
