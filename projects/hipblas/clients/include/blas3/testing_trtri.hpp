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

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

/* ============================================================================================ */

using hipblasTrtriModel = ArgumentModel<e_a_type, e_uplo, e_diag, e_N, e_lda>;

inline void testname_trtri(const Arguments& arg, std::string& name)
{
    hipblasTrtriModel{}.test_name(arg, name);
}

template <typename T>
void testing_trtri_bad_arg(const Arguments& arg)
{
    bool FORTRAN        = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasTrtriFn = FORTRAN ? hipblasTrtri<T, true> : hipblasTrtri<T, false>;

    hipblasLocalHandle handle(arg);

    int64_t           N    = 100;
    int64_t           lda  = 102;
    hipblasFillMode_t uplo = HIPBLAS_FILL_MODE_LOWER;
    hipblasDiagType_t diag = HIPBLAS_DIAG_NON_UNIT;

    // Allocate device memory
    device_matrix<T> dA(N, N, lda);
    device_matrix<T> dinvA(N, N, lda);

    EXPECT_HIPBLAS_STATUS(hipblasTrtriFn(nullptr, uplo, diag, N, dA, lda, dinvA, lda),
                          HIPBLAS_STATUS_NOT_INITIALIZED);

    EXPECT_HIPBLAS_STATUS(
        hipblasTrtriFn(handle, HIPBLAS_FILL_MODE_FULL, diag, N, dA, lda, dinvA, lda),
        HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_HIPBLAS_STATUS(
        hipblasTrtriFn(handle, (hipblasFillMode_t)HIPBLAS_OP_N, diag, N, dA, lda, dinvA, lda),
        HIPBLAS_STATUS_INVALID_ENUM);
    EXPECT_HIPBLAS_STATUS(
        hipblasTrtriFn(handle, uplo, (hipblasDiagType_t)HIPBLAS_OP_N, N, dA, lda, dinvA, lda),
        HIPBLAS_STATUS_INVALID_ENUM);

    if(arg.bad_arg_all)
    {
        EXPECT_HIPBLAS_STATUS(hipblasTrtriFn(handle, uplo, diag, N, nullptr, lda, dinvA, lda),
                              HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_HIPBLAS_STATUS(hipblasTrtriFn(handle, uplo, diag, N, nullptr, lda, nullptr, lda),
                              HIPBLAS_STATUS_INVALID_VALUE);
    }

    // If N == 0, can have nullptrs
    CHECK_HIPBLAS_ERROR(hipblasTrtriFn(handle, uplo, diag, 0, nullptr, lda, nullptr, lda));
}

template <typename T>
void testing_trtri(const Arguments& arg)
{
    bool FORTRAN        = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasTrtriFn = FORTRAN ? hipblasTrtri<T, true> : hipblasTrtri<T, false>;

    const double rel_error = get_epsilon<T>() * 1000;

    hipblasFillMode_t uplo = char2hipblas_fill(arg.uplo);
    hipblasDiagType_t diag = char2hipblas_diagonal(arg.diag);
    int               N    = arg.N;
    int               lda  = arg.lda;

    int ldinvA = lda;

    // check here to prevent undefined memory allocation error
    if(N < 0 || lda < 0 || lda < N)
    {
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_matrix<T> hA(N, N, lda);
    host_matrix<T> hB(N, N, lda);

    // Allocate device memory
    device_matrix<T> dA(N, N, lda);
    device_matrix<T> dinvA(N, N, lda);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dinvA.memcheck());

    double             gpu_time_used{0}, hipblas_error{0};
    hipblasLocalHandle handle(arg);

    // Initial Data on CPU
    hipblas_init_matrix(hA, arg, hipblas_client_never_set_nan, hipblas_triangular_matrix, true);

    T* A = (T*)hA;

    // proprocess the matrix to avoid ill-conditioned matrix
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            A[i + j * lda] *= 0.01;

            if(j % 2)
                A[i + j * lda] *= -1;

            if(i == j)
            {
                if(diag == HIPBLAS_DIAG_UNIT)
                    A[i + j * lda] = 1.0;
                else
                    A[i + j * lda] *= 100.0;
            }
        }
    }

    hB = hA;

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dinvA.transfer_from(hA));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasTrtriFn(handle, uplo, diag, N, dA, lda, dinvA, ldinvA));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hA.transfer_from(dinvA));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        ref_trtri<T>(arg.uplo, arg.diag, N, hB, lda);

        if(arg.unit_check)
        {
            near_check_general<T>(N, N, lda, hB.data(), hA.data(), rel_error);
        }
        if(arg.norm_check)
        {
            hipblas_error = norm_check_general<T>('F', N, N, lda, hB, hA);
        }
    }

    if(arg.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasTrtriFn(handle, uplo, diag, N, dA, lda, dinvA, ldinvA));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasTrtriModel{}.log_args<T>(std::cout,
                                        arg,
                                        gpu_time_used,
                                        trtri_gflop_count<T>(N),
                                        trtri_gbyte_count<T>(N),
                                        hipblas_error);
    }
}
