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

using hipblasTrtriStridedBatchedModel
    = ArgumentModel<e_a_type, e_uplo, e_diag, e_N, e_lda, e_stride_scale, e_batch_count>;

inline void testname_trtri_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasTrtriStridedBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_trtri_strided_batched_bad_arg(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasTrtriStridedBatchedFn
        = FORTRAN ? hipblasTrtriStridedBatched<T, true> : hipblasTrtriStridedBatched<T, false>;

    hipblasLocalHandle handle(arg);

    int64_t           N           = 100;
    int64_t           lda         = 102;
    int64_t           batch_count = 2;
    hipblasFillMode_t uplo        = HIPBLAS_FILL_MODE_LOWER;
    hipblasDiagType_t diag        = HIPBLAS_DIAG_NON_UNIT;

    hipblasStride stride_A    = N * lda;
    hipblasStride stride_invA = N * lda;

    // Allocate device memory
    device_strided_batch_matrix<T> dA(N, N, lda, stride_A, batch_count);
    device_strided_batch_matrix<T> dinvA(N, N, lda, stride_invA, batch_count);

    EXPECT_HIPBLAS_STATUS(
        hipblasTrtriStridedBatchedFn(
            nullptr, uplo, diag, N, dA, lda, stride_A, dinvA, lda, stride_invA, batch_count),
        HIPBLAS_STATUS_NOT_INITIALIZED);

    EXPECT_HIPBLAS_STATUS(hipblasTrtriStridedBatchedFn(handle,
                                                       HIPBLAS_FILL_MODE_FULL,
                                                       diag,
                                                       N,
                                                       dA,
                                                       lda,
                                                       stride_A,
                                                       dinvA,
                                                       lda,
                                                       stride_invA,
                                                       batch_count),
                          HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_HIPBLAS_STATUS(hipblasTrtriStridedBatchedFn(handle,
                                                       (hipblasFillMode_t)HIPBLAS_OP_N,
                                                       diag,
                                                       N,
                                                       dA,
                                                       lda,
                                                       stride_A,
                                                       dinvA,
                                                       lda,
                                                       stride_invA,
                                                       batch_count),
                          HIPBLAS_STATUS_INVALID_ENUM);
    EXPECT_HIPBLAS_STATUS(hipblasTrtriStridedBatchedFn(handle,
                                                       uplo,
                                                       (hipblasDiagType_t)HIPBLAS_OP_N,
                                                       N,
                                                       dA,
                                                       lda,
                                                       stride_A,
                                                       dinvA,
                                                       lda,
                                                       stride_invA,
                                                       batch_count),
                          HIPBLAS_STATUS_INVALID_ENUM);

    if(arg.bad_arg_all)
    {
        EXPECT_HIPBLAS_STATUS(hipblasTrtriStridedBatchedFn(handle,
                                                           uplo,
                                                           diag,
                                                           N,
                                                           nullptr,
                                                           lda,
                                                           stride_A,
                                                           dinvA,
                                                           lda,
                                                           stride_invA,
                                                           batch_count),
                              HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_HIPBLAS_STATUS(hipblasTrtriStridedBatchedFn(handle,
                                                           uplo,
                                                           diag,
                                                           N,
                                                           nullptr,
                                                           lda,
                                                           stride_A,
                                                           nullptr,
                                                           lda,
                                                           stride_invA,
                                                           batch_count),
                              HIPBLAS_STATUS_INVALID_VALUE);
    }

    // If N == 0, can have nullptrs
    CHECK_HIPBLAS_ERROR(hipblasTrtriStridedBatchedFn(
        handle, uplo, diag, 0, nullptr, lda, stride_A, nullptr, lda, stride_invA, batch_count));
    CHECK_HIPBLAS_ERROR(hipblasTrtriStridedBatchedFn(
        handle, uplo, diag, N, nullptr, lda, stride_A, nullptr, lda, stride_invA, 0));
}

template <typename T>
void testing_trtri_strided_batched(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasTrtriStridedBatchedFn
        = FORTRAN ? hipblasTrtriStridedBatched<T, true> : hipblasTrtriStridedBatched<T, false>;

    const double rel_error = get_epsilon<T>() * 1000;

    hipblasFillMode_t uplo         = char2hipblas_fill(arg.uplo);
    hipblasDiagType_t diag         = char2hipblas_diagonal(arg.diag);
    int               N            = arg.N;
    int               lda          = arg.lda;
    double            stride_scale = arg.stride_scale;
    int               batch_count  = arg.batch_count;

    int           ldinvA   = lda;
    hipblasStride stride_A = lda * N * stride_scale;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    bool invalid_size = N < 0 || lda < N || batch_count < 0;
    if(invalid_size || lda == 0 || N == 0 || batch_count == 0)
    {
        EXPECT_HIPBLAS_STATUS(
            hipblasTrtriStridedBatchedFn(
                handle, uplo, diag, N, nullptr, lda, stride_A, nullptr, lda, stride_A, batch_count),
            invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS);
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_strided_batch_matrix<T> hA(N, N, lda, stride_A, batch_count);
    host_strided_batch_matrix<T> hB(N, N, lda, stride_A, batch_count);

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hB.memcheck());

    // Allocate device memory
    device_strided_batch_matrix<T> dA(N, N, lda, stride_A, batch_count);
    device_strided_batch_matrix<T> dinvA(N, N, lda, stride_A, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dinvA.memcheck());

    double gpu_time_used{0}, hipblas_error{0};

    // Initial Data on CPU
    hipblas_init_matrix(hA, arg, hipblas_client_never_set_nan, hipblas_triangular_matrix, true);

    for(int b = 0; b < batch_count; b++)
    {
        // proprocess the matrix to avoid ill-conditioned matrix
        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < N; j++)
            {
                hA[b][i + j * lda] *= 0.01;

                if(j % 2)
                    hA[b][i + j * lda] *= -1;

                if(i == j)
                {
                    if(diag == HIPBLAS_DIAG_UNIT)
                        hA[b][i + j * lda] = 1.0;
                    else
                        hA[b][i + j * lda] *= 100.0;
                }
            }
        }
    }

    hB.copy_from(hA);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dinvA.transfer_from(hA));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasTrtriStridedBatchedFn(
            handle, uplo, diag, N, dA, lda, stride_A, dinvA, ldinvA, stride_A, batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hA.transfer_from(dinvA));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            ref_trtri<T>(arg.uplo, arg.diag, N, hB.data() + b * stride_A, lda);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            near_check_general<T>(N, N, batch_count, lda, stride_A, hB, hA, rel_error);
        }
        if(arg.norm_check)
        {
            hipblas_error = norm_check_general<T>('F', N, N, lda, stride_A, hB, hA, batch_count);
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

            CHECK_HIPBLAS_ERROR(hipblasTrtriStridedBatchedFn(
                handle, uplo, diag, N, dA, lda, stride_A, dinvA, ldinvA, stride_A, batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasTrtriStridedBatchedModel{}.log_args<T>(std::cout,
                                                      arg,
                                                      gpu_time_used,
                                                      trtri_gflop_count<T>(N),
                                                      trtri_gbyte_count<T>(N),
                                                      hipblas_error);
    }
}
