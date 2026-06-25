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

using hipblasGeqrfStridedBatchedModel
    = ArgumentModel<e_a_type, e_M, e_N, e_lda, e_stride_scale, e_batch_count>;

inline void testname_geqrf_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasGeqrfStridedBatchedModel{}.test_name(arg, name);
}

template <typename T>
void setup_geqrf_strided_batched_testing(const Arguments&                arg,
                                         host_strided_batch_matrix<T>&   hA,
                                         device_strided_batch_matrix<T>& dA,
                                         device_vector<T>&               dIpiv,
                                         int                             M,
                                         int                             N,
                                         int                             lda,
                                         hipblasStride                   strideA,
                                         hipblasStride                   strideP,
                                         int                             batch_count)
{
    // Initial hA on CPU
    hipblas_init_matrix(hA, arg, hipblas_client_never_set_nan, hipblas_general_matrix, true);

    size_t Ipiv_size = strideP * batch_count;

    // Initial hA on CPU
    for(int b = 0; b < batch_count; b++)
    {
        // scale A to avoid singularities
        for(int i = 0; i < M; i++)
        {
            for(int j = 0; j < N; j++)
            {
                if(i == j)
                    hA[b][i + j * lda] += 400;
                else
                    hA[b][i + j * lda] -= 4;
            }
        }
    }

    // Copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(hipMemset(dIpiv, 0, Ipiv_size * sizeof(T)));
}

template <typename T>
void testing_geqrf_strided_batched_bad_arg(const Arguments& arg)
{
    auto hipblasGeqrfStridedBatchedFn = arg.api == hipblas_client_api::FORTRAN
                                            ? hipblasGeqrfStridedBatched<T, true>
                                            : hipblasGeqrfStridedBatched<T, false>;

    hipblasLocalHandle handle(arg);
    const int          M           = 100;
    const int          N           = 101;
    int                K           = std::min(M, N);
    const int          lda         = 102;
    const int          batch_count = 2;

    hipblasStride strideA = size_t(lda) * N;
    hipblasStride strideP = K;

    size_t Ipiv_size = strideP * batch_count;

    host_strided_batch_matrix<T> hA(M, N, lda, strideA, batch_count);

    device_strided_batch_matrix<T> dA(M, N, lda, strideA, batch_count);
    device_vector<T>               dIpiv(Ipiv_size);

    int info = 0;
    int expectedInfo;

    setup_geqrf_strided_batched_testing(
        arg, hA, dA, dIpiv, M, N, lda, strideA, strideP, batch_count);

    EXPECT_HIPBLAS_STATUS(hipblasGeqrfStridedBatchedFn(
                              handle, M, N, dA, lda, strideA, dIpiv, strideP, nullptr, batch_count),
                          HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasGeqrfStridedBatchedFn(
                              handle, -1, N, dA, lda, strideA, dIpiv, strideP, &info, batch_count),
                          HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -1;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(hipblasGeqrfStridedBatchedFn(
                              handle, M, -1, dA, lda, strideA, dIpiv, strideP, &info, batch_count),
                          HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -2;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(
        hipblasGeqrfStridedBatchedFn(
            handle, M, N, nullptr, lda, strideA, dIpiv, strideP, &info, batch_count),
        HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -3;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(hipblasGeqrfStridedBatchedFn(
                              handle, M, N, dA, M - 1, strideA, dIpiv, strideP, &info, batch_count),
                          HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -4;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(hipblasGeqrfStridedBatchedFn(
                              handle, M, N, dA, lda, strideA, nullptr, strideP, &info, batch_count),
                          HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -6;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(
        hipblasGeqrfStridedBatchedFn(handle, M, N, dA, lda, strideA, dIpiv, strideP, &info, -1),
        HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -9;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    // If M == 0 || N == 0, A and ipiv can be nullptr
    EXPECT_HIPBLAS_STATUS(
        hipblasGeqrfStridedBatchedFn(
            handle, 0, N, nullptr, lda, strideA, nullptr, strideP, &info, batch_count),
        HIPBLAS_STATUS_SUCCESS);
    expectedInfo = 0;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(
        hipblasGeqrfStridedBatchedFn(
            handle, M, 0, nullptr, lda, strideA, nullptr, strideP, &info, batch_count),
        HIPBLAS_STATUS_SUCCESS);
    expectedInfo = 0;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    // can't make any assumptions about ptrs when batch_count < 0, this is handled by rocSOLVER
}

template <typename T>
void testing_geqrf_strided_batched(const Arguments& arg)
{
    using U      = real_t<T>;
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasGeqrfStridedBatchedFn
        = FORTRAN ? hipblasGeqrfStridedBatched<T, true> : hipblasGeqrfStridedBatched<T, false>;

    int    M            = arg.M;
    int    N            = arg.N;
    int    K            = std::min(M, N);
    int    lda          = arg.lda;
    double stride_scale = arg.stride_scale;
    int    batch_count  = arg.batch_count;

    hipblasStride strideA = lda * N * stride_scale;
    hipblasStride strideP = K * stride_scale;

    int Ipiv_size = strideP * batch_count;
    int info;

    hipblasLocalHandle handle(arg);

    // Check to prevent memory allocation error
    bool invalid_size = M < 0 || N < 0 || lda < std::max(1, M) || batch_count < 0;
    if(invalid_size || !M || !N || !batch_count)
    {
        // including pointers so can test other params
        device_vector<T> dA(1);
        device_vector<T> dIpiv(1);
        hipblasStatus_t  status = hipblasGeqrfStridedBatchedFn(
            handle, M, N, dA, lda, strideA, dIpiv, strideP, &info, batch_count);
        EXPECT_HIPBLAS_STATUS(
            status, (invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS));

        int expected_info = 0;
        if(M < 0)
            expected_info = -1;
        else if(N < 0)
            expected_info = -2;
        else if(lda < std::max(1, M))
            expected_info = -4;
        else if(batch_count < 0)
            expected_info = -9;
        unit_check_general(1, 1, 1, &expected_info, &info);

        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_strided_batch_matrix<T> hA(M, N, lda, strideA, batch_count);
    host_strided_batch_matrix<T> hA1(M, N, lda, strideA, batch_count);
    host_vector<T>               hIpiv(Ipiv_size);
    host_vector<T>               hIpiv1(Ipiv_size);

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hA1.memcheck());
    CHECK_HIP_ERROR(hIpiv.memcheck());
    CHECK_HIP_ERROR(hIpiv1.memcheck());

    device_strided_batch_matrix<T> dA(M, N, lda, strideA, batch_count);
    device_vector<T>               dIpiv(Ipiv_size);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dIpiv.memcheck());

    double gpu_time_used{0}, hipblas_error{0};

    setup_geqrf_strided_batched_testing(
        arg, hA, dA, dIpiv, M, N, lda, strideA, strideP, batch_count);

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    CHECK_HIPBLAS_ERROR(hipblasGeqrfStridedBatchedFn(
        handle, M, N, dA, lda, strideA, dIpiv, strideP, &info, batch_count));

    // Copy output from device to CPU
    CHECK_HIP_ERROR(hIpiv1.transfer_from(dIpiv));
    CHECK_HIP_ERROR(hA1.transfer_from(dA));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
           CPU LAPACK
        =================================================================== */

        // Workspace query
        host_vector<T> work(1);
        ref_geqrf(M, N, hA.data(), lda, hIpiv.data(), work.data(), -1);
        int lwork = type2int(work[0]);

        // Perform factorization
        work = host_vector<T>(lwork);
        for(int b = 0; b < batch_count; b++)
        {
            ref_geqrf(M, N, hA[b], lda, hIpiv.data() + b * strideP, work.data(), N);
        }

        double e1     = norm_check_general<T>('F', M, N, lda, strideA, hA, hA1, batch_count);
        double e2     = norm_check_general<T>('F', K, 1, K, strideP, hIpiv, hIpiv1, batch_count);
        hipblas_error = e1 + e2;

        if(arg.unit_check)
        {
            U      eps       = std::numeric_limits<U>::epsilon();
            double tolerance = eps * 2000;

            unit_check_error(e1, tolerance);
            unit_check_error(e2, tolerance);
            int zero = 0;
            unit_check_general(1, 1, 1, &zero, &info);
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

            CHECK_HIPBLAS_ERROR(hipblasGeqrfStridedBatchedFn(
                handle, M, N, dA, lda, strideA, dIpiv, strideP, &info, batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasGeqrfStridedBatchedModel{}.log_args<T>(std::cout,
                                                      arg,
                                                      gpu_time_used,
                                                      geqrf_gflop_count<T>(N, M),
                                                      ArgumentLogging::NA_value,
                                                      hipblas_error);
    }
}
