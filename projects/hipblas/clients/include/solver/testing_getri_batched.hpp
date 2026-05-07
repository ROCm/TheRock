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

using hipblasGetriBatchedModel = ArgumentModel<e_a_type, e_N, e_lda, e_batch_count>;

inline void testname_getri_batched(const Arguments& arg, std::string& name)
{
    hipblasGetriBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_getri_batched_bad_arg(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasGetriBatchedFn
        = FORTRAN ? hipblasGetriBatched<T, true> : hipblasGetriBatched<T, false>;

    hipblasLocalHandle handle(arg);
    int64_t            N           = 101;
    int64_t            M           = N;
    int64_t            lda         = 102;
    int64_t            batch_count = 2;
    int64_t            A_size      = N * lda;
    int64_t            Ipiv_size   = std::min(M, N) * batch_count;

    // Allocate device memory
    device_batch_matrix<T> dA(M, N, lda, batch_count);
    device_batch_matrix<T> dC(M, N, lda, batch_count);
    device_vector<int>     dIpiv(Ipiv_size);
    device_vector<int>     dInfo(batch_count);

    EXPECT_HIPBLAS_STATUS(hipblasGetriBatchedFn(nullptr,
                                                N,
                                                dA.ptr_on_device(),
                                                lda,
                                                dIpiv,
                                                dC.ptr_on_device(),
                                                lda,
                                                dInfo,
                                                batch_count),
                          HIPBLAS_STATUS_NOT_INITIALIZED);

    EXPECT_HIPBLAS_STATUS(hipblasGetriBatchedFn(handle,
                                                -1,
                                                dA.ptr_on_device(),
                                                lda,
                                                dIpiv,
                                                dC.ptr_on_device(),
                                                lda,
                                                dInfo,
                                                batch_count),
                          HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasGetriBatchedFn(handle,
                                                N,
                                                dA.ptr_on_device(),
                                                N - 1,
                                                dIpiv,
                                                dC.ptr_on_device(),
                                                lda,
                                                dInfo,
                                                batch_count),
                          HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasGetriBatchedFn(handle,
                                                N,
                                                dA.ptr_on_device(),
                                                lda,
                                                dIpiv,
                                                dC.ptr_on_device(),
                                                N - 1,
                                                dInfo,
                                                batch_count),
                          HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(
        hipblasGetriBatchedFn(
            handle, N, dA.ptr_on_device(), lda, dIpiv, dC.ptr_on_device(), lda, dInfo, -1),
        HIPBLAS_STATUS_INVALID_VALUE);

    // If N == 0 || batch_count == 0, A, C, and ipiv can be nullptr. rocSolver doesn't allow nullptr with batch_count == 0
    CHECK_HIPBLAS_ERROR(
        hipblasGetriBatchedFn(handle, 0, nullptr, lda, nullptr, nullptr, lda, dInfo, batch_count));

    if(arg.bad_arg_all)
    {
        EXPECT_HIPBLAS_STATUS(
            hipblasGetriBatchedFn(
                handle, N, nullptr, lda, dIpiv, dC.ptr_on_device(), lda, dInfo, batch_count),
            HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_HIPBLAS_STATUS(
            hipblasGetriBatchedFn(
                handle, N, dA.ptr_on_device(), lda, dIpiv, nullptr, lda, dInfo, batch_count),
            HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_HIPBLAS_STATUS(hipblasGetriBatchedFn(handle,
                                                    N,
                                                    dA.ptr_on_device(),
                                                    lda,
                                                    dIpiv,
                                                    dC.ptr_on_device(),
                                                    lda,
                                                    nullptr,
                                                    batch_count),
                              HIPBLAS_STATUS_INVALID_VALUE);
    }
}

template <typename T>
void testing_getri_batched(const Arguments& arg)
{
    using U      = real_t<T>;
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasGetriBatchedFn
        = FORTRAN ? hipblasGetriBatched<T, true> : hipblasGetriBatched<T, false>;

    int M           = arg.N;
    int N           = arg.N;
    int lda         = arg.lda;
    int batch_count = arg.batch_count;

    hipblasStride strideP   = std::min(M, N);
    size_t        Ipiv_size = strideP * batch_count;

    // Check to prevent memory allocation error
    if(M < 0 || N < 0 || lda < M || batch_count < 0)
    {
        return;
    }
    if(batch_count == 0)
    {
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_batch_matrix<T> hA(M, N, lda, batch_count);
    host_batch_matrix<T> hC(M, N, lda, batch_count);
    host_batch_matrix<T> hA1(M, N, lda, batch_count);
    host_vector<int>     hIpiv(Ipiv_size);
    host_vector<int64_t> hIpiv64(Ipiv_size);
    host_vector<int>     hInfo(batch_count);
    host_vector<int>     hInfo1(batch_count);

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hA1.memcheck());
    CHECK_HIP_ERROR(hC.memcheck());

    // Allocate device memory
    device_batch_matrix<T> dA(M, N, lda, batch_count);
    device_batch_matrix<T> dC(M, N, lda, batch_count);
    device_vector<int>     dIpiv(Ipiv_size);
    device_vector<int>     dInfo(batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(dIpiv.memcheck());
    CHECK_DEVICE_ALLOCATION(dInfo.memcheck());

    double             gpu_time_used{0}, hipblas_error{0};
    hipblasLocalHandle handle(arg);

    // Initial hA on CPU
    hipblas_init_matrix(hA, arg, hipblas_client_never_set_nan, hipblas_general_matrix, true);

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

        // perform LU factorization on A
        int64_t* hIpivb = hIpiv64.data() + b * strideP;
        hInfo[b]        = ref_getrf(M, N, hA[b], lda, hIpivb);
    }

    for(int i = 0; i < Ipiv_size; i++)
        hIpiv[i] = hIpiv64[i];

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dC.transfer_from(hC));
    CHECK_HIP_ERROR(hipMemcpy(dIpiv, hIpiv, Ipiv_size * sizeof(int), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemset(dInfo, 0, batch_count * sizeof(int)));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasGetriBatchedFn(handle,
                                                  N,
                                                  dA.ptr_on_device(),
                                                  lda,
                                                  dIpiv,
                                                  dC.ptr_on_device(),
                                                  lda,
                                                  dInfo,
                                                  batch_count));

        // Copy output from device to CPU
        CHECK_HIP_ERROR(hA1.transfer_from(dC));
        CHECK_HIP_ERROR(
            hipMemcpy(hIpiv.data(), dIpiv, Ipiv_size * sizeof(int), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hInfo1.data(), dInfo, batch_count * sizeof(int), hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU LAPACK
        =================================================================== */
        for(int i = 0; i < Ipiv_size; i++)
            hIpiv64[i] = hIpiv[i];

        for(int b = 0; b < batch_count; b++)
        {
            // Workspace query
            host_vector<T> work(1);
            ref_getri(N, hA[b], lda, hIpiv64.data() + b * strideP, work.data(), -1);
            int lwork = type2int(work[0]);

            // Perform inversion
            work     = host_vector<T>(lwork);
            hInfo[b] = ref_getri(N, hA[b], lda, hIpiv64.data() + b * strideP, work.data(), lwork);

            hipblas_error = norm_check_general<T>('F', M, N, lda, hA[b], hA1[b]);
            if(arg.unit_check)
            {
                U      eps       = std::numeric_limits<U>::epsilon();
                double tolerance = eps * 2000;
                unit_check_error(hipblas_error, tolerance);
            }
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

            CHECK_HIPBLAS_ERROR(hipblasGetriBatchedFn(handle,
                                                      N,
                                                      dA.ptr_on_device(),
                                                      lda,
                                                      dIpiv,
                                                      dC.ptr_on_device(),
                                                      lda,
                                                      dInfo,
                                                      batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasGetriBatchedModel{}.log_args<T>(std::cout,
                                               arg,
                                               gpu_time_used,
                                               getri_gflop_count<T>(N),
                                               ArgumentLogging::NA_value,
                                               hipblas_error);
    }
}
