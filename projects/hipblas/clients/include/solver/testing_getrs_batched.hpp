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
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

using hipblasGetrsBatchedModel = ArgumentModel<e_a_type, e_N, e_lda, e_ldb, e_batch_count>;

inline void testname_getrs_batched(const Arguments& arg, std::string& name)
{
    hipblasGetrsBatchedModel{}.test_name(arg, name);
}

template <typename T>
void setup_getrs_batched_testing(const Arguments&        arg,
                                 host_batch_matrix<T>&   hA,
                                 host_batch_matrix<T>&   hB,
                                 host_batch_matrix<T>&   hX,
                                 host_vector<int64_t>&   hIpiv64,
                                 device_batch_matrix<T>& dA,
                                 device_batch_matrix<T>& dB,
                                 device_vector<int>&     dIpiv,
                                 int                     N,
                                 int                     lda,
                                 int                     ldb,
                                 int                     batch_count)
{
    hipblasStride    strideP   = N;
    size_t           Ipiv_size = strideP * batch_count;
    host_vector<int> hIpiv32(Ipiv_size);

    // Initial hA, hX on CPU
    hipblas_init_matrix(hA, arg, hipblas_client_never_set_nan, hipblas_general_matrix, true);
    hipblas_init_matrix(hX, arg, hipblas_client_never_set_nan, hipblas_general_matrix, false, true);

    hipblasOperation_t op = HIPBLAS_OP_N;
    for(int b = 0; b < batch_count; b++)
    {
        // scale A to avoid singularities
        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < N; j++)
            {
                if(i == j)
                    hA[b][i + j * lda] += 400;
                else
                    hA[b][i + j * lda] -= 4;
            }
        }

        // Calculate hB = hA*hX;
        ref_gemm<T>(op, op, N, 1, N, (T)1, hA[b], lda, hX[b], ldb, (T)0, hB[b], ldb);

        // LU factorize hA on the CPU
        int info = ref_getrf<T>(N, N, hA[b], lda, hIpiv64.data() + b * strideP);
        if(info != 0)
        {
            std::cerr << "LU decomposition failed" << std::endl;
            int expectedInfo = 0;
            unit_check_general(1, 1, 1, &expectedInfo, &info);
        }
    }

    for(int i = 0; i < Ipiv_size; i++)
        hIpiv32[i] = hIpiv64[i];

    // Copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));
    CHECK_HIP_ERROR(
        hipMemcpy(dIpiv, hIpiv32.data(), Ipiv_size * sizeof(int), hipMemcpyHostToDevice));
}

template <typename T>
void testing_getrs_batched_bad_arg(const Arguments& arg)
{
    auto hipblasGetrsBatchedFn = arg.api == hipblas_client_api::FORTRAN
                                     ? hipblasGetrsBatched<T, true>
                                     : hipblasGetrsBatched<T, false>;

    hipblasLocalHandle handle(arg);
    const int          N           = 100;
    const int          nrhs        = 1;
    const int          lda         = 101;
    const int          ldb         = 102;
    const int          batch_count = 2;

    const size_t Ipiv_size = size_t(N) * batch_count;

    const hipblasOperation_t op = HIPBLAS_OP_N;

    host_batch_matrix<T> hA(N, N, lda, batch_count);
    host_batch_matrix<T> hB(N, 1, ldb, batch_count);
    host_batch_matrix<T> hX(N, 1, ldb, batch_count);
    host_vector<int64_t> hIpiv64(Ipiv_size);

    device_batch_matrix<T> dA(N, N, lda, batch_count);
    device_batch_matrix<T> dB(N, 1, ldb, batch_count);
    device_vector<int>     dIpiv(Ipiv_size);
    int                    info         = 0;
    int                    expectedInfo = 0;

    hipblas_internal_type<T>* const* dAp = dA.ptr_on_device();
    hipblas_internal_type<T>* const* dBp = dB.ptr_on_device();

    // Need initialization code because even with bad params we call roc/cu-solver
    // so want to give reasonable data

    setup_getrs_batched_testing(arg, hA, hB, hX, hIpiv64, dA, dB, dIpiv, N, lda, ldb, batch_count);

    EXPECT_HIPBLAS_STATUS(
        hipblasGetrsBatchedFn(handle, op, -1, nrhs, dAp, lda, dIpiv, dBp, ldb, &info, batch_count),
        HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -2;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(
        hipblasGetrsBatchedFn(handle, op, N, -1, dAp, lda, dIpiv, dBp, ldb, &info, batch_count),
        HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -3;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(
        hipblasGetrsBatchedFn(handle, op, N, nrhs, dAp, N - 1, dIpiv, dBp, ldb, &info, batch_count),
        HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -5;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(
        hipblasGetrsBatchedFn(handle, op, N, nrhs, dAp, lda, dIpiv, dBp, N - 1, &info, batch_count),
        HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -8;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    // cuBLAS returns HIPBLAS_STATUS_EXECUTION_FAILED and gives info == 0
#ifndef __HIP_PLATFORM_NVCC__
    EXPECT_HIPBLAS_STATUS(
        hipblasGetrsBatchedFn(handle, op, N, nrhs, dAp, lda, dIpiv, dBp, ldb, &info, -1),
        HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -10;
    unit_check_general(1, 1, 1, &expectedInfo, &info);
#endif

    // If N == 0, A, B, and ipiv can be nullptr
    EXPECT_HIPBLAS_STATUS(
        hipblasGetrsBatchedFn(
            handle, op, 0, nrhs, nullptr, lda, nullptr, nullptr, ldb, &info, batch_count),
        HIPBLAS_STATUS_SUCCESS);
    expectedInfo = 0;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    // if nrhs == 0, B can be nullptr
    EXPECT_HIPBLAS_STATUS(
        hipblasGetrsBatchedFn(handle, op, N, 0, dAp, lda, dIpiv, nullptr, ldb, &info, batch_count),
        HIPBLAS_STATUS_SUCCESS);
    expectedInfo = 0;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    // can't make any assumptions about ptrs when batch_count < 0, this is handled by rocSOLVER

    // cuBLAS beckend doesn't check for nullptrs, including info, hipBLAS/rocSOLVER does
#ifndef __HIP_PLATFORM_NVCC__
    EXPECT_HIPBLAS_STATUS(
        hipblasGetrsBatchedFn(handle, op, N, nrhs, dAp, lda, dIpiv, dBp, ldb, nullptr, batch_count),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(
        hipblasGetrsBatchedFn(
            handle, op, N, nrhs, nullptr, lda, dIpiv, dBp, ldb, &info, batch_count),
        HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -4;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(
        hipblasGetrsBatchedFn(handle, op, N, nrhs, dAp, lda, nullptr, dBp, ldb, &info, batch_count),
        HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -6;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(
        hipblasGetrsBatchedFn(
            handle, op, N, nrhs, dAp, lda, dIpiv, nullptr, ldb, &info, batch_count),
        HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -7;
    unit_check_general(1, 1, 1, &expectedInfo, &info);
#endif
}

template <typename T>
void testing_getrs_batched(const Arguments& arg)
{
    using U      = real_t<T>;
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasGetrsBatchedFn
        = FORTRAN ? hipblasGetrsBatched<T, true> : hipblasGetrsBatched<T, false>;

    int N           = arg.N;
    int lda         = arg.lda;
    int ldb         = arg.ldb;
    int batch_count = arg.batch_count;

    hipblasStride strideP   = N;
    size_t        A_size    = size_t(lda) * N;
    size_t        B_size    = size_t(ldb) * 1;
    size_t        Ipiv_size = strideP * batch_count;

    // Check to prevent memory allocation error
    if(N < 0 || lda < N || ldb < N || batch_count < 0)
    {
        return;
    }
    if(batch_count == 0)
    {
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_batch_matrix<T> hA(N, N, lda, batch_count);
    host_batch_matrix<T> hB(N, 1, ldb, batch_count);
    host_batch_matrix<T> hB1(N, 1, ldb, batch_count);
    host_batch_matrix<T> hX(N, 1, ldb, batch_count);
    host_vector<int>     hIpiv(Ipiv_size);
    host_vector<int64_t> hIpiv64(Ipiv_size);
    int                  info;

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hB.memcheck());
    CHECK_HIP_ERROR(hB1.memcheck());
    CHECK_HIP_ERROR(hX.memcheck());

    device_batch_matrix<T> dA(N, N, lda, batch_count);
    device_batch_matrix<T> dB(N, 1, ldb, batch_count);
    device_vector<int>     dIpiv(Ipiv_size);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dIpiv.memcheck());

    double             gpu_time_used{0}, hipblas_error{0};
    hipblasLocalHandle handle(arg);
    hipblasOperation_t op = HIPBLAS_OP_N;

    setup_getrs_batched_testing(arg, hA, hB, hX, hIpiv64, dA, dB, dIpiv, N, lda, ldb, batch_count);

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasGetrsBatchedFn(handle,
                                                  op,
                                                  N,
                                                  1,
                                                  dA.ptr_on_device(),
                                                  lda,
                                                  dIpiv,
                                                  dB.ptr_on_device(),
                                                  ldb,
                                                  &info,
                                                  batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hB1.transfer_from(dB));
        CHECK_HIP_ERROR(
            hipMemcpy(hIpiv.data(), dIpiv, Ipiv_size * sizeof(int), hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU LAPACK
        =================================================================== */

        for(int i = 0; i < Ipiv_size; i++)
            hIpiv64[i] = hIpiv[i];

        for(int b = 0; b < batch_count; b++)
        {
            ref_getrs('N', N, 1, hA[b], lda, hIpiv64.data() + b * strideP, hB[b], ldb);
        }

        hipblas_error = norm_check_general<T>('F', N, 1, ldb, hB, hB1, batch_count);
        if(arg.unit_check)
        {
            U      eps       = std::numeric_limits<U>::epsilon();
            double tolerance = N * eps * 100;
            int    zero      = 0;

            unit_check_error(hipblas_error, tolerance);
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

            CHECK_HIPBLAS_ERROR(hipblasGetrsBatchedFn(handle,
                                                      op,
                                                      N,
                                                      1,
                                                      dA.ptr_on_device(),
                                                      lda,
                                                      dIpiv,
                                                      dB.ptr_on_device(),
                                                      ldb,
                                                      &info,
                                                      batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasGetrsBatchedModel{}.log_args<T>(std::cout,
                                               arg,
                                               gpu_time_used,
                                               getrs_gflop_count<T>(N, 1),
                                               ArgumentLogging::NA_value,
                                               hipblas_error);
    }
}
