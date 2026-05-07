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

using hipblasGelsBatchedModel
    = ArgumentModel<e_a_type, e_transA, e_M, e_N, e_lda, e_ldb, e_batch_count>;

inline void testname_gels_batched(const Arguments& arg, std::string& name)
{
    hipblasGelsBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_gels_batched_bad_arg(const Arguments& arg)
{
    auto hipblasGelsBatchedFn = arg.api == hipblas_client_api::FORTRAN
                                    ? hipblasGelsBatched<T, true>
                                    : hipblasGelsBatched<T, false>;

    hipblasLocalHandle       handle(arg);
    const int                M           = 100;
    const int                N           = 101;
    const int                nrhs        = 10;
    const int                lda         = 102;
    const int                ldb         = 103;
    const int                batch_count = 2;
    const hipblasOperation_t opN         = HIPBLAS_OP_N;
    const hipblasOperation_t opBad       = is_complex<T> ? HIPBLAS_OP_T : HIPBLAS_OP_C;

    // Allocate device memory
    device_batch_matrix<T> dA(M, N, lda, batch_count);
    device_batch_matrix<T> dB(M, nrhs, ldb, batch_count);

    device_vector<int> dInfo(batch_count);
    int                info = 0;
    int                expectedInfo;

    hipblas_internal_type<T>* const* dAp = dA.ptr_on_device();
    hipblas_internal_type<T>* const* dBp = dB.ptr_on_device();

    EXPECT_HIPBLAS_STATUS(
        hipblasGelsBatchedFn(
            handle, opN, M, N, nrhs, dAp, lda, dBp, ldb, nullptr, dInfo, batch_count),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(
        hipblasGelsBatchedFn(
            handle, opN, -1, N, nrhs, dAp, lda, dBp, ldb, &info, dInfo, batch_count),
        HIPBLAS_STATUS_INVALID_VALUE);

    if(arg.bad_arg_all)
    {
        expectedInfo = -2; // cublas gets -1
        unit_check_general(1, 1, 1, &expectedInfo, &info);
    }

    EXPECT_HIPBLAS_STATUS(
        hipblasGelsBatchedFn(
            handle, opN, M, -1, nrhs, dAp, lda, dBp, ldb, &info, dInfo, batch_count),
        HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -3;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(
        hipblasGelsBatchedFn(handle, opN, M, N, -1, dAp, lda, dBp, ldb, &info, dInfo, batch_count),
        HIPBLAS_STATUS_INVALID_VALUE);

    if(arg.bad_arg_all)
    {
        expectedInfo = -4; // cublas gets -2
        unit_check_general(1, 1, 1, &expectedInfo, &info);
    }

    EXPECT_HIPBLAS_STATUS(
        hipblasGelsBatchedFn(
            handle, opN, M, N, nrhs, dAp, M - 1, dBp, ldb, &info, dInfo, batch_count),
        HIPBLAS_STATUS_INVALID_VALUE);

    if(arg.bad_arg_all)
    {
        expectedInfo = -6; // cublas gets -5
        unit_check_general(1, 1, 1, &expectedInfo, &info);
    }

    // Explicit values to check for ldb < M and ldb < N
    EXPECT_HIPBLAS_STATUS(
        hipblasGelsBatchedFn(
            handle, opN, 200, 100, nrhs, dAp, 201, dBp, 199, &info, dInfo, batch_count),
        HIPBLAS_STATUS_INVALID_VALUE);

    if(arg.bad_arg_all)
    {
        expectedInfo = -8; // cublas gets -7
        unit_check_general(1, 1, 1, &expectedInfo, &info);
    }

    EXPECT_HIPBLAS_STATUS(
        hipblasGelsBatchedFn(handle, opN, M, N, nrhs, dAp, lda, dBp, ldb, &info, dInfo, -1),
        HIPBLAS_STATUS_INVALID_VALUE);

    if(arg.bad_arg_all)
    {
        expectedInfo = -11; // cublas gets -8
        unit_check_general(1, 1, 1, &expectedInfo, &info);
    }

    // If M == 0 || N == 0, A can be nullptr
    EXPECT_HIPBLAS_STATUS(
        hipblasGelsBatchedFn(
            handle, opN, 0, N, nrhs, nullptr, lda, dBp, ldb, &info, dInfo, batch_count),
        HIPBLAS_STATUS_SUCCESS);
    expectedInfo = 0;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(
        hipblasGelsBatchedFn(
            handle, opN, M, 0, nrhs, nullptr, lda, dBp, ldb, &info, dInfo, batch_count),
        HIPBLAS_STATUS_SUCCESS);
    expectedInfo = 0;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    // If M == 0 && N == 0, B can be nullptr
    EXPECT_HIPBLAS_STATUS(
        hipblasGelsBatchedFn(
            handle, opN, 0, 0, nrhs, nullptr, lda, nullptr, ldb, &info, dInfo, batch_count),
        HIPBLAS_STATUS_SUCCESS);
    expectedInfo = 0;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    // If batch_count == 0, dInfo can be nullptr
    EXPECT_HIPBLAS_STATUS(
        hipblasGelsBatchedFn(handle, opN, M, N, nrhs, dAp, lda, dBp, ldb, &info, nullptr, 0),
        HIPBLAS_STATUS_SUCCESS);
    expectedInfo = 0;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    if(arg.bad_arg_all)
    {
        // cuBLAS returns HIPBLAS_STATUS_NOT_SUPPORTED for these cases, not checking  for now.
        EXPECT_HIPBLAS_STATUS(
            hipblasGelsBatchedFn(
                handle, opBad, M, N, nrhs, dAp, lda, dBp, ldb, &info, dInfo, batch_count),
            HIPBLAS_STATUS_INVALID_VALUE);
        expectedInfo = -1;
        unit_check_general(1, 1, 1, &expectedInfo, &info);

        EXPECT_HIPBLAS_STATUS(
            hipblasGelsBatchedFn(
                handle, opN, M, N, nrhs, nullptr, lda, dBp, ldb, &info, dInfo, batch_count),
            HIPBLAS_STATUS_INVALID_VALUE);
        expectedInfo = -5;
        unit_check_general(1, 1, 1, &expectedInfo, &info);

        EXPECT_HIPBLAS_STATUS(
            hipblasGelsBatchedFn(
                handle, opN, M, N, nrhs, dAp, lda, nullptr, ldb, &info, dInfo, batch_count),
            HIPBLAS_STATUS_INVALID_VALUE);
        expectedInfo = -7;
        unit_check_general(1, 1, 1, &expectedInfo, &info);

        EXPECT_HIPBLAS_STATUS(
            hipblasGelsBatchedFn(
                handle, opN, 100, 200, nrhs, dAp, lda, dBp, 199, &info, dInfo, batch_count),
            HIPBLAS_STATUS_INVALID_VALUE);
        expectedInfo = -8;
        unit_check_general(1, 1, 1, &expectedInfo, &info);

        EXPECT_HIPBLAS_STATUS(
            hipblasGelsBatchedFn(
                handle, opN, M, N, nrhs, dAp, lda, dBp, ldb, &info, nullptr, batch_count),
            HIPBLAS_STATUS_INVALID_VALUE);
        expectedInfo = -10;
        unit_check_general(1, 1, 1, &expectedInfo, &info);

        // If nrhs == 0, B can be nullptr
        EXPECT_HIPBLAS_STATUS(
            hipblasGelsBatchedFn(
                handle, opN, M, N, 0, dAp, lda, nullptr, ldb, &info, dInfo, batch_count),
            HIPBLAS_STATUS_SUCCESS);
        expectedInfo = 0;
        unit_check_general(1, 1, 1, &expectedInfo, &info);
    }
}

template <typename T>
void testing_gels_batched(const Arguments& arg)
{
    using U      = real_t<T>;
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasGelsBatchedFn
        = FORTRAN ? hipblasGelsBatched<T, true> : hipblasGelsBatched<T, false>;

    char transc      = arg.transA;
    int  N           = arg.N;
    int  M           = arg.M;
    int  nrhs        = arg.K;
    int  lda         = arg.lda;
    int  ldb         = arg.ldb;
    int  batch_count = arg.batch_count;

    if(is_complex<T> && transc == 'T')
        transc = 'C';
    else if(!is_complex<T> && transc == 'C')
        transc = 'T';

    hipblasOperation_t trans = char2hipblas_operation(transc);

    // Check to prevent memory allocation error
    if(M < 0 || N < 0 || nrhs < 0 || lda < M || ldb < M || ldb < N || batch_count < 0)
    {
        return;
    }
    if(batch_count == 0)
    {
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_batch_matrix<T> hA(M, N, lda, batch_count);
    host_batch_matrix<T> hB(M, nrhs, ldb, batch_count);
    host_batch_matrix<T> hB_res(M, nrhs, ldb, batch_count);
    host_vector<T>       info_res(batch_count);
    host_vector<T>       info(batch_count);
    int                  info_input(-1);

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hB.memcheck());
    CHECK_HIP_ERROR(hB_res.memcheck());
    CHECK_HIP_ERROR(info_res.memcheck());
    CHECK_HIP_ERROR(info.memcheck());

    // Allocate device memory
    device_batch_matrix<T> dA(M, N, lda, batch_count);
    device_batch_matrix<T> dB(M, nrhs, ldb, batch_count);
    device_vector<int>     dInfo(batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dInfo.memcheck());

    double             gpu_time_used{0}, hipblas_error{0};
    hipblasLocalHandle handle(arg);

    // Initial hA, hB, hX on CPU
    hipblas_init_matrix(hA, arg, hipblas_client_never_set_nan, hipblas_general_matrix, true);
    hipblas_init_matrix(hB, arg, hipblas_client_never_set_nan, hipblas_general_matrix, false, true);
    hB_res.copy_from(hB);

    // scale A to avoid singularities
    for(int b = 0; b < batch_count; b++)
    {
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
    }

    // Copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasGelsBatchedFn(handle,
                                                 trans,
                                                 M,
                                                 N,
                                                 nrhs,
                                                 dA.ptr_on_device(),
                                                 lda,
                                                 dB.ptr_on_device(),
                                                 ldb,
                                                 &info_input,
                                                 dInfo,
                                                 batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hB_res.transfer_from(dB));
        CHECK_HIP_ERROR(
            hipMemcpy(info_res.data(), dInfo, sizeof(int) * batch_count, hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU LAPACK
        =================================================================== */
        int            sizeW = std::max(1, std::min(M, N) + std::max(std::min(M, N), nrhs));
        host_vector<T> hW(sizeW);

        for(int b = 0; b < batch_count; b++)
        {
            info[b] = ref_gels(transc, M, N, nrhs, hA[b], lda, hB[b], ldb, hW.data(), sizeW);
        }

        hipblas_error
            = norm_check_general<T>('F', std::max(M, N), nrhs, ldb, hB, hB_res, batch_count);

        if(info_input != 0)
            hipblas_error += 1.0;
        for(int b = 0; b < batch_count; b++)
        {
            if(info[b] != info_res[b])
                hipblas_error += 1.0;
        }

        if(arg.unit_check)
        {
            double eps       = std::numeric_limits<U>::epsilon();
            double tolerance = N * eps * 100;
            int    zero      = 0;

            unit_check_error(hipblas_error, tolerance);
            unit_check_general(1, 1, 1, &zero, &info_input);
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

            CHECK_HIPBLAS_ERROR(hipblasGelsBatchedFn(handle,
                                                     trans,
                                                     M,
                                                     N,
                                                     nrhs,
                                                     dA.ptr_on_device(),
                                                     lda,
                                                     dB.ptr_on_device(),
                                                     ldb,
                                                     &info_input,
                                                     dInfo,
                                                     batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasGelsBatchedModel{}.log_args<T>(std::cout,
                                              arg,
                                              gpu_time_used,
                                              ArgumentLogging::NA_value,
                                              ArgumentLogging::NA_value,
                                              hipblas_error);
    }
}
