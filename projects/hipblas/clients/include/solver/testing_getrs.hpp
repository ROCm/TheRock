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

using hipblasGetrsModel = ArgumentModel<e_a_type, e_N, e_lda, e_ldb>;

inline void testname_getrs(const Arguments& arg, std::string& name)
{
    hipblasGetrsModel{}.test_name(arg, name);
}

template <typename T>
void setup_getrs_testing(const Arguments&      arg,
                         host_matrix<T>&       hA,
                         host_matrix<T>&       hB,
                         host_matrix<T>&       hX,
                         host_vector<int64_t>& hIpiv64,
                         device_matrix<T>&     dA,
                         device_matrix<T>&     dB,
                         device_vector<int>&   dIpiv,
                         int                   N,
                         int                   lda,
                         int                   ldb)
{
    const size_t Ipiv_size = N;

    host_vector<int> hIpiv32(Ipiv_size);

    // Initial hA, hX on CPU
    hipblas_init_matrix(hA, arg, hipblas_client_never_set_nan, hipblas_general_matrix, true);
    hipblas_init_matrix(hX, arg, hipblas_client_never_set_nan, hipblas_general_matrix, false, true);

    T* A = (T*)hA;
    // scale A to avoid singularities
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            if(i == j)
                A[i + j * lda] += 400;
            else
                A[i + j * lda] -= 4;
        }
    }

    // Calculate hB = hA*hX;
    hipblasOperation_t opN = HIPBLAS_OP_N;
    ref_gemm<T>(opN, opN, N, 1, N, (T)1, hA.data(), lda, hX.data(), ldb, (T)0, hB.data(), ldb);

    // LU factorize hA on the CPU
    int info = ref_getrf<T>(N, N, hA.data(), lda, hIpiv64.data());
    if(info != 0)
    {
        std::cerr << "LU decomposition failed" << std::endl;
        int expectedInfo = 0;
        unit_check_general(1, 1, 1, &expectedInfo, &info);
    }

    for(int i = 0; i < Ipiv_size; i++)
        hIpiv32[i] = hIpiv64[i];

    // Copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));
    CHECK_HIP_ERROR(hipMemcpy(dIpiv, hIpiv32, Ipiv_size * sizeof(int), hipMemcpyHostToDevice));
}

template <typename T>
void testing_getrs_bad_arg(const Arguments& arg)
{
    auto hipblasGetrsFn
        = arg.api == hipblas_client_api::FORTRAN ? hipblasGetrs<T, true> : hipblasGetrs<T, false>;

    hipblasLocalHandle handle(arg);
    const int          N         = 100;
    const int          nrhs      = 1;
    const int          lda       = 101;
    const int          ldb       = 102;
    const int          Ipiv_size = N;

    const hipblasOperation_t op = HIPBLAS_OP_N;

    host_matrix<T>       hA(N, N, lda);
    host_matrix<T>       hB(N, 1, ldb);
    host_matrix<T>       hX(N, 1, ldb);
    host_vector<int64_t> hIpiv64(Ipiv_size);

    device_matrix<T>   dA(N, N, lda);
    device_matrix<T>   dB(N, 1, ldb);
    device_vector<int> dIpiv(Ipiv_size);
    int                info = 0;
    int                expectedInfo;

    // Need initialization code because even with bad params we call roc/cu-solver
    // so want to give reasonable data
    setup_getrs_testing(arg, hA, hB, hX, hIpiv64, dA, dB, dIpiv, N, lda, ldb);

    EXPECT_HIPBLAS_STATUS(hipblasGetrsFn(handle, op, N, nrhs, dA, lda, dIpiv, dB, ldb, nullptr),
                          HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasGetrsFn(handle, op, -1, nrhs, dA, lda, dIpiv, dB, ldb, &info),
                          HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -2;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(hipblasGetrsFn(handle, op, N, -1, dA, lda, dIpiv, dB, ldb, &info),
                          HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -3;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(hipblasGetrsFn(handle, op, N, nrhs, nullptr, lda, dIpiv, dB, ldb, &info),
                          HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -4;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(hipblasGetrsFn(handle, op, N, nrhs, dA, N - 1, dIpiv, dB, ldb, &info),
                          HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -5;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(hipblasGetrsFn(handle, op, N, nrhs, dA, lda, nullptr, dB, ldb, &info),
                          HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -6;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(hipblasGetrsFn(handle, op, N, nrhs, dA, lda, dIpiv, nullptr, ldb, &info),
                          HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -7;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(hipblasGetrsFn(handle, op, N, nrhs, dA, lda, dIpiv, dB, N - 1, &info),
                          HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -8;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    // If N == 0, A, B, and ipiv can be nullptr
    EXPECT_HIPBLAS_STATUS(
        hipblasGetrsFn(handle, op, 0, nrhs, nullptr, lda, nullptr, nullptr, ldb, &info),
        HIPBLAS_STATUS_SUCCESS);
    expectedInfo = 0;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    // if nrhs == 0, B can be nullptr
    EXPECT_HIPBLAS_STATUS(hipblasGetrsFn(handle, op, N, 0, dA, lda, dIpiv, nullptr, ldb, &info),
                          HIPBLAS_STATUS_SUCCESS);
    expectedInfo = 0;
    unit_check_general(1, 1, 1, &expectedInfo, &info);
}

template <typename T>
void testing_getrs(const Arguments& arg)
{
    using U             = real_t<T>;
    bool FORTRAN        = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasGetrsFn = FORTRAN ? hipblasGetrs<T, true> : hipblasGetrs<T, false>;

    int N   = arg.N;
    int lda = arg.lda;
    int ldb = arg.ldb;

    size_t Ipiv_size = N;

    // Check to prevent memory allocation error
    if(N < 0 || lda < N || ldb < N)
    {
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_matrix<T>       hA(N, N, lda);
    host_matrix<T>       hX(N, 1, ldb);
    host_matrix<T>       hB(N, 1, ldb);
    host_matrix<T>       hB1(N, 1, ldb);
    host_vector<int>     hIpiv(Ipiv_size);
    host_vector<int64_t> hIpiv64(Ipiv_size);
    int                  info;

    device_matrix<T>   dA(N, N, lda);
    device_matrix<T>   dB(N, 1, ldb);
    device_vector<int> dIpiv(Ipiv_size);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dIpiv.memcheck());

    double             gpu_time_used{0}, hipblas_error{0};
    hipblasLocalHandle handle(arg);
    hipblasOperation_t op = HIPBLAS_OP_N;

    setup_getrs_testing(arg, hA, hB, hX, hIpiv64, dA, dB, dIpiv, N, lda, ldb);

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasGetrsFn(handle, op, N, 1, dA, lda, dIpiv, dB, ldb, &info));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hB1.transfer_from(dB));
        CHECK_HIP_ERROR(hipMemcpy(hIpiv, dIpiv, Ipiv_size * sizeof(int), hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU LAPACK
        =================================================================== */
        for(int i = 0; i < Ipiv_size; i++)
            hIpiv64[i] = hIpiv[i];

        ref_getrs('N', N, 1, hA.data(), lda, hIpiv64.data(), hB.data(), ldb);

        hipblas_error = norm_check_general<T>('F', N, 1, ldb, hB.data(), hB1.data());

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

            CHECK_HIPBLAS_ERROR(hipblasGetrsFn(handle, op, N, 1, dA, lda, dIpiv, dB, ldb, &info));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasGetrsModel{}.log_args<T>(std::cout,
                                        arg,
                                        gpu_time_used,
                                        getrs_gflop_count<T>(N, 1),
                                        ArgumentLogging::NA_value,
                                        hipblas_error);
    }
}
