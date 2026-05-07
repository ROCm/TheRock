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

/* ============================================================================================ */

using hipblasHerkModel
    = ArgumentModel<e_a_type, e_uplo, e_transA, e_N, e_K, e_alpha, e_lda, e_beta, e_ldc>;

inline void testname_herk(const Arguments& arg, std::string& name)
{
    hipblasHerkModel{}.test_name(arg, name);
}

template <typename T>
void testing_herk_bad_arg(const Arguments& arg)
{
    using U            = real_t<T>;
    auto hipblasHerkFn = arg.api == FORTRAN ? hipblasHerk<T, U, true> : hipblasHerk<T, U, false>;
    auto hipblasHerkFn_64
        = arg.api == FORTRAN_64 ? hipblasHerk_64<T, U, true> : hipblasHerk_64<T, U, false>;

    hipblasLocalHandle handle(arg);

    int64_t            N      = 101;
    int64_t            K      = 100;
    int64_t            lda    = 102;
    int64_t            ldc    = 104;
    hipblasOperation_t transA = HIPBLAS_OP_N;
    hipblasFillMode_t  uplo   = HIPBLAS_FILL_MODE_LOWER;

    size_t rows = (transA != HIPBLAS_OP_N ? std::max(K, int64_t(1)) : N);
    size_t cols = (transA == HIPBLAS_OP_N ? std::max(K, int64_t(1)) : N);

    // Allocate device memory
    device_matrix<T> dA(rows, cols, lda);
    device_matrix<T> dC(N, N, ldc);

    device_vector<U> d_alpha(1), d_zero(1), d_beta(1), d_one(1);
    const U          h_alpha{1}, h_zero{0}, h_beta{2}, h_one{1};

    const U* alpha = &h_alpha;
    const U* beta  = &h_beta;
    const U* one   = &h_one;
    const U* zero  = &h_zero;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        if(pointer_mode == HIPBLAS_POINTER_MODE_DEVICE)
        {
            CHECK_HIP_ERROR(hipMemcpy(d_alpha, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(d_beta, beta, sizeof(*beta), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(d_one, one, sizeof(*one), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(d_zero, zero, sizeof(*zero), hipMemcpyHostToDevice));
            alpha = d_alpha;
            beta  = d_beta;
            one   = d_one;
            zero  = d_zero;
        }

        DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                    hipblasHerkFn,
                    (nullptr, uplo, transA, N, K, alpha, dA, lda, beta, dC, ldc));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasHerkFn,
                    (handle, HIPBLAS_FILL_MODE_FULL, transA, N, K, alpha, dA, lda, beta, dC, ldc));

        DAPI_EXPECT(
            HIPBLAS_STATUS_INVALID_ENUM,
            hipblasHerkFn,
            (handle, (hipblasFillMode_t)HIPBLAS_OP_N, transA, N, K, alpha, dA, lda, beta, dC, ldc));

        // TODO: Supported in cuBLAS but not in rocBLAS? Need to investigate.
        // DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
        //     hipblasHerkFn, (handle, uplo, HIPBLAS_OP_T, N, K, alpha, dA, lda, beta, dC, ldc));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasHerkFn,
                    (handle,
                     uplo,
                     (hipblasOperation_t)HIPBLAS_FILL_MODE_FULL,
                     N,
                     K,
                     alpha,
                     dA,
                     lda,
                     beta,
                     dC,
                     ldc));

        if(arg.bad_arg_all)
        {
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasHerkFn,
                        (handle, uplo, transA, N, K, nullptr, dA, lda, beta, dC, ldc));
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasHerkFn,
                        (handle, uplo, transA, N, K, alpha, dA, lda, nullptr, dC, ldc));

            if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
            {
                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasHerkFn,
                            (handle, uplo, transA, N, K, alpha, nullptr, lda, beta, dC, ldc));
                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasHerkFn,
                            (handle, uplo, transA, N, K, alpha, dA, lda, beta, nullptr, ldc));

                // herk will quick-return with alpha == 0 && beta == 1. Here, c_i32_overflow will rollover in the case of 32-bit params,
                // and quick-return with 64-bit params. This depends on implementation so only testing rocBLAS backend
                DAPI_EXPECT((arg.api & c_API_64) ? HIPBLAS_STATUS_SUCCESS
                                                 : HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasHerkFn,
                            (handle,
                             uplo,
                             transA,
                             c_i32_overflow,
                             c_i32_overflow,
                             zero,
                             nullptr,
                             c_i32_overflow,
                             one,
                             nullptr,
                             c_i32_overflow));
            }

            // If k == 0 && beta == 1, A, C may be nullptr
            DAPI_CHECK(hipblasHerkFn,
                       (handle, uplo, transA, N, 0, alpha, nullptr, lda, one, nullptr, ldc));

            // If alpha == 0 && beta == 1, A, C may be nullptr
            DAPI_CHECK(hipblasHerkFn,
                       (handle, uplo, transA, N, K, zero, nullptr, lda, one, nullptr, ldc));
        }

        // If N == 0, can have nullptrs
        DAPI_CHECK(hipblasHerkFn,
                   (handle, uplo, transA, 0, K, nullptr, nullptr, lda, nullptr, nullptr, ldc));
    }
}

template <typename T>
void testing_herk(const Arguments& arg)
{
    using U            = real_t<T>;
    bool FORTRAN       = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasHerkFn = arg.api == FORTRAN ? hipblasHerk<T, U, true> : hipblasHerk<T, U, false>;
    auto hipblasHerkFn_64
        = arg.api == FORTRAN_64 ? hipblasHerk_64<T, U, true> : hipblasHerk_64<T, U, false>;

    int64_t N   = arg.N;
    int64_t K   = arg.K;
    int64_t lda = arg.lda;
    int64_t ldc = arg.ldc;

    hipblasFillMode_t  uplo   = char2hipblas_fill(arg.uplo);
    hipblasOperation_t transA = char2hipblas_operation(arg.transA);

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = N < 0 || K < 0 || ldc < N || (transA == HIPBLAS_OP_N && lda < N)
                        || (transA != HIPBLAS_OP_N && lda < K);
    if(invalid_size || !N)
    {
        DAPI_EXPECT(invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS,
                    hipblasHerkFn,
                    (handle, uplo, transA, N, K, nullptr, nullptr, lda, nullptr, nullptr, ldc));
        return;
    }

    size_t rows = (transA != HIPBLAS_OP_N ? std::max(K, int64_t(1)) : N);
    size_t cols = (transA == HIPBLAS_OP_N ? std::max(K, int64_t(1)) : N);

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_matrix<T> hA(rows, cols, lda);
    host_matrix<T> hC_host(N, N, ldc);
    host_matrix<T> hC_device(N, N, ldc);
    host_matrix<T> hC_gold(N, N, ldc);

    // Allocate device memory
    device_matrix<T> dA(rows, cols, lda);
    device_matrix<T> dC(N, N, ldc);
    device_vector<U> d_alpha(1);
    device_vector<U> d_beta(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta.memcheck());

    U h_alpha = arg.get_alpha<U>();
    U h_beta  = arg.get_beta<U>();

    double gpu_time_used{0}, hipblas_error_host{0}, hipblas_error_device{0};

    // Initial Data on CPU
    hipblas_init_matrix(hA, arg, hipblas_client_alpha_sets_nan, hipblas_general_matrix, true);
    hipblas_init_matrix(
        hC_host, arg, hipblas_client_beta_sets_nan, hipblas_hermitian_matrix, false);

    hC_device = hC_host;
    hC_gold   = hC_host;

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dC.transfer_from(hC_host));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(U), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(U), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        DAPI_CHECK(hipblasHerkFn,
                   (handle, uplo, transA, N, K, &h_alpha, dA, lda, &h_beta, dC, ldc));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hC_host.transfer_from(dC));

        CHECK_HIP_ERROR(dC.transfer_from(hC_device));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasHerkFn, (handle, uplo, transA, N, K, d_alpha, dA, lda, d_beta, dC, ldc));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hC_device.transfer_from(dC));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        ref_herk<T>(uplo, transA, N, K, h_alpha, hA, lda, h_beta, hC_gold, ldc);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(N, N, ldc, hC_gold, hC_host);
            unit_check_general<T>(N, N, ldc, hC_gold, hC_device);
        }

        if(arg.norm_check)
        {
            hipblas_error_host   = norm_check_general<T>('F', N, N, ldc, hC_gold, hC_host);
            hipblas_error_device = norm_check_general<T>('F', N, N, ldc, hC_gold, hC_device);
        }
    }

    if(arg.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(hipblasHerkFn,
                          (handle, uplo, transA, N, K, d_alpha, dA, lda, d_beta, dC, ldc));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        hipblasHerkModel{}.log_args<T>(std::cout,
                                       arg,
                                       gpu_time_used,
                                       herk_gflop_count<T>(N, K),
                                       herk_gbyte_count<T>(N, K),
                                       hipblas_error_host,
                                       hipblas_error_device);
    }
}
