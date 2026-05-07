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

using hipblasHemmModel
    = ArgumentModel<e_a_type, e_side, e_uplo, e_M, e_N, e_alpha, e_lda, e_ldb, e_beta, e_ldc>;

inline void testname_hemm(const Arguments& arg, std::string& name)
{
    hipblasHemmModel{}.test_name(arg, name);
}

template <typename T>
void testing_hemm_bad_arg(const Arguments& arg)
{
    using Ts           = hipblas_internal_type<T>;
    auto hipblasHemmFn = arg.api == FORTRAN ? hipblasHemm<T, true> : hipblasHemm<T, false>;
    auto hipblasHemmFn_64
        = arg.api == FORTRAN_64 ? hipblasHemm_64<T, true> : hipblasHemm_64<T, false>;

    hipblasLocalHandle handle(arg);

    int64_t           M    = 101;
    int64_t           N    = 100;
    int64_t           lda  = 102;
    int64_t           ldb  = 103;
    int64_t           ldc  = 104;
    hipblasSideMode_t side = HIPBLAS_SIDE_LEFT;
    hipblasFillMode_t uplo = HIPBLAS_FILL_MODE_LOWER;

    size_t dim_A = (side == HIPBLAS_SIDE_LEFT ? M : N);

    // Allocate device memory
    device_matrix<T> dA(dim_A, dim_A, lda);
    device_matrix<T> dB(M, N, ldb);
    device_matrix<T> dC(M, N, ldc);

    device_vector<T> d_alpha(1), d_beta(1), d_one(1), d_zero(1);
    Ts               h_alpha{1}, h_beta{2}, h_one{1}, h_zero{0};
    if constexpr(is_complex<T>)
        h_one = {1, 0};

    const Ts* alpha = &h_alpha;
    const Ts* beta  = &h_beta;
    const Ts* one   = &h_one;
    const Ts* zero  = &h_zero;

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
                    hipblasHemmFn,
                    (nullptr, side, uplo, M, N, alpha, dA, lda, dB, ldb, beta, dC, ldc));

        DAPI_EXPECT(
#ifdef __HIP_PLATFORM_NVCC__
            HIPBLAS_STATUS_INVALID_ENUM,
#else
            HIPBLAS_STATUS_INVALID_VALUE,
#endif
            hipblasHemmFn,
            (handle, HIPBLAS_SIDE_BOTH, uplo, M, N, alpha, dA, lda, dB, ldb, beta, dC, ldc));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasHemmFn,
                    (handle,
                     (hipblasSideMode_t)HIPBLAS_OP_N,
                     uplo,
                     M,
                     N,
                     alpha,
                     dA,
                     lda,
                     dB,
                     ldb,
                     beta,
                     dC,
                     ldc));
        DAPI_EXPECT(
            HIPBLAS_STATUS_INVALID_VALUE,
            hipblasHemmFn,
            (handle, side, HIPBLAS_FILL_MODE_FULL, M, N, alpha, dA, lda, dB, ldb, beta, dC, ldc));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasHemmFn,
                    (handle,
                     side,
                     (hipblasFillMode_t)HIPBLAS_OP_N,
                     M,
                     N,
                     alpha,
                     dA,
                     lda,
                     dB,
                     ldb,
                     beta,
                     dC,
                     ldc));

        if(arg.bad_arg_all)
        {
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasHemmFn,
                        (handle, side, uplo, M, N, nullptr, dA, lda, dB, ldb, beta, dC, ldc));
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasHemmFn,
                        (handle, side, uplo, M, N, alpha, dA, lda, dB, ldb, nullptr, dC, ldc));

            if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
            {
                DAPI_EXPECT(
                    HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasHemmFn,
                    (handle, side, uplo, M, N, alpha, nullptr, lda, dB, ldb, beta, dC, ldc));
                DAPI_EXPECT(
                    HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasHemmFn,
                    (handle, side, uplo, M, N, alpha, dA, lda, nullptr, ldb, beta, dC, ldc));
                DAPI_EXPECT(
                    HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasHemmFn,
                    (handle, side, uplo, M, N, alpha, dA, lda, dB, ldb, beta, nullptr, ldc));

                // hemm will quick-return with alpha == 0 && beta == 1. Here, c_i32_overflow will rollover in the case of 32-bit params,
                // and quick-return with 64-bit params. This depends on implementation so only testing rocBLAS backend
                DAPI_EXPECT((arg.api & c_API_64) ? HIPBLAS_STATUS_SUCCESS
                                                 : HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasHemmFn,
                            (handle,
                             side,
                             uplo,
                             c_i32_overflow,
                             c_i32_overflow,
                             zero,
                             nullptr,
                             c_i32_overflow,
                             nullptr,
                             c_i32_overflow,
                             one,
                             nullptr,
                             c_i32_overflow));
            }

            // alpha == 0 && beta == 1, can have all nullptrs
            DAPI_CHECK(
                hipblasHemmFn,
                (handle, side, uplo, M, N, zero, nullptr, lda, nullptr, ldb, one, nullptr, ldc));
        }

        // If M == 0 || N == 0, can have nullptrs
        DAPI_CHECK(
            hipblasHemmFn,
            (handle, side, uplo, 0, N, nullptr, nullptr, lda, nullptr, ldb, nullptr, nullptr, ldc));
        DAPI_CHECK(
            hipblasHemmFn,
            (handle, side, uplo, M, 0, nullptr, nullptr, lda, nullptr, ldb, nullptr, nullptr, ldc));
    }
}

template <typename T>
void testing_hemm(const Arguments& arg)
{
    using Ts           = hipblas_internal_type<T>;
    auto hipblasHemmFn = arg.api == FORTRAN ? hipblasHemm<T, true> : hipblasHemm<T, false>;
    auto hipblasHemmFn_64
        = arg.api == FORTRAN_64 ? hipblasHemm_64<T, true> : hipblasHemm_64<T, false>;

    hipblasSideMode_t side = char2hipblas_side(arg.side);
    hipblasFillMode_t uplo = char2hipblas_fill(arg.uplo);
    int64_t           M    = arg.M;
    int64_t           N    = arg.N;
    int64_t           lda  = arg.lda;
    int64_t           ldb  = arg.ldb;
    int64_t           ldc  = arg.ldc;

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    size_t dim_A = (side == HIPBLAS_SIDE_LEFT ? M : N);

    hipblasLocalHandle handle(arg);

    // check here to prevent undefined memory allocation error
    bool invalid_size = M < 0 || N < 0 || ldc < M || ldb < M || lda < dim_A;
    if(invalid_size || !M || !N)
    {
        DAPI_EXPECT(
            invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS,
            hipblasHemmFn,
            (handle, side, uplo, M, N, nullptr, nullptr, lda, nullptr, ldb, nullptr, nullptr, ldc));
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_matrix<T> hA(dim_A, dim_A, lda);
    host_matrix<T> hB(M, N, ldb);
    host_matrix<T> hC_host(M, N, ldc);
    host_matrix<T> hC_device(M, N, ldc);
    host_matrix<T> hC_cpu(M, N, ldc);

    // Allocate device memory
    device_matrix<T> dA(dim_A, dim_A, lda);
    device_matrix<T> dB(M, N, ldb);
    device_matrix<T> dC(M, N, ldc);
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta.memcheck());

    double gpu_time_used{0}, hipblas_error_host{0}, hipblas_error_device{0};
    // Initial Data on CPU
    hipblas_init_matrix(hA, arg, hipblas_client_never_set_nan, hipblas_hermitian_matrix, true);
    hipblas_init_matrix(
        hB, arg, hipblas_client_alpha_sets_nan, hipblas_general_matrix, false, true);
    hipblas_init_matrix(hC_host, arg, hipblas_client_beta_sets_nan, hipblas_general_matrix);

    hC_cpu    = hC_host;
    hC_device = hC_host;

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));
    CHECK_HIP_ERROR(dC.transfer_from(hC_host));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        DAPI_CHECK(hipblasHemmFn,
                   (handle,
                    side,
                    uplo,
                    M,
                    N,
                    reinterpret_cast<Ts*>(&h_alpha),
                    dA,
                    lda,
                    dB,
                    ldb,
                    reinterpret_cast<Ts*>(&h_beta),
                    dC,
                    ldc));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hC_host.transfer_from(dC));

        CHECK_HIP_ERROR(dC.transfer_from(hC_device));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasHemmFn,
                   (handle, side, uplo, M, N, d_alpha, dA, lda, dB, ldb, d_beta, dC, ldc));

        CHECK_HIP_ERROR(hC_device.transfer_from(dC));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        ref_hemm<T>(
            side, uplo, M, N, h_alpha, hA.data(), lda, hB.data(), ldb, h_beta, hC_cpu.data(), ldc);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(M, N, ldc, hC_cpu, hC_host);
            unit_check_general<T>(M, N, ldc, hC_cpu, hC_device);
        }

        if(arg.norm_check)
        {
            hipblas_error_host   = norm_check_general<T>('F', M, N, ldc, hC_cpu, hC_host);
            hipblas_error_device = norm_check_general<T>('F', M, N, ldc, hC_cpu, hC_device);
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

            DAPI_DISPATCH(hipblasHemmFn,
                          (handle, side, uplo, M, N, d_alpha, dA, lda, dB, ldb, d_beta, dC, ldc));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        hipblasHemmModel{}.log_args<T>(std::cout,
                                       arg,
                                       gpu_time_used,
                                       hemm_gflop_count<T>(M, N, dim_A),
                                       hemm_gbyte_count<T>(M, N, dim_A),
                                       hipblas_error_host,
                                       hipblas_error_device);
    }
}
