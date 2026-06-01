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

using hipblasTrsmModel = ArgumentModel<e_a_type,
                                       e_side,
                                       e_uplo,
                                       e_transA,
                                       e_diag,
                                       e_M,
                                       e_N,
                                       e_alpha,
                                       e_lda,
                                       e_ldb,
                                       e_user_allocated_workspace>;

inline void testname_trsm(const Arguments& arg, std::string& name)
{
    hipblasTrsmModel{}.test_name(arg, name);
}

template <typename T>
void testing_trsm_bad_arg(const Arguments& arg)
{
    using Ts           = hipblas_internal_type<T>;
    auto hipblasTrsmFn = arg.api == FORTRAN ? hipblasTrsm<T, true> : hipblasTrsm<T, false>;
    auto hipblasTrsmFn_64
        = arg.api == FORTRAN_64 ? hipblasTrsm_64<T, true> : hipblasTrsm_64<T, false>;

    hipblasLocalHandle handle(arg);

    int64_t            M      = 101;
    int64_t            N      = 100;
    int64_t            lda    = 102;
    int64_t            ldb    = 103;
    hipblasSideMode_t  side   = HIPBLAS_SIDE_LEFT;
    hipblasFillMode_t  uplo   = HIPBLAS_FILL_MODE_LOWER;
    hipblasOperation_t transA = HIPBLAS_OP_N;
    hipblasDiagType_t  diag   = HIPBLAS_DIAG_NON_UNIT;

    int64_t K = side == HIPBLAS_SIDE_LEFT ? M : N;

    // Allocate device memory
    device_matrix<T> dA(K, K, lda);
    device_matrix<T> dB(M, N, ldb);

    device_vector<T> d_alpha(1), d_zero(1);
    const Ts         h_alpha{1}, h_zero{0};

    const Ts* alpha = &h_alpha;
    const Ts* zero  = &h_zero;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        if(pointer_mode == HIPBLAS_POINTER_MODE_DEVICE)
        {
            CHECK_HIP_ERROR(hipMemcpy(d_alpha, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(d_zero, zero, sizeof(*zero), hipMemcpyHostToDevice));
            alpha = d_alpha;
            zero  = d_zero;
        }

        DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                    hipblasTrsmFn,
                    (nullptr, side, uplo, transA, diag, M, N, alpha, dA, lda, dB, ldb));

        DAPI_EXPECT(
#ifdef __HIP_PLATFORM_NVCC__
            HIPBLAS_STATUS_INVALID_ENUM,
#else
            HIPBLAS_STATUS_INVALID_VALUE,
#endif
            hipblasTrsmFn,
            (handle, HIPBLAS_SIDE_BOTH, uplo, transA, diag, M, N, alpha, dA, lda, dB, ldb));
        DAPI_EXPECT(
            HIPBLAS_STATUS_INVALID_VALUE,
            hipblasTrsmFn,
            (handle, side, HIPBLAS_FILL_MODE_FULL, transA, diag, M, N, alpha, dA, lda, dB, ldb));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasTrsmFn,
                    (handle,
                     side,
                     uplo,
                     (hipblasOperation_t)HIPBLAS_FILL_MODE_FULL,
                     diag,
                     M,
                     N,
                     alpha,
                     dA,
                     lda,
                     dB,
                     ldb));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasTrsmFn,
                    (handle,
                     side,
                     uplo,
                     transA,
                     (hipblasDiagType_t)HIPBLAS_FILL_MODE_FULL,
                     M,
                     N,
                     alpha,
                     dA,
                     lda,
                     dB,
                     ldb));

        if(arg.bad_arg_all)
        {
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasTrsmFn,
                        (handle, side, uplo, transA, diag, M, N, nullptr, dA, lda, dB, ldb));

            if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
            {
                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasTrsmFn,
                            (handle, side, uplo, transA, diag, M, N, alpha, nullptr, lda, dB, ldb));
                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasTrsmFn,
                            (handle, side, uplo, transA, diag, M, N, alpha, dA, lda, nullptr, ldb));
            }

            // If alpha == 0, then A can be nullptr
            DAPI_CHECK(hipblasTrsmFn,
                       (handle, side, uplo, transA, diag, M, N, zero, nullptr, lda, dB, ldb));

            // trsm will quick-return with N == 0 || M == 0. Here, c_i32_overflow will rollover in the case of 32-bit params,
            // and quick-return with 64-bit params. This depends on implementation so only testing rocBLAS backend
            DAPI_EXPECT((arg.api & c_API_64) ? HIPBLAS_STATUS_SUCCESS
                                             : HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasTrsmFn,
                        (handle,
                         side,
                         uplo,
                         transA,
                         diag,
                         0,
                         c_i32_overflow,
                         nullptr,
                         nullptr,
                         c_i32_overflow,
                         nullptr,
                         c_i32_overflow));
            DAPI_EXPECT((arg.api & c_API_64) ? HIPBLAS_STATUS_SUCCESS
                                             : HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasTrsmFn,
                        (handle,
                         side,
                         uplo,
                         transA,
                         diag,
                         c_i32_overflow,
                         0,
                         nullptr,
                         nullptr,
                         c_i32_overflow,
                         nullptr,
                         c_i32_overflow));
        }

        // If M == 0 || N == 0, can have nullptrs
        DAPI_CHECK(hipblasTrsmFn,
                   (handle, side, uplo, transA, diag, 0, N, nullptr, nullptr, lda, nullptr, ldb));
        DAPI_CHECK(hipblasTrsmFn,
                   (handle, side, uplo, transA, diag, M, 0, nullptr, nullptr, lda, nullptr, ldb));
    }
}

template <typename T>
void testing_trsm(const Arguments& arg)
{
    using Ts           = hipblas_internal_type<T>;
    auto hipblasTrsmFn = arg.api == FORTRAN ? hipblasTrsm<T, true> : hipblasTrsm<T, false>;
    auto hipblasTrsmFn_64
        = arg.api == FORTRAN_64 ? hipblasTrsm_64<T, true> : hipblasTrsm_64<T, false>;

    hipblasSideMode_t  side   = char2hipblas_side(arg.side);
    hipblasFillMode_t  uplo   = char2hipblas_fill(arg.uplo);
    hipblasOperation_t transA = char2hipblas_operation(arg.transA);
    hipblasDiagType_t  diag   = char2hipblas_diagonal(arg.diag);
    int64_t            M      = arg.M;
    int64_t            N      = arg.N;
    int64_t            lda    = arg.lda;
    int64_t            ldb    = arg.ldb;

    T h_alpha = arg.get_alpha<T>();

    int64_t K = (side == HIPBLAS_SIDE_LEFT ? M : N);

    hipblasLocalHandle handle(arg);

    // check here to prevent undefined memory allocation error
    bool invalid_size = M < 0 || N < 0 || lda < K || ldb < M;
    if(invalid_size)
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasTrsmFn,
                    (handle, side, uplo, transA, diag, M, N, nullptr, nullptr, lda, nullptr, ldb));

        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_matrix<T> hA(K, K, lda);
    host_matrix<T> hB_host(M, N, ldb);
    host_matrix<T> hB_device(M, N, ldb);
    host_matrix<T> hB_cpu(M, N, ldb);

    // Allocate device memory
    device_matrix<T> dA(K, K, lda);
    device_matrix<T> dB(M, N, ldb);
    device_vector<T> d_alpha(1, 1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());

    double gpu_time_used{0}, hipblas_error_host{0}, hipblas_error_device{0};

    // Initial data on CPU
    hipblas_init_matrix(
        hA, arg, hipblas_client_never_set_nan, hipblas_diagonally_dominant_triangular_matrix, true);
    hipblas_init_matrix(
        hB_host, arg, hipblas_client_never_set_nan, hipblas_general_matrix, false, true);

    //  make hA unit diagonal if diag == HIPBLAS_DIAG_UNIT
    if(diag == HIPBLAS_DIAG_UNIT)
    {
        make_unit_diagonal(uplo, (T*)hA, lda, K);
    }

    // Calculate hB = hA*hX;
    ref_trmm<T>(side, uplo, transA, diag, M, N, T(1.0) / h_alpha, (const T*)hA, lda, hB_host, ldb);

    hB_cpu    = hB_host; // original solution hX
    hB_device = hB_host;

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB_host));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    if(arg.unit_check || arg.norm_check)
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        DAPI_CHECK(hipblasTrsmFn,
                   (handle,
                    side,
                    uplo,
                    transA,
                    diag,
                    M,
                    N,
                    reinterpret_cast<Ts*>(&h_alpha),
                    dA,
                    lda,
                    dB,
                    ldb));

        CHECK_HIP_ERROR(hB_host.transfer_from(dB));
        CHECK_HIP_ERROR(dB.transfer_from(hB_device));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasTrsmFn,
                   (handle, side, uplo, transA, diag, M, N, d_alpha, dA, lda, dB, ldb));

        CHECK_HIP_ERROR(hB_device.transfer_from(dB));

        /* =====================================================================
           CPU BLAS
        =================================================================== */

        ref_trsm<T>(side, uplo, transA, diag, M, N, h_alpha, (const T*)hA, lda, hB_cpu, ldb);

        // if enable norm check, norm check is invasive
        real_t<T> eps       = std::numeric_limits<real_t<T>>::epsilon();
        double    tolerance = eps * 40 * M;

        hipblas_error_host   = norm_check_general<T>('F', M, N, ldb, hB_cpu, hB_host);
        hipblas_error_device = norm_check_general<T>('F', M, N, ldb, hB_cpu, hB_device);
        if(arg.unit_check)
        {
            unit_check_error(hipblas_error_host, tolerance);
            unit_check_error(hipblas_error_device, tolerance);
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

            DAPI_DISPATCH(hipblasTrsmFn,
                          (handle, side, uplo, transA, diag, M, N, d_alpha, dA, lda, dB, ldb));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasTrsmModel{}.log_args<T>(std::cout,
                                       arg,
                                       gpu_time_used,
                                       trsm_gflop_count<T>(M, N, K),
                                       trsm_gbyte_count<T>(M, N, K),
                                       hipblas_error_host,
                                       hipblas_error_device);
    }
}
