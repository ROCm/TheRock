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

using hipblasTrmmBatchedModel = ArgumentModel<e_a_type,
                                              e_side,
                                              e_uplo,
                                              e_transA,
                                              e_diag,
                                              e_M,
                                              e_N,
                                              e_alpha,
                                              e_lda,
                                              e_ldb,
                                              e_batch_count>;

inline void testname_trmm_batched(const Arguments& arg, std::string& name)
{
    hipblasTrmmBatchedModel{}.test_name(arg, name);
}

template <typename T>
inline void testing_trmm_batched_bad_arg(const Arguments& arg)
{
    using Ts                     = hipblas_internal_type<T>;
    auto hipblasTrmmBatchedFn    = arg.api == hipblas_client_api::FORTRAN
                                       ? hipblasTrmmBatched<T, true>
                                       : hipblasTrmmBatched<T, false>;
    auto hipblasTrmmBatchedFn_64 = arg.api == hipblas_client_api::FORTRAN_64
                                       ? hipblasTrmmBatched_64<T, true>
                                       : hipblasTrmmBatched_64<T, false>;
    bool inplace                 = arg.inplace;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        hipblasSideMode_t  side        = HIPBLAS_SIDE_LEFT;
        hipblasFillMode_t  uplo        = HIPBLAS_FILL_MODE_LOWER;
        hipblasOperation_t transA      = HIPBLAS_OP_N;
        hipblasDiagType_t  diag        = HIPBLAS_DIAG_NON_UNIT;
        int64_t            M           = 100;
        int64_t            N           = 101;
        int64_t            lda         = 102;
        int64_t            ldb         = 103;
        int64_t            ldc         = 104;
        int64_t            batch_count = 2;
        int64_t            ldOut       = inplace ? ldb : ldc;
        int64_t            K           = side == HIPBLAS_SIDE_LEFT ? M : N;

        device_vector<T> alpha_d(1), zero_d(1);

        const Ts alpha_h{1}, zero_h{0};

        const Ts* alpha = &alpha_h;
        const Ts* zero  = &zero_h;

        hipblasLocalHandle handle(arg);
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        if(pointer_mode == HIPBLAS_POINTER_MODE_DEVICE)
        {
            CHECK_HIP_ERROR(hipMemcpy(alpha_d, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            alpha = alpha_d;
            CHECK_HIP_ERROR(hipMemcpy(zero_d, zero, sizeof(*zero), hipMemcpyHostToDevice));
            zero = zero_d;
        }

        // Allocate device memory
        device_batch_matrix<T> dA(K, K, lda, batch_count);
        device_batch_matrix<T> dB(M, N, ldb, batch_count);

        int64_t dC_M   = inplace ? 1 : M;
        int64_t dC_N   = inplace ? 1 : N;
        int64_t dC_ldc = inplace ? 1 : ldc;

        device_batch_matrix<T> dC(dC_M, dC_N, dC_ldc, batch_count);

        device_batch_matrix<T>* dOut = inplace ? &dB : &dC;

        // invalid enums
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasTrmmBatchedFn,
                    (handle,
                     HIPBLAS_SIDE_BOTH,
                     uplo,
                     transA,
                     diag,
                     M,
                     N,
                     alpha,
                     dA.ptr_on_device(),
                     lda,
                     dB.ptr_on_device(),
                     ldb,
                     (*dOut).ptr_on_device(),
                     ldOut,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasTrmmBatchedFn,
                    (handle,
                     side,
                     HIPBLAS_FILL_MODE_FULL,
                     transA,
                     diag,
                     M,
                     N,
                     alpha,
                     dA.ptr_on_device(),
                     lda,
                     dB.ptr_on_device(),
                     ldb,
                     (*dOut).ptr_on_device(),
                     ldOut,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasTrmmBatchedFn,
                    (handle,
                     side,
                     uplo,
                     (hipblasOperation_t)HIPBLAS_SIDE_BOTH,
                     diag,
                     M,
                     N,
                     alpha,
                     dA.ptr_on_device(),
                     lda,
                     dB.ptr_on_device(),
                     ldb,
                     (*dOut).ptr_on_device(),
                     ldOut,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasTrmmBatchedFn,
                    (handle,
                     side,
                     uplo,
                     transA,
                     (hipblasDiagType_t)HIPBLAS_SIDE_BOTH,
                     M,
                     N,
                     alpha,
                     dA.ptr_on_device(),
                     lda,
                     dB.ptr_on_device(),
                     ldb,
                     (*dOut).ptr_on_device(),
                     ldOut,
                     batch_count));

        // nullptr checks
        DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                    hipblasTrmmBatchedFn,
                    (nullptr,
                     side,
                     uplo,
                     transA,
                     diag,
                     M,
                     N,
                     alpha,
                     dA.ptr_on_device(),
                     lda,
                     dB.ptr_on_device(),
                     ldb,
                     (*dOut).ptr_on_device(),
                     ldOut,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasTrmmBatchedFn,
                    (handle,
                     side,
                     uplo,
                     transA,
                     diag,
                     M,
                     N,
                     nullptr,
                     dA.ptr_on_device(),
                     lda,
                     dB.ptr_on_device(),
                     ldb,
                     (*dOut).ptr_on_device(),
                     ldOut,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasTrmmBatchedFn,
                    (handle,
                     side,
                     uplo,
                     transA,
                     diag,
                     M,
                     N,
                     alpha,
                     nullptr,
                     lda,
                     dB.ptr_on_device(),
                     ldb,
                     (*dOut).ptr_on_device(),
                     ldOut,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasTrmmBatchedFn,
                    (handle,
                     side,
                     uplo,
                     transA,
                     diag,
                     M,
                     N,
                     alpha,
                     dA.ptr_on_device(),
                     lda,
                     nullptr,
                     ldb,
                     (*dOut).ptr_on_device(),
                     ldOut,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasTrmmBatchedFn,
                    (handle,
                     side,
                     uplo,
                     transA,
                     diag,
                     M,
                     N,
                     alpha,
                     dA.ptr_on_device(),
                     lda,
                     dB.ptr_on_device(),
                     ldb,
                     nullptr,
                     ldOut,
                     batch_count));

        // quick return: if alpha == 0, both A & B can be nullptr
        DAPI_EXPECT(HIPBLAS_STATUS_SUCCESS,
                    hipblasTrmmBatchedFn,
                    (handle,
                     side,
                     uplo,
                     transA,
                     diag,
                     M,
                     N,
                     zero,
                     nullptr,
                     lda,
                     nullptr,
                     ldb,
                     (*dOut).ptr_on_device(),
                     ldOut,
                     batch_count));

        // quick return: if M == 0, then all ptrs can be nullptr
        DAPI_EXPECT(HIPBLAS_STATUS_SUCCESS,
                    hipblasTrmmBatchedFn,
                    (handle,
                     side,
                     uplo,
                     transA,
                     diag,
                     0,
                     N,
                     nullptr,
                     nullptr,
                     lda,
                     nullptr,
                     ldb,
                     nullptr,
                     ldOut,
                     batch_count));

        // quick return: if N == 0, then all ptrs can be nullptr
        DAPI_EXPECT(HIPBLAS_STATUS_SUCCESS,
                    hipblasTrmmBatchedFn,
                    (handle,
                     side,
                     uplo,
                     transA,
                     diag,
                     M,
                     0,
                     nullptr,
                     nullptr,
                     lda,
                     nullptr,
                     ldb,
                     nullptr,
                     ldOut,
                     batch_count));

        // quick return: if batch_count == 0, then all ptrs can be nullptr
        DAPI_EXPECT(HIPBLAS_STATUS_SUCCESS,
                    hipblasTrmmBatchedFn,
                    (handle,
                     side,
                     uplo,
                     transA,
                     diag,
                     M,
                     N,
                     nullptr,
                     nullptr,
                     lda,
                     nullptr,
                     ldb,
                     nullptr,
                     ldOut,
                     0));

        // 64-bit interface tests
        DAPI_EXPECT((arg.api & c_API_64) ? HIPBLAS_STATUS_SUCCESS : HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasTrmmBatchedFn,
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
                     c_i32_overflow,
                     nullptr,
                     c_i32_overflow,
                     c_i32_overflow));

        DAPI_EXPECT((arg.api & c_API_64) ? HIPBLAS_STATUS_SUCCESS : HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasTrmmBatchedFn,
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
                     c_i32_overflow,
                     nullptr,
                     c_i32_overflow,
                     c_i32_overflow));

        // in-place only checks
        if(inplace)
        {
            // if inplace, must have ldb == ldc
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasTrmmBatchedFn,
                        (handle,
                         side,
                         uplo,
                         transA,
                         diag,
                         M,
                         N,
                         alpha,
                         dA.ptr_on_device(),
                         lda,
                         dB.ptr_on_device(),
                         ldb,
                         (*dOut).ptr_on_device(),
                         ldb + 1,
                         batch_count));
        }
    }
}

template <typename T>
void testing_trmm_batched(const Arguments& arg)
{
    using Ts                     = hipblas_internal_type<T>;
    auto hipblasTrmmBatchedFn    = arg.api == hipblas_client_api::FORTRAN
                                       ? hipblasTrmmBatched<T, true>
                                       : hipblasTrmmBatched<T, false>;
    auto hipblasTrmmBatchedFn_64 = arg.api == hipblas_client_api::FORTRAN_64
                                       ? hipblasTrmmBatched_64<T, true>
                                       : hipblasTrmmBatched_64<T, false>;
    bool inplace                 = arg.inplace;

    hipblasSideMode_t  side        = char2hipblas_side(arg.side);
    hipblasFillMode_t  uplo        = char2hipblas_fill(arg.uplo);
    hipblasOperation_t transA      = char2hipblas_operation(arg.transA);
    hipblasDiagType_t  diag        = char2hipblas_diagonal(arg.diag);
    int64_t            M           = arg.M;
    int64_t            N           = arg.N;
    int64_t            lda         = arg.lda;
    int64_t            ldb         = arg.ldb;
    int64_t            ldc         = arg.ldc;
    int64_t            ldOut       = inplace ? ldb : ldc;
    int64_t            batch_count = arg.batch_count;

    T h_alpha = arg.get_alpha<T>();

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    int64_t K = (side == HIPBLAS_SIDE_LEFT ? M : N);

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = M < 0 || N < 0 || lda < K || ldb < M || ldc < M || batch_count < 0;
    if(M == 0 || N == 0 || batch_count == 0 || invalid_size)
    {
        DAPI_EXPECT(invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS,
                    hipblasTrmmBatchedFn,
                    (handle,
                     side,
                     uplo,
                     transA,
                     diag,
                     M,
                     N,
                     nullptr,
                     nullptr,
                     lda,
                     nullptr,
                     ldb,
                     nullptr,
                     ldOut,
                     batch_count));
        return;
    }

    double gpu_time_used{0}, hipblas_error_host{0}, hipblas_error_device{0};

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_batch_matrix<T> hA(K, K, lda, batch_count);
    host_batch_matrix<T> hB(M, N, ldb, batch_count);
    host_batch_matrix<T> hC = (inplace) ? host_batch_matrix<T>(1, 1, 1, 1)
                                        : host_batch_matrix<T>(M, N, ldc, batch_count);
    host_batch_matrix<T> hOut_host(M, N, ldOut, batch_count);
    host_batch_matrix<T> hOut_device(M, N, ldOut, batch_count);
    host_batch_matrix<T> hOut_cpu(M, N, ldOut, batch_count);

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hB.memcheck());
    CHECK_HIP_ERROR(hC.memcheck());
    CHECK_HIP_ERROR(hOut_host.memcheck());
    CHECK_HIP_ERROR(hOut_device.memcheck());
    CHECK_HIP_ERROR(hOut_cpu.memcheck());

    // Allocate device memory
    device_batch_matrix<T> dA(K, K, lda, batch_count);
    device_batch_matrix<T> dB(M, N, ldb, batch_count);
    device_batch_matrix<T> dC = (inplace) ? device_batch_matrix<T>(1, 1, 1, 1)
                                          : device_batch_matrix<T>(M, N, ldc, batch_count);
    device_vector<T>       d_alpha(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());

    device_batch_matrix<T>* dOut = inplace ? &dB : &dC;

    // Initial Data on CPU
    hipblas_init_matrix(hA, arg, hipblas_client_alpha_sets_nan, hipblas_triangular_matrix, true);
    hipblas_init_matrix(
        hB, arg, hipblas_client_alpha_sets_nan, hipblas_general_matrix, false, true);

    if(!inplace)
        hipblas_init_matrix(
            hC, arg, hipblas_client_alpha_sets_nan, hipblas_general_matrix, false, true);

    hOut_host.copy_from(inplace ? hB : hC);
    hOut_device.copy_from(hOut_host);
    hOut_cpu.copy_from(hOut_host);

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        DAPI_CHECK(hipblasTrmmBatchedFn,
                   (handle,
                    side,
                    uplo,
                    transA,
                    diag,
                    M,
                    N,
                    reinterpret_cast<Ts*>(&h_alpha),
                    dA.ptr_on_device(),
                    lda,
                    dB.ptr_on_device(),
                    ldb,
                    (*dOut).ptr_on_device(),
                    ldOut,
                    batch_count));

        CHECK_HIP_ERROR(hOut_host.transfer_from(*dOut));
        CHECK_HIP_ERROR(dB.transfer_from(hB));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasTrmmBatchedFn,
                   (handle,
                    side,
                    uplo,
                    transA,
                    diag,
                    M,
                    N,
                    d_alpha,
                    dA.ptr_on_device(),
                    lda,
                    dB.ptr_on_device(),
                    ldb,
                    (*dOut).ptr_on_device(),
                    ldOut,
                    batch_count));

        CHECK_HIP_ERROR(hOut_device.transfer_from(*dOut));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(int64_t b = 0; b < batch_count; b++)
        {
            ref_trmm<T>(side, uplo, transA, diag, M, N, h_alpha, hA[b], lda, hB[b], ldb);
        }

        copy_matrix_with_different_leading_dimensions_batched(hB, hOut_cpu, M, N, ldb, ldOut);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(M, N, batch_count, ldOut, hOut_cpu, hOut_host);
            unit_check_general<T>(M, N, batch_count, ldOut, hOut_cpu, hOut_device);
        }
        if(arg.norm_check)
        {
            hipblas_error_host
                = norm_check_general<T>('F', M, N, ldOut, hOut_cpu, hOut_host, batch_count);
            hipblas_error_device
                = norm_check_general<T>('F', M, N, ldOut, hOut_cpu, hOut_device, batch_count);
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

            DAPI_DISPATCH(hipblasTrmmBatchedFn,
                          (handle,
                           side,
                           uplo,
                           transA,
                           diag,
                           M,
                           N,
                           d_alpha,
                           dA.ptr_on_device(),
                           lda,
                           dB.ptr_on_device(),
                           ldb,
                           (*dOut).ptr_on_device(),
                           ldOut,
                           batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasTrmmBatchedModel{}.log_args<T>(std::cout,
                                              arg,
                                              gpu_time_used,
                                              trmm_gflop_count<T>(M, N, K),
                                              trmm_gbyte_count<T>(M, N, K),
                                              hipblas_error_host,
                                              hipblas_error_device);
    }
}
