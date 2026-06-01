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
#include <typeinfo>
#include <vector>

#include "hipblas_unique_ptr.hpp"
#include "testing_common.hpp"

/* ============================================================================================ */

using hipblasGemmStridedBatchedModel = ArgumentModel<e_a_type,
                                                     e_transA,
                                                     e_transB,
                                                     e_M,
                                                     e_N,
                                                     e_K,
                                                     e_alpha,
                                                     e_lda,
                                                     e_ldb,
                                                     e_beta,
                                                     e_ldc,
                                                     e_stride_scale,
                                                     e_batch_count>;

inline void testname_gemm_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasGemmStridedBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_gemm_strided_batched_bad_arg(const Arguments& arg)
{
    using Ts                            = hipblas_internal_type<T>;
    auto hipblasGemmStridedBatchedFn    = arg.api == FORTRAN ? hipblasGemmStridedBatched<T, true>
                                                             : hipblasGemmStridedBatched<T, false>;
    auto hipblasGemmStridedBatchedFn_64 = arg.api == FORTRAN_64
                                              ? hipblasGemmStridedBatched_64<T, true>
                                              : hipblasGemmStridedBatched_64<T, false>;

    hipblasLocalHandle handle(arg);

    int64_t M           = 101;
    int64_t N           = 100;
    int64_t K           = 102;
    int64_t lda         = 103;
    int64_t ldb         = 104;
    int64_t ldc         = 105;
    int64_t batch_count = 2;

    hipblasOperation_t transA = HIPBLAS_OP_N;
    hipblasOperation_t transB = HIPBLAS_OP_N;

    int64_t A_row = transA == HIPBLAS_OP_N ? M : std::max(K, int64_t(1));
    int64_t A_col = transA == HIPBLAS_OP_N ? std::max(K, int64_t(1)) : M;
    int64_t B_row = transB == HIPBLAS_OP_N ? std::max(K, int64_t(1)) : N;
    int64_t B_col = transB == HIPBLAS_OP_N ? N : std::max(K, int64_t(1));

    hipblasStride stride_A = A_col * lda;
    hipblasStride stride_B = B_row * ldb;
    hipblasStride stride_C = N * ldc;

    // Allocate device memory
    device_strided_batch_matrix<T> dA(A_row, A_col, lda, stride_A, batch_count);
    device_strided_batch_matrix<T> dB(B_row, B_col, ldb, stride_B, batch_count);
    device_strided_batch_matrix<T> dC(M, N, ldc, stride_C, batch_count);

    device_vector<T> d_alpha(1), d_beta(1), d_one(1), d_zero(1);
    Ts               h_alpha{1.0f}, h_beta{2.0f}, h_one{1.0f}, h_zero{0.0f};

    if constexpr(std::is_same_v<T, hipblasHalf>)
        h_one = float_to_half(1.0f);
    else if constexpr(is_complex<T>)
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
                    hipblasGemmStridedBatchedFn,
                    (nullptr,
                     transA,
                     transB,
                     M,
                     N,
                     K,
                     alpha,
                     dA,
                     lda,
                     stride_A,
                     dB,
                     ldb,
                     stride_B,
                     beta,
                     dC,
                     ldc,
                     stride_C,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasGemmStridedBatchedFn,
                    (handle,
                     (hipblasOperation_t)HIPBLAS_FILL_MODE_FULL,
                     transB,
                     M,
                     N,
                     K,
                     alpha,
                     dA,
                     lda,
                     stride_A,
                     dB,
                     ldb,
                     stride_B,
                     beta,
                     dC,
                     ldc,
                     stride_C,
                     batch_count));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasGemmStridedBatchedFn,
                    (handle,
                     transA,
                     (hipblasOperation_t)HIPBLAS_FILL_MODE_FULL,
                     M,
                     N,
                     K,
                     alpha,
                     dA,
                     lda,
                     stride_A,
                     dB,
                     ldb,
                     stride_B,
                     beta,
                     dC,
                     ldc,
                     stride_C,
                     batch_count));

        if(arg.bad_arg_all)
        {
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasGemmStridedBatchedFn,
                        (handle,
                         transA,
                         transB,
                         M,
                         N,
                         K,
                         alpha,
                         dA,
                         lda,
                         stride_A,
                         dB,
                         ldb,
                         stride_B,
                         nullptr,
                         dC,
                         ldc,
                         stride_C,
                         batch_count));

            if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
            {
                // alpha check only for host mode. rocBLAS can handle this in device mode too but shouldn't assume in case this changes.
                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasGemmStridedBatchedFn,
                            (handle,
                             transA,
                             transB,
                             M,
                             N,
                             K,
                             nullptr,
                             dA,
                             lda,
                             stride_A,
                             dB,
                             ldb,
                             stride_B,
                             beta,
                             dC,
                             ldc,
                             stride_C,
                             batch_count));

                // again, rocBLAS can handle this in device mode but shouldn't assume
                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasGemmStridedBatchedFn,
                            (handle,
                             transA,
                             transB,
                             M,
                             N,
                             K,
                             alpha,
                             nullptr,
                             lda,
                             stride_A,
                             dB,
                             ldb,
                             stride_B,
                             beta,
                             dC,
                             ldc,
                             stride_C,
                             batch_count));
                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasGemmStridedBatchedFn,
                            (handle,
                             transA,
                             transB,
                             M,
                             N,
                             K,
                             alpha,
                             dA,
                             lda,
                             stride_A,
                             nullptr,
                             ldb,
                             stride_B,
                             beta,
                             dC,
                             ldc,
                             stride_C,
                             batch_count));
                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasGemmStridedBatchedFn,
                            (handle,
                             transA,
                             transB,
                             M,
                             N,
                             K,
                             alpha,
                             dA,
                             lda,
                             stride_A,
                             dB,
                             ldb,
                             stride_B,
                             beta,
                             nullptr,
                             ldc,
                             stride_C,
                             batch_count));
            }

            // If alpha == 0 && beta == 1, can have A, B, C be nullptr
            DAPI_CHECK(hipblasGemmStridedBatchedFn,
                       (handle,
                        transA,
                        transB,
                        M,
                        N,
                        K,
                        zero,
                        nullptr,
                        lda,
                        stride_A,
                        nullptr,
                        ldb,
                        stride_B,
                        one,
                        nullptr,
                        ldc,
                        stride_C,
                        batch_count));

            // If alpha == 0, A and B can be nullptr
            DAPI_CHECK(hipblasGemmStridedBatchedFn,
                       (handle,
                        transA,
                        transB,
                        M,
                        N,
                        K,
                        zero,
                        nullptr,
                        lda,
                        stride_A,
                        nullptr,
                        ldb,
                        stride_B,
                        beta,
                        dC,
                        ldc,
                        stride_C,
                        batch_count));

            // If K == 0, alpha, A, and B can be nullptr
            DAPI_CHECK(hipblasGemmStridedBatchedFn,
                       (handle,
                        transA,
                        transB,
                        M,
                        N,
                        0,
                        nullptr,
                        nullptr,
                        lda,
                        stride_A,
                        nullptr,
                        ldb,
                        stride_B,
                        beta,
                        dC,
                        ldc,
                        stride_C,
                        batch_count));

            // 64-bit interface test
            DAPI_EXPECT((arg.api & c_API_64) ? HIPBLAS_STATUS_SUCCESS
                                             : HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasGemmStridedBatchedFn,
                        (handle,
                         transA,
                         transB,
                         c_i32_overflow,
                         c_i32_overflow,
                         c_i32_overflow,
                         zero,
                         nullptr,
                         c_i32_overflow,
                         stride_A,
                         nullptr,
                         c_i32_overflow,
                         stride_B,
                         one,
                         nullptr,
                         c_i32_overflow,
                         stride_C,
                         c_i32_overflow));
        }

        // If M == 0 || N == 0 || batch_count == 0, can have nullptrs
        DAPI_CHECK(hipblasGemmStridedBatchedFn,
                   (handle,
                    transA,
                    transB,
                    0,
                    N,
                    K,
                    nullptr,
                    nullptr,
                    lda,
                    stride_A,
                    nullptr,
                    lda,
                    stride_B,
                    nullptr,
                    nullptr,
                    lda,
                    stride_C,
                    batch_count));
        DAPI_CHECK(hipblasGemmStridedBatchedFn,
                   (handle,
                    transA,
                    transB,
                    M,
                    0,
                    K,
                    nullptr,
                    nullptr,
                    lda,
                    stride_A,
                    nullptr,
                    lda,
                    stride_B,
                    nullptr,
                    nullptr,
                    lda,
                    stride_C,
                    batch_count));
        DAPI_CHECK(hipblasGemmStridedBatchedFn,
                   (handle,
                    transA,
                    transB,
                    M,
                    N,
                    K,
                    nullptr,
                    nullptr,
                    lda,
                    stride_A,
                    nullptr,
                    lda,
                    stride_B,
                    nullptr,
                    nullptr,
                    lda,
                    stride_C,
                    0));
    }
}

template <typename T>
void testing_gemm_strided_batched(const Arguments& arg)
{
    using Ts                            = hipblas_internal_type<T>;
    auto hipblasGemmStridedBatchedFn    = arg.api == FORTRAN ? hipblasGemmStridedBatched<T, true>
                                                             : hipblasGemmStridedBatched<T, false>;
    auto hipblasGemmStridedBatchedFn_64 = arg.api == FORTRAN_64
                                              ? hipblasGemmStridedBatched_64<T, true>
                                              : hipblasGemmStridedBatched_64<T, false>;

    hipblasOperation_t transA       = char2hipblas_operation(arg.transA);
    hipblasOperation_t transB       = char2hipblas_operation(arg.transB);
    int64_t            M            = arg.M;
    int64_t            N            = arg.N;
    int64_t            K            = arg.K;
    int64_t            lda          = arg.lda;
    int64_t            ldb          = arg.ldb;
    int64_t            ldc          = arg.ldc;
    double             stride_scale = arg.stride_scale;
    int64_t            batch_count  = arg.batch_count;

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    hipblasLocalHandle handle(arg);

    int64_t A_row = transA == HIPBLAS_OP_N ? M : std::max(K, int64_t(1));
    int64_t A_col = transA == HIPBLAS_OP_N ? std::max(K, int64_t(1)) : M;
    int64_t B_row = transB == HIPBLAS_OP_N ? std::max(K, int64_t(1)) : N;
    int64_t B_col = transB == HIPBLAS_OP_N ? N : std::max(K, int64_t(1));

    hipblasStride stride_A = lda * A_col * stride_scale;
    hipblasStride stride_B = ldb * B_col * stride_scale;
    hipblasStride stride_C = ldc * N * stride_scale;

    // check here to prevent undefined memory allocation error
    bool invalid_size
        = M < 0 || N < 0 || K < 0 || batch_count < 0 || lda < A_row || ldb < B_row || ldc < M;
    if(invalid_size || !M || !N || !batch_count)
    {
        DAPI_EXPECT(invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS,
                    hipblasGemmStridedBatchedFn,
                    (handle,
                     transA,
                     transB,
                     M,
                     N,
                     K,
                     nullptr,
                     nullptr,
                     lda,
                     stride_A,
                     nullptr,
                     ldb,
                     stride_B,
                     nullptr,
                     nullptr,
                     ldc,
                     stride_C,
                     batch_count));
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_strided_batch_matrix<T> hA(A_row, A_col, lda, stride_A, batch_count);
    host_strided_batch_matrix<T> hB(B_row, B_col, ldb, stride_B, batch_count);
    host_strided_batch_matrix<T> hC_host(M, N, ldc, stride_C, batch_count);
    host_strided_batch_matrix<T> hC_device(M, N, ldc, stride_C, batch_count);
    host_strided_batch_matrix<T> hC_cpu(M, N, ldc, stride_C, batch_count);

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hB.memcheck());
    CHECK_HIP_ERROR(hC_host.memcheck());
    CHECK_HIP_ERROR(hC_device.memcheck());
    CHECK_HIP_ERROR(hC_cpu.memcheck());

    // Allocate device memory
    device_strided_batch_matrix<T> dA(A_row, A_col, lda, stride_A, batch_count);
    device_strided_batch_matrix<T> dB(B_row, B_col, ldb, stride_B, batch_count);
    device_strided_batch_matrix<T> dC(M, N, ldc, stride_C, batch_count);
    device_vector<T>               d_alpha(1);
    device_vector<T>               d_beta(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta.memcheck());

    // Initial Data on CPU
    hipblas_init_matrix(hA, arg, hipblas_client_alpha_sets_nan, hipblas_general_matrix, true);
    hipblas_init_matrix(
        hB, arg, hipblas_client_alpha_sets_nan, hipblas_general_matrix, false, true);
    hipblas_init_matrix(hC_host, arg, hipblas_client_beta_sets_nan, hipblas_general_matrix);

    // copy vector
    hC_device.copy_from(hC_host);
    hC_cpu.copy_from(hC_host);

    // copy data from CPU to device, does not work for lda != A_row
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));
    CHECK_HIP_ERROR(dC.transfer_from(hC_host));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    double gpu_time_used{0}, hipblas_error_host{0}, hipblas_error_device{0};

    /* =====================================================================
         HIPBLAS
    =================================================================== */
    if(arg.unit_check || arg.norm_check)
    {
        // host mode
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));

        // library interface
        DAPI_CHECK(hipblasGemmStridedBatchedFn,
                   (handle,
                    transA,
                    transB,
                    M,
                    N,
                    K,
                    reinterpret_cast<Ts*>(&h_alpha),
                    dA,
                    lda,
                    stride_A,
                    dB,
                    ldb,
                    stride_B,
                    reinterpret_cast<Ts*>(&h_beta),
                    dC,
                    ldc,
                    stride_C,
                    batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hC_host.transfer_from(dC));

        // device mode
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIP_ERROR(dC.transfer_from(hC_device));
        DAPI_CHECK(hipblasGemmStridedBatchedFn,
                   (handle,
                    transA,
                    transB,
                    M,
                    N,
                    K,
                    d_alpha,
                    dA,
                    lda,
                    stride_A,
                    dB,
                    ldb,
                    stride_B,
                    d_beta,
                    dC,
                    ldc,
                    stride_C,
                    batch_count));
        CHECK_HIP_ERROR(hC_device.transfer_from(dC));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int64_t b = 0; b < batch_count; b++)
        {
            ref_gemm<T>(
                transA, transB, M, N, K, h_alpha, hA[b], lda, hB[b], ldb, h_beta, hC_cpu[b], ldc);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            if(std::is_same_v<T, hipblasHalf> && (getArchMajor() == 11))
            {
                const double tol = K * sum_error_tolerance_for_gfx11<T, T, T>;
                near_check_general<T>(M, N, batch_count, ldc, stride_C, hC_cpu, hC_host, tol);
                near_check_general<T>(M, N, batch_count, ldc, stride_C, hC_cpu, hC_device, tol);
            }
            else
            {
                unit_check_general<T>(M, N, batch_count, ldc, stride_C, hC_cpu, hC_host);
                unit_check_general<T>(M, N, batch_count, ldc, stride_C, hC_cpu, hC_device);
            }
        }
        if(arg.norm_check)
        {
            hipblas_error_host
                = norm_check_general<T>('F', M, N, ldc, stride_C, hC_cpu, hC_host, batch_count);
            hipblas_error_device
                = norm_check_general<T>('F', M, N, ldc, stride_C, hC_cpu, hC_device, batch_count);
        }
    }

    if(arg.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));

        // gemm has better performance in host mode. In rocBLAS in device mode
        // we need to copy alpha and beta to the host.
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(hipblasGemmStridedBatchedFn,
                          (handle,
                           transA,
                           transB,
                           M,
                           N,
                           K,
                           reinterpret_cast<Ts*>(&h_alpha),
                           dA,
                           lda,
                           stride_A,
                           dB,
                           ldb,
                           stride_B,
                           reinterpret_cast<Ts*>(&h_beta),
                           dC,
                           ldc,
                           stride_C,
                           batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasGemmStridedBatchedModel{}.log_args<T>(std::cout,
                                                     arg,
                                                     gpu_time_used,
                                                     gemm_gflop_count<T>(M, N, K),
                                                     gemm_gbyte_count<T>(M, N, K),
                                                     hipblas_error_host,
                                                     hipblas_error_device);
    }
}
