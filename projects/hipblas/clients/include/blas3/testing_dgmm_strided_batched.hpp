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

/* ============================================================================================ */

using hipblasDgmmStridedBatchedModel = ArgumentModel<e_a_type,
                                                     e_side,
                                                     e_M,
                                                     e_N,
                                                     e_lda,
                                                     e_incx,
                                                     e_ldc,
                                                     e_stride_scale,
                                                     e_batch_count>;

inline void testname_dgmm_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasDgmmStridedBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_dgmm_strided_batched_bad_arg(const Arguments& arg)
{
    auto hipblasDgmmStridedBatchedFn    = arg.api == FORTRAN ? hipblasDgmmStridedBatched<T, true>
                                                             : hipblasDgmmStridedBatched<T, false>;
    auto hipblasDgmmStridedBatchedFn_64 = arg.api == FORTRAN_64
                                              ? hipblasDgmmStridedBatched_64<T, true>
                                              : hipblasDgmmStridedBatched_64<T, false>;

    hipblasLocalHandle handle(arg);

    int64_t M           = 101;
    int64_t N           = 100;
    int64_t lda         = 102;
    int64_t incx        = 1;
    int64_t ldc         = 103;
    int64_t batch_count = 2;

    hipblasSideMode_t side = HIPBLAS_SIDE_LEFT;

    int64_t K = side == HIPBLAS_SIDE_LEFT ? M : N;

    hipblasStride stride_A = N * lda;
    hipblasStride stride_x = K * incx;
    hipblasStride stride_C = N * ldc;

    // Allocate device memory
    device_strided_batch_matrix<T> dA(M, N, lda, stride_A, batch_count);
    device_strided_batch_vector<T> dx(K, incx, stride_x, batch_count);
    device_strided_batch_matrix<T> dC(M, N, ldc, stride_C, batch_count);

    DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                hipblasDgmmStridedBatchedFn,
                (nullptr,
                 side,
                 M,
                 N,
                 dA,
                 lda,
                 stride_A,
                 dx,
                 incx,
                 stride_x,
                 dC,
                 ldc,
                 stride_C,
                 batch_count));

    DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                hipblasDgmmStridedBatchedFn,
                (handle,
                 (hipblasSideMode_t)HIPBLAS_FILL_MODE_FULL,
                 M,
                 N,
                 dA,
                 lda,
                 stride_A,
                 dx,
                 incx,
                 stride_x,
                 dC,
                 ldc,
                 stride_C,
                 batch_count));

    if(arg.bad_arg_all)
    {
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasDgmmStridedBatchedFn,
                    (handle,
                     side,
                     M,
                     N,
                     nullptr,
                     lda,
                     stride_A,
                     dx,
                     incx,
                     stride_x,
                     dC,
                     ldc,
                     stride_C,
                     batch_count));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasDgmmStridedBatchedFn,
                    (handle,
                     side,
                     M,
                     N,
                     dA,
                     lda,
                     stride_A,
                     nullptr,
                     incx,
                     stride_x,
                     dC,
                     ldc,
                     stride_C,
                     batch_count));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasDgmmStridedBatchedFn,
                    (handle,
                     side,
                     M,
                     N,
                     dA,
                     lda,
                     stride_A,
                     dx,
                     incx,
                     stride_x,
                     nullptr,
                     ldc,
                     stride_C,
                     batch_count));

        // 64-bit interface tests
        DAPI_EXPECT((arg.api & c_API_64) ? HIPBLAS_STATUS_SUCCESS : HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasDgmmStridedBatchedFn,
                    (handle,
                     side,
                     0,
                     c_i32_overflow,
                     nullptr,
                     c_i32_overflow,
                     stride_A,
                     nullptr,
                     incx,
                     stride_x,
                     nullptr,
                     c_i32_overflow,
                     stride_C,
                     c_i32_overflow));
        DAPI_EXPECT((arg.api & c_API_64) ? HIPBLAS_STATUS_SUCCESS : HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasDgmmStridedBatchedFn,
                    (handle,
                     side,
                     c_i32_overflow,
                     0,
                     nullptr,
                     c_i32_overflow,
                     stride_A,
                     nullptr,
                     incx,
                     stride_x,
                     nullptr,
                     c_i32_overflow,
                     stride_C,
                     c_i32_overflow));
    }

    // If M == 0 || N == 0 || batch_count == 0, can have nullptrs
    DAPI_CHECK(hipblasDgmmStridedBatchedFn,
               (handle,
                side,
                0,
                N,
                nullptr,
                lda,
                stride_A,
                nullptr,
                incx,
                stride_x,
                nullptr,
                ldc,
                stride_C,
                batch_count));
    DAPI_CHECK(hipblasDgmmStridedBatchedFn,
               (handle,
                side,
                M,
                0,
                nullptr,
                lda,
                stride_A,
                nullptr,
                incx,
                stride_x,
                nullptr,
                ldc,
                stride_C,
                batch_count));
    DAPI_CHECK(hipblasDgmmStridedBatchedFn,
               (handle,
                side,
                M,
                N,
                nullptr,
                lda,
                stride_A,
                nullptr,
                incx,
                stride_x,
                nullptr,
                ldc,
                stride_C,
                0));
}

template <typename T>
void testing_dgmm_strided_batched(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasDgmmStridedBatchedFn
        = FORTRAN ? hipblasDgmmStridedBatched<T, true> : hipblasDgmmStridedBatched<T, false>;
    auto hipblasDgmmStridedBatchedFn_64 = arg.api == FORTRAN_64
                                              ? hipblasDgmmStridedBatched_64<T, true>
                                              : hipblasDgmmStridedBatched_64<T, false>;

    hipblasSideMode_t side = char2hipblas_side(arg.side);

    int64_t M            = arg.M;
    int64_t N            = arg.N;
    int64_t lda          = arg.lda;
    int64_t incx         = arg.incx;
    int64_t ldc          = arg.ldc;
    int64_t batch_count  = arg.batch_count;
    double  stride_scale = arg.stride_scale;
    int64_t K            = (side == HIPBLAS_SIDE_RIGHT ? N : M);

    int64_t       abs_incx = incx >= 0 ? incx : -incx;
    hipblasStride stride_A = lda * N * stride_scale;
    hipblasStride stride_x = abs_incx * K * stride_scale;
    hipblasStride stride_C = ldc * N * stride_scale;
    if(!stride_x)
        stride_x = 1;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = M < 0 || N < 0 || ldc < M || lda < M || batch_count < 0;
    if(invalid_size || !N || !M || !batch_count)
    {
        DAPI_EXPECT((invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS),
                    hipblasDgmmStridedBatchedFn,
                    (handle,
                     side,
                     M,
                     N,
                     nullptr,
                     lda,
                     stride_A,
                     nullptr,
                     incx,
                     stride_x,
                     nullptr,
                     ldc,
                     stride_C,
                     batch_count));
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_strided_batch_matrix<T> hA(M, N, lda, stride_A, batch_count);
    host_strided_batch_vector<T> hx(K, incx, stride_x, batch_count);
    host_strided_batch_matrix<T> hC(M, N, ldc, stride_C, batch_count);
    host_strided_batch_matrix<T> hC_gold(M, N, ldc, stride_C, batch_count);

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hx.memcheck());
    CHECK_HIP_ERROR(hC.memcheck());
    CHECK_HIP_ERROR(hC_gold.memcheck());

    // Allocate device memory
    device_strided_batch_matrix<T> dA(M, N, lda, stride_A, batch_count);
    device_strided_batch_vector<T> dx(K, incx, stride_x, batch_count);
    device_strided_batch_matrix<T> dC(M, N, ldc, stride_C, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());

    double gpu_time_used{0}, hipblas_error{0};

    // Initial Data on CPU
    hipblas_init_matrix(hA, arg, hipblas_client_never_set_nan, hipblas_general_matrix, true);
    hipblas_init_vector(hx, arg, hipblas_client_never_set_nan, false, true);
    hipblas_init_matrix(hC, arg, hipblas_client_never_set_nan, hipblas_general_matrix);

    hC_gold.copy_from(hC_gold);

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dC.transfer_from(hC));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        DAPI_CHECK(hipblasDgmmStridedBatchedFn,
                   (handle,
                    side,
                    M,
                    N,
                    dA,
                    lda,
                    stride_A,
                    dx,
                    incx,
                    stride_x,
                    dC,
                    ldc,
                    stride_C,
                    batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hC.transfer_from(dC));

        /* =====================================================================
           CPU BLAS
        =================================================================== */

        // reference calculation
        ptrdiff_t shift_x = incx < 0 ? -ptrdiff_t(incx) * (N - 1) : 0;
        for(int64_t b = 0; b < batch_count; b++)
        {
            ref_dgmm<T>(side,
                        M,
                        N,
                        hA + b * stride_A,
                        lda,
                        hx + b * stride_x,
                        incx,
                        hC_gold + b * stride_C,
                        ldc);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(M, N, batch_count, ldc, stride_C, hC_gold, hC);
        }

        if(arg.norm_check)
        {
            hipblas_error
                = norm_check_general<T>('F', M, N, ldc, stride_C, hC_gold, hC, batch_count);
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

            DAPI_DISPATCH(hipblasDgmmStridedBatchedFn,
                          (handle,
                           side,
                           M,
                           N,
                           dA,
                           lda,
                           stride_A,
                           dx,
                           incx,
                           stride_x,
                           dC,
                           ldc,
                           stride_C,
                           batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        hipblasDgmmStridedBatchedModel{}.log_args<T>(std::cout,
                                                     arg,
                                                     gpu_time_used,
                                                     dgmm_gflop_count<T>(M, N),
                                                     dgmm_gbyte_count<T>(M, N, K),
                                                     hipblas_error);
    }
}
