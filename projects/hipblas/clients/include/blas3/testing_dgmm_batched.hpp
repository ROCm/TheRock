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

using hipblasDgmmBatchedModel
    = ArgumentModel<e_a_type, e_side, e_M, e_N, e_lda, e_incx, e_ldc, e_batch_count>;

inline void testname_dgmm_batched(const Arguments& arg, std::string& name)
{
    hipblasDgmmBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_dgmm_batched_bad_arg(const Arguments& arg)
{
    auto hipblasDgmmBatchedFn
        = arg.api == FORTRAN ? hipblasDgmmBatched<T, true> : hipblasDgmmBatched<T, false>;
    auto hipblasDgmmBatchedFn_64
        = arg.api == FORTRAN_64 ? hipblasDgmmBatched_64<T, true> : hipblasDgmmBatched_64<T, false>;

    hipblasLocalHandle handle(arg);

    int64_t M           = 101;
    int64_t N           = 100;
    int64_t lda         = 102;
    int64_t incx        = 1;
    int64_t ldc         = 103;
    int64_t batch_count = 2;

    hipblasSideMode_t side = HIPBLAS_SIDE_LEFT;

    int64_t K = side == HIPBLAS_SIDE_LEFT ? M : N;

    // Allocate device memory
    device_batch_matrix<T> dA(M, N, lda, batch_count);
    device_batch_vector<T> dx(K, incx, batch_count);
    device_batch_matrix<T> dC(M, N, ldc, batch_count);

    DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                hipblasDgmmBatchedFn,
                (nullptr,
                 side,
                 M,
                 N,
                 dA.ptr_on_device(),
                 lda,
                 dx.ptr_on_device(),
                 incx,
                 dC.ptr_on_device(),
                 ldc,
                 batch_count));

    DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                hipblasDgmmBatchedFn,
                (handle,
                 (hipblasSideMode_t)HIPBLAS_FILL_MODE_FULL,
                 M,
                 N,
                 dA.ptr_on_device(),
                 lda,
                 dx.ptr_on_device(),
                 incx,
                 dC.ptr_on_device(),
                 ldc,
                 batch_count));

    if(arg.bad_arg_all)
    {
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasDgmmBatchedFn,
                    (handle,
                     side,
                     M,
                     N,
                     nullptr,
                     lda,
                     dx.ptr_on_device(),
                     incx,
                     dC.ptr_on_device(),
                     ldc,
                     batch_count));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasDgmmBatchedFn,
                    (handle,
                     side,
                     M,
                     N,
                     dA.ptr_on_device(),
                     lda,
                     nullptr,
                     incx,
                     dC.ptr_on_device(),
                     ldc,
                     batch_count));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasDgmmBatchedFn,
                    (handle,
                     side,
                     M,
                     N,
                     dA.ptr_on_device(),
                     lda,
                     dx.ptr_on_device(),
                     incx,
                     nullptr,
                     ldc,
                     batch_count));

        // 64-bit interface tests
        DAPI_EXPECT((arg.api & c_API_64) ? HIPBLAS_STATUS_SUCCESS : HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasDgmmBatchedFn,
                    (handle,
                     side,
                     0,
                     c_i32_overflow,
                     nullptr,
                     c_i32_overflow,
                     nullptr,
                     incx,
                     nullptr,
                     c_i32_overflow,
                     c_i32_overflow));
        DAPI_EXPECT((arg.api & c_API_64) ? HIPBLAS_STATUS_SUCCESS : HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasDgmmBatchedFn,
                    (handle,
                     side,
                     c_i32_overflow,
                     0,
                     nullptr,
                     c_i32_overflow,
                     nullptr,
                     incx,
                     nullptr,
                     c_i32_overflow,
                     c_i32_overflow));
    }

    // If M == 0 || N == 0 || batch_count == 0, can have all nullptrs
    DAPI_CHECK(hipblasDgmmBatchedFn,
               (handle, side, 0, N, nullptr, lda, nullptr, incx, nullptr, ldc, batch_count));
    DAPI_CHECK(hipblasDgmmBatchedFn,
               (handle, side, M, 0, nullptr, lda, nullptr, incx, nullptr, ldc, batch_count));
    DAPI_CHECK(hipblasDgmmBatchedFn,
               (handle, side, M, N, nullptr, lda, nullptr, incx, nullptr, ldc, 0));
}

template <typename T>
void testing_dgmm_batched(const Arguments& arg)
{
    auto hipblasDgmmBatchedFn
        = arg.api == FORTRAN ? hipblasDgmmBatched<T, true> : hipblasDgmmBatched<T, false>;
    auto hipblasDgmmBatchedFn_64
        = arg.api == FORTRAN_64 ? hipblasDgmmBatched_64<T, true> : hipblasDgmmBatched_64<T, false>;

    hipblasSideMode_t side = char2hipblas_side(arg.side);

    int64_t M           = arg.M;
    int64_t N           = arg.N;
    int64_t lda         = arg.lda;
    int64_t incx        = arg.incx;
    int64_t ldc         = arg.ldc;
    int64_t batch_count = arg.batch_count;
    int64_t K           = (side == HIPBLAS_SIDE_RIGHT ? N : M);

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = M < 0 || N < 0 || ldc < M || lda < M || batch_count < 0;
    if(invalid_size || !N || !M || !batch_count)
    {
        DAPI_EXPECT((invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS),
                    hipblasDgmmBatchedFn,
                    (handle, side, M, N, nullptr, lda, nullptr, incx, nullptr, ldc, batch_count));
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_batch_matrix<T> hA(M, N, lda, batch_count);
    host_batch_vector<T> hx(K, incx, batch_count);
    host_batch_matrix<T> hC(M, N, ldc, batch_count);
    host_batch_matrix<T> hC_gold(M, N, ldc, batch_count);

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hx.memcheck());
    CHECK_HIP_ERROR(hC.memcheck());
    CHECK_HIP_ERROR(hC_gold.memcheck());

    // Allocate device memory
    device_batch_matrix<T> dA(M, N, lda, batch_count);
    device_batch_vector<T> dx(K, incx, batch_count);
    device_batch_matrix<T> dC(M, N, ldc, batch_count);

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
        DAPI_CHECK(hipblasDgmmBatchedFn,
                   (handle,
                    side,
                    M,
                    N,
                    dA.ptr_on_device(),
                    lda,
                    dx.ptr_on_device(),
                    incx,
                    dC.ptr_on_device(),
                    ldc,
                    batch_count));
        CHECK_HIP_ERROR(hC.transfer_from(dC));

        /* =====================================================================
           CPU BLAS
        =================================================================== */

        // reference calculation
        ptrdiff_t shift_x = incx < 0 ? -ptrdiff_t(incx) * (N - 1) : 0;
        for(int64_t b = 0; b < batch_count; b++)
        {
            ref_dgmm<T>(side, M, N, hA[b], lda, hx[b], incx, hC_gold[b], ldc);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(M, N, batch_count, ldc, hC_gold, hC);
        }

        if(arg.norm_check)
        {
            hipblas_error = norm_check_general<T>('F', M, N, ldc, hC_gold, hC, batch_count);
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

            DAPI_DISPATCH(hipblasDgmmBatchedFn,
                          (handle,
                           side,
                           M,
                           N,
                           dA.ptr_on_device(),
                           lda,
                           dx.ptr_on_device(),
                           incx,
                           dC.ptr_on_device(),
                           ldc,
                           batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        hipblasDgmmBatchedModel{}.log_args<T>(std::cout,
                                              arg,
                                              gpu_time_used,
                                              dgmm_gflop_count<T>(M, N),
                                              dgmm_gbyte_count<T>(M, N, K),
                                              hipblas_error);
    }
}
