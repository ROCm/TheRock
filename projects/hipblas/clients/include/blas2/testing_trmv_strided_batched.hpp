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

using hipblasTrmvStridedBatchedModel = ArgumentModel<e_a_type,
                                                     e_uplo,
                                                     e_transA,
                                                     e_diag,
                                                     e_N,
                                                     e_lda,
                                                     e_incx,
                                                     e_stride_scale,
                                                     e_batch_count>;

inline void testname_trmv_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasTrmvStridedBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_trmv_strided_batched_bad_arg(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasTrmvStridedBatchedFn
        = FORTRAN ? hipblasTrmvStridedBatched<T, true> : hipblasTrmvStridedBatched<T, false>;

    auto hipblasTrmvStridedBatchedFn_64 = arg.api == FORTRAN_64
                                              ? hipblasTrmvStridedBatched_64<T, true>
                                              : hipblasTrmvStridedBatched_64<T, false>;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        hipblasLocalHandle handle(arg);
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        hipblasOperation_t transA      = HIPBLAS_OP_N;
        hipblasFillMode_t  uplo        = HIPBLAS_FILL_MODE_UPPER;
        hipblasDiagType_t  diag        = HIPBLAS_DIAG_NON_UNIT;
        int64_t            N           = 100;
        int64_t            lda         = 100;
        int64_t            incx        = 1;
        int64_t            batch_count = 2;
        hipblasStride      stride_A    = N * lda;
        hipblasStride      stride_x    = N * incx;

        // Allocate device memory
        device_strided_batch_matrix<T> dA(N, N, lda, stride_A, batch_count);
        device_strided_batch_vector<T> dx(N, incx, stride_x, batch_count);

        DAPI_EXPECT(
            HIPBLAS_STATUS_NOT_INITIALIZED,
            hipblasTrmvStridedBatchedFn,
            (nullptr, uplo, transA, diag, N, dA, lda, stride_A, dx, incx, stride_x, batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasTrmvStridedBatchedFn,
                    (handle,
                     HIPBLAS_FILL_MODE_FULL,
                     transA,
                     diag,
                     N,
                     dA,
                     lda,
                     stride_A,
                     dx,
                     incx,
                     stride_x,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasTrmvStridedBatchedFn,
                    (handle,
                     (hipblasFillMode_t)HIPBLAS_OP_N,
                     transA,
                     diag,
                     N,
                     dA,
                     lda,
                     stride_A,
                     dx,
                     incx,
                     stride_x,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasTrmvStridedBatchedFn,
                    (handle,
                     uplo,
                     (hipblasOperation_t)HIPBLAS_FILL_MODE_FULL,
                     diag,
                     N,
                     dA,
                     lda,
                     stride_A,
                     dx,
                     incx,
                     stride_x,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasTrmvStridedBatchedFn,
                    (handle,
                     uplo,
                     transA,
                     (hipblasDiagType_t)HIPBLAS_FILL_MODE_FULL,
                     N,
                     dA,
                     lda,
                     stride_A,
                     dx,
                     incx,
                     stride_x,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasTrmvStridedBatchedFn,
                    (handle,
                     uplo,
                     transA,
                     diag,
                     N,
                     nullptr,
                     lda,
                     stride_A,
                     dx,
                     incx,
                     stride_x,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasTrmvStridedBatchedFn,
                    (handle,
                     uplo,
                     transA,
                     diag,
                     N,
                     dA,
                     lda,
                     stride_A,
                     nullptr,
                     incx,
                     stride_x,
                     batch_count));

        // With N == 0, can have all nullptrs
        DAPI_CHECK(hipblasTrmvStridedBatchedFn,
                   (handle,
                    uplo,
                    transA,
                    diag,
                    0,
                    nullptr,
                    lda,
                    stride_A,
                    nullptr,
                    incx,
                    stride_x,
                    batch_count));
        DAPI_CHECK(
            hipblasTrmvStridedBatchedFn,
            (handle, uplo, transA, diag, N, nullptr, lda, stride_A, nullptr, incx, stride_x, 0));
    }
}

template <typename T>
void testing_trmv_strided_batched(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasTrmvStridedBatchedFn
        = FORTRAN ? hipblasTrmvStridedBatched<T, true> : hipblasTrmvStridedBatched<T, false>;

    auto hipblasTrmvStridedBatchedFn_64 = arg.api == FORTRAN_64
                                              ? hipblasTrmvStridedBatched_64<T, true>
                                              : hipblasTrmvStridedBatched_64<T, false>;

    hipblasFillMode_t  uplo         = char2hipblas_fill(arg.uplo);
    hipblasOperation_t transA       = char2hipblas_operation(arg.transA);
    hipblasDiagType_t  diag         = char2hipblas_diagonal(arg.diag);
    int64_t            N            = arg.N;
    int64_t            lda          = arg.lda;
    int64_t            incx         = arg.incx;
    double             stride_scale = arg.stride_scale;
    int64_t            batch_count  = arg.batch_count;

    size_t        abs_incx = incx >= 0 ? incx : -incx;
    hipblasStride stride_A = lda * N * stride_scale;
    hipblasStride stride_x = N * abs_incx * stride_scale;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = N < 0 || lda < N || lda < 1 || !incx || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        DAPI_EXPECT(invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS,
                    hipblasTrmvStridedBatchedFn,
                    (handle,
                     uplo,
                     transA,
                     diag,
                     N,
                     nullptr,
                     lda,
                     stride_A,
                     nullptr,
                     incx,
                     stride_x,
                     batch_count));
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_strided_batch_matrix<T> hA(N, N, lda, stride_A, batch_count);
    host_strided_batch_vector<T> hx(N, incx, stride_x, batch_count);
    host_strided_batch_vector<T> hres(N, incx, stride_x, batch_count);

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hx.memcheck());
    CHECK_HIP_ERROR(hres.memcheck());

    // Allocate device memory
    device_strided_batch_matrix<T> dA(N, N, lda, stride_A, batch_count);
    device_strided_batch_vector<T> dx(N, incx, stride_x, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());

    double hipblas_error{0};

    // Initial Data on CPU
    hipblas_init_matrix(
        hA, arg, hipblas_client_never_set_nan, hipblas_triangular_matrix, true, false);
    hipblas_init_vector(hx, arg, hipblas_client_never_set_nan, false, true);

    // copy vector
    hres.copy_from(hx);

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        DAPI_CHECK(
            hipblasTrmvStridedBatchedFn,
            (handle, uplo, transA, diag, N, dA, lda, stride_A, dx, incx, stride_x, batch_count));

        CHECK_HIP_ERROR(hres.transfer_from(dx));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(size_t b = 0; b < batch_count; b++)
        {
            ref_trmv<T>(uplo, transA, diag, N, hA[b], lda, hx[b], incx);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, batch_count, abs_incx, stride_x, hx, hres);
        }
        if(arg.norm_check)
        {
            hipblas_error
                = norm_check_general<T>('F', 1, N, abs_incx, stride_x, hx, hres, batch_count);
        }
    }

    if(arg.timing)
    {
        double      gpu_time_used{0};
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(hipblasTrmvStridedBatchedFn,
                          (handle,
                           uplo,
                           transA,
                           diag,
                           N,
                           dA,
                           lda,
                           stride_A,
                           dx,
                           incx,
                           stride_x,
                           batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasTrmvStridedBatchedModel{}.log_args<T>(std::cout,
                                                     arg,
                                                     gpu_time_used,
                                                     trmv_gflop_count<T>(N),
                                                     trmv_gbyte_count<T>(N),
                                                     hipblas_error);
    }
}
