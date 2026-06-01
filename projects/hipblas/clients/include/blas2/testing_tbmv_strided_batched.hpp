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

using hipblasTbmvStridedBatchedModel = ArgumentModel<e_a_type,
                                                     e_uplo,
                                                     e_transA,
                                                     e_diag,
                                                     e_M,
                                                     e_K,
                                                     e_lda,
                                                     e_incx,
                                                     e_stride_scale,
                                                     e_batch_count>;

inline void testname_tbmv_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasTbmvStridedBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_tbmv_strided_batched_bad_arg(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasTbmvStridedBatchedFn
        = FORTRAN ? hipblasTbmvStridedBatched<T, true> : hipblasTbmvStridedBatched<T, false>;

    auto hipblasTbmvStridedBatchedFn_64 = arg.api == FORTRAN_64
                                              ? hipblasTbmvStridedBatched_64<T, true>
                                              : hipblasTbmvStridedBatched_64<T, false>;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        hipblasLocalHandle handle(arg);
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        hipblasFillMode_t  uplo              = HIPBLAS_FILL_MODE_UPPER;
        hipblasOperation_t transA            = HIPBLAS_OP_N;
        hipblasDiagType_t  diag              = HIPBLAS_DIAG_NON_UNIT;
        int64_t            M                 = 100;
        int64_t            K                 = 5;
        int64_t            banded_matrix_row = K + 1;
        int64_t            lda               = 100;
        int64_t            incx              = 1;
        int64_t            batch_count       = 2;
        hipblasStride      stride_A          = M * lda;
        hipblasStride      stride_x          = M * incx;

        // Allocate device memory
        device_strided_batch_matrix<T> dAb(banded_matrix_row, M, lda, stride_A, batch_count);
        device_strided_batch_vector<T> dx(M, incx, stride_x, batch_count);

        DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                    hipblasTbmvStridedBatchedFn,
                    (nullptr,
                     uplo,
                     transA,
                     diag,
                     M,
                     K,
                     dAb,
                     lda,
                     stride_A,
                     dx,
                     incx,
                     stride_x,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasTbmvStridedBatchedFn,
                    (handle,
                     HIPBLAS_FILL_MODE_FULL,
                     transA,
                     diag,
                     M,
                     K,
                     dAb,
                     lda,
                     stride_A,
                     dx,
                     incx,
                     stride_x,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasTbmvStridedBatchedFn,
                    (handle,
                     (hipblasFillMode_t)HIPBLAS_OP_N,
                     transA,
                     diag,
                     M,
                     K,
                     dAb,
                     lda,
                     stride_A,
                     dx,
                     incx,
                     stride_x,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasTbmvStridedBatchedFn,
                    (handle,
                     uplo,
                     (hipblasOperation_t)HIPBLAS_FILL_MODE_FULL,
                     diag,
                     M,
                     K,
                     dAb,
                     lda,
                     stride_A,
                     dx,
                     incx,
                     stride_x,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasTbmvStridedBatchedFn,
                    (handle,
                     uplo,
                     transA,
                     (hipblasDiagType_t)HIPBLAS_FILL_MODE_FULL,
                     M,
                     K,
                     dAb,
                     lda,
                     stride_A,
                     dx,
                     incx,
                     stride_x,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasTbmvStridedBatchedFn,
                    (handle,
                     uplo,
                     transA,
                     diag,
                     M,
                     K,
                     nullptr,
                     lda,
                     stride_A,
                     dx,
                     incx,
                     stride_x,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasTbmvStridedBatchedFn,
                    (handle,
                     uplo,
                     transA,
                     diag,
                     M,
                     K,
                     dAb,
                     lda,
                     stride_A,
                     nullptr,
                     incx,
                     stride_x,
                     batch_count));

        // With M == 0, can have all nullptrs
        DAPI_CHECK(hipblasTbmvStridedBatchedFn,
                   (handle,
                    uplo,
                    transA,
                    diag,
                    0,
                    K,
                    nullptr,
                    lda,
                    stride_A,
                    nullptr,
                    incx,
                    stride_x,
                    batch_count));
        DAPI_CHECK(
            hipblasTbmvStridedBatchedFn,
            (handle, uplo, transA, diag, M, K, nullptr, lda, stride_A, nullptr, incx, stride_x, 0));
    }
}

template <typename T>
void testing_tbmv_strided_batched(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasTbmvStridedBatchedFn
        = FORTRAN ? hipblasTbmvStridedBatched<T, true> : hipblasTbmvStridedBatched<T, false>;

    auto hipblasTbmvStridedBatchedFn_64 = arg.api == FORTRAN_64
                                              ? hipblasTbmvStridedBatched_64<T, true>
                                              : hipblasTbmvStridedBatched_64<T, false>;

    hipblasFillMode_t  uplo              = char2hipblas_fill(arg.uplo);
    hipblasOperation_t transA            = char2hipblas_operation(arg.transA);
    hipblasDiagType_t  diag              = char2hipblas_diagonal(arg.diag);
    int64_t            M                 = arg.M;
    int64_t            K                 = arg.K;
    int64_t            lda               = arg.lda;
    int64_t            incx              = arg.incx;
    double             stride_scale      = arg.stride_scale;
    int64_t            batch_count       = arg.batch_count;
    const int64_t      banded_matrix_row = K + 1;

    size_t        abs_incx = incx >= 0 ? incx : -incx;
    hipblasStride stride_A = lda * M * stride_scale;
    hipblasStride stride_x = M * abs_incx * stride_scale;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = M < 0 || K < 0 || lda < K + 1 || !incx || batch_count < 0;
    if(invalid_size || !M || !batch_count)
    {
        DAPI_EXPECT(invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS,
                    hipblasTbmvStridedBatchedFn,
                    (handle,
                     uplo,
                     transA,
                     diag,
                     M,
                     K,
                     nullptr,
                     lda,
                     stride_A,
                     nullptr,
                     incx,
                     stride_x,
                     batch_count));
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hAb), `d` is in GPU (device) memory (eg dAb).
    // Allocate host memory
    host_strided_batch_matrix<T> hAb(banded_matrix_row, M, lda, stride_A, batch_count);
    host_strided_batch_vector<T> hx(M, incx, stride_x, batch_count);
    host_strided_batch_vector<T> hx_cpu(M, incx, stride_x, batch_count);
    host_strided_batch_vector<T> hx_res(M, incx, stride_x, batch_count);

    // Check host memory allocation
    CHECK_HIP_ERROR(hAb.memcheck());
    CHECK_HIP_ERROR(hx.memcheck());
    CHECK_HIP_ERROR(hx_cpu.memcheck());
    CHECK_HIP_ERROR(hx_res.memcheck());

    // Allocate device memory
    device_strided_batch_matrix<T> dAb(banded_matrix_row, M, lda, stride_A, batch_count);
    device_strided_batch_vector<T> dx(M, incx, stride_x, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dAb.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());

    double hipblas_error{0};

    // Initial Data on CPU
    hipblas_init_matrix(
        hAb, arg, hipblas_client_never_set_nan, hipblas_general_matrix, true, false);
    hipblas_init_vector(hx, arg, hipblas_client_never_set_nan, false, true);

    // copy vector
    hx_cpu.copy_from(hx);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dAb.transfer_from(hAb));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        DAPI_CHECK(hipblasTbmvStridedBatchedFn,
                   (handle,
                    uplo,
                    transA,
                    diag,
                    M,
                    K,
                    dAb,
                    lda,
                    stride_A,
                    dx,
                    incx,
                    stride_x,
                    batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hx_res.transfer_from(dx));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(size_t b = 0; b < batch_count; b++)
        {
            ref_tbmv<T>(uplo, transA, diag, M, K, hAb[b], lda, hx_cpu[b], incx);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(1, M, batch_count, abs_incx, stride_x, hx_cpu, hx_res);
        }
        if(arg.norm_check)
        {
            hipblas_error = norm_check_general<T>(
                'F', 1, M, abs_incx, stride_x, hx_cpu.data(), hx_res.data(), batch_count);
        }
    }

    if(arg.timing)
    {
        double gpu_time_used{0};
        CHECK_HIP_ERROR(dx.transfer_from(hx));

        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(hipblasTbmvStridedBatchedFn,
                          (handle,
                           uplo,
                           transA,
                           diag,
                           M,
                           K,
                           dAb,
                           lda,
                           stride_A,
                           dx,
                           incx,
                           stride_x,
                           batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasTbmvStridedBatchedModel{}.log_args<T>(std::cout,
                                                     arg,
                                                     gpu_time_used,
                                                     tbmv_gflop_count<T>(M, K),
                                                     tbmv_gbyte_count<T>(M, K),
                                                     hipblas_error);
    }
}
