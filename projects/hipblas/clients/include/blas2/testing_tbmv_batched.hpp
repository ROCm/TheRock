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

using hipblasTbmvBatchedModel
    = ArgumentModel<e_a_type, e_uplo, e_transA, e_diag, e_M, e_K, e_lda, e_incx, e_batch_count>;

inline void testname_tbmv_batched(const Arguments& arg, std::string& name)
{
    hipblasTbmvBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_tbmv_batched_bad_arg(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasTbmvBatchedFn
        = FORTRAN ? hipblasTbmvBatched<T, true> : hipblasTbmvBatched<T, false>;

    auto hipblasTbmvBatchedFn_64
        = arg.api == FORTRAN_64 ? hipblasTbmvBatched_64<T, true> : hipblasTbmvBatched_64<T, false>;

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

        // Allocate device memory
        device_batch_matrix<T> dAb(banded_matrix_row, M, lda, batch_count);
        device_batch_vector<T> dx(M, incx, batch_count);

        DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                    hipblasTbmvBatchedFn,
                    (nullptr,
                     uplo,
                     transA,
                     diag,
                     M,
                     K,
                     dAb.ptr_on_device(),
                     lda,
                     dx.ptr_on_device(),
                     incx,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasTbmvBatchedFn,
                    (handle,
                     HIPBLAS_FILL_MODE_FULL,
                     transA,
                     diag,
                     M,
                     K,
                     dAb.ptr_on_device(),
                     lda,
                     dx.ptr_on_device(),
                     incx,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasTbmvBatchedFn,
                    (handle,
                     (hipblasFillMode_t)HIPBLAS_OP_N,
                     transA,
                     diag,
                     M,
                     K,
                     dAb.ptr_on_device(),
                     lda,
                     dx.ptr_on_device(),
                     incx,
                     batch_count));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasTbmvBatchedFn,
                    (handle,
                     uplo,
                     (hipblasOperation_t)HIPBLAS_FILL_MODE_FULL,
                     diag,
                     M,
                     K,
                     dAb.ptr_on_device(),
                     lda,
                     dx.ptr_on_device(),
                     incx,
                     batch_count));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasTbmvBatchedFn,
                    (handle,
                     uplo,
                     transA,
                     (hipblasDiagType_t)HIPBLAS_FILL_MODE_FULL,
                     M,
                     K,
                     dAb.ptr_on_device(),
                     lda,
                     dx.ptr_on_device(),
                     incx,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasTbmvBatchedFn,
                    (handle,
                     uplo,
                     transA,
                     diag,
                     M,
                     K,
                     nullptr,
                     lda,
                     dx.ptr_on_device(),
                     incx,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasTbmvBatchedFn,
                    (handle,
                     uplo,
                     transA,
                     diag,
                     M,
                     K,
                     dAb.ptr_on_device(),
                     lda,
                     nullptr,
                     incx,
                     batch_count));

        // With M == 0, can have all nullptrs
        DAPI_CHECK(hipblasTbmvBatchedFn,
                   (handle, uplo, transA, diag, 0, K, nullptr, lda, nullptr, incx, batch_count));
        DAPI_CHECK(hipblasTbmvBatchedFn,
                   (handle, uplo, transA, diag, M, K, nullptr, lda, nullptr, incx, 0));
    }
}

template <typename T>
void testing_tbmv_batched(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasTbmvBatchedFn
        = FORTRAN ? hipblasTbmvBatched<T, true> : hipblasTbmvBatched<T, false>;

    auto hipblasTbmvBatchedFn_64
        = arg.api == FORTRAN_64 ? hipblasTbmvBatched_64<T, true> : hipblasTbmvBatched_64<T, false>;

    hipblasFillMode_t  uplo        = char2hipblas_fill(arg.uplo);
    hipblasOperation_t transA      = char2hipblas_operation(arg.transA);
    hipblasDiagType_t  diag        = char2hipblas_diagonal(arg.diag);
    int64_t            M           = arg.M;
    int64_t            K           = arg.K;
    int64_t            lda         = arg.lda;
    int64_t            incx        = arg.incx;
    int64_t            batch_count = arg.batch_count;

    const int64_t banded_matrix_row = K + 1;
    size_t        abs_incx          = incx >= 0 ? incx : -incx;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = M < 0 || K < 0 || lda < banded_matrix_row || !incx || batch_count < 0;
    if(invalid_size || !M || !batch_count)
    {
        DAPI_EXPECT(invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS,
                    hipblasTbmvBatchedFn,
                    (handle, uplo, transA, diag, M, K, nullptr, lda, nullptr, incx, batch_count));
        return;
    }

    double hipblas_error{0};

    // Naming: `h` is in CPU (host) memory(eg hAb), `d` is in GPU (device) memory (eg dAb).
    // Allocate host memory
    host_batch_matrix<T> hAb(banded_matrix_row, M, lda, batch_count);
    host_batch_vector<T> hx(M, incx, batch_count);
    host_batch_vector<T> hx_cpu(M, incx, batch_count);
    host_batch_vector<T> hx_res(M, incx, batch_count);

    // Check host memory allocation
    CHECK_HIP_ERROR(hAb.memcheck());
    CHECK_HIP_ERROR(hx.memcheck());
    CHECK_HIP_ERROR(hx_cpu.memcheck());
    CHECK_HIP_ERROR(hx_res.memcheck());

    // Allocate device memory
    device_batch_matrix<T> dAb(banded_matrix_row, M, lda, batch_count);
    device_batch_vector<T> dx(M, incx, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dAb.memcheck());

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
        DAPI_CHECK(hipblasTbmvBatchedFn,
                   (handle,
                    uplo,
                    transA,
                    diag,
                    M,
                    K,
                    dAb.ptr_on_device(),
                    lda,
                    dx.ptr_on_device(),
                    incx,
                    batch_count));

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
            unit_check_general<T>(1, M, batch_count, abs_incx, hx_cpu, hx_res);
        }
        if(arg.norm_check)
        {
            hipblas_error = norm_check_general<T>('F', 1, M, abs_incx, hx_cpu, hx_res, batch_count);
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

            DAPI_DISPATCH(hipblasTbmvBatchedFn,
                          (handle,
                           uplo,
                           transA,
                           diag,
                           M,
                           K,
                           dAb.ptr_on_device(),
                           lda,
                           dx.ptr_on_device(),
                           incx,
                           batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasTbmvBatchedModel{}.log_args<T>(std::cout,
                                              arg,
                                              gpu_time_used,
                                              tbmv_gflop_count<T>(M, K),
                                              tbmv_gbyte_count<T>(M, K),
                                              hipblas_error);
    }
}
