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

using hipblasTpmvBatchedModel
    = ArgumentModel<e_a_type, e_uplo, e_transA, e_diag, e_N, e_incx, e_batch_count>;

inline void testname_tpmv_batched(const Arguments& arg, std::string& name)
{
    hipblasTpmvBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_tpmv_batched_bad_arg(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasTpmvBatchedFn
        = FORTRAN ? hipblasTpmvBatched<T, true> : hipblasTpmvBatched<T, false>;

    auto hipblasTpmvBatchedFn_64
        = arg.api == FORTRAN_64 ? hipblasTpmvBatched_64<T, true> : hipblasTpmvBatched_64<T, false>;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        hipblasLocalHandle handle(arg);
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        hipblasOperation_t transA      = HIPBLAS_OP_N;
        hipblasFillMode_t  uplo        = HIPBLAS_FILL_MODE_UPPER;
        hipblasDiagType_t  diag        = HIPBLAS_DIAG_NON_UNIT;
        int64_t            N           = 100;
        int64_t            incx        = 1;
        int64_t            batch_count = 2;

        // Allocate device memory
        device_batch_matrix<T> dAp(1, hipblas_packed_matrix_size(N), 1, batch_count);
        device_batch_vector<T> dx(N, incx, batch_count);

        DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                    hipblasTpmvBatchedFn,
                    (nullptr,
                     uplo,
                     transA,
                     diag,
                     N,
                     dAp.ptr_on_device(),
                     dx.ptr_on_device(),
                     incx,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasTpmvBatchedFn,
                    (handle,
                     HIPBLAS_FILL_MODE_FULL,
                     transA,
                     diag,
                     N,
                     dAp.ptr_on_device(),
                     dx.ptr_on_device(),
                     incx,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasTpmvBatchedFn,
                    (handle,
                     (hipblasFillMode_t)HIPBLAS_OP_N,
                     transA,
                     diag,
                     N,
                     dAp.ptr_on_device(),
                     dx.ptr_on_device(),
                     incx,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasTpmvBatchedFn,
                    (handle,
                     uplo,
                     (hipblasOperation_t)HIPBLAS_FILL_MODE_FULL,
                     diag,
                     N,
                     dAp.ptr_on_device(),
                     dx.ptr_on_device(),
                     incx,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasTpmvBatchedFn,
                    (handle,
                     uplo,
                     transA,
                     (hipblasDiagType_t)HIPBLAS_FILL_MODE_FULL,
                     N,
                     dAp.ptr_on_device(),
                     dx.ptr_on_device(),
                     incx,
                     batch_count));

        DAPI_EXPECT(
            HIPBLAS_STATUS_INVALID_VALUE,
            hipblasTpmvBatchedFn,
            (handle, uplo, transA, diag, N, nullptr, dx.ptr_on_device(), incx, batch_count));
        DAPI_EXPECT(
            HIPBLAS_STATUS_INVALID_VALUE,
            hipblasTpmvBatchedFn,
            (handle, uplo, transA, diag, N, dAp.ptr_on_device(), nullptr, incx, batch_count));

        // With N == 0, can have all nullptrs
        DAPI_CHECK(hipblasTpmvBatchedFn,
                   (handle, uplo, transA, diag, 0, nullptr, nullptr, incx, batch_count));
        DAPI_CHECK(hipblasTpmvBatchedFn,
                   (handle, uplo, transA, diag, N, nullptr, nullptr, incx, 0));
    }
}

template <typename T>
void testing_tpmv_batched(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasTpmvBatchedFn
        = FORTRAN ? hipblasTpmvBatched<T, true> : hipblasTpmvBatched<T, false>;

    auto hipblasTpmvBatchedFn_64
        = arg.api == FORTRAN_64 ? hipblasTpmvBatched_64<T, true> : hipblasTpmvBatched_64<T, false>;

    hipblasFillMode_t  uplo        = char2hipblas_fill(arg.uplo);
    hipblasOperation_t transA      = char2hipblas_operation(arg.transA);
    hipblasDiagType_t  diag        = char2hipblas_diagonal(arg.diag);
    int64_t            N           = arg.N;
    int64_t            incx        = arg.incx;
    int64_t            batch_count = arg.batch_count;

    size_t abs_incx = incx >= 0 ? incx : -incx;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = N < 0 || !incx || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        DAPI_EXPECT(invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS,
                    hipblasTpmvBatchedFn,
                    (handle, uplo, transA, diag, N, nullptr, nullptr, incx, batch_count));
        return;
    }

    double hipblas_error{0};

    // Naming: `h` is in CPU (host) memory(eg hAp), `d` is in GPU (device) memory (eg dAp).
    // Allocate host memory
    host_batch_matrix<T> hA(N, N, N, batch_count);
    host_batch_matrix<T> hAp(1, hipblas_packed_matrix_size(N), 1, batch_count);
    host_batch_vector<T> hx_cpu(N, incx, batch_count);
    host_batch_vector<T> hx_res(N, incx, batch_count);

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hAp.memcheck());
    CHECK_HIP_ERROR(hx_cpu.memcheck());
    CHECK_HIP_ERROR(hx_res.memcheck());

    // Allocate device memory
    device_batch_matrix<T> dAp(1, hipblas_packed_matrix_size(N), 1, batch_count);
    device_batch_vector<T> dx(N, incx, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dAp.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());

    // Initial Data on CPU
    hipblas_init_matrix(
        hA, arg, hipblas_client_never_set_nan, hipblas_triangular_matrix, true, false);
    hipblas_init_vector(hx_cpu, arg, hipblas_client_never_set_nan, false, true);

    // helper function to convert Regular matrix `hA` to packed matrix `hAp`
    regular_to_packed(uplo == HIPBLAS_FILL_MODE_UPPER, hA, hAp, N);

    CHECK_HIP_ERROR(dAp.transfer_from(hAp));
    CHECK_HIP_ERROR(dx.transfer_from(hx_cpu));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        DAPI_CHECK(hipblasTpmvBatchedFn,
                   (handle,
                    uplo,
                    transA,
                    diag,
                    N,
                    dAp.ptr_on_device(),
                    dx.ptr_on_device(),
                    incx,
                    batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hx_res.transfer_from(dx));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(size_t b = 0; b < batch_count; b++)
        {
            ref_tpmv<T>(uplo, transA, diag, N, hAp[b], hx_cpu[b], incx);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, batch_count, abs_incx, hx_cpu, hx_res);
        }
        if(arg.norm_check)
        {
            hipblas_error = norm_check_general<T>('F', 1, N, abs_incx, hx_cpu, hx_res, batch_count);
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

            DAPI_DISPATCH(hipblasTpmvBatchedFn,
                          (handle,
                           uplo,
                           transA,
                           diag,
                           N,
                           dAp.ptr_on_device(),
                           dx.ptr_on_device(),
                           incx,
                           batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        hipblasTpmvBatchedModel{}.log_args<T>(std::cout,
                                              arg,
                                              gpu_time_used,
                                              tpmv_gflop_count<T>(N),
                                              tpmv_gbyte_count<T>(N),
                                              hipblas_error);
    }
}
