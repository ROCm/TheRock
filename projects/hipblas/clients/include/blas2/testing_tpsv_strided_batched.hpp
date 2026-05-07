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

using hipblasTpsvStridedBatchedModel
    = ArgumentModel<e_a_type, e_uplo, e_transA, e_diag, e_N, e_incx, e_stride_scale, e_batch_count>;

inline void testname_tpsv_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasTpsvStridedBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_tpsv_strided_batched_bad_arg(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasTpsvStridedBatchedFn
        = FORTRAN ? hipblasTpsvStridedBatched<T, true> : hipblasTpsvStridedBatched<T, false>;

    auto hipblasTpsvStridedBatchedFn_64 = arg.api == FORTRAN_64
                                              ? hipblasTpsvStridedBatched_64<T, true>
                                              : hipblasTpsvStridedBatched_64<T, false>;

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
        hipblasStride      stride_A    = hipblas_packed_matrix_size(N);
        hipblasStride      stride_x    = N * incx;

        // Allocate device memory
        device_strided_batch_matrix<T> dAp(
            1, hipblas_packed_matrix_size(N), 1, stride_A, batch_count);
        device_strided_batch_vector<T> dx(N, incx, stride_x, batch_count);

        DAPI_EXPECT(
            HIPBLAS_STATUS_NOT_INITIALIZED,
            hipblasTpsvStridedBatchedFn,
            (nullptr, uplo, transA, diag, N, dAp, stride_A, dx, incx, stride_x, batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasTpsvStridedBatchedFn,
                    (handle,
                     HIPBLAS_FILL_MODE_FULL,
                     transA,
                     diag,
                     N,
                     dAp,
                     stride_A,
                     dx,
                     incx,
                     stride_x,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasTpsvStridedBatchedFn,
                    (handle,
                     (hipblasFillMode_t)HIPBLAS_OP_N,
                     transA,
                     diag,
                     N,
                     dAp,
                     stride_A,
                     dx,
                     incx,
                     stride_x,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasTpsvStridedBatchedFn,
                    (handle,
                     uplo,
                     (hipblasOperation_t)HIPBLAS_FILL_MODE_FULL,
                     diag,
                     N,
                     dAp,
                     stride_A,
                     dx,
                     incx,
                     stride_x,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasTpsvStridedBatchedFn,
                    (handle,
                     uplo,
                     transA,
                     (hipblasDiagType_t)HIPBLAS_FILL_MODE_FULL,
                     N,
                     dAp,
                     stride_A,
                     dx,
                     incx,
                     stride_x,
                     batch_count));

        DAPI_EXPECT(
            HIPBLAS_STATUS_INVALID_VALUE,
            hipblasTpsvStridedBatchedFn,
            (handle, uplo, transA, diag, N, nullptr, stride_A, dx, incx, stride_x, batch_count));

        DAPI_EXPECT(
            HIPBLAS_STATUS_INVALID_VALUE,
            hipblasTpsvStridedBatchedFn,
            (handle, uplo, transA, diag, N, dAp, stride_A, nullptr, incx, stride_x, batch_count));

        // With N == 0, can have all nullptrs
        DAPI_CHECK(hipblasTpsvStridedBatchedFn,
                   (handle,
                    uplo,
                    transA,
                    diag,
                    0,
                    nullptr,
                    stride_A,
                    nullptr,
                    incx,
                    stride_x,
                    batch_count));
        DAPI_CHECK(hipblasTpsvStridedBatchedFn,
                   (handle, uplo, transA, diag, N, nullptr, stride_A, nullptr, incx, stride_x, 0));
    }
}

template <typename T>
void testing_tpsv_strided_batched(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasTpsvStridedBatchedFn
        = FORTRAN ? hipblasTpsvStridedBatched<T, true> : hipblasTpsvStridedBatched<T, false>;

    auto hipblasTpsvStridedBatchedFn_64 = arg.api == FORTRAN_64
                                              ? hipblasTpsvStridedBatched_64<T, true>
                                              : hipblasTpsvStridedBatched_64<T, false>;

    hipblasFillMode_t  uplo         = char2hipblas_fill(arg.uplo);
    hipblasDiagType_t  diag         = char2hipblas_diagonal(arg.diag);
    hipblasOperation_t transA       = char2hipblas_operation(arg.transA);
    int64_t            N            = arg.N;
    int64_t            incx         = arg.incx;
    double             stride_scale = arg.stride_scale;
    int64_t            batch_count  = arg.batch_count;

    int64_t       size_AP   = hipblas_packed_matrix_size(N);
    size_t        abs_incx  = incx < 0 ? -incx : incx;
    hipblasStride stride_A  = N * N; // only for test setup
    hipblasStride stride_AP = size_AP * stride_scale;
    hipblasStride stride_x  = abs_incx * N * stride_scale;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = N < 0 || !incx || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        DAPI_EXPECT(invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS,
                    hipblasTpsvStridedBatchedFn,
                    (handle,
                     uplo,
                     transA,
                     diag,
                     N,
                     nullptr,
                     stride_AP,
                     nullptr,
                     incx,
                     stride_x,
                     batch_count));
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hAp), `d` is in GPU (device) memory (eg dAp).
    // Allocate host memory
    host_strided_batch_matrix<T> hA(N, N, N, stride_A, batch_count);
    host_strided_batch_matrix<T> hAp(1, hipblas_packed_matrix_size(N), 1, stride_AP, batch_count);
    host_strided_batch_vector<T> hb(N, incx, stride_x, batch_count);
    host_strided_batch_vector<T> hx(N, incx, stride_x, batch_count);
    host_strided_batch_vector<T> hx_or_b(N, incx, stride_x, batch_count);
    host_strided_batch_vector<T> cpu_x_or_b(N, incx, stride_x, batch_count);

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hAp.memcheck());
    CHECK_HIP_ERROR(hb.memcheck());
    CHECK_HIP_ERROR(hx.memcheck());
    CHECK_HIP_ERROR(hx_or_b.memcheck());
    CHECK_HIP_ERROR(cpu_x_or_b.memcheck());

    // Allocate device memory
    device_strided_batch_matrix<T> dAp(1, hipblas_packed_matrix_size(N), 1, stride_AP, batch_count);
    device_strided_batch_vector<T> dx_or_b(N, incx, stride_x, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dAp.memcheck());
    CHECK_DEVICE_ALLOCATION(dx_or_b.memcheck());

    double hipblas_error{0}, cumulative_hipblas_error{0};

    // Initial Data on CPU
    hipblas_init_matrix(hA,
                        arg,
                        hipblas_client_never_set_nan,
                        hipblas_diagonally_dominant_triangular_matrix,
                        true,
                        false);
    hipblas_init_vector(hx, arg, hipblas_client_never_set_nan, false, true);

    hb.copy_from(hx);

    if(diag == HIPBLAS_DIAG_UNIT)
    {
        make_unit_diagonal(uplo, hA);
    }

    for(size_t b = 0; b < batch_count; b++)
    {
        // Calculate hb = hA*hx;
        ref_trmv<T>(uplo, transA, diag, N, hA[b], N, hb[b], incx);
    }

    // helper function to convert Regular matrix `hA` to packed matrix `hAp`
    regular_to_packed(uplo == HIPBLAS_FILL_MODE_UPPER, hA, hAp, N);

    cpu_x_or_b.copy_from(hb);
    hx_or_b.copy_from(hb);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dAp.transfer_from(hAp));
    CHECK_HIP_ERROR(dx_or_b.transfer_from(hx_or_b));

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    if(arg.unit_check || arg.norm_check)
    {
        DAPI_CHECK(
            hipblasTpsvStridedBatchedFn,
            (handle, uplo, transA, diag, N, dAp, stride_AP, dx_or_b, incx, stride_x, batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hx_or_b.transfer_from(dx_or_b));

        // Calculating error
        // For norm_check/bench, currently taking the cumulative sum of errors over all batches
        for(size_t b = 0; b < batch_count; b++)
        {
            hipblas_error = hipblas_abs(vector_norm_1<T>(N, abs_incx, hx[b], hx_or_b[b]));

            if(arg.unit_check)
            {
                double tolerance = std::numeric_limits<real_t<T>>::epsilon() * 40 * N;
                unit_check_error(hipblas_error, tolerance);
            }

            cumulative_hipblas_error += hipblas_error;
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

            DAPI_DISPATCH(hipblasTpsvStridedBatchedFn,
                          (handle,
                           uplo,
                           transA,
                           diag,
                           N,
                           dAp,
                           stride_AP,
                           dx_or_b,
                           incx,
                           stride_x,
                           batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        hipblasTpsvStridedBatchedModel{}.log_args<T>(std::cout,
                                                     arg,
                                                     gpu_time_used,
                                                     tpsv_gflop_count<T>(N),
                                                     tpsv_gbyte_count<T>(N),
                                                     cumulative_hipblas_error);
    }
}
