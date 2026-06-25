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

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

/* ============================================================================================ */

using hipblasSwapStridedBatchedModel
    = ArgumentModel<e_a_type, e_N, e_incx, e_incy, e_stride_scale, e_batch_count>;

inline void testname_swap_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasSwapStridedBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_swap_strided_batched_bad_arg(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasSwapStridedBatchedFn
        = FORTRAN ? hipblasSwapStridedBatched<T, true> : hipblasSwapStridedBatched<T, false>;
    auto hipblasSwapStridedBatchedFn_64 = arg.api == FORTRAN_64
                                              ? hipblasSwapStridedBatched_64<T, true>
                                              : hipblasSwapStridedBatched_64<T, false>;

    hipblasLocalHandle handle(arg);

    int64_t       N           = 100;
    int64_t       incx        = 1;
    int64_t       incy        = 1;
    int64_t       batch_count = 2;
    hipblasStride stride_x    = N * incx;
    hipblasStride stride_y    = N * incy;

    device_strided_batch_vector<T> dx(N, incx, stride_x, batch_count);
    device_strided_batch_vector<T> dy(N, incy, stride_y, batch_count);

    DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                hipblasSwapStridedBatchedFn,
                (nullptr, N, dx, incx, stride_x, dy, incy, stride_y, batch_count));
    DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                hipblasSwapStridedBatchedFn,
                (handle, N, nullptr, incx, stride_x, dy, incy, stride_y, batch_count));
    DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                hipblasSwapStridedBatchedFn,
                (handle, N, dx, incx, stride_x, nullptr, incy, stride_y, batch_count));
}

template <typename T>
void testing_swap_strided_batched(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasSwapStridedBatchedFn
        = FORTRAN ? hipblasSwapStridedBatched<T, true> : hipblasSwapStridedBatched<T, false>;
    auto hipblasSwapStridedBatchedFn_64 = arg.api == FORTRAN_64
                                              ? hipblasSwapStridedBatched_64<T, true>
                                              : hipblasSwapStridedBatched_64<T, false>;

    int64_t N            = arg.N;
    int64_t incx         = arg.incx;
    int64_t incy         = arg.incy;
    double  stride_scale = arg.stride_scale;
    int64_t batch_count  = arg.batch_count;
    int     unit_check   = arg.unit_check;
    int     norm_check   = arg.norm_check;
    int     timing       = arg.timing;

    int64_t       abs_incx = incx >= 0 ? incx : -incx;
    int64_t       abs_incy = incy >= 0 ? incy : -incy;
    hipblasStride stride_x = N * abs_incx * stride_scale;
    hipblasStride stride_y = N * abs_incy * stride_scale;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0 || batch_count <= 0)
    {
        DAPI_CHECK(hipblasSwapStridedBatchedFn,
                   (handle, N, nullptr, incx, stride_x, nullptr, incy, stride_y, batch_count));
        return;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_strided_batch_vector<T> hx(N, incx, stride_x, batch_count);
    host_strided_batch_vector<T> hx_cpu(N, incx, stride_x, batch_count);
    host_strided_batch_vector<T> hy(N, incy, stride_y, batch_count);
    host_strided_batch_vector<T> hy_cpu(N, incy, stride_y, batch_count);

    device_strided_batch_vector<T> dx(N, incx, stride_x, batch_count);
    device_strided_batch_vector<T> dy(N, incy, stride_y, batch_count);

    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());

    double hipblas_error = 0.0;
    double gpu_time_used = 0.0;

    // Initial Data on CPU
    hipblas_init_vector(hx, arg, hipblas_client_alpha_sets_nan, true);
    hipblas_init_vector(hy, arg, hipblas_client_alpha_sets_nan, true);

    hx_cpu.copy_from(hx);
    hy_cpu.copy_from(hy);

    // copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));

    if(unit_check || norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        DAPI_CHECK(hipblasSwapStridedBatchedFn,
                   (handle, N, dx, incx, stride_x, dy, incy, stride_y, batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hx.transfer_from(dx));
        CHECK_HIP_ERROR(hy.transfer_from(dy));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int64_t b = 0; b < batch_count; b++)
        {
            ref_swap<T>(N, hx_cpu[b], incx, hy_cpu[b], incy);
        }

        if(unit_check)
        {
            unit_check_general<T>(1, N, batch_count, abs_incx, stride_x, hx_cpu.data(), hx.data());
            unit_check_general<T>(1, N, batch_count, abs_incy, stride_y, hy_cpu.data(), hy.data());
        }
        if(norm_check)
        {
            hipblas_error = std::max(
                norm_check_general<T>(
                    'F', 1, N, abs_incx, stride_x, hx_cpu.data(), hx.data(), batch_count),
                norm_check_general<T>(
                    'F', 1, N, abs_incy, stride_y, hy_cpu.data(), hy.data(), batch_count));
        }

    } // end of if unit/norm check

    if(timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_CHECK(hipblasSwapStridedBatchedFn,
                       (handle, N, dx, incx, stride_x, dy, incy, stride_y, batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasSwapStridedBatchedModel{}.log_args<T>(std::cout,
                                                     arg,
                                                     gpu_time_used,
                                                     swap_gflop_count<T>(N),
                                                     swap_gbyte_count<T>(N),
                                                     hipblas_error);
    }
}
