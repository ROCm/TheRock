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

using hipblasCopyBatchedModel = ArgumentModel<e_a_type, e_N, e_incx, e_incy, e_batch_count>;

inline void testname_copy_batched(const Arguments& arg, std::string& name)
{
    hipblasCopyBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_copy_batched_bad_arg(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasCopyBatchedFn
        = FORTRAN ? hipblasCopyBatched<T, true> : hipblasCopyBatched<T, false>;
    auto hipblasCopyBatchedFn_64
        = arg.api == FORTRAN_64 ? hipblasCopyBatched_64<T, true> : hipblasCopyBatched_64<T, false>;

    hipblasLocalHandle handle(arg);

    int64_t N           = 100;
    int64_t incx        = 1;
    int64_t incy        = 1;
    int64_t batch_count = 2;

    device_batch_vector<T> dx(N, incx, batch_count);
    device_batch_vector<T> dy(N, incy, batch_count);

    DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                hipblasCopyBatchedFn,
                (nullptr, N, dx, incx, dy, incy, batch_count));
    DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                hipblasCopyBatchedFn,
                (handle, N, nullptr, incx, dy, incy, batch_count));
    DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                hipblasCopyBatchedFn,
                (handle, N, dx, incx, nullptr, incy, batch_count));
}

template <typename T>
void testing_copy_batched(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasCopyBatchedFn
        = FORTRAN ? hipblasCopyBatched<T, true> : hipblasCopyBatched<T, false>;
    auto hipblasCopyBatchedFn_64
        = arg.api == FORTRAN_64 ? hipblasCopyBatched_64<T, true> : hipblasCopyBatched_64<T, false>;

    int64_t N           = arg.N;
    int64_t incx        = arg.incx;
    int64_t incy        = arg.incy;
    int64_t batch_count = arg.batch_count;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0 || batch_count <= 0)
    {
        DAPI_CHECK(hipblasCopyBatchedFn, (handle, N, nullptr, incx, nullptr, incy, batch_count));
        return;
    }

    int64_t abs_incy = incy >= 0 ? incy : -incy;

    double hipblas_error = 0.0;
    double gpu_time_used = 0.0;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_batch_vector<T> hx(N, incx, batch_count);
    host_batch_vector<T> hy(N, incy, batch_count);
    host_batch_vector<T> hx_cpu(N, incx, batch_count);
    host_batch_vector<T> hy_cpu(N, incy, batch_count);

    device_batch_vector<T> dx(N, incx, batch_count);
    device_batch_vector<T> dy(N, incy, batch_count);

    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());

    hipblas_init_vector(hx, arg, hipblas_client_alpha_sets_nan, true);
    hipblas_init_vector(hy, arg, hipblas_client_alpha_sets_nan, false);

    hx_cpu.copy_from(hx);
    hy_cpu.copy_from(hy);

    CHECK_DEVICE_ALLOCATION(dx.transfer_from(hx));
    CHECK_DEVICE_ALLOCATION(dy.transfer_from(hy));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
                    HIPBLAS
        =================================================================== */
        DAPI_CHECK(hipblasCopyBatchedFn,
                   (handle, N, dx.ptr_on_device(), incx, dy.ptr_on_device(), incy, batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hy.transfer_from(dy));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int64_t b = 0; b < batch_count; b++)
        {
            ref_copy<T>(N, hx_cpu[b], incx, hy_cpu[b], incy);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, batch_count, abs_incy, hy_cpu, hy);
        }
        if(arg.norm_check)
        {
            hipblas_error = norm_check_general<T>('F', 1, N, abs_incy, hy_cpu, hy, batch_count);
        }
    } // end of if unit check

    if(arg.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_CHECK(
                hipblasCopyBatchedFn,
                (handle, N, dx.ptr_on_device(), incx, dy.ptr_on_device(), incy, batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasCopyBatchedModel{}.log_args<T>(std::cout,
                                              arg,
                                              gpu_time_used,
                                              copy_gflop_count<T>(N),
                                              copy_gbyte_count<T>(N),
                                              hipblas_error);
    }
}
