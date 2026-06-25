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

using hipblasRotmBatchedModel = ArgumentModel<e_a_type, e_N, e_incx, e_incy, e_batch_count>;

inline void testname_rotm_batched(const Arguments& arg, std::string& name)
{
    hipblasRotmBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_rotm_batched_bad_arg(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasRotmBatchedFn
        = FORTRAN ? hipblasRotmBatched<T, true> : hipblasRotmBatched<T, false>;
    auto hipblasRotmBatchedFn_64
        = arg.api == FORTRAN_64 ? hipblasRotmBatched_64<T, true> : hipblasRotmBatched_64<T, false>;

    int64_t N           = 100;
    int64_t incx        = 1;
    int64_t incy        = 1;
    int64_t batch_count = 2;

    hipblasLocalHandle handle(arg);

    device_batch_vector<T> dx(N, incx, batch_count);
    device_batch_vector<T> dy(N, incy, batch_count);
    device_batch_vector<T> dparam(5, 1, batch_count);

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        // No param checking for batched version

        // None of these test cases will write to result so using device pointer is fine for both modes
        DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                    hipblasRotmBatchedFn,
                    (nullptr, N, dx, incx, dy, incy, dparam, batch_count));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasRotmBatchedFn,
                    (handle, N, nullptr, incx, dy, incy, dparam, batch_count));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasRotmBatchedFn,
                    (handle, N, dx, incx, nullptr, incy, dparam, batch_count));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasRotmBatchedFn,
                    (handle, N, dx, incx, dy, incy, nullptr, batch_count));
    }
}

template <typename T>
void testing_rotm_batched(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasRotmBatchedFn
        = FORTRAN ? hipblasRotmBatched<T, true> : hipblasRotmBatched<T, false>;
    auto hipblasRotmBatchedFn_64
        = arg.api == FORTRAN_64 ? hipblasRotmBatched_64<T, true> : hipblasRotmBatched_64<T, false>;

    int64_t N           = arg.N;
    int64_t incx        = arg.incx;
    int64_t incy        = arg.incy;
    int64_t batch_count = arg.batch_count;

    const T rel_error = std::numeric_limits<T>::epsilon() * 1000;

    hipblasLocalHandle handle(arg);

    // check to prevent undefined memory allocation error
    if(N <= 0 || batch_count <= 0)
    {
        DAPI_CHECK(hipblasRotmBatchedFn,
                   (handle, N, nullptr, incx, nullptr, incy, nullptr, batch_count));
        return;
    }

    int64_t abs_incx = incx >= 0 ? incx : -incx;
    int64_t abs_incy = incy >= 0 ? incy : -incy;

    double gpu_time_used{0}, hipblas_error_device{0};

    host_batch_vector<T> hx(N, incx, batch_count);
    host_batch_vector<T> hy(N, incy, batch_count);
    host_batch_vector<T> hdata(4, 1, batch_count);
    host_batch_vector<T> hparam(5, 1, batch_count);

    device_batch_vector<T> dx(N, incx, batch_count);
    device_batch_vector<T> dy(N, incy, batch_count);
    device_batch_vector<T> dparam(5, 1, batch_count);

    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(dparam.memcheck());

    hipblas_init_vector(hx, arg, hipblas_client_alpha_sets_nan, true);
    hipblas_init_vector(hy, arg, hipblas_client_alpha_sets_nan, false);
    hipblas_init_vector(hdata, arg, hipblas_client_alpha_sets_nan, false);

    for(int64_t b = 0; b < batch_count; b++)
        ref_rotmg<T>(&hdata[b][0], &hdata[b][1], &hdata[b][2], &hdata[b][3], hparam[b]);

    constexpr int FLAG_COUNT        = 4;
    const T       FLAGS[FLAG_COUNT] = {-1, 0, 1, -2};

    for(int i = 0; i < FLAG_COUNT; i++)
    {
        if(arg.unit_check || arg.norm_check)
        {
            for(int64_t b = 0; b < batch_count; b++)
            {
                hparam[b][0] = FLAGS[i];
            }

            // Test device
            CHECK_HIP_ERROR(dx.transfer_from(hx));
            CHECK_HIP_ERROR(dy.transfer_from(hy));
            CHECK_HIP_ERROR(dparam.transfer_from(hparam));
            CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
            DAPI_CHECK(hipblasRotmBatchedFn,
                       (handle,
                        N,
                        dx.ptr_on_device(),
                        incx,
                        dy.ptr_on_device(),
                        incy,
                        dparam.ptr_on_device(),
                        batch_count));

            host_batch_vector<T> rx(N, incx, batch_count);
            host_batch_vector<T> ry(N, incy, batch_count);
            CHECK_HIP_ERROR(rx.transfer_from(dx));
            CHECK_HIP_ERROR(ry.transfer_from(dy));

            host_batch_vector<T> cx(N, incx, batch_count);
            host_batch_vector<T> cy(N, incy, batch_count);
            cx.copy_from(hx);
            cy.copy_from(hy);

            for(int64_t b = 0; b < batch_count; b++)
            {
                // CPU BLAS reference data
                ref_rotm<T>(N, cx[b], incx, cy[b], incy, hparam[b]);
            }

            if(arg.unit_check)
            {
                for(int64_t b = 0; b < batch_count; b++)
                {
                    near_check_general<T>(1, N, abs_incx, cx[b], rx[b], rel_error);
                    near_check_general<T>(1, N, abs_incy, cy[b], ry[b], rel_error);
                }
            }
            if(arg.norm_check)
            {
                hipblas_error_device
                    = norm_check_general<T>('F', 1, N, abs_incx, cx, rx, batch_count);
                hipblas_error_device
                    += norm_check_general<T>('F', 1, N, abs_incy, cy, ry, batch_count);
            }
        }
    }

    if(arg.timing)
    {
        for(int64_t b = 0; b < batch_count; b++)
        {
            hparam[b][0] = 0;
        }

        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIP_ERROR(dx.transfer_from(hx));
        CHECK_HIP_ERROR(dy.transfer_from(hy));
        CHECK_HIP_ERROR(dparam.transfer_from(hparam));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_CHECK(hipblasRotmBatchedFn,
                       (handle,
                        N,
                        dx.ptr_on_device(),
                        incx,
                        dy.ptr_on_device(),
                        incy,
                        dparam.ptr_on_device(),
                        batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasRotmBatchedModel{}.log_args<T>(std::cout,
                                              arg,
                                              gpu_time_used,
                                              rotm_gflop_count<T>(N, 0),
                                              rotm_gbyte_count<T>(N, 0),
                                              0,
                                              hipblas_error_device);
    }
}
