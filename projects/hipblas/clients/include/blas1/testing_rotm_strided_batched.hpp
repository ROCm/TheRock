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

using hipblasRotmStridedBatchedModel
    = ArgumentModel<e_a_type, e_N, e_incx, e_incy, e_stride_scale, e_batch_count>;

inline void testname_rotm_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasRotmStridedBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_rotm_strided_batched_bad_arg(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasRotmStridedBatchedFn
        = FORTRAN ? hipblasRotmStridedBatched<T, true> : hipblasRotmStridedBatched<T, false>;
    auto hipblasRotmStridedBatchedFn_64 = arg.api == FORTRAN_64
                                              ? hipblasRotmStridedBatched_64<T, true>
                                              : hipblasRotmStridedBatched_64<T, false>;

    int64_t       N           = 100;
    int64_t       incx        = 1;
    int64_t       incy        = 1;
    int64_t       batch_count = 2;
    hipblasStride stride_x    = N * incx;
    hipblasStride stride_y    = N * incy;
    hipblasStride stride_p    = 5;

    hipblasLocalHandle handle(arg);

    T h_param[10];

    device_strided_batch_vector<T> dx(N, incx, stride_x, batch_count);
    device_strided_batch_vector<T> dy(N, incy, stride_y, batch_count);
    device_strided_batch_vector<T> dparam(5, 1, stride_p, batch_count);
    T*                             param = dparam;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
        {
            param = h_param;

            // if pointer_mode_host, param[0] == -2, and strideparam = 0, can quick return
            param[0] = -2;
            DAPI_CHECK(hipblasRotmStridedBatchedFn,
                       (handle,
                        N,
                        nullptr,
                        incx,
                        stride_x,
                        nullptr,
                        incy,
                        stride_y,
                        param,
                        0,
                        batch_count));
            param[0] = 0;
        }

        DAPI_EXPECT(
            HIPBLAS_STATUS_NOT_INITIALIZED,
            hipblasRotmStridedBatchedFn,
            (nullptr, N, dx, incx, stride_x, dy, incy, stride_y, param, stride_p, batch_count));
        DAPI_EXPECT(
            HIPBLAS_STATUS_INVALID_VALUE,
            hipblasRotmStridedBatchedFn,
            (handle, N, nullptr, incx, stride_x, dy, incy, stride_y, param, stride_p, batch_count));
        DAPI_EXPECT(
            HIPBLAS_STATUS_INVALID_VALUE,
            hipblasRotmStridedBatchedFn,
            (handle, N, dx, incx, stride_x, nullptr, incy, stride_y, param, stride_p, batch_count));
        DAPI_EXPECT(
            HIPBLAS_STATUS_INVALID_VALUE,
            hipblasRotmStridedBatchedFn,
            (handle, N, dx, incx, stride_x, dy, incy, stride_y, nullptr, stride_p, batch_count));
    }
}

template <typename T>
void testing_rotm_strided_batched(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasRotmStridedBatchedFn
        = FORTRAN ? hipblasRotmStridedBatched<T, true> : hipblasRotmStridedBatched<T, false>;
    auto hipblasRotmStridedBatchedFn_64 = arg.api == FORTRAN_64
                                              ? hipblasRotmStridedBatched_64<T, true>
                                              : hipblasRotmStridedBatched_64<T, false>;

    double stride_scale = arg.stride_scale;

    int64_t       N            = arg.N;
    int64_t       incx         = arg.incx;
    int64_t       incy         = arg.incy;
    hipblasStride stride_param = 5 * stride_scale;
    hipblasStride stride_data  = 4 * stride_scale;
    int64_t       batch_count  = arg.batch_count;

    const T rel_error = std::numeric_limits<T>::epsilon() * 1000;

    hipblasLocalHandle handle(arg);

    int64_t       abs_incx = incx >= 0 ? incx : -incx;
    int64_t       abs_incy = incy >= 0 ? incy : -incy;
    hipblasStride stride_x = N * abs_incx * stride_scale;
    hipblasStride stride_y = N * abs_incy * stride_scale;

    // check to prevent undefined memory allocation error
    if(N <= 0 || batch_count <= 0)
    {
        DAPI_CHECK(hipblasRotmStridedBatchedFn,
                   (handle,
                    N,
                    nullptr,
                    incx,
                    stride_x,
                    nullptr,
                    incy,
                    stride_y,
                    nullptr,
                    stride_param,
                    batch_count));

        return;
    }

    double gpu_time_used{0}, hipblas_error_device{0};

    // Initial Data on CPU
    host_strided_batch_vector<T> hx(N, incx, stride_x, batch_count);
    host_strided_batch_vector<T> hy(N, incy, stride_y, batch_count);
    host_strided_batch_vector<T> hdata(4, 1, stride_data, batch_count);
    host_strided_batch_vector<T> hparam(5, 1, stride_param, batch_count);

    device_strided_batch_vector<T> dx(N, incx, stride_x, batch_count);
    device_strided_batch_vector<T> dy(N, incy, stride_y, batch_count);
    device_strided_batch_vector<T> dparam(5, 1, stride_param, batch_count);

    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(dparam.memcheck());

    hipblas_init_vector(hx, arg, hipblas_client_alpha_sets_nan, true);
    hipblas_init_vector(hy, arg, hipblas_client_alpha_sets_nan, false);
    hipblas_init_vector(hdata, arg, hipblas_client_alpha_sets_nan, false);

    // generating simply one set of hparam which will not be appropriate for testing
    // that it zeros out the second element of the rotm vector parameter
    hipblas_init_vector_zero<T>(hparam);

    for(int64_t b = 0; b < batch_count; b++)
        ref_rotmg<T>(&hdata[b][0], &hdata[b][1], &hdata[b][2], &hdata[b][3], hparam[b]);

    constexpr int FLAG_COUNT        = 4;
    const T       FLAGS[FLAG_COUNT] = {-1, 0, 1, -2};

    for(int i = 0; i < FLAG_COUNT; i++)
    {
        if(arg.unit_check || arg.norm_check)
        {
            for(int64_t b = 0; b < batch_count; b++)
                hparam[b][0] = FLAGS[i];

            // Test device
            CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

            CHECK_HIP_ERROR(dx.transfer_from(hx));
            CHECK_HIP_ERROR(dy.transfer_from(hy));
            CHECK_HIP_ERROR(dparam.transfer_from(hparam));

            DAPI_CHECK(hipblasRotmStridedBatchedFn,
                       (handle,
                        N,
                        dx,
                        incx,
                        stride_x,
                        dy,
                        incy,
                        stride_y,
                        dparam,
                        stride_param,
                        batch_count));

            host_strided_batch_vector<T> rx(N, incx, stride_x, batch_count);
            host_strided_batch_vector<T> ry(N, incy, stride_y, batch_count);

            CHECK_HIP_ERROR(rx.transfer_from(dx));
            CHECK_HIP_ERROR(ry.transfer_from(dy));

            host_strided_batch_vector<T> cx(N, incx, stride_x, batch_count);
            host_strided_batch_vector<T> cy(N, incy, stride_y, batch_count);

            cx.copy_from(hx);
            cy.copy_from(hy);

            // CPU BLAS reference data
            for(int64_t b = 0; b < batch_count; b++)
            {
                ref_rotm<T>(N, cx[b], incx, cy[b], incy, hparam[b]);
            }

            if(arg.unit_check)
            {
                near_check_general<T>(1, N, batch_count, abs_incx, stride_x, cx, rx, rel_error);
                near_check_general<T>(1, N, batch_count, abs_incy, stride_y, cy, ry, rel_error);
            }
            if(arg.norm_check)
            {
                hipblas_error_device
                    = norm_check_general<T>('F', 1, N, abs_incx, stride_x, cx, rx, batch_count);
                hipblas_error_device
                    += norm_check_general<T>('F', 1, N, abs_incy, stride_y, cy, ry, batch_count);
            }
        }
    }

    if(arg.timing)
    {
        for(int64_t b = 0; b < batch_count; b++)
            hparam[b][0] = 0;

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

            DAPI_CHECK(hipblasRotmStridedBatchedFn,
                       (handle,
                        N,
                        dx,
                        incx,
                        stride_x,
                        dy,
                        incy,
                        stride_y,
                        dparam,
                        stride_param,
                        batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasRotmStridedBatchedModel{}.log_args<T>(std::cout,
                                                     arg,
                                                     gpu_time_used,
                                                     rotm_gflop_count<T>(N, 0),
                                                     rotm_gbyte_count<T>(N, 0),
                                                     0,
                                                     hipblas_error_device);
    }
}
