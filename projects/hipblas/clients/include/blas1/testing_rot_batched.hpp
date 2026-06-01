/* ************************************************************************
 * Copyright (C) 2016-2025 Advanced Micro Devices, Inc. All rights reserved.
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

using hipblasRotBatchedModel
    = ArgumentModel<e_a_type, e_c_type, e_compute_type, e_N, e_incx, e_incy, e_batch_count>;

inline void testname_rot_batched(const Arguments& arg, std::string& name)
{
    hipblasRotBatchedModel{}.test_name(arg, name);
}

template <typename T, typename U = T, typename V = T>
void testing_rot_batched_bad_arg(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasRotBatchedFn
        = FORTRAN ? hipblasRotBatched<T, U, V, true> : hipblasRotBatched<T, U, V, false>;
    auto hipblasRotBatchedFn_64 = arg.api == FORTRAN_64 ? hipblasRotBatched_64<T, U, V, true>
                                                        : hipblasRotBatched_64<T, U, V, false>;

    int64_t N           = 100;
    int64_t incx        = 1;
    int64_t incy        = 1;
    int64_t batch_count = 2;

    hipblasLocalHandle handle(arg);

    device_batch_vector<T> dx(N, incx, batch_count);
    device_batch_vector<T> dy(N, incy, batch_count);
    device_vector<U>       dc(1);
    device_vector<V>       ds(1);

    DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                hipblasRotBatchedFn,
                (nullptr, N, dx, incx, dy, incy, dc, ds, batch_count));
    DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                hipblasRotBatchedFn,
                (handle, N, nullptr, incx, dy, incy, dc, ds, batch_count));
    DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                hipblasRotBatchedFn,
                (handle, N, dx, incx, nullptr, incy, dc, ds, batch_count));
    DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                hipblasRotBatchedFn,
                (handle, N, dx, incx, dy, incy, nullptr, ds, batch_count));
    DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                hipblasRotBatchedFn,
                (handle, N, dx, incx, dy, incy, dc, nullptr, batch_count));
}

template <typename T, typename U = T, typename V = T>
void testing_rot_batched(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasRotBatchedFn
        = FORTRAN ? hipblasRotBatched<T, U, V, true> : hipblasRotBatched<T, U, V, false>;
    auto hipblasRotBatchedFn_64 = arg.api == FORTRAN_64 ? hipblasRotBatched_64<T, U, V, true>
                                                        : hipblasRotBatched_64<T, U, V, false>;

    int64_t N           = arg.N;
    int64_t incx        = arg.incx;
    int64_t incy        = arg.incy;
    int64_t batch_count = arg.batch_count;

    const U rel_error = std::numeric_limits<U>::epsilon() * 1000;

    hipblasLocalHandle handle(arg);

    // check to prevent undefined memory allocation error
    if(N <= 0 || batch_count <= 0)
    {
        DAPI_CHECK(hipblasRotBatchedFn,
                   (handle, N, nullptr, incx, nullptr, incy, nullptr, nullptr, batch_count));
        return;
    }

    int64_t abs_incx = incx >= 0 ? incx : -incx;
    int64_t abs_incy = incy >= 0 ? incy : -incy;

    double gpu_time_used{0}, hipblas_error_host{0}, hipblas_error_device{0};

    device_batch_vector<T> dx(N, incx, batch_count);
    device_batch_vector<T> dy(N, incy, batch_count);
    device_vector<U>       dc(1);
    device_vector<V>       ds(1);

    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(dc.memcheck());
    CHECK_DEVICE_ALLOCATION(ds.memcheck());

    // Initial Data on CPU
    host_batch_vector<T> hx(N, incx, batch_count);
    host_batch_vector<T> hy(N, incy, batch_count);
    host_vector<U>       hc(1);
    host_vector<V>       hs(1);

    // Random alpha (0 - 10)
    host_vector<int> alpha(1);

    hipblas_init_vector(hx, arg, hipblas_client_never_set_nan, true);
    hipblas_init_vector(hy, arg, hipblas_client_never_set_nan, false);
    hipblas_init_vector(alpha, arg, hipblas_client_never_set_nan, false);

    // cos and sin of alpha (in rads)
    hc[0] = cos(alpha[0]);
    hs[0] = sin(alpha[0]);

    // CPU BLAS reference data
    host_batch_vector<T> cx(N, incx, batch_count);
    host_batch_vector<T> cy(N, incy, batch_count);
    cx.copy_from(hx);
    cy.copy_from(hy);

    // ref_rotg<T, U>(cx, cy, hc, hs);
    // cx[0] = hx[0];
    // cy[0] = hy[0];
    for(int64_t b = 0; b < batch_count; b++)
    {
        ref_rot<T, U, V>(N, cx[b], incx, cy[b], incy, *hc, *hs);
    }

    if(arg.unit_check || arg.norm_check)
    {
        // Test host
        {
            CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
            CHECK_HIP_ERROR(dx.transfer_from(hx));
            CHECK_HIP_ERROR(dy.transfer_from(hy));
            DAPI_CHECK(hipblasRotBatchedFn,
                       (handle,
                        N,
                        dx.ptr_on_device(),
                        incx,
                        dy.ptr_on_device(),
                        incy,
                        hc.internal_type(),
                        hs.internal_type(),
                        batch_count));

            host_batch_vector<T> rx(N, incx, batch_count);
            host_batch_vector<T> ry(N, incy, batch_count);
            CHECK_HIP_ERROR(rx.transfer_from(dx));
            CHECK_HIP_ERROR(ry.transfer_from(dy));

            if(arg.unit_check)
            {
                for(int64_t b = 0; b < batch_count; b++)
                {
                    near_check_general(1, N, abs_incx, cx[b], rx[b], rel_error);
                    near_check_general(1, N, abs_incy, cy[b], ry[b], rel_error);
                }
            }
            if(arg.norm_check)
            {
                hipblas_error_host
                    = norm_check_general<T>('F', 1, N, abs_incx, cx, rx, batch_count);
                hipblas_error_host
                    += norm_check_general<T>('F', 1, N, abs_incy, cy, ry, batch_count);
            }
        }

        // Test device
        {
            CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
            CHECK_HIP_ERROR(dx.transfer_from(hx));
            CHECK_HIP_ERROR(dy.transfer_from(hy));
            CHECK_HIP_ERROR(hipMemcpy(dc, hc, sizeof(U), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(ds, hs, sizeof(V), hipMemcpyHostToDevice));
            DAPI_CHECK(hipblasRotBatchedFn,
                       (handle,
                        N,
                        dx.ptr_on_device(),
                        incx,
                        dy.ptr_on_device(),
                        incy,
                        dc,
                        ds,
                        batch_count));

            host_batch_vector<T> rx(N, incx, batch_count);
            host_batch_vector<T> ry(N, incy, batch_count);
            CHECK_HIP_ERROR(rx.transfer_from(dx));
            CHECK_HIP_ERROR(ry.transfer_from(dy));

            if(arg.unit_check)
            {
                for(int64_t b = 0; b < batch_count; b++)
                {
                    near_check_general(1, N, abs_incx, cx[b], rx[b], rel_error);
                    near_check_general(1, N, abs_incy, cy[b], ry[b], rel_error);
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
        CHECK_HIP_ERROR(dx.transfer_from(hx));
        CHECK_HIP_ERROR(dy.transfer_from(hy));
        CHECK_HIP_ERROR(hipMemcpy(dc, hc, sizeof(U), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(ds, hs, sizeof(V), hipMemcpyHostToDevice));
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_CHECK(hipblasRotBatchedFn,
                       (handle,
                        N,
                        dx.ptr_on_device(),
                        incx,
                        dy.ptr_on_device(),
                        incy,
                        dc,
                        ds,
                        batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasRotBatchedModel{}.log_args<T>(std::cout,
                                             arg,
                                             gpu_time_used,
                                             rot_gflop_count<T, T, U, V>(N),
                                             rot_gbyte_count<T>(N),
                                             hipblas_error_host,
                                             hipblas_error_device);
    }
}
