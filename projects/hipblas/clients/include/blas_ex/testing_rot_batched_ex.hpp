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

using hipblasRotBatchedExModel = ArgumentModel<e_a_type,
                                               e_b_type,
                                               e_c_type,
                                               e_compute_type,
                                               e_N,
                                               e_incx,
                                               e_incy,
                                               e_batch_count>;

inline void testname_rot_batched_ex(const Arguments& arg, std::string& name)
{
    hipblasRotBatchedExModel{}.test_name(arg, name);
}

template <typename Tx, typename Ty = Tx, typename Tcs = Ty, typename Tex = Tcs>
void testing_rot_batched_ex_bad_arg(const Arguments& arg)
{
    auto hipblasRotBatchedExFn
        = arg.api == FORTRAN ? hipblasRotBatchedExFortran : hipblasRotBatchedEx;
    auto hipblasRotBatchedExFn_64
        = arg.api == FORTRAN_64 ? hipblasRotBatchedEx_64Fortran : hipblasRotBatchedEx_64;

    hipDataType xType         = arg.a_type;
    hipDataType yType         = arg.b_type;
    hipDataType csType        = arg.c_type;
    hipDataType executionType = arg.compute_type;

    int64_t N           = 100;
    int64_t incx        = 1;
    int64_t incy        = 1;
    int64_t batch_count = 2;

    hipblasLocalHandle handle(arg);

    device_batch_vector<Tx> dx(N, incx, batch_count);
    device_batch_vector<Ty> dy(N, incy, batch_count);
    device_vector<Tcs>      dc(batch_count);
    device_vector<Tcs>      ds(batch_count);

    DAPI_EXPECT(
        HIPBLAS_STATUS_NOT_INITIALIZED,
        hipblasRotBatchedExFn,
        (nullptr, N, dx, xType, incx, dy, yType, incy, dc, ds, csType, batch_count, executionType));

    if(arg.bad_arg_all)
    {
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasRotBatchedExFn,
                    (handle,
                     N,
                     nullptr,
                     xType,
                     incx,
                     dy,
                     yType,
                     incy,
                     dc,
                     ds,
                     csType,
                     batch_count,
                     executionType));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasRotBatchedExFn,
                    (handle,
                     N,
                     dx,
                     xType,
                     incx,
                     nullptr,
                     yType,
                     incy,
                     dc,
                     ds,
                     csType,
                     batch_count,
                     executionType));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasRotBatchedExFn,
                    (handle,
                     N,
                     dx,
                     xType,
                     incx,
                     dy,
                     yType,
                     incy,
                     nullptr,
                     ds,
                     csType,
                     batch_count,
                     executionType));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasRotBatchedExFn,
                    (handle,
                     N,
                     dx,
                     xType,
                     incx,
                     dy,
                     yType,
                     incy,
                     dc,
                     nullptr,
                     csType,
                     batch_count,
                     executionType));

        // This is a little different than the checks for L2. In rocBLAS implementation n <= 0 is a quick-return success before other arg checks.
        // Here, for 32-bit API, I'm counting on the rollover to return success, and for the 64-bit API I'm passing in invalid
        // pointers to get invalid_value returns
        DAPI_EXPECT((arg.api & c_API_64) ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS,
                    hipblasRotBatchedExFn,
                    (handle,
                     c_i32_overflow,
                     nullptr,
                     xType,
                     1,
                     nullptr,
                     yType,
                     1,
                     nullptr,
                     nullptr,
                     csType,
                     c_i32_overflow,
                     executionType));
    }
}

template <typename Tx, typename Ty = Tx, typename Tcs = Ty, typename Tex = Tcs>
void testing_rot_batched_ex(const Arguments& arg)
{
    auto hipblasRotBatchedExFn
        = arg.api == FORTRAN ? hipblasRotBatchedExFortran : hipblasRotBatchedEx;
    auto hipblasRotBatchedExFn_64
        = arg.api == FORTRAN_64 ? hipblasRotBatchedEx_64Fortran : hipblasRotBatchedEx_64;

    int64_t N           = arg.N;
    int64_t incx        = arg.incx;
    int64_t incy        = arg.incy;
    int64_t batch_count = arg.batch_count;

    hipDataType xType         = arg.a_type;
    hipDataType yType         = arg.b_type;
    hipDataType csType        = arg.c_type;
    hipDataType executionType = arg.compute_type;

    hipblasLocalHandle handle(arg);

    // check to prevent undefined memory allocation error
    if(N <= 0 || batch_count <= 0)
    {
        DAPI_CHECK(hipblasRotBatchedExFn,
                   (handle,
                    N,
                    nullptr,
                    xType,
                    incx,
                    nullptr,
                    yType,
                    incy,
                    nullptr,
                    nullptr,
                    csType,
                    batch_count,
                    executionType));

        return;
    }

    int64_t abs_incx = incx >= 0 ? incx : -incx;
    int64_t abs_incy = incy >= 0 ? incy : -incy;

    double gpu_time_used{0}, hipblas_error_host{0}, hipblas_error_device{0};

    host_batch_vector<Tx> hx_host(N, incx, batch_count);
    host_batch_vector<Ty> hy_host(N, incy, batch_count);
    host_batch_vector<Tx> hx_device(N, incx, batch_count);
    host_batch_vector<Ty> hy_device(N, incy, batch_count);
    host_batch_vector<Tx> hx_cpu(N, incx, batch_count);
    host_batch_vector<Ty> hy_cpu(N, incy, batch_count);
    host_vector<Tcs>      hc(1);
    host_vector<Tcs>      hs(1);

    device_batch_vector<Tx> dx(N, incx, batch_count);
    device_batch_vector<Ty> dy(N, incy, batch_count);
    device_vector<Tcs>      dc(1);
    device_vector<Tcs>      ds(1);

    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(dc.memcheck());
    CHECK_DEVICE_ALLOCATION(ds.memcheck());

    hipblas_init_vector(hx_host, arg, hipblas_client_never_set_nan, true);
    hipblas_init_vector(hy_host, arg, hipblas_client_never_set_nan, false);
    hipblas_init_vector(hc, arg, hipblas_client_never_set_nan, false);
    hipblas_init_vector(hs, arg, hipblas_client_never_set_nan, false);

    hx_device.copy_from(hx_host);
    hx_cpu.copy_from(hx_host);
    hy_device.copy_from(hy_host);
    hy_cpu.copy_from(hy_host);

    CHECK_HIP_ERROR(dx.transfer_from(hx_host));
    CHECK_HIP_ERROR(dy.transfer_from(hy_host));

    CHECK_HIP_ERROR(hipMemcpy(dc, hc, sizeof(Tcs), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(ds, hs, sizeof(Tcs), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        // HIPBLAS
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        DAPI_CHECK(hipblasRotBatchedExFn,
                   (handle,
                    N,
                    dx.ptr_on_device(),
                    xType,
                    incx,
                    dy.ptr_on_device(),
                    yType,
                    incy,
                    hc,
                    hs,
                    csType,
                    batch_count,
                    executionType));

        CHECK_HIP_ERROR(hx_host.transfer_from(dx));
        CHECK_HIP_ERROR(hy_host.transfer_from(dy));
        CHECK_HIP_ERROR(dx.transfer_from(hx_device));
        CHECK_HIP_ERROR(dy.transfer_from(hy_device));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasRotBatchedExFn,
                   (handle,
                    N,
                    dx.ptr_on_device(),
                    xType,
                    incx,
                    dy.ptr_on_device(),
                    yType,
                    incy,
                    dc,
                    ds,
                    csType,
                    batch_count,
                    executionType));

        CHECK_HIP_ERROR(hx_device.transfer_from(dx));
        CHECK_HIP_ERROR(hy_device.transfer_from(dy));

        // CBLAS
        for(int64_t b = 0; b < batch_count; b++)
        {
            ref_rot<Tx, Tcs, Tcs>(N, hx_cpu[b], incx, hy_cpu[b], incy, *hc, *hs);
        }

        if(arg.unit_check)
        {
            unit_check_general<Tx>(1, N, batch_count, abs_incx, hx_cpu, hx_host);
            unit_check_general<Ty>(1, N, batch_count, abs_incy, hy_cpu, hy_host);
            unit_check_general<Tx>(1, N, batch_count, abs_incx, hx_cpu, hx_device);
            unit_check_general<Ty>(1, N, batch_count, abs_incy, hy_cpu, hy_device);
        }
        if(arg.norm_check)
        {
            hipblas_error_host
                = norm_check_general<Tx>('F', 1, N, abs_incx, hx_cpu, hx_host, batch_count);
            hipblas_error_host
                += norm_check_general<Ty>('F', 1, N, abs_incy, hy_cpu, hy_host, batch_count);
            hipblas_error_device
                = norm_check_general<Tx>('F', 1, N, abs_incx, hx_cpu, hx_device, batch_count);
            hipblas_error_device
                += norm_check_general<Ty>('F', 1, N, abs_incy, hy_cpu, hy_device, batch_count);
        }
    }

    if(arg.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(hipblasRotBatchedExFn,
                          (handle,
                           N,
                           dx.ptr_on_device(),
                           xType,
                           incx,
                           dy.ptr_on_device(),
                           yType,
                           incy,
                           dc,
                           ds,
                           csType,
                           batch_count,
                           executionType));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasRotBatchedExModel{}.log_args<Tx>(std::cout,
                                                arg,
                                                gpu_time_used,
                                                rot_gflop_count<Tx, Ty, Tcs, Tcs>(N),
                                                rot_gbyte_count<Tx>(N),
                                                hipblas_error_host,
                                                hipblas_error_device);
    }
}
