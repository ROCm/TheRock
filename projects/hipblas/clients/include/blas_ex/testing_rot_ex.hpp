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

using hipblasRotExModel
    = ArgumentModel<e_a_type, e_b_type, e_c_type, e_compute_type, e_N, e_incx, e_incy>;

inline void testname_rot_ex(const Arguments& arg, std::string& name)
{
    hipblasRotExModel{}.test_name(arg, name);
}

template <typename Tx, typename Ty = Tx, typename Tcs = Ty, typename Tex = Tcs>
void testing_rot_ex_bad_arg(const Arguments& arg)
{
    auto hipblasRotExFn    = arg.api == FORTRAN ? hipblasRotExFortran : hipblasRotEx;
    auto hipblasRotExFn_64 = arg.api == FORTRAN_64 ? hipblasRotEx_64Fortran : hipblasRotEx_64;

    hipDataType xType         = arg.a_type;
    hipDataType yType         = arg.b_type;
    hipDataType csType        = arg.c_type;
    hipDataType executionType = arg.compute_type;

    int64_t N    = 100;
    int64_t incx = 1;
    int64_t incy = 1;

    hipblasLocalHandle handle(arg);

    device_vector<Tx>  dx(N, incx);
    device_vector<Ty>  dy(N, incy);
    device_vector<Tcs> dc(1);
    device_vector<Tcs> ds(1);

    DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                hipblasRotExFn,
                (nullptr, N, dx, xType, incx, dy, yType, incy, dc, ds, csType, executionType));

    if(arg.bad_arg_all)
    {
        DAPI_EXPECT(
            HIPBLAS_STATUS_INVALID_VALUE,
            hipblasRotExFn,
            (handle, N, nullptr, xType, incx, dy, yType, incy, dc, ds, csType, executionType));
        DAPI_EXPECT(
            HIPBLAS_STATUS_INVALID_VALUE,
            hipblasRotExFn,
            (handle, N, dx, xType, incx, nullptr, yType, incy, dc, ds, csType, executionType));
        DAPI_EXPECT(
            HIPBLAS_STATUS_INVALID_VALUE,
            hipblasRotExFn,
            (handle, N, dx, xType, incx, dy, yType, incy, nullptr, ds, csType, executionType));
        DAPI_EXPECT(
            HIPBLAS_STATUS_INVALID_VALUE,
            hipblasRotExFn,
            (handle, N, dx, xType, incx, dy, yType, incy, dc, nullptr, csType, executionType));

        // This is a little different than the checks for L2. In rocBLAS implementation n <= 0 is a quick-return success before other arg checks.
        // Here, for 32-bit API, I'm counting on the rollover to return success, and for the 64-bit API I'm passing in invalid
        // pointers to get invalid_value returns
        DAPI_EXPECT((arg.api & c_API_64) ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS,
                    hipblasRotExFn,
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
                     executionType));
    }
}

template <typename Tx, typename Ty = Tx, typename Tcs = Ty, typename Tex = Tcs>
void testing_rot_ex(const Arguments& arg)
{
    auto hipblasRotExFn    = arg.api == FORTRAN ? hipblasRotExFortran : hipblasRotEx;
    auto hipblasRotExFn_64 = arg.api == FORTRAN_64 ? hipblasRotEx_64Fortran : hipblasRotEx_64;

    int64_t N    = arg.N;
    int64_t incx = arg.incx;
    int64_t incy = arg.incy;

    hipDataType xType         = arg.a_type;
    hipDataType yType         = arg.b_type;
    hipDataType csType        = arg.c_type;
    hipDataType executionType = arg.compute_type;

    hipblasLocalHandle handle(arg);

    // check to prevent undefined memory allocation error
    if(N <= 0)
    {
        DAPI_CHECK(hipblasRotExFn,
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
                    executionType));
        return;
    }

    double gpu_time_used{0}, hipblas_error_host{0}, hipblas_error_device{0};

    int64_t abs_incx = incx >= 0 ? incx : -incx;
    int64_t abs_incy = incy >= 0 ? incy : -incy;

    device_vector<Tx>  dx(N, incx);
    device_vector<Ty>  dy(N, incy);
    device_vector<Tcs> dc(1);
    device_vector<Tcs> ds(1);

    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(dc.memcheck());
    CHECK_DEVICE_ALLOCATION(ds.memcheck());

    // Initial Data on CPU
    host_vector<Tx>  hx_host(N, incx);
    host_vector<Ty>  hy_host(N, incy);
    host_vector<Tx>  hx_device(N, incx);
    host_vector<Ty>  hy_device(N, incy);
    host_vector<Tx>  hx_cpu(N, incx);
    host_vector<Ty>  hy_cpu(N, incy);
    host_vector<Tcs> hc(1);
    host_vector<Tcs> hs(1);

    // Random alpha (0 - 10)
    host_vector<int> alpha(1);

    hipblas_init_vector(hx_host, arg, hipblas_client_never_set_nan, true);
    hipblas_init_vector(hy_host, arg, hipblas_client_never_set_nan, false);
    hipblas_init_vector(alpha, arg, hipblas_client_never_set_nan, false);
    hipblas_init_vector(hc, arg, hipblas_client_never_set_nan, false);
    hipblas_init_vector(hs, arg, hipblas_client_never_set_nan, false);

    // // cos and sin of alpha (in rads)
    // hc[0] = cos(alpha[0]);
    // hs[0] = sin(alpha[0]);

    // CPU BLAS reference data
    hx_device = hx_host;
    hx_cpu    = hx_host;
    hy_device = hy_host;
    hy_cpu    = hy_host;

    CHECK_HIP_ERROR(dx.transfer_from(hx_host));
    CHECK_HIP_ERROR(dy.transfer_from(hy_host));

    CHECK_HIP_ERROR(hipMemcpy(dc, hc, sizeof(Tcs), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(ds, hs, sizeof(Tcs), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        // HIPBLAS
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        DAPI_CHECK(hipblasRotExFn,
                   (handle, N, dx, xType, incx, dy, yType, incy, hc, hs, csType, executionType));

        CHECK_HIP_ERROR(hx_host.transfer_from(dx));
        CHECK_HIP_ERROR(hy_host.transfer_from(dy));
        CHECK_HIP_ERROR(dx.transfer_from(hx_device));
        CHECK_HIP_ERROR(dy.transfer_from(hy_device));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasRotExFn,
                   (handle, N, dx, xType, incx, dy, yType, incy, dc, ds, csType, executionType));

        CHECK_HIP_ERROR(hx_device.transfer_from(dx));
        CHECK_HIP_ERROR(hy_device.transfer_from(dy));

        // CBLAS
        // TODO: execution type in cblas_rot
        ref_rot<Tx, Tcs, Tcs>(N, hx_cpu.data(), incx, hy_cpu.data(), incy, *hc, *hs);

        if(arg.unit_check)
        {
            unit_check_general<Tx>(1, N, abs_incx, hx_cpu, hx_host);
            unit_check_general<Ty>(1, N, abs_incy, hy_cpu, hy_host);
            unit_check_general<Tx>(1, N, abs_incx, hx_cpu, hx_device);
            unit_check_general<Ty>(1, N, abs_incy, hy_cpu, hy_device);
        }
        if(arg.norm_check)
        {
            hipblas_error_host = norm_check_general<Tx>('F', 1, N, abs_incx, hx_cpu, hx_host);
            hipblas_error_host += norm_check_general<Ty>('F', 1, N, abs_incy, hy_cpu, hy_host);
            hipblas_error_device = norm_check_general<Tx>('F', 1, N, abs_incx, hx_cpu, hx_device);
            hipblas_error_device += norm_check_general<Ty>('F', 1, N, abs_incy, hy_cpu, hy_device);
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

            DAPI_DISPATCH(
                hipblasRotExFn,
                (handle, N, dx, xType, incx, dy, yType, incy, dc, ds, csType, executionType));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasRotExModel{}.log_args<Tx>(std::cout,
                                         arg,
                                         gpu_time_used,
                                         rot_gflop_count<Tx, Ty, Tcs, Tcs>(N),
                                         rot_gbyte_count<Tx>(N),
                                         hipblas_error_host,
                                         hipblas_error_device);
    }
}
