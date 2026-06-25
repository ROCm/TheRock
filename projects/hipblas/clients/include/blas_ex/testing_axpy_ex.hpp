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

using hipblasAxpyExModel
    = ArgumentModel<e_a_type, e_b_type, e_c_type, e_compute_type, e_N, e_alpha, e_incx, e_incy>;

inline void testname_axpy_ex(const Arguments& arg, std::string& name)
{
    hipblasAxpyExModel{}.test_name(arg, name);
}

template <typename Ta, typename Tx = Ta, typename Ty = Tx, typename Tex = Ty>
void testing_axpy_ex_bad_arg(const Arguments& arg)
{
    using Ts                = hipblas_internal_type<Ta>;
    auto hipblasAxpyExFn    = arg.api == FORTRAN ? hipblasAxpyExFortran : hipblasAxpyEx;
    auto hipblasAxpyExFn_64 = arg.api == FORTRAN_64 ? hipblasAxpyEx_64Fortran : hipblasAxpyEx_64;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        hipblasLocalHandle handle(arg);
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        hipDataType alphaType     = arg.a_type;
        hipDataType xType         = arg.b_type;
        hipDataType yType         = arg.c_type;
        hipDataType executionType = arg.compute_type;

        int64_t N    = 100;
        int64_t incx = 1;
        int64_t incy = 1;

        device_vector<Ta> d_alpha(1), d_zero(1);
        device_vector<Tx> dx(N, incx);
        device_vector<Ty> dy(N, incy);

        const Ts  h_alpha{1.0f}, h_zero{0.0f};
        const Ts* alpha = &h_alpha;
        const Ts* zero  = &h_zero;

        if(pointer_mode == HIPBLAS_POINTER_MODE_DEVICE)
        {
            CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(h_alpha), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(d_zero, &h_zero, sizeof(h_zero), hipMemcpyHostToDevice));
            alpha = d_alpha;
            zero  = d_zero;
        }

        DAPI_EXPECT(
            HIPBLAS_STATUS_NOT_INITIALIZED,
            hipblasAxpyExFn,
            (nullptr, N, alpha, alphaType, dx, xType, incx, dy, yType, incy, executionType));

        if(arg.bad_arg_all)
        {
            DAPI_EXPECT(
                HIPBLAS_STATUS_INVALID_VALUE,
                hipblasAxpyExFn,
                (handle, N, nullptr, alphaType, dx, xType, incx, dy, yType, incy, executionType));

            // Can only check for nullptr for dx/dy with host mode because
            // device mode may not check as it could be quick-return success
            if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
            {
                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasAxpyExFn,
                            (handle,
                             N,
                             alpha,
                             alphaType,
                             nullptr,
                             xType,
                             incx,
                             dy,
                             yType,
                             incy,
                             executionType));
                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasAxpyExFn,
                            (handle,
                             N,
                             alpha,
                             alphaType,
                             dx,
                             xType,
                             incx,
                             nullptr,
                             yType,
                             incy,
                             executionType));

                // This is a little different than the checks for L2. In rocBLAS implementation n <= 0 is a quick-return success before other arg checks.
                // Here, for 32-bit API, I'm counting on the rollover to return success, and for the 64-bit API I'm passing in invalid
                // pointers to get invalid_value returns
                DAPI_EXPECT((arg.api & c_API_64) ? HIPBLAS_STATUS_INVALID_VALUE
                                                 : HIPBLAS_STATUS_SUCCESS,
                            hipblasAxpyExFn,
                            (handle,
                             c_i32_overflow,
                             nullptr,
                             alphaType,
                             nullptr,
                             xType,
                             1,
                             nullptr,
                             yType,
                             incy,
                             executionType));
            }
        }

        DAPI_CHECK(hipblasAxpyExFn,
                   (handle,
                    0,
                    nullptr,
                    alphaType,
                    nullptr,
                    xType,
                    incx,
                    nullptr,
                    yType,
                    incy,
                    executionType));
        DAPI_CHECK(hipblasAxpyExFn,
                   (handle,
                    N,
                    zero,
                    alphaType,
                    nullptr,
                    xType,
                    incx,
                    nullptr,
                    yType,
                    incy,
                    executionType));
    }
}

template <typename Ta, typename Tx = Ta, typename Ty = Tx, typename Tex = Ty>
void testing_axpy_ex(const Arguments& arg)
{
    using Ts                = hipblas_internal_type<Ta>;
    auto hipblasAxpyExFn    = arg.api == FORTRAN ? hipblasAxpyExFortran : hipblasAxpyEx;
    auto hipblasAxpyExFn_64 = arg.api == FORTRAN_64 ? hipblasAxpyEx_64Fortran : hipblasAxpyEx_64;

    int64_t N    = arg.N;
    int64_t incx = arg.incx;
    int64_t incy = arg.incy;

    hipblasLocalHandle handle(arg);

    hipDataType alphaType     = arg.a_type;
    hipDataType xType         = arg.b_type;
    hipDataType yType         = arg.c_type;
    hipDataType executionType = arg.compute_type;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0)
    {
        DAPI_CHECK(hipblasAxpyExFn,
                   (handle,
                    N,
                    nullptr,
                    alphaType,
                    nullptr,
                    xType,
                    incx,
                    nullptr,
                    yType,
                    incy,
                    executionType));
        return;
    }

    int64_t abs_incx = incx < 0 ? -incx : incx;
    int64_t abs_incy = incy < 0 ? -incy : incy;

    Ta h_alpha = arg.get_alpha<Ta>();

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<Tx> hx(N, incx);
    host_vector<Ty> hy_host(N, incy);
    host_vector<Ty> hy_device(N, incy);
    host_vector<Ty> hy_cpu(N, incy);

    device_vector<Tx> dx(N, incx);
    device_vector<Ty> dy(N, incy);
    device_vector<Ta> d_alpha(1);

    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());

    double gpu_time_used{0}, hipblas_error_host{0}, hipblas_error_device{0};

    // Initial Data on CPU
    hipblas_init_vector(hx, arg, hipblas_client_alpha_sets_nan, true);
    hipblas_init_vector(hy_host, arg, hipblas_client_alpha_sets_nan, false);

    hy_device = hy_host;
    hy_cpu    = hy_host;

    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy_host));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(Ta), hipMemcpyHostToDevice));

    /* =====================================================================
         HIPBLAS
    =================================================================== */
    CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
    DAPI_CHECK(hipblasAxpyExFn,
               (handle,
                N,
                reinterpret_cast<Ts*>(&h_alpha),
                alphaType,
                dx,
                xType,
                incx,
                dy,
                yType,
                incy,
                executionType));

    // copy output from device to CPU
    CHECK_HIP_ERROR(hy_host.transfer_from(dy));
    CHECK_HIP_ERROR(dy.transfer_from(hy_device));

    CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
    DAPI_CHECK(hipblasAxpyExFn,
               (handle, N, d_alpha, alphaType, dx, xType, incx, dy, yType, incy, executionType));

    CHECK_HIP_ERROR(hy_device.transfer_from(dy));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        ref_axpy(N, h_alpha, hx.data(), incx, hy_cpu.data(), incy);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<Ty>(1, N, abs_incy, hy_cpu, hy_host);
            unit_check_general<Ty>(1, N, abs_incy, hy_cpu, hy_device);
        }
        if(arg.norm_check)
        {
            hipblas_error_host   = norm_check_general<Ty>('F', 1, N, abs_incy, hy_cpu, hy_host);
            hipblas_error_device = norm_check_general<Ty>('F', 1, N, abs_incy, hy_cpu, hy_device);
        }

    } // end of if unit check

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
                hipblasAxpyExFn,
                (handle, N, d_alpha, alphaType, dx, xType, incx, dy, yType, incy, executionType));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasAxpyExModel{}.log_args<Ta>(std::cout,
                                          arg,
                                          gpu_time_used,
                                          axpy_gflop_count<Ta>(N),
                                          axpy_gbyte_count<Ta>(N),
                                          hipblas_error_host,
                                          hipblas_error_device);
    }
}
