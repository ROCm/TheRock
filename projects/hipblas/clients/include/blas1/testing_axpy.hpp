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

using hipblasAxpyModel = ArgumentModel<e_a_type, e_N, e_alpha, e_incx, e_incy>;

inline void testname_axpy(const Arguments& arg, std::string& name)
{
    hipblasAxpyModel{}.test_name(arg, name);
}

template <typename T>
void testing_axpy_bad_arg(const Arguments& arg)
{
    using Ts           = hipblas_internal_type<T>;
    bool FORTRAN       = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasAxpyFn = FORTRAN ? hipblasAxpy<T, true> : hipblasAxpy<T, false>;
    auto hipblasAxpyFn_64
        = arg.api == FORTRAN_64 ? hipblasAxpy_64<T, true> : hipblasAxpy_64<T, false>;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        hipblasLocalHandle handle(arg);
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        int64_t N    = 100;
        int64_t incx = 1;
        int64_t incy = 1;

        device_vector<T> d_alpha(1), d_zero(1);
        device_vector<T> dx(N, incx);
        device_vector<T> dy(N, incy);

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
            HIPBLAS_STATUS_NOT_INITIALIZED, hipblasAxpyFn, (nullptr, N, alpha, dx, incx, dy, incy));

        DAPI_CHECK(hipblasAxpyFn, (handle, 0, nullptr, nullptr, incx, nullptr, incy));
        DAPI_CHECK(hipblasAxpyFn, (handle, N, zero, nullptr, incx, nullptr, incy));

        if(arg.bad_arg_all)
        {
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasAxpyFn,
                        (handle, N, nullptr, dx, incx, dy, incy));

            // Can only check for nullptr for dx/dy with host mode because
            //device mode may not check as it could be quick-return success
            if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
            {
                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasAxpyFn,
                            (handle, N, alpha, nullptr, incx, dy, incy));
                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasAxpyFn,
                            (handle, N, alpha, dx, incx, nullptr, incy));
            }
        }
    }
}

template <typename T>
void testing_axpy(const Arguments& arg)
{
    using Ts           = hipblas_internal_type<T>;
    bool FORTRAN       = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasAxpyFn = FORTRAN ? hipblasAxpy<T, true> : hipblasAxpy<T, false>;
    auto hipblasAxpyFn_64
        = arg.api == FORTRAN_64 ? hipblasAxpy_64<T, true> : hipblasAxpy_64<T, false>;

    int64_t N    = arg.N;
    int64_t incx = arg.incx;
    int64_t incy = arg.incy;

    int64_t abs_incx = incx < 0 ? -incx : incx;
    int64_t abs_incy = incy < 0 ? -incy : incy;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0)
    {
        DAPI_CHECK(hipblasAxpyFn, (handle, N, nullptr, nullptr, incx, nullptr, incy));
        return;
    }

    T alpha = arg.get_alpha<T>();

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hx(N, incx);
    host_vector<T> hx_cpu(N, incx);
    host_vector<T> hy_host(N, incy);
    host_vector<T> hy_device(N, incy);
    host_vector<T> hy_cpu(N, incy);

    device_vector<T> dx(N, incx);
    device_vector<T> dy_host(N, incy);
    device_vector<T> dy_device(N, incy);
    device_vector<T> d_alpha(1);

    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy_host.memcheck());
    CHECK_DEVICE_ALLOCATION(dy_device.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());

    double gpu_time_used{0}, hipblas_error_host{0}, hipblas_error_device{0};

    // Initial Data on CPU
    hipblas_init_vector(hx, arg, hipblas_client_alpha_sets_nan, true);
    hipblas_init_vector(hy_host, arg, hipblas_client_alpha_sets_nan, false);
    hy_device = hy_host;

    // copy vector is easy in STL; hx_cpu = hx: save a copy in hx_cpu which will be output of CPU BLAS
    hx_cpu = hx;
    hy_cpu = hy_host;

    // copy data from CPU to device
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy_host.transfer_from(hy_host));
    CHECK_HIP_ERROR(dy_device.transfer_from(hy_device));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &alpha, sizeof(T), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
                    HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasAxpyFn, (handle, N, d_alpha, dx, incx, dy_device, incy));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        DAPI_CHECK(hipblasAxpyFn,
                   (handle, N, reinterpret_cast<Ts*>(&alpha), dx, incx, dy_host, incy));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hy_host.transfer_from(dy_host));
        CHECK_HIP_ERROR(hy_device.transfer_from(dy_device));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        ref_axpy<T>(N, alpha, hx_cpu.data(), incx, hy_cpu.data(), incy);

        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, abs_incy, hy_cpu.data(), hy_host.data());
            unit_check_general<T>(1, N, abs_incy, hy_cpu.data(), hy_device.data());
        }
        if(arg.norm_check)
        {
            hipblas_error_host
                = norm_check_general<T>('F', 1, N, abs_incy, hy_cpu.data(), hy_host.data());
            hipblas_error_device
                = norm_check_general<T>('F', 1, N, abs_incy, hy_cpu.data(), hy_device.data());
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

            DAPI_CHECK(hipblasAxpyFn, (handle, N, d_alpha, dx, incx, dy_device, incy));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasAxpyModel{}.log_args<T>(std::cout,
                                       arg,
                                       gpu_time_used,
                                       axpy_gflop_count<T>(N),
                                       axpy_gbyte_count<T>(N),
                                       hipblas_error_host,
                                       hipblas_error_device);
    }
}
