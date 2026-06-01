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

using hipblasScalExModel = ArgumentModel<e_a_type, e_b_type, e_compute_type, e_N, e_alpha, e_incx>;

inline void testname_scal_ex(const Arguments& arg, std::string& name)
{
    hipblasScalExModel{}.test_name(arg, name);
}

template <typename Ta, typename Tx = Ta, typename Tex = Tx>
void testing_scal_ex_bad_arg(const Arguments& arg)
{
    using Ts                = hipblas_internal_type<Ta>;
    auto hipblasScalExFn    = arg.api == FORTRAN ? hipblasScalExFortran : hipblasScalEx;
    auto hipblasScalExFn_64 = arg.api == FORTRAN_64 ? hipblasScalEx_64Fortran : hipblasScalEx_64;

    hipDataType alphaType     = arg.a_type;
    hipDataType xType         = arg.b_type;
    hipDataType executionType = arg.compute_type;

    int64_t N     = 100;
    int64_t incx  = 1;
    Ta      alpha = Ta(0.6);

    hipblasLocalHandle handle(arg);

    device_vector<Tx> dx(N, incx);

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        // Notably scal differs from axpy such that x can /never/ be a nullptr, regardless of alpha.

        // None of these test cases will write to result so using device pointer is fine for both modes
        DAPI_EXPECT(
            HIPBLAS_STATUS_NOT_INITIALIZED,
            hipblasScalExFn,
            (nullptr, N, reinterpret_cast<Ts*>(&alpha), alphaType, dx, xType, incx, executionType));

        if(arg.bad_arg_all)
        {
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasScalExFn,
                        (handle, N, nullptr, alphaType, dx, xType, incx, executionType));
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasScalExFn,
                        (handle,
                         N,
                         reinterpret_cast<Ts*>(&alpha),
                         alphaType,
                         nullptr,
                         xType,
                         incx,
                         executionType));

            // This is a little different than the checks for L2. In rocBLAS implementation n <= 0 is a quick-return success before other arg checks.
            // Here, for 32-bit API, I'm counting on the rollover to return success, and for the 64-bit API I'm passing in invalid
            // pointers to get invalid_value returns
            DAPI_EXPECT(
                (arg.api & c_API_64) ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS,
                hipblasScalExFn,
                (handle, c_i32_overflow, nullptr, alphaType, nullptr, xType, 1, executionType));
        }
    }
}

template <typename Ta, typename Tx = Ta, typename Tex = Tx>
void testing_scal_ex(const Arguments& arg)
{
    using Ts                = hipblas_internal_type<Ta>;
    auto hipblasScalExFn    = arg.api == FORTRAN ? hipblasScalExFortran : hipblasScalEx;
    auto hipblasScalExFn_64 = arg.api == FORTRAN_64 ? hipblasScalEx_64Fortran : hipblasScalEx_64;

    int64_t N    = arg.N;
    int64_t incx = arg.incx;

    int unit_check = arg.unit_check;
    int timing     = arg.timing;
    int norm_check = arg.norm_check;

    Ta h_alpha = arg.get_alpha<Ta>();

    hipblasLocalHandle handle(arg);

    hipDataType alphaType     = arg.a_type;
    hipDataType xType         = arg.b_type;
    hipDataType executionType = arg.compute_type;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0 || incx <= 0)
    {
        DAPI_CHECK(hipblasScalExFn,
                   (handle, N, nullptr, alphaType, nullptr, xType, incx, executionType));
        return;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<Tx> hx_host(N, incx);
    host_vector<Tx> hx_device(N, incx);
    host_vector<Tx> hx_cpu(N, incx);

    host_vector<Tex> hx_cpu_ex(N, incx);
    host_vector<Tex> hx_host_ex(N, incx);
    host_vector<Tex> hx_device_ex(N, incx);

    device_vector<Tx> dx(N, incx);
    device_vector<Ta> d_alpha(1);

    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());

    double gpu_time_used{0}, hipblas_error_host{0}, hipblas_error_device{0};

    // Initial Data on CPU
    hipblas_init_vector(hx_host, arg, hipblas_client_alpha_sets_nan, true);

    // copy vector is easy in STL; hz = hx: save a copy in hz which will be output of CPU BLAS
    hx_device = hx_cpu = hx_host;

    // copy data from CPU to device
    CHECK_HIP_ERROR(dx.transfer_from(hx_host));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(Ta), hipMemcpyHostToDevice));

    if(unit_check || norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        DAPI_CHECK(hipblasScalExFn,
                   (handle,
                    N,
                    reinterpret_cast<Ts*>(&h_alpha),
                    alphaType,
                    dx,
                    xType,
                    incx,
                    executionType));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hx_host.transfer_from(dx));
        CHECK_HIP_ERROR(dx.transfer_from(hx_device));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasScalExFn,
                   (handle, N, d_alpha, alphaType, dx, xType, incx, executionType));

        CHECK_HIP_ERROR(hx_device.transfer_from(dx));

        /*======================================================================
                    CPU BLAS
        ========================================================================*/
        ref_scal<Tx, Ta>(N, h_alpha, hx_cpu, incx);

        for(size_t i = 0; i < N; i++)
        {
            hx_cpu_ex[i * incx]    = (Tex)hx_cpu[i * incx];
            hx_host_ex[i * incx]   = (Tex)hx_host[i * incx];
            hx_device_ex[i * incx] = (Tex)hx_device[i * incx];
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(unit_check)
        {
            unit_check_general<Tex>(1, N, incx, hx_cpu_ex, hx_host_ex);
            unit_check_general<Tex>(1, N, incx, hx_cpu_ex, hx_device_ex);
        }

        if(norm_check)
        {
            hipblas_error_host = norm_check_general<Tex>('F', 1, N, incx, hx_cpu_ex, hx_host_ex);
            hipblas_error_host = norm_check_general<Tex>('F', 1, N, incx, hx_cpu_ex, hx_device_ex);
        }

    } // end of if unit check

    if(timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(hipblasScalExFn,
                          (handle, N, d_alpha, alphaType, dx, xType, incx, executionType));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasScalExModel{}.log_args<Tx>(std::cout,
                                          arg,
                                          gpu_time_used,
                                          scal_gflop_count<Tx, Ta>(N),
                                          scal_gbyte_count<Tx>(N),
                                          hipblas_error_host,
                                          hipblas_error_device);
    }
}
