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

using hipblasCopyModel = ArgumentModel<e_a_type, e_N, e_incx, e_incy>;

inline void testname_copy(const Arguments& arg, std::string& name)
{
    hipblasCopyModel{}.test_name(arg, name);
}

template <typename T>
void testing_copy_bad_arg(const Arguments& arg)
{
    bool FORTRAN       = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasCopyFn = FORTRAN ? hipblasCopy<T, true> : hipblasCopy<T, false>;
    auto hipblasCopyFn_64
        = arg.api == FORTRAN_64 ? hipblasCopy_64<T, true> : hipblasCopy_64<T, false>;

    hipblasLocalHandle handle(arg);

    int64_t N    = 100;
    int64_t incx = 1;
    int64_t incy = 1;

    device_vector<T> dx(N, incx);
    device_vector<T> dy(N, incy);

    DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED, hipblasCopyFn, (nullptr, N, dx, incx, dy, incy));

    if(arg.bad_arg_all)
    {
        DAPI_EXPECT(
            HIPBLAS_STATUS_INVALID_VALUE, hipblasCopyFn, (handle, N, nullptr, incx, dy, incy));
        DAPI_EXPECT(
            HIPBLAS_STATUS_INVALID_VALUE, hipblasCopyFn, (handle, N, dx, incx, nullptr, incy));
    }
}

template <typename T>
void testing_copy(const Arguments& arg)
{
    bool FORTRAN       = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasCopyFn = FORTRAN ? hipblasCopy<T, true> : hipblasCopy<T, false>;
    auto hipblasCopyFn_64
        = arg.api == FORTRAN_64 ? hipblasCopy_64<T, true> : hipblasCopy_64<T, false>;

    int64_t N    = arg.N;
    int64_t incx = arg.incx;
    int64_t incy = arg.incy;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0)
    {
        DAPI_CHECK(hipblasCopyFn, (handle, N, nullptr, incx, nullptr, incy));
        return;
    }

    int64_t abs_incx = incx >= 0 ? incx : -incx;
    int64_t abs_incy = incy >= 0 ? incy : -incy;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hx(N, incx);
    host_vector<T> hy(N, incy);
    host_vector<T> hx_cpu(N, incy);
    host_vector<T> hy_cpu(N, incy);

    // allocate memory on device
    device_vector<T> dx(N, incx);
    device_vector<T> dy(N, incy);

    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());

    double hipblas_error = 0.0;
    double gpu_time_used = 0.0;

    // Initial Data on CPU
    hipblas_init_vector(hx, arg, hipblas_client_alpha_sets_nan, true);
    hipblas_init_vector(hy, arg, hipblas_client_alpha_sets_nan, false);

    hx_cpu = hx;
    hy_cpu = hy;

    // copy data from CPU to device
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
                    HIPBLAS
        =================================================================== */
        DAPI_CHECK(hipblasCopyFn, (handle, N, dx, incx, dy, incy));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hy.transfer_from(dy));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        // TODO: remove casts
        ref_copy<T>(N, hx_cpu.data(), incx, hy_cpu.data(), incy);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, abs_incy, hy_cpu.data(), hy.data());
        }
        if(arg.norm_check)
        {
            hipblas_error = norm_check_general<T>('F', 1, N, abs_incy, hy_cpu, hy);
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

            DAPI_CHECK(hipblasCopyFn, (handle, N, dx, incx, dy, incy));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasCopyModel{}.log_args<T>(std::cout,
                                       arg,
                                       gpu_time_used,
                                       copy_gflop_count<T>(N),
                                       copy_gbyte_count<T>(N),
                                       hipblas_error);
    }
}
