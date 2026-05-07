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

using hipblasAsumModel = ArgumentModel<e_a_type, e_N, e_incx>;

inline void testname_asum(const Arguments& arg, std::string& name)
{
    hipblasAsumModel{}.test_name(arg, name);
}

template <typename T>
void testing_asum_bad_arg(const Arguments& arg)
{
    using Tr           = real_t<T>;
    bool FORTRAN       = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasAsumFn = FORTRAN ? hipblasAsum<T, Tr, true> : hipblasAsum<T, Tr, false>;
    auto hipblasAsumFn_64
        = arg.api == FORTRAN_64 ? hipblasAsum_64<T, Tr, true> : hipblasAsum_64<T, Tr, false>;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        hipblasLocalHandle handle(arg);
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        int64_t N    = 100;
        int64_t incx = 1;

        // Host-side result invalid for device mode, but shouldn't matter for bad-arg test cases
        Tr res = 10;

        device_vector<T> dx(N * incx);

        DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED, hipblasAsumFn, (nullptr, N, dx, incx, &res));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE, hipblasAsumFn, (handle, N, dx, incx, nullptr));

        // extra tests supported with rocBLAS backend
        if(arg.bad_arg_all)
        {
            DAPI_EXPECT(
                HIPBLAS_STATUS_INVALID_VALUE, hipblasAsumFn, (handle, N, nullptr, incx, &res));
        }
    }
}

template <typename T>
void testing_asum(const Arguments& arg)
{
    using Tr           = real_t<T>;
    bool FORTRAN       = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasAsumFn = FORTRAN ? hipblasAsum<T, Tr, true> : hipblasAsum<T, Tr, false>;
    auto hipblasAsumFn_64
        = arg.api == FORTRAN_64 ? hipblasAsum_64<T, Tr, true> : hipblasAsum_64<T, Tr, false>;

    int64_t N    = arg.N;
    int64_t incx = arg.incx;

    hipblasLocalHandle handle(arg);

    // check to prevent undefined memory allocation error
    if(N <= 0 || incx <= 0)
    {
        device_vector<Tr> d_hipblas_result_0(1);
        host_vector<Tr>   h_hipblas_result_0(1);
        hipblas_init_nan(h_hipblas_result_0.data(), 1);
        CHECK_HIP_ERROR(
            hipMemcpy(d_hipblas_result_0, h_hipblas_result_0, sizeof(Tr), hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasAsumFn, (handle, N, nullptr, incx, d_hipblas_result_0));

        host_vector<Tr> cpu_0(1);
        host_vector<Tr> gpu_0(1);
        CHECK_HIP_ERROR(hipMemcpy(gpu_0, d_hipblas_result_0, sizeof(Tr), hipMemcpyDeviceToHost));
        unit_check_general<Tr>(1, 1, 1, cpu_0, gpu_0);

        return;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hx(N, incx);

    device_vector<T>  dx(N, incx);
    device_vector<Tr> d_hipblas_result(1);

    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(d_hipblas_result.memcheck());

    Tr cpu_result{0}, hipblas_result_host{0}, hipblas_result_device{0};

    double gpu_time_used, hipblas_error_host = 0, hipblas_error_device = 0;

    // Initial Data on CPU
    hipblas_init_vector(hx, arg, hipblas_client_alpha_sets_nan, true);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
                    HIPBLAS
        =================================================================== */
        // hipblasAsum accept both dev/host pointer for the scalar
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasAsumFn, (handle, N, dx, incx, d_hipblas_result));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        DAPI_CHECK(hipblasAsumFn, (handle, N, dx, incx, &hipblas_result_host));

        CHECK_HIP_ERROR(
            hipMemcpy(&hipblas_result_device, d_hipblas_result, sizeof(Tr), hipMemcpyDeviceToHost));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */

        ref_asum<T>(N, hx.data(), incx, &cpu_result);

        // Near check for asum ILP64 bit
        bool near_check = arg.initialization == hipblas_initialization::hpl;
        Tr   abs_error  = hipblas_type_epsilon<Tr> * cpu_result;
        Tr   tolerance  = 20.0;
        abs_error *= tolerance;

        if(arg.unit_check)
        {
            if(near_check)
            {
                near_check_general<Tr>(1, 1, 1, &cpu_result, &hipblas_result_host, abs_error);
                near_check_general<Tr>(1, 1, 1, &cpu_result, &hipblas_result_device, abs_error);
            }
            else
            {
                unit_check_general<Tr>(1, 1, 1, &cpu_result, &hipblas_result_host);
                unit_check_general<Tr>(1, 1, 1, &cpu_result, &hipblas_result_device);
            }
        }
        if(arg.norm_check)
        {
            hipblas_error_host
                = norm_check_general<Tr>('F', 1, 1, 1, &cpu_result, &hipblas_result_host);
            hipblas_error_device
                = norm_check_general<Tr>('F', 1, 1, 1, &cpu_result, &hipblas_result_device);
        }

    } // end of if unit/norm check

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

            DAPI_CHECK(hipblasAsumFn, (handle, N, dx, incx, d_hipblas_result));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasAsumModel{}.log_args<T>(std::cout,
                                       arg,
                                       gpu_time_used,
                                       asum_gflop_count<T>(N),
                                       asum_gbyte_count<T>(N),
                                       hipblas_error_host,
                                       hipblas_error_device);
    }

    return;
}
