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

using hipblasNrm2Model = ArgumentModel<e_a_type, e_N, e_incx>;

inline void testname_nrm2(const Arguments& arg, std::string& name)
{
    hipblasNrm2Model{}.test_name(arg, name);
}

template <typename T>
void testing_nrm2_bad_arg(const Arguments& arg)
{
    using Tr           = real_t<T>;
    bool FORTRAN       = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasNrm2Fn = FORTRAN ? hipblasNrm2<T, Tr, true> : hipblasNrm2<T, Tr, false>;
    auto hipblasNrm2Fn_64
        = arg.api == FORTRAN_64 ? hipblasNrm2_64<T, Tr, true> : hipblasNrm2_64<T, Tr, false>;

    int64_t N    = 100;
    int64_t incx = 1;

    hipblasLocalHandle handle(arg);

    device_vector<T>  dx(N, incx);
    device_vector<Tr> d_res(1);

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        // None of these test cases will write to result so using device pointer is fine for both modes
        DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED, hipblasNrm2Fn, (nullptr, N, dx, incx, d_res));

        if(arg.bad_arg_all)
        {
            DAPI_EXPECT(
                HIPBLAS_STATUS_INVALID_VALUE, hipblasNrm2Fn, (handle, N, nullptr, incx, d_res));
            DAPI_EXPECT(
                HIPBLAS_STATUS_INVALID_VALUE, hipblasNrm2Fn, (handle, N, dx, incx, nullptr));
        }
    }
}

template <typename T>
void testing_nrm2(const Arguments& arg)
{
    using Tr           = real_t<T>;
    bool FORTRAN       = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasNrm2Fn = FORTRAN ? hipblasNrm2<T, Tr, true> : hipblasNrm2<T, Tr, false>;
    auto hipblasNrm2Fn_64
        = arg.api == FORTRAN_64 ? hipblasNrm2_64<T, Tr, true> : hipblasNrm2_64<T, Tr, false>;

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
        DAPI_CHECK(hipblasNrm2Fn, (handle, N, nullptr, incx, d_hipblas_result_0));

        host_vector<Tr> cpu_0(1);
        host_vector<Tr> gpu_0(1);
        CHECK_HIP_ERROR(hipMemcpy(gpu_0, d_hipblas_result_0, sizeof(Tr), hipMemcpyDeviceToHost));
        unit_check_general<Tr>(1, 1, 1, cpu_0, gpu_0);

        return;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hx(N, incx);
    Tr             cpu_result{0}, hipblas_result_host{0}, hipblas_result_device{0};

    device_vector<T>  dx(N, incx);
    device_vector<Tr> d_hipblas_result(1);

    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(d_hipblas_result.memcheck());

    double gpu_time_used{0}, hipblas_error_host{0}, hipblas_error_device{0};

    // Initial Data on CPU
    hipblas_init_vector(hx, arg, hipblas_client_alpha_sets_nan, true);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    if(arg.unit_check || arg.norm_check)
    {
        // hipblasNrm2 accept both dev/host pointer for the scalar
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasNrm2Fn, (handle, N, dx, incx, d_hipblas_result));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        DAPI_CHECK(hipblasNrm2Fn, (handle, N, dx, incx, &hipblas_result_host));

        CHECK_HIP_ERROR(
            hipMemcpy(&hipblas_result_device, d_hipblas_result, sizeof(Tr), hipMemcpyDeviceToHost));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */

        ref_nrm2<T, Tr>(N, hx.data(), incx, &cpu_result);

        if(arg.unit_check)
        {
            unit_check_nrm2<Tr>(cpu_result, hipblas_result_host, N);
            unit_check_nrm2<Tr>(cpu_result, hipblas_result_device, N);
        }

        if(arg.norm_check)
        {
            hipblas_error_host   = vector_norm_1(1, 1, &cpu_result, &hipblas_result_host);
            hipblas_error_device = vector_norm_1(1, 1, &cpu_result, &hipblas_result_device);
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

            DAPI_CHECK(hipblasNrm2Fn, (handle, N, dx, incx, d_hipblas_result));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasNrm2Model{}.log_args<T>(std::cout,
                                       arg,
                                       gpu_time_used,
                                       nrm2_gflop_count<T>(N),
                                       nrm2_gbyte_count<T>(N),
                                       hipblas_error_host,
                                       hipblas_error_device);
    }
}
