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

using hipblasDotModel = ArgumentModel<e_a_type, e_N, e_incx, e_incy>;

inline void testname_dot(const Arguments& arg, std::string& name)
{
    hipblasDotModel{}.test_name(arg, name);
}

inline void testname_dotc(const Arguments& arg, std::string& name)
{
    hipblasDotModel{}.test_name(arg, name);
}

template <typename T, bool CONJ = false>
void testing_dot_bad_arg(const Arguments& arg)
{
    bool FORTRAN         = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasDotFn    = FORTRAN ? (CONJ ? hipblasDotc<T, true> : hipblasDot<T, true>)
                                   : (CONJ ? hipblasDotc<T, false> : hipblasDot<T, false>);
    auto hipblasDotFn_64 = arg.api == FORTRAN_64
                               ? (CONJ ? hipblasDotc_64<T, true> : hipblasDot_64<T, true>)
                               : (CONJ ? hipblasDotc_64<T, false> : hipblasDot_64<T, false>);

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        hipblasLocalHandle handle(arg);
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        int64_t N    = 100;
        int64_t incx = 1;
        int64_t incy = 1;

        device_vector<T> dx(N, incx);
        device_vector<T> dy(N, incy);
        device_vector<T> d_res(1);

        // None of these test cases will write to result so using device pointer is fine for both modes
        DAPI_EXPECT(
            HIPBLAS_STATUS_NOT_INITIALIZED, hipblasDotFn, (nullptr, N, dx, incx, dy, incy, d_res));

        if(arg.bad_arg_all)
        {
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasDotFn,
                        (handle, N, nullptr, incx, dy, incy, d_res));
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasDotFn,
                        (handle, N, dx, incx, nullptr, incy, d_res));
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasDotFn,
                        (handle, N, dx, incx, dy, incy, nullptr));
        }
    }
}

template <typename T>
void testing_dotc_bad_arg(const Arguments& arg)
{
    testing_dot_bad_arg<T, true>(arg);
}

template <typename T, bool CONJ = false>
void testing_dot(const Arguments& arg)
{
    bool FORTRAN         = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasDotFn    = FORTRAN ? (CONJ ? hipblasDotc<T, true> : hipblasDot<T, true>)
                                   : (CONJ ? hipblasDotc<T, false> : hipblasDot<T, false>);
    auto hipblasDotFn_64 = arg.api == FORTRAN_64
                               ? (CONJ ? hipblasDotc_64<T, true> : hipblasDot_64<T, true>)
                               : (CONJ ? hipblasDotc_64<T, false> : hipblasDot_64<T, false>);

    int64_t N    = arg.N;
    int64_t incx = arg.incx;
    int64_t incy = arg.incy;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0)
    {
        device_vector<T> d_hipblas_result_0(1);
        host_vector<T>   h_hipblas_result_0(1);
        hipblas_init_nan(h_hipblas_result_0.data(), 1);
        CHECK_HIP_ERROR(
            hipMemcpy(d_hipblas_result_0, h_hipblas_result_0, sizeof(T), hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasDotFn, (handle, N, nullptr, incx, nullptr, incy, d_hipblas_result_0));

        host_vector<T> cpu_0(1);
        host_vector<T> gpu_0(1);

        CHECK_HIP_ERROR(hipMemcpy(gpu_0, d_hipblas_result_0, sizeof(T), hipMemcpyDeviceToHost));
        unit_check_general<T>(1, 1, 1, cpu_0, gpu_0);

        return;
    }

    int64_t abs_incx = incx >= 0 ? incx : -incx;
    int64_t abs_incy = incy >= 0 ? incy : -incy;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hx(N, incx);
    host_vector<T> hy(N, incy);

    T                cpu_result, h_hipblas_result_1, h_hipblas_result_2;
    device_vector<T> dx(N, incx);
    device_vector<T> dy(N, incy);
    device_vector<T> d_hipblas_result(1);

    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(d_hipblas_result.memcheck());

    double gpu_time_used{0}, hipblas_error_host{0}, hipblas_error_device{0};

    // Initial Data on CPU
    hipblas_init_vector(hx, arg, hipblas_client_alpha_sets_nan, true, true);
    hipblas_init_vector(hy, arg, hipblas_client_alpha_sets_nan, false);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        // hipblasDot accept both dev/host pointer for the scalar
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasDotFn, (handle, N, dx, incx, dy, incy, d_hipblas_result));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        DAPI_CHECK(hipblasDotFn,
                   (handle,
                    N,
                    dx,
                    incx,
                    dy,
                    incy,
                    reinterpret_cast<hipblas_internal_type<T>*>(&h_hipblas_result_1)));

        CHECK_HIP_ERROR(
            hipMemcpy(&h_hipblas_result_2, d_hipblas_result, sizeof(T), hipMemcpyDeviceToHost));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        (CONJ ? ref_dotc<T> : ref_dot<T>)(N, hx.data(), incx, hy.data(), incy, &cpu_result);

        bool   near_check = arg.initialization == hipblas_initialization::hpl;
        double abs_error  = hipblas_type_epsilon<T> * N;

        if(arg.unit_check)
        {
            if(near_check)
            {
                near_check_general<T>(1, 1, 1, &cpu_result, &h_hipblas_result_1, abs_error);
                near_check_general<T>(1, 1, 1, &cpu_result, &h_hipblas_result_2, abs_error);
            }
            else
            {
                unit_check_general<T>(1, 1, 1, &cpu_result, &h_hipblas_result_1);
                unit_check_general<T>(1, 1, 1, &cpu_result, &h_hipblas_result_2);
            }
        }
        if(arg.norm_check)
        {
            hipblas_error_host
                = norm_check_general<T>('F', 1, 1, 1, &cpu_result, &h_hipblas_result_1);
            hipblas_error_device
                = norm_check_general<T>('F', 1, 1, 1, &cpu_result, &h_hipblas_result_2);
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

            DAPI_CHECK(hipblasDotFn, (handle, N, dx, incx, dy, incy, d_hipblas_result));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasDotModel{}.log_args<T>(std::cout,
                                      arg,
                                      gpu_time_used,
                                      dot_gflop_count<CONJ, T>(N),
                                      dot_gbyte_count<T>(N),
                                      hipblas_error_host,
                                      hipblas_error_device);
    }
}

template <typename T>
void testing_dotc(const Arguments& arg)
{
    testing_dot<T, true>(arg);
}
