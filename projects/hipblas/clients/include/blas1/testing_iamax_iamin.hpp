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

#include "hipblas_iamax_iamin_ref.hpp"
#include "testing_common.hpp"

using hipblasIamaxIaminModel = ArgumentModel<e_a_type, e_N, e_incx>;

template <typename T, typename R, typename FUNC>
void testing_iamax_iamin_bad_arg(const Arguments& arg, FUNC func)
{
    hipblasLocalHandle handle(arg);

    R       N     = 100;
    int64_t incx  = 1;
    R       h_res = -1;

    device_vector<T> dx(N, incx);
    device_vector<R> d_res(1);
    R*               res = d_res;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        // need host-side result for cuBLAS test
        if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
            res = &h_res;

        EXPECT_HIPBLAS_STATUS(func(nullptr, N, dx, incx, res), HIPBLAS_STATUS_NOT_INITIALIZED);

        if(arg.bad_arg_all)
        {
            EXPECT_HIPBLAS_STATUS(func(handle, N, nullptr, incx, res),
                                  HIPBLAS_STATUS_INVALID_VALUE);
            EXPECT_HIPBLAS_STATUS(func(handle, N, dx, incx, nullptr), HIPBLAS_STATUS_INVALID_VALUE);
        }
    }
}

template <typename T>
void testing_iamax_bad_arg(const Arguments& arg)
{
    auto hipblasIamaxFn = arg.api == FORTRAN ? hipblasIamax<T, true> : hipblasIamax<T, false>;
    auto hipblasIamaxFn_64
        = arg.api == FORTRAN_64 ? hipblasIamax_64<T, true> : hipblasIamax_64<T, false>;

    if(arg.api & c_API_64)
        testing_iamax_iamin_bad_arg<T, int64_t>(arg, hipblasIamaxFn_64);
    else
        testing_iamax_iamin_bad_arg<T, int>(arg, hipblasIamaxFn);
}

template <typename T>
void testing_iamin_bad_arg(const Arguments& arg)
{
    auto hipblasIaminFn = arg.api == FORTRAN ? hipblasIamin<T, true> : hipblasIamin<T, false>;
    auto hipblasIaminFn_64
        = arg.api == FORTRAN_64 ? hipblasIamin_64<T, true> : hipblasIamin_64<T, false>;

    if(arg.api & c_API_64)
        testing_iamax_iamin_bad_arg<T, int64_t>(arg, hipblasIaminFn_64);
    else
        testing_iamax_iamin_bad_arg<T, int>(arg, hipblasIaminFn);
}

template <typename T,
          void REFBLAS_FUNC(int64_t, const T*, int64_t, int64_t*),
          typename R,
          typename FUNC>
void testing_iamax_iamin(const Arguments& arg, FUNC func)
{
    int64_t N    = arg.N;
    int64_t incx = arg.incx;

    hipblasLocalHandle handle(arg);

    // check to prevent undefined memory allocation error
    if(N <= 0 || incx <= 0)
    {
        device_vector<R> d_hipblas_result_0(1);
        host_vector<R>   h_hipblas_result_0(1);
        hipblas_init_nan(h_hipblas_result_0.data(), 1);
        CHECK_HIP_ERROR(
            hipMemcpy(d_hipblas_result_0, h_hipblas_result_0, sizeof(R), hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(func(handle, N, nullptr, incx, d_hipblas_result_0));

        host_vector<R> cpu_0(1);
        host_vector<R> gpu_0(1);
        CHECK_HIP_ERROR(hipMemcpy(gpu_0, d_hipblas_result_0, sizeof(R), hipMemcpyDeviceToHost));
        unit_check_general<R>(1, 1, 1, cpu_0, gpu_0);

        return;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this
    // practice
    host_vector<T> hx(N, incx);
    R              cpu_result{0}, hipblas_result_host{0}, hipblas_result_device{0};

    device_vector<T> dx(N, incx);
    device_vector<R> d_hipblas_result(1);

    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(d_hipblas_result.memcheck());

    // Initial Data on CPU
    hipblas_init_vector(hx, arg, hipblas_client_alpha_sets_nan, true);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    double gpu_time_used{0};
    R      hipblas_error_host{0}, hipblas_error_device{0};

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
                    HIPBLAS
        =================================================================== */
        // device_pointer
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(func(handle, N, dx, incx, d_hipblas_result));

        CHECK_HIP_ERROR(
            hipMemcpy(&hipblas_result_device, d_hipblas_result, sizeof(R), hipMemcpyDeviceToHost));

        // host_pointer
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(func(handle, N, dx, incx, &hipblas_result_host));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        int64_t result_i64;
        REFBLAS_FUNC(N, hx.data(), incx, &result_i64);
        cpu_result = result_i64;

        if(arg.unit_check)
        {
            unit_check_general<R>(1, 1, 1, &cpu_result, &hipblas_result_host);
            unit_check_general<R>(1, 1, 1, &cpu_result, &hipblas_result_device);
        }
        if(arg.norm_check)
        {
            hipblas_error_host   = hipblas_abs(hipblas_result_host - cpu_result);
            hipblas_error_device = hipblas_abs(hipblas_result_device - cpu_result);
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

            CHECK_HIPBLAS_ERROR(func(handle, N, dx, incx, d_hipblas_result));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasIamaxIaminModel{}.log_args<T>(std::cout,
                                             arg,
                                             gpu_time_used,
                                             iamax_gflop_count<T>(N),
                                             iamax_gbyte_count<T>(N),
                                             hipblas_error_host,
                                             hipblas_error_device);
    }
}

inline void testname_iamax(const Arguments& arg, std::string& name)
{
    hipblasIamaxIaminModel{}.test_name(arg, name);
}

template <typename T>
void testing_iamax(const Arguments& arg)
{
    auto hipblasIamaxFn = arg.api == FORTRAN ? hipblasIamax<T, true> : hipblasIamax<T, false>;
    auto hipblasIamaxFn_64
        = arg.api == FORTRAN_64 ? hipblasIamax_64<T, true> : hipblasIamax_64<T, false>;

    if(arg.api & c_API_64)
        testing_iamax_iamin<T, hipblas_iamax_iamin_ref::iamax<T>, int64_t>(arg, hipblasIamaxFn_64);
    else
        testing_iamax_iamin<T, hipblas_iamax_iamin_ref::iamax<T>, int>(arg, hipblasIamaxFn);
}

inline void testname_iamin(const Arguments& arg, std::string& name)
{
    hipblasIamaxIaminModel{}.test_name(arg, name);
}

template <typename T>
void testing_iamin(const Arguments& arg)
{
    auto hipblasIaminFn = arg.api == FORTRAN ? hipblasIamin<T, true> : hipblasIamin<T, false>;
    auto hipblasIaminFn_64
        = arg.api == FORTRAN_64 ? hipblasIamin_64<T, true> : hipblasIamin_64<T, false>;

    if(arg.api & c_API_64)
        testing_iamax_iamin<T, hipblas_iamax_iamin_ref::iamin<T>, int64_t>(arg, hipblasIaminFn_64);
    else
        testing_iamax_iamin<T, hipblas_iamax_iamin_ref::iamin<T>, int>(arg, hipblasIaminFn);
}
