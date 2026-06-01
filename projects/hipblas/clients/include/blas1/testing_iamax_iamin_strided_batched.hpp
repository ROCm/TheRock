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

using hipblasIamaxIaminStridedBatchedModel
    = ArgumentModel<e_a_type, e_N, e_incx, e_stride_scale, e_batch_count>;

template <typename T, typename R, typename FUNC>
void testing_iamax_iamin_strided_batched_bad_arg(const Arguments& arg, FUNC func)
{
    hipblasLocalHandle handle(arg);

    int64_t       N           = 100;
    int64_t       incx        = 1;
    int64_t       batch_count = 2;
    hipblasStride stride_x    = N * incx;

    device_strided_batch_vector<T> dx(N, incx, stride_x, batch_count);
    device_vector<R>               d_res(batch_count);

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        // None of these test cases will write to result so using device pointer is fine for both modes
        EXPECT_HIPBLAS_STATUS(func(nullptr, N, dx, incx, stride_x, batch_count, d_res),
                              HIPBLAS_STATUS_NOT_INITIALIZED);
        EXPECT_HIPBLAS_STATUS(func(handle, N, nullptr, incx, stride_x, batch_count, d_res),
                              HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_HIPBLAS_STATUS(func(handle, N, dx, incx, stride_x, batch_count, nullptr),
                              HIPBLAS_STATUS_INVALID_VALUE);
    }
}

template <typename T>
void testing_iamax_strided_batched_bad_arg(const Arguments& arg)
{
    auto hipblasIamaxStridedBatchedFn    = arg.api == FORTRAN ? hipblasIamaxStridedBatched<T, true>
                                                              : hipblasIamaxStridedBatched<T, false>;
    auto hipblasIamaxStridedBatched_64Fn = arg.api == FORTRAN_64
                                               ? hipblasIamaxStridedBatched_64<T, true>
                                               : hipblasIamaxStridedBatched_64<T, false>;

    if(arg.api & c_API_64)
        testing_iamax_iamin_strided_batched_bad_arg<T, int64_t>(arg,
                                                                hipblasIamaxStridedBatched_64Fn);
    else
        testing_iamax_iamin_strided_batched_bad_arg<T, int>(arg, hipblasIamaxStridedBatchedFn);
}

template <typename T>
void testing_iamin_strided_batched_bad_arg(const Arguments& arg)
{
    auto hipblasIaminStridedBatchedFn    = arg.api == FORTRAN ? hipblasIaminStridedBatched<T, true>
                                                              : hipblasIaminStridedBatched<T, false>;
    auto hipblasIaminStridedBatched_64Fn = arg.api == FORTRAN_64
                                               ? hipblasIaminStridedBatched_64<T, true>
                                               : hipblasIaminStridedBatched_64<T, false>;

    if(arg.api & c_API_64)
        testing_iamax_iamin_strided_batched_bad_arg<T, int64_t>(arg,
                                                                hipblasIaminStridedBatched_64Fn);
    else
        testing_iamax_iamin_strided_batched_bad_arg<T, int>(arg, hipblasIaminStridedBatchedFn);
}

template <typename T,
          void REFBLAS_FUNC(int64_t, const T*, int64_t, int64_t*),
          typename R,
          typename FUNC>
void testing_iamax_iamin_strided_batched(const Arguments& arg, FUNC func)
{
    int64_t N            = arg.N;
    int64_t incx         = arg.incx;
    double  stride_scale = arg.stride_scale;
    int64_t batch_count  = arg.batch_count;

    hipblasStride stridex = N * incx * stride_scale;

    hipblasLocalHandle handle(arg);

    // check to prevent undefined memory allocation error
    if(batch_count <= 0 || N <= 0 || incx <= 0)
    {
        // quick return success or invalid value
        int64_t          batches = std::max(int64_t(1), batch_count);
        device_vector<R> d_hipblas_result_0(batches);
        host_vector<R>   h_hipblas_result_0(batches);
        hipblas_init_nan(h_hipblas_result_0.data(), batches);
        CHECK_HIP_ERROR(hipMemcpy(
            d_hipblas_result_0, h_hipblas_result_0, sizeof(R) * batches, hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(
            func(handle, N, nullptr, incx, stridex, batch_count, d_hipblas_result_0));

        if(batch_count > 0)
        {
            host_vector<R> cpu_0(batch_count);
            host_vector<R> gpu_0(batch_count);
            CHECK_HIP_ERROR(hipMemcpy(
                gpu_0, d_hipblas_result_0, sizeof(R) * batch_count, hipMemcpyDeviceToHost));
            unit_check_general<R>(1, batch_count, 1, cpu_0, gpu_0);
        }

        return;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this
    // practice
    host_strided_batch_vector<T> hx(N, incx, stridex, batch_count);
    host_vector<R>               cpu_result(batch_count);
    host_vector<int64_t>         cpu_result_64(batch_count);
    host_vector<R>               hipblas_result_host(batch_count);
    host_vector<R>               hipblas_result_device(batch_count);

    device_strided_batch_vector<T> dx(N, incx, stridex, batch_count);
    device_vector<R>               d_hipblas_result(batch_count);

    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(d_hipblas_result.memcheck());

    // Initial Data on CPU
    hipblas_init_vector(hx, arg, hipblas_client_alpha_sets_nan, true);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    double gpu_time_used{0};
    R      hipblas_error_host = 0, hipblas_error_device = 0;

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
                    HIPBLAS
        =================================================================== */
        // device_pointer
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(func(handle, N, dx, incx, stridex, batch_count, d_hipblas_result));

        CHECK_HIP_ERROR(hipMemcpy(hipblas_result_device,
                                  d_hipblas_result,
                                  sizeof(R) * batch_count,
                                  hipMemcpyDeviceToHost));

        // host_pointer
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(func(handle, N, dx, incx, stridex, batch_count, hipblas_result_host));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int64_t b = 0; b < batch_count; b++)
        {
            REFBLAS_FUNC(N, hx.data() + b * stridex, incx, &(cpu_result_64[b]));
            cpu_result[b] = cpu_result_64[b];
        }

        if(arg.unit_check)
        {
            unit_check_general<R>(1, 1, batch_count, cpu_result.data(), hipblas_result_host.data());
            unit_check_general<R>(
                1, 1, batch_count, cpu_result.data(), hipblas_result_device.data());
        }
        if(arg.norm_check)
        {
            for(int64_t b = 0; b < batch_count; b++)
            {
                hipblas_error_host
                    = std::max(int64_t(hipblas_error_host),
                               int64_t(hipblas_abs(hipblas_result_host[b] - cpu_result[b])));
                hipblas_error_device
                    = std::max(int64_t(hipblas_error_device),
                               int64_t(hipblas_abs(hipblas_result_device[b] - cpu_result[b])));
            }
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

            CHECK_HIPBLAS_ERROR(func(handle, N, dx, incx, stridex, batch_count, d_hipblas_result));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasIamaxIaminStridedBatchedModel{}.log_args<T>(std::cout,
                                                           arg,
                                                           gpu_time_used,
                                                           iamax_gflop_count<T>(N),
                                                           iamax_gbyte_count<T>(N),
                                                           hipblas_error_host,
                                                           hipblas_error_device);
    }
}

inline void testname_iamax_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasIamaxIaminStridedBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_iamax_strided_batched(const Arguments& arg)
{
    auto hipblasIamaxStridedBatchedFn    = arg.api == FORTRAN ? hipblasIamaxStridedBatched<T, true>
                                                              : hipblasIamaxStridedBatched<T, false>;
    auto hipblasIamaxStridedBatched_64Fn = arg.api == FORTRAN_64
                                               ? hipblasIamaxStridedBatched_64<T, true>
                                               : hipblasIamaxStridedBatched_64<T, false>;

    if(arg.api & c_API_64)
        testing_iamax_iamin_strided_batched<T, hipblas_iamax_iamin_ref::iamax<T>, int64_t>(
            arg, hipblasIamaxStridedBatched_64Fn);
    else
        testing_iamax_iamin_strided_batched<T, hipblas_iamax_iamin_ref::iamax<T>, int>(
            arg, hipblasIamaxStridedBatchedFn);
}

inline void testname_iamin_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasIamaxIaminStridedBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_iamin_strided_batched(const Arguments& arg)
{
    auto hipblasIaminStridedBatchedFn    = arg.api == FORTRAN ? hipblasIaminStridedBatched<T, true>
                                                              : hipblasIaminStridedBatched<T, false>;
    auto hipblasIaminStridedBatched_64Fn = arg.api == FORTRAN_64
                                               ? hipblasIaminStridedBatched_64<T, true>
                                               : hipblasIaminStridedBatched_64<T, false>;

    if(arg.api & c_API_64)
        testing_iamax_iamin_strided_batched<T, hipblas_iamax_iamin_ref::iamin<T>, int64_t>(
            arg, hipblasIaminStridedBatched_64Fn);
    else
        testing_iamax_iamin_strided_batched<T, hipblas_iamax_iamin_ref::iamin<T>, int>(
            arg, hipblasIaminStridedBatchedFn);
}
