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

using hipblasAsumStridedBatchedModel
    = ArgumentModel<e_a_type, e_N, e_incx, e_stride_scale, e_batch_count>;

inline void testname_asum_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasAsumStridedBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_asum_strided_batched_bad_arg(const Arguments& arg)
{
    using Tr                            = real_t<T>;
    bool FORTRAN                        = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasAsumStridedBatchedFn    = FORTRAN ? hipblasAsumStridedBatched<T, Tr, true>
                                                  : hipblasAsumStridedBatched<T, Tr, false>;
    auto hipblasAsumStridedBatchedFn_64 = arg.api == FORTRAN_64
                                              ? hipblasAsumStridedBatched_64<T, Tr, true>
                                              : hipblasAsumStridedBatched_64<T, Tr, false>;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        hipblasLocalHandle handle(arg);
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        int64_t       N           = 100;
        int64_t       incx        = 1;
        hipblasStride stride_x    = N;
        int64_t       batch_count = 2;

        // Host-side result invalid for device mode, but shouldn't matter for bad-arg test cases
        Tr res = 10;

        device_vector<T> dx(stride_x * batch_count);

        DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                    hipblasAsumStridedBatchedFn,
                    (nullptr, N, dx, incx, stride_x, batch_count, &res));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasAsumStridedBatchedFn,
                    (handle, N, nullptr, incx, stride_x, batch_count, &res));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasAsumStridedBatchedFn,
                    (handle, N, dx, incx, stride_x, batch_count, nullptr));
    }
}

template <typename T>
void testing_asum_strided_batched(const Arguments& arg)
{
    using Tr                            = real_t<T>;
    bool FORTRAN                        = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasAsumStridedBatchedFn    = FORTRAN ? hipblasAsumStridedBatched<T, Tr, true>
                                                  : hipblasAsumStridedBatched<T, Tr, false>;
    auto hipblasAsumStridedBatchedFn_64 = arg.api == FORTRAN_64
                                              ? hipblasAsumStridedBatched_64<T, Tr, true>
                                              : hipblasAsumStridedBatched_64<T, Tr, false>;

    int64_t N            = arg.N;
    int64_t incx         = arg.incx;
    double  stride_scale = arg.stride_scale;
    int64_t batch_count  = arg.batch_count;

    hipblasStride stridex = N * incx * stride_scale;

    double             gpu_time_used{0}, hipblas_error_host{0}, hipblas_error_device{0};
    hipblasLocalHandle handle(arg);

    // check to prevent undefined memory allocation error
    if(N <= 0 || incx <= 0 || batch_count <= 0)
    {
        int64_t           batches = std::max(int64_t(1), batch_count);
        device_vector<Tr> d_hipblas_result_0(batches);
        host_vector<Tr>   h_hipblas_result_0(batches);
        hipblas_init_nan(h_hipblas_result_0.data(), batches);
        CHECK_HIP_ERROR(hipMemcpy(
            d_hipblas_result_0, h_hipblas_result_0, sizeof(Tr) * batches, hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasAsumStridedBatchedFn,
                   (handle, N, nullptr, incx, stridex, batch_count, d_hipblas_result_0));

        if(batch_count > 0)
        {
            host_vector<Tr> cpu_0(batch_count);
            host_vector<Tr> gpu_0(batch_count);
            CHECK_HIP_ERROR(hipMemcpy(
                gpu_0, d_hipblas_result_0, sizeof(Tr) * batch_count, hipMemcpyDeviceToHost));
            unit_check_general<Tr>(1, batch_count, 1, cpu_0, gpu_0);
        }
        return;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_strided_batch_vector<T> hx(N, incx, stridex, batch_count);
    host_vector<Tr>              cpu_result(batch_count);
    host_vector<Tr>              hipblas_result_host(batch_count);
    host_vector<Tr>              hipblas_result_device(batch_count);

    device_strided_batch_vector<T> dx(N, incx, stridex, batch_count);
    device_vector<Tr>              d_hipblas_result(batch_count);

    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(d_hipblas_result.memcheck());

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
        DAPI_CHECK(hipblasAsumStridedBatchedFn,
                   (handle, N, dx, incx, stridex, batch_count, d_hipblas_result));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        DAPI_CHECK(hipblasAsumStridedBatchedFn,
                   (handle, N, dx, incx, stridex, batch_count, hipblas_result_host));

        CHECK_HIP_ERROR(hipMemcpy(hipblas_result_device,
                                  d_hipblas_result,
                                  sizeof(Tr) * batch_count,
                                  hipMemcpyDeviceToHost));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int64_t b = 0; b < batch_count; b++)
        {
            ref_asum<T>(N, hx.data() + b * stridex, incx, &cpu_result[b]);
        }

        bool near_check = arg.initialization == hipblas_initialization::hpl;

        Tr abs_error = hipblas_type_epsilon<Tr> * cpu_result[0];
        Tr tolerance = 20.0;
        abs_error *= tolerance;

        if(arg.unit_check)
        {
            if(near_check)
            {
                near_check_general<Tr>(
                    batch_count, 1, 1, cpu_result.data(), hipblas_result_host.data(), abs_error);
                near_check_general<Tr>(
                    batch_count, 1, 1, cpu_result.data(), hipblas_result_device.data(), abs_error);
            }
            else
            {
                unit_check_general<Tr>(1, batch_count, 1, cpu_result, hipblas_result_host);
                unit_check_general<Tr>(1, batch_count, 1, cpu_result, hipblas_result_device);
            }
        }
        if(arg.norm_check)
        {
            hipblas_error_host
                = norm_check_general<Tr>('F', 1, batch_count, 1, cpu_result, hipblas_result_host);
            hipblas_error_device
                = norm_check_general<Tr>('F', 1, batch_count, 1, cpu_result, hipblas_result_device);
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

            DAPI_CHECK(hipblasAsumStridedBatchedFn,
                       (handle, N, dx, incx, stridex, batch_count, d_hipblas_result));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasAsumStridedBatchedModel{}.log_args<T>(std::cout,
                                                     arg,
                                                     gpu_time_used,
                                                     asum_gflop_count<T>(N),
                                                     asum_gbyte_count<T>(N),
                                                     hipblas_error_host,
                                                     hipblas_error_device);
    }
}
