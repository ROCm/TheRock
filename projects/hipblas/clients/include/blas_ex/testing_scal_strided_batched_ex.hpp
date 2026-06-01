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

using hipblasScalStridedBatchedExModel = ArgumentModel<e_a_type,
                                                       e_b_type,
                                                       e_compute_type,
                                                       e_N,
                                                       e_alpha,
                                                       e_incx,
                                                       e_stride_scale,
                                                       e_batch_count>;

inline void testname_scal_strided_batched_ex(const Arguments& arg, std::string& name)
{
    hipblasScalStridedBatchedExModel{}.test_name(arg, name);
}

template <typename Ta, typename Tx = Ta, typename Tex = Tx>
void testing_scal_strided_batched_ex_bad_arg(const Arguments& arg)
{
    using Ts = hipblas_internal_type<Ta>;
    auto hipblasScalStridedBatchedExFn
        = arg.api == FORTRAN ? hipblasScalStridedBatchedExFortran : hipblasScalStridedBatchedEx;
    auto hipblasScalStridedBatchedExFn_64 = arg.api == FORTRAN_64
                                                ? hipblasScalStridedBatchedEx_64Fortran
                                                : hipblasScalStridedBatchedEx_64;

    hipDataType alphaType     = arg.a_type;
    hipDataType xType         = arg.b_type;
    hipDataType executionType = arg.compute_type;

    int64_t N           = 100;
    int64_t incx        = 1;
    int64_t batch_count = 2;

    hipblasStride stridex = N * incx;

    Ta alpha = (Ta)0.6;

    hipblasLocalHandle handle(arg);

    device_strided_batch_vector<Tx> dx(N, incx, stridex, batch_count);

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        // Notably scal differs from axpy such that x can /never/ be a nullptr, regardless of alpha.

        // None of these test cases will write to result so using device pointer is fine for both modes
        DAPI_EXPECT(
            HIPBLAS_STATUS_NOT_INITIALIZED,
            hipblasScalStridedBatchedExFn,
            (nullptr, N, &alpha, alphaType, dx, xType, incx, stridex, batch_count, executionType));

        if(arg.bad_arg_all)
        {
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasScalStridedBatchedExFn,
                        (handle,
                         N,
                         nullptr,
                         alphaType,
                         dx,
                         xType,
                         incx,
                         stridex,
                         batch_count,
                         executionType));
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasScalStridedBatchedExFn,
                        (handle,
                         N,
                         &alpha,
                         alphaType,
                         nullptr,
                         xType,
                         incx,
                         stridex,
                         batch_count,
                         executionType));

            // This is a little different than the checks for L2. In rocBLAS implementation n <= 0 is a quick-return success before other arg checks.
            // Here, for 32-bit API, I'm counting on the rollover to return success, and for the 64-bit API I'm passing in invalid
            // pointers to get invalid_value returns
            DAPI_EXPECT((arg.api & c_API_64) ? HIPBLAS_STATUS_INVALID_VALUE
                                             : HIPBLAS_STATUS_SUCCESS,
                        hipblasScalStridedBatchedExFn,
                        (handle,
                         c_i32_overflow,
                         nullptr,
                         alphaType,
                         nullptr,
                         xType,
                         1,
                         stridex,
                         c_i32_overflow,
                         executionType));
        }
    }
}

template <typename Ta, typename Tx = Ta, typename Tex = Tx>
void testing_scal_strided_batched_ex(const Arguments& arg)
{
    using Ts = hipblas_internal_type<Ta>;
    auto hipblasScalStridedBatchedExFn
        = arg.api == FORTRAN ? hipblasScalStridedBatchedExFortran : hipblasScalStridedBatchedEx;
    auto hipblasScalStridedBatchedExFn_64 = arg.api == FORTRAN_64
                                                ? hipblasScalStridedBatchedEx_64Fortran
                                                : hipblasScalStridedBatchedEx_64;

    int64_t N            = arg.N;
    int64_t incx         = arg.incx;
    double  stride_scale = arg.stride_scale;
    int64_t batch_count  = arg.batch_count;

    int unit_check = arg.unit_check;
    int timing     = arg.timing;
    int norm_check = arg.norm_check;

    hipblasStride stridex = size_t(N) * incx * stride_scale;
    size_t        sizeX   = stridex * batch_count;

    Ta h_alpha = arg.get_alpha<Ta>();

    hipblasLocalHandle handle(arg);

    hipDataType alphaType     = arg.a_type;
    hipDataType xType         = arg.b_type;
    hipDataType executionType = arg.compute_type;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0 || incx <= 0 || batch_count <= 0)
    {
        DAPI_CHECK(hipblasScalStridedBatchedExFn,
                   (handle,
                    N,
                    nullptr,
                    alphaType,
                    nullptr,
                    xType,
                    incx,
                    stride_scale,
                    batch_count,
                    executionType));
        return;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_strided_batch_vector<Tx> hx_host(N, incx, stridex, batch_count);
    host_strided_batch_vector<Tx> hx_device(N, incx, stridex, batch_count);
    host_strided_batch_vector<Tx> hx_cpu(N, incx, stridex, batch_count);

    host_strided_batch_vector<Tex> hx_host_ex(N, incx, stridex, batch_count);
    host_strided_batch_vector<Tex> hx_device_ex(N, incx, stridex, batch_count);
    host_strided_batch_vector<Tex> hx_cpu_ex(N, incx, stridex, batch_count);

    device_strided_batch_vector<Tx> dx(N, incx, stridex, batch_count);
    device_vector<Ta>               d_alpha(1);

    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());

    double gpu_time_used{0}, hipblas_error_host{0}, hipblas_error_device{0};

    // Initial Data on CPU
    hipblas_init_vector(hx_host, arg, hipblas_client_alpha_sets_nan, true);

    // copy vector is easy in STL; hz = hx: save a copy in hz which will be output of CPU BLAS
    hx_device.copy_from(hx_host);
    hx_cpu.copy_from(hx_host);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dx.transfer_from(hx_host));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(Ta), hipMemcpyHostToDevice));

    if(unit_check || norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        DAPI_CHECK(hipblasScalStridedBatchedExFn,
                   (handle,
                    N,
                    reinterpret_cast<Ts*>(&h_alpha),
                    alphaType,
                    dx,
                    xType,
                    incx,
                    stridex,
                    batch_count,
                    executionType));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hx_host.transfer_from(dx));
        CHECK_HIP_ERROR(dx.transfer_from(hx_device));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(
            hipblasScalStridedBatchedExFn,
            (handle, N, d_alpha, alphaType, dx, xType, incx, stridex, batch_count, executionType));

        CHECK_HIP_ERROR(hx_device.transfer_from(dx));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int64_t b = 0; b < batch_count; b++)
        {
            ref_scal<Tx, Ta>(N, h_alpha, hx_cpu[b], incx);
        }

        for(size_t b = 0; b < batch_count; b++)
        {
            for(size_t i = 0; i < N; i++)
            {
                hx_host_ex[b][i * incx]   = (Tex)hx_host[b][i * incx];
                hx_device_ex[b][i * incx] = (Tex)hx_device[b][i * incx];
                hx_cpu_ex[b][i * incx]    = (Tex)hx_cpu[b][i * incx];
            }
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(unit_check)
        {
            unit_check_general<Tex>(1, N, batch_count, incx, stridex, hx_cpu_ex, hx_host_ex);
            unit_check_general<Tex>(1, N, batch_count, incx, stridex, hx_cpu_ex, hx_device_ex);
        }

        if(norm_check)
        {
            hipblas_error_host = norm_check_general<Tex>(
                'F', 1, N, incx, stridex, hx_cpu_ex, hx_host_ex, batch_count);
            hipblas_error_device = norm_check_general<Tex>(
                'F', 1, N, incx, stridex, hx_cpu_ex, hx_device_ex, batch_count);
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

            DAPI_DISPATCH(hipblasScalStridedBatchedExFn,
                          (handle,
                           N,
                           d_alpha,
                           alphaType,
                           dx,
                           xType,
                           incx,
                           stridex,
                           batch_count,
                           executionType));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasScalStridedBatchedExModel{}.log_args<Tx>(std::cout,
                                                        arg,
                                                        gpu_time_used,
                                                        scal_gflop_count<Tx, Ta>(N),
                                                        scal_gbyte_count<Tx>(N),
                                                        hipblas_error_host,
                                                        hipblas_error_device);
    }
}
