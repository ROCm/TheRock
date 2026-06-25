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

using hipblasAxpyBatchedExModel = ArgumentModel<e_a_type,
                                                e_b_type,
                                                e_c_type,
                                                e_compute_type,
                                                e_N,
                                                e_alpha,
                                                e_incx,
                                                e_incy,
                                                e_batch_count>;

inline void testname_axpy_batched_ex(const Arguments& arg, std::string& name)
{
    hipblasAxpyBatchedExModel{}.test_name(arg, name);
}

template <typename Ta, typename Tx = Ta, typename Ty = Tx, typename Tex = Ty>
void testing_axpy_batched_ex_bad_arg(const Arguments& arg)
{
    using Ts = hipblas_internal_type<Ta>;
    auto hipblasAxpyBatchedExFn
        = arg.api == FORTRAN ? hipblasAxpyBatchedExFortran : hipblasAxpyBatchedEx;
    auto hipblasAxpyBatchedExFn_64
        = arg.api == FORTRAN_64 ? hipblasAxpyBatchedEx_64Fortran : hipblasAxpyBatchedEx_64;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        hipblasLocalHandle handle(arg);
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        hipDataType alphaType     = arg.a_type;
        hipDataType xType         = arg.b_type;
        hipDataType yType         = arg.c_type;
        hipDataType executionType = arg.compute_type;

        int64_t N           = 100;
        int64_t incx        = 1;
        int64_t incy        = 1;
        int64_t batch_count = 2;

        device_vector<Ta>       d_alpha(1), d_zero(1);
        device_batch_vector<Tx> dx(N, incx, batch_count);
        device_batch_vector<Ty> dy(N, incy, batch_count);

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

        DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                    hipblasAxpyBatchedExFn,
                    (nullptr,
                     N,
                     alpha,
                     alphaType,
                     dx,
                     xType,
                     incx,
                     dy,
                     yType,
                     incy,
                     batch_count,
                     executionType));

        if(arg.bad_arg_all)
        {
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasAxpyBatchedExFn,
                        (handle,
                         N,
                         nullptr,
                         alphaType,
                         dx,
                         xType,
                         incx,
                         dy,
                         yType,
                         incy,
                         batch_count,
                         executionType));

            // Can only check for nullptr for dx/dy with host mode because
            // device mode may not check as it could be quick-return success
            if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
            {
                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasAxpyBatchedExFn,
                            (handle,
                             N,
                             alpha,
                             alphaType,
                             nullptr,
                             xType,
                             incx,
                             dy,
                             yType,
                             incy,
                             batch_count,
                             executionType));
                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasAxpyBatchedExFn,
                            (handle,
                             N,
                             alpha,
                             alphaType,
                             dx,
                             xType,
                             incx,
                             nullptr,
                             yType,
                             incy,
                             batch_count,
                             executionType));

                DAPI_EXPECT((arg.api & c_API_64) ? HIPBLAS_STATUS_INVALID_VALUE
                                                 : HIPBLAS_STATUS_SUCCESS,
                            hipblasAxpyBatchedExFn,
                            (handle,
                             c_i32_overflow,
                             nullptr,
                             alphaType,
                             nullptr,
                             xType,
                             1,
                             nullptr,
                             yType,
                             incy,
                             c_i32_overflow,
                             executionType));
            }
        }

        DAPI_CHECK(hipblasAxpyBatchedExFn,
                   (handle,
                    0,
                    nullptr,
                    alphaType,
                    nullptr,
                    xType,
                    incx,
                    nullptr,
                    yType,
                    incy,
                    batch_count,
                    executionType));
        DAPI_CHECK(hipblasAxpyBatchedExFn,
                   (handle,
                    N,
                    zero,
                    alphaType,
                    nullptr,
                    xType,
                    incx,
                    nullptr,
                    yType,
                    incy,
                    batch_count,
                    executionType));
        DAPI_CHECK(hipblasAxpyBatchedExFn,
                   (handle,
                    N,
                    nullptr,
                    alphaType,
                    nullptr,
                    xType,
                    incx,
                    nullptr,
                    yType,
                    incy,
                    0,
                    executionType));
    }
}

template <typename Ta, typename Tx = Ta, typename Ty = Tx, typename Tex = Ty>
void testing_axpy_batched_ex(const Arguments& arg)
{
    using Ts = hipblas_internal_type<Ta>;
    auto hipblasAxpyBatchedExFn
        = arg.api == FORTRAN ? hipblasAxpyBatchedExFortran : hipblasAxpyBatchedEx;
    auto hipblasAxpyBatchedExFn_64
        = arg.api == FORTRAN_64 ? hipblasAxpyBatchedEx_64Fortran : hipblasAxpyBatchedEx_64;

    int64_t N           = arg.N;
    int64_t incx        = arg.incx;
    int64_t incy        = arg.incy;
    int64_t batch_count = arg.batch_count;

    hipDataType alphaType     = arg.a_type;
    hipDataType xType         = arg.b_type;
    hipDataType yType         = arg.c_type;
    hipDataType executionType = arg.compute_type;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0 || batch_count <= 0)
    {
        DAPI_CHECK(hipblasAxpyBatchedExFn,
                   (handle,
                    N,
                    nullptr,
                    alphaType,
                    nullptr,
                    xType,
                    incx,
                    nullptr,
                    yType,
                    incy,
                    batch_count,
                    executionType));
        return;
    }

    int64_t abs_incx = incx < 0 ? -incx : incx;
    int64_t abs_incy = incy < 0 ? -incy : incy;

    Ta h_alpha = arg.get_alpha<Ta>();

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_batch_vector<Tx> hx(N, incx, batch_count);
    host_batch_vector<Ty> hy_host(N, incy, batch_count);
    host_batch_vector<Ty> hy_device(N, incy, batch_count);
    host_batch_vector<Ty> hy_cpu(N, incy, batch_count);

    device_batch_vector<Tx> dx(N, incx, batch_count);
    device_batch_vector<Ty> dy(N, incy, batch_count);
    device_vector<Ta>       d_alpha(1);

    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());

    double gpu_time_used{0}, hipblas_error_host{0}, hipblas_error_device{0};

    // Initial Data on CPU
    hipblas_init_vector(hx, arg, hipblas_client_alpha_sets_nan, true);
    hipblas_init_vector(hy_host, arg, hipblas_client_alpha_sets_nan, false);

    hy_device.copy_from(hy_host);
    hy_cpu.copy_from(hy_host);

    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy_host));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(Ta), hipMemcpyHostToDevice));

    /* =====================================================================
         HIPBLAS
    =================================================================== */
    CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
    DAPI_CHECK(hipblasAxpyBatchedExFn,
               (handle,
                N,
                reinterpret_cast<Ts*>(&h_alpha),
                alphaType,
                dx.ptr_on_device(),
                xType,
                incx,
                dy.ptr_on_device(),
                yType,
                incy,
                batch_count,
                executionType));

    CHECK_HIP_ERROR(hy_host.transfer_from(dy));
    CHECK_HIP_ERROR(dy.transfer_from(hy_device));

    CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
    DAPI_CHECK(hipblasAxpyBatchedExFn,
               (handle,
                N,
                d_alpha,
                alphaType,
                dx.ptr_on_device(),
                xType,
                incx,
                dy.ptr_on_device(),
                yType,
                incy,
                batch_count,
                executionType));

    CHECK_HIP_ERROR(hy_device.transfer_from(dy));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int64_t b = 0; b < batch_count; b++)
        {
            ref_axpy(N, h_alpha, hx[b], incx, hy_cpu[b], incy);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<Ty>(1, N, batch_count, abs_incy, hy_cpu, hy_host);
            unit_check_general<Ty>(1, N, batch_count, abs_incy, hy_cpu, hy_device);
        }
        if(arg.norm_check)
        {
            hipblas_error_host
                = norm_check_general<Ty>('F', 1, N, abs_incy, hy_cpu, hy_host, batch_count);
            hipblas_error_device
                = norm_check_general<Ty>('F', 1, N, abs_incy, hy_cpu, hy_device, batch_count);
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

            DAPI_DISPATCH(hipblasAxpyBatchedExFn,
                          (handle,
                           N,
                           d_alpha,
                           alphaType,
                           dx.ptr_on_device(),
                           xType,
                           incx,
                           dy.ptr_on_device(),
                           yType,
                           incy,
                           batch_count,
                           executionType));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasAxpyBatchedExModel{}.log_args<Ta>(std::cout,
                                                 arg,
                                                 gpu_time_used,
                                                 axpy_gflop_count<Ta>(N),
                                                 axpy_gbyte_count<Ta>(N),
                                                 hipblas_error_host,
                                                 hipblas_error_device);
    }
}
