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

using hipblasDotStridedBatchedExModel = ArgumentModel<e_a_type,
                                                      e_b_type,
                                                      e_c_type,
                                                      e_compute_type,
                                                      e_N,
                                                      e_incx,
                                                      e_incy,
                                                      e_stride_scale,
                                                      e_batch_count>;

inline void testname_dot_strided_batched_ex(const Arguments& arg, std::string& name)
{
    hipblasDotStridedBatchedExModel{}.test_name(arg, name);
}

inline void testname_dotc_strided_batched_ex(const Arguments& arg, std::string& name)
{
    hipblasDotStridedBatchedExModel{}.test_name(arg, name);
}

template <typename Tx, typename Ty = Tx, typename Tr = Ty, typename Tex = Tr, bool CONJ = false>
void testing_dot_strided_batched_ex_bad_arg(const Arguments& arg)
{
    auto hipblasDotStridedBatchedExFn
        = arg.api == FORTRAN
              ? (CONJ ? hipblasDotcStridedBatchedExFortran : hipblasDotStridedBatchedExFortran)
              : (CONJ ? hipblasDotcStridedBatchedEx : hipblasDotStridedBatchedEx);
    auto hipblasDotStridedBatchedExFn_64
        = arg.api == FORTRAN_64
              ? (CONJ ? hipblasDotcStridedBatchedEx_64Fortran
                      : hipblasDotStridedBatchedEx_64Fortran)
              : (CONJ ? hipblasDotcStridedBatchedEx_64 : hipblasDotStridedBatchedEx_64);

    hipDataType xType         = arg.a_type;
    hipDataType yType         = arg.b_type;
    hipDataType resultType    = arg.c_type;
    hipDataType executionType = arg.compute_type;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        hipblasLocalHandle handle(arg);
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        int64_t N           = 100;
        int64_t incx        = 1;
        int64_t incy        = 1;
        int64_t batch_count = 2;

        hipblasStride stridex = N * incx;
        hipblasStride stridey = N * incy;

        device_strided_batch_vector<Tx> dx(N, incx, stridex, batch_count);
        device_strided_batch_vector<Ty> dy(N, incy, stridey, batch_count);
        device_vector<Tr>               d_res(batch_count);

        // None of these test cases will write to result so using device pointer is fine for both modes
        DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                    hipblasDotStridedBatchedExFn,
                    (nullptr,
                     N,
                     dx,
                     xType,
                     incx,
                     stridex,
                     dy,
                     yType,
                     incy,
                     stridey,
                     batch_count,
                     d_res,
                     resultType,
                     executionType));

        if(arg.bad_arg_all)
        {
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasDotStridedBatchedExFn,
                        (handle,
                         N,
                         nullptr,
                         xType,
                         incx,
                         stridex,
                         dy,
                         yType,
                         incy,
                         stridey,
                         batch_count,
                         d_res,
                         resultType,
                         executionType));
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasDotStridedBatchedExFn,
                        (handle,
                         N,
                         dx,
                         xType,
                         incx,
                         stridex,
                         nullptr,
                         yType,
                         incy,
                         stridey,
                         batch_count,
                         d_res,
                         resultType,
                         executionType));
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasDotStridedBatchedExFn,
                        (handle,
                         N,
                         dx,
                         xType,
                         incx,
                         stridex,
                         dy,
                         yType,
                         incy,
                         stridey,
                         batch_count,
                         nullptr,
                         resultType,
                         executionType));

            // This is a little different than the checks for L2. In rocBLAS implementation n <= 0 is a quick-return success before other arg checks.
            // Here, for 32-bit API, I'm counting on the rollover to return success, and for the 64-bit API I'm passing in invalid
            // pointers to get invalid_value returns. Result can be nullptr in batched versions as batch_count < 0 will return success without updating result
            DAPI_EXPECT((arg.api & c_API_64) ? HIPBLAS_STATUS_INVALID_VALUE
                                             : HIPBLAS_STATUS_SUCCESS,
                        hipblasDotStridedBatchedExFn,
                        (handle,
                         c_i32_overflow,
                         nullptr,
                         xType,
                         1,
                         stridex,
                         nullptr,
                         yType,
                         incy,
                         stridey,
                         c_i32_overflow,
                         nullptr,
                         resultType,
                         executionType));
        }
    }
}

template <typename Tx, typename Ty = Tx, typename Tr = Ty, typename Tex = Tr>
void testing_dotc_strided_batched_ex_bad_arg(const Arguments& arg)
{
    testing_dot_strided_batched_ex_bad_arg<Tx, Ty, Tr, Tex, true>(arg);
}

template <typename Tx, typename Ty = Tx, typename Tr = Ty, typename Tex = Tr, bool CONJ = false>
void testing_dot_strided_batched_ex(const Arguments& arg)
{
    auto hipblasDotStridedBatchedExFn
        = arg.api == FORTRAN
              ? (CONJ ? hipblasDotcStridedBatchedExFortran : hipblasDotStridedBatchedExFortran)
              : (CONJ ? hipblasDotcStridedBatchedEx : hipblasDotStridedBatchedEx);
    auto hipblasDotStridedBatchedExFn_64
        = arg.api == FORTRAN_64
              ? (CONJ ? hipblasDotcStridedBatchedEx_64Fortran
                      : hipblasDotStridedBatchedEx_64Fortran)
              : (CONJ ? hipblasDotcStridedBatchedEx_64 : hipblasDotStridedBatchedEx_64);

    int64_t N            = arg.N;
    int64_t incx         = arg.incx;
    int64_t incy         = arg.incy;
    double  stride_scale = arg.stride_scale;
    int64_t batch_count  = arg.batch_count;

    int64_t       abs_incx = incx >= 0 ? incx : -incx;
    int64_t       abs_incy = incy >= 0 ? incy : -incy;
    hipblasStride stridex  = N * abs_incx * stride_scale;
    hipblasStride stridey  = N * abs_incy * stride_scale;

    hipblasLocalHandle handle(arg);

    hipDataType xType         = arg.a_type;
    hipDataType yType         = arg.b_type;
    hipDataType resultType    = arg.c_type;
    hipDataType executionType = arg.compute_type;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0 || batch_count <= 0)
    {
        int64_t           batches = std::max(batch_count, int64_t(1));
        device_vector<Tr> d_hipblas_result_0(batches);
        host_vector<Tr>   h_hipblas_result_0(batches);
        hipblas_init_nan(h_hipblas_result_0.data(), batches);
        CHECK_HIP_ERROR(hipMemcpy(
            d_hipblas_result_0, h_hipblas_result_0, sizeof(Tr) * batches, hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasDotStridedBatchedExFn,
                   (handle,
                    N,
                    nullptr,
                    xType,
                    incx,
                    stridex,
                    nullptr,
                    yType,
                    incy,
                    stridey,
                    batch_count,
                    d_hipblas_result_0,
                    resultType,
                    executionType));

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
    host_strided_batch_vector<Tx> hx(N, incx, stridex, batch_count);
    host_strided_batch_vector<Ty> hy(N, incy, stridey, batch_count);
    host_vector<Tr>               h_hipblas_result_host(batch_count);
    host_vector<Tr>               h_hipblas_result_device(batch_count);
    host_vector<Tr>               h_cpu_result(batch_count);

    device_strided_batch_vector<Tx> dx(N, incx, stridex, batch_count);
    device_strided_batch_vector<Ty> dy(N, incy, stridey, batch_count);
    device_vector<Tr>               d_hipblas_result(batch_count);

    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(d_hipblas_result.memcheck());

    double gpu_time_used{0}, hipblas_error_host{0}, hipblas_error_device{0};

    // Initial Data on CPU
    hipblas_init_vector(hx, arg, hipblas_client_alpha_sets_nan, true, false);
    hipblas_init_vector(hy, arg, hipblas_client_alpha_sets_nan, false, true);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        DAPI_CHECK(hipblasDotStridedBatchedExFn,
                   (handle,
                    N,
                    dx,
                    xType,
                    incx,
                    stridex,
                    dy,
                    yType,
                    incy,
                    stridey,
                    batch_count,
                    h_hipblas_result_host,
                    resultType,
                    executionType));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasDotStridedBatchedExFn,
                   (handle,
                    N,
                    dx,
                    xType,
                    incx,
                    stridex,
                    dy,
                    yType,
                    incy,
                    stridey,
                    batch_count,
                    d_hipblas_result,
                    resultType,
                    executionType));

        CHECK_HIP_ERROR(hipMemcpy(h_hipblas_result_device,
                                  d_hipblas_result,
                                  sizeof(Tr) * batch_count,
                                  hipMemcpyDeviceToHost));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int64_t b = 0; b < batch_count; b++)
        {
            (CONJ ? ref_dotc<Tx> : ref_dot<Tx>)(N, hx[b], incx, hy[b], incy, &h_cpu_result[b]);
        }

        if(arg.unit_check)
        {
            if(std::is_same<Tr, hipblasHalf>{})
            {
                double tol = error_tolerance<Tr> * N;
                near_check_general(1,
                                   1,
                                   batch_count,
                                   1,
                                   1,
                                   h_cpu_result.data(),
                                   h_hipblas_result_host.data(),
                                   tol);
                near_check_general(1,
                                   1,
                                   batch_count,
                                   1,
                                   1,
                                   h_cpu_result.data(),
                                   h_hipblas_result_device.data(),
                                   tol);
            }
            else
            {
                unit_check_general<Tr>(1, batch_count, 1, h_cpu_result, h_hipblas_result_host);
                unit_check_general<Tr>(1, batch_count, 1, h_cpu_result, h_hipblas_result_device);
            }
        }
        if(arg.norm_check)
        {
            hipblas_error_host = norm_check_general<Tr>(
                'F', 1, batch_count, 1, h_cpu_result, h_hipblas_result_host);
            hipblas_error_device = norm_check_general<Tr>(
                'F', 1, batch_count, 1, h_cpu_result, h_hipblas_result_device);
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

            DAPI_DISPATCH(hipblasDotStridedBatchedExFn,
                          (handle,
                           N,
                           dx,
                           xType,
                           incx,
                           stridex,
                           dy,
                           yType,
                           incy,
                           stridey,
                           batch_count,
                           d_hipblas_result,
                           resultType,
                           executionType));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasDotStridedBatchedExModel{}.log_args<Tx>(std::cout,
                                                       arg,
                                                       gpu_time_used,
                                                       dot_gflop_count<CONJ, Tx>(N),
                                                       dot_gbyte_count<Tx>(N),
                                                       hipblas_error_host,
                                                       hipblas_error_device);
    }
}

template <typename Tx, typename Ty = Tx, typename Tr = Ty, typename Tex = Tr>
void testing_dotc_strided_batched_ex(const Arguments& arg)
{
    testing_dot_strided_batched_ex<Tx, Ty, Tr, Tex, true>(arg);
}
