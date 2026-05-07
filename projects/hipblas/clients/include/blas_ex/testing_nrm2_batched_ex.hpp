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

using hipblasNrm2BatchedExModel
    = ArgumentModel<e_a_type, e_b_type, e_compute_type, e_N, e_incx, e_batch_count>;

inline void testname_nrm2_batched_ex(const Arguments& arg, std::string& name)
{
    hipblasNrm2BatchedExModel{}.test_name(arg, name);
}

template <typename Tx, typename Tr = Tx, typename Tex = Tr>
void testing_nrm2_batched_ex_bad_arg(const Arguments& arg)
{
    auto hipblasNrm2BatchedExFn
        = arg.api == FORTRAN ? hipblasNrm2BatchedExFortran : hipblasNrm2BatchedEx;
    auto hipblasNrm2BatchedExFn_64
        = arg.api == FORTRAN_64 ? hipblasNrm2BatchedEx_64Fortran : hipblasNrm2BatchedEx_64;

    int64_t N           = 100;
    int64_t incx        = 1;
    int64_t batch_count = 2;

    hipDataType xType         = arg.a_type;
    hipDataType resultType    = arg.b_type;
    hipDataType executionType = arg.compute_type;

    hipblasLocalHandle handle(arg);

    device_batch_vector<Tx> dx(N, incx, batch_count);
    device_vector<Tr>       d_res(batch_count);

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        // None of these test cases will write to result so using device pointer is fine for both modes
        DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                    hipblasNrm2BatchedExFn,
                    (nullptr, N, dx, xType, incx, batch_count, d_res, resultType, executionType));

        if(arg.bad_arg_all)
        {
            DAPI_EXPECT(
                HIPBLAS_STATUS_INVALID_VALUE,
                hipblasNrm2BatchedExFn,
                (handle, N, nullptr, xType, incx, batch_count, d_res, resultType, executionType));
            DAPI_EXPECT(
                HIPBLAS_STATUS_INVALID_VALUE,
                hipblasNrm2BatchedExFn,
                (handle, N, dx, xType, incx, batch_count, nullptr, resultType, executionType));

            // This is a little different than the checks for L2. In rocBLAS implementation n <= 0 is a quick-return success before other arg checks.
            // Here, for 32-bit API, I'm counting on the rollover to return success, and for the 64-bit API I'm passing in invalid
            // pointers to get invalid_value returns
            DAPI_EXPECT((arg.api & c_API_64) ? HIPBLAS_STATUS_INVALID_VALUE
                                             : HIPBLAS_STATUS_SUCCESS,
                        hipblasNrm2BatchedExFn,
                        (handle,
                         c_i32_overflow,
                         nullptr,
                         xType,
                         1,
                         c_i32_overflow,
                         d_res,
                         resultType,
                         executionType));
        }
    }
}

template <typename Tx, typename Tr = Tx, typename Tex = Tr>
void testing_nrm2_batched_ex(const Arguments& arg)
{
    auto hipblasNrm2BatchedExFn
        = arg.api == FORTRAN ? hipblasNrm2BatchedExFortran : hipblasNrm2BatchedEx;
    auto hipblasNrm2BatchedExFn_64
        = arg.api == FORTRAN_64 ? hipblasNrm2BatchedEx_64Fortran : hipblasNrm2BatchedEx_64;

    int64_t N           = arg.N;
    int64_t incx        = arg.incx;
    int64_t batch_count = arg.batch_count;

    hipDataType xType         = arg.a_type;
    hipDataType resultType    = arg.b_type;
    hipDataType executionType = arg.compute_type;

    hipblasLocalHandle handle(arg);

    // check to prevent undefined memory allocation error
    if(N <= 0 || incx <= 0 || batch_count <= 0)
    {
        int64_t           batches = std::max(batch_count, int64_t(1));
        device_vector<Tr> d_hipblas_result_0(batches);
        host_vector<Tr>   h_hipblas_result_0(batches);
        hipblas_init_nan(h_hipblas_result_0.data(), batches);
        CHECK_HIP_ERROR(hipMemcpy(
            d_hipblas_result_0, h_hipblas_result_0, sizeof(Tr) * batches, hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasNrm2BatchedExFn,
                   (handle,
                    N,
                    nullptr,
                    xType,
                    incx,
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
    host_batch_vector<Tx> hx(N, incx, batch_count);
    host_vector<Tr>       h_cpu_result(batch_count);
    host_vector<Tr>       h_hipblas_result_host(batch_count);
    host_vector<Tr>       h_hipblas_result_device(batch_count);

    device_batch_vector<Tx> dx(N, incx, batch_count);
    device_vector<Tr>       d_hipblas_result(batch_count);

    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(d_hipblas_result.memcheck());

    double gpu_time_used{0}, hipblas_error_host{0}, hipblas_error_device{0};

    // Initial Data on CPU
    hipblas_init_vector(hx, arg, hipblas_client_alpha_sets_nan, true);
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    if(arg.unit_check || arg.norm_check)
    {
        // hipblasNrm2 accept both dev/host pointer for the scalar
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasNrm2BatchedExFn,
                   (handle,
                    N,
                    dx.ptr_on_device(),
                    xType,
                    incx,
                    batch_count,
                    d_hipblas_result,
                    resultType,
                    executionType));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        DAPI_CHECK(hipblasNrm2BatchedExFn,
                   (handle,
                    N,
                    dx.ptr_on_device(),
                    xType,
                    incx,
                    batch_count,
                    h_hipblas_result_host,
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
            ref_nrm2<Tx, Tr>(N, hx[b], incx, &(h_cpu_result[b]));
        }

        double abs_result
            = (float)h_cpu_result[0] > 0.0f ? (double)h_cpu_result[0] : (double)-h_cpu_result[0];
        double abs_error;

        abs_error = abs_result > 0 ? hipblas_type_epsilon<Tr> * N * abs_result
                                   : hipblas_type_epsilon<Tr> * N;

        double tolerance = 2.0; //  accounts for rounding in reduction sum. depends on n.
            //  If test fails, try decreasing n or increasing tolerance.
        abs_error *= tolerance;

        if(arg.unit_check)
        {
            near_check_general<Tr>(
                batch_count, 1, 1, h_cpu_result.data(), h_hipblas_result_host.data(), abs_error);
            near_check_general<Tr>(
                batch_count, 1, 1, h_cpu_result.data(), h_hipblas_result_device.data(), abs_error);
        }
        if(arg.norm_check)
        {
            for(int b = 0; b < batch_count; b++)
            {
                hipblas_error_host
                    = std::max(vector_norm_1(1, 1, &(h_cpu_result[b]), &(h_hipblas_result_host[b])),
                               hipblas_error_host);
                hipblas_error_device = std::max(
                    vector_norm_1(1, 1, &(h_cpu_result[b]), &(h_hipblas_result_device[b])),
                    hipblas_error_device);
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

            DAPI_DISPATCH(hipblasNrm2BatchedExFn,
                          (handle,
                           N,
                           dx.ptr_on_device(),
                           xType,
                           incx,
                           batch_count,
                           d_hipblas_result,
                           resultType,
                           executionType));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasNrm2BatchedExModel{}.log_args<Tx>(std::cout,
                                                 arg,
                                                 gpu_time_used,
                                                 nrm2_gflop_count<Tx>(N),
                                                 nrm2_gbyte_count<Tx>(N),
                                                 hipblas_error_host,
                                                 hipblas_error_device);
    }
}
