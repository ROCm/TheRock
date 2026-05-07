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

using hipblasScalStridedBatchedModel
    = ArgumentModel<e_a_type, e_c_type, e_N, e_alpha, e_incx, e_stride_scale, e_batch_count>;

inline void testname_scal_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasScalStridedBatchedModel{}.test_name(arg, name);
}

template <typename T, typename U = T>
void testing_scal_strided_batched_bad_arg(const Arguments& arg)
{
    using Ts     = hipblas_internal_type<U>;
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasScalStridedBatchedFn
        = FORTRAN ? hipblasScalStridedBatched<T, U, true> : hipblasScalStridedBatched<T, U, false>;
    auto hipblasScalStridedBatchedFn_64 = arg.api == FORTRAN_64
                                              ? hipblasScalStridedBatched_64<T, U, true>
                                              : hipblasScalStridedBatched_64<T, U, false>;

    int64_t       N           = 100;
    int64_t       incx        = 1;
    int64_t       batch_count = 2;
    hipblasStride stride_x    = N * incx;
    U             alpha       = (U)0.6;

    hipblasLocalHandle handle(arg);

    device_strided_batch_vector<T> dx(N, incx, stride_x, batch_count);

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        // Notably scal differs from axpy such that x can /never/ be a nullptr, regardless of alpha.

        // None of these test cases will write to result so using device pointer is fine for both modes
        DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                    hipblasScalStridedBatchedFn,
                    (nullptr, N, reinterpret_cast<Ts*>(&alpha), dx, incx, stride_x, batch_count));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasScalStridedBatchedFn,
                    (handle, N, nullptr, dx, incx, stride_x, batch_count));
        DAPI_EXPECT(
            HIPBLAS_STATUS_INVALID_VALUE,
            hipblasScalStridedBatchedFn,
            (handle, N, reinterpret_cast<Ts*>(&alpha), nullptr, incx, stride_x, batch_count));
    }
}

template <typename T, typename U = T>
void testing_scal_strided_batched(const Arguments& arg)
{
    using Ts     = hipblas_internal_type<U>;
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasScalStridedBatchedFn
        = FORTRAN ? hipblasScalStridedBatched<T, U, true> : hipblasScalStridedBatched<T, U, false>;
    auto hipblasScalStridedBatchedFn_64 = arg.api == FORTRAN_64
                                              ? hipblasScalStridedBatched_64<T, U, true>
                                              : hipblasScalStridedBatched_64<T, U, false>;

    int64_t N            = arg.N;
    int64_t incx         = arg.incx;
    int     stride_scale = arg.stride_scale;
    int64_t batch_count  = arg.batch_count;

    int unit_check = arg.unit_check;
    int timing     = arg.timing;

    hipblasStride stride_x = N * incx * stride_scale;

    U alpha = arg.get_alpha<U>();

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0 || incx <= 0 || batch_count <= 0)
    {
        DAPI_CHECK(hipblasScalStridedBatchedFn,
                   (handle, N, nullptr, nullptr, incx, stride_x, batch_count));
        return;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_strided_batch_vector<T> hx(N, incx, stride_x, batch_count);
    host_strided_batch_vector<T> hz(N, incx, stride_x, batch_count);

    device_strided_batch_vector<T> dx(N, incx, stride_x, batch_count);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());

    double gpu_time_used = 0.0, cpu_time_used = 0.0;
    double hipblas_error = 0.0;

    // Initial Data on CPU
    hipblas_init_vector(hx, arg, hipblas_client_alpha_sets_nan, true);

    hz.copy_from(hx);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    if(arg.unit_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        DAPI_CHECK(hipblasScalStridedBatchedFn,
                   (handle, N, reinterpret_cast<Ts*>(&alpha), dx, incx, stride_x, batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hx.transfer_from(dx));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int64_t b = 0; b < batch_count; b++)
        {
            ref_scal<T, U>(N, alpha, hz[b], incx);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, batch_count, incx, stride_x, hz, hx);
        }

    } // end of if unit check

    if(timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_CHECK(hipblasScalStridedBatchedFn,
                       (handle, N, reinterpret_cast<Ts*>(&alpha), dx, incx, stride_x, batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasScalStridedBatchedModel{}.log_args<T>(std::cout,
                                                     arg,
                                                     gpu_time_used,
                                                     scal_gflop_count<T, U>(N),
                                                     scal_gbyte_count<T>(N),
                                                     hipblas_error);
    }
}
