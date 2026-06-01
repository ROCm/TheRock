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

using hipblasScalBatchedModel
    = ArgumentModel<e_a_type, e_c_type, e_N, e_alpha, e_incx, e_batch_count>;

inline void testname_scal_batched(const Arguments& arg, std::string& name)
{
    hipblasScalBatchedModel{}.test_name(arg, name);
}

template <typename T, typename U = T>
void testing_scal_batched_bad_arg(const Arguments& arg)
{
    using Ts     = hipblas_internal_type<U>;
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasScalBatchedFn
        = FORTRAN ? hipblasScalBatched<T, U, true> : hipblasScalBatched<T, U, false>;
    auto hipblasScalBatchedFn_64 = arg.api == FORTRAN_64 ? hipblasScalBatched_64<T, U, true>
                                                         : hipblasScalBatched_64<T, U, false>;

    int64_t N           = 100;
    int64_t incx        = 1;
    int64_t batch_count = 2;
    U       alpha       = (U)0.6;

    hipblasLocalHandle handle(arg);

    device_batch_vector<T> dx(N, incx, batch_count);

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        // Notably scal differs from axpy such that x can /never/ be a nullptr, regardless of alpha.

        // None of these test cases will write to result so using host pointer is fine for both modes
        DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                    hipblasScalBatchedFn,
                    (nullptr, N, reinterpret_cast<Ts*>(&alpha), dx, incx, batch_count));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasScalBatchedFn,
                    (handle, N, nullptr, dx, incx, batch_count));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasScalBatchedFn,
                    (handle, N, reinterpret_cast<Ts*>(&alpha), nullptr, incx, batch_count));
    }
}

template <typename T, typename U = T>
void testing_scal_batched(const Arguments& arg)
{
    using Ts     = hipblas_internal_type<U>;
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasScalBatchedFn
        = FORTRAN ? hipblasScalBatched<T, U, true> : hipblasScalBatched<T, U, false>;
    auto hipblasScalBatchedFn_64 = arg.api == FORTRAN_64 ? hipblasScalBatched_64<T, U, true>
                                                         : hipblasScalBatched_64<T, U, false>;

    int64_t N           = arg.N;
    int64_t incx        = arg.incx;
    int64_t batch_count = arg.batch_count;

    int unit_check = arg.unit_check;
    int norm_check = arg.norm_check;
    int timing     = arg.timing;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0 || incx <= 0 || batch_count <= 0)
    {
        DAPI_CHECK(hipblasScalBatchedFn, (handle, N, nullptr, nullptr, incx, batch_count));
        return;
    }

    size_t sizeX         = size_t(N) * incx;
    U      alpha         = arg.get_alpha<U>();
    double gpu_time_used = 0.0, cpu_time_used = 0.0;
    double hipblas_error = 0.0;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_batch_vector<T> hx(N, incx, batch_count);
    host_batch_vector<T> hz(N, incx, batch_count);

    device_batch_vector<T> dx(N, incx, batch_count);

    CHECK_DEVICE_ALLOCATION(dx.memcheck());

    hipblas_init_vector(hx, arg, hipblas_client_alpha_sets_nan, true);
    hz.copy_from(hx);

    CHECK_HIP_ERROR(dx.transfer_from(hx));

    if(unit_check || norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        DAPI_CHECK(
            hipblasScalBatchedFn,
            (handle, N, reinterpret_cast<Ts*>(&alpha), dx.ptr_on_device(), incx, batch_count));

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
        if(unit_check)
        {
            unit_check_general<T>(1, N, batch_count, incx, hz, hx);
        }
        if(norm_check)
        {
            hipblas_error = norm_check_general<T>('F', 1, N, incx, hz, hx, batch_count);
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

            DAPI_CHECK(
                hipblasScalBatchedFn,
                (handle, N, reinterpret_cast<Ts*>(&alpha), dx.ptr_on_device(), incx, batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasScalBatchedModel{}.log_args<T>(std::cout,
                                              arg,
                                              gpu_time_used,
                                              scal_gflop_count<T, U>(N),
                                              scal_gbyte_count<T>(N),
                                              hipblas_error);
    }
}
