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

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

/* ============================================================================================ */

using hipblasSprBatchedModel = ArgumentModel<e_a_type, e_uplo, e_N, e_alpha, e_incx, e_batch_count>;

inline void testname_spr_batched(const Arguments& arg, std::string& name)
{
    hipblasSprBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_spr_batched_bad_arg(const Arguments& arg)
{
    using Ts = hipblas_internal_type<T>;
    auto hipblasSprBatchedFn
        = arg.api == FORTRAN ? hipblasSprBatched<T, true> : hipblasSprBatched<T, false>;
    auto hipblasSprBatchedFn_64
        = arg.api == FORTRAN_64 ? hipblasSprBatched_64<T, true> : hipblasSprBatched_64<T, false>;

    const Ts          h_alpha{1}, h_zero{0};
    const Ts*         alpha = &h_alpha;
    const Ts*         zero  = &h_zero;
    hipblasFillMode_t uplo  = HIPBLAS_FILL_MODE_UPPER;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        hipblasLocalHandle handle(arg);
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        int64_t N           = 100;
        int64_t incx        = 1;
        int64_t batch_count = 2;
        int64_t size_A      = hipblas_packed_matrix_size(N);
        ;

        device_vector<T> d_alpha(1), d_zero(1);

        if(pointer_mode == HIPBLAS_POINTER_MODE_DEVICE)
        {
            CHECK_HIP_ERROR(hipMemcpy(d_alpha, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(d_zero, zero, sizeof(*zero), hipMemcpyHostToDevice));
            alpha = d_alpha;
            zero  = d_zero;
        }

        // Allocate device memory
        device_batch_matrix<T> dAp(1, hipblas_packed_matrix_size(N), 1, batch_count);
        device_batch_vector<T> dx(N, incx, batch_count);

        DAPI_EXPECT(
            HIPBLAS_STATUS_NOT_INITIALIZED,
            hipblasSprBatchedFn,
            (nullptr, uplo, N, alpha, dx.ptr_on_device(), incx, dAp.ptr_on_device(), batch_count));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasSprBatchedFn,
                    (handle,
                     HIPBLAS_FILL_MODE_FULL,
                     N,
                     alpha,
                     dx.ptr_on_device(),
                     incx,
                     dAp.ptr_on_device(),
                     batch_count));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasSprBatchedFn,
                    (handle,
                     (hipblasFillMode_t)HIPBLAS_OP_N,
                     N,
                     alpha,
                     dx.ptr_on_device(),
                     incx,
                     dAp.ptr_on_device(),
                     batch_count));

        DAPI_EXPECT(
            HIPBLAS_STATUS_INVALID_VALUE,
            hipblasSprBatchedFn,
            (handle, uplo, N, nullptr, dx.ptr_on_device(), incx, dAp.ptr_on_device(), batch_count));

        if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
        {
            // For device mode in rocBLAS we don't have checks for dAp, dx as we may be able to quick return
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasSprBatchedFn,
                        (handle, uplo, N, alpha, nullptr, incx, dAp.ptr_on_device(), batch_count));
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasSprBatchedFn,
                        (handle, uplo, N, alpha, dx.ptr_on_device(), incx, nullptr, batch_count));

            // rocBLAS implementation has alpha == 0 quick return after arg checks, so if we're using 32-bit params,
            // this should fail with invalid-value
            // Note that this strategy can't check incx as rocBLAS supports negative. Also depends on implementation so not testing cuBLAS for now
            DAPI_EXPECT((arg.api & c_API_64) ? HIPBLAS_STATUS_SUCCESS
                                             : HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasSprBatchedFn,
                        (handle, uplo, c_i32_overflow, zero, nullptr, 1, nullptr, c_i32_overflow));
        }

        // With N == 0, can have all nullptrs
        DAPI_CHECK(hipblasSprBatchedFn,
                   (handle, uplo, 0, nullptr, nullptr, incx, nullptr, batch_count));
        DAPI_CHECK(hipblasSprBatchedFn, (handle, uplo, N, nullptr, nullptr, incx, nullptr, 0));

        // With alpha == 0, can have all nullptrs
        DAPI_CHECK(hipblasSprBatchedFn,
                   (handle, uplo, N, zero, nullptr, incx, nullptr, batch_count));
    }
}

template <typename T>
void testing_spr_batched(const Arguments& arg)
{
    using Ts = hipblas_internal_type<T>;
    auto hipblasSprBatchedFn
        = arg.api == FORTRAN ? hipblasSprBatched<T, true> : hipblasSprBatched<T, false>;
    auto hipblasSprBatchedFn_64
        = arg.api == FORTRAN_64 ? hipblasSprBatched_64<T, true> : hipblasSprBatched_64<T, false>;

    hipblasFillMode_t uplo        = char2hipblas_fill(arg.uplo);
    int64_t           N           = arg.N;
    int64_t           incx        = arg.incx;
    int64_t           batch_count = arg.batch_count;

    int64_t abs_incx = incx < 0 ? -incx : incx;
    size_t  size_A   = hipblas_packed_matrix_size(N);
    ;

    T h_alpha = arg.get_alpha<T>();

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = N < 0 || !incx || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        DAPI_EXPECT(invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS,
                    hipblasSprBatchedFn,
                    (handle, uplo, N, nullptr, nullptr, incx, nullptr, batch_count));
        return;
    }

    double gpu_time_used{0}, hipblas_error_host{0}, hipblas_error_device{0};

    // Naming: `h` is in CPU (host) memory(eg hAp), `d` is in GPU (device) memory (eg dAp).
    // Allocate host memory
    host_batch_matrix<T> hA(N, N, N, batch_count);
    host_batch_matrix<T> hAp(1, size_A, 1, batch_count);
    host_batch_matrix<T> hAp_host(1, size_A, 1, batch_count);
    host_batch_matrix<T> hAp_device(1, size_A, 1, batch_count);
    host_batch_matrix<T> hAp_cpu(1, size_A, 1, batch_count);
    host_batch_vector<T> hx(N, incx, batch_count);
    host_vector<T>       halpha(1);

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hAp.memcheck());
    CHECK_HIP_ERROR(hAp_host.memcheck());
    CHECK_HIP_ERROR(hAp_device.memcheck());
    CHECK_HIP_ERROR(hAp_cpu.memcheck());
    CHECK_HIP_ERROR(hx.memcheck());

    halpha[0] = h_alpha;

    // Allocate device memory
    device_batch_matrix<T> dAp(1, size_A, 1, batch_count);
    device_batch_vector<T> dx(N, incx, batch_count);
    device_vector<T>       d_alpha(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dAp.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());

    // Initial Data on CPU
    hipblas_init_matrix(hA, arg, hipblas_client_never_set_nan, hipblas_symmetric_matrix, true);
    hipblas_init_vector(hx, arg, hipblas_client_alpha_sets_nan, false, true);

    // helper function to convert Regular matrix `hA` to packed matrix `hAp`
    regular_to_packed(uplo == HIPBLAS_FILL_MODE_UPPER, hA, hAp, N);

    hAp_cpu.copy_from(hAp);

    CHECK_HIP_ERROR(dAp.transfer_from(hAp));
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, halpha, sizeof(T), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        DAPI_CHECK(hipblasSprBatchedFn,
                   (handle,
                    uplo,
                    N,
                    reinterpret_cast<Ts*>(&h_alpha),
                    dx.ptr_on_device(),
                    incx,
                    dAp.ptr_on_device(),
                    batch_count));

        CHECK_HIP_ERROR(hAp_host.transfer_from(dAp));
        CHECK_HIP_ERROR(dAp.transfer_from(hAp));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(
            hipblasSprBatchedFn,
            (handle, uplo, N, d_alpha, dx.ptr_on_device(), incx, dAp.ptr_on_device(), batch_count));

        CHECK_HIP_ERROR(hAp_device.transfer_from(dAp));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(int64_t b = 0; b < batch_count; b++)
        {
            ref_spr<T>(uplo, N, h_alpha, hx[b], incx, hAp_cpu[b]);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(1, size_A, batch_count, 1, hAp_cpu, hAp_host);
            unit_check_general<T>(1, size_A, batch_count, 1, hAp_cpu, hAp_device);
        }
        if(arg.norm_check)
        {
            hipblas_error_host
                = norm_check_general<T>('F', 1, size_A, 1, hAp_cpu, hAp_host, batch_count);
            hipblas_error_device
                = norm_check_general<T>('F', 1, size_A, 1, hAp_cpu, hAp_device, batch_count);
        }
    }

    if(arg.timing)
    {
        CHECK_HIP_ERROR(dAp.transfer_from(hAp));
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(hipblasSprBatchedFn,
                          (handle,
                           uplo,
                           N,
                           d_alpha,
                           dx.ptr_on_device(),
                           incx,
                           dAp.ptr_on_device(),
                           batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasSprBatchedModel{}.log_args<T>(std::cout,
                                             arg,
                                             gpu_time_used,
                                             spr_gflop_count<T>(N),
                                             spr_gbyte_count<T>(N),
                                             hipblas_error_host,
                                             hipblas_error_device);
    }
}
