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

using hipblasSyr2BatchedModel
    = ArgumentModel<e_a_type, e_uplo, e_N, e_alpha, e_incx, e_incy, e_lda, e_batch_count>;

inline void testname_syr2_batched(const Arguments& arg, std::string& name)
{
    hipblasSyr2BatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_syr2_batched_bad_arg(const Arguments& arg)
{
    using Ts = hipblas_internal_type<T>;
    auto hipblasSyr2BatchedFn
        = arg.api == FORTRAN ? hipblasSyr2Batched<T, true> : hipblasSyr2Batched<T, false>;
    auto hipblasSyr2BatchedFn_64
        = arg.api == FORTRAN_64 ? hipblasSyr2Batched_64<T, true> : hipblasSyr2Batched_64<T, false>;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        hipblasLocalHandle handle(arg);
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        hipblasFillMode_t uplo        = HIPBLAS_FILL_MODE_UPPER;
        int64_t           N           = 100;
        int64_t           lda         = 100;
        int64_t           incx        = 1;
        int64_t           incy        = 1;
        int64_t           batch_count = 2;

        device_vector<T> d_alpha(1), d_zero(1);

        const Ts  h_alpha{1}, h_zero{0};
        const Ts* alpha = &h_alpha;
        const Ts* zero  = &h_zero;

        if(pointer_mode == HIPBLAS_POINTER_MODE_DEVICE)
        {
            CHECK_HIP_ERROR(hipMemcpy(d_alpha, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(d_zero, zero, sizeof(*zero), hipMemcpyHostToDevice));
            alpha = d_alpha;
            zero  = d_zero;
        }

        // Allocate device memory
        device_batch_matrix<T> dA(N, N, lda, batch_count);
        device_batch_vector<T> dx(N, incx, batch_count);
        device_batch_vector<T> dy(N, incy, batch_count);

        DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                    hipblasSyr2BatchedFn,
                    (nullptr,
                     uplo,
                     N,
                     alpha,
                     dx.ptr_on_device(),
                     incx,
                     dy.ptr_on_device(),
                     incy,
                     dA.ptr_on_device(),
                     lda,
                     batch_count));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasSyr2BatchedFn,
                    (handle,
                     HIPBLAS_FILL_MODE_FULL,
                     N,
                     alpha,
                     dx.ptr_on_device(),
                     incx,
                     dy.ptr_on_device(),
                     incy,
                     dA.ptr_on_device(),
                     lda,
                     batch_count));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasSyr2BatchedFn,
                    (handle,
                     (hipblasFillMode_t)HIPBLAS_OP_N,
                     N,
                     alpha,
                     dx.ptr_on_device(),
                     incx,
                     dy.ptr_on_device(),
                     incy,
                     dA.ptr_on_device(),
                     lda,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasSyr2BatchedFn,
                    (handle,
                     uplo,
                     N,
                     nullptr,
                     dx.ptr_on_device(),
                     incx,
                     dy.ptr_on_device(),
                     incy,
                     dA.ptr_on_device(),
                     lda,
                     batch_count));

        if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
        {
            // For device mode in rocBLAS we don't have checks for dA, dx as we may be able to quick return
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasSyr2BatchedFn,
                        (handle,
                         uplo,
                         N,
                         alpha,
                         nullptr,
                         incx,
                         dy.ptr_on_device(),
                         incy,
                         dA.ptr_on_device(),
                         lda,
                         batch_count));
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasSyr2BatchedFn,
                        (handle,
                         uplo,
                         N,
                         alpha,
                         dx.ptr_on_device(),
                         incx,
                         nullptr,
                         incy,
                         dA.ptr_on_device(),
                         lda,
                         batch_count));
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasSyr2BatchedFn,
                        (handle,
                         uplo,
                         N,
                         alpha,
                         dx.ptr_on_device(),
                         incx,
                         dy.ptr_on_device(),
                         incy,
                         nullptr,
                         lda,
                         batch_count));

            // testing the 64-bit interface for n, lda, and batch_count
            DAPI_EXPECT((arg.api & c_API_64) ? HIPBLAS_STATUS_SUCCESS
                                             : HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasSyr2BatchedFn,
                        (handle,
                         uplo,
                         c_i32_overflow,
                         zero,
                         nullptr,
                         incx,
                         nullptr,
                         incy,
                         nullptr,
                         c_i32_overflow,
                         c_i32_overflow));
        }

        // With N == 0, can have all nullptrs
        DAPI_CHECK(
            hipblasSyr2BatchedFn,
            (handle, uplo, 0, nullptr, nullptr, incx, nullptr, incy, nullptr, lda, batch_count));
        DAPI_CHECK(hipblasSyr2BatchedFn,
                   (handle, uplo, N, nullptr, nullptr, incx, nullptr, incy, nullptr, lda, 0));

        // With alpha == 0, can have all nullptrs
        DAPI_CHECK(
            hipblasSyr2BatchedFn,
            (handle, uplo, N, zero, nullptr, incx, nullptr, incy, nullptr, lda, batch_count));
    }
}

template <typename T>
void testing_syr2_batched(const Arguments& arg)
{
    using Ts = hipblas_internal_type<T>;
    auto hipblasSyr2BatchedFn
        = arg.api == FORTRAN ? hipblasSyr2Batched<T, true> : hipblasSyr2Batched<T, false>;
    auto hipblasSyr2BatchedFn_64
        = arg.api == FORTRAN_64 ? hipblasSyr2Batched_64<T, true> : hipblasSyr2Batched_64<T, false>;

    hipblasFillMode_t uplo        = char2hipblas_fill(arg.uplo);
    int64_t           N           = arg.N;
    int64_t           incx        = arg.incx;
    int64_t           incy        = arg.incy;
    int64_t           lda         = arg.lda;
    int64_t           batch_count = arg.batch_count;

    int64_t abs_incx = incx < 0 ? -incx : incx;
    int64_t abs_incy = incy < 0 ? -incy : incy;
    size_t  A_size   = size_t(lda) * N;

    T h_alpha = arg.get_alpha<T>();

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = N < 0 || !incx || !incy || lda < N || lda < 1 || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        DAPI_EXPECT(
            (invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS),
            hipblasSyr2BatchedFn,
            (handle, uplo, N, nullptr, nullptr, incx, nullptr, incy, nullptr, lda, batch_count));
        return;
    }

    double gpu_time_used{0}, hipblas_error_host{0}, hipblas_error_device{0};

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_batch_matrix<T> hA(N, N, lda, batch_count);
    host_batch_matrix<T> hA_cpu(N, N, lda, batch_count);
    host_batch_matrix<T> hA_host(N, N, lda, batch_count);
    host_batch_matrix<T> hA_device(N, N, lda, batch_count);
    host_batch_vector<T> hx(N, incx, batch_count);
    host_batch_vector<T> hy(N, incy, batch_count);

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hA_cpu.memcheck());
    CHECK_HIP_ERROR(hA_host.memcheck());
    CHECK_HIP_ERROR(hA_device.memcheck());
    CHECK_HIP_ERROR(hx.memcheck());
    CHECK_HIP_ERROR(hy.memcheck());

    // Allocate device memory
    device_batch_matrix<T> dA(N, N, lda, batch_count);
    device_batch_vector<T> dx(N, incx, batch_count);
    device_batch_vector<T> dy(N, incy, batch_count);
    device_vector<T>       d_alpha(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());

    // Initial Data on CPU
    hipblas_init_matrix(
        hA, arg, hipblas_client_never_set_nan, hipblas_symmetric_matrix, true, false);
    hipblas_init_vector(hx, arg, hipblas_client_alpha_sets_nan, false, true);
    hipblas_init_vector(hy, arg, hipblas_client_alpha_sets_nan);

    //copy vector
    hA_cpu.copy_from(hA);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        DAPI_CHECK(hipblasSyr2BatchedFn,
                   (handle,
                    uplo,
                    N,
                    reinterpret_cast<Ts*>(&h_alpha),
                    dx.ptr_on_device(),
                    incx,
                    dy.ptr_on_device(),
                    incy,
                    dA.ptr_on_device(),
                    lda,
                    batch_count));

        CHECK_HIP_ERROR(hA_host.transfer_from(dA));
        CHECK_HIP_ERROR(dA.transfer_from(hA));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasSyr2BatchedFn,
                   (handle,
                    uplo,
                    N,
                    d_alpha,
                    dx.ptr_on_device(),
                    incx,
                    dy.ptr_on_device(),
                    incy,
                    dA.ptr_on_device(),
                    lda,
                    batch_count));

        CHECK_HIP_ERROR(hA_device.transfer_from(dA));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(int64_t b = 0; b < batch_count; b++)
        {
            ref_syr2<T>(uplo, N, h_alpha, hx[b], incx, hy[b], incy, hA_cpu[b], lda);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(N, N, batch_count, lda, hA_cpu, hA_host);
            unit_check_general<T>(N, N, batch_count, lda, hA_cpu, hA_device);
        }
        if(arg.norm_check)
        {
            hipblas_error_host
                = norm_check_general<T>('F', N, N, lda, hA_cpu, hA_host, batch_count);
            hipblas_error_device
                = norm_check_general<T>('F', N, N, lda, hA_cpu, hA_device, batch_count);
        }
    }

    if(arg.timing)
    {
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(hipblasSyr2BatchedFn,
                          (handle,
                           uplo,
                           N,
                           d_alpha,
                           dx.ptr_on_device(),
                           incx,
                           dy.ptr_on_device(),
                           incy,
                           dA.ptr_on_device(),
                           lda,
                           batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasSyr2BatchedModel{}.log_args<T>(std::cout,
                                              arg,
                                              gpu_time_used,
                                              syr2_gflop_count<T>(N),
                                              syr2_gbyte_count<T>(N),
                                              hipblas_error_host,
                                              hipblas_error_device);
    }
}
