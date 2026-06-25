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

using hipblasGerBatchedModel
    = ArgumentModel<e_a_type, e_M, e_N, e_alpha, e_incx, e_incy, e_lda, e_batch_count>;

inline void testname_ger_batched(const Arguments& arg, std::string& name)
{
    hipblasGerBatchedModel{}.test_name(arg, name);
}

template <typename T, bool CONJ = false>
void testing_ger_batched_bad_arg(const Arguments& arg)
{
    using Ts     = hipblas_internal_type<T>;
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasGerBatchedFn
        = FORTRAN ? (CONJ ? hipblasGerBatched<T, true, true> : hipblasGerBatched<T, false, true>)
                  : (CONJ ? hipblasGerBatched<T, true, false> : hipblasGerBatched<T, false, false>);

    auto hipblasGerBatchedFn_64
        = arg.api == FORTRAN_64
              ? (CONJ ? hipblasGerBatched_64<T, true, true> : hipblasGerBatched_64<T, false, true>)
              : (CONJ ? hipblasGerBatched_64<T, true, false>
                      : hipblasGerBatched_64<T, false, false>);

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        hipblasLocalHandle handle(arg);
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        int64_t N           = 100;
        int64_t M           = 100;
        int64_t lda         = 100;
        int64_t incx        = 1;
        int64_t incy        = 1;
        int64_t batch_count = 2;

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

        device_batch_vector<T> dA(N * lda, 1, batch_count);
        device_batch_vector<T> dx(N, incx, batch_count);
        device_batch_vector<T> dy(M, incy, batch_count);

        DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                    hipblasGerBatchedFn,
                    (nullptr,
                     M,
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
                    hipblasGerBatchedFn,
                    (handle,
                     M,
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
            // For device mode in rocBLAS we don't have checks for dA, dx, dy as we may be able to quick return
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasGerBatchedFn,
                        (handle,
                         M,
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
                        hipblasGerBatchedFn,
                        (handle,
                         M,
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
                        hipblasGerBatchedFn,
                        (handle,
                         M,
                         N,
                         alpha,
                         dx.ptr_on_device(),
                         incx,
                         dy.ptr_on_device(),
                         incy,
                         nullptr,
                         lda,
                         batch_count));

            // rocBLAS implementation has alpha == 0 quick return after arg checks, so if we're using 32-bit params,
            // this should fail with invalid-value as c_i32_overflow will rollover to -2147483648
            // Note: that this strategy can't check incx as rocBLAS supports negative. Also depends on implementation so not testing cuBLAS for now

            DAPI_EXPECT((arg.api & c_API_64) ? HIPBLAS_STATUS_SUCCESS
                                             : HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasGerBatchedFn,
                        (handle,
                         c_i32_overflow,
                         c_i32_overflow,
                         zero,
                         nullptr,
                         incx,
                         nullptr,
                         incy,
                         nullptr,
                         c_i32_overflow + 1,
                         batch_count));
        }

        // With N == 0, can have all nullptrs
        DAPI_CHECK(
            hipblasGerBatchedFn,
            (handle, M, 0, nullptr, nullptr, incx, nullptr, incy, nullptr, lda, batch_count));

        // With alpha == 0 can have all nullptrs
        DAPI_CHECK(hipblasGerBatchedFn,
                   (handle, M, N, zero, nullptr, incx, nullptr, incy, nullptr, lda, batch_count));
    }
}

template <typename T, bool CONJ>
void testing_ger_batched(const Arguments& arg)
{
    using Ts     = hipblas_internal_type<T>;
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasGerBatchedFn
        = FORTRAN ? (CONJ ? hipblasGerBatched<T, true, true> : hipblasGerBatched<T, false, true>)
                  : (CONJ ? hipblasGerBatched<T, true, false> : hipblasGerBatched<T, false, false>);

    auto hipblasGerBatchedFn_64
        = arg.api == FORTRAN_64
              ? (CONJ ? hipblasGerBatched_64<T, true, true> : hipblasGerBatched_64<T, false, true>)
              : (CONJ ? hipblasGerBatched_64<T, true, false>
                      : hipblasGerBatched_64<T, false, false>);

    int64_t M           = arg.M;
    int64_t N           = arg.N;
    int64_t incx        = arg.incx;
    int64_t incy        = arg.incy;
    int64_t lda         = arg.lda;
    int64_t batch_count = arg.batch_count;

    size_t A_size = lda * N;

    double hipblas_error_host{0}, hipblas_error_device{0};

    T h_alpha = arg.get_alpha<T>();

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = M < 0 || N < 0 || !incx || !incy || lda < M || lda < 1 || batch_count < 0;
    if(invalid_size || !M || !N || !batch_count)
    {
        DAPI_EXPECT(
            invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS,
            hipblasGerBatchedFn,
            (handle, M, N, nullptr, nullptr, incx, nullptr, incy, nullptr, lda, batch_count));
        return;
    }

    // Naming: dA is in GPU (device) memory. hA is in CPU (host) memory
    host_batch_matrix<T> hA(M, N, lda, batch_count);
    host_batch_matrix<T> hA_cpu(M, N, lda, batch_count);
    host_batch_matrix<T> hA_host(M, N, lda, batch_count);
    host_batch_matrix<T> hA_device(M, N, lda, batch_count);
    host_batch_vector<T> hx(M, incx, batch_count);
    host_batch_vector<T> hy(N, incy, batch_count);

    device_batch_matrix<T> dA(M, N, lda, batch_count);
    device_batch_vector<T> dx(M, incx, batch_count);
    device_batch_vector<T> dy(N, incy, batch_count);
    device_vector<T>       d_alpha(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());

    hipblas_init_matrix(hA, arg, hipblas_client_never_set_nan, hipblas_general_matrix, true);
    hipblas_init_vector(hx, arg, hipblas_client_alpha_sets_nan, false, true);
    hipblas_init_vector(hy, arg, hipblas_client_alpha_sets_nan);

    // copy matrix
    hA_cpu.copy_from(hA);
    hA_host.copy_from(hA);
    hA_device.copy_from(hA);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /*=====================================================================
            HIPBLAS
        ======================================================================= */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        DAPI_CHECK(hipblasGerBatchedFn,
                   (handle,
                    M,
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
        DAPI_CHECK(hipblasGerBatchedFn,
                   (handle,
                    M,
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
        for(size_t b = 0; b < batch_count; b++)
        {
            ref_ger<T, CONJ>(M, N, h_alpha, hx[b], incx, hy[b], incy, hA_cpu[b], lda);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(M, N, batch_count, lda, hA_cpu, hA_host);
            unit_check_general<T>(M, N, batch_count, lda, hA_cpu, hA_device);
        }
        if(arg.norm_check)
        {
            hipblas_error_host
                = norm_check_general<T>('F', M, N, lda, hA_cpu, hA_host, batch_count);
            hipblas_error_device
                = norm_check_general<T>('F', M, N, lda, hA_cpu, hA_device, batch_count);
        }
    }

    if(arg.timing)
    {
        double gpu_time_used{0};
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(hipblasGerBatchedFn,
                          (handle,
                           M,
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

        hipblasGerBatchedModel{}.log_args<T>(std::cout,
                                             arg,
                                             gpu_time_used,
                                             ger_gflop_count<T>(M, N),
                                             ger_gbyte_count<T>(M, N),
                                             hipblas_error_host,
                                             hipblas_error_device);
    }
}
