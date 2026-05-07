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

using hipblasHbmvStridedBatchedModel = ArgumentModel<e_a_type,
                                                     e_uplo,
                                                     e_N,
                                                     e_K,
                                                     e_alpha,
                                                     e_lda,
                                                     e_incx,
                                                     e_beta,
                                                     e_incy,
                                                     e_stride_scale,
                                                     e_batch_count>;

inline void testname_hbmv_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasHbmvStridedBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_hbmv_strided_batched_bad_arg(const Arguments& arg)
{
    using Ts     = hipblas_internal_type<T>;
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasHbmvStridedBatchedFn
        = FORTRAN ? hipblasHbmvStridedBatched<T, true> : hipblasHbmvStridedBatched<T, false>;

    auto hipblasHbmvStridedBatchedFn_64 = arg.api == FORTRAN_64
                                              ? hipblasHbmvStridedBatched_64<T, true>
                                              : hipblasHbmvStridedBatched_64<T, false>;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        hipblasLocalHandle handle(arg);
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        hipblasFillMode_t uplo              = HIPBLAS_FILL_MODE_UPPER;
        int64_t           N                 = 100;
        int64_t           K                 = 5;
        int64_t           banded_matrix_row = K + 1;
        int64_t           lda               = 100;
        int64_t           incx              = 1;
        int64_t           incy              = 1;
        int64_t           batch_count       = 2;
        hipblasStride     stride_A          = N * lda;
        hipblasStride     stride_x          = N * incx;
        hipblasStride     stride_y          = N * incy;

        device_vector<T> d_alpha(1), d_beta(1), d_one(1), d_zero(1);

        Ts h_alpha{1}, h_beta{2}, h_one{1}, h_zero{0};
        if constexpr(is_complex<T>)
            h_one = {1, 0};

        const Ts* alpha = &h_alpha;
        const Ts* beta  = &h_beta;
        const Ts* one   = &h_one;
        const Ts* zero  = &h_zero;

        if(pointer_mode == HIPBLAS_POINTER_MODE_DEVICE)
        {
            CHECK_HIP_ERROR(hipMemcpy(d_alpha, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(d_beta, beta, sizeof(*beta), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(d_one, one, sizeof(*one), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(d_zero, zero, sizeof(*zero), hipMemcpyHostToDevice));
            alpha = d_alpha;
            beta  = d_beta;
            one   = d_one;
            zero  = d_zero;
        }

        device_strided_batch_matrix<T> dA(banded_matrix_row, N, lda, stride_A, batch_count);
        device_strided_batch_vector<T> dx(N, incx, stride_x, batch_count);
        device_strided_batch_vector<T> dy(N, incy, stride_y, batch_count);

        DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                    hipblasHbmvStridedBatchedFn,
                    (nullptr,
                     uplo,
                     N,
                     K,
                     alpha,
                     dA,
                     lda,
                     stride_A,
                     dx,
                     incx,
                     stride_x,
                     beta,
                     dy,
                     incy,
                     stride_y,
                     batch_count));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasHbmvStridedBatchedFn,
                    (handle,
                     HIPBLAS_FILL_MODE_FULL,
                     N,
                     K,
                     alpha,
                     dA,
                     lda,
                     stride_A,
                     dx,
                     incx,
                     stride_x,
                     beta,
                     dy,
                     incy,
                     stride_y,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasHbmvStridedBatchedFn,
                    (handle,
                     (hipblasFillMode_t)HIPBLAS_OP_N,
                     N,
                     K,
                     alpha,
                     dA,
                     lda,
                     stride_A,
                     dx,
                     incx,
                     stride_x,
                     beta,
                     dy,
                     incy,
                     stride_y,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasHbmvStridedBatchedFn,
                    (handle,
                     uplo,
                     N,
                     K,
                     nullptr,
                     dA,
                     lda,
                     stride_A,
                     dx,
                     incx,
                     stride_x,
                     beta,
                     dy,
                     incy,
                     stride_y,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasHbmvStridedBatchedFn,
                    (handle,
                     uplo,
                     N,
                     K,
                     alpha,
                     dA,
                     lda,
                     stride_A,
                     dx,
                     incx,
                     stride_x,
                     nullptr,
                     dy,
                     incy,
                     stride_y,
                     batch_count));

        if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
        {
            // For device mode in rocBLAS we don't have checks for dA, dx, dy as we may be able to quick return
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasHbmvStridedBatchedFn,
                        (handle,
                         uplo,
                         N,
                         K,
                         alpha,
                         nullptr,
                         lda,
                         stride_A,
                         dx,
                         incx,
                         stride_x,
                         beta,
                         dy,
                         incy,
                         stride_y,
                         batch_count));

            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasHbmvStridedBatchedFn,
                        (handle,
                         uplo,
                         N,
                         K,
                         alpha,
                         dA,
                         lda,
                         stride_A,
                         nullptr,
                         incx,
                         stride_x,
                         beta,
                         dy,
                         incy,
                         stride_y,
                         batch_count));

            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasHbmvStridedBatchedFn,
                        (handle,
                         uplo,
                         N,
                         K,
                         alpha,
                         dA,
                         lda,
                         stride_A,
                         dx,
                         incx,
                         stride_x,
                         beta,
                         nullptr,
                         incy,
                         stride_y,
                         batch_count));
        }

        // With N == 0, can have all nullptrs
        DAPI_CHECK(hipblasHbmvStridedBatchedFn,
                   (handle,
                    uplo,
                    0,
                    K,
                    nullptr,
                    nullptr,
                    lda,
                    stride_A,
                    nullptr,
                    incx,
                    stride_x,
                    nullptr,
                    nullptr,
                    incy,
                    stride_y,
                    batch_count));
        DAPI_CHECK(hipblasHbmvStridedBatchedFn,
                   (handle,
                    uplo,
                    N,
                    K,
                    nullptr,
                    nullptr,
                    lda,
                    stride_A,
                    nullptr,
                    incx,
                    stride_x,
                    nullptr,
                    nullptr,
                    incy,
                    stride_y,
                    0));

        // With alpha == 0 can have A and x nullptr
        DAPI_CHECK(hipblasHbmvStridedBatchedFn,
                   (handle,
                    uplo,
                    N,
                    K,
                    zero,
                    nullptr,
                    lda,
                    stride_A,
                    nullptr,
                    incx,
                    stride_x,
                    beta,
                    dy,
                    incy,
                    stride_y,
                    batch_count));

        // With alpha == 0 && beta == 1, all other ptrs can be nullptr
        DAPI_CHECK(hipblasHbmvStridedBatchedFn,
                   (handle,
                    uplo,
                    N,
                    K,
                    zero,
                    nullptr,
                    lda,
                    stride_A,
                    nullptr,
                    incx,
                    stride_x,
                    one,
                    nullptr,
                    incy,
                    stride_y,
                    batch_count));
    }
}

template <typename T>
void testing_hbmv_strided_batched(const Arguments& arg)
{
    using Ts     = hipblas_internal_type<T>;
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;

    auto hipblasHbmvStridedBatchedFn
        = FORTRAN ? hipblasHbmvStridedBatched<T, true> : hipblasHbmvStridedBatched<T, false>;

    auto hipblasHbmvStridedBatchedFn_64 = arg.api == FORTRAN_64
                                              ? hipblasHbmvStridedBatched_64<T, true>
                                              : hipblasHbmvStridedBatched_64<T, false>;

    hipblasFillMode_t uplo              = char2hipblas_fill(arg.uplo);
    int64_t           N                 = arg.N;
    int64_t           K                 = arg.K;
    int64_t           lda               = arg.lda;
    int64_t           incx              = arg.incx;
    int64_t           incy              = arg.incy;
    double            stride_scale      = arg.stride_scale;
    int64_t           batch_count       = arg.batch_count;
    int64_t           banded_matrix_row = K + 1;

    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t abs_incy = incy >= 0 ? incy : -incy;

    hipblasStride stride_A = lda * N * stride_scale;
    hipblasStride stride_x = N * abs_incx * stride_scale;
    hipblasStride stride_y = N * abs_incy * stride_scale;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size
        = N < 0 || K < 0 || lda < banded_matrix_row || !incx || !incy || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        DAPI_EXPECT(invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS,
                    hipblasHbmvStridedBatchedFn,
                    (handle,
                     uplo,
                     N,
                     K,
                     nullptr,
                     nullptr,
                     lda,
                     stride_A,
                     nullptr,
                     incx,
                     stride_x,
                     nullptr,
                     nullptr,
                     incy,
                     stride_y,
                     batch_count));
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_strided_batch_matrix<T> hA(banded_matrix_row, N, lda, stride_A, batch_count);
    host_strided_batch_vector<T> hx(N, incx, stride_x, batch_count);
    host_strided_batch_vector<T> hy(N, incy, stride_y, batch_count);
    host_strided_batch_vector<T> hy_cpu(N, incy, stride_y, batch_count);
    host_strided_batch_vector<T> hy_host(N, incy, stride_y, batch_count);
    host_strided_batch_vector<T> hy_device(N, incy, stride_y, batch_count);

    device_strided_batch_matrix<T> dA(banded_matrix_row, N, lda, stride_A, batch_count);
    device_strided_batch_vector<T> dx(N, incx, stride_x, batch_count);
    device_strided_batch_vector<T> dy(N, incy, stride_y, batch_count);
    device_vector<T>               d_alpha(1);
    device_vector<T>               d_beta(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta.memcheck());

    double hipblas_error_host{0}, hipblas_error_device{0};

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    // Initial Data on CPU
    //Matrix `hA` is initialized as a triangular matrix because only the upper triangular or lower triangular portion of the matrix `hAb` is referenced.
    hipblas_init_matrix(hA, arg, hipblas_client_alpha_sets_nan, hipblas_triangular_matrix, true);
    hipblas_init_vector(hx, arg, hipblas_client_alpha_sets_nan, false, true);
    hipblas_init_vector(hy, arg, hipblas_client_beta_sets_nan);

    // copy vector is easy in STL; hy_cpu = hy: save a copy in hy_cpu which will be output of CPU BLAS
    hy_cpu.copy_from(hy);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        DAPI_CHECK(hipblasHbmvStridedBatchedFn,
                   (handle,
                    uplo,
                    N,
                    K,
                    reinterpret_cast<Ts*>(&h_alpha),
                    dA,
                    lda,
                    stride_A,
                    dx,
                    incx,
                    stride_x,
                    reinterpret_cast<Ts*>(&h_beta),
                    dy,
                    incy,
                    stride_y,
                    batch_count));

        CHECK_HIP_ERROR(hy_host.transfer_from(dy));
        CHECK_HIP_ERROR(dy.transfer_from(hy));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasHbmvStridedBatchedFn,
                   (handle,
                    uplo,
                    N,
                    K,
                    d_alpha,
                    dA,
                    lda,
                    stride_A,
                    dx,
                    incx,
                    stride_x,
                    d_beta,
                    dy,
                    incy,
                    stride_y,
                    batch_count));

        CHECK_HIP_ERROR(hy_device.transfer_from(dy));

        /* =====================================================================
           CPU BLAS
        =================================================================== */

        for(size_t b = 0; b < batch_count; b++)
        {
            ref_hbmv<T>(uplo, N, K, h_alpha, hA[b], lda, hx[b], incx, h_beta, hy_cpu[b], incy);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, batch_count, abs_incy, stride_y, hy_cpu, hy_host);
            unit_check_general<T>(1, N, batch_count, abs_incy, stride_y, hy_cpu, hy_device);
        }
        if(arg.norm_check)
        {
            hipblas_error_host = norm_check_general<T>(
                'F', 1, N, abs_incy, stride_y, hy_cpu, hy_host, batch_count);
            hipblas_error_device = norm_check_general<T>(
                'F', 1, N, abs_incy, stride_y, hy_cpu, hy_device, batch_count);
        }
    }

    if(arg.timing)
    {
        double gpu_time_used{0};
        CHECK_HIP_ERROR(dy.transfer_from(hy));
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(hipblasHbmvStridedBatchedFn,
                          (handle,
                           uplo,
                           N,
                           K,
                           d_alpha,
                           dA,
                           lda,
                           stride_A,
                           dx,
                           incx,
                           stride_x,
                           d_beta,
                           dy,
                           incy,
                           stride_y,
                           batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasHbmvStridedBatchedModel{}.log_args<T>(std::cout,
                                                     arg,
                                                     gpu_time_used,
                                                     hbmv_gflop_count<T>(N, K),
                                                     hbmv_gbyte_count<T>(N, K),
                                                     hipblas_error_host,
                                                     hipblas_error_device);
    }
}
