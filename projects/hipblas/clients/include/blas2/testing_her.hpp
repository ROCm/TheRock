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

using hipblasHerModel = ArgumentModel<e_a_type, e_uplo, e_N, e_alpha, e_incx, e_lda>;

inline void testname_her(const Arguments& arg, std::string& name)
{
    hipblasHerModel{}.test_name(arg, name);
}

template <typename T>
void testing_her_bad_arg(const Arguments& arg)
{
    using U           = real_t<T>;
    bool FORTRAN      = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasHerFn = FORTRAN ? hipblasHer<T, U, true> : hipblasHer<T, U, false>;
    auto hipblasHerFn_64
        = arg.api == FORTRAN_64 ? hipblasHer_64<T, U, true> : hipblasHer_64<T, U, false>;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        hipblasLocalHandle handle(arg);
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        hipblasFillMode_t uplo = HIPBLAS_FILL_MODE_UPPER;
        int64_t           N    = 100;
        int64_t           lda  = 100;
        int64_t           incx = 1;

        device_vector<U> d_alpha(1), d_zero(1);

        const U  h_alpha{1}, h_zero{0};
        const U* alpha = &h_alpha;
        const U* zero  = &h_zero;

        if(pointer_mode == HIPBLAS_POINTER_MODE_DEVICE)
        {
            CHECK_HIP_ERROR(hipMemcpy(d_alpha, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(d_zero, zero, sizeof(*zero), hipMemcpyHostToDevice));
            alpha = d_alpha;
            zero  = d_zero;
        }

        device_matrix<T> dA(N, N, lda);
        device_vector<T> dx(N, incx);

        DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                    hipblasHerFn,
                    (nullptr, uplo, N, alpha, dx, incx, dA, lda));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasHerFn,
                    (handle, HIPBLAS_FILL_MODE_FULL, N, alpha, dx, incx, dA, lda));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasHerFn,
                    (handle, (hipblasFillMode_t)HIPBLAS_OP_N, N, alpha, dx, incx, dA, lda));

        if(arg.bad_arg_all)
        {
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasHerFn,
                        (handle, uplo, N, nullptr, dx, incx, dA, lda));

            if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
            {
                // For device mode in rocBLAS we don't have checks for dA, dx as we may be able to quick return
                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasHerFn,
                            (handle, uplo, N, alpha, nullptr, incx, dA, lda));

                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasHerFn,
                            (handle, uplo, N, alpha, dx, incx, nullptr, lda));
            }

            // With alpha == 0, can have all nullptrs
            DAPI_CHECK(hipblasHerFn, (handle, uplo, N, zero, nullptr, incx, nullptr, lda));
        }

        // With N == 0, can have all nullptrs
        DAPI_CHECK(hipblasHerFn, (handle, uplo, 0, nullptr, nullptr, incx, nullptr, lda));
    }
}

template <typename T>
void testing_her(const Arguments& arg)
{
    using U           = real_t<T>;
    bool FORTRAN      = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasHerFn = FORTRAN ? hipblasHer<T, U, true> : hipblasHer<T, U, false>;
    auto hipblasHerFn_64
        = arg.api == FORTRAN_64 ? hipblasHer_64<T, U, true> : hipblasHer_64<T, U, false>;

    hipblasFillMode_t uplo = char2hipblas_fill(arg.uplo);
    int64_t           N    = arg.N;
    int64_t           incx = arg.incx;
    int64_t           lda  = arg.lda;

    size_t abs_incx = incx >= 0 ? incx : -incx;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = N < 0 || lda < N || lda < 1 || !incx;
    if(invalid_size || !N)
    {
        DAPI_EXPECT(invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS,
                    hipblasHerFn,
                    (handle, uplo, N, nullptr, nullptr, incx, nullptr, lda));
        return;
    }

    // Naming: dA is in GPU (device) memory. hA is in CPU (host) memory
    host_matrix<T> hA(N, N, lda);
    host_matrix<T> hA_cpu(N, N, lda);
    host_matrix<T> hA_host(N, N, lda);
    host_matrix<T> hA_device(N, N, lda);
    host_vector<T> hx(N, incx);

    device_matrix<T> dA(N, N, lda);
    device_vector<T> dx(N, incx);
    device_vector<U> d_alpha(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());

    double hipblas_error_host{0}, hipblas_error_device{0};

    U h_alpha = arg.get_alpha<U>();

    // Initial Data on CPU
    hipblas_init_matrix(hA, arg, hipblas_client_never_set_nan, hipblas_hermitian_matrix, true);
    hipblas_init_vector(hx, arg, hipblas_client_alpha_sets_nan, false, true);

    // copy matrix is easy in STL; hA_cpu = hA: save a copy in hA_cpu which will be output of CPU BLAS
    hA_cpu = hA;

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(U), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        DAPI_CHECK(hipblasHerFn, (handle, uplo, N, (U*)&h_alpha, dx, incx, dA, lda));

        CHECK_HIP_ERROR(hA_host.transfer_from(dA));
        CHECK_HIP_ERROR(dA.transfer_from(hA));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasHerFn, (handle, uplo, N, d_alpha, dx, incx, dA, lda));

        CHECK_HIP_ERROR(hA_device.transfer_from(dA));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        ref_her<T>(uplo, N, h_alpha, hx.data(), incx, hA_cpu.data(), lda);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(N, N, lda, hA_cpu.data(), hA_host.data());

            // NOTE: on cuBLAS, with alpha == 0 and alpha on the device, there is not a quick-return,
            // instead, the imaginary part of the diagonal elements are set to 0. in rocBLAS, we are quick-returning
            // as well as in our reference code. For this reason, I've disabled the check here.
            if(h_alpha)
                unit_check_general<T>(N, N, lda, hA_cpu.data(), hA_device.data());
        }
        if(arg.norm_check)
        {
            hipblas_error_host   = norm_check_general<T>('F', N, N, lda, hA_cpu, hA_host);
            hipblas_error_device = norm_check_general<T>('F', N, N, lda, hA_cpu, hA_device);
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
            DAPI_DISPATCH(hipblasHerFn, (handle, uplo, N, d_alpha, dx, incx, dA, lda));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasHerModel{}.log_args<U>(std::cout,
                                      arg,
                                      gpu_time_used,
                                      her_gflop_count<T>(N),
                                      her_gbyte_count<T>(N),
                                      hipblas_error_host,
                                      hipblas_error_device);
    }
}
