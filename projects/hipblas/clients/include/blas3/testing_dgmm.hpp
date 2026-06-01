/* ************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
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

using hipblasDgmmModel = ArgumentModel<e_a_type, e_side, e_M, e_N, e_lda, e_incx, e_ldc>;

inline void testname_dgmm(const Arguments& arg, std::string& name)
{
    hipblasDgmmModel{}.test_name(arg, name);
}

template <typename T>
void testing_dgmm_bad_arg(const Arguments& arg)
{
    auto hipblasDgmmFn = arg.api == FORTRAN ? hipblasDgmm<T, true> : hipblasDgmm<T, false>;
    auto hipblasDgmmFn_64
        = arg.api == FORTRAN_64 ? hipblasDgmm_64<T, true> : hipblasDgmm_64<T, false>;

    hipblasLocalHandle handle(arg);

    int64_t M    = 101;
    int64_t N    = 100;
    int64_t lda  = 102;
    int64_t incx = 1;
    int64_t ldc  = 103;

    hipblasSideMode_t side = HIPBLAS_SIDE_LEFT;

    int64_t K = side == HIPBLAS_SIDE_LEFT ? M : N;

    // Allocate device memory
    device_matrix<T> dA(M, N, lda);
    device_vector<T> dx(K, incx);
    device_matrix<T> dC(M, N, ldc);

    DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                hipblasDgmmFn,
                (nullptr, side, M, N, dA, lda, dx, incx, dC, ldc));

    DAPI_EXPECT(
        HIPBLAS_STATUS_INVALID_ENUM,
        hipblasDgmmFn,
        (handle, (hipblasSideMode_t)HIPBLAS_FILL_MODE_FULL, M, N, dA, lda, dx, incx, dC, ldc));

    if(arg.bad_arg_all)
    {
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasDgmmFn,
                    (handle, side, M, N, nullptr, lda, dx, incx, dC, ldc));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasDgmmFn,
                    (handle, side, M, N, dA, lda, nullptr, incx, dC, ldc));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasDgmmFn,
                    (handle, side, M, N, dA, lda, dx, incx, nullptr, ldc));

        // dgmm will quick-return with M == 0 || N == 0. Here, c_i32_overflow will rollover in the case of 32-bit params,
        // and quick-return with 64-bit params. This depends on implementation so only testing rocBLAS backend
        DAPI_EXPECT((arg.api & c_API_64) ? HIPBLAS_STATUS_SUCCESS : HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasDgmmFn,
                    (handle,
                     side,
                     0,
                     c_i32_overflow,
                     nullptr,
                     c_i32_overflow,
                     nullptr,
                     incx,
                     nullptr,
                     c_i32_overflow));
        DAPI_EXPECT((arg.api & c_API_64) ? HIPBLAS_STATUS_SUCCESS : HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasDgmmFn,
                    (handle,
                     side,
                     c_i32_overflow,
                     0,
                     nullptr,
                     c_i32_overflow,
                     nullptr,
                     incx,
                     nullptr,
                     c_i32_overflow));
    }

    // If M == 0 || N == 0, can have nullptrs
    DAPI_CHECK(hipblasDgmmFn, (handle, side, 0, N, nullptr, lda, nullptr, incx, nullptr, ldc));
    DAPI_CHECK(hipblasDgmmFn, (handle, side, M, 0, nullptr, lda, nullptr, incx, nullptr, ldc));
}

template <typename T>
void testing_dgmm(const Arguments& arg)
{
    auto hipblasDgmmFn = arg.api == FORTRAN ? hipblasDgmm<T, true> : hipblasDgmm<T, false>;
    auto hipblasDgmmFn_64
        = arg.api == FORTRAN_64 ? hipblasDgmm_64<T, true> : hipblasDgmm_64<T, false>;

    hipblasSideMode_t side = char2hipblas_side(arg.side);

    int64_t M    = arg.M;
    int64_t N    = arg.N;
    int64_t lda  = arg.lda;
    int64_t incx = arg.incx;
    int64_t ldc  = arg.ldc;

    int64_t abs_incx = incx >= 0 ? incx : -incx;
    int64_t K        = (side == HIPBLAS_SIDE_RIGHT ? N : M);

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = M < 0 || N < 0 || ldc < M || lda < M;
    if(invalid_size || !N || !M)
    {
        DAPI_EXPECT((invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS),
                    hipblasDgmmFn,
                    (handle, side, M, N, nullptr, lda, nullptr, incx, nullptr, ldc));
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_matrix<T> hA(M, N, lda);
    host_vector<T> hx(K, incx);
    host_matrix<T> hC(M, N, ldc);
    host_matrix<T> hC_gold(M, N, ldc);

    // Allocate device memory
    device_matrix<T> dA(M, N, lda);
    device_vector<T> dx(K, incx);
    device_matrix<T> dC(M, N, ldc);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());

    double gpu_time_used{0}, hipblas_error{0};

    // Initial Data on CPU
    hipblas_init_matrix(hA, arg, hipblas_client_never_set_nan, hipblas_general_matrix, true);
    hipblas_init_vector(hx, arg, hipblas_client_never_set_nan, false, true);
    hipblas_init_matrix(hC, arg, hipblas_client_never_set_nan, hipblas_general_matrix);

    hC_gold = hC;

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dC.transfer_from(hC));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        DAPI_CHECK(hipblasDgmmFn, (handle, side, M, N, dA, lda, dx, incx, dC, ldc));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hC.transfer_from(dC));

        /* =====================================================================
           CPU BLAS
        =================================================================== */

        ref_dgmm<T>(side, M, N, hA, lda, hx, incx, hC_gold, ldc);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(M, N, ldc, hC_gold, hC);
        }

        if(arg.norm_check)
        {
            hipblas_error = norm_check_general<T>('F', M, N, ldc, hC_gold, hC);
        }
    }

    if(arg.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(hipblasDgmmFn, (handle, side, M, N, dA, lda, dx, incx, dC, ldc));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        hipblasDgmmModel{}.log_args<T>(std::cout,
                                       arg,
                                       gpu_time_used,
                                       dgmm_gflop_count<T>(M, N),
                                       dgmm_gbyte_count<T>(M, N, K),
                                       hipblas_error);
    }
}
