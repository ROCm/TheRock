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

using hipblasTrsvModel = ArgumentModel<e_a_type, e_uplo, e_transA, e_diag, e_N, e_lda, e_incx>;

inline void testname_trsv(const Arguments& arg, std::string& name)
{
    hipblasTrsvModel{}.test_name(arg, name);
}

template <typename T>
void testing_trsv_bad_arg(const Arguments& arg)
{
    bool FORTRAN       = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasTrsvFn = FORTRAN ? hipblasTrsv<T, true> : hipblasTrsv<T, false>;
    auto hipblasTrsvFn_64
        = arg.api == FORTRAN_64 ? hipblasTrsv_64<T, true> : hipblasTrsv_64<T, false>;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        hipblasLocalHandle handle(arg);
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        hipblasOperation_t transA = HIPBLAS_OP_N;
        hipblasFillMode_t  uplo   = HIPBLAS_FILL_MODE_UPPER;
        hipblasDiagType_t  diag   = HIPBLAS_DIAG_NON_UNIT;
        int64_t            N      = 100;
        int64_t            lda    = 100;
        int64_t            incx   = 1;

        // Allocate device memory
        device_matrix<T> dA(N, N, lda);
        device_vector<T> dx(N, incx);

        DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                    hipblasTrsvFn,
                    (nullptr, uplo, transA, diag, N, dA, lda, dx, incx));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasTrsvFn,
                    (handle, HIPBLAS_FILL_MODE_FULL, transA, diag, N, dA, lda, dx, incx));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasTrsvFn,
                    (handle, (hipblasFillMode_t)HIPBLAS_OP_N, transA, diag, N, dA, lda, dx, incx));

        DAPI_EXPECT(
            HIPBLAS_STATUS_INVALID_ENUM,
            hipblasTrsvFn,
            (handle, uplo, (hipblasOperation_t)HIPBLAS_FILL_MODE_FULL, diag, N, dA, lda, dx, incx));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasTrsvFn,
                    (handle,
                     uplo,
                     transA,
                     (hipblasDiagType_t)HIPBLAS_FILL_MODE_FULL,
                     N,
                     dA,
                     lda,
                     dx,
                     incx));

        if(arg.bad_arg_all)
        {
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasTrsvFn,
                        (handle, uplo, transA, diag, N, nullptr, lda, dx, incx));

            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasTrsvFn,
                        (handle, uplo, transA, diag, N, dA, lda, nullptr, incx));
        }

        // With N == 0, can have all nullptrs
        DAPI_CHECK(hipblasTrsvFn, (handle, uplo, transA, diag, 0, nullptr, lda, nullptr, incx));
    }
}

template <typename T>
void testing_trsv(const Arguments& arg)
{
    bool FORTRAN       = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasTrsvFn = FORTRAN ? hipblasTrsv<T, true> : hipblasTrsv<T, false>;
    auto hipblasTrsvFn_64
        = arg.api == FORTRAN_64 ? hipblasTrsv_64<T, true> : hipblasTrsv_64<T, false>;

    hipblasFillMode_t  uplo   = char2hipblas_fill(arg.uplo);
    hipblasDiagType_t  diag   = char2hipblas_diagonal(arg.diag);
    hipblasOperation_t transA = char2hipblas_operation(arg.transA);
    int64_t            N      = arg.N;
    int64_t            incx   = arg.incx;
    int64_t            lda    = arg.lda;

    int64_t abs_incx = incx < 0 ? -incx : incx;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = N < 0 || lda < N || lda < 1 || !incx;
    if(invalid_size || !N)
    {
        DAPI_EXPECT(invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS,
                    hipblasTrsvFn,
                    (handle, uplo, transA, diag, N, nullptr, lda, nullptr, incx));
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_matrix<T> hA(N, N, lda);
    host_vector<T> hb(N, incx);
    host_vector<T> hx(N, incx);
    host_vector<T> hx_or_b(N, incx);

    // Allocate device memory
    device_matrix<T> dA(N, N, lda);
    device_vector<T> dx_or_b(N, incx);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dx_or_b.memcheck());

    double hipblas_error{0};

    // Initial Data on CPU
    hipblas_init_matrix(hA,
                        arg,
                        hipblas_client_never_set_nan,
                        hipblas_diagonally_dominant_triangular_matrix,
                        true,
                        false);
    hipblas_init_vector(hx, arg, hipblas_client_never_set_nan, false, true);

    hb = hx;

    if(diag == HIPBLAS_DIAG_UNIT)
    {
        make_unit_diagonal(uplo, (T*)hA, lda, N);
    }

    // Calculate hb = hA*hx;
    ref_trmv<T>(uplo, transA, diag, N, hA.data(), lda, hb.data(), incx);
    hx_or_b = hb;

    // copy data from CPU to device

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dx_or_b.transfer_from(hx_or_b));

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    if(arg.unit_check || arg.norm_check)
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        DAPI_CHECK(hipblasTrsvFn, (handle, uplo, transA, diag, N, dA, lda, dx_or_b, incx));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hx_or_b.transfer_from(dx_or_b));

        // Calculating error
        hipblas_error = hipblas_abs(vector_norm_1<T>(N, abs_incx, hx.data(), hx_or_b.data()));

        if(arg.unit_check)
        {
            double tolerance = std::numeric_limits<real_t<T>>::epsilon() * 40 * N;
            unit_check_error(hipblas_error, tolerance);
        }
    }

    if(arg.timing)
    {
        double      gpu_time_used{0};
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(hipblasTrsvFn, (handle, uplo, transA, diag, N, dA, lda, dx_or_b, incx));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        hipblasTrsvModel{}.log_args<T>(std::cout,
                                       arg,
                                       gpu_time_used,
                                       trsv_gflop_count<T>(N),
                                       trsv_gbyte_count<T>(N),
                                       hipblas_error);
    }
}
