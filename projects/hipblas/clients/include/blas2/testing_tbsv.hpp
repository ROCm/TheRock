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

using hipblasTbsvModel = ArgumentModel<e_a_type, e_uplo, e_transA, e_diag, e_N, e_K, e_lda, e_incx>;

inline void testname_tbsv(const Arguments& arg, std::string& name)
{
    hipblasTbsvModel{}.test_name(arg, name);
}

template <typename T>
void testing_tbsv_bad_arg(const Arguments& arg)
{
    bool FORTRAN       = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasTbsvFn = FORTRAN ? hipblasTbsv<T, true> : hipblasTbsv<T, false>;
    auto hipblasTbsvFn_64
        = arg.api == FORTRAN_64 ? hipblasTbsv_64<T, true> : hipblasTbsv_64<T, false>;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        hipblasLocalHandle handle(arg);
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        hipblasFillMode_t  uplo              = HIPBLAS_FILL_MODE_UPPER;
        hipblasOperation_t transA            = HIPBLAS_OP_N;
        hipblasDiagType_t  diag              = HIPBLAS_DIAG_NON_UNIT;
        int64_t            N                 = 100;
        int64_t            K                 = 5;
        int64_t            banded_matrix_row = K + 1;
        int64_t            lda               = 100;
        int64_t            incx              = 1;

        // Allocate device memory
        device_matrix<T> dAb(banded_matrix_row, N, lda);
        device_vector<T> dx(N, incx);

        DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                    hipblasTbsvFn,
                    (nullptr, uplo, transA, diag, N, K, dAb, lda, dx, incx));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasTbsvFn,
                    (handle, HIPBLAS_FILL_MODE_FULL, transA, diag, N, K, dAb, lda, dx, incx));

        DAPI_EXPECT(
            HIPBLAS_STATUS_INVALID_ENUM,
            hipblasTbsvFn,
            (handle, (hipblasFillMode_t)HIPBLAS_OP_N, transA, diag, N, K, dAb, lda, dx, incx));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasTbsvFn,
                    (handle,
                     uplo,
                     (hipblasOperation_t)HIPBLAS_FILL_MODE_FULL,
                     diag,
                     N,
                     K,
                     dAb,
                     lda,
                     dx,
                     incx));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasTbsvFn,
                    (handle,
                     uplo,
                     transA,
                     (hipblasDiagType_t)HIPBLAS_FILL_MODE_FULL,
                     N,
                     K,
                     dAb,
                     lda,
                     dx,
                     incx));

        if(arg.bad_arg_all)
        {
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasTbsvFn,
                        (handle, uplo, transA, diag, N, K, nullptr, lda, dx, incx));

            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasTbsvFn,
                        (handle, uplo, transA, diag, N, K, dAb, lda, nullptr, incx));
        }

        // With N == 0, can have all nullptrs
        DAPI_CHECK(hipblasTbsvFn, (handle, uplo, transA, diag, 0, K, nullptr, lda, nullptr, incx));
    }
}

template <typename T>
void testing_tbsv(const Arguments& arg)
{
    bool FORTRAN       = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasTbsvFn = FORTRAN ? hipblasTbsv<T, true> : hipblasTbsv<T, false>;
    auto hipblasTbsvFn_64
        = arg.api == FORTRAN_64 ? hipblasTbsv_64<T, true> : hipblasTbsv_64<T, false>;

    hipblasFillMode_t  uplo              = char2hipblas_fill(arg.uplo);
    hipblasDiagType_t  diag              = char2hipblas_diagonal(arg.diag);
    hipblasOperation_t transA            = char2hipblas_operation(arg.transA);
    int64_t            N                 = arg.N;
    int64_t            K                 = arg.K;
    int64_t            incx              = arg.incx;
    int64_t            lda               = arg.lda;
    const int64_t      banded_matrix_row = K + 1;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = N < 0 || K < 0 || lda < banded_matrix_row || !incx;
    if(invalid_size || !N)
    {
        DAPI_EXPECT(invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS,
                    hipblasTbsvFn,
                    (handle, uplo, transA, diag, N, K, nullptr, lda, nullptr, incx));
        return;
    }

    int abs_incx = incx < 0 ? -incx : incx;

    // Naming: `h` is in CPU (host) memory(eg hAb), `d` is in GPU (device) memory (eg dAb).
    // Allocate host memory
    host_matrix<T> hA(N, N, N);
    host_matrix<T> hAb(banded_matrix_row, N, lda);
    host_vector<T> hb(N, incx);
    host_vector<T> hx(N, incx);
    host_vector<T> hx_or_b(N, incx);

    // Allocate device memory
    device_matrix<T> dAb(banded_matrix_row, N, lda);
    device_vector<T> dx_or_b(N, incx);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dAb.memcheck());
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

    banded_matrix_setup(uplo == HIPBLAS_FILL_MODE_UPPER, (T*)hA, N, N, K);

    if(diag == HIPBLAS_DIAG_UNIT)
    {
        make_unit_diagonal(uplo, (T*)hA, N, N);
    }

    regular_to_banded(uplo == HIPBLAS_FILL_MODE_UPPER, (T*)hA, N, (T*)hAb, lda, N, K);

    ref_tbmv<T>(uplo, transA, diag, N, K, hAb, lda, hb, incx);

    hx_or_b = hb;

    // copy data from CPU to device
    CHECK_HIP_ERROR(dAb.transfer_from(hAb));
    CHECK_HIP_ERROR(dx_or_b.transfer_from(hx_or_b));

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    if(arg.unit_check || arg.norm_check)
    {
        DAPI_CHECK(hipblasTbsvFn, (handle, uplo, transA, diag, N, K, dAb, lda, dx_or_b, incx));

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

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(hipblasTbsvFn,
                          (handle, uplo, transA, diag, N, K, dAb, lda, dx_or_b, incx));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        hipblasTbsvModel{}.log_args<T>(std::cout,
                                       arg,
                                       gpu_time_used,
                                       tbsv_gflop_count<T>(N, K),
                                       tbsv_gbyte_count<T>(N, K),
                                       hipblas_error);
    }
}
