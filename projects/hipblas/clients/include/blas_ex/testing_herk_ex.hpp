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

#include "utility.h"
#include <fstream>
#include <iostream>
#include <limits>
#include <stdlib.h>
#include <typeinfo>
#include <vector>

#include "testing_common.hpp"

/* ============================================================================================ */

using hipblasHerkExModel = ArgumentModel<e_a_type,
                                         e_c_type,
                                         e_compute_type,
                                         e_uplo,
                                         e_transA,
                                         e_N,
                                         e_K,
                                         e_alpha,
                                         e_lda,
                                         e_beta,
                                         e_ldc>;

inline void testname_herk_ex(const Arguments& arg, std::string& name)
{
    hipblasHerkExModel{}.test_name(arg, name);
}

template <typename Ti, typename To = Ti, typename Tex = To>
void testing_herk_ex_bad_arg(const Arguments& arg)
{
    using Ts     = hipblas_internal_type<Tex>;
    using Tab_ex = real_t<Tex>;

    auto hipblasHerkExFn    = arg.api == FORTRAN ? hipblasHerkExFortran : hipblasHerkEx;
    auto hipblasHerkExFn_64 = arg.api == FORTRAN ? hipblasHerkExFortran : hipblasHerkEx;
    //auto hipblasHerkExFn_64 = arg.api == FORTRAN_64 ? hipblasHerkEx_64Fortran : hipblasHerkEx_64;

    hipblasLocalHandle handle(arg);

    hipDataType aType       = arg.a_type;
    hipDataType cType       = arg.c_type;
    hipDataType computeType = arg.compute_type;

    int64_t N   = 100;
    int64_t K   = 101;
    int64_t lda = 102;
    int64_t ldc = 103;

    hipblasFillMode_t  uplo   = HIPBLAS_FILL_MODE_UPPER;
    hipblasOperation_t transA = HIPBLAS_OP_N;

    int64_t A_row = transA == HIPBLAS_OP_N ? N : std::max(K, int64_t(1));
    int64_t A_col = transA == HIPBLAS_OP_N ? std::max(K, int64_t(1)) : N;

    // Allocate device memory
    device_matrix<Ti> dA(A_row, A_col, lda);
    device_matrix<To> dC(N, N, ldc);

    device_vector<Tab_ex> d_alpha(1), d_beta(1), d_one(1), d_zero(1);
    Tab_ex                h_alpha{1.0f}, h_beta{2.0f}, h_one{1.0f}, h_zero{0.0f};

    if constexpr(std::is_same_v<Tab_ex, hipblasHalf>)
        h_one = float_to_half(1.0f);

    const Tab_ex* alpha = &h_alpha;
    const Tab_ex* beta  = &h_beta;
    const Tab_ex* one   = &h_one;
    const Tab_ex* zero  = &h_zero;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

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

        DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                    hipblasHerkExFn,
                    (nullptr,
                     uplo,
                     transA,
                     N,
                     K,
                     alpha,
                     dA,
                     aType,
                     lda,
                     beta,
                     dC,
                     cType,
                     ldc,
                     computeType));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE, // enum is valid but not for this function
                    hipblasHerkExFn,
                    (handle,
                     HIPBLAS_FILL_MODE_FULL,
                     transA,
                     N,
                     K,
                     alpha,
                     dA,
                     aType,
                     lda,
                     beta,
                     dC,
                     cType,
                     ldc,
                     computeType));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE, // enum is valid but not for this function
                    hipblasHerkExFn,
                    (handle,
                     uplo,
                     HIPBLAS_OP_T,
                     N,
                     K,
                     alpha,
                     dA,
                     aType,
                     lda,
                     beta,
                     dC,
                     cType,
                     ldc,
                     computeType));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasHerkExFn,
                    (handle,
                     uplo,
                     (hipblasOperation_t)HIPBLAS_FILL_MODE_FULL,
                     N,
                     K,
                     alpha,
                     dA,
                     aType,
                     lda,
                     beta,
                     dC,
                     cType,
                     ldc,
                     computeType));

        if(arg.bad_arg_all)
        {
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasHerkExFn,
                        (handle,
                         uplo,
                         transA,
                         N,
                         K,
                         nullptr,
                         dA,
                         aType,
                         lda,
                         beta,
                         dC,
                         cType,
                         ldc,
                         computeType));
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasHerkExFn,
                        (handle,
                         uplo,
                         transA,
                         N,
                         K,
                         alpha,
                         nullptr,
                         aType,
                         lda,
                         beta,
                         dC,
                         cType,
                         ldc,
                         computeType));
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasHerkExFn,
                        (handle,
                         uplo,
                         transA,
                         N,
                         K,
                         alpha,
                         dA,
                         aType,
                         lda,
                         nullptr,
                         dC,
                         cType,
                         ldc,
                         computeType));
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasHerkExFn,
                        (handle,
                         uplo,
                         transA,
                         N,
                         K,
                         alpha,
                         dA,
                         aType,
                         lda,
                         beta,
                         nullptr,
                         cType,
                         ldc,
                         computeType));
        }

        // With N == 0, can have all nullptrs
        DAPI_CHECK(hipblasHerkExFn,
                   (handle,
                    uplo,
                    transA,
                    0,
                    K,
                    nullptr,
                    nullptr,
                    aType,
                    lda,
                    nullptr,
                    nullptr,
                    cType,
                    ldc,
                    computeType));

        // With alpha == 0, can have A be nullptr
        DAPI_CHECK(hipblasHerkExFn,
                   (handle,
                    uplo,
                    transA,
                    N,
                    K,
                    zero,
                    nullptr,
                    aType,
                    lda,
                    beta,
                    dC,
                    cType,
                    ldc,
                    computeType));
    }
}

template <typename Ti, typename To = Ti, typename Tex = To>
void testing_herk_ex(const Arguments& arg)
{
    using Ts     = hipblas_internal_type<Tex>;
    using Tab_ex = real_t<Tex>;

    auto hipblasHerkExFn    = arg.api == FORTRAN ? hipblasHerkExFortran : hipblasHerkEx;
    auto hipblasHerkExFn_64 = arg.api == FORTRAN ? hipblasHerkExFortran : hipblasHerkEx;
    //auto hipblasHerkExFn_64 = arg.api == FORTRAN_64 ? hipblasHerkEx_64Fortran : hipblasHerkEx_64;

    hipDataType aType       = arg.a_type;
    hipDataType cType       = arg.c_type;
    hipDataType computeType = arg.compute_type;

    hipblasFillMode_t  uplo   = char2hipblas_fill(arg.uplo);
    hipblasOperation_t transA = char2hipblas_operation(arg.transA);

    int N   = arg.N;
    int K   = arg.K;
    int lda = arg.lda;
    int ldc = arg.ldc;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    Tab_ex h_alpha = arg.get_alpha<Tab_ex>();
    Tab_ex h_beta  = arg.get_beta<Tab_ex>();

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = N < 0 || K < 0 || ldc < N || (transA == HIPBLAS_OP_N && lda < N)
                        || (transA != HIPBLAS_OP_N && lda < K);
    if(invalid_size || !N)
    {
        DAPI_EXPECT(invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS,
                    hipblasHerkExFn,
                    (handle,
                     uplo,
                     transA,
                     N,
                     K,
                     nullptr,
                     nullptr,
                     aType,
                     lda,
                     nullptr,
                     nullptr,
                     cType,
                     ldc,
                     computeType));
        return;
    }

    // Naming: `h` is in CPU (host) memory, `d` is in GPU (device) memory
    size_t A_row  = (transA == HIPBLAS_OP_N ? N : K);
    size_t A_col  = (transA == HIPBLAS_OP_N ? K : N);
    size_t A_size = size_t(lda) * A_col;
    size_t C_size = size_t(ldc) * N;

    // Allocate host memory
    host_matrix<Ti> hA(A_row, A_col, lda);
    host_matrix<To> hC_host(N, N, ldc);
    host_matrix<To> hC_device(N, N, ldc);
    host_matrix<To> hC_cpu(N, N, ldc);

    // Allocate device memory
    device_matrix<Ti>     dA(A_row, A_col, lda);
    device_matrix<To>     dC(N, N, ldc);
    device_vector<Tab_ex> d_alpha(1);
    device_vector<Tab_ex> d_beta(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta.memcheck());

    double gpu_time_used{0}, hipblas_error_host{0}, hipblas_error_device{0};

    // Initial Data on CPU
    hipblas_init_matrix(hA, arg, hipblas_client_never_set_nan, hipblas_general_matrix, true);
    hipblas_init_matrix(hC_host, arg, hipblas_client_never_set_nan, hipblas_hermitian_matrix);

    // copy matrix is easy in STL; hC_device = hC_host: save a copy in hC_device which will be output of device function
    hC_device = hC_host;
    hC_cpu    = hC_host;

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dC.transfer_from(hC_host));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(Tab_ex), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(Tab_ex), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));

        DAPI_CHECK(hipblasHerkExFn,
                   (handle,
                    uplo,
                    transA,
                    N,
                    K,
                    &h_alpha,
                    dA,
                    aType,
                    lda,
                    &h_beta,
                    dC,
                    cType,
                    ldc,
                    computeType));

        CHECK_HIP_ERROR(hC_host.transfer_from(dC));
        CHECK_HIP_ERROR(dC.transfer_from(hC_device));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        DAPI_CHECK(hipblasHerkExFn,
                   (handle,
                    uplo,
                    transA,
                    N,
                    K,
                    d_alpha,
                    dA,
                    aType,
                    lda,
                    d_beta,
                    dC,
                    cType,
                    ldc,
                    computeType));

        CHECK_HIP_ERROR(hC_device.transfer_from(dC));

        /* =====================================================================
           CPU BLAS
        =================================================================== */

        // reference calculation for golden result
        ref_herk_ex<Ti, To, Tab_ex>(
            uplo, transA, N, K, h_alpha, hA.data(), lda, h_beta, hC_cpu.data(), ldc);

        if(arg.unit_check)
        {
            unit_check_general<To>(N, N, ldc, hC_cpu, hC_host);
            unit_check_general<To>(N, N, ldc, hC_cpu, hC_device);
        }

        if(arg.norm_check)
        {
            hipblas_error_host   = norm_check_general<To>('F', N, N, ldc, hC_cpu, hC_host);
            hipblas_error_device = norm_check_general<To>('F', N, N, ldc, hC_cpu, hC_device);
        }
    }

    if(arg.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);
            DAPI_DISPATCH(hipblasHerkExFn,
                          (handle,
                           uplo,
                           transA,
                           N,
                           K,
                           &h_alpha,
                           dA,
                           aType,
                           lda,
                           &h_beta,
                           dC,
                           cType,
                           ldc,
                           computeType));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasHerkExModel{}.log_args<To>(std::cout,
                                          arg,
                                          gpu_time_used,
                                          herk_gflop_count<Ts>(N, K),
                                          herk_gbyte_count<Ts>(N, K),
                                          hipblas_error_host,
                                          hipblas_error_device);
    }
}
