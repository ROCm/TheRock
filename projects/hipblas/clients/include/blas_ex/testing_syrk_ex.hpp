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

using hipblasSyrkExModel = ArgumentModel<e_a_type,
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

inline void testname_syrk_ex(const Arguments& arg, std::string& name)
{
    hipblasSyrkExModel{}.test_name(arg, name);
}

template <typename Ti, typename To = Ti, typename Tex = To>
void testing_syrk_ex_bad_arg(const Arguments& arg)
{
    using Ts = hipblas_internal_type<Tex>;

    auto hipblasSyrkExFn    = arg.api == FORTRAN ? hipblasSyrkExFortran : hipblasSyrkEx;
    auto hipblasSyrkExFn_64 = arg.api == FORTRAN ? hipblasSyrkExFortran : hipblasSyrkEx;
    //auto hipblasSyrkExFn_64 = arg.api == FORTRAN_64 ? hipblasSyrkEx_64Fortran : hipblasSyrkEx_64;

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

    device_vector<Tex> d_alpha(1), d_beta(1), d_one(1), d_zero(1);
    Ts                 h_alpha{1.0f}, h_beta{2.0f}, h_one{1.0f}, h_zero{0.0f};

    if constexpr(std::is_same_v<Tex, hipblasHalf>)
        h_one = float_to_half(1.0f);
    else if constexpr(is_complex<Tex>)
        h_one = {1, 0};

    const Ts* alpha = &h_alpha;
    const Ts* beta  = &h_beta;
    const Ts* one   = &h_one;
    const Ts* zero  = &h_zero;

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

        // clang-format off

        DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
            hipblasSyrkExFn, (nullptr, uplo, transA, N, K, alpha,
                           dA, aType, lda,
                           beta,
                           dC, cType, ldc,
                           computeType));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM, hipblasSyrkExFn, (handle,
                                            (hipblasFillMode_t)HIPBLAS_OP_N,
                                            transA, N, K, alpha,
                                            dA, aType, lda,
                                            beta,
                                            dC, cType, ldc,
                                            computeType));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM, hipblasSyrkExFn, (handle,
                                            uplo,
                                            (hipblasOperation_t)HIPBLAS_FILL_MODE_FULL, 
                                            N, K, alpha,
                                            dA, aType, lda,
                                            beta,
                                            dC, cType, ldc,
                                            computeType));

        if(arg.bad_arg_all)
        {
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                hipblasSyrkExFn, (
                    handle, uplo, transA, N, K, alpha,
                    dA, aType, lda,
                    nullptr,
                    dC, cType, ldc,
                    computeType));

            if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
            {
                // alpha check only for host mode. rocBLAS can handle this in device mode too but shouldn't assume in case this changes.
                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasSyrkExFn, (
                        handle, uplo, transA, N, K, nullptr,
                        dA, aType, lda,
                        beta,
                        dC, cType, ldc,
                        computeType));

                // again, rocBLAS can handle this in device mode but shouldn't assume
                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE, hipblasSyrkExFn, (handle, uplo, transA, N, K, alpha,
                                                    nullptr, aType, lda,
                                                    beta,
                                                    dC, cType, ldc,
                                                    computeType));

                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE, hipblasSyrkExFn, (handle, uplo, transA, N, K, alpha,
                                                    dA, aType, lda,
                                                    beta,
                                                    nullptr, cType, ldc,
                                                    computeType));

                // If alpha == 0, A can be nullptr
                DAPI_CHECK(hipblasSyrkExFn, (
                    handle, uplo, transA, N, K, zero,
                    nullptr, aType, lda,
                    beta,
                    dC, cType, ldc,
                    computeType));

                // TODO, rocblas change
                // If K == 0, alpha, A can be nullptr
                // DAPI_CHECK(hipblasSyrkExFn, (handle, uplo, transA, N, 0, nullptr,
                //                                   nullptr, aType, lda,
                //                                   beta,
                //                                   dC, cType, ldc,
                //                                   computeType));

            }
                    

                
        }

        // If N == 0, can have nullptrs
        DAPI_CHECK(hipblasSyrkExFn, (handle, uplo, transA, 0, K, nullptr,
                                          nullptr, aType, lda,
                                          nullptr,
                                          nullptr, cType, ldc,
                                          computeType));

        // clang-format on
    }
}

template <typename Ti, typename To = Ti, typename Tex = To>
void testing_syrk_ex(const Arguments& arg)
{
    using Ts = hipblas_internal_type<Tex>;

    auto hipblasSyrkExFn    = arg.api == FORTRAN ? hipblasSyrkExFortran : hipblasSyrkEx;
    auto hipblasSyrkExFn_64 = arg.api == FORTRAN ? hipblasSyrkExFortran : hipblasSyrkEx;
    //auto hipblasSyrkExFn_64 = arg.api == FORTRAN_64 ? hipblasSyrkEx_64Fortran : hipblasSyrkEx_64;

    hipblasFillMode_t  uplo   = char2hipblas_fill(arg.uplo);
    hipblasOperation_t transA = char2hipblas_operation(arg.transA);
    int64_t            N      = arg.N;
    int64_t            K      = arg.K;
    int64_t            lda    = arg.lda;
    int64_t            ldc    = arg.ldc;

    hipDataType a_type       = arg.a_type;
    hipDataType c_type       = arg.c_type;
    hipDataType compute_type = arg.compute_type;

    Tex h_alpha_Tex = arg.get_alpha<Tex>();
    Tex h_beta_Tex  = arg.get_beta<Tex>();

    int norm_check = arg.norm_check;
    int unit_check = arg.unit_check;
    int timing     = arg.timing;

    int64_t A_row = transA == HIPBLAS_OP_N ? N : K;
    int64_t A_col = transA == HIPBLAS_OP_N ? K : N;

    hipblasLocalHandle handle(arg);

    // check here to prevent undefined memory allocation error
    bool invalid_size = N < 0 || K < 0 || lda < A_row || ldc < N;
    if(invalid_size || !N)
    {
        DAPI_EXPECT(invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS,
                    hipblasSyrkExFn,
                    (handle,
                     uplo,
                     transA,
                     N,
                     K,
                     nullptr,
                     nullptr,
                     a_type,
                     lda,
                     nullptr,
                     nullptr,
                     c_type,
                     ldc,
                     compute_type));

        return;
    }

    // Allocate host memory
    host_matrix<Ti> hA(A_row, A_col, lda);
    host_matrix<To> hC_host(N, N, ldc);
    host_matrix<To> hC_device(N, N, ldc);
    host_matrix<To> hC_gold(N, N, ldc);

    // Allocate device memory
    device_matrix<Ti>  dA(A_row, A_col, lda);
    device_matrix<To>  dC(N, N, ldc);
    device_vector<Tex> d_alpha(1);
    device_vector<Tex> d_beta(1);

    double gpu_time_used{0}, hipblas_error_host{0}, hipblas_error_device{0};

    // Initial Data on CPU
    hipblas_init_matrix(hA, arg, hipblas_client_alpha_sets_nan, hipblas_general_matrix, true);
    hipblas_init_matrix(hC_host, arg, hipblas_client_beta_sets_nan, hipblas_symmetric_matrix);

    hC_device = hC_host;
    hC_gold   = hC_host;

    // copy data from CPU to device

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dC.transfer_from(hC_host));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha_Tex, sizeof(Tex), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta_Tex, sizeof(Tex), hipMemcpyHostToDevice));

    if(unit_check || norm_check)
    {
        // hipBLAS
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));

        DAPI_CHECK(hipblasSyrkExFn,
                   (handle,
                    uplo,
                    transA,
                    N,
                    K,
                    &h_alpha_Tex,
                    dA,
                    a_type,
                    lda,
                    &h_beta_Tex,
                    dC,
                    c_type,
                    ldc,
                    compute_type));

        CHECK_HIP_ERROR(hC_host.transfer_from(dC));
        CHECK_HIP_ERROR(dC.transfer_from(hC_device));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        DAPI_CHECK(hipblasSyrkExFn,
                   (handle,
                    uplo,
                    transA,
                    N,
                    K,
                    d_alpha,
                    dA,
                    a_type,
                    lda,
                    d_beta,
                    dC,
                    c_type,
                    ldc,
                    compute_type));

        CHECK_HIP_ERROR(hC_device.transfer_from(dC));

        // reference BLAS
        ref_syrk_ex<Ti, To, Tex>(
            uplo, transA, N, K, h_alpha_Tex, hA.data(), lda, h_beta_Tex, hC_gold.data(), ldc);

        if(unit_check)
        {
            // check for float16/bfloat16 input
            if(arg.initialization != rand_int
               || ((getArchMajor() == 11)
                   && ((std::is_same<Tex, float>{} && std::is_same<Ti, hipblasBfloat16>{})
                       || (std::is_same<Tex, float>{} && std::is_same<Ti, hipblasHalf>{})
                       || (std::is_same<Tex, hipblasHalf>{} && std::is_same<Ti, hipblasHalf>{}))))
            {
                const double tol = K * sum_error_tolerance_for_gfx11<Tex, Ti, To>;
                near_check_general<To>(N, N, ldc, hC_gold.data(), hC_host.data(), tol);
                near_check_general<To>(N, N, ldc, hC_gold.data(), hC_device.data(), tol);
            }
            else
            {
                unit_check_general<To>(N, N, ldc, hC_gold, hC_host);
                unit_check_general<To>(N, N, ldc, hC_gold, hC_device);
            }
        }
        if(norm_check)
        {
            hipblas_error_host = hipblas_abs(
                norm_check_general<To>('F', N, N, ldc, hC_gold.data(), hC_host.data()));
            hipblas_error_device = hipblas_abs(
                norm_check_general<To>('F', N, N, ldc, hC_gold.data(), hC_device.data()));
        }
    }

    if(timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(hipblasSyrkExFn,
                          (handle,
                           uplo,
                           transA,
                           N,
                           K,
                           &h_alpha_Tex,
                           dA,
                           a_type,
                           lda,
                           &h_beta_Tex,
                           dC,
                           c_type,
                           ldc,
                           compute_type));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasSyrkExModel{}.log_args<To>(std::cout,
                                          arg,
                                          gpu_time_used,
                                          syrk_gflop_count<Tex>(N, K),
                                          syrk_gbyte_count<Tex>(N, K),
                                          hipblas_error_host,
                                          hipblas_error_device);
    }
}
