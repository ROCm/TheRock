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

using hipblasGemmExModel = ArgumentModel<e_a_type,
                                         e_c_type,
                                         e_compute_type,
                                         e_transA,
                                         e_transB,
                                         e_M,
                                         e_N,
                                         e_K,
                                         e_alpha,
                                         e_lda,
                                         e_ldb,
                                         e_beta,
                                         e_ldc,
                                         e_with_flags,
                                         e_flags>;

inline void testname_gemm_ex(const Arguments& arg, std::string& name)
{
    hipblasGemmExModel{}.test_name(arg, name);
}

template <typename Ti, typename To = Ti, typename Tex = To>
void testing_gemm_ex_bad_arg(const Arguments& arg)
{
    using Ts = hipblas_internal_type<Tex>;
    // Note: hipblasGemmEx and hipblasGemmExWithFlags are essentially the exact same.
    //       Only testing WithFlags version as it has slightly more functionality.
    auto hipblasGemmExFn
        = arg.api == FORTRAN ? hipblasGemmExWithFlagsFortran : hipblasGemmExWithFlags;
    auto hipblasGemmExFn_64
        = arg.api == FORTRAN_64 ? hipblasGemmExWithFlags_64Fortran : hipblasGemmExWithFlags_64;

    hipblasLocalHandle handle(arg);

    hipDataType          aType       = arg.a_type;
    hipDataType          bType       = arg.b_type;
    hipDataType          cType       = arg.c_type;
    hipblasComputeType_t computeType = arg.compute_type_gemm;
    hipblasGemmFlags_t   flags       = HIPBLAS_GEMM_FLAGS_NONE;
    hipblasGemmAlgo_t    algo        = HIPBLAS_GEMM_DEFAULT;

    int64_t M   = 101;
    int64_t N   = 100;
    int64_t K   = 102;
    int64_t lda = 103;
    int64_t ldb = 104;
    int64_t ldc = 105;

    hipblasOperation_t transA = HIPBLAS_OP_N;
    hipblasOperation_t transB = HIPBLAS_OP_N;

    int64_t A_row = transA == HIPBLAS_OP_N ? M : std::max(K, int64_t(1));
    int64_t A_col = transA == HIPBLAS_OP_N ? std::max(K, int64_t(1)) : M;
    int64_t B_row = transB == HIPBLAS_OP_N ? std::max(K, int64_t(1)) : N;
    int64_t B_col = transB == HIPBLAS_OP_N ? N : std::max(K, int64_t(1));

    // Allocate device memory
    device_matrix<Ti> dA(A_row, A_col, lda);
    device_matrix<Ti> dB(B_row, B_col, ldb);
    device_matrix<To> dC(M, N, ldc);

    device_vector<Tex> d_alpha(1), d_beta(1), d_one(1), d_zero(1);
    Ts                 h_alpha, h_beta, h_one, h_zero;

    if constexpr(is_complex<Tex>)
    {
        h_alpha = {1.0, 0.0};
        h_beta  = {2.0, 0.0};
        h_one   = {1.0, 0.0};
        h_zero  = {0.0, 0.0};
    }
    else
    {
        h_alpha = Ts(1.0f);
        h_beta  = Ts(2.0f);
        h_one   = Ts(1.0f);
        h_zero  = Ts(0.0f);
    }

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
            hipblasGemmExFn, (nullptr, transA, transB, M, N, K, alpha,
                           dA, aType, lda,
                           dB, bType, ldb, beta,
                           dC, cType, ldc,
                           computeType,
                           algo, flags));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM, hipblasGemmExFn, (handle,
                                            (hipblasOperation_t)HIPBLAS_FILL_MODE_FULL,
                                            transB, M, N, K, alpha,
                                            dA, aType, lda,
                                            dB, bType, ldb, beta,
                                            dC, cType, ldc,
                                            computeType,
                                            algo, flags));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM, hipblasGemmExFn, (handle, transA,
                                            (hipblasOperation_t)HIPBLAS_FILL_MODE_FULL,
                                            M, N, K, alpha,
                                            dA, aType, lda,
                                            dB, bType, ldb, beta,
                                            dC, cType, ldc,
                                            computeType,
                                            algo, flags));

                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM, hipblasGemmExFn, (handle, transA, transB, M, N, K, alpha,
                                            dA, aType, lda,
                                            dB, bType, ldb, beta,
                                            dC, cType, ldc,
                                            computeType,
                                            (hipblasGemmAlgo_t)HIPBLAS_OP_N,
                                            flags));

        if(arg.bad_arg_all)
        {
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                hipblasGemmExFn, (
                    handle, transA, transB, M, N, K, alpha,
                    dA, aType, lda,
                    dB, bType, ldb, nullptr,
                    dC, cType, ldc,
                    computeType,
                    algo, flags));

            if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
            {
                // alpha check only for host mode. rocBLAS can handle this in device mode too but shouldn't assume in case this changes.
                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasGemmExFn, (
                        handle, transA, transB, M, N, K, nullptr,
                        dA, aType, lda,
                        dB, bType, ldb, beta,
                        dC, cType, ldc,
                        computeType,
                        algo, flags));

                // again, rocBLAS can handle this in device mode but shouldn't assume
                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE, hipblasGemmExFn, (handle, transA, transB, M, N, K, alpha,
                                                    nullptr, aType, lda,
                                                    dB, bType, ldb, beta,
                                                    dC, cType, ldc,
                                                    computeType,
                                                    algo, flags));
                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE, hipblasGemmExFn, (handle, transA, transB, M, N, K, alpha,
                                                    dA, aType, lda,
                                                    nullptr, bType, ldb, beta,
                                                    dC, cType, ldc,
                                                    computeType,
                                                    algo, flags));
                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE, hipblasGemmExFn, (handle, transA, transB, M, N, K, alpha,
                                                    dA, aType, lda,
                                                    dB, bType, ldb, beta,
                                                    nullptr, cType, ldc,
                                                    computeType,
                                                    algo, flags));

                // gemm_ex doesn't have immediate quick return for alpha == 0 && beta == 1. First checking with all args except N > 32-bit, then testing N > 32-bit
                DAPI_EXPECT((arg.api & c_API_64) ? HIPBLAS_STATUS_SUCCESS
                                                : HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasGemmExFn,
                            (handle,
                            transA,
                            transB,
                            c_i32_overflow,
                            0,
                            c_i32_overflow,
                            zero,
                            nullptr,
                            aType,
                            c_i32_overflow,
                            nullptr,
                            bType,
                            c_i32_overflow,
                            one,
                            nullptr,
                            cType,
                            c_i32_overflow,
                            computeType,
                            algo,
                            flags));
                DAPI_EXPECT((arg.api & c_API_64) ? HIPBLAS_STATUS_SUCCESS
                                                : HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasGemmExFn,
                            (handle,
                            transA,
                            transB,
                            0,
                            c_i32_overflow,
                            c_i32_overflow,
                            zero,
                            nullptr,
                            aType,
                            c_i32_overflow,
                            nullptr,
                            bType,
                            c_i32_overflow,
                            one,
                            nullptr,
                            cType,
                            c_i32_overflow,
                            computeType,
                            algo,
                            flags));
            }

            // If alpha == 0, A and B can be nullptr
            DAPI_CHECK(hipblasGemmExFn, (
                handle, transA, transB, M, N, K, zero,
                nullptr, aType, lda,
                nullptr, bType, ldb, beta,
                dC, cType, ldc,
                computeType,
                algo, flags));

            // If K == 0, alpha, A, and B can be nullptr
            DAPI_CHECK(hipblasGemmExFn, (handle, transA, transB, M, N, 0, nullptr,
                                              nullptr, aType, lda,
                                              nullptr, bType, ldb, beta,
                                              dC, cType, ldc,
                                              computeType,
                                              algo, flags));
        }

        // If M == 0 || N == 0, can have nullptrs
        DAPI_CHECK(hipblasGemmExFn, (handle, transA, transB, 0, N, K, nullptr,
                                          nullptr, aType, lda,
                                          nullptr, bType, ldb, nullptr,
                                          nullptr, cType, ldc,
                                          computeType,
                                          algo, flags));
        DAPI_CHECK(hipblasGemmExFn, (handle, transA, transB, M, 0, K, nullptr,
                                          nullptr, aType, lda,
                                          nullptr, bType, ldb, nullptr,
                                          nullptr, cType, ldc,
                                          computeType,
                                          algo, flags));

        // clang-format on
    }
}

template <typename Ti, typename To = Ti, typename Tex = To>
void testing_gemm_ex(const Arguments& arg)
{
    using Ts             = hipblas_internal_type<Tex>;
    auto hipblasGemmExFn = arg.api == FORTRAN ? hipblasGemmExFortran : hipblasGemmEx;
    auto hipblasGemmExWithFlagsFn
        = arg.api == FORTRAN ? hipblasGemmExWithFlagsFortran : hipblasGemmExWithFlags;
    auto hipblasGemmExFn_64 = arg.api == FORTRAN_64 ? hipblasGemmEx_64Fortran : hipblasGemmEx_64;
    auto hipblasGemmExWithFlagsFn_64
        = arg.api == FORTRAN_64 ? hipblasGemmExWithFlags_64Fortran : hipblasGemmExWithFlags_64;

    hipblasGemmAlgo_t algo           = HIPBLAS_GEMM_DEFAULT;
    size_t*           workspace_size = 0;
    void*             workspace      = 0;

    hipblasOperation_t transA = char2hipblas_operation(arg.transA);
    hipblasOperation_t transB = char2hipblas_operation(arg.transB);
    int64_t            M      = arg.M;
    int64_t            N      = arg.N;
    int64_t            K      = arg.K;
    int64_t            lda    = arg.lda;
    int64_t            ldb    = arg.ldb;
    int64_t            ldc    = arg.ldc;

    hipDataType          a_type       = arg.a_type;
    hipDataType          b_type       = arg.b_type;
    hipDataType          c_type       = arg.c_type;
    hipblasComputeType_t compute_type = arg.compute_type_gemm;
    hipblasGemmFlags_t   flags        = hipblasGemmFlags_t(arg.flags);

    Tex h_alpha_Tex = arg.get_alpha<Tex>();
    Tex h_beta_Tex  = arg.get_beta<Tex>();

    int norm_check = arg.norm_check;
    int unit_check = arg.unit_check;
    int timing     = arg.timing;

    int64_t A_row = transA == HIPBLAS_OP_N ? M : K;
    int64_t A_col = transA == HIPBLAS_OP_N ? K : M;
    int64_t B_row = transB == HIPBLAS_OP_N ? K : N;
    int64_t B_col = transB == HIPBLAS_OP_N ? N : K;

    hipblasLocalHandle handle(arg);

    // check here to prevent undefined memory allocation error
    bool invalid_size = M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M;
    if(invalid_size || !M || !N)
    {
        DAPI_EXPECT(invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS,
                    hipblasGemmExFn,
                    (handle,
                     transA,
                     transB,
                     M,
                     N,
                     K,
                     nullptr,
                     nullptr,
                     a_type,
                     lda,
                     nullptr,
                     b_type,
                     ldb,
                     nullptr,
                     nullptr,
                     c_type,
                     ldc,
                     compute_type,
                     algo));

        return;
    }

    // Allocate host memory
    host_matrix<Ti> hA(A_row, A_col, lda);
    host_matrix<Ti> hB(B_row, B_col, ldb);
    host_matrix<To> hC_host(M, N, ldc);
    host_matrix<To> hC_device(M, N, ldc);
    host_matrix<To> hC_gold(M, N, ldc);

    // Allocate device memory
    device_matrix<Ti>  dA(A_row, A_col, lda);
    device_matrix<Ti>  dB(B_row, B_col, ldb);
    device_matrix<To>  dC(M, N, ldc);
    device_vector<Tex> d_alpha(1);
    device_vector<Tex> d_beta(1);

    double gpu_time_used{0}, hipblas_error_host{0}, hipblas_error_device{0};

    // Initial Data on CPU
    hipblas_init_matrix(hA, arg, hipblas_client_alpha_sets_nan, hipblas_general_matrix, true);
    hipblas_init_matrix(
        hB, arg, hipblas_client_alpha_sets_nan, hipblas_general_matrix, false, true);
    hipblas_init_matrix(hC_host, arg, hipblas_client_beta_sets_nan, hipblas_general_matrix);

    hC_gold = hC_device = hC_host;

    // copy data from CPU to device

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));
    CHECK_HIP_ERROR(dC.transfer_from(hC_host));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha_Tex, sizeof(Tex), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta_Tex, sizeof(Tex), hipMemcpyHostToDevice));

    if(unit_check || norm_check)
    {
        // hipBLAS
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        if(!arg.with_flags)
        {
            DAPI_CHECK(hipblasGemmExFn,
                       (handle,
                        transA,
                        transB,
                        M,
                        N,
                        K,
                        reinterpret_cast<Ts*>(&h_alpha_Tex),
                        dA,
                        a_type,
                        lda,
                        dB,
                        b_type,
                        ldb,
                        reinterpret_cast<Ts*>(&h_beta_Tex),
                        dC,
                        c_type,
                        ldc,
                        compute_type,
                        algo));
        }
        else
        {
            DAPI_CHECK(hipblasGemmExWithFlagsFn,
                       (handle,
                        transA,
                        transB,
                        M,
                        N,
                        K,
                        &h_alpha_Tex,
                        dA,
                        a_type,
                        lda,
                        dB,
                        b_type,
                        ldb,
                        &h_beta_Tex,
                        dC,
                        c_type,
                        ldc,
                        compute_type,
                        algo,
                        flags));
        }

        CHECK_HIP_ERROR(hC_host.transfer_from(dC));
        CHECK_HIP_ERROR(dC.transfer_from(hC_device));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        if(!arg.with_flags)
        {
            DAPI_CHECK(hipblasGemmExFn,
                       (handle,
                        transA,
                        transB,
                        M,
                        N,
                        K,
                        d_alpha,
                        dA,
                        a_type,
                        lda,
                        dB,
                        b_type,
                        ldb,
                        d_beta,
                        dC,
                        c_type,
                        ldc,
                        compute_type,
                        algo));
        }
        else
        {
            DAPI_CHECK(hipblasGemmExWithFlagsFn,
                       (handle,
                        transA,
                        transB,
                        M,
                        N,
                        K,
                        d_alpha,
                        dA,
                        a_type,
                        lda,
                        dB,
                        b_type,
                        ldb,
                        d_beta,
                        dC,
                        c_type,
                        ldc,
                        compute_type,
                        algo,
                        flags));
        }

        CHECK_HIP_ERROR(hC_device.transfer_from(dC));

        // reference BLAS
        ref_gemm<Ti, To, Tex>(transA,
                              transB,
                              M,
                              N,
                              K,
                              h_alpha_Tex,
                              hA.data(),
                              lda,
                              hB.data(),
                              ldb,
                              h_beta_Tex,
                              hC_gold.data(),
                              ldc);

        if(unit_check)
        {
            // check for float16/bfloat16 input
            if((getArchMajor() == 11)
               && ((std::is_same<Tex, float>{} && std::is_same<Ti, hipblasBfloat16>{})
                   || (std::is_same<Tex, float>{} && std::is_same<Ti, hipblasHalf>{})
                   || (std::is_same<Tex, hipblasHalf>{} && std::is_same<Ti, hipblasHalf>{})))
            {
                const double tol = K * sum_error_tolerance_for_gfx11<Tex, Ti, To>;
                near_check_general<To>(M, N, ldc, hC_gold.data(), hC_host.data(), tol);
                near_check_general<To>(M, N, ldc, hC_gold.data(), hC_device.data(), tol);
            }
            else
            {
                unit_check_general<To>(M, N, ldc, hC_gold, hC_host);
                unit_check_general<To>(M, N, ldc, hC_gold, hC_device);
            }
        }
        if(norm_check)
        {
            hipblas_error_host = hipblas_abs(
                norm_check_general<To>('F', M, N, ldc, hC_gold.data(), hC_host.data()));
            hipblas_error_device = hipblas_abs(
                norm_check_general<To>('F', M, N, ldc, hC_gold.data(), hC_device.data()));
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

            if(!arg.with_flags)
            {
                DAPI_DISPATCH(hipblasGemmExFn,
                              (handle,
                               transA,
                               transB,
                               M,
                               N,
                               K,
                               &h_alpha_Tex,
                               dA,
                               a_type,
                               lda,
                               dB,
                               b_type,
                               ldb,
                               &h_beta_Tex,
                               dC,
                               c_type,
                               ldc,
                               compute_type,
                               algo));
            }
            else
            {
                DAPI_DISPATCH(hipblasGemmExWithFlagsFn,
                              (handle,
                               transA,
                               transB,
                               M,
                               N,
                               K,
                               &h_alpha_Tex,
                               dA,
                               a_type,
                               lda,
                               dB,
                               b_type,
                               ldb,
                               &h_beta_Tex,
                               dC,
                               c_type,
                               ldc,
                               compute_type,
                               algo,
                               flags));
            }
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasGemmExModel{}.log_args<To>(std::cout,
                                          arg,
                                          gpu_time_used,
                                          gemm_gflop_count<Tex>(M, N, K),
                                          gemm_gbyte_count<Tex>(M, N, K),
                                          hipblas_error_host,
                                          hipblas_error_device);
    }
}
