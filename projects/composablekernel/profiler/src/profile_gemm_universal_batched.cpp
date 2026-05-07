// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "profiler/profile_gemm_universal_batched_impl.hpp"
#include "profiler_operation_registry.hpp"

#include "ck/library/tensor_operation_instance/gpu/gemm_universal_batched.hpp"

enum struct GemmMatrixLayout
{
    MK_KN_MN, // 0
    MK_NK_MN, // 1
    KM_KN_MN, // 2
    KM_NK_MN, // 3
};

enum struct GemmDataType
{
    BF16_BF16_BF16, // 0
    F8_F8_BF16,     // 1
};

#define OP_NAME "gemm_universal_batched"
#define OP_DESC "Batched GEMM Universal"

int profile_batched_gemm_universal(int argc, char* argv[])
{
    if(argc != 19 && argc != 22)
    {
        // clang-format off
        printf("arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n");
        printf("arg2: data type (0: bf16, 1: fp8->bf16)\n");
        printf("arg3: matrix layout (0: A[g, m, k] * B[g, k, n] = C[g, m, n];\n");
        printf("                     1: A[g, m, k] * B[g, n, k] = C[g, m, n];\n");
        printf("                     2: A[g, k, m] * B[g, k, n] = C[g, m, n];\n");
        printf("                     3: A[g, k, m] * B[g, n, k] = C[g, m, n])\n");
        printf("arg4: verification (0: no; 1: yes)\n");
        printf("arg5: initialization (0: no init; 1: integer value; 2: decimal value)\n");
        printf("arg6: print tensor value (0: no; 1: yes)\n");
        printf("arg7: time kernel (0=n0, 1=yes)\n");
        printf("arg8 to 18: M, N, K, StrideA, StrideB, StrideC, BatchStrideA, BatchStrideB, BatchStrideC, BatchCount, KBatch\n");
        printf("optional:\n");
        printf("arg19: number of warm-up cycles (default 1)\n");
        printf("arg20: number of iterations (default 10)\n");
        printf("arg21: memory for rotating buffer (default 0, size in MB)\n");
        // clang-format on
        exit(1);
    }

    int n_warmup      = 1;
    int n_iter        = 10;
    uint64_t rotating = 0;
    if(argc == 22)
    {
        n_warmup = std::stoi(argv[19]);
        n_iter   = std::stoi(argv[20]);
        rotating = std::stoull(argv[21]) * 1024 * 1024;
    }

    const auto data_type       = static_cast<GemmDataType>(std::stoi(argv[2]));
    const auto layout          = static_cast<GemmMatrixLayout>(std::stoi(argv[3]));
    const bool do_verification = std::stoi(argv[4]);
    const int init_method      = std::stoi(argv[5]);
    const bool do_log          = std::stoi(argv[6]);
    const bool time_kernel     = std::stoi(argv[7]);

    const int M = std::stoi(argv[8]);
    const int N = std::stoi(argv[9]);
    const int K = std::stoi(argv[10]);

    const int StrideA = std::stoi(argv[11]);
    const int StrideB = std::stoi(argv[12]);
    const int StrideC = std::stoi(argv[13]);

    const int BatchStrideA = std::stoi(argv[14]);
    const int BatchStrideB = std::stoi(argv[15]);
    const int BatchStrideC = std::stoi(argv[16]);

    const int BatchCount = std::stoi(argv[17]);
    const int KBatch     = std::stoi(argv[18]);

#if defined(CK_USE_FP8_ON_UNSUPPORTED_ARCH) || defined(CK_USE_GFX94)
    using F8 = ck::f8_t;
#endif
    using BF16 = ck::bhalf_t;

    using Row = ck::tensor_layout::gemm::RowMajor;
    using Col = ck::tensor_layout::gemm::ColumnMajor;

    auto profile =
        [&](auto a_type, auto b_type, auto c_type, auto a_layout, auto b_layout, auto c_layout) {
            using ADataType  = decltype(a_type);
            using BDataType  = decltype(b_type);
            using DsDataType = ck::Tuple<>;
            using CDataType  = decltype(c_type);

            using ALayout  = decltype(a_layout);
            using BLayout  = decltype(b_layout);
            using DsLayout = ck::Tuple<>;
            using CLayout  = decltype(c_layout);

            const int DefaultStrideA = ck::is_same_v<ALayout, Row> ? K : M;
            const int DefaultStrideB = ck::is_same_v<BLayout, Row> ? N : K;
            const int DefaultStrideC = ck::is_same_v<CLayout, Row> ? N : M;

            const int StrideA_ = (StrideA < 0) ? DefaultStrideA : StrideA;
            const int StrideB_ = (StrideB < 0) ? DefaultStrideB : StrideB;
            const int StrideC_ = (StrideC < 0) ? DefaultStrideC : StrideC;

            const int DefaultBatchStrideA = (ck::is_same_v<ALayout, Row> ? M : K) * StrideA_;
            const int DefaultBatchStrideB = (ck::is_same_v<BLayout, Row> ? K : N) * StrideB_;
            const int DefaultBatchStrideC = (ck::is_same_v<CLayout, Row> ? M : N) * StrideC_;

            const int BatchStrideA_ = (BatchStrideA < 0) ? DefaultBatchStrideA : BatchStrideA;
            const int BatchStrideB_ = (BatchStrideB < 0) ? DefaultBatchStrideB : BatchStrideB;
            const int BatchStrideC_ = (BatchStrideC < 0) ? DefaultBatchStrideC : BatchStrideC;

            using AElementOp = ck::tensor_operation::element_wise::PassThrough;
            using BElementOp = ck::tensor_operation::element_wise::PassThrough;
            using CElementOp = ck::tensor_operation::element_wise::PassThrough;

            using DeviceOp = ck::tensor_operation::device::DeviceBatchedGemmV2MultiD<ALayout,
                                                                                     BLayout,
                                                                                     DsLayout,
                                                                                     CLayout,
                                                                                     ADataType,
                                                                                     BDataType,
                                                                                     DsDataType,
                                                                                     CDataType,
                                                                                     AElementOp,
                                                                                     BElementOp,
                                                                                     CElementOp>;

            bool pass = ck::profiler::profile_gemm_universal_batched_impl<ADataType,
                                                                          BDataType,
                                                                          CDataType,
                                                                          ALayout,
                                                                          BLayout,
                                                                          CLayout,
                                                                          AElementOp,
                                                                          BElementOp,
                                                                          CElementOp,
                                                                          DeviceOp>(do_verification,
                                                                                    init_method,
                                                                                    do_log,
                                                                                    time_kernel,
                                                                                    M,
                                                                                    N,
                                                                                    K,
                                                                                    BatchStrideA_,
                                                                                    BatchStrideB_,
                                                                                    BatchStrideC_,
                                                                                    StrideA_,
                                                                                    StrideB_,
                                                                                    StrideC_,
                                                                                    BatchCount,
                                                                                    KBatch,
                                                                                    n_warmup,
                                                                                    n_iter,
                                                                                    rotating);

            return pass ? 0 : 1;
        };

    if(data_type == GemmDataType::BF16_BF16_BF16 && layout == GemmMatrixLayout::MK_NK_MN)
    {
        return profile(BF16{}, BF16{}, BF16{}, Row{}, Col{}, Row{});
    }
#if defined(CK_USE_FP8_ON_UNSUPPORTED_ARCH) || defined(CK_USE_GFX94)
    else if(data_type == GemmDataType::F8_F8_BF16 && layout == GemmMatrixLayout::MK_NK_MN)
    {
        return profile(F8{}, F8{}, BF16{}, Row{}, Col{}, Row{});
    }
#endif
    else
    {
        std::cout << "this data_type & layout is not implemented" << std::endl;

        return 1;
    }
}

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_batched_gemm_universal);
