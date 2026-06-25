// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "profiler/profile_gemm_mx_impl.hpp"
#include "profiler_operation_registry.hpp"

enum struct GemmMatrixLayout
{
    MK_KN_MN,   // 0
    MK_NK_MN,   // 1
    MK_MFMA_MN, // 2
};

enum struct GemmDataType
{
    F4_F4_F16,  // 0
    F8_F8_F16,  // 1
    F8_F8_BF16, // 2
};

#define OP_NAME "gemm_mx"
#define OP_DESC "GEMM_mx"

int profile_gemm_mx(int argc, char* argv[])
{
    if(argc != 11 && argc != 14 && argc != 18)
    {
        printf("arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n");
        printf("arg2: data type (0: f4->f16   ;\n");
        printf("                 1: fp8->f16  ;\n");
        printf("                 2: fp8->bf16 )\n");
        printf("arg3: matrix layout (0: A[m, k] * B[k, n] = C[m, n]  ;\n");
        printf("                     1: A[m, k] * B[n, k] = C[m, n]  ;\n");
        printf("                     2: A[k, m] * BPreShuff = C[m, n])\n");
        printf("arg4: verification (0: no; 1: yes)\n");
        printf("arg5: initialization (0: no init; 1: integer value; 2: decimal value)\n");
        printf("arg6: print tensor value (0: no; 1: yes)\n");
        printf("arg7: time kernel (0=no, 1=yes)\n");
        printf("arg8 to 13: M, N, K, StrideA, StrideB, StrideC\n");
        printf("optional:\n");
        printf("arg14: number of kbatch (default 1)\n");
        printf("arg15: number of warm-up cycles (default 1)\n");
        printf("arg16: number of iterations (default 10)\n");
        printf("arg17: memory for rotating buffer (default 0, size in MB)\n");
        exit(1);
    }
    int arg_index              = 2;
    const auto data_type       = static_cast<GemmDataType>(std::stoi(argv[arg_index++]));
    const auto layout          = static_cast<GemmMatrixLayout>(std::stoi(argv[arg_index++]));
    const bool do_verification = std::stoi(argv[arg_index++]);
    const int init_method      = std::stoi(argv[arg_index++]);
    const bool do_log          = std::stoi(argv[arg_index++]);
    const bool time_kernel     = std::stoi(argv[arg_index++]);

    const int M = std::stoi(argv[arg_index++]);
    const int N = std::stoi(argv[arg_index++]);
    const int K = std::stoi(argv[arg_index++]);

    int StrideA = -1, StrideB = -1, StrideC = -1;
    if(argc > arg_index)
    {
        StrideA = std::stoi(argv[arg_index++]);
        StrideB = std::stoi(argv[arg_index++]);
        StrideC = std::stoi(argv[arg_index++]);
    }

    int KBatch        = 1;
    int n_warmup      = 1;
    int n_iter        = 10;
    uint64_t rotating = 0;
    if(argc > arg_index)
    {
        KBatch   = std::stoi(argv[arg_index++]);
        n_warmup = std::stoi(argv[arg_index++]);
        n_iter   = std::stoi(argv[arg_index++]);
        rotating = std::stoull(argv[arg_index++]) * 1024 * 1024;
    }

    using F16  = ck::half_t;
    using BF16 = ck::bhalf_t;
    using F4   = ck::f4x2_pk_t;
    using F8   = ck::f8_t;

    using Row  = ck::tensor_layout::gemm::RowMajor;
    using Col  = ck::tensor_layout::gemm::ColumnMajor;
    using MFMA = ck::tensor_layout::gemm::MFMA;

    auto profile =
        [&](auto a_type, auto b_type, auto c_type, auto a_layout, auto b_layout, auto c_layout) {
            using ADataType = decltype(a_type);
            using BDataType = decltype(b_type);
            using CDataType = decltype(c_type);
            using ALayout   = decltype(a_layout);
            using BLayout   = decltype(b_layout);
            using CLayout   = decltype(c_layout);

            const int DefaultStrideA = ck::is_same_v<ALayout, Row> ? K : M;
            const int DefaultStrideB = ck::is_same_v<BLayout, Row> ? N : K;
            const int DefaultStrideC = ck::is_same_v<CLayout, Row> ? N : M;

            bool pass = ck::profiler::profile_gemm_mx_impl<ADataType,
                                                           BDataType,
                                                           CDataType,
                                                           ALayout,
                                                           BLayout,
                                                           CLayout,
                                                           32>( //
                do_verification,
                init_method,
                do_log,
                time_kernel,
                M,
                N,
                K,
                (StrideA < 0) ? DefaultStrideA : StrideA,
                (StrideB < 0) ? DefaultStrideB : StrideB,
                (StrideC < 0) ? DefaultStrideC : StrideC,
                KBatch,
                n_warmup,
                n_iter,
                rotating);

            return pass ? 0 : 1;
        };

    if(data_type == GemmDataType::F4_F4_F16 && layout == GemmMatrixLayout::MK_NK_MN)
    {
        return profile(F4{}, F4{}, F16{}, Row{}, Col{}, Row{});
    }
    else if(data_type == GemmDataType::F4_F4_F16 && layout == GemmMatrixLayout::MK_MFMA_MN)
    {
        return profile(F4{}, F4{}, F16{}, Row{}, MFMA{}, Row{});
    }
    else if(data_type == GemmDataType::F8_F8_F16 && layout == GemmMatrixLayout::MK_NK_MN)
    {
        return profile(F8{}, F8{}, F16{}, Row{}, Col{}, Row{});
    }
    else if(data_type == GemmDataType::F8_F8_BF16 && layout == GemmMatrixLayout::MK_NK_MN)
    {
        return profile(F8{}, F8{}, BF16{}, Row{}, Col{}, Row{});
    }
    else
    {
        std::cout << "this data_type & layout is not implemented" << std::endl;

        return 1;
    }
}

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_gemm_mx);
