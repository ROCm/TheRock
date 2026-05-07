// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "profiler/profile_gemm_multiply_multiply_wp_impl.hpp"
#include "profiler_operation_registry.hpp"

enum struct GemmMatrixLayout
{
    MK_MFMA_MN, // 0
};

enum struct GemmDataType
{
    F8_F8_F16,  // 0
    F8_F8_BF16, // 1
};

#define OP_NAME "gemm_multiply_multiply_weight_preshuffle"
#define OP_DESC "GEMM_Multiply_Multiply_Weight_PreShuffle"

int profile_gemm_multiply_multiply_weight_preshuffle(int argc, char* argv[])
{
    if(argc != 16 && argc != 20)
    {
        printf("arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n");
        printf("arg2: data type (0: f8->f16; 1: f8->bf16;\n");
        printf("arg3: matrix layout (0: A[m, k] * B[MFMA] = C[m, n];\n");
        printf("arg4: verification (0: no; 1: yes)\n");
        printf("arg5: initialization (0: no init; 1: integer value; 2: decimal value)\n");
        printf("arg6: print tensor value (0: no; 1: yes)\n");
        printf("arg7: time kernel (0=no, 1=yes)\n");
        printf("arg8 to 15: M, N, K, StrideA, StrideB, StrideD0, StrideD1, StrideE\n");
        printf("optional:\n");
        printf("arg16: number of kbatch (default 1)\n");
        printf("arg17: number of warm-up cycles (default 1)\n");
        printf("arg18: number of iterations (default 10)\n");
        printf("arg19: memory for rotating buffer (default 0, size in MB)\n");
        exit(1);
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

    const int StrideA  = std::stoi(argv[11]);
    const int StrideB  = std::stoi(argv[12]);
    const int StrideD0 = std::stoi(argv[13]);
    const int StrideD1 = std::stoi(argv[14]);
    const int StrideE  = std::stoi(argv[15]);

    int n_warmup      = 1;
    int n_iter        = 10;
    uint64_t rotating = 0;
    int KBatch        = 1;
    if(argc == 20)
    {
        KBatch   = std::stoi(argv[16]);
        n_warmup = std::stoi(argv[17]);
        n_iter   = std::stoi(argv[18]);
        rotating = std::stoull(argv[19]) * 1024 * 1024;
    }

    using F32  = float;
    using BF16 = ck::bhalf_t;
    using F16  = ck::half_t;
    using F8   = ck::f8_t;
    // using I8   = int8_t;
    // using I32  = int;

    using Row = ck::tensor_layout::gemm::RowMajor;
    using Col = ck::tensor_layout::gemm::ColumnMajor;

    auto profile = [&](auto a_type,
                       auto b_type,
                       auto comp_type,
                       auto acc_type,
                       auto d0_type,
                       auto d1_type,
                       auto c_type,
                       auto a_layout,
                       auto b_layout,
                       auto d0_layout,
                       auto d1_layout,
                       auto e_layout) {
        using ADataType       = decltype(a_type);
        using BDataType       = decltype(b_type);
        using ComputeDataType = decltype(comp_type);
        using D0DataType      = decltype(d0_type);
        using D1DataType      = decltype(d1_type);
        using AccDataType     = decltype(acc_type);
        using EDataType       = decltype(c_type);

        using ALayout  = decltype(a_layout);
        using BLayout  = decltype(b_layout);
        using D0Layout = decltype(d0_layout);
        using D1Layout = decltype(d1_layout);
        using ELayout  = decltype(e_layout);

        const int DefaultStrideA  = ck::is_same_v<ALayout, Row> ? K : M;
        const int DefaultStrideB  = ck::is_same_v<BLayout, Row> ? N : K;
        const int DefaultStrideD0 = ck::is_same_v<D0Layout, Row> ? N : M;
        const int DefaultStrideD1 = ck::is_same_v<D1Layout, Row> ? N : M;
        const int DefaultStrideE  = ck::is_same_v<ELayout, Row> ? N : M;

        bool pass =
            ck::profiler::profile_gemm_multiply_multiply_weight_preshuffle_impl<ADataType,
                                                                                BDataType,
                                                                                ComputeDataType,
                                                                                AccDataType,
                                                                                D0DataType,
                                                                                D1DataType,
                                                                                EDataType,
                                                                                ALayout,
                                                                                BLayout,
                                                                                D0Layout,
                                                                                D1Layout,
                                                                                ELayout>(
                do_verification,
                init_method,
                do_log,
                time_kernel,
                M,
                N,
                K,
                (StrideA < 0) ? DefaultStrideA : StrideA,
                (StrideB < 0) ? DefaultStrideB : StrideB,
                (StrideD0 < 0) ? DefaultStrideD0 : StrideD0,
                (StrideD1 < 0) ? DefaultStrideD1 : StrideD1,
                (StrideE < 0) ? DefaultStrideE : StrideE,
                KBatch,
                n_warmup,
                n_iter,
                rotating);

        return pass ? 0 : 1;
    };
    if(data_type == GemmDataType::F8_F8_F16 && layout == GemmMatrixLayout::MK_MFMA_MN)
    {
        return profile(
            F8{}, F8{}, F8{}, F32{}, F32{}, F32{}, F16{}, Row{}, Col{}, Row{}, Col{}, Row{});
    }
    else if(data_type == GemmDataType::F8_F8_BF16 && layout == GemmMatrixLayout::MK_MFMA_MN)
    {
        return profile(
            F8{}, F8{}, F8{}, F32{}, F32{}, F32{}, BF16{}, Row{}, Col{}, Row{}, Col{}, Row{});
    }
    else
    {
        std::cout << "this data_type & layout is not implemented" << std::endl;

        return 1;
    }
}

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_gemm_multiply_multiply_weight_preshuffle);
