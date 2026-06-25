// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <cstdio>

#include "profiler/profile_gemm_quantization_impl.hpp"
#include "profiler_operation_registry.hpp"

#define OP_NAME "gemm_quantization"
#define OP_DESC "GEMM Quantization"

using INT8  = int8_t;
using INT32 = int32_t;

int profile_gemm_quantization(int argc, char* argv[])
{
    enum struct MatrixLayout
    {
        MK_KN_MN, // 0:
        MK_NK_MN, // 1:
        KM_KN_MN, // 2:
        KM_NK_MN, // 3:
    };

    if(argc != 14)
    {
        // clang-format off
        printf("arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n");
        printf("arg2: matrix layout (0: E[m, n] = A[m, k] * B[k, n];\n");
        printf("                     1: E[m, n] = A[m, k] * B[n, k];\n");
        printf("                     2: E[m, n] = A[k, m] * B[k, n];\n");
        printf("                     3: E[m, n] = A[k, m] * B[n, k])\n");
        printf("arg3: verification (0: no; 1: yes)\n");
        printf("arg4: initialization (0: no init; default: integer value)\n");
        printf("arg5: print tensor value (0: no; 1: yes)\n");
        printf("arg6: time kernel (0=no, 1=yes)\n");
        printf("arg7 to 12: M, N, K, StrideA, StrideB, StrideE\n");
        printf("arg13: requant_scale (float, e.g., 0.03)\n");
        // clang-format on
        exit(1);
    }

    const auto layout          = static_cast<MatrixLayout>(std::stoi(argv[2]));
    const bool do_verification = std::stoi(argv[3]);
    const int init_method      = std::stoi(argv[4]);
    const bool do_log          = std::stoi(argv[5]);
    const bool time_kernel     = std::stoi(argv[6]);

    const int M = std::stoi(argv[7]);
    const int N = std::stoi(argv[8]);
    const int K = std::stoi(argv[9]);

    const int StrideA = std::stoi(argv[10]);
    const int StrideB = std::stoi(argv[11]);
    const int StrideE = std::stoi(argv[12]);

    const float requant_scale = std::stof(argv[13]);

    using Row = ck::tensor_layout::gemm::RowMajor;
    using Col = ck::tensor_layout::gemm::ColumnMajor;

    auto profile = [&](auto a_layout, auto b_layout, auto e_layout) {
        using ALayout = decltype(a_layout);
        using BLayout = decltype(b_layout);
        using ELayout = decltype(e_layout);

        bool pass = ck::profiler::profile_gemm_quantization_impl<int8_t,
                                                                 int8_t,
                                                                 int32_t,
                                                                 int8_t,
                                                                 ALayout,
                                                                 BLayout,
                                                                 ELayout>(do_verification,
                                                                          init_method,
                                                                          do_log,
                                                                          time_kernel,
                                                                          M,
                                                                          N,
                                                                          K,
                                                                          StrideA,
                                                                          StrideB,
                                                                          StrideE,
                                                                          requant_scale);

        return pass ? 0 : 1;
    };

    if(layout == MatrixLayout::MK_KN_MN)
    {
        return profile(Row{}, Row{}, Row{});
    }
    else if(layout == MatrixLayout::MK_NK_MN)
    {
        return profile(Row{}, Col{}, Row{});
    }
    else if(layout == MatrixLayout::KM_KN_MN)
    {
        return profile(Col{}, Row{}, Row{});
    }
    else if(layout == MatrixLayout::KM_NK_MN)
    {
        return profile(Col{}, Col{}, Row{});
    }
    else
    {
        std::cout << "this layout is not implemented" << std::endl;
        return 1;
    }
}

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_gemm_quantization);
