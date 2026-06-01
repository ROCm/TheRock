// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "profiler/profile_gemm_multiply_add_impl.hpp"
#include "profiler_operation_registry.hpp"

#define OP_NAME "gemm_multiply_add"
#define OP_DESC "GEMM+MULTIPLY+ADD"

int profile_gemm_multiply_add(int argc, char* argv[])
{
    enum struct MatrixLayout
    {
        MK_KN_MN_MN_MN, // 0
        MK_NK_MN_MN_MN, // 1
    };

    enum struct MatrixDataType
    {
        F16_F16_F16_F16_F16, // 0
        F16_F8_F32_F32_F16,  // 1
    };

    if(argc != 16)
    {
        // clang-format off
        printf("arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n");
        printf("arg2: data type (0: fp16; 1: fp16Afp8B)\n");
        printf("arg3: matrix layout (0: E[m, n] = Multiply_Add((A[m, k] * B[k, n]) x D1[m, n] + D0[m, n]);\n");
        printf("                     1: E[m, n] = Multiply_Add((A[m, k] * B[n, k]) x D1[m, n] + D0[m, n]);\n");
        printf("arg4: verification (0: no; 1: yes)\n");
        printf("arg5: initialization (0: no init; 1: integer value; 2: decimal value)\n");
        printf("arg6: print tensor value (0: no; 1: yes)\n");
        printf("arg7: time kernel (0=no, 1=yes)\n");
        printf("arg8 to 15: M, N, K, StrideA, StrideB, StrideD0, StrideD1, StrideE\n");
        // clang-format on
        exit(1);
    }

    const auto data_type       = static_cast<MatrixDataType>(std::stoi(argv[2]));
    const auto layout          = static_cast<MatrixLayout>(std::stoi(argv[3]));
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

    using F16 = ck::half_t;
    using F32 = float;
#if defined CK_ENABLE_FP8
    using F8 = ck::f8_t;
#endif

    using Row = ck::tensor_layout::gemm::RowMajor;
    using Col = ck::tensor_layout::gemm::ColumnMajor;

    auto profile = [&](auto a_type,
                       auto b_type,
                       auto acc_type,
                       auto d0_type,
                       auto d1_type,
                       auto e_type,
                       auto a_layout,
                       auto b_layout,
                       auto d0_layout,
                       auto d1_layout,
                       auto e_layout) {
        using ADataType   = decltype(a_type);
        using BDataType   = decltype(b_type);
        using AccDataType = decltype(acc_type);
        using D0DataType  = decltype(d0_type);
        using D1DataType  = decltype(d1_type);
        using EDataType   = decltype(e_type);

        using ALayout  = decltype(a_layout);
        using BLayout  = decltype(b_layout);
        using D0Layout = decltype(d0_layout);
        using D1Layout = decltype(d1_layout);
        using ELayout  = decltype(e_layout);

        bool pass = ck::profiler::profile_gemm_multiply_add_impl<ADataType,
                                                                 BDataType,
                                                                 AccDataType,
                                                                 D0DataType,
                                                                 D1DataType,
                                                                 EDataType,
                                                                 ALayout,
                                                                 BLayout,
                                                                 D0Layout,
                                                                 D1Layout,
                                                                 ELayout>(do_verification,
                                                                          init_method,
                                                                          do_log,
                                                                          time_kernel,
                                                                          M,
                                                                          N,
                                                                          K,
                                                                          StrideA,
                                                                          StrideB,
                                                                          StrideD0,
                                                                          StrideD1,
                                                                          StrideE);

        return pass ? 0 : 1;
    };

    if(data_type == MatrixDataType::F16_F16_F16_F16_F16 && layout == MatrixLayout::MK_KN_MN_MN_MN)
    {
        return profile(F16{}, F16{}, F32{}, F16{}, F16{}, F16{}, Row{}, Row{}, Row{}, Row{}, Row{});
    }
    else if(data_type == MatrixDataType::F16_F16_F16_F16_F16 &&
            layout == MatrixLayout::MK_NK_MN_MN_MN)
    {
        return profile(F16{}, F16{}, F32{}, F16{}, F16{}, F16{}, Row{}, Col{}, Row{}, Row{}, Row{});
    }
#if defined CK_ENABLE_FP8
    else if(data_type == MatrixDataType::F16_F8_F32_F32_F16 &&
            layout == MatrixLayout::MK_KN_MN_MN_MN)
    {
        return profile(F16{}, F8{}, F32{}, F32{}, F32{}, F16{}, Row{}, Row{}, Row{}, Row{}, Row{});
    }
    else if(data_type == MatrixDataType::F16_F8_F32_F32_F16 &&
            layout == MatrixLayout::MK_NK_MN_MN_MN)
    {
        return profile(F16{}, F8{}, F32{}, F32{}, F32{}, F16{}, Row{}, Col{}, Row{}, Row{}, Row{});
    }
#endif
    else
    {
        std::cout << "this data_type & layout is not implemented" << std::endl;

        return 1;
    }
}

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_gemm_multiply_add);
