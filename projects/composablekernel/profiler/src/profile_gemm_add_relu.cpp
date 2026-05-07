// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "profiler/profile_gemm_add_relu_impl.hpp"
#include "profiler_operation_registry.hpp"

#define OP_NAME "gemm_add_relu"
#define OP_DESC "GEMM+Add+ReLU"

using INT8 = int8_t;
using BF16 = ck::bhalf_t;

int profile_gemm_add_relu(int argc, char* argv[])
{
    enum struct MatrixLayout
    {
        MK_KN_MN_MN, // 0
        MK_NK_MN_MN, // 1
        KM_KN_MN_MN, // 2
        KM_NK_MN_MN, // 3
    };

    enum struct MatrixDataType
    {
        F16_INT8_F16_F16,    // 0
        BF16_INT8_BF16_BF16, // 1
    };

    if(argc != 15)
    {
        // clang-format off
        printf("arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n");
        printf("arg2: data type (0: f16&i8 1: bf16&i8)\n");
        printf("arg3: matrix layout (0: E[m, n] = ReLU(A[m, k] * B[k, n] + D0[m, n]);\n");
        printf("                     1: E[m, n] = ReLU(A[m, k] * B[n, k] + D0[m, n]);\n");
        printf("                     2: E[m, n] = ReLU(A[k, m] * B[k, n] + D0[m, n]);\n");
        printf("                     3: E[m, n] = ReLU(A[k, m] * B[n, k] + D0[m, n]))\n");
        printf("arg4: verification (0: no; 1: yes)\n");
        printf("arg5: initialization (0: no init; 1: integer value; 2: decimal value)\n");
        printf("arg6: print tensor value (0: no; 1: yes)\n");
        printf("arg7: time kernel (0=no, 1=yes)\n");
        printf("arg8 to 14: M, N, K, StrideA, StrideB, StrideD0, StrideE\n");
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
    const int StrideE  = std::stoi(argv[14]);

    using F16 = ck::half_t;
    using F32 = float;

    using Row = ck::tensor_layout::gemm::RowMajor;
    // using Col = ck::tensor_layout::gemm::ColumnMajor;

    auto profile = [&](auto a_type,
                       auto b_type,
                       auto acc_type,
                       auto d0_type,
                       auto e_type,
                       auto a_layout,
                       auto b_layout,
                       auto d0_layout,
                       auto e_layout) {
        using ADataType   = decltype(a_type);
        using BDataType   = decltype(b_type);
        using AccDataType = decltype(acc_type);
        using D0DataType  = decltype(d0_type);
        using EDataType   = decltype(e_type);

        using ALayout  = decltype(a_layout);
        using BLayout  = decltype(b_layout);
        using D0Layout = decltype(d0_layout);
        using ELayout  = decltype(e_layout);

        const int DefaultStrideA  = ck::is_same_v<ALayout, Row> ? K : M;
        const int DefaultStrideB  = ck::is_same_v<BLayout, Row> ? N : K;
        const int DefaultStrideD0 = ck::is_same_v<D0Layout, Row> ? N : M;
        const int DefaultStrideE  = ck::is_same_v<ELayout, Row> ? N : M;

        bool pass = ck::profiler::profile_gemm_add_relu_impl<ADataType,
                                                             BDataType,
                                                             AccDataType,
                                                             D0DataType,
                                                             EDataType,
                                                             ALayout,
                                                             BLayout,
                                                             D0Layout,
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
            (StrideE < 0) ? DefaultStrideE : StrideE);

        return pass ? 0 : 1;
    };

    if(data_type == MatrixDataType::F16_INT8_F16_F16 && layout == MatrixLayout::MK_KN_MN_MN)
    {
        return profile(F16{}, INT8{}, F32{}, F16{}, F16{}, Row{}, Row{}, Row{}, Row{});
    }
    else if(data_type == MatrixDataType::BF16_INT8_BF16_BF16 && layout == MatrixLayout::MK_KN_MN_MN)
    {
        return profile(BF16{}, INT8{}, F32{}, BF16{}, BF16{}, Row{}, Row{}, Row{}, Row{});
    }
    else
    {
        std::cout << "this data_type & layout is not implemented" << std::endl;

        return 1;
    }
}

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_gemm_add_relu);
