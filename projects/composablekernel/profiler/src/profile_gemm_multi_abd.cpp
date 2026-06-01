// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "profiler/profile_gemm_multi_abd_impl.hpp"
#include "profiler_operation_registry.hpp"

enum struct GemmMatrixLayout
{
    MK_KN_MN, // 0
    MK_NK_MN, // 1
    KM_KN_MN, // 2
    KM_NK_MN, // 3
};

enum struct GemmDataType
{
    BF16_I8_BF16_BF16, // 0
};

enum struct GemmElementOp
{
    PASS_THROUGH,          // 0
    MULTIPLY,              // 1
    ADD,                   // 2
    FASTGELU,              // 3
    ADD_FASTGELU,          // 4
    MULTIPLY_ADD,          // 5
    MULTIPLY_FASTGELU,     // 6
    MULTIPLY_ADD_FASTGELU, // 7
};

#define OP_NAME "gemm_multi_abd"
#define OP_DESC "GEMM_Multiple_ABD"

int profile_gemm_multi_abd(int argc, char* argv[])
{
    if(argc != 18)
    {
        // clang-format off
        printf("arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n");
        printf("arg2: data type (0: bf16@int8/bf16->bf16;)\n");
        printf("arg3: matrix layout (0: E[m, n] = A[m, k] * B[k, n];\n");
        printf("                     1: E[m, n] = A[m, k] * B[n, k];\n");
        printf("                     2: E[m, n] = A[k, m] * B[k, n];\n");
        printf("                     3: E[m, n] = A[k, m] * B[n, k])\n");
        printf("arg4: verification (0: no; 1: yes)\n");
        printf("arg5: initialization (0: no init; 1: integer value; 2: decimal value)\n");
        printf("arg6: print tensor value (0: no; 1: yes)\n");
        printf("arg7: time kernel (0=no, 1=yes)\n");
        printf("arg8: number of As (1)\n");
        printf("arg9: number of Bs (1/2)\n");
        printf("arg10: number of Ds (0/1/2)\n");
        printf("arg11 to 17: M, N, K, StrideA, StrideB, StrideE, StrideD\n");
        // clang-format on
        exit(1);
    }

    const auto data_type       = static_cast<GemmDataType>(std::stoi(argv[2]));
    const auto layout          = static_cast<GemmMatrixLayout>(std::stoi(argv[3]));
    const bool do_verification = std::stoi(argv[4]);
    const int init_method      = std::stoi(argv[5]);
    const bool do_log          = std::stoi(argv[6]);
    const bool time_kernel     = std::stoi(argv[7]);

    const int num_as = std::stoi(argv[8]);
    const int num_bs = std::stoi(argv[9]);
    const int num_ds = std::stoi(argv[10]);

    const int M = std::stoi(argv[11]);
    const int N = std::stoi(argv[12]);
    const int K = std::stoi(argv[13]);

    const int StrideA = std::stoi(argv[14]);
    const int StrideB = std::stoi(argv[15]);
    const int StrideE = std::stoi(argv[16]);
    const int StrideD = std::stoi(argv[17]);

    using F32  = float;
    using BF16 = ck::bhalf_t;
    using I8   = int8_t;

    using Row = ck::tensor_layout::gemm::RowMajor;
    using Col = ck::tensor_layout::gemm::ColumnMajor;

    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
    using Multiply    = ck::tensor_operation::element_wise::Multiply;
    using FastGelu    = ck::tensor_operation::element_wise::FastGelu;
    using AddFastGelu = ck::tensor_operation::element_wise::AddFastGelu;

    auto profile = [&](auto b_layout, auto b_element_op, auto cde_element_op, auto num_d_tensor) {
        using ADataType  = BF16;
        using B0DataType = I8;
        using B1DataType = BF16;
        using DDataType  = BF16;
        using EDataType  = BF16;

        using ALayout = Row;
        using BLayout = decltype(b_layout);
        using DLayout = Row;
        using ELayout = Row;

        using AElementOp         = PassThrough;
        using BElementOp         = decltype(b_element_op);
        using CDEElementOp       = decltype(cde_element_op);
        const int DefaultStrideA = ck::is_same_v<ALayout, Row> ? K : M;
        const int DefaultStrideB = ck::is_same_v<BLayout, Row> ? N : K;
        const int DefaultStrideD = ck::is_same_v<DLayout, Row> ? N : M;
        const int DefaultStrideE = ck::is_same_v<ELayout, Row> ? N : M;

        constexpr auto NumberDTensor = decltype(num_d_tensor){};

        // Only num_d_tensor == 0 and 1 are supported
        using DsDataType = typename std::
            conditional<(NumberDTensor == 0), ck::Tuple<>, ck::Tuple<DDataType>>::type;
        using DsLayout =
            typename std::conditional<(NumberDTensor == 0), ck::Tuple<>, ck::Tuple<DLayout>>::type;

        bool pass = ck::profiler::profile_gemm_multi_abd_impl<ck::Tuple<ADataType>,
                                                              ck::Tuple<B0DataType, B1DataType>,
                                                              F32,
                                                              DsDataType,
                                                              EDataType,
                                                              ck::Tuple<ALayout>,
                                                              ck::Tuple<BLayout, BLayout>,
                                                              DsLayout,
                                                              ELayout,
                                                              AElementOp,
                                                              BElementOp,
                                                              CDEElementOp>(
            do_verification,
            init_method,
            do_log,
            time_kernel,
            M,
            N,
            K,
            (StrideA < 0) ? DefaultStrideA : StrideA,
            (StrideB < 0) ? DefaultStrideB : StrideB,
            (StrideD < 0) ? DefaultStrideD : StrideD,
            (StrideE < 0) ? DefaultStrideE : StrideE);

        return pass ? 0 : 1;
    };

    // num_as == 1 is only supported
    if(data_type != GemmDataType::BF16_I8_BF16_BF16 || num_as != 1)
    {
        std::cout << "The provided input parameters are not supported" << std::endl;
        return 1;
    }

    // Supported configurations
    if(layout == GemmMatrixLayout::MK_KN_MN && num_bs == 2 && num_ds == 1)
    {
        return profile(Row{}, Multiply{}, AddFastGelu{}, ck::Number<1>{});
    }
    else if(layout == GemmMatrixLayout::MK_KN_MN && num_bs == 2 && num_ds == 0)
    {
        return profile(Row{}, Multiply{}, FastGelu{}, ck::Number<0>{});
    }
    else if(layout == GemmMatrixLayout::MK_NK_MN && num_bs == 2 && num_ds == 1)
    {
        return profile(Col{}, Multiply{}, AddFastGelu{}, ck::Number<1>{});
    }
    else if(layout == GemmMatrixLayout::MK_NK_MN && num_bs == 2 && num_ds == 0)
    {
        return profile(Col{}, Multiply{}, FastGelu{}, ck::Number<0>{});
    }

    std::cout << "The provided input parameters are not supported" << std::endl;

    return 1;
}

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_gemm_multi_abd);
