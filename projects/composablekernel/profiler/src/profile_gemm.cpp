// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "profiler/profile_gemm_impl.hpp"
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
    F32_F32_F32,    // 0
    F16_F16_F16,    // 1
    BF16_BF16_BF16, // 2
    INT8_INT8_INT8, // 3
    F8_F8_F8,       // 4
};

#define OP_NAME "gemm"
#define OP_DESC "GEMM"

static void print_helper_msg()
{
    std::cout << "arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n"
              << "arg2: data type (0: fp32; 1: fp16; 2: bf16; 3: int8; 4: fp8)\n"
              << "arg3: matrix layout (0: A[m, k] * B[k, n] = C[m, n];\n"
              << "                     1: A[m, k] * B[n, k] = C[m, n];\n"
              << "                     2: A[k, m] * B[k, n] = C[m, n];\n"
              << "                     3: A[k, m] * B[n, k] = C[m, n])\n"
              << "arg4: verification (0: no; 1: yes)\n"
              << "arg5: initialization (0: no init; 1: integer value; 2: decimal value)\n"
              << "arg6: print tensor value (0: no; 1: yes)\n"
              << "arg7: time kernel (0: no, 1: yes)\n"
              << "arg8 to 13: M, N, K, StrideA, StrideB, StrideC\n"
              << "optional:\n"
              << "arg14: number of warm-up cycles (default 1)\n"
              << "arg15: number of iterations (default 10)\n"
              << std::endl;
}

int profile_gemm(int argc, char* argv[])
{
    if(argc != 14 && argc != 16)
    {
        print_helper_msg();
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

    const int StrideA = std::stoi(argv[11]);
    const int StrideB = std::stoi(argv[12]);
    const int StrideC = std::stoi(argv[13]);

    int n_warmup = 1;
    int n_iter   = 10;
    if(argc == 16)
    {
        n_warmup = std::stoi(argv[14]);
        n_iter   = std::stoi(argv[15]);
    }
    using F32 = float;
    using F16 = ck::half_t;
#ifdef CK_ENABLE_BF16
    using BF16 = ck::bhalf_t;
#endif
#ifdef CK_ENABLE_INT8
    using INT8  = int8_t;
    using INT32 = int32_t;
#endif
#ifdef CK_ENABLE_FP8
    using F8 = ck::f8_t;
#endif

    using Row = ck::tensor_layout::gemm::RowMajor;
    using Col = ck::tensor_layout::gemm::ColumnMajor;

    auto profile = [&](auto a_layout,
                       auto b_layout,
                       auto c_layout,
                       auto a_type,
                       auto b_type,
                       auto acc_type,
                       auto c_type) {
        using ALayout = decltype(a_layout);
        using BLayout = decltype(b_layout);
        using CLayout = decltype(c_layout);

        using ADataType   = decltype(a_type);
        using BDataType   = decltype(b_type);
        using AccDataType = decltype(acc_type);
        using CDataType   = decltype(c_type);

        const int DefaultStrideA = ck::is_same_v<ALayout, Row> ? K : M;
        const int DefaultStrideB = ck::is_same_v<BLayout, Row> ? N : K;
        const int DefaultStrideC = ck::is_same_v<CLayout, Row> ? N : M;

        bool pass =
            ck::profiler::profile_gemm_impl<ALayout,
                                            BLayout,
                                            CLayout,
                                            ADataType,
                                            BDataType,
                                            AccDataType,
                                            CDataType>(do_verification,
                                                       init_method,
                                                       do_log,
                                                       time_kernel,
                                                       M,
                                                       N,
                                                       K,
                                                       (StrideA < 0) ? DefaultStrideA : StrideA,
                                                       (StrideB < 0) ? DefaultStrideB : StrideB,
                                                       (StrideC < 0) ? DefaultStrideC : StrideC,
                                                       n_warmup,
                                                       n_iter);

        return pass ? 0 : 1;
    };

    if(data_type != GemmDataType::F32_F32_F32 && data_type != GemmDataType::F16_F16_F16 &&
       data_type != GemmDataType::BF16_BF16_BF16 && data_type != GemmDataType::INT8_INT8_INT8 &&
       data_type != GemmDataType::F8_F8_F8)
    {
        // dummy clause before the else clauses for different data types
        std::cout << "Gemm: this data_type is not implemented" << std::endl;
        return 1;
    }
#ifdef CK_ENABLE_FP32
    else if(data_type == GemmDataType::F32_F32_F32 && layout == GemmMatrixLayout::MK_KN_MN)
    {
        return profile(Row{}, Row{}, Row{}, F32{}, F32{}, F32{}, F32{});
    }
    else if(data_type == GemmDataType::F32_F32_F32 && layout == GemmMatrixLayout::MK_NK_MN)
    {
        return profile(Row{}, Col{}, Row{}, F32{}, F32{}, F32{}, F32{});
    }
    else if(data_type == GemmDataType::F32_F32_F32 && layout == GemmMatrixLayout::KM_KN_MN)
    {
        return profile(Col{}, Row{}, Row{}, F32{}, F32{}, F32{}, F32{});
    }
    else if(data_type == GemmDataType::F32_F32_F32 && layout == GemmMatrixLayout::KM_NK_MN)
    {
        return profile(Col{}, Col{}, Row{}, F32{}, F32{}, F32{}, F32{});
    }
#endif
#ifdef CK_ENABLE_FP16
    else if(data_type == GemmDataType::F16_F16_F16 && layout == GemmMatrixLayout::MK_KN_MN)
    {
        return profile(Row{}, Row{}, Row{}, F16{}, F16{}, F32{}, F16{});
    }
    else if(data_type == GemmDataType::F16_F16_F16 && layout == GemmMatrixLayout::MK_NK_MN)
    {
        return profile(Row{}, Col{}, Row{}, F16{}, F16{}, F32{}, F16{});
    }
    else if(data_type == GemmDataType::F16_F16_F16 && layout == GemmMatrixLayout::KM_KN_MN)
    {
        return profile(Col{}, Row{}, Row{}, F16{}, F16{}, F32{}, F16{});
    }
    else if(data_type == GemmDataType::F16_F16_F16 && layout == GemmMatrixLayout::KM_NK_MN)
    {
        return profile(Col{}, Col{}, Row{}, F16{}, F16{}, F32{}, F16{});
    }
#endif
#ifdef CK_ENABLE_BF16
    else if(data_type == GemmDataType::BF16_BF16_BF16 && layout == GemmMatrixLayout::MK_KN_MN)
    {
        return profile(Row{}, Row{}, Row{}, BF16{}, BF16{}, F32{}, BF16{});
    }
    else if(data_type == GemmDataType::BF16_BF16_BF16 && layout == GemmMatrixLayout::MK_NK_MN)
    {
        return profile(Row{}, Col{}, Row{}, BF16{}, BF16{}, F32{}, BF16{});
    }
    else if(data_type == GemmDataType::BF16_BF16_BF16 && layout == GemmMatrixLayout::KM_KN_MN)
    {
        return profile(Col{}, Row{}, Row{}, BF16{}, BF16{}, F32{}, BF16{});
    }
    else if(data_type == GemmDataType::BF16_BF16_BF16 && layout == GemmMatrixLayout::KM_NK_MN)
    {
        return profile(Col{}, Col{}, Row{}, BF16{}, BF16{}, F32{}, BF16{});
    }
#endif
#ifdef CK_ENABLE_INT8
    else if(data_type == GemmDataType::INT8_INT8_INT8 && layout == GemmMatrixLayout::MK_KN_MN)
    {
        return profile(Row{}, Row{}, Row{}, INT8{}, INT8{}, INT32{}, INT8{});
    }
    else if(data_type == GemmDataType::INT8_INT8_INT8 && layout == GemmMatrixLayout::MK_NK_MN)
    {
        return profile(Row{}, Col{}, Row{}, INT8{}, INT8{}, INT32{}, INT8{});
    }
    else if(data_type == GemmDataType::INT8_INT8_INT8 && layout == GemmMatrixLayout::KM_KN_MN)
    {
        return profile(Col{}, Row{}, Row{}, INT8{}, INT8{}, INT32{}, INT8{});
    }
    else if(data_type == GemmDataType::INT8_INT8_INT8 && layout == GemmMatrixLayout::KM_NK_MN)
    {
        return profile(Col{}, Col{}, Row{}, INT8{}, INT8{}, INT32{}, INT8{});
    }
#endif
#ifdef CK_ENABLE_FP8
    else if(data_type == GemmDataType::F8_F8_F8 && layout == GemmMatrixLayout::MK_KN_MN)
    {
        return profile(Row{}, Row{}, Row{}, F8{}, F8{}, F32{}, F8{});
    }
    else if(data_type == GemmDataType::F8_F8_F8 && layout == GemmMatrixLayout::MK_NK_MN)
    {
        return profile(Row{}, Col{}, Row{}, F8{}, F8{}, F32{}, F8{});
    }
    else if(data_type == GemmDataType::F8_F8_F8 && layout == GemmMatrixLayout::KM_KN_MN)
    {
        return profile(Col{}, Row{}, Row{}, F8{}, F8{}, F32{}, F8{});
    }
    else if(data_type == GemmDataType::F8_F8_F8 && layout == GemmMatrixLayout::KM_NK_MN)
    {
        return profile(Col{}, Col{}, Row{}, F8{}, F8{}, F32{}, F8{});
    }
#endif
    else
    {
        std::cout << "Gemm: this data_type & layout is not implemented" << std::endl;

        return 1;
    }
}

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_gemm);
