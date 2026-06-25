// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "profiler/profile_gemm_universal_streamk_impl.hpp"
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
    F8_F16_F16,     // 4
    F16_F8_F16,     // 5
    F16_F16_F16_F8, // 6
    F8_F8_BF16,     // 7
};

#define OP_NAME "gemm_universal_streamk"
#define OP_DESC "Universal Streamk GEMM"

int profile_gemm_universal_streamk(int argc, char* argv[])
{
    if(argc != 16 && argc != 19)
    {
        printf("arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n");
        printf("arg2: data type (0: fp32; 1: fp16; 2: bf16; 3: int8; 4: f8@f16; 5: f16@f8; 6: f16, "
               "comp f8;  7: f8->bf16,)\n");
        printf("arg3: matrix layout (0: A[m, k] * B[k, n] = C[m, n];\n");
        printf("                     1: A[m, k] * B[n, k] = C[m, n];\n");
        printf("                     2: A[k, m] * B[k, n] = C[m, n];\n");
        printf("                     3: A[k, m] * B[n, k] = C[m, n])\n");
        printf("arg4: verification (0: no; 1: yes)\n");
        printf("arg5: initialization (0: no init; 1: integer value; 2: decimal value)\n");
        printf("arg6: print tensor value (0: no; 1: yes)\n");
        printf("arg7: time kernel (0=no, 1=yes)\n");
        printf("arg8 to 13: M, N, K, StrideA, StrideB, StrideC\n");
        printf("arg14: Stream-k select strategy 0: all DP, 1: 1-tile SK, 2: 2-tile SK\n");
        printf("arg15: Grid-size, -1 for max persistent kernel occupancy\n");
        printf("optional:\n");
        printf("arg16: number of warm-up cycles (default 1)\n");
        printf("arg17: number of iterations (default 10)\n");
        printf("arg18: memory for rotating buffer (default 0, size in MB)\n");
        exit(1);
    }

    int M;
    int N;
    int StrideA;
    int StrideB;
    // Analyze the unsupported matrix shapes, switch the M and N number
    if(std::stoi(argv[9]) % 8 != 0 && std::stoi(argv[8]) % 8 == 0)
    {
        M       = std::stoi(argv[9]);
        StrideA = std::stoi(argv[12]);
        N       = std::stoi(argv[8]);
        StrideB = std::stoi(argv[11]);
    }
    else
    {
        M       = std::stoi(argv[8]);
        StrideA = std::stoi(argv[11]);
        N       = std::stoi(argv[9]);
        StrideB = std::stoi(argv[12]);
    }

    const auto data_type       = static_cast<GemmDataType>(std::stoi(argv[2]));
    const auto layout          = static_cast<GemmMatrixLayout>(std::stoi(argv[3]));
    const bool do_verification = std::stoi(argv[4]);
    const int init_method      = std::stoi(argv[5]);
    const bool do_log          = std::stoi(argv[6]);
    const bool time_kernel     = std::stoi(argv[7]);

    const int K = std::stoi(argv[10]);

    const int StrideC     = std::stoi(argv[13]);
    const int Streamk_sel = std::stoi(argv[14]);
    const int Grid_size   = std::stoi(argv[15]);

    int n_warmup      = 1;
    int n_iter        = 10;
    uint64_t rotating = 0;
    if(argc == 19)
    {
        n_warmup = std::stoi(argv[16]);
        n_iter   = std::stoi(argv[17]);
        rotating = std::stoull(argv[18]) * 1024 * 1024;
    }

    using F32  = float;
    using F16  = ck::half_t;
    using BF16 = ck::bhalf_t;

#if defined(CK_USE_FP8_ON_UNSUPPORTED_ARCH) || defined(CK_USE_GFX94)
    using F8 = ck::f8_t;
#endif

    using Row = ck::tensor_layout::gemm::RowMajor;
    using Col = ck::tensor_layout::gemm::ColumnMajor;

    auto profile = [&](auto a_type,
                       auto b_type,
                       auto comp_type,
                       auto acc_type,
                       auto c_type,
                       auto a_layout,
                       auto b_layout,
                       auto c_layout) {
        using ADataType       = decltype(a_type);
        using BDataType       = decltype(b_type);
        using ComputeDataType = decltype(comp_type);
        using AccDataType     = decltype(acc_type);
        using CDataType       = decltype(c_type);

        using ALayout = decltype(a_layout);
        using BLayout = decltype(b_layout);
        using CLayout = decltype(c_layout);

        const int DefaultStrideA = ck::is_same_v<ALayout, Row> ? K : M;
        const int DefaultStrideB = ck::is_same_v<BLayout, Row> ? N : K;
        const int DefaultStrideC = ck::is_same_v<CLayout, Row> ? N : M;

        bool pass = ck::profiler::profile_gemm_universal_streamk_impl<ADataType,
                                                                      BDataType,
                                                                      ComputeDataType,
                                                                      AccDataType,
                                                                      CDataType,
                                                                      ALayout,
                                                                      BLayout,
                                                                      CLayout>(
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
            Streamk_sel,
            Grid_size,
            n_warmup,
            n_iter,
            rotating);

        return pass ? 0 : 1;
    };

    if(data_type == GemmDataType::F16_F16_F16 && layout == GemmMatrixLayout::MK_KN_MN)
    {
        return profile(F16{}, F16{}, F32{}, F32{}, F16{}, Row{}, Row{}, Row{});
    }
    else if(data_type == GemmDataType::F16_F16_F16 && layout == GemmMatrixLayout::MK_NK_MN)
    {
        return profile(F16{}, F16{}, F32{}, F32{}, F16{}, Row{}, Col{}, Row{});
    }
#if defined(CK_USE_FP8_ON_UNSUPPORTED_ARCH) || defined(CK_USE_GFX94)
    else if(data_type == GemmDataType::F16_F8_F16 && layout == GemmMatrixLayout::MK_KN_MN)
    {
        return profile(F16{}, F8{}, F32{}, F32{}, F16{}, Row{}, Row{}, Row{});
    }
    else if(data_type == GemmDataType::F16_F8_F16 && layout == GemmMatrixLayout::MK_NK_MN)
    {
        return profile(F16{}, F8{}, F32{}, F32{}, F16{}, Row{}, Col{}, Row{});
    }
    else if(data_type == GemmDataType::F8_F16_F16 && layout == GemmMatrixLayout::MK_KN_MN)
    {
        return profile(F8{}, F16{}, F32{}, F32{}, F16{}, Row{}, Row{}, Row{});
    }
    else if(data_type == GemmDataType::F8_F16_F16 && layout == GemmMatrixLayout::MK_NK_MN)
    {
        return profile(F8{}, F16{}, F32{}, F32{}, F16{}, Row{}, Col{}, Row{});
    }
#endif
    else if(data_type == GemmDataType::BF16_BF16_BF16 && layout == GemmMatrixLayout::MK_KN_MN)
    {
        return profile(BF16{}, BF16{}, F32{}, F32{}, BF16{}, Row{}, Row{}, Row{});
    }
    else if(data_type == GemmDataType::BF16_BF16_BF16 && layout == GemmMatrixLayout::MK_NK_MN)
    {
        return profile(BF16{}, BF16{}, F32{}, F32{}, BF16{}, Row{}, Col{}, Row{});
    }
    else if(data_type == GemmDataType::BF16_BF16_BF16 && layout == GemmMatrixLayout::KM_KN_MN)
    {
        return profile(BF16{}, BF16{}, F32{}, F32{}, BF16{}, Col{}, Row{}, Row{});
    }
    else if(data_type == GemmDataType::BF16_BF16_BF16 && layout == GemmMatrixLayout::KM_NK_MN)
    {
        return profile(BF16{}, BF16{}, F32{}, F32{}, BF16{}, Col{}, Col{}, Row{});
    }
#if defined(CK_USE_FP8_ON_UNSUPPORTED_ARCH) || defined(CK_USE_GFX94)
    else if(data_type == GemmDataType::F8_F8_BF16 && layout == GemmMatrixLayout::MK_KN_MN)
    {
        return profile(F8{}, F8{}, F8{}, F32{}, BF16{}, Row{}, Row{}, Row{});
    }
    else if(data_type == GemmDataType::F8_F8_BF16 && layout == GemmMatrixLayout::MK_NK_MN)
    {
        return profile(F8{}, F8{}, F8{}, F32{}, BF16{}, Row{}, Col{}, Row{});
    }
#endif
    else
    {
        std::cout << "this data_type & layout is not implemented" << std::endl;

        return 1;
    }
}

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_gemm_universal_streamk);
