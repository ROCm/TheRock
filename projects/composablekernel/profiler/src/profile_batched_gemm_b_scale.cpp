// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <inttypes.h>

#include "profiler/profile_batched_gemm_b_scale_impl.hpp"
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
    F16_I4_F16,     // 8
};

enum struct BScaleBlockTile
{
    K_64,  // 0
    K_128, // 1
};

#define OP_NAME "batched_gemm_b_scale"
#define OP_DESC "Int4-dequant batched GEMM"

int profile_batched_gemm_b_scale(int argc, char* argv[])
{
    if(argc != 17 && argc != 20)
    {
        printf("arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n");
        printf("arg2: data type (0: fp32; 1: fp16; 2: bf16; 3: int8; 4: f8@f16; 5: f16@f8; 6: "
               "f16->f8; 7: f8->bf16, "
               "comp f8; 8: f16@i4)\n");
        printf("arg3: matrix layout (0: A[m, k] * B[k, n] = C[m, n];\n");
        printf("                     1: A[m, k] * B[n, k] = C[m, n];\n");
        printf("                     2: A[k, m] * B[k, n] = C[m, n];\n");
        printf("                     3: A[k, m] * B[n, k] = C[m, n])\n");
        printf("arg4: B scale block tile (0: 64, 1: 128):\n");
        printf("arg5: verification (0: no; 1: yes)\n");
        printf("arg6: initialization (0: no init; 1: integer value; 2: decimal value)\n");
        printf("arg7: print tensor value (0: no; 1: yes)\n");
        printf("arg8: time kernel (0=no, 1=yes)\n");
        printf("arg9 to 15: M, N, K, StrideA, StrideB, StrideC, BatchCount\n");
        printf("arg16: split k into mulitiple batch\n");
        printf("optional:\n");
        printf("arg17: number of warm-up cycles (default 1)\n");
        printf("arg18: number of iterations (default 10)\n");
        printf("arg19: memory for rotating buffer (default 0, size in MB)\n");
        exit(1);
    }

    printf("Start profiling\n");
    const auto data_type       = static_cast<GemmDataType>(std::stoi(argv[2]));
    const auto layout          = static_cast<GemmMatrixLayout>(std::stoi(argv[3]));
    const auto B_scale_block   = static_cast<BScaleBlockTile>(std::stoi(argv[4]));
    const bool do_verification = std::stoi(argv[5]);
    const int init_method      = std::stoi(argv[6]);
    const bool do_log          = std::stoi(argv[7]);
    const bool time_kernel     = std::stoi(argv[8]);

    const int M = std::stoi(argv[9]);
    const int N = std::stoi(argv[10]);
    const int K = std::stoi(argv[11]);

    const int StrideA = std::stoi(argv[12]);
    const int StrideB = std::stoi(argv[13]);
    const int StrideC = std::stoi(argv[14]);

    const int BatchStrideA = M * N;
    const int BatchStrideB = N * K;
    const int BatchStrideC = M * N;
    const int BatchStrideScaleB =
        (K + static_cast<int>(B_scale_block) - 1) / static_cast<int>(B_scale_block) * N;
    const int BatchSize = std::stoi(argv[15]);
    const int KBatch    = std::stoi(argv[16]);

    printf("M:%d, N:%d, K:%d, StrideA:%d, StrideB:%d, StrideC:%d, BatchStrideA:%d, "
           "BatchStrideB:%d, BatchStrideC:%d, BatchStrideScaleB:%d, BatchSize:%d, KBatch:%d,\n",
           M,
           N,
           K,
           StrideA,
           StrideB,
           StrideC,
           BatchStrideA,
           BatchStrideB,
           BatchStrideC,
           BatchStrideScaleB,
           BatchSize,
           KBatch);

    int n_warmup      = 1;
    int n_iter        = 10;
    uint64_t rotating = 0;
    if(argc == 20)
    {
        n_warmup = std::stoi(argv[17]);
        n_iter   = std::stoi(argv[18]);
        rotating = std::stoull(argv[19]) * 1024 * 1024;

        printf("n_warmup:%d, n_iter:%d, rotating:%" PRIu64 "\n", n_warmup, n_iter, rotating);
    }

    using F32 = float;
    using F16 = ck::half_t;
    using I4  = ck::pk_i4_t;

    using Row = ck::tensor_layout::gemm::RowMajor;
    using Col = ck::tensor_layout::gemm::ColumnMajor;

    auto profile = [&](auto a_type,
                       auto b_type,
                       auto b_scale_type,
                       auto comp_type,
                       auto acc_type,
                       auto c_type,
                       auto scale_block_k,
                       auto a_layout,
                       auto b_layout,
                       auto c_layout) {
        using ADataType       = decltype(a_type);
        using BDataType       = decltype(b_type);
        using BScaleDataType  = decltype(b_scale_type);
        using ComputeDataType = decltype(comp_type);
        using AccDataType     = decltype(acc_type);
        using CDataType       = decltype(c_type);

        using ALayout = decltype(a_layout);
        using BLayout = decltype(b_layout);
        using CLayout = decltype(c_layout);

        const int DefaultStrideA = ck::is_same_v<ALayout, Row> ? K : M;
        const int DefaultStrideB = ck::is_same_v<BLayout, Row> ? N : K;
        const int DefaultStrideC = ck::is_same_v<CLayout, Row> ? N : M;

        bool pass = ck::profiler::profile_batched_gemm_b_scale_impl<ADataType,
                                                                    BDataType,
                                                                    BScaleDataType,
                                                                    ComputeDataType,
                                                                    AccDataType,
                                                                    CDataType,
                                                                    scale_block_k,
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
            BatchStrideA,
            BatchStrideB,
            BatchStrideC,
            BatchStrideScaleB,
            BatchSize,
            KBatch,
            n_warmup,
            n_iter,
            rotating);

        return pass ? 0 : 1;
    };

    if(data_type == GemmDataType::F16_I4_F16 && layout == GemmMatrixLayout::MK_NK_MN &&
       B_scale_block == BScaleBlockTile::K_128)
    {
        printf("F16_I4_F16 MK_NK_MN K_128\n");
        return profile(
            F16{}, I4{}, F16{}, F16{}, F32{}, F16{}, ck::Number<128>{}, Row{}, Col{}, Row{});
    }
    else
    {
        std::cout << "this data_type & layout is not implemented" << std::endl;

        return 1;
    }
}

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_batched_gemm_b_scale);
