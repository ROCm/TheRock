// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "profiler/profile_gemm_ab_scale_impl.hpp"
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

enum struct ScaleBlockTile
{
    Tile_128_128_128, // 0
    Tile_1_128_128,   // 1
};

#define OP_NAME "gemm_ab_scale"
#define OP_DESC "GEMM_AB_Scale"

int profile_gemm_ab_scale(int argc, char* argv[])
{
    if(argc != 15 && argc != 16 && argc != 19)
    {
        printf("arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n");
        printf("arg2: data type (0: fp32; 1: fp16; 2: bf16; 3: int8; 4: f8@f16; 5: f16@f8; 6: "
               "f16->f8; 7: f8->bf16, "
               "comp f8)\n");
        printf("arg3: matrix layout (0: A[m, k] * B[k, n] = C[m, n];\n");
        printf("                     1: A[m, k] * B[n, k] = C[m, n];\n");
        printf("                     2: A[k, m] * B[k, n] = C[m, n];\n");
        printf("                     3: A[k, m] * B[n, k] = C[m, n])\n");
        printf("arg4: scale block tile (0: ScaleBlockM/N/K = [128, 128, 128]; 1: ScaleBlockM/N/K = "
               "[1, 128, 128];\n");
        printf("arg5: verification (0: no; 1: yes)\n");
        printf("arg6: initialization (0: no init; 1: integer value; 2: decimal value)\n");
        printf("arg7: print tensor value (0: no; 1: yes)\n");
        printf("arg8: time kernel (0=no, 1=yes)\n");
        printf("arg9 to 14: M, N, K, StrideA, StrideB, StrideE\n");
        printf("arg15: KBatch (default: 1)\n");
        printf("optional:\n");
        printf("arg16: number of warm-up cycles (default 1)\n");
        printf("arg17: number of iterations (default 10)\n");
        printf("arg18: memory for rotating buffer (default 0, size in MB)\n");
        exit(1);
    }

    const auto data_type        = static_cast<GemmDataType>(std::stoi(argv[2]));
    const auto layout           = static_cast<GemmMatrixLayout>(std::stoi(argv[3]));
    const auto scale_block_tile = static_cast<ScaleBlockTile>(std::stoi(argv[4]));
    const bool do_verification  = std::stoi(argv[5]);
    const int init_method       = std::stoi(argv[6]);
    const bool do_log           = std::stoi(argv[7]);
    const bool time_kernel      = std::stoi(argv[8]);

    const int M = std::stoi(argv[9]);
    const int N = std::stoi(argv[10]);
    const int K = std::stoi(argv[11]);

    const int StrideA = std::stoi(argv[12]);
    const int StrideB = std::stoi(argv[13]);
    const int StrideE = std::stoi(argv[14]);
    const int KBatch  = (argc > 15) ? std::stoi(argv[15]) : 1;

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
    using BF16 = ck::bhalf_t;
    using F8   = ck::f8_t;

    using Row = ck::tensor_layout::gemm::RowMajor;
    using Col = ck::tensor_layout::gemm::ColumnMajor;

    auto profile = [&](auto a0_type,
                       auto a1_type,
                       auto b0_type,
                       auto b1_type,
                       auto comp_type,
                       auto acc_type,
                       auto c_type,
                       auto scale_block_m,
                       auto scale_block_n,
                       auto scale_block_k,
                       auto a_layout,
                       auto b_layout,
                       auto e_layout) {
        using A0DataType      = decltype(a0_type);
        using A1DataType      = decltype(a1_type);
        using B0DataType      = decltype(b0_type);
        using B1DataType      = decltype(b1_type);
        using ComputeDataType = decltype(comp_type);
        using AccDataType     = decltype(acc_type);
        using EDataType       = decltype(c_type);

        using ALayout = decltype(a_layout);
        using BLayout = decltype(b_layout);
        using ELayout = decltype(e_layout);

        const int DefaultStrideA = ck::is_same_v<ALayout, Row> ? K : M;
        const int DefaultStrideB = ck::is_same_v<BLayout, Row> ? N : K;
        const int DefaultStrideE = ck::is_same_v<ELayout, Row> ? N : M;

        bool pass = ck::profiler::profile_gemm_ab_scale_impl<A0DataType,
                                                             A1DataType,
                                                             B0DataType,
                                                             B1DataType,
                                                             ComputeDataType,
                                                             AccDataType,
                                                             EDataType,
                                                             scale_block_m,
                                                             scale_block_n,
                                                             scale_block_k,
                                                             ALayout,
                                                             BLayout,
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
            (StrideE < 0) ? DefaultStrideE : StrideE,
            KBatch,
            n_warmup,
            n_iter,
            rotating);

        return pass ? 0 : 1;
    };

    if(data_type == GemmDataType::F8_F8_BF16 && layout == GemmMatrixLayout::MK_NK_MN &&
       scale_block_tile == ScaleBlockTile::Tile_1_128_128)
    {
        return profile(F8{},
                       F32{},
                       F8{},
                       F32{},
                       F8{},
                       F32{},
                       BF16{},
                       ck::Number<1>{},
                       ck::Number<128>{},
                       ck::Number<128>{},
                       Row{},
                       Col{},
                       Row{});
    }
    else if(data_type == GemmDataType::F8_F8_BF16 && layout == GemmMatrixLayout::MK_KN_MN &&
            scale_block_tile == ScaleBlockTile::Tile_1_128_128)
    {
        return profile(F8{},
                       F32{},
                       F8{},
                       F32{},
                       F8{},
                       F32{},
                       BF16{},
                       ck::Number<1>{},
                       ck::Number<128>{},
                       ck::Number<128>{},
                       Row{},
                       Row{},
                       Row{});
    }
    else if(data_type == GemmDataType::F8_F8_BF16 && layout == GemmMatrixLayout::KM_KN_MN &&
            scale_block_tile == ScaleBlockTile::Tile_1_128_128)
    {
        return profile(F8{},
                       F32{},
                       F8{},
                       F32{},
                       F8{},
                       F32{},
                       BF16{},
                       ck::Number<1>{},
                       ck::Number<128>{},
                       ck::Number<128>{},
                       Col{},
                       Row{},
                       Row{});
    }
    else
    {
        std::cout << "this data_type & layout is not implemented" << std::endl;

        return 1;
    }
}

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_gemm_ab_scale);
