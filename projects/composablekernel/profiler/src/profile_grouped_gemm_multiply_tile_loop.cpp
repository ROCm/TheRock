// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "profiler/profile_grouped_gemm_multiply_tile_loop_impl.hpp"
#include "profiler_operation_registry.hpp"

enum struct GemmMatrixLayout
{
    MK_KN_MN, // 0
};

enum struct GemmDataType
{
    BF16_INT8_BF16_BF16, // 0
};

#define OP_NAME "grouped_gemm_multiply_tile_loop"
#define OP_DESC "Grouped GEMM Multiply Multiple D Tile Loop"

namespace {

std::vector<int> argToIntArray(char* input)
{
    std::vector<int> out;
    std::istringstream in(input);
    std::string item;

    while(std::getline(in, item, ','))
    {
        out.push_back(std::stoi(item));
    }
    return out;
}

int profile_grouped_gemm_tile_loop(int argc, char* argv[])
{
    if(argc < 14)
    {
        std::cout
            << "arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n"
            << "arg2: data type (0: bf16@int8)\n"
            << "arg3: matrix layout (0: A[m, k] * B[k, n] = C[m, n]);\n"
            << "arg4: verification (0: no; 1: yes)\n"
            << "arg5: initialization (0: no init; 1: integer value; 2: decimal value)\n"
            << "arg6: print tensor value (0: no; 1: yes)\n"
            << "arg7: time kernel (0=n0, 1=yes)\n"
            << "arg8 to 13: Ms, Ns, Ks, StrideAs, StrideBs, StrideCs (e.g., 256,256 128,128 64,64 "
               "64,64 64,64 128,128)\n"
            << "optional:\n"
            << "arg14: number of warm-up cycles (default 1)\n"
            << "arg15: number of iterations (default 10)\n"
            << std::endl;

        exit(1);
    }

    const auto data_type       = static_cast<GemmDataType>(std::stoi(argv[2]));
    const auto layout          = static_cast<GemmMatrixLayout>(std::stoi(argv[3]));
    const bool do_verification = std::stoi(argv[4]);
    const int init_method      = std::stoi(argv[5]);
    const bool do_log          = std::stoi(argv[6]);
    const bool time_kernel     = std::stoi(argv[7]);

    const auto Ms = argToIntArray(argv[8]);
    const auto Ns = argToIntArray(argv[9]);
    const auto Ks = argToIntArray(argv[10]);

    auto StrideAs = argToIntArray(argv[11]);
    auto StrideBs = argToIntArray(argv[12]);
    auto StrideCs = argToIntArray(argv[13]);

    const int DefaultStrideA = Ks[0];
    const int DefaultStrideB = Ns[0];
    const int DefaultStrideC = Ns[0];

    for(size_t i = 0; i < Ms.size(); ++i)
    {
        StrideAs[i] = StrideAs[i] == -1 ? DefaultStrideA : StrideAs[i];
        StrideBs[i] = StrideBs[i] == -1 ? DefaultStrideB : StrideBs[i];
        StrideCs[i] = StrideCs[i] == -1 ? DefaultStrideC : StrideCs[i];
    }

    std::vector<int> StrideDs(StrideCs);

    int n_warmup = 10;
    int n_iter   = 50;
    if(argc == 16)
    {
        n_warmup = std::stoi(argv[14]);
        n_iter   = std::stoi(argv[15]);
    }

    if(data_type == GemmDataType::BF16_INT8_BF16_BF16 && layout == GemmMatrixLayout::MK_KN_MN)
    {
        ck::profiler::profile_grouped_gemm_multiply_tile_loop_impl<
            ck::bhalf_t,
            int8_t,
            ck::bhalf_t,
            ck::bhalf_t,
            float,
            ck::tensor_layout::gemm::RowMajor,
            ck::tensor_layout::gemm::RowMajor,
            ck::tensor_layout::gemm::RowMajor,
            ck::tensor_layout::gemm::RowMajor>(do_verification,
                                               init_method,
                                               do_log,
                                               time_kernel,
                                               Ms,
                                               Ns,
                                               Ks,
                                               StrideAs,
                                               StrideBs,
                                               StrideDs,
                                               StrideCs,
                                               n_warmup,
                                               n_iter);
    }
    else
    {
        throw std::runtime_error("wrong! this GEMM data_type & layout is not implemented");
    }
    return 0;
}

} // anonymous namespace

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_grouped_gemm_tile_loop);
