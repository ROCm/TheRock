// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <numeric>
#include <unordered_map>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/utility/data_type.hpp"

#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/reference_tensor_operation/gpu/reference_gemm.hpp"
#include "ck/host_utility/kernel_launch.hpp"

using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

struct ProblemSize final
{
    ck::index_t M = 3840;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    ck::index_t StrideA = -1;
    ck::index_t StrideB = -1;
    ck::index_t StrideC = -1;
};

struct ProblemSizeStreamK final
{
    ck::index_t M = 3840;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    ck::index_t StrideA = -1;
    ck::index_t StrideB = -1;
    ck::index_t StrideC = -1;

    ck::index_t NumSKBlocks = -1; // number of stream-k blocks
};
struct ProblemSizeStreamK_universal final
{
    ck::index_t M = 3840;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    ck::index_t StrideA = -1;
    ck::index_t StrideB = -1;
    ck::index_t StrideC = -1;

    ck::index_t Grid_size                           = -1; // defaults to max occupancy
    ck::index_t Streamk_sel                         = 1;  // defaults to 1-tile SK
    ck::StreamKReductionStrategy reduction_strategy = ck::StreamKReductionStrategy::Atomic;
};

struct ProblemSizeSplitK final
{
    ck::index_t M = 3840;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    ck::index_t StrideA = -1;
    ck::index_t StrideB = -1;
    ck::index_t StrideC = -1;

    ck::index_t KBatch = 1;
};

struct ExecutionConfig final
{
    // 0 - no verification, 1 - CPU, 2 - GPU, 3 - CPU + GPU
    int do_verification = 1;
    int init_method     = 2;
    bool time_kernel    = false;
};

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

template <typename ProblemType>
bool parse_cmd_args(int, char*[], ProblemType&, ExecutionConfig&)
{
    return false;
}

template <>
bool parse_cmd_args<ProblemSize>(int argc,
                                 char* argv[],
                                 ProblemSize& problem_size,
                                 ExecutionConfig& config)
{
    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 4)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);
    }
    else if(argc == 10)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);

        problem_size.M = std::stoi(argv[4]);
        problem_size.N = std::stoi(argv[5]);
        problem_size.K = std::stoi(argv[6]);

        problem_size.StrideA = std::stoi(argv[7]);
        problem_size.StrideB = std::stoi(argv[8]);
        problem_size.StrideC = std::stoi(argv[9]);
    }
    else
    {
        std::cerr
            << "arg1: verification (0=no, 1=CPU, 2=GPU, 3=CPU and GPU)" << std::endl
            << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)" << std::endl
            << "arg3: time kernel (0=no, 1=yes)" << std::endl
            << "arg4 to 9: M (256x), N(128x), K(32x), StrideA, StrideB, StrideC (default: -1 or 0)"
            << std::endl;
        return false;
    }

    return true;
}

template <>
bool parse_cmd_args<ProblemSizeStreamK_universal>(int argc,
                                                  char* argv[],
                                                  ProblemSizeStreamK_universal& problem_size,
                                                  ExecutionConfig& config)
{
    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 4)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);
    }
    else if(argc >= 10)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);

        problem_size.M = std::stoi(argv[4]);
        problem_size.N = std::stoi(argv[5]);
        problem_size.K = std::stoi(argv[6]);

        problem_size.StrideA = std::stoi(argv[7]);
        problem_size.StrideB = std::stoi(argv[8]);
        problem_size.StrideC = std::stoi(argv[9]);

        if(argc >= 11)
        {
            problem_size.Streamk_sel = std::stoi(argv[10]);

            if(argc >= 12)
            {
                problem_size.Grid_size = std::stoi(argv[11]);

                if(argc >= 13)
                {
                    int reduction_strategy          = std::stoi(argv[12]);
                    problem_size.reduction_strategy = reduction_strategy == 0
                                                          ? ck::StreamKReductionStrategy::Atomic
                                                          : ck::StreamKReductionStrategy::Reduction;
                }
            }
        }
    }
    else
    {
        std::cerr
            << "arg1: verification (0=no, 1=CPU, 2=GPU, 3=CPU and GPU)" << std::endl
            << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)" << std::endl
            << "arg3: time kernel (0=no, 1=yes)" << std::endl
            << "arg4 to 9: M (256x), N(128x), K(32x), StrideA, StrideB, StrideC (default: -1 or 0)"
            << std::endl
            << "arg10: stream-k select (-1: default config, 0: all DP, 1: 1-tile SK, 2: 2-tile SK)"
            << std::endl
            << "arg11: Grid_size(-1 for max occupancy)" << std::endl
            << "arg12: Reduction strategy (0: Atomic, 1: Reduction)" << std::endl;
        return false;
    }

    return true;
}

template <>
bool parse_cmd_args<ProblemSizeStreamK>(int argc,
                                        char* argv[],
                                        ProblemSizeStreamK& problem_size,
                                        ExecutionConfig& config)
{
    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 4)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);
    }
    else if(argc >= 10)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);

        problem_size.M = std::stoi(argv[4]);
        problem_size.N = std::stoi(argv[5]);
        problem_size.K = std::stoi(argv[6]);

        problem_size.StrideA = std::stoi(argv[7]);
        problem_size.StrideB = std::stoi(argv[8]);
        problem_size.StrideC = std::stoi(argv[9]);

        if(argc >= 11)
        {
            problem_size.NumSKBlocks = std::stoi(argv[10]);
        }
    }
    else
    {
        std::cerr
            << "arg1: verification (0=no, 1=CPU, 2=GPU, 3=CPU and GPU)" << std::endl
            << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)" << std::endl
            << "arg3: time kernel (0=no, 1=yes)" << std::endl
            << "arg4 to 9: M (256x), N(128x), K(32x), StrideA, StrideB, StrideC (default: -1 or 0)"
            << std::endl
            << "arg10: stream-k select (0: all DP, 1: 1-tile SK, 2: 2-tile SK)"
            << "\narg11: Grid_size(-1 for max occupancy)" << std::endl;
        return false;
    }

    return true;
}

template <>
bool parse_cmd_args<ProblemSizeSplitK>(int argc,
                                       char* argv[],
                                       ProblemSizeSplitK& problem_size,
                                       ExecutionConfig& config)
{
    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 4)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);
    }
    else if(argc >= 10)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);

        problem_size.M = std::stoi(argv[4]);
        problem_size.N = std::stoi(argv[5]);
        problem_size.K = std::stoi(argv[6]);

        problem_size.StrideA = std::stoi(argv[7]);
        problem_size.StrideB = std::stoi(argv[8]);
        problem_size.StrideC = std::stoi(argv[9]);

        if(argc >= 11)
        {
            problem_size.KBatch = std::stoi(argv[10]);
        }
    }
    else
    {
        std::cerr
            << "arg1: verification (0=no, 1=CPU, 2=GPU, 3=CPU and GPU)" << std::endl
            << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)" << std::endl
            << "arg3: time kernel (0=no, 1=yes)" << std::endl
            << "arg4 to 9: M (256x), N(128x), K(32x), StrideA, StrideB, StrideC (default: -1 or 0)"
            << std::endl
            << "arg10: KBatch" << std::endl;
        return false;
    }

    return true;
}

template <typename DataType, typename ComputeDataType = DataType>
inline __host__ __device__ constexpr double get_rtol()
{
    if constexpr(std::is_same_v<DataType, float> && std::is_same_v<ComputeDataType, ck::tf32_t>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, float>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, double>)
    {
        return 1e-6;
    }
    else if constexpr(std::is_same_v<DataType, ck::half_t>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, ck::bhalf_t>)
    {
        return 5e-2;
    }
    else if constexpr(std::is_same_v<DataType, int32_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, int8_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, ck::f8_t>)
    {
        return 1e-1; // 240 and 224 are acceptable
    }
    else if constexpr(std::is_same_v<DataType, ck::bf8_t>)
    {
        return 1.5e-1; // 57344 and 49152 are acceptable
    }
    else
    {
        return 1e-3;
    }
}

template <typename DataType, typename ComputeDataType = DataType>
inline __host__ __device__ constexpr double get_atol()
{
    if constexpr(std::is_same_v<DataType, float> && std::is_same_v<ComputeDataType, ck::tf32_t>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, float>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, double>)
    {
        return 1e-6;
    }
    else if constexpr(std::is_same_v<DataType, ck::half_t>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, ck::bhalf_t>)
    {
        return 5e-2;
    }
    else if constexpr(std::is_same_v<DataType, int32_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, int8_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, ck::f8_t>)
    {
        return 16.1; // 240 and 224 are acceptable
    }
    else if constexpr(std::is_same_v<DataType, ck::bf8_t>)
    {
        return 8192.1; // 57344 and 49152 are acceptable
    }
    else
    {
        return 1e-3;
    }
}

float i4_to_f32_gfx9(uint8_t i4)
{
    static std::unordered_map<uint8_t, float> u = {{0b1000, -0.5000f},
                                                   {0b1001, -0.4375f},
                                                   {0b1010, -0.3750f},
                                                   {0b1011, -0.3125f},
                                                   {0b1100, -0.2500f},
                                                   {0b1101, -0.1875f},
                                                   {0b1110, -0.1250f},
                                                   {0b1111, -0.0625f},
                                                   {0b0, +0.0000f},
                                                   {0b1, +0.0625f},
                                                   {0b10, +0.1250f},
                                                   {0b11, +0.1875f},
                                                   {0b100, +0.2500f},
                                                   {0b101, +0.3125f},
                                                   {0b110, +0.3750f},
                                                   {0b111, +0.4375f}};

    return u[i4];
}
