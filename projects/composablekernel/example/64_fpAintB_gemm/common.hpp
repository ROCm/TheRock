// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <numeric>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/utility/data_type.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_fpAintB_gemm.hpp"

using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::make_ParallelTensorFunctor;
using ::ck::Tensor;

struct ProblemSize final
{
    ck::index_t M = 3840;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    ck::index_t StrideA = 4096;
    ck::index_t StrideB = 4096;
    ck::index_t StrideC = 4096;
};

struct ExecutionConfig final
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;
};

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

template <typename IntType>
struct UnsignedWeightPreprocessor
{
};

template <>
struct UnsignedWeightPreprocessor<int8_t>
{
    using UnsignedWeight = Tensor<uint8_t>;
    using SignedWeight   = Tensor<int8_t>;
    static UnsignedWeight convert(SignedWeight const& Input)
    {

        UnsignedWeight Output = Input.template CopyAsType<uint8_t>();

        auto f_kn = [&](auto k, auto n) {
            const uint8_t adder = 128;
            int8_t v_signed_weight;
            uint8_t v_unsigned_weight;

            ck::tensor_operation::element_wise::PassThrough{}(v_signed_weight, Input(k, n));
            v_unsigned_weight = ck::type_convert<uint8_t>(v_signed_weight) + adder;
            Output(k, n)      = v_unsigned_weight;
        };

        make_ParallelTensorFunctor(f_kn, Input.mDesc.GetLengths()[0], Input.mDesc.GetLengths()[1])(
            std::thread::hardware_concurrency());

        return Output;
    }

    UnsignedWeight operator()(SignedWeight const& Input) { return convert(Input); }
};

inline bool
parse_cmd_args(int argc, char* argv[], ProblemSize& problem_size, ExecutionConfig& config)
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
        std::cerr << "arg1: verification (0=no, 1=yes)" << std::endl
                  << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)"
                  << std::endl
                  << "arg3: time kernel (0=no, 1=yes)" << std::endl
                  << "arg4 to 9: M (256x), N(128x), K(32x), StrideA, StrideB, StrideC" << std::endl;
        return false;
    }

    return true;
}
