// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gemm_utils.hpp"
#include "run_gemm_example.inc"
#include "run_gemm_example_common.hpp"
#include "gemm_basic_invoker.hpp"
#include "ck_tile/core/utility/gemm_validation.hpp"

int run_gemm_example(ck_tile::ArgParser& arg_parser)
{
    std::string data_type = arg_parser.get_str("prec");
    std::string a_layout  = arg_parser.get_str("a_layout");
    std::string b_layout  = arg_parser.get_str("b_layout");
    std::string c_layout  = arg_parser.get_str("c_layout");

    std::tuple<ck_tile::index_t, ck_tile::index_t, ck_tile::index_t> gemm_sizes =
        parse_gemm_size(arg_parser);

    int m = std::get<0>(gemm_sizes);
    int n = std::get<1>(gemm_sizes);
    int k = std::get<2>(gemm_sizes);

    int stride_a = arg_parser.get_int("stride_a");
    int stride_b = arg_parser.get_int("stride_b");
    int stride_c = arg_parser.get_int("stride_c");

    using GemmConfig = GemmConfigBase;
    using Invoker    = BasicInvoker;

    ck_tile::validate_gemm_stride(
        a_layout, b_layout, c_layout, m, n, k, stride_a, stride_b, stride_c);

    if(data_type == "fp16")
    {
        return run_gemm_example_prec_type<GemmConfig, Invoker, ck_tile::half_t>(
            a_layout, b_layout, arg_parser);
    }
    else if(data_type == "bf16")
    {
        return run_gemm_example_prec_type<GemmConfig, Invoker, ck_tile::bf16_t>(
            a_layout, b_layout, arg_parser);
    }
    else if(data_type == "fp8")
    {
        return run_gemm_example_prec_type<GemmConfig,
                                          Invoker,
                                          ck_tile::fp8_t,
                                          ck_tile::fp8_t,
                                          ck_tile::half_t>(a_layout, b_layout, arg_parser);
    }
    else if(data_type == "bf8")
    {
        return run_gemm_example_prec_type<GemmConfig,
                                          Invoker,
                                          ck_tile::bf8_t,
                                          ck_tile::bf8_t,
                                          ck_tile::half_t>(a_layout, b_layout, arg_parser);
    }
    else if(data_type == "i8")
    {
        return run_gemm_example_prec_type<GemmConfig,
                                          Invoker,
                                          ck_tile::int8_t,
                                          ck_tile::int8_t,
                                          int32_t>(a_layout, b_layout, arg_parser);
    }
    else if(data_type == "pk_int4_t")
    {
        // TODO: Add support for bhalf_t ADataType
        if constexpr(GemmConfig::Pipeline == ck_tile::GemmPipeline::COMPUTE_V3)
        {
            return run_gemm_example_prec_type<GemmConfig,
                                              Invoker,
                                              ck_tile::half_t,
                                              ck_tile::pk_int4_t,
                                              ck_tile::half_t>(a_layout, b_layout, arg_parser);
        }
        else
        {
            throw std::runtime_error("Unsupported data type for this operation !!!");
        }
    }
    else
    {
        throw std::runtime_error("Unsupported data type for this operation !!!");
    }
}

int main(int argc, char* argv[])
{
    auto arg_parser = create_args();
    auto result     = arg_parser.parse(argc, argv);

    if(!result)
        return -1;

    try
    {
        return !run_gemm_example(arg_parser);
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Runtime error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
