// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <string>

#include "gemm_utils.hpp"
#include "run_gemm_example.inc"
#include "run_gemm_example_common.hpp"
#include "universal_gemm_invoker.hpp"

template <template <typename PrecType> typename GemmConfig>
int run_gemm_example(ck_tile::ArgParser& arg_parser)
{
    std::string data_type = arg_parser.get_str("prec");
    std::string a_layout  = arg_parser.get_str("a_layout");
    std::string b_layout  = arg_parser.get_str("b_layout");

    using Invoker = UniversalInvoker;

    if(data_type == "fp16")
    {
        return run_gemm_example_prec_type<GemmConfig<ck_tile::half_t>, Invoker, ck_tile::half_t>(
            a_layout, b_layout, arg_parser);
    }
    else if(data_type == "bf16")
    {
        return run_gemm_example_prec_type<GemmConfig<ck_tile::bf16_t>, Invoker, ck_tile::bf16_t>(
            a_layout, b_layout, arg_parser);
    }
    else if(data_type == "fp8")
    {
        return run_gemm_example_prec_type<GemmConfig<ck_tile::fp8_t>,
                                          Invoker,
                                          ck_tile::fp8_t,
                                          ck_tile::fp8_t,
                                          ck_tile::half_t>(a_layout, b_layout, arg_parser);
    }
    else if(data_type == "bf8")
    {
        return run_gemm_example_prec_type<GemmConfig<ck_tile::bf8_t>,
                                          Invoker,
                                          ck_tile::bf8_t,
                                          ck_tile::bf8_t,
                                          ck_tile::half_t>(a_layout, b_layout, arg_parser);
    }
    else if(data_type == "int8")
    {
        return run_gemm_example_prec_type<GemmConfig<ck_tile::int8_t>,
                                          Invoker,
                                          ck_tile::int8_t,
                                          ck_tile::int8_t,
                                          ck_tile::int32_t>(a_layout, b_layout, arg_parser);
    }
    else if(data_type == "fp16i4")
    {
        // TODO: Add support for bhalf_t ADataType
        if constexpr(GemmConfig<ck_tile::half_t>::Pipeline == ck_tile::GemmPipeline::COMPUTE_V3)
        {
            return run_gemm_example_prec_type<GemmConfig<ck_tile::half_t>,
                                              Invoker,
                                              ck_tile::half_t,
                                              ck_tile::pk_int4_t,
                                              ck_tile::half_t>(a_layout, b_layout, arg_parser);
        }
        else
        {
            throw std::runtime_error("Unsupported pipeline for this operation !!!");
        }
    }
    else if(data_type == "fp8i4")
    {
        if constexpr(GemmConfig<ck_tile::fp8_t>::Pipeline == ck_tile::GemmPipeline::COMPUTE_V3)
        {
            return run_gemm_example_prec_type<GemmConfig<ck_tile::fp8_t>,
                                              Invoker,
                                              ck_tile::fp8_t,
                                              ck_tile::pk_int4_t,
                                              ck_tile::half_t>(a_layout, b_layout, arg_parser);
        }
        else
        {
            throw std::runtime_error("Unsupported pipeline for this operation !!!");
        }
    }
    else if(data_type == "bf8i4")
    {
        if constexpr(GemmConfig<ck_tile::bf8_t>::Pipeline == ck_tile::GemmPipeline::COMPUTE_V3)
        {
            return run_gemm_example_prec_type<GemmConfig<ck_tile::bf8_t>,
                                              Invoker,
                                              ck_tile::bf8_t,
                                              ck_tile::pk_int4_t,
                                              ck_tile::half_t>(a_layout, b_layout, arg_parser);
        }
        else
        {
            throw std::runtime_error("Unsupported pipeline for this operation !!!");
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
#if CK_TILE_USE_WMMA
        return !run_gemm_example<GemmConfigComputeV3_WMMA>(arg_parser);
#else
        return !run_gemm_example<GemmConfigComputeV3_2>(arg_parser);
#endif
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Caught runtime error: " << e.what() << '\n';
        // Return a non-zero code to indicate failure
        return EXIT_FAILURE;
    }
}
