// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include "quant_run_grouped_gemm_example.hpp"

extern template int run_gemm_example_persistency<ck_tile::fp8_t, ck_tile::QuantType::TensorQuant>(
    const ck_tile::ArgParser&, std::string, std::string, bool);
extern template int run_gemm_example_persistency<ck_tile::fp8_t, ck_tile::QuantType::RowColQuant>(
    const ck_tile::ArgParser&, std::string, std::string, bool);
extern template int run_gemm_example_persistency<ck_tile::fp8_t, ck_tile::QuantType::AQuantGrouped>(
    const ck_tile::ArgParser&, std::string, std::string, bool);
extern template int run_gemm_example_persistency<ck_tile::fp8_t, ck_tile::QuantType::BQuantGrouped>(
    const ck_tile::ArgParser&, std::string, std::string, bool);
extern template int run_gemm_example_persistency<ck_tile::bf8_t, ck_tile::QuantType::TensorQuant>(
    const ck_tile::ArgParser&, std::string, std::string, bool);
extern template int run_gemm_example_persistency<ck_tile::bf8_t, ck_tile::QuantType::RowColQuant>(
    const ck_tile::ArgParser&, std::string, std::string, bool);
extern template int run_gemm_example_persistency<ck_tile::bf8_t, ck_tile::QuantType::AQuantGrouped>(
    const ck_tile::ArgParser&, std::string, std::string, bool);
extern template int run_gemm_example_persistency<ck_tile::bf8_t, ck_tile::QuantType::BQuantGrouped>(
    const ck_tile::ArgParser&, std::string, std::string, bool);

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("Ms", "", "M dimensions - empty by default.")
        .insert("Ns", "", "N dimensions - empty by default.")
        .insert("Ks", "", "K dimensions - empty by default.")
        .insert(
            "stride_As",
            "",
            "Tensor A strides - it is empty by default.") // stride_As/stride_Bs/stride_Cs/stride_AQs/stride_BQs
                                                          // can be set to zero if
                                                          // Ms/Ns/Ks is not empty
        .insert("stride_Bs", "", "Tensor B strides - it is empty by default.")
        .insert("stride_Cs", "", "Tensor C strides - it is empty by default.")
        .insert("stride_AQs", "", "Tensor AQ strides - it is empty by default.")
        .insert("stride_BQs", "", "Tensor BQ strides - it is empty by default.")
        .insert("a_layout", "R", "A tensor data layout - Row by default.")
        .insert("b_layout", "C", "B tensor data layout - Column by default.")
        .insert("c_layout", "R", "C tensor data layout - Row by default.")
        .insert("validate", "1", "0. No validation, 1. Validation on CPU.")
        .insert("prec", "fp8", "data type. fp16/bf16/fp8/bf8")
        .insert("warmup", "10", "number of iterations before benchmark the kernel.")
        .insert("repeat", "100", "number of iterations to benchmark the kernel.")
        .insert("group_count", "8", "group count.")
        .insert("kbatch", "1", "kbatch for SplitK")
        .insert("quant_mode", "bquant", "Choose aquant, bquant (default), tensor, or rowcol")
        .insert("init", "0", "0. Random, 2. One(s) (Constant)")
        .insert("persistent", "0", "Kernel persistency. 0: non-persistent. 1: persistent.");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
    {
        return -1;
    }

    const std::string a_layout  = arg_parser.get_str("a_layout");
    const std::string b_layout  = arg_parser.get_str("b_layout");
    const std::string data_type = arg_parser.get_str("prec");
    std::string quant_mode      = arg_parser.get_str("quant_mode");
    bool persistent             = arg_parser.get_bool("persistent");

    if(data_type == "fp8")
    {
        if(quant_mode == "tensor")
        {
            return run_gemm_example_persistency<ck_tile::fp8_t, ck_tile::QuantType::TensorQuant>(
                arg_parser, a_layout, b_layout, persistent);
        }
        else if(quant_mode == "rowcol")
        {
            return run_gemm_example_persistency<ck_tile::fp8_t, ck_tile::QuantType::RowColQuant>(
                arg_parser, a_layout, b_layout, persistent);
        }
        else if(quant_mode == "aquant")
        {
            return run_gemm_example_persistency<ck_tile::fp8_t, ck_tile::QuantType::AQuantGrouped>(
                arg_parser, a_layout, b_layout, persistent);
        }
        else if(quant_mode == "bquant")
        {
            return run_gemm_example_persistency<ck_tile::fp8_t, ck_tile::QuantType::BQuantGrouped>(
                arg_parser, a_layout, b_layout, persistent);
        }
        else
        {
            throw std::runtime_error("Unsupported quantization mode!");
        }
    }
    if(data_type == "bf8")
    {
        if(quant_mode == "tensor")
        {
            return run_gemm_example_persistency<ck_tile::bf8_t, ck_tile::QuantType::TensorQuant>(
                arg_parser, a_layout, b_layout, persistent);
        }
        else if(quant_mode == "rowcol")
        {
            return run_gemm_example_persistency<ck_tile::bf8_t, ck_tile::QuantType::RowColQuant>(
                arg_parser, a_layout, b_layout, persistent);
        }
        else if(quant_mode == "aquant")
        {
            return run_gemm_example_persistency<ck_tile::bf8_t, ck_tile::QuantType::AQuantGrouped>(
                arg_parser, a_layout, b_layout, persistent);
        }
        else if(quant_mode == "bquant")
        {
            return run_gemm_example_persistency<ck_tile::bf8_t, ck_tile::QuantType::BQuantGrouped>(
                arg_parser, a_layout, b_layout, persistent);
        }
        else
        {
            throw std::runtime_error("Unsupported quantization mode!");
        }
    }
    else
    {
        throw std::runtime_error("Unsupported data type configuration.");
    }
}
