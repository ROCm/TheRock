// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/grouped_convolution.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
#include "conv_configs.hpp"

template <typename InDataType, typename WeiDataType, typename AccDataType, typename OutDataType>
auto calculate_rtol_atol(const ck_tile::index_t GemmK,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    using ComputeType =
        std::conditional_t<sizeof(InDataType) < sizeof(WeiDataType), InDataType, WeiDataType>;
    // Calculate thresholds
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, OutDataType, AccDataType>(
        ck_tile::integer_divide_ceil(GemmK, kbatch));
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, OutDataType, AccDataType>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(GemmK, kbatch));
    // Calculate error due to split_k accumulation
    const auto rtol_split_k =
        ck_tile::get_relative_threshold<OutDataType, OutDataType, OutDataType>(kbatch);
    const auto atol_split_k =
        ck_tile::get_absolute_threshold<OutDataType, OutDataType, OutDataType>(
            max_accumulated_value, kbatch);
    // Use higher threshold
    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}

ck_tile::index_t fill_spatial_dimensions(std::vector<ck_tile::index_t>& filter_spatial_lengths,
                                         std::vector<ck_tile::index_t>& image_spatial_lengths,
                                         std::vector<ck_tile::index_t>& strides,
                                         std::vector<ck_tile::index_t>& dilations,
                                         std::vector<ck_tile::index_t>& lpads,
                                         std::vector<ck_tile::index_t>& rpads,
                                         ck_tile::ArgParser& arg_parser)
{

    constexpr ck_tile::index_t non_sp_dims = 3;
    const ck_tile::index_t n_dim_sp        = arg_parser.get_str("in_layout").size() - non_sp_dims;

    if(!(n_dim_sp >= 1 && n_dim_sp <= 3))
    {
        throw std::runtime_error("Wrong layout!\n");
    }

    if(n_dim_sp == 3)
    {
        filter_spatial_lengths.push_back(arg_parser.get_int("z"));
        image_spatial_lengths.push_back(arg_parser.get_int("d"));
        strides.push_back(arg_parser.get_int("stride_d"));
        dilations.push_back(arg_parser.get_int("dilation_d"));
        lpads.push_back(arg_parser.get_int("lpad_d"));
        rpads.push_back(arg_parser.get_int("rpad_d"));
    }
    if(n_dim_sp >= 2)
    {
        filter_spatial_lengths.push_back(arg_parser.get_int("y"));
        image_spatial_lengths.push_back(arg_parser.get_int("h"));
        strides.push_back(arg_parser.get_int("stride_h"));
        dilations.push_back(arg_parser.get_int("dilation_h"));
        lpads.push_back(arg_parser.get_int("lpad_h"));
        rpads.push_back(arg_parser.get_int("rpad_h"));
    }
    filter_spatial_lengths.push_back(arg_parser.get_int("x"));
    image_spatial_lengths.push_back(arg_parser.get_int("w"));
    strides.push_back(arg_parser.get_int("stride_w"));
    dilations.push_back(arg_parser.get_int("dilation_w"));
    lpads.push_back(arg_parser.get_int("lpad_w"));
    rpads.push_back(arg_parser.get_int("rpad_w"));

    return n_dim_sp;
}

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("g", "2", "group dimension")
        .insert("n", "32", "n dimension")
        .insert("k", "32", "k dimension")
        .insert("c", "32", "c dimension")

        .insert("d", "64", "d dimension")
        .insert("h", "64", "h dimension")
        .insert("w", "64", "w dimension")

        .insert("z", "4", "z dimension")
        .insert("y", "4", "y dimension")
        .insert("x", "4", "x dimension")

        .insert("stride_d", "1", "d stride")
        .insert("stride_h", "1", "h stride")
        .insert("stride_w", "1", "w stride")

        .insert("dilation_d", "1", "d dilation")
        .insert("dilation_h", "1", "h dilation")
        .insert("dilation_w", "1", "w dilation")

        .insert("lpad_d", "0", "left pad for d dimension")
        .insert("lpad_h", "0", "left pad for h dimension")
        .insert("lpad_w", "0", "left pad for w dimension")

        .insert("rpad_d", "0", "right pad for d dimension")
        .insert("rpad_h", "0", "right pad for h dimension")
        .insert("rpad_w", "0", "right pad for w dimension")

        .insert("in_layout", "NHWGC", "Input image layout - NHWGC by default")
        .insert("wei_layout", "GKYXC", "Weight layout - GKYXC by default")
        .insert("out_layout", "NHWGK", "Output image layout - NHWGK by default")
        .insert("v", "1", "0. No validation, 1. Validation on CPU, 2. Validation on GPU")
        .insert("prec", "fp16", "data type. fp16/bf16/fp8/bf8")
        .insert("warmup", "50", "number of iterations before benchmark the kernel")
        .insert("repeat", "100", "number of iterations to benchmark the kernel")
        .insert("timer", "gpu", "gpu:gpu timer, cpu:cpu timer")
        .insert("split_k", "1", "splitK value")
        .insert("init", "0", "0:random, 1:linear, 2:constant(1)")
        .insert("json", "0", "0: No Json, 1: Dump Results in Json format");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

struct InvokerResult
{
    float ave_time;
    ck_tile::index_t split_k;
};
