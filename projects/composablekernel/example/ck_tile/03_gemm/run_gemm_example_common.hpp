// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include "gemm_utils.hpp"

template <typename GemmConfig,
          typename Invoker,
          typename APrecType,
          typename BPrecType = APrecType,
          typename CPrecType = APrecType>
int run_gemm_example_prec_type(std::string a_layout,
                               std::string b_layout,
                               ck_tile::ArgParser& arg_parser)
{
    using Row       = ck_tile::tensor_layout::gemm::RowMajor;
    using Col       = ck_tile::tensor_layout::gemm::ColumnMajor;
    bool preshuffle = GemmConfig::Preshuffle;

    if(preshuffle && std::is_same_v<BPrecType, ck_tile::pk_int4_t>)
    {
        throw std::runtime_error("Preshuffle is not supported for this int4 datatype!");
    }

    if(preshuffle && a_layout != "R" && b_layout != "C")
    {
        throw std::runtime_error(
            "Preshuffle is supported only for A(Row major), B(column major) input matrices!");
    }

    using LayoutVariant = std::variant<Row, Col>;

    auto string_to_layout = [](const std::string& layout) -> LayoutVariant {
        if(layout == "R")
            return Row{};
        if(layout == "C")
            return Col{};
        throw std::runtime_error("Unsupported layout: " + layout);
    };

    auto a_layout_variant = string_to_layout(a_layout);
    auto b_layout_variant = string_to_layout(b_layout);

    return std::visit(
        [&](auto a_layout_type, auto b_layout_type) -> int {
            if constexpr(std::is_same_v<BPrecType, ck_tile::pk_int4_t> &&
                         std::is_same_v<decltype(b_layout_type), Row>)
            {
                throw std::runtime_error("Unsupported memory layout for the input matrices when "
                                         "BPrecType is ck_tile::pk_int4_t!");
            }
            else
            {
                return run_gemm_example_with_layouts<GemmConfig,
                                                     Invoker,
                                                     APrecType,
                                                     BPrecType,
                                                     CPrecType>(
                    arg_parser, a_layout_type, b_layout_type, Row{});
            }
        },
        a_layout_variant,
        b_layout_variant);
}
