// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// @file
/// @brief Implementation of the describe() function template for convolution kernels

#pragma once

#include "ck_tile/builder/reflect/conv_description.hpp"
#include "ck_tile/builder/reflect/conv_traits.hpp"

namespace ck_tile::reflect {

/// @brief Factory function to create ConvDescription from a convolution instance type
/// @tparam Instance The convolution instance type (must have ConvTraits)
/// @return A ConvDescription object populated with the instance's configuration details
template <conv::HasConvTraits Instance>
conv::ConvDescription describe()
{
    using Traits = conv::ConvTraits<Instance>;

    return conv::ConvDescription(
        conv::ConvSignatureInfo{
            .spatial_dim       = Traits::spatial_dim,
            .direction         = Traits::direction,
            .input_layout      = Traits::layout[0],
            .weight_layout     = Traits::layout[1],
            .output_layout     = Traits::layout[2],
            .data_type         = Traits::data_type,
            .input_element_op  = Traits::input_element_op,
            .weight_element_op = Traits::weight_element_op,
            .output_element_op = Traits::output_element_op,
        },
        conv::GemmAlgorithmInfo{
            .thread_block_size   = Traits::thread_block_size,
            .tile_dims           = Traits::tile_dims,
            .warp_gemm           = Traits::warp_gemm,
            .a_tile_transfer     = Traits::a_tile_transfer,
            .b_tile_transfer     = Traits::b_tile_transfer,
            .c_tile_transfer     = Traits::c_tile_transfer,
            .pipeline_version    = Traits::pipeline_version,
            .pipeline_scheduler  = Traits::pipeline_scheduler,
            .conv_specialization = Traits::conv_specialization,
            .padding             = Traits::gemm_padding,
        },
        []() { return reflect::instance_string<Instance>(); });
}

} // namespace ck_tile::reflect
