// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/testing/conv_fwd.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_elementwise_op.hpp"
#include <type_traits>
#include <array>

/// This file contains the implementation details for invoking/testing
/// grouped convolution operations in old CK. The main item is the
/// `run()` function, which is the main implementation used to invoke
/// CK grouped forward convolution kernels.

namespace ck_tile::builder::test {

namespace detail {

/// @brief Concept for checking whether this is the reference convolution
/// implementation.
///
/// This is the same as `::ck_tile::builder::test::CkConvInstance`, except
/// with some utility aliases. For that reason, its moved to this detail
/// namespace.
template <typename Conv,
          auto SIGNATURE,
          size_t SPATIAL_DIM = SIGNATURE.spatial_dim,
          // TODO: We shouldn't need to call into an internal namespace here.
          typename Ops = factory::internal::ElementwiseOps<SIGNATURE>>
concept CkConvInstance = requires(Conv& conv,
                                  // TODO: This should be changed depending on IsMultiA etc.
                                  // Currently that is not yet supported elsewhere anyway.
                                  const void* p_a,
                                  const void* p_b,
                                  void* p_e,
                                  std::array<index_t, SPATIAL_DIM + 3> lengths,
                                  std::array<index_t, SPATIAL_DIM + 3> strides,
                                  std::array<index_t, SPATIAL_DIM> filter,
                                  Ops::AElementwiseOp elementwise_a,
                                  Ops::BElementwiseOp elementwise_b,
                                  Ops::CDEElementwiseOp elementwise_cde) {
    {
        conv.MakeArgument(p_a,
                          p_b,
                          // TODO: Support multiple D outputs.
                          {},
                          p_e,
                          // A lengths/strides
                          lengths,
                          strides,
                          // B lengths/strides
                          lengths,
                          strides,
                          // TODO: Ds lengths/strides
                          {},
                          {},
                          // E lengths/strides
                          lengths,
                          strides,
                          // strides/dilations/pads
                          filter,
                          filter,
                          filter,
                          filter,
                          // element-wise operations.
                          elementwise_a,
                          elementwise_b,
                          elementwise_cde)
    };
};

} // namespace detail

/// @brief Concept for checking whether a convolution is invoked like old CK.
///
/// This concept is used to tell whether a convolution implementation is
/// likely to be an "old CK" implementation - that is, whether we should
/// invoke it as an old CK kernel. This is mainly used with `run()` to
/// differentiate which implementation that should be invoked.
///
/// - SIGNATURE is the operation signature.
/// - Conv is a convolution instance created by the CK Builder API.
template <typename Conv, auto SIGNATURE>
concept CkConvInstance = detail::CkConvInstance<Conv, SIGNATURE>;

/// @brief `run()` specialization for forward convolution and old CK.
///
/// @tparam SIGNATURE Forward convolution signature.
/// @throws std::runtime_error if the arguments werent actually valid for the
/// operation. This should be caught and reported by the testing framework.
///
/// @see run()
template <auto SIGNATURE>
    requires ValidConvSignature<SIGNATURE> && ConvDirectionIsForward<SIGNATURE>
void run(CkConvInstance<SIGNATURE> auto& conv,
         const Args<SIGNATURE>& args,
         const Inputs<SIGNATURE>& inputs,
         const Outputs<SIGNATURE>& outputs)
{
    constexpr auto spatial_dim = SIGNATURE.spatial_dim;

    const auto copy = [](const auto& src, auto& dst) {
        std::copy(src.begin(), src.end(), dst.begin());
    };

    const auto to_ck_lengths = [&](const auto& src) {
        std::array<ck::index_t, spatial_dim + 3> result;
        copy(src, result);
        return result;
    };

    const auto to_ck_extent = [&](const auto& extent) {
        std::array<ck::index_t, spatial_dim> result;
        copy(extent, result);
        return result;
    };

    const auto param = args.to_ck_conv_param();

    const auto input_desc  = args.make_input_descriptor();
    const auto weight_desc = args.make_weight_descriptor();
    const auto output_desc = args.make_output_descriptor();

    auto ck_args = conv.MakeArgument(inputs.input,
                                     inputs.weight,
                                     {},
                                     outputs.output,
                                     to_ck_lengths(input_desc.get_lengths()),
                                     to_ck_lengths(input_desc.get_strides()),
                                     to_ck_lengths(weight_desc.get_lengths()),
                                     to_ck_lengths(weight_desc.get_strides()),
                                     {},
                                     {},
                                     to_ck_lengths(output_desc.get_lengths()),
                                     to_ck_lengths(output_desc.get_strides()),
                                     to_ck_extent(param.conv_filter_strides_),
                                     to_ck_extent(param.conv_filter_dilations_),
                                     to_ck_extent(param.input_left_pads_),
                                     to_ck_extent(param.input_right_pads_),
                                     args.a_elementwise_op,
                                     args.b_elementwise_op,
                                     args.cde_elementwise_op);

    if(!conv.IsSupportedArgument(ck_args))
    {
        throw std::runtime_error("invalid argument");
    }

    conv.MakeInvoker().Run(ck_args, {});
}

} // namespace ck_tile::builder::test
