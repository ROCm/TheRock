// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_tensor_layout.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_elementwise_op.hpp"
#include "ck_tile/builder/testing/testing.hpp"
#include "ck_tile/builder/testing/filter_extent.hpp"
#include "ck_tile/builder/testing/tensor_buffer.hpp"
#include "ck_tile/builder/testing/tensor_initialization.hpp"
#include "ck_tile/builder/testing/tensor_descriptor.hpp"
#include "ck_tile/builder/testing/validation.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"

/// This file implements common functionality for invoking/testing grouped
/// forward convolutions created through the CK Builder API. The main item
/// of it is the ConvArgs structure - which contains a complete description
/// of a convolution operation.
///
/// It is not intended that this file contains implementation details for
/// actually launching a convolution operation. As this can be done
/// through different APIs depending on the kernel (CK, CK Tile, or a
/// reference implementation), the code dealing with that is split out
/// into a separate header for each implementation.

namespace ck_tile::builder::test {

/// @brief Convolution tensor dimensions.
///
/// This structure is used to describe lengths of a convolution problem. In
/// fact, this structure is a complete description of ALL inputs and outputs
/// lengths of a convolution problem, as this structure contains all of the
/// combined parameters. Note that we can't also use this structure to describe
/// tensor strides: whereas the lengths are all governed by a common set of
/// parameters, strides of the input, weight, and output tensor are all
/// independent.
template <int SPATIAL_DIM>
struct ConvTensorLengths
{
    size_t batch_size                = 1;  // N
    size_t groups                    = 1;  // G
    size_t input_channels            = 1;  // C
    size_t output_channels           = 1;  // K
    FilterExtent<SPATIAL_DIM> image  = {}; // W, H, D
    FilterExtent<SPATIAL_DIM> filter = {}; // X, Y, Z
};

/// @brief `Args` specialization for forward convolution.
///
/// @tparam SIGNATURE Forward convolution signature.
///
/// @see Args
template <auto SIGNATURE>
    requires ValidConvSignature<SIGNATURE> && ConvDirectionIsForward<SIGNATURE>
struct Args<SIGNATURE>
{
    constexpr static auto SPATIAL_DIM = SIGNATURE.spatial_dim;
    constexpr static auto INPUT_TYPE  = SIGNATURE.data_type;
    constexpr static auto WEIGHT_TYPE = SIGNATURE.data_type;
    constexpr static auto OUTPUT_TYPE = SIGNATURE.data_type;

    constexpr static int INPUT_RANK  = 3 + SPATIAL_DIM;
    constexpr static int WEIGHT_RANK = 3 + SPATIAL_DIM;
    constexpr static int OUTPUT_RANK = 3 + SPATIAL_DIM;

    using InputDescriptor  = TensorDescriptor<INPUT_TYPE, INPUT_RANK>;
    using WeightDescriptor = TensorDescriptor<WEIGHT_TYPE, WEIGHT_RANK>;
    using OutputDescriptor = TensorDescriptor<OUTPUT_TYPE, OUTPUT_RANK>;

    // TODO: We shouldn't need to call into an internal namespace here.
    using Ops = factory::internal::ElementwiseOps<SIGNATURE>;

    // TODO: We shouldn't need to call into an internal namespace here.
    using Layouts =
        factory::internal::ConvTensorLayouts<SIGNATURE, SPATIAL_DIM, ConvDirection::FORWARD>;

    ConvTensorLengths<SPATIAL_DIM> lengths;

    // TODO: Tensor strides. This needs a new structure as well as some
    // reworking of the make_*_descriptor() functions, as the current
    // implementation (based on ConvParam in old CK / CK Tile) does not
    // support strides at all.

    FilterExtent<SPATIAL_DIM> filter_strides;
    FilterExtent<SPATIAL_DIM> filter_dilation;
    FilterExtent<SPATIAL_DIM> input_left_pad;
    FilterExtent<SPATIAL_DIM> input_right_pad;

    Ops::AElementwiseOp a_elementwise_op;
    Ops::BElementwiseOp b_elementwise_op;
    Ops::CDEElementwiseOp cde_elementwise_op;

    /// This function returns the `TensorDescriptor` corresponding to
    /// the input-tensor of the convolution problem. This can then
    /// be used to, for example, allocate memory.
    InputDescriptor make_input_descriptor() const
    {
        // TODO: We're using old CK functionality to compute the right
        // values here, mainly because CK tile does not support the
        // right tensor layouts here. We should probably change that
        // because CK currently prints an annoying message about it,
        // plus that would let us get rid of the `to_ck_conv_param()`
        // function.
        const auto param = to_ck_conv_param();
        const auto desc  = ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<
             typename Layouts::ALayout>(param);
        using Extent = typename InputDescriptor::Extent;
        return InputDescriptor(Extent::from_vector(desc.GetLengths()),
                               Extent::from_vector(desc.GetStrides()));
    }

    /// This function returns the `TensorDescriptor` corresponding to
    /// the weight-tensor of  the convolution problem. This can then
    /// be used to, for example, allocate memory.
    WeightDescriptor make_weight_descriptor() const
    {
        // See note in implementation of `make_input_descriptor`.
        const auto param = to_ck_conv_param();
        const auto desc  = ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<
             typename Layouts::BLayout>(param);
        using Extent = typename WeightDescriptor::Extent;
        return WeightDescriptor(Extent::from_vector(desc.GetLengths()),
                                Extent::from_vector(desc.GetStrides()));
    }

    /// This function returns the `TensorDescriptor` corresponding to
    /// the output-tensor of the convolution problem. This can then
    /// be used to, for example, allocate memory.
    OutputDescriptor make_output_descriptor() const
    {
        // See note in implementation of `make_input_descriptor`.
        const auto param = to_ck_conv_param();
        const auto desc  = ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<
             typename Layouts::ELayout>(param);
        using Extent = typename OutputDescriptor::Extent;
        return OutputDescriptor(Extent::from_vector(desc.GetLengths()),
                                Extent::from_vector(desc.GetStrides()));
    }

    /// Convert the Args structure into a CK conv_param structure. This
    /// function is mainly used to be able to use the existing
    /// CK-functionality to obtain tensor descriptors.
    ck::utils::conv::ConvParam to_ck_conv_param() const
    {
        const auto to_vector = [](const auto& extent) {
            if constexpr(SPATIAL_DIM == 1)
                return std::vector<ck::index_t>{ck::index_t(extent.width)};
            else if constexpr(SPATIAL_DIM == 2)
                return std::vector<ck::index_t>{ck::index_t(extent.height),
                                                ck::index_t(extent.width)};
            else
                return std::vector<ck::index_t>{ck::index_t(extent.depth),
                                                ck::index_t(extent.height),
                                                ck::index_t(extent.width)};
        };

        return ck::utils::conv::ConvParam(SPATIAL_DIM,
                                          this->lengths.groups,
                                          this->lengths.batch_size,
                                          this->lengths.output_channels,
                                          this->lengths.input_channels,
                                          to_vector(this->lengths.filter),
                                          to_vector(this->lengths.image),
                                          to_vector(this->filter_strides),
                                          to_vector(this->filter_dilation),
                                          to_vector(this->input_left_pad),
                                          to_vector(this->input_right_pad));
    }
};

/// @brief `Inputs` specialization for forward convolution.
///
/// @tparam SIGNATURE Forward convolution signature.
///
/// @see Inputs
template <auto SIGNATURE>
    requires ValidConvSignature<SIGNATURE> && ConvDirectionIsForward<SIGNATURE>
struct Inputs<SIGNATURE>
{
    void* input;
    void* weight;
};

/// @brief `Outputs` specialization for forward convolution.
///
/// @tparam SIGNATURE Forward convolution signature.
///
/// @see Outputs
template <auto SIGNATURE>
    requires ValidConvSignature<SIGNATURE> && ConvDirectionIsForward<SIGNATURE>
struct Outputs<SIGNATURE>
{
    void* output;
};

/// @brief `UniqueInputs` specialization for forward convolution.
///
/// @tparam SIGNATURE Forward convolution signature.
///
/// @see UniqueInputs
/// @see ValidUniqueInputs
template <auto SIGNATURE>
    requires ValidConvSignature<SIGNATURE> && ConvDirectionIsForward<SIGNATURE>
struct UniqueInputs<SIGNATURE>
{
    DeviceBuffer input_buf;
    DeviceBuffer weight_buf;

    /// @see ValidUniqueInputs
    Inputs<SIGNATURE> get()
    {
        return {
            .input  = input_buf.get(),
            .weight = weight_buf.get(),
        };
    }
};

/// @brief `UniqueOutputs` specialization for forward convolution.
///
/// @tparam SIGNATURE Forward convolution signature.
///
/// @see UniqueOutputs
/// @see ValidUniqueOutputs
template <auto SIGNATURE>
    requires ValidConvSignature<SIGNATURE> && ConvDirectionIsForward<SIGNATURE>
struct UniqueOutputs<SIGNATURE>
{
    DeviceBuffer output_buf;

    /// @see ValidUniqueOutputs
    Outputs<SIGNATURE> get()
    {
        return {
            .output = output_buf.get(),
        };
    }
};

/// @brief `alloc_inputs()` specialization for forward convolution.
///
/// @tparam SIGNATURE Forward convolution signature.
///
/// @see alloc_inputs()
template <auto SIGNATURE>
    requires ValidConvSignature<SIGNATURE> && ConvDirectionIsForward<SIGNATURE> &&
             ValidUniqueInputs<SIGNATURE>
UniqueInputs<SIGNATURE> alloc_inputs(const Args<SIGNATURE>& args)
{
    return {
        .input_buf  = alloc_tensor_buffer(args.make_input_descriptor()),
        .weight_buf = alloc_tensor_buffer(args.make_weight_descriptor()),
    };
}

/// @brief `init_inputs()` specialization for forward convolution.
///
/// @tparam SIGNATURE Forward convolution signature.
///
/// @see alloc_inputs()
template <auto SIGNATURE>
    requires ValidConvSignature<SIGNATURE> && ConvDirectionIsForward<SIGNATURE>
void init_inputs(const Args<SIGNATURE>& args, Inputs<SIGNATURE> inputs)
{
    init_tensor_buffer_uniform_fp(inputs.input, args.make_input_descriptor(), -2.0f, 2.0f);
    init_tensor_buffer_uniform_fp(inputs.weight, args.make_weight_descriptor(), -2.0f, 2.0f);
}

/// @brief `alloc_outputs()` specialization for forward convolution.
///
/// @tparam SIGNATURE Forward convolution signature.
///
/// @see alloc_outputs()
template <auto SIGNATURE>
    requires ValidConvSignature<SIGNATURE> && ConvDirectionIsForward<SIGNATURE> &&
             ValidUniqueOutputs<SIGNATURE>
UniqueOutputs<SIGNATURE> alloc_outputs(const Args<SIGNATURE>& args)
{
    return {
        .output_buf = alloc_tensor_buffer(args.make_output_descriptor()),
    };
}

/// @brief `validate()` specialization for forward convolution.
///
/// @tparam SIGNATURE Forward convolution signature.
///
/// @see validate()
template <auto SIGNATURE>
    requires ValidConvSignature<SIGNATURE> && ConvDirectionIsForward<SIGNATURE>
ValidationReport
validate(const Args<SIGNATURE>& args, Outputs<SIGNATURE> actual, Outputs<SIGNATURE> expected)
{
    ValidationReport report;
    report.check("output", args.make_output_descriptor(), actual.output, expected.output);
    return report;
}

} // namespace ck_tile::builder::test
