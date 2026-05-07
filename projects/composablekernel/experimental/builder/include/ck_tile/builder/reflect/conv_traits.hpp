// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include "ck/tensor_operation/gpu/device/convolution_backward_data_specialization.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_weight_specialization.hpp"
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/utility/pipeline_enum.hpp"
#include "ck/utility/scheduler_enum.hpp"
#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/reflect/conv_types.hpp"
#include "ck_tile/builder/reflect/instance_traits.hpp"
#include "ck_tile/builder/reflect/instance_traits_util.hpp"
#include "ck_tile/builder/types.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/grouped_convolution.hpp"

namespace ck_tile::reflect::conv {

// Forward convolution layout concept - checks for A/B/E layout types
template <typename T>
concept HasFwdConvLayouts = requires {
    typename T::ALayout;
    typename T::BLayout;
    typename T::ELayout;
};

// GEMM specialization concept - checks for kGemmSpecialization member
template <typename T>
concept HasGemmSpec = requires {
    {
        T::kGemmSpecialization
    } -> std::convertible_to<ck::tensor_operation::device::GemmSpecialization>;
};

// Data types concept - checks for ADataType member
template <typename T>
concept HasDataTypes = requires { typename T::ADataType; };

// Elementwise operations concept - checks for A/B/CDE elementwise operation types
template <typename T>
concept HasElementwiseOps = requires {
    typename T::AElementwiseOperation;
    typename T::BElementwiseOperation;
    typename T::CDEElementwiseOperation;
};

// Tile parameters concept - checks for tile dimension and transfer members
template <typename T>
concept HasTileParams = requires {
    { T::kKPerBlock } -> std::convertible_to<int>;
    { T::kMPerBlock } -> std::convertible_to<int>;
    { T::kNPerBlock } -> std::convertible_to<int>;
    { T::kAK1 } -> std::convertible_to<int>;
    { T::kBK1 } -> std::convertible_to<int>;
    T::kCThreadClusterLengths;
};

// Comprehensive concept that checks if an instance has all XDL forward convolution traits
// This concept is used to constrain ConvTraits specialization that expect XDL forward convolutions
template <typename T>
concept IsXdlFwdConv = HasFwdConvLayouts<T> && HasGemmSpec<T> && HasDataTypes<T> &&
                       HasElementwiseOps<T> && HasTileParams<T>;

// Primary concept for checking if a type can be described
// Currently only forward convolutions are supported, but this can be extended
// in the future to include backward data and backward weight convolutions
template <typename T>
concept HasConvTraits = IsXdlFwdConv<InstanceTraits<T>>;

// Helper metafunctions to convert from ck enums to builder enums

/// @brief Converts a CK BlockGemmPipelineVersion enum to a builder PipelineVersion enum.
/// @tparam ck_ver The CK BlockGemmPipelineVersion enum value to convert.
/// @return The corresponding builder::PipelineVersion enum value (V1, V2, V3, V4, or V5).
/// @details This function maps CK's block GEMM pipeline version identifiers to the
/// builder framework's standardized pipeline version enum. The pipeline version
/// determines the strategy used for data movement and computation overlap in the
/// GEMM kernel's main loop.
template <ck::BlockGemmPipelineVersion ck_ver>
constexpr auto convert_pipeline_version()
{
    using enum ck::BlockGemmPipelineVersion;
    using enum builder::PipelineVersion;

    switch(ck_ver)
    {
    case v1: return V1;
    case v2: return V2;
    case v3: return V3;
    case v4: return V4;
    case v5: return V5;
    }
}

/// @brief Converts a CK PipelineVersion enum to a builder PipelineVersion enum.
/// @tparam ck_ver The CK PipelineVersion enum value to convert.
/// @return The corresponding builder::PipelineVersion enum value (V1, V2, V4, or WEIGHT_ONLY).
/// @details This function maps CK's general pipeline version identifiers to the
/// builder framework's standardized pipeline version enum. Note that this overload
/// handles a different set of pipeline versions compared to the BlockGemmPipelineVersion
/// variant, including support for specialized weight-only pipelines.
template <ck::PipelineVersion ck_ver>
constexpr auto convert_pipeline_version()
{
    using enum ck::PipelineVersion;
    using enum builder::PipelineVersion;

    switch(ck_ver)
    {
    case v1: return V1;
    case v2: return V2;
    case v4: return V4;
    case weight_only: return WEIGHT_ONLY;
    }
}

/// @brief Converts a CK BlockGemmPipelineScheduler enum to a builder PipelineScheduler enum.
/// @tparam ck_sched The CK BlockGemmPipelineScheduler enum value to convert.
/// @return The corresponding builder::PipelineScheduler enum value (INTRAWAVE or INTERWAVE).
/// @details This function maps CK's block GEMM pipeline scheduler identifiers to the
/// builder framework's standardized scheduler enum. The scheduler determines how work
/// is distributed and synchronized within and across wavefronts during pipeline execution.
/// INTRAWAVE scheduling operates within a single wavefront, while INTERWAVE coordinates
/// across multiple wavefronts.
template <ck::BlockGemmPipelineScheduler ck_sched>
constexpr auto convert_pipeline_scheduler()
{
    using enum ck::BlockGemmPipelineScheduler;
    using enum builder::PipelineScheduler;

    switch(ck_sched)
    {
    case Intrawave: return INTRAWAVE;
    case Interwave: return INTERWAVE;
    }
}

/// @brief Converts a CK LoopScheduler enum to a builder PipelineScheduler enum.
/// @tparam ck_sched The CK LoopScheduler enum value to convert.
/// @return The corresponding builder::PipelineScheduler enum value (DEFAULT or INTERWAVE).
/// @details This function maps CK's loop scheduler identifiers to the builder framework's
/// standardized pipeline scheduler enum. The loop scheduler controls how iterations of
/// the main computational loop are scheduled across threads. DEFAULT uses the standard
/// scheduling strategy, while INTERWAVE enables cross-wavefront coordination for improved
/// performance in certain scenarios.
template <ck::LoopScheduler ck_sched>
constexpr auto convert_pipeline_scheduler()
{
    using enum ck::LoopScheduler;
    using enum builder::PipelineScheduler;

    switch(ck_sched)
    {
    case Default: return DEFAULT;
    case Interwave: return INTERWAVE;
    }
}

// Helper metafunctions to derive signature information from Instance types

/// @brief Helper function to report unsupported convolution direction with a clear error message.
template <typename Instance>
[[noreturn]] consteval void report_unsupported_conv_direction_error()
{
    throw "Unsupported convolution direction detected!\n"
          "The kernel instance does not have a recognized convolution specialization.\n"
          "Expected one of: kConvForwardSpecialization, kConvBwdDataSpecialization, or "
          "kConvBwdWeightSpecialization.\n"
          "Please verify that your kernel instance is properly configured.";
}

/// @brief Derives the convolution direction from a device kernel `Instance` type.
/// @tparam Instance The device kernel instance type.
/// @return A `builder::ConvDirection` enum value (FORWARD, BACKWARD_DATA, or BACKWARD_WEIGHT).
template <typename Instance>
constexpr builder::ConvDirection conv_direction()
{
    using InstTraits = InstanceTraits<Instance>;

    if constexpr(requires { &InstTraits::kConvForwardSpecialization; })
        return builder::ConvDirection::FORWARD;
    else if constexpr(requires { &InstTraits::kConvBwdDataSpecialization; })
        return builder::ConvDirection::BACKWARD_DATA;
    else if constexpr(requires { &InstTraits::kConvBwdWeightSpecialization; })
        return builder::ConvDirection::BACKWARD_WEIGHT;
    else
    {
        report_unsupported_conv_direction_error<Instance>();
        return builder::ConvDirection::FORWARD; // Unreachable
    }
}

/// @brief Derives the convolution-specific specialization from a device kernel `Instance` type.
/// @tparam Instance The device kernel instance type.
/// @return A `builder::ConvFwdSpecialization`, `builder::ConvBwdDataSpecialization`, or
/// `builder::ConvBwdWeightSpecialization` enum value.
template <typename Instance>
constexpr auto conv_spec()
{
    using InstTraits = InstanceTraits<Instance>;

    if constexpr(requires { InstTraits::kConvForwardSpecialization; })
    {
        using enum ck::tensor_operation::device::ConvolutionForwardSpecialization;
        using enum builder::ConvFwdSpecialization;

        switch(InstTraits::kConvForwardSpecialization)
        {
        case Default: return DEFAULT;
        case Filter1x1Pad0: return FILTER_1X1_PAD0;
        case Filter1x1Stride1Pad0: return FILTER_1X1_STRIDE1_PAD0;
        case Filter3x3: return FILTER_3x3;
        case OddC: return ODD_C;
        }
    }
    else if constexpr(requires { InstTraits::kConvBwdDataSpecialization; })
    {
        using enum ck::tensor_operation::device::ConvolutionBackwardDataSpecialization;
        using enum builder::ConvBwdDataSpecialization;

        switch(InstTraits::kConvBwdDataSpecialization)
        {
        case Default: return DEFAULT;
        case Filter1x1Stride1Pad0: return FILTER_1X1_STRIDE1_PAD0;
        }
    }
    else if constexpr(requires { InstTraits::kConvBwdWeightSpecialization; })
    {
        using enum ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization;
        using enum builder::ConvBwdWeightSpecialization;

        switch(InstTraits::kConvBwdWeightSpecialization)
        {
        case Default: return DEFAULT;
        case Filter1x1Stride1Pad0: return FILTER_1X1_STRIDE1_PAD0;
        case Filter1x1Pad0: return FILTER_1X1_PAD0;
        case OddC: return ODD_C;
        }
    }
}

// Helper variable template to check if CK layout enums match
template <typename A,
          typename B,
          typename E,
          typename ExpectedA,
          typename ExpectedB,
          typename ExpectedE>
inline constexpr bool layouts_are =
    std::is_same_v<A, ExpectedA> && std::is_same_v<B, ExpectedB> && std::is_same_v<E, ExpectedE>;

/// @brief Helper function to report unsupported layout combinations with a clear error message.
/// @details This consteval function is designed to fail at compile time with a descriptive
/// error message when an unsupported layout combination is encountered.
template <typename A, typename B, typename E, int SpatialDim>
[[noreturn]] consteval void report_unsupported_layout_error()
{
    // This will produce a compile-time error with the exception message
    throw "Unsupported convolution layout combination detected!\n"
          "The combination of ALayout, BLayout, and ELayout template parameters\n"
          "is not recognized for the given spatial dimension.\n"
          "Please verify that your convolution instance uses a supported layout configuration.\n"
          "Check the conv_layout() function for the list of supported layout combinations.";
}

/// @brief Derives the grouped convolution layout from a device kernel `Instance` type.
/// @tparam Instance The device kernel instance type.
/// @return An std::array corresponding to the tensor layouts:
///             index 0 -> Input layout
///             index 1 -> Weight layout
///             index 2 -> Output layout
template <typename Instance>
constexpr auto conv_layout()
    requires HasFwdConvLayouts<InstanceTraits<Instance>>
{
    // Helper lambda to construct layout array
    auto layouts = [](auto... Ls) { return std::array<builder::TensorLayout, 3>{Ls...}; };

    using A       = typename InstanceTraits<Instance>::ALayout;
    using B       = typename InstanceTraits<Instance>::BLayout;
    using E       = typename InstanceTraits<Instance>::ELayout;
    namespace ctl = ck::tensor_layout::convolution;
    using enum builder::TensorLayout;

    switch(InstanceTraits<Instance>::kSpatialDim)
    {
    case 1:
        if constexpr(layouts_are<A, B, E, ctl::GNWC, ctl::GKXC, ctl::GNWK>)
            return layouts(GNWC, GKXC, GNWK);
        if constexpr(layouts_are<A, B, E, ctl::G_NW_C, ctl::G_K_X_C, ctl::G_NW_K>)
            return layouts(GNWC, GKXC, GNWK);
        if constexpr(layouts_are<A, B, E, ctl::NWGC, ctl::GKXC, ctl::NWGK>)
            return layouts(NWGC, GKXC, NWGK);
        if constexpr(layouts_are<A, B, E, ctl::NGCW, ctl::GKXC, ctl::NGKW>)
            return layouts(NGCW, GKXC, NGKW);
        if constexpr(layouts_are<A, B, E, ctl::NGCW, ctl::GKCX, ctl::NGKW>)
            return layouts(NGCW, GKCX, NGKW);
        break;
    case 2:
        if constexpr(layouts_are<A, B, E, ctl::GNHWC, ctl::GKYXC, ctl::GNHWK>)
            return layouts(GNHWC, GKYXC, GNHWK);
        if constexpr(layouts_are<A, B, E, ctl::G_NHW_C, ctl::G_K_YX_C, ctl::G_NHW_K>)
            return layouts(GNHWC, GKYXC, GNHWK);
        if constexpr(layouts_are<A, B, E, ctl::NHWGC, ctl::GKYXC, ctl::NHWGK>)
            return layouts(NHWGC, GKYXC, NHWGK);
        if constexpr(layouts_are<A, B, E, ctl::NHWGC, ctl::KYXGC, ctl::NHWGK>)
            return layouts(NHWGC, GKYXC, NHWGK);
        if constexpr(layouts_are<A, B, E, ctl::NGCHW, ctl::GKYXC, ctl::NGKHW>)
            return layouts(NGCHW, GKYXC, NGKHW);
        if constexpr(layouts_are<A, B, E, ctl::NGCHW, ctl::GKCYX, ctl::NGKHW>)
            return layouts(NGCHW, GKCYX, NGKHW);
        break;
    case 3:
        if constexpr(layouts_are<A, B, E, ctl::GNDHWC, ctl::GKZYXC, ctl::GNDHWK>)
            return layouts(GNDHWC, GKZYXC, GNDHWK);
        if constexpr(layouts_are<A, B, E, ctl::G_NDHW_C, ctl::G_K_ZYX_C, ctl::G_NDHW_K>)
            return layouts(GNDHWC, GKZYXC, GNDHWK);
        if constexpr(layouts_are<A, B, E, ctl::NDHWGC, ctl::GKZYXC, ctl::NDHWGK>)
            return layouts(NDHWGC, GKZYXC, NDHWGK);
        if constexpr(layouts_are<A, B, E, ctl::NGCDHW, ctl::GKZYXC, ctl::NGKDHW>)
            return layouts(NGCDHW, GKZYXC, NGKDHW);
        if constexpr(layouts_are<A, B, E, ctl::NGCDHW, ctl::GKCZYX, ctl::NGKDHW>)
            return layouts(NGCDHW, GKCZYX, NGKDHW);
        break;
    }

    // If we reach here, the layout combination is not supported
    // Call consteval function to trigger a compile-time error with a clear message
    report_unsupported_layout_error<A, B, E, InstanceTraits<Instance>::kSpatialDim>();

    // This return is unreachable but needed to satisfy the compiler
    return layouts(GNHWC, GKYXC, GNHWK);
}

/// @brief Helper function to report unsupported data type with a clear error message.
template <typename ADataType>
[[noreturn]] consteval void report_unsupported_data_type_error()
{
    throw "Unsupported data type detected!\n"
          "The ADataType is not recognized.\n"
          "Supported types are: ck::half_t (FP16), ck::Tuple<ck::half_t, ck::half_t> (FP16_FP16), "
          "ck::bhalf_t (BF16), ck::Tuple<ck::bhalf_t, ck::bhalf_t> (BF16_BF16), float (FP32), "
          "ck::Tuple<float, float> (FP32_FP32), double (FP64), ck::f8_t (FP8), ck::bf8_fnuz_t "
          "(BF8), "
          "int8_t (I8), ck::Tuple<int8_t, int8_t> (I8_I8), uint8_t (U8).\n"
          "Please verify that your kernel instance uses a supported data type.";
}

/// @brief Derives the data type from a device kernel `Instance` type.
/// Returns a `builder::DataType` enum value (e.g., FP16, BF16, FP32, BF8).
template <typename Instance>
constexpr builder::DataType conv_data_type()
    requires HasDataTypes<InstanceTraits<Instance>>
{
    using InstTraits = InstanceTraits<Instance>;
    using ADataType  = typename InstTraits::ADataType;
    using enum builder::DataType;

    if constexpr(std::is_same_v<ADataType, ck::half_t>)
        return FP16;
    else if constexpr(std::is_same_v<ADataType, ck::Tuple<ck::half_t, ck::half_t>>)
        return FP16_FP16;
    else if constexpr(std::is_same_v<ADataType, ck::bhalf_t>)
        return BF16;
    else if constexpr(std::is_same_v<ADataType, ck::Tuple<ck::bhalf_t, ck::bhalf_t>>)
        return BF16_BF16;
    else if constexpr(std::is_same_v<ADataType, float>)
        return FP32;
    else if constexpr(std::is_same_v<ADataType, ck::Tuple<float, float>>)
        return FP32_FP32;
    else if constexpr(std::is_same_v<ADataType, double>)
        return FP64;
    else if constexpr(std::is_same_v<ADataType, ck::f8_t>)
        return FP8;
    else if constexpr(std::is_same_v<ADataType, ck::bf8_fnuz_t>)
        return BF8;
    else if constexpr(std::is_same_v<ADataType, ck::bf8_ocp_t>)
        return BF8;
    else if constexpr(std::is_same_v<ADataType, int8_t>)
        return I8;
    else if constexpr(std::is_same_v<ADataType, ck::Tuple<int8_t, int8_t>>)
        return I8_I8;
    else if constexpr(std::is_same_v<ADataType, uint8_t>)
        return U8;
    else
    {
        report_unsupported_data_type_error<ADataType>();
        return FP32; // Unreachable
    }
}

/// @brief Helper function to report unsupported elementwise operation with a clear error message.
template <typename ElementwiseOp>
[[noreturn]] consteval void report_unsupported_elementwise_op_error()
{
    throw "Unsupported elementwise operation detected!\n"
          "The elementwise operation type is not recognized.\n"
          "Supported operations are: AddClamp, AddReluAdd, BiasBnormClamp, Bilinear, "
          "BiasNormalizeInInferClamp, Clamp, ConvInvscale, ConvScale, ConvScaleAdd, "
          "ConvScaleRelu, Scale, ScaleAdd, PassThrough, ScaleAddScaleAddRelu, DynamicUnaryOp, "
          "UnaryCombinedOp, Activation_Mul2_Clamp, Activation_Mul_Clamp, Add_Activation_Mul_Clamp, "
          "Add_Activation_Mul2_Clamp, Add_Mul_Activation_Mul_Clamp, Add_Mul2_Activation_Mul_Clamp, "
          "UnaryConvert.\n"
          "Please verify that your kernel instance uses a supported elementwise operation.";
}

/// @brief Derives the elementwise operation from op type.
/// @tparam ElementwiseOp Elementwise operation functor type.
/// @return A `builder::ElementwiseOperation` enum value corresponding to elementwise operation.
template <typename ElementwiseOp>
constexpr builder::ElementwiseOperation elementwise_op()
{
    using enum builder::ElementwiseOperation;
    constexpr std::string_view name = detail::elementwise_op_name<ElementwiseOp>();

    if constexpr(detail::case_insensitive_equal(name, "AddClamp"))
        return ADD_CLAMP;
    else if constexpr(detail::case_insensitive_equal(name, "AddReluAdd"))
        return ADD_RELU_ADD;
    else if constexpr(detail::case_insensitive_equal(name, "BiasBnormClamp"))
        return BIAS_BNORM_CLAMP;
    else if constexpr(detail::case_insensitive_equal(name, "Bilinear"))
        return BILINEAR;
    else if constexpr(detail::case_insensitive_equal(name, "BiasNormalizeInInferClamp"))
        return BIAS_BNORM_CLAMP;
    else if constexpr(detail::case_insensitive_equal(name, "Clamp"))
        return CLAMP;
    else if constexpr(detail::case_insensitive_equal(name, "ConvInvscale"))
        return CONV_INVSCALE;
    else if constexpr(detail::case_insensitive_equal(name, "ConvScale"))
        return CONV_SCALE;
    else if constexpr(detail::case_insensitive_equal(name, "ConvScaleAdd"))
        return CONV_SCALE_ADD;
    else if constexpr(detail::case_insensitive_equal(name, "ConvScaleRelu"))
        return CONV_SCALE_RELU;
    else if constexpr(detail::case_insensitive_equal(name, "Scale"))
        return SCALE;
    else if constexpr(detail::case_insensitive_equal(name, "ScaleAdd"))
        return SCALE_ADD;
    else if constexpr(detail::case_insensitive_equal(name, "PassThrough"))
        return PASS_THROUGH;
    else if constexpr(detail::case_insensitive_equal(name, "ScaleAddScaleAddRelu"))
        return SCALEADD_SCALEADD_RELU;
    else if constexpr(detail::case_insensitive_equal(name, "DynamicUnaryOp"))
        return DYNAMIC_UNARY_OP;
    else if constexpr(detail::case_insensitive_equal(name, "UnaryCombinedOp"))
        return UNARY_COMBINED_OP;
    else if constexpr(detail::case_insensitive_equal(name, "Activation_Mul2_Clamp"))
        return ACTIVATION_MUL2_CLAMP;
    else if constexpr(detail::case_insensitive_equal(name, "Activation_Mul_Clamp"))
        return ACTIVATION_MUL_CLAMP;
    else if constexpr(detail::case_insensitive_equal(name, "Add_Activation_Mul_Clamp"))
        return ADD_ACTIVATION_MUL_CLAMP;
    else if constexpr(detail::case_insensitive_equal(name, "Add_Activation_Mul2_Clamp"))
        return ADD_ACTIVATION_MUL2_CLAMP;
    else if constexpr(detail::case_insensitive_equal(name, "Add_Mul_Activation_Mul_Clamp"))
        return ADD_MUL_ACTIVATION_MUL_CLAMP;
    else if constexpr(detail::case_insensitive_equal(name, "Add_Mul2_Activation_Mul_Clamp"))
        return ADD_MUL2_ACTIVATION_MUL_CLAMP;
    else if constexpr(detail::case_insensitive_equal(name, "UnaryConvert"))
        return UNARY_CONVERT;
    else if constexpr(detail::case_insensitive_equal(name, "Logistic"))
        return LOGISTIC;
    else if constexpr(detail::case_insensitive_equal(name, "ClippedRelu"))
        return CLIPPED_RELU;
    else if constexpr(detail::case_insensitive_equal(name, "Swish"))
        return SWISH;
    else if constexpr(detail::case_insensitive_equal(name, "Elu"))
        return ELU;
    else if constexpr(detail::case_insensitive_equal(name, "Power"))
        return POWER;
    else if constexpr(detail::case_insensitive_equal(name, "LeakyRelu"))
        return LEAKY_RELU;
    else if constexpr(detail::case_insensitive_equal(name, "UnaryAbs"))
        return UNARY_ABS;
    else if constexpr(detail::case_insensitive_equal(name, "Relu"))
        return RELU;
    else if constexpr(detail::case_insensitive_equal(name, "SoftRelu"))
        return SOFT_RELU;
    else if constexpr(detail::case_insensitive_equal(name, "Sigmoid"))
        return SIGMOID;
    else if constexpr(detail::case_insensitive_equal(name, "TanH"))
        return TANH;
    else if constexpr(detail::case_insensitive_equal(name, "Gelu"))
        return GELU;
    else if constexpr(detail::case_insensitive_equal(name, "Silu"))
        return SILU;
    else
    {
        report_unsupported_elementwise_op_error<ElementwiseOp>();
        return PASS_THROUGH; // Unreachable
    }
}

/// @brief Derives a gemm padding from a kernel instance type.
/// @tparam Instance - A Device Kernel object type.
/// @return A `builder::GemmPadding` enum value corresponding to kernel padding.
template <typename Instance>
constexpr builder::GemmPadding gemm_spec()
    requires HasGemmSpec<InstanceTraits<Instance>>
{
    using InstTraits = InstanceTraits<Instance>;
    using enum builder::GemmPadding;
    using enum ck::tensor_operation::device::GemmSpecialization;

    constexpr auto gemm_spec = InstTraits::kGemmSpecialization;

    switch(gemm_spec)
    {
    case Default: return DEFAULT;
    case MPadding: return M_PADDING;
    case NPadding: return N_PADDING;
    case KPadding: return K_PADDING;
    case MNPadding: return MN_PADDING;
    case MKPadding: return MK_PADDING;
    case NKPadding: return NK_PADDING;
    case MNKPadding: return MNK_PADDING;
    case OPadding: return O_PADDING;
    case MOPadding: return MO_PADDING;
    case NOPadding: return NO_PADDING;
    case KOPadding: return KO_PADDING;
    case MNOPadding: return MNO_PADDING;
    case MKOPadding: return MKO_PADDING;
    case NKOPadding: return NKO_PADDING;
    case MNKOPadding: return MNKO_PADDING;
    }
}

/// @brief Primary template for extracting convolution traits.
/// @details This struct is the main entry point for reflecting on a convolution
/// kernel's properties. It is specialized to handle different kinds of input types.
template <typename T>
struct ConvTraits;

/// @brief Specialization of `ConvTraits` for a direct device kernel `Instance`.
/// @details This is the primary specialization used to extract a comprehensive
/// set of traits directly from a fully-formed device kernel `Instance` type.
/// It uses `InstanceTraits` to access the kernel's template parameters.
template <HasInstanceTraits Instance>
    requires IsXdlFwdConv<InstanceTraits<Instance>>
struct ConvTraits<Instance>
{
    using InstTraits = InstanceTraits<Instance>;

    // --- Signature Information ---
    /// @brief The number of spatial dimensions in the convolution (1, 2, or 3).
    static constexpr int spatial_dim = InstTraits::kSpatialDim;
    /// @brief The direction of the convolution (Forward, Backward Data, or Backward Weight).
    static constexpr builder::ConvDirection direction = conv_direction<Instance>();
    /// @brief The memory layout of the convolution tensors (e.g., GNHWC_GKYXC_GNHWK).
    static constexpr auto layout = conv_layout<Instance>();
    /// @brief The primary data type used in the computation (e.g., FP16, FP32).
    static constexpr builder::DataType data_type = conv_data_type<Instance>();

    static constexpr builder::ElementwiseOperation input_element_op =
        elementwise_op<typename InstTraits::AElementwiseOperation>();
    static constexpr builder::ElementwiseOperation weight_element_op =
        elementwise_op<typename InstTraits::BElementwiseOperation>();
    static constexpr builder::ElementwiseOperation output_element_op =
        elementwise_op<typename InstTraits::CDEElementwiseOperation>();

    /// @brief The GEMM specialization used by the kernel - padding
    static constexpr auto gemm_padding = gemm_spec<Instance>();
    /// @brief The convolution-specific specialization (e.g., Default, 1x1).
    static constexpr auto conv_specialization = conv_spec<Instance>();

    // --- Algorithm Information ---
    /// @brief The total number of threads in a thread block (workgroup).
    static constexpr int thread_block_size = InstTraits::kBlockSize;
    /// @brief The dimensions of the data tile processed by the thread block.
    static constexpr DataTileInfo tile_dims = {
        .m = InstTraits::kMPerBlock, .n = InstTraits::kNPerBlock, .k = InstTraits::kKPerBlock};

    /// @brief Configuration for the A-matrix (input) tile transfer.
    static constexpr InputTileTransferInfo a_tile_transfer = {
        .tile_dimensions = {.k0     = InstTraits::kKPerBlock / InstTraits::kAK1,
                            .m_or_n = InstTraits::kMPerBlock,
                            .k1     = InstTraits::kAK1},
        .transfer_params = {.k1                    = InstTraits::kAK1,
                            .thread_cluster_dims   = InstTraits::kAThreadClusterLengths,
                            .thread_cluster_order  = InstTraits::kAThreadClusterArrangeOrder,
                            .src_access_order      = InstTraits::kABlockTransferSrcAccessOrder,
                            .src_vector_dim        = InstTraits::kABlockTransferSrcVectorDim,
                            .src_scalar_per_vector = InstTraits::kABlockTransferSrcScalarPerVector,
                            .dst_scalar_per_vector_k1 =
                                InstTraits::kABlockTransferDstScalarPerVectorK1,
                            .lds_padding = static_cast<bool>(InstTraits::kABlockLdsExtraM)}};

    /// @brief Configuration for the B-matrix (weights) tile transfer.
    static constexpr InputTileTransferInfo b_tile_transfer = {
        .tile_dimensions = {.k0     = InstTraits::kKPerBlock / InstTraits::kBK1,
                            .m_or_n = InstTraits::kNPerBlock,
                            .k1     = InstTraits::kBK1},
        .transfer_params = {.k1                    = InstTraits::kBK1,
                            .thread_cluster_dims   = InstTraits::kBThreadClusterLengths,
                            .thread_cluster_order  = InstTraits::kBThreadClusterArrangeOrder,
                            .src_access_order      = InstTraits::kBBlockTransferSrcAccessOrder,
                            .src_vector_dim        = InstTraits::kBBlockTransferSrcVectorDim,
                            .src_scalar_per_vector = InstTraits::kBBlockTransferSrcScalarPerVector,
                            .dst_scalar_per_vector_k1 =
                                InstTraits::kBBlockTransferDstScalarPerVectorK1,
                            .lds_padding = static_cast<bool>(InstTraits::kBBlockLdsExtraN)}};

    /// @brief Parameters for the warp-level GEMM computation.
    static constexpr WarpGemmParams warp_gemm = {.gemm_m = InstTraits::kMPerXDL,
                                                 .gemm_n = InstTraits::kNPerXDL,
                                                 .m_iter = InstTraits::kMXdlPerWave,
                                                 .n_iter = InstTraits::kNXdlPerWave};

    /// @brief Configuration for the C-matrix (output) tile transfer.
    static constexpr OutputTileTransferInfo c_tile_transfer = {
        .shuffle_params      = {.m_gemms_per_shuffle = InstTraits::kCShuffleMXdlPerWavePerShuffle,
                                .n_gemms_per_shuffle = InstTraits::kCShuffleNXdlPerWavePerShuffle},
        .thread_cluster_dims = {InstTraits::kCThreadClusterLengths[0],
                                InstTraits::kCThreadClusterLengths[1],
                                InstTraits::kCThreadClusterLengths[2],
                                InstTraits::kCThreadClusterLengths[3]},
        .scalar_per_vector   = InstTraits::kCBlockTransferScalarPerVector};

    /// @brief Helper to safely get the pipeline version.
    /// @details This is only available for some convolutions (e.g., forward).
    /// If not present in `InstanceTraits`, it returns a default value.
    template <typename T = InstTraits>
    static constexpr auto get_pipeline_version()
    {
        if constexpr(requires { T::kPipelineVersion; })
        {
            return convert_pipeline_version<T::kPipelineVersion>();
        }
        else
        {
            // Return a default or indicate not available
            return builder::PipelineVersion::V1;
        }
    }

    /// @brief The block GEMM pipeline version used by the kernel.
    static constexpr auto pipeline_version = get_pipeline_version();

    /// @brief Helper to safely get the pipeline scheduler.
    /// @details This is only available for some convolutions. If not present
    /// in `InstanceTraits`, it returns a default value.
    template <typename T = InstTraits>
    static constexpr auto get_pipeline_scheduler()
    {
        if constexpr(requires { T::kPipelineScheduler; })
        {
            return convert_pipeline_scheduler<T::kPipelineScheduler>();
        }
        else if constexpr(requires { T::kLoopScheduler; })
        {
            return convert_pipeline_scheduler<T::kLoopScheduler>();
        }
        else
        {
            // Return a default or indicate not available
            return builder::PipelineScheduler::DEFAULT;
        }
    }

    /// @brief The pipeline scheduler used by the kernel.
    static constexpr auto pipeline_scheduler = get_pipeline_scheduler();
};

} // namespace ck_tile::reflect::conv
