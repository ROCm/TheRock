// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>
#include <string_view>
#include <variant>
#include <bit>
#include <array>

namespace ck_tile::builder {

// TODO: Handle tuple types and FP8/BF8 properly
enum class DataType
{
    UNDEFINED_DATA_TYPE = 0,
    FP32,
    FP32_FP32,
    FP16,
    FP16_FP16,
    BF16,
    BF16_BF16,
    FP8,
    BF8,
    FP64,
    I32,
    I8,
    I8_I8,
    U8
};

enum class TensorLayout
{
    UNDEFINED_TENSOR_LAYOUT = 0,

    // Bias tensors
    GC,
    G_C_strided,
    G_K_strided,

    // 1D conv input tensor
    GNCW,
    GNWC,
    NWGC,
    NGCW,
    G_NW_C_strided,

    // 2D conv input tensor
    GNCHW,
    GNHWC,
    NHWGC,
    NGCHW,
    G_NHW_C_strided,

    // 3D conv input tensor
    GNCDHW,
    GNDHWC,
    NDHWGC,
    NGCDHW,
    G_NDHW_C_strided,

    // 1D conv weight tensor
    GKXC,
    GKCX,
    KXGC,
    G_K_X_C_strided,

    // 2D conv weight tensor
    GKYXC,
    GKCYX,
    KYXGC,
    G_K_YX_C_strided,

    // 3D conv weight tensor
    GKZYXC,
    GKCZYX,
    KZYXGC,
    G_K_ZYX_C_strided,

    // 1D conv output tensor
    GNKW,
    GNWK,
    NWGK,
    NGKW,
    G_NW_K_strided,

    // 2D conv output tensor
    GNKHW,
    GNHWK,
    NHWGK,
    NGKHW,
    G_NHW_K_strided,

    // 3D conv output tensor
    GNKDHW,
    GNDHWK,
    NDHWGK,
    NGKDHW,
    G_NDHW_K_strided
};

// Direction of the convolution operation.
enum class ConvDirection
{
    FORWARD,
    BACKWARD_DATA,
    BACKWARD_WEIGHT
};

// Fused element-wise operations.
// TODO: Generalize design rather than enumerating all possible ops.
enum class ElementwiseOperation
{
    ADD_CLAMP,
    ADD_RELU_ADD,
    ACTIVATION_MUL2_CLAMP,
    ACTIVATION_MUL_CLAMP,
    ADD_ACTIVATION_MUL_CLAMP,
    ADD_ACTIVATION_MUL2_CLAMP,
    ADD_MUL_ACTIVATION_MUL_CLAMP,
    ADD_MUL2_ACTIVATION_MUL_CLAMP,
    BIAS_BNORM_CLAMP,
    BILINEAR,
    SCALE,
    SCALE_ADD,
    CLAMP,
    CONV_INVSCALE,
    CONV_SCALE,
    CONV_SCALE_ADD,
    CONV_SCALE_RELU,
    PASS_THROUGH,
    SCALEADD_SCALEADD_RELU,
    DYNAMIC_UNARY_OP,
    UNARY_COMBINED_OP,
    UNARY_CONVERT,
    LOGISTIC,
    CLIPPED_RELU,
    SWISH,
    ELU,
    POWER,
    LEAKY_RELU,
    UNARY_ABS,
    RELU,
    SOFT_RELU,
    SIGMOID,
    TANH,
    GELU,
    SILU
};

// Enums for pipeline versions & schedulers
enum class PipelineVersion
{
    V1,
    V2,
    V3,
    V4,
    V5,
    WEIGHT_ONLY
};

// Enums for the GEMM specialization.
enum struct GemmSpecialization
{
    // Gemm
    Default,
    MPadding,
    NPadding,
    KPadding,
    MNPadding,
    MKPadding,
    NKPadding,
    MNKPadding,
    // Gemm + Gemm
    OPadding,
    MOPadding,
    NOPadding,
    KOPadding,
    MNOPadding,
    MKOPadding,
    NKOPadding,
    MNKOPadding
};

// Enums for the CK Tile convolution specialization.
enum class TileConvSpecialization
{
    DEFAULT,
    FILTER_1X1_PAD0,
    FILTER_1X1_STRIDE1_PAD0,
    FILTER_3x3
};

// Enums for the forward convolution specialization.
enum class ConvFwdSpecialization
{
    DEFAULT,
    FILTER_1X1_PAD0,
    FILTER_1X1_STRIDE1_PAD0,
    FILTER_3x3,
    ODD_C
};

// Enums for the backward data convolution specialization.
enum class ConvBwdDataSpecialization
{
    DEFAULT,
    FILTER_1X1_STRIDE1_PAD0,
};

// Enums for the backward weight convolution specialization.
enum class ConvBwdWeightSpecialization
{
    DEFAULT,
    FILTER_1X1_STRIDE1_PAD0,
    FILTER_1X1_PAD0,
    ODD_C,
};

// Enums for the Gemm padding.
enum class GemmPadding
{
    DEFAULT,
    M_PADDING,
    N_PADDING,
    K_PADDING,
    MN_PADDING,
    MK_PADDING,
    NK_PADDING,
    MNK_PADDING,
    O_PADDING,
    MO_PADDING,
    NO_PADDING,
    KO_PADDING,
    MNO_PADDING,
    MKO_PADDING,
    NKO_PADDING,
    MNKO_PADDING,
};

enum class PipelineScheduler
{
    DEFAULT,
    INTRAWAVE,
    INTERWAVE
};

enum class ConvAlgorithmSpecialization
{
    LARGE_TENSOR,
    REFERENCE // GPU reference implementation for validation
};

// to_string methods for enum classes
inline std::string_view to_string(DataType dt)
{
    using enum DataType;
    switch(dt)
    {
    case FP16: return "FP16";
    case FP16_FP16: return "FP16_FP16";
    case FP32: return "FP32";
    case FP32_FP32: return "FP32_FP32";
    case BF16: return "BF16";
    case BF16_BF16: return "BF16_BF16";
    case FP8: return "FP8";
    case BF8: return "BF8";
    case FP64: return "FP64";
    case I32: return "I32";
    case I8: return "I8";
    case I8_I8: return "I8_I8";
    case U8: return "U8";
    case UNDEFINED_DATA_TYPE: return "UNDEFINED_DATA_TYPE";
    default: return "Unknown";
    }
}

inline std::string_view to_string(ConvDirection dir)
{
    using enum ConvDirection;
    switch(dir)
    {
    case FORWARD: return "Forward";
    case BACKWARD_DATA: return "Backward Data";
    case BACKWARD_WEIGHT: return "Backward Weight";
    default: return "Unknown";
    }
}

inline std::string_view to_string(ElementwiseOperation op)
{
    using enum ElementwiseOperation;
    switch(op)
    {
    case ADD_CLAMP: return "ADD_CLAMP";
    case ADD_RELU_ADD: return "ADD_RELU_ADD";
    case ACTIVATION_MUL2_CLAMP: return "ACTIVATION_MUL2_CLAMP";
    case ACTIVATION_MUL_CLAMP: return "ACTIVATION_MUL_CLAMP";
    case ADD_ACTIVATION_MUL_CLAMP: return "ADD_ACTIVATION_MUL_CLAMP";
    case ADD_ACTIVATION_MUL2_CLAMP: return "ADD_ACTIVATION_MUL2_CLAMP";
    case ADD_MUL_ACTIVATION_MUL_CLAMP: return "ADD_MUL_ACTIVATION_MUL_CLAMP";
    case ADD_MUL2_ACTIVATION_MUL_CLAMP: return "ADD_MUL2_ACTIVATION_MUL_CLAMP";
    case BIAS_BNORM_CLAMP: return "BIAS_BNORM_CLAMP";
    case BILINEAR: return "BILINEAR";
    case CLAMP: return "CLAMP";
    case SCALE: return "SCALE";
    case SCALE_ADD: return "SCALE_ADD";
    case CONV_INVSCALE: return "CONV_INVSCALE";
    case CONV_SCALE: return "CONV_SCALE";
    case CONV_SCALE_ADD: return "CONV_SCALE_ADD";
    case CONV_SCALE_RELU: return "CONV_SCALE_RELU";
    case PASS_THROUGH: return "PASS_THROUGH";
    case SCALEADD_SCALEADD_RELU: return "SCALEADD_SCALEADD_RELU";
    case DYNAMIC_UNARY_OP: return "DYNAMIC_UNARY_OP";
    case UNARY_COMBINED_OP: return "UNARY_COMBINED_OP";
    case UNARY_CONVERT: return "UNARY_CONVERT";
    case LOGISTIC: return "LOGISTIC";
    case CLIPPED_RELU: return "CLIPPED_RELU";
    case SWISH: return "SWISH";
    case ELU: return "ELU";
    case POWER: return "POWER";
    case LEAKY_RELU: return "LEAKY_RELU";
    case UNARY_ABS: return "UNARY_ABS";
    case RELU: return "RELU";
    case SOFT_RELU: return "SOFT_RELU";
    case SIGMOID: return "SIGMOID";
    case TANH: return "TANH";
    case GELU: return "GELU";
    case SILU: return "SILU";
    default: return "Unknown";
    }
}

inline std::string_view to_string(PipelineVersion ver)
{
    using enum PipelineVersion;
    switch(ver)
    {
    case V1: return "V1";
    case V2: return "V2";
    case V3: return "V3";
    case V4: return "V4";
    case V5: return "V5";
    case WEIGHT_ONLY: return "WEIGHT_ONLY";
    default: return "Unknown";
    }
}

inline std::string_view to_string(GemmSpecialization spec)
{
    using enum GemmSpecialization;
    switch(spec)
    {
    case Default: return "Default";
    case MPadding: return "MPadding";
    case NPadding: return "NPadding";
    case KPadding: return "KPadding";
    case MNPadding: return "MNPadding";
    case MKPadding: return "MKPadding";
    case NKPadding: return "NKPadding";
    case MNKPadding: return "MNKPadding";
    case OPadding: return "OPadding";
    case MOPadding: return "MOPadding";
    case NOPadding: return "NOPadding";
    case KOPadding: return "KOPadding";
    case MNOPadding: return "MNOPadding";
    case MKOPadding: return "MKOPadding";
    case NKOPadding: return "NKOPadding";
    case MNKOPadding: return "MNKOPadding";
    default: return "Unknown";
    }
}

inline std::string_view to_string(ConvFwdSpecialization spec)
{
    using enum ConvFwdSpecialization;
    switch(spec)
    {
    case DEFAULT: return "DEFAULT";
    case FILTER_1X1_PAD0: return "FILTER_1X1_PAD0";
    case FILTER_1X1_STRIDE1_PAD0: return "FILTER_1X1_STRIDE1_PAD0";
    case FILTER_3x3: return "FILTER_3x3";
    case ODD_C: return "ODD_C";
    default: return "Unknown";
    }
}

inline std::string_view to_string(ConvBwdDataSpecialization spec)
{
    using enum ConvBwdDataSpecialization;
    switch(spec)
    {
    case DEFAULT: return "DEFAULT";
    case FILTER_1X1_STRIDE1_PAD0: return "FILTER_1X1_STRIDE1_PAD0";
    default: return "Unknown";
    }
}

inline std::string_view to_string(ConvBwdWeightSpecialization spec)
{
    using enum ConvBwdWeightSpecialization;
    switch(spec)
    {
    case DEFAULT: return "DEFAULT";
    case FILTER_1X1_STRIDE1_PAD0: return "FILTER_1X1_STRIDE1_PAD0";
    case FILTER_1X1_PAD0: return "FILTER_1X1_PAD0";
    case ODD_C: return "ODD_C";
    default: return "Unknown";
    }
}

inline std::string_view to_string(GemmPadding padding)
{
    using enum GemmPadding;
    switch(padding)
    {
    case DEFAULT: return "DEFAULT";
    case M_PADDING: return "M_PADDING";
    case N_PADDING: return "N_PADDING";
    case K_PADDING: return "K_PADDING";
    case MN_PADDING: return "MN_PADDING";
    case MK_PADDING: return "MK_PADDING";
    case NK_PADDING: return "NK_PADDING";
    case MNK_PADDING: return "MNK_PADDING";
    case O_PADDING: return "O_PADDING";
    case MO_PADDING: return "MO_PADDING";
    case NO_PADDING: return "NO_PADDING";
    case KO_PADDING: return "KO_PADDING";
    case MNO_PADDING: return "MNO_PADDING";
    case MKO_PADDING: return "MKO_PADDING";
    case NKO_PADDING: return "NKO_PADDING";
    case MNKO_PADDING: return "MNKO_PADDING";
    default: return "Unknown";
    }
}

inline std::string_view to_string(PipelineScheduler sched)
{
    using enum PipelineScheduler;
    switch(sched)
    {
    case DEFAULT: return "DEFAULT";
    case INTRAWAVE: return "INTRAWAVE";
    case INTERWAVE: return "INTERWAVE";
    default: return "Unknown";
    }
}

inline std::string_view to_string(TensorLayout layout)
{
    using enum TensorLayout;
    switch(layout)
    {
    case GNCW: return "GNCW";
    case GNWC: return "GNWC";
    case NWGC: return "NWGC";
    case NGCW: return "NGCW";
    case G_NW_C_strided: return "G_NW_C_strided";
    case GNCHW: return "GNCHW";
    case GNHWC: return "GNHWC";
    case NHWGC: return "NHWGC";
    case NGCHW: return "NGCHW";
    case G_NHW_C_strided: return "G_NHW_C_strided";
    case GNCDHW: return "GNCDHW";
    case GNDHWC: return "GNDHWC";
    case NDHWGC: return "NDHWGC";
    case NGCDHW: return "NGCDHW";
    case G_NDHW_C_strided: return "G_NDHW_C_strided";
    case GKXC: return "GKXC";
    case GKCX: return "GKCX";
    case KXGC: return "KXGC";
    case G_K_X_C_strided: return "G_K_X_C_strided";
    case GKYXC: return "GKYXC";
    case GKCYX: return "GKCYX";
    case KYXGC: return "KYXGC";
    case G_K_YX_C_strided: return "G_K_YX_C_strided";
    case GKZYXC: return "GKZYXC";
    case GKCZYX: return "GKCZYX";
    case KZYXGC: return "KZYXGC";
    case G_K_ZYX_C_strided: return "G_K_ZYX_C_strided";
    case GNKW: return "GNKW";
    case GNWK: return "GNWK";
    case NWGK: return "NWGK";
    case NGKW: return "NGKW";
    case G_NW_K_strided: return "G_NW_K_strided";
    case GNKHW: return "GNKHW";
    case GNHWK: return "GNHWK";
    case NHWGK: return "NHWGK";
    case NGKHW: return "NGKHW";
    case G_NHW_K_strided: return "G_NHW_K_strided";
    case GNKDHW: return "GNKDHW";
    case GNDHWK: return "GNDHWK";
    case NDHWGK: return "NDHWGK";
    case NGKDHW: return "NGKDHW";
    case G_NDHW_K_strided: return "G_NDHW_K_strided";
    case GC: return "GC";
    case G_C_strided: return "G_C_strided";
    case G_K_strided: return "G_K_strided";
    case UNDEFINED_TENSOR_LAYOUT: return "UNDEFINED_TENSOR_LAYOUT";
    default: return "Unknown";
    }
}

// ostream operator overloads for enum classes
inline std::ostream& operator<<(std::ostream& os, DataType dt) { return os << to_string(dt); }

inline std::ostream& operator<<(std::ostream& os, ConvDirection dir)
{
    return os << to_string(dir);
}

inline std::ostream& operator<<(std::ostream& os, ElementwiseOperation op)
{
    return os << to_string(op);
}

inline std::ostream& operator<<(std::ostream& os, PipelineVersion ver)
{
    return os << to_string(ver);
}

inline std::ostream& operator<<(std::ostream& os, GemmSpecialization spec)
{
    return os << to_string(spec);
}

inline std::ostream& operator<<(std::ostream& os, ConvFwdSpecialization spec)
{
    return os << to_string(spec);
}

inline std::ostream& operator<<(std::ostream& os, ConvBwdDataSpecialization spec)
{
    return os << to_string(spec);
}

inline std::ostream& operator<<(std::ostream& os, ConvBwdWeightSpecialization spec)
{
    return os << to_string(spec);
}

inline std::ostream& operator<<(std::ostream& os, GemmPadding padding)
{
    return os << to_string(padding);
}

inline std::ostream& operator<<(std::ostream& os, PipelineScheduler sched)
{
    return os << to_string(sched);
}

inline std::ostream& operator<<(std::ostream& os, TensorLayout layout)
{
    return os << to_string(layout);
}

// ostream operator overload for std::variant of convolution specializations
inline std::ostream& operator<<(std::ostream& os,
                                const std::variant<ConvFwdSpecialization,
                                                   ConvBwdDataSpecialization,
                                                   ConvBwdWeightSpecialization>& spec)
{
    std::visit([&os](const auto& s) { os << s; }, spec);
    return os;
}

} // namespace ck_tile::builder
