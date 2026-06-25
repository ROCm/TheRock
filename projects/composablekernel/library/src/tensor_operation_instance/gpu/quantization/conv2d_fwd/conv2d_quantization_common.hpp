// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using Empty_Tuple = ck::Tuple<>;
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using NHWGC       = ck::tensor_layout::convolution::NHWGC;
using GKYXC       = ck::tensor_layout::convolution::GKYXC;
using NHWGK       = ck::tensor_layout::convolution::NHWGK;
using G_K         = ck::tensor_layout::convolution::G_K;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Relu        = ck::tensor_operation::element_wise::Relu;
using TanH        = ck::tensor_operation::element_wise::TanH;

using GK_Tuple      = ck::Tuple<G_K>;
using GK_GK_Tuple   = ck::Tuple<G_K, G_K>;
using I32_Tuple     = ck::Tuple<int32_t>;
using F32_Tuple     = ck::Tuple<float>;
using I32_F32_Tuple = ck::Tuple<int32_t, float>;

// perlayer
using Mul_Clamp      = ck::tensor_operation::element_wise::Activation_Mul_Clamp<PassThrough>;
using Relu_Mul_Clamp = ck::tensor_operation::element_wise::Activation_Mul_Clamp<Relu>;

// bias + perlayer
using Add_Mul_Clamp = ck::tensor_operation::element_wise::Add_Activation_Mul_Clamp<PassThrough>;
using Add_Relu_Mul_Clamp = ck::tensor_operation::element_wise::Add_Activation_Mul_Clamp<Relu>;
using Add_Mul_TanH_Mul_Clamp =
    ck::tensor_operation::element_wise::Add_Mul_Activation_Mul_Clamp<TanH>;

// perchannel
using Mul2_Clamp      = ck::tensor_operation::element_wise::Activation_Mul2_Clamp<PassThrough>;
using Relu_Mul2_Clamp = ck::tensor_operation::element_wise::Activation_Mul2_Clamp<Relu>;

// bias + perchannel
using Add_Mul2_Clamp = ck::tensor_operation::element_wise::Add_Activation_Mul2_Clamp<PassThrough>;
using Add_Relu_Mul2_Clamp = ck::tensor_operation::element_wise::Add_Activation_Mul2_Clamp<Relu>;
using Add_Mul2_TanH_Mul_Clamp =
    ck::tensor_operation::element_wise::Add_Mul2_Activation_Mul_Clamp<TanH>;

static constexpr ck::index_t NDimSpatial = 2;
static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;
static constexpr auto ConvFwdDefault =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;
static constexpr auto ConvFwd1x1P0 =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Pad0;
static constexpr auto ConvFwd1x1S1P0 =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
