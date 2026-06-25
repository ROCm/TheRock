// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using Empty_Tuple   = ck::Tuple<>;
using Row_Row_Tuple = ck::Tuple<Row, Row>;
using Col_Col_Tuple = ck::Tuple<Col, Col>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Relu        = ck::tensor_operation::element_wise::Relu;

using Mul_Clamp      = ck::tensor_operation::element_wise::Activation_Mul_Clamp<PassThrough>;
using Relu_Mul_Clamp = ck::tensor_operation::element_wise::Activation_Mul_Clamp<Relu>;

using Add_Mul_Clamp = ck::tensor_operation::element_wise::Add_Activation_Mul_Clamp<PassThrough>;
using Add_Relu_Mul_Clamp = ck::tensor_operation::element_wise::Add_Activation_Mul_Clamp<Relu>;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;
static constexpr auto MNKPadding  = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
