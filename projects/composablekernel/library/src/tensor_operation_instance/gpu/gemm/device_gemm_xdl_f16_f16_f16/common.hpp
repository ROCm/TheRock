// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/utility/data_type.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto GemmDefault   = ck::tensor_operation::device::GemmSpecialization::Default;
static constexpr auto GemmMNPadding = ck::tensor_operation::device::GemmSpecialization::MNPadding;

using InstanceNT = DeviceGemm<Col, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>;
using InstanceNN = DeviceGemm<Col, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>;
using InstanceTT = DeviceGemm<Row, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>;
using InstanceTN = DeviceGemm<Row, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>;

template <typename Instance>
using OwnerList = std::vector<std::unique_ptr<Instance>>;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
