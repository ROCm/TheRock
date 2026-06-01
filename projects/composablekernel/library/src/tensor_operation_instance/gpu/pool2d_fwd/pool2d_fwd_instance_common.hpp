// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_pool2d_fwd_nhwc_nhwc.hpp"
#include "ck/utility/data_type.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using I32  = int32_t;
using F32  = float;
using F16  = ck::half_t;
using BF16 = ck::bhalf_t;
using I8   = int8_t;
using F8   = ck::f8_t;
using NHWC = ck::tensor_layout::convolution::NHWC;

template <typename InDataType,
          typename OutDataType,
          typename IndexDataType,
          typename ComputeDataType,
          ReduceTensorOp ReduceOpId,
          bool OutputIndex>
using device_pool2d_fwd_nhwc_instances =
    // clang-format off
    std::tuple <
        DevicePool2dFwd_NHWC_NHWC<InDataType, OutDataType, IndexDataType, ComputeDataType, ReduceOpId, OutputIndex, 256, 256, 1, 1, 1, 1>,
        DevicePool2dFwd_NHWC_NHWC<InDataType, OutDataType, IndexDataType, ComputeDataType, ReduceOpId, OutputIndex, 256, 256, 1, 2, 1, 2>,
        DevicePool2dFwd_NHWC_NHWC<InDataType, OutDataType, IndexDataType, ComputeDataType, ReduceOpId, OutputIndex, 256, 256, 1, 4, 1, 4>
               // clang-format on
               >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
