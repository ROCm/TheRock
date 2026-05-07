// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_pool2d_fwd_nhwc_nhwc.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_pool3d_fwd_ndhwc_ndhwc.hpp"
#include "ck/utility/data_type.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using I8    = int8_t;
using F8    = ck::f8_t;
using I32   = int32_t;
using F16   = ck::half_t;
using BF16  = ck::bhalf_t;
using F32   = float;
using NDHWC = ck::tensor_layout::convolution::NDHWC;

template <typename InDataType,
          typename OutDataType,
          typename IndexDataType,
          typename ComputeDataType,
          ReduceTensorOp ReduceOpId,
          bool OutputIndex>
using device_pool3d_fwd_ndhwc_instances =
    // clang-format off
    std::tuple <
        DevicePool3dFwd_NDHWC_NDHWC<InDataType, OutDataType, IndexDataType, ComputeDataType, ReduceOpId, OutputIndex, 256, 256, 1, 1, 1, 1>,
        DevicePool3dFwd_NDHWC_NDHWC<InDataType, OutDataType, IndexDataType, ComputeDataType, ReduceOpId, OutputIndex, 256, 256, 1, 2, 1, 2>,
        DevicePool3dFwd_NDHWC_NDHWC<InDataType, OutDataType, IndexDataType, ComputeDataType, ReduceOpId, OutputIndex, 256, 256, 1, 4, 1, 4>
               // clang-format on
               >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
