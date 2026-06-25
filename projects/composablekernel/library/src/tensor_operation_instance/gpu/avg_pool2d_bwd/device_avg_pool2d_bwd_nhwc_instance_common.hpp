// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_avgpool2d_bwd_nhwc_nhwc.hpp"
#include "ck/utility/data_type.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F16  = ck::half_t;
using BF16 = ck::bhalf_t;
using F8   = ck::f8_t;
using I8   = int8_t;
using I32  = int32_t;
using F32  = float;
using NHWC = ck::tensor_layout::convolution::NHWC;

template <typename OutType, typename InType, typename ComputeType>
using device_avgpool_2D_bwd_nhwc_instances = std::tuple<
    // clang-format off
        DeviceAvgPool2dBwd_NHWC_NHWC<OutType, InType, ComputeType, 256, 256, 1, 1, 1, 1>,
        DeviceAvgPool2dBwd_NHWC_NHWC<OutType, InType, ComputeType, 256, 256, 1, 2, 2, 2>,
        DeviceAvgPool2dBwd_NHWC_NHWC<OutType, InType, ComputeType, 256, 256, 1, 4, 4, 4>,
        DeviceAvgPool2dBwd_NHWC_NHWC<OutType, InType, ComputeType, 256, 256, 1, 8, 8, 8>,
        DeviceAvgPool2dBwd_NHWC_NHWC<OutType, InType, ComputeType, 256, 32, 8, 8, 8, 8>
    // clang-format on
    >;
} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
