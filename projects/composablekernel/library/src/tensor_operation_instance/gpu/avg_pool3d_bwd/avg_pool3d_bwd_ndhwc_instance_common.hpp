// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_avgpool3d_bwd_ndhwc_ndhwc.hpp"
#include "ck/utility/data_type.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using I32   = int32_t;
using F16   = ck::half_t;
using BF16  = ck::bhalf_t;
using F32   = float;
using NDHWC = ck::tensor_layout::convolution::NDHWC;

using device_avgpool_bwd_ndhwc_f16_instances =
    // clang-format off
    std::tuple <
        DeviceAvgPool3dBwd_NDHWC_NDHWC<F16, F16, F32, 256, 256, 1, 1, 1, 1>,
        DeviceAvgPool3dBwd_NDHWC_NDHWC<F16, F16, F32, 256, 256, 1, 2, 2, 2>,
        DeviceAvgPool3dBwd_NDHWC_NDHWC<F16, F16, F32, 256, 256, 1, 4, 4, 4>,
        DeviceAvgPool3dBwd_NDHWC_NDHWC<F16, F16, F32, 256, 256, 1, 8, 8, 8>,
        DeviceAvgPool3dBwd_NDHWC_NDHWC<F16, F16, F32, 256, 32, 8, 8, 8, 8>
               // clang-format on
               >;

using device_avgpool_bwd_ndhwc_bf16_instances =
    // clang-format off
    std::tuple <
        DeviceAvgPool3dBwd_NDHWC_NDHWC<BF16, BF16, F32, 256, 256, 1, 1, 1, 1>,
        DeviceAvgPool3dBwd_NDHWC_NDHWC<BF16, BF16, F32, 256, 256, 1, 2, 2, 2>,
        DeviceAvgPool3dBwd_NDHWC_NDHWC<BF16, BF16, F32, 256, 256, 1, 4, 4, 4>,
        DeviceAvgPool3dBwd_NDHWC_NDHWC<BF16, BF16, F32, 256, 256, 1, 8, 8, 8>,
        DeviceAvgPool3dBwd_NDHWC_NDHWC<BF16, BF16, F32, 256, 32, 8, 8, 8, 8>
               // clang-format on
               >;

using device_avgpool_bwd_ndhwc_f32_instances =
    // clang-format off
    std::tuple <
        DeviceAvgPool3dBwd_NDHWC_NDHWC<F32, F32, F32, 256, 256, 1, 1, 1, 1>,
        DeviceAvgPool3dBwd_NDHWC_NDHWC<F32, F32, F32, 256, 256, 1, 2, 2, 2>,
        DeviceAvgPool3dBwd_NDHWC_NDHWC<F32, F32, F32, 256, 256, 1, 4, 4, 4>,
        DeviceAvgPool3dBwd_NDHWC_NDHWC<F32, F32, F32, 256, 256, 1, 8, 8, 8>,
        DeviceAvgPool3dBwd_NDHWC_NDHWC<F32, F32, F32, 256, 32, 8, 8, 8, 8>
               // clang-format on
               >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
