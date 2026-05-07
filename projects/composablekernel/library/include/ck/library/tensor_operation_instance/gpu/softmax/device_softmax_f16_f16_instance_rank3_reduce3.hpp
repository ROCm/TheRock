// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"
#include "ck/tensor_operation/gpu/device/device_softmax.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_softmax_f16_f16_rank3_reduce3_instances(
    std::vector<DeviceSoftmaxPtr<F16, F32, F16, PassThrough, PassThrough, 3, 3>>& instances);

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
