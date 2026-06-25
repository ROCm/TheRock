// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "pool2d_fwd_instance_common.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

static constexpr auto ReduceOpId = ck::ReduceTensorOp::MAX;

void add_device_pool2d_fwd_nhwc_i8_instances(
    std::vector<std::unique_ptr<DevicePoolFwd<4, 2, I8, I8, I32, NHWC, NHWC, ReduceOpId, false>>>&
        instances)
{
    add_device_operation_instances(
        instances, device_pool2d_fwd_nhwc_instances<I8, I8, I32, F32, ReduceOpId, false>{});
}

void add_device_pool2d_fwd_nhwc_index_i8_instances(
    std::vector<std::unique_ptr<DevicePoolFwd<4, 2, I8, I8, I32, NHWC, NHWC, ReduceOpId, true>>>&
        instances)
{
    add_device_operation_instances(
        instances, device_pool2d_fwd_nhwc_instances<I8, I8, I32, F32, ReduceOpId, true>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
