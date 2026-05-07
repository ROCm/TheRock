// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "max_pool_bwd_instance_common.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_maxpool_bwd_f32_instances(
    std::vector<std::unique_ptr<DeviceMaxPoolBwd<F32, I32, F32>>>& instances)
{
    add_device_operation_instances(instances, device_maxpool_bwd_instances<F32, I32, F32>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
