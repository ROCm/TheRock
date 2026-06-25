// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "max_pool_bwd_instance_common.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_maxpool_bwd_f8_instances(
    std::vector<std::unique_ptr<DeviceMaxPoolBwd<F8, I32, F8>>>& instances)
{
    add_device_operation_instances(instances, device_maxpool_bwd_instances<F8, I32, F8>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
