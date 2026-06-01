// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "max_pool_bwd_instance_common.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_maxpool_bwd_bf16_instances(
    std::vector<std::unique_ptr<DeviceMaxPoolBwd<BF16, I32, BF16>>>& instances)
{
    add_device_operation_instances(instances, device_maxpool_bwd_instances<BF16, I32, BF16>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
