// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "normalization_bwd_data_instance_common.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_groupnorm_bwd_data_f32_instances(
    std::vector<std::unique_ptr<DeviceNormalizationBwdData<F32, F32, F32, F32, F32, 5, 3>>>&
        instances)
{
    add_device_operation_instances(instances, device_groupnorm_bwd_data_f32_generic_instance{});
    add_device_operation_instances(instances, device_groupnorm_bwd_data_f32_instances{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
