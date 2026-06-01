// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/gpu/permute_scale/device_permute_scale_instances.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using Scale = element_wise::Scale;

void add_device_permute_scale_2d_f32_instances(
    std::vector<std::unique_ptr<DeviceElementwise<ck::Tuple<F32>, ck::Tuple<F32>, Scale, 2>>>&
        instances)
{
    add_device_operation_instances(instances, device_permute_scale_f32_instances<2, Scale>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
