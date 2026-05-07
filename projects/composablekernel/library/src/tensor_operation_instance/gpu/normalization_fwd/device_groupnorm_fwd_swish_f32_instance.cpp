// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "normalization_fwd_instance_common.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using Swish = ck::tensor_operation::element_wise::Swish;

void add_device_normalization_fwd_rank_5_3_swish_f32_instances(
    std::vector<std::unique_ptr<DeviceNormalizationFwd<F32, F32, F32, F32, F32, Swish, 5, 3>>>&
        instances)
{
    add_device_operation_instances(instances,
                                   device_normalization_f32_generic_instance<Swish, 5, 3>{});
    add_device_operation_instances(instances, device_normalization_f32_instances<Swish, 5, 3>{});
    add_device_operation_instances(instances,
                                   device_normalization_splitk_f32_instances<Swish, 5, 3>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
