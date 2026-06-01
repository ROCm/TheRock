// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <vector>

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"
#include "ck/library/tensor_operation_instance/gpu/softmax/device_softmax_f32_f32_instance_rank3_reduce1.hpp"
#include "ck/library/tensor_operation_instance/gpu/softmax/device_softmax_f32_f32_instance_type.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_softmax_f32_f32_rank3_reduce1_instances(
    std::vector<DeviceSoftmaxPtr<F32, F32, F32, PassThrough, PassThrough, 3, 1>>& instances)
{
    add_device_operation_instances(instances, device_softmax_f32_f32_generic_instance<3, 1>{});
    add_device_operation_instances(instances, device_softmax_f32_f32_instances<3, 1>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
