// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/transpose/device_transpose_instance.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F16         = ck::half_t;
using F32         = float;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

void add_device_transpose_f16_instances(
    std::vector<std::unique_ptr<DeviceElementwise<ck::Tuple<F16>, ck::Tuple<F16>, PassThrough, 5>>>&
        instances)
{
    add_device_operation_instances(instances, device_transpose_f16_instances{});
}

void add_device_transpose_f32_instances(
    std::vector<std::unique_ptr<DeviceElementwise<ck::Tuple<F32>, ck::Tuple<F32>, PassThrough, 5>>>&
        instances)
{
    add_device_operation_instances(instances, device_transpose_f32_instances{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
