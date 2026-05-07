// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "device_avg_pool2d_bwd_nhwc_instance_common.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_avgpool_2D_bwd_nhwc_int8_instances(
    std::vector<std::unique_ptr<DeviceAvgPoolBwd<2, I8, I8, NHWC, NHWC>>>& instances)
{
    add_device_operation_instances(instances, device_avgpool_2D_bwd_nhwc_instances<I8, I8, I32>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
