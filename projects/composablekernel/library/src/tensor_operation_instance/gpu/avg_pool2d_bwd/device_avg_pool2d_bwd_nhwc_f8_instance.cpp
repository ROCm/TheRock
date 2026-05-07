// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "device_avg_pool2d_bwd_nhwc_instance_common.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_avgpool_2D_bwd_nhwc_f8_instances(
    std::vector<std::unique_ptr<DeviceAvgPoolBwd<2, F8, F8, NHWC, NHWC>>>& instances)
{
    add_device_operation_instances(instances, device_avgpool_2D_bwd_nhwc_instances<F8, F8, F32>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
