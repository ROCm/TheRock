// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_dl_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_grouped_conv1d_bwd_weight_dl_nwgc_gkxc_nwgk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<1,
                                                           NWGC,
                                                           GKXC,
                                                           NWGK,
                                                           F32,
                                                           F32,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances)
{
    // 1. Default
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_weight_dl_f32_instances<1,
                                                        NWGC,
                                                        GKXC,
                                                        NWGK,
                                                        ConvBwdWeightDefault>{});
    // 2. Filter1x1Stride1Pad0
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_weight_dl_f32_instances<1,
                                                        NWGC,
                                                        GKXC,
                                                        NWGK,
                                                        ConvBwdWeightFilter1x1Stride1Pad0>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
