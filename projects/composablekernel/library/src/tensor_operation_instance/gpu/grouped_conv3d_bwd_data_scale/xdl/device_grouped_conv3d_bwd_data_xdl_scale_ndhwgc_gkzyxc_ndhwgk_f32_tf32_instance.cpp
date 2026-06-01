// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_data/device_grouped_conv_bwd_data_xdl_scale_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_grouped_conv3d_bwd_data_xdl_scale_ndhwgk_gkzyxc_ndhwgc_f32_tf32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NDHWGK,
                                                                  GKZYXC,
                                                                  Tuple<>,
                                                                  NDHWGC,
                                                                  F32,
                                                                  F32,
                                                                  Tuple<>,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  Scale,
                                                                  TF32,
                                                                  TF32>>>& instances)
{
    // 1. Default
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_data_xdl_scale_f32_tf32_instances<3,
                                                                  NDHWGK,
                                                                  GKZYXC,
                                                                  Tuple<>,
                                                                  NDHWGC,
                                                                  ConvBwdDataDefault>{});
    // 2. Filter1x1Stride1Pad0
    add_device_operation_instances(instances,
                                   device_grouped_conv_bwd_data_xdl_scale_f32_tf32_instances<
                                       3,
                                       NDHWGK,
                                       GKZYXC,
                                       Tuple<>,
                                       NDHWGC,
                                       ConvBwdDataFilter1x1Stride1Pad0>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
