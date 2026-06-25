// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_merged_groups_instance.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_grouped_conv3d_fwd_bias_clamp_xdl_merged_groups_ndhwgc_gkzyxc_ndhwgk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Tuple<NDHWGK>,
                                                                NDHWGK,
                                                                F32,
                                                                F32,
                                                                Tuple<F32>,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                AddClamp>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_merged_groups_f32_instances<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Tuple<NDHWGK>,
                                                                NDHWGK,
                                                                ConvFwdDefault,
                                                                Tuple<F32>,
                                                                AddClamp>{});
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_merged_groups_f32_instances<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Tuple<NDHWGK>,
                                                                NDHWGK,
                                                                ConvFwd3x3,
                                                                Tuple<F32>,
                                                                AddClamp>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
