// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_merged_groups_instance.hpp"
#include "ck/host_utility/device_prop.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
// Compilation parameters for in[n, hi, wi, g, c] * wei[g, k, y, x, c] = out[n, ho, wo, g, k]
void add_device_grouped_conv2d_fwd_xdl_merged_groups_nhwgc_gkyxc_nhwgk_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                NHWGK,
                                                                BF16,
                                                                BF16,
                                                                Empty_Tuple,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances)
{
    if(ck::get_device_name() == "gfx950")
    {
        add_device_operation_instances(
            instances,
            device_grouped_conv_fwd_xdl_merged_groups_bf16_instances_2x<2,
                                                                        NHWGC,
                                                                        GKYXC,
                                                                        Empty_Tuple,
                                                                        NHWGK,
                                                                        ConvFwdDefault>{});

        add_device_operation_instances(
            instances,
            device_grouped_conv_fwd_xdl_merged_groups_bf16_instances_2x<2,
                                                                        NHWGC,
                                                                        GKYXC,
                                                                        Empty_Tuple,
                                                                        NHWGK,
                                                                        ConvFwd3x3>{});
    }
    else
    {
        add_device_operation_instances(
            instances,
            device_grouped_conv_fwd_xdl_merged_groups_bf16_instances<2,
                                                                     NHWGC,
                                                                     GKYXC,
                                                                     Empty_Tuple,
                                                                     NHWGK,
                                                                     ConvFwdDefault>{});

        add_device_operation_instances(
            instances,
            device_grouped_conv_fwd_xdl_merged_groups_bf16_instances<2,
                                                                     NHWGC,
                                                                     GKYXC,
                                                                     Empty_Tuple,
                                                                     NHWGK,
                                                                     ConvFwd3x3>{});
    }
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
