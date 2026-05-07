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
void add_device_grouped_conv2d_fwd_bias_clamp_xdl_merged_groups_nhwgc_gkyxc_nhwgk_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<NHWGK>,
                                                                NHWGK,
                                                                BF16,
                                                                BF16,
                                                                Tuple<BF16>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                AddClamp>>>& instances)
{
    if(ck::get_device_name() == "gfx950")
    {
        add_device_operation_instances(
            instances,
            device_grouped_conv_fwd_xdl_merged_groups_bf16_instances_2x<2,
                                                                        NHWGC,
                                                                        GKYXC,
                                                                        Tuple<NHWGK>,
                                                                        NHWGK,
                                                                        ConvFwdDefault,
                                                                        Tuple<BF16>,
                                                                        AddClamp>{});

        add_device_operation_instances(
            instances,
            device_grouped_conv_fwd_xdl_merged_groups_bf16_instances_2x<2,
                                                                        NHWGC,
                                                                        GKYXC,
                                                                        Tuple<NHWGK>,
                                                                        NHWGK,
                                                                        ConvFwd3x3,
                                                                        Tuple<BF16>,
                                                                        AddClamp>{});
    }
    else
    {
        add_device_operation_instances(
            instances,
            device_grouped_conv_fwd_xdl_merged_groups_bf16_instances<2,
                                                                     NHWGC,
                                                                     GKYXC,
                                                                     Tuple<NHWGK>,
                                                                     NHWGK,
                                                                     ConvFwdDefault,
                                                                     Tuple<BF16>,
                                                                     AddClamp>{});

        add_device_operation_instances(
            instances,
            device_grouped_conv_fwd_xdl_merged_groups_bf16_instances<2,
                                                                     NHWGC,
                                                                     GKYXC,
                                                                     Tuple<NHWGK>,
                                                                     NHWGK,
                                                                     ConvFwd3x3,
                                                                     Tuple<BF16>,
                                                                     AddClamp>{});
    }
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
