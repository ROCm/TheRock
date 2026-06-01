// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
void add_device_grouped_conv2d_fwd_clamp_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_bf16_instances_part3(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<>,
                                                                NHWGK,
                                                                BF16,
                                                                BF16,
                                                                Tuple<>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_wmma_cshufflev3_bf16_instances_part3<2,
                                                                     NHWGC,
                                                                     GKYXC,
                                                                     Tuple<>,
                                                                     NHWGK,
                                                                     ConvFwdDefault,
                                                                     Tuple<>,
                                                                     Clamp>{});

    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_wmma_cshufflev3_bf16_instances_part3<2,
                                                                     NHWGC,
                                                                     GKYXC,
                                                                     Tuple<>,
                                                                     NHWGK,
                                                                     ConvFwd1x1P0,
                                                                     Tuple<>,
                                                                     Clamp>{});

    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_wmma_cshufflev3_bf16_instances_part3<2,
                                                                     NHWGC,
                                                                     GKYXC,
                                                                     Tuple<>,
                                                                     NHWGK,
                                                                     ConvFwd1x1S1P0,
                                                                     Tuple<>,
                                                                     Clamp>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
