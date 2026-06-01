// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_instance.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_grouped_conv1d_fwd_xdl_gnwc_gkxc_gnwk_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<1,
                                                                GNWC,
                                                                GKXC,
                                                                Empty_Tuple,
                                                                GNWK,
                                                                BF16,
                                                                BF16,
                                                                Empty_Tuple,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances)
{
    add_device_operation_instances(instances,
                                   device_grouped_conv_fwd_xdl_bf16_instances<1,
                                                                              GNWC,
                                                                              GKXC,
                                                                              Empty_Tuple,
                                                                              GNWK,
                                                                              ConvFwdDefault>{});
    add_device_operation_instances(instances,
                                   device_grouped_conv_fwd_xdl_bf16_instances<1,
                                                                              GNWC,
                                                                              GKXC,
                                                                              Empty_Tuple,
                                                                              GNWK,
                                                                              ConvFwd1x1P0>{});
    add_device_operation_instances(instances,
                                   device_grouped_conv_fwd_xdl_bf16_instances<1,
                                                                              GNWC,
                                                                              GKXC,
                                                                              Empty_Tuple,
                                                                              GNWK,
                                                                              ConvFwd1x1S1P0>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
