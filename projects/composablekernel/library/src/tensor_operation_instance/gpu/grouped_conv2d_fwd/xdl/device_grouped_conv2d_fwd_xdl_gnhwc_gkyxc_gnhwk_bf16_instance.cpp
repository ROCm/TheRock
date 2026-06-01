// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
// Compilation parameters for in[g, n, hi, wi, c] * wei[g, k, y, x, c] = out[g, n, ho, wo, k]
void add_device_grouped_conv2d_fwd_xdl_gnhwc_gkyxc_gnhwk_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                GNHWC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                GNHWK,
                                                                BF16,
                                                                BF16,
                                                                Empty_Tuple,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances)
{
    add_device_operation_instances(instances,
                                   device_grouped_conv_fwd_xdl_bf16_instances<2,
                                                                              GNHWC,
                                                                              GKYXC,
                                                                              Empty_Tuple,
                                                                              GNHWK,
                                                                              ConvFwdDefault>{});

    add_device_operation_instances(instances,
                                   device_grouped_conv_fwd_xdl_bf16_instances<2,
                                                                              GNHWC,
                                                                              GKYXC,
                                                                              Empty_Tuple,
                                                                              GNHWK,
                                                                              ConvFwd1x1P0>{});

    add_device_operation_instances(instances,
                                   device_grouped_conv_fwd_xdl_bf16_instances<2,
                                                                              GNHWC,
                                                                              GKYXC,
                                                                              Empty_Tuple,
                                                                              GNHWK,
                                                                              ConvFwd1x1S1P0>{});

    add_device_operation_instances(instances,
                                   device_grouped_conv_fwd_xdl_bf16_instances<2,
                                                                              GNHWC,
                                                                              GKYXC,
                                                                              Empty_Tuple,
                                                                              GNHWK,
                                                                              ConvFwdOddC>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
