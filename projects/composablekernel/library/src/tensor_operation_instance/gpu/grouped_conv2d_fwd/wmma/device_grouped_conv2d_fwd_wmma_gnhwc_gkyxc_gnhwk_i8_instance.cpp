// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
// Compilation parameters for in[g, n, hi, wi, c] * wei[g, k, y, x, c] = out[g, n, ho, wo, k]
void add_device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_i8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                GNHWC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                GNHWK,
                                                                int8_t,
                                                                int8_t,
                                                                Empty_Tuple,
                                                                int8_t,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances)
{
    add_device_operation_instances(instances,
                                   device_grouped_conv_fwd_wmma_i8_instances<2,
                                                                             GNHWC,
                                                                             GKYXC,
                                                                             Empty_Tuple,
                                                                             GNHWK,
                                                                             Empty_Tuple,
                                                                             PassThrough,
                                                                             ConvFwdDefault>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
