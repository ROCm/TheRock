// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
// Compilation parameters for in[g, n, di, hi, wi, c] * wei[g, k, z, y, x, c] = out[g, n, do, ho,
// wo, k]
void add_device_grouped_conv3d_fwd_wmma_gndhwc_gkzyxc_gndhwk_f16_oddc_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                GNDHWC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                GNDHWK,
                                                                F16,
                                                                F16,
                                                                Empty_Tuple,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances)
{
    add_device_operation_instances(instances,
                                   device_grouped_conv_fwd_wmma_f16_instances<3,
                                                                              GNDHWC,
                                                                              GKZYXC,
                                                                              Empty_Tuple,
                                                                              GNDHWK,
                                                                              Empty_Tuple,
                                                                              PassThrough,
                                                                              ConvFwdOddC>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
