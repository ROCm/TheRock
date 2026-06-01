// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
// Compilation parameters for in[n, di, hi, wi, g, c] * wei[g, k, z, y, x, c] = out[n, do, ho, wo,
// g, k]
void add_device_grouped_conv3d_fwd_wmma_ndhwgc_gkzyxc_ndhwgk_f16_1x1p0_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                NDHWGK,
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
                                                                              NDHWGC,
                                                                              GKZYXC,
                                                                              Empty_Tuple,
                                                                              NDHWGK,
                                                                              Empty_Tuple,
                                                                              PassThrough,
                                                                              ConvFwd1x1P0>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
