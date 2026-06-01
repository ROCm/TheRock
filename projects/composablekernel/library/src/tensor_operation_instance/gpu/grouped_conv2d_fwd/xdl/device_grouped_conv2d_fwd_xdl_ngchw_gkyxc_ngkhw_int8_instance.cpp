// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
// Compilation parameters for in[n, hi, wi, g, c] * wei[g, k, y, x, c] = out[n, ho, wo, g, k]
void add_device_grouped_conv2d_fwd_xdl_ngchw_gkyxc_ngkhw_int8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NGCHW,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                NGKHW,
                                                                int8_t,
                                                                int8_t,
                                                                Empty_Tuple,
                                                                int8_t,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_int8_generic_instances<2,
                                                           NGCHW,
                                                           GKYXC,
                                                           Empty_Tuple,
                                                           NGKHW,
                                                           ConvFwdDefault>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
