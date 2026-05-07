// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_instance.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_f8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                NDHWGK,
                                                                F8,
                                                                F8,
                                                                Empty_Tuple,
                                                                F8,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough,
                                                                F8>>>& instances)
{
    add_device_operation_instances(instances,
                                   device_grouped_conv_fwd_xdl_f8_instances<3,
                                                                            NDHWGC,
                                                                            GKZYXC,
                                                                            Empty_Tuple,
                                                                            NDHWGK,
                                                                            ConvFwdDefault>{});
    add_device_operation_instances(instances,
                                   device_grouped_conv_fwd_xdl_f8_instances<3,
                                                                            NDHWGC,
                                                                            GKZYXC,
                                                                            Empty_Tuple,
                                                                            NDHWGK,
                                                                            ConvFwd1x1P0>{});
    add_device_operation_instances(instances,
                                   device_grouped_conv_fwd_xdl_f8_instances<3,
                                                                            NDHWGC,
                                                                            GKZYXC,
                                                                            Empty_Tuple,
                                                                            NDHWGK,
                                                                            ConvFwd1x1S1P0>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
