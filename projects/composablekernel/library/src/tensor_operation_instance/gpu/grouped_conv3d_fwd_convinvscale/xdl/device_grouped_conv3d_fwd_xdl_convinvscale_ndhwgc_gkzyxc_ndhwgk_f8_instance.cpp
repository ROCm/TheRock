// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_outelementop_instance.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using ConvInvscale = ck::tensor_operation::element_wise::ConvInvscale;

void add_device_grouped_conv3d_fwd_xdl_convinvscale_ndhwgc_gkzyxc_ndhwgk_f8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<>,
                                                                NDHWGK,
                                                                F8,
                                                                F8,
                                                                ck::Tuple<>,
                                                                F8,
                                                                PassThrough,
                                                                PassThrough,
                                                                ConvInvscale,
                                                                F8,
                                                                F8>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_outelementop_f8_instances<3,
                                                              NDHWGC,
                                                              GKZYXC,
                                                              ck::Tuple<>,
                                                              NDHWGK,
                                                              ConvFwdDefault,
                                                              ConvInvscale>{});
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_outelementop_f8_instances<3,
                                                              NDHWGC,
                                                              GKZYXC,
                                                              ck::Tuple<>,
                                                              NDHWGK,
                                                              ConvFwd1x1P0,
                                                              ConvInvscale>{});
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_outelementop_f8_instances<3,
                                                              NDHWGC,
                                                              GKZYXC,
                                                              ck::Tuple<>,
                                                              NDHWGK,
                                                              ConvFwd1x1S1P0,
                                                              ConvInvscale>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
