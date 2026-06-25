// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_scaleadd_scaleadd_relu_instance.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
void add_device_grouped_conv3d_fwd_xdl_scaleadd_scaleadd_relu_ndhwgc_gkzyxc_ndhwgk_int8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<NDHWGK, G_K>,
                                                                NDHWGK,
                                                                int8_t,
                                                                int8_t,
                                                                ck::Tuple<F32, F32>,
                                                                int8_t,
                                                                PassThrough,
                                                                PassThrough,
                                                                ScaleAddScaleAddRelu>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_scaleadd_scaleadd_relu_int8_instances<3,
                                                                          NDHWGC,
                                                                          GKZYXC,
                                                                          ck::Tuple<NDHWGK, G_K>,
                                                                          NDHWGK,
                                                                          ConvFwdDefault>{});
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_scaleadd_scaleadd_relu_int8_instances<3,
                                                                          NDHWGC,
                                                                          GKZYXC,
                                                                          ck::Tuple<NDHWGK, G_K>,
                                                                          NDHWGK,
                                                                          ConvFwd1x1P0>{});
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_scaleadd_scaleadd_relu_int8_instances<3,
                                                                          NDHWGC,
                                                                          GKZYXC,
                                                                          ck::Tuple<NDHWGK, G_K>,
                                                                          NDHWGK,
                                                                          ConvFwd1x1S1P0>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
