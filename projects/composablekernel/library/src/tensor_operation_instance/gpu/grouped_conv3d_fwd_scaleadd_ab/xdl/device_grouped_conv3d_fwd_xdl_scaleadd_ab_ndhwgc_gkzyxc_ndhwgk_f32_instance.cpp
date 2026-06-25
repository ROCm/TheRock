// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_scaleadd_ab_instance.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_grouped_conv3d_fwd_xdl_scaleadd_ab_ndhwgc_gkzyxc_ndhwgk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<>,
                                                                NDHWGK,
                                                                ck::Tuple<F32, F32>,
                                                                ck::Tuple<F32, F32>,
                                                                ck::Tuple<>,
                                                                F32,
                                                                ScaleAdd,
                                                                ScaleAdd,
                                                                PassThrough>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_scaleadd_ab_f32_instances<3,
                                                              NDHWGC,
                                                              GKZYXC,
                                                              NDHWGK,
                                                              ConvFwdDefault>{});
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_scaleadd_ab_f32_instances<3,
                                                              NDHWGC,
                                                              GKZYXC,
                                                              NDHWGK,
                                                              ConvFwd1x1P0>{});
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_scaleadd_ab_f32_instances<3,
                                                              NDHWGC,
                                                              GKZYXC,
                                                              NDHWGK,
                                                              ConvFwd1x1S1P0>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
