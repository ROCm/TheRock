// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_dynamic_op_instance.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_grouped_conv3d_fwd_xdl_dynamic_op_ndhwgc_gkzyxc_ndhwgk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<>,
                                                                NDHWGK,
                                                                F16,
                                                                F16,
                                                                ck::Tuple<>,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                DynamicUnaryOp>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_dynamic_op_f16_instances<3,
                                                             NDHWGC,
                                                             GKZYXC,
                                                             Tuple<>,
                                                             NDHWGK,
                                                             ConvFwdDefault>{});
#if 0 // Enable with dynamic op optimizations (at now generating a lot of virtual functions cause
      // long compilation time)
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_dynamic_op_f16_instances<3,
                                                             NDHWGC,
                                                             GKZYXC,
                                                             Tuple<>,
                                                             NDHWGK,
                                                             ConvFwd1x1P0>{});
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_dynamic_op_f16_instances<3,
                                                             NDHWGC,
                                                             GKZYXC,
                                                             Tuple<>,
                                                             NDHWGK,
                                                             ConvFwd1x1S1P0>{});
#endif
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
