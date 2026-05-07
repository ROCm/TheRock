// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_dynamic_op_instance.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_grouped_conv2d_fwd_xdl_dynamic_op_nhwgc_gkyxc_nhwgk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                ck::Tuple<>,
                                                                NHWGK,
                                                                F32,
                                                                F32,
                                                                ck::Tuple<>,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                DynamicUnaryOp>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_dynamic_op_f32_instances<2,
                                                             NHWGC,
                                                             GKYXC,
                                                             Tuple<>,
                                                             NHWGK,
                                                             ConvFwdDefault>{});
#if 0 // Enable with dynamic op optimizations (at now generating a lot of virtual functions cause
      // long compilation time)
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_dynamic_op_f32_instances<2,
                                                             NHWGC,
                                                             GKYXC,
                                                             Tuple<>,
                                                             NHWGK,
                                                             ConvFwd1x1P0>{});
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_dynamic_op_f32_instances<2,
                                                             NHWGC,
                                                             GKYXC,
                                                             Tuple<>,
                                                             NHWGK,
                                                             ConvFwd1x1S1P0>{});
#endif
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
