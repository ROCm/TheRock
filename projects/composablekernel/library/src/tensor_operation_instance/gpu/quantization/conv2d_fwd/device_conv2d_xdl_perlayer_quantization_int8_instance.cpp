// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "device_conv2d_xdl_int8_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
void add_device_conv2d_xdl_perlayer_quantization_int8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<NDimSpatial,
                                                                NHWGC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                NHWGK,
                                                                int8_t,
                                                                int8_t,
                                                                Empty_Tuple,
                                                                int8_t,
                                                                PassThrough,
                                                                PassThrough,
                                                                Mul_Clamp>>>& instances)
{
    add_device_operation_instances(instances,
                                   device_grouped_conv2d_xdl_int8_instances<NHWGC,
                                                                            GKYXC,
                                                                            Empty_Tuple,
                                                                            NHWGK,
                                                                            Empty_Tuple,
                                                                            Mul_Clamp,
                                                                            ConvFwdDefault,
                                                                            16>{});
    add_device_operation_instances(instances,
                                   device_grouped_conv2d_xdl_int8_instances<NHWGC,
                                                                            GKYXC,
                                                                            Empty_Tuple,
                                                                            NHWGK,
                                                                            Empty_Tuple,
                                                                            Mul_Clamp,
                                                                            ConvFwd1x1P0,
                                                                            16>{});
    add_device_operation_instances(instances,
                                   device_grouped_conv2d_xdl_int8_instances<NHWGC,
                                                                            GKYXC,
                                                                            Empty_Tuple,
                                                                            NHWGK,
                                                                            Empty_Tuple,
                                                                            Mul_Clamp,
                                                                            ConvFwd1x1S1P0,
                                                                            16>{});
}

void add_device_conv2d_xdl_relu_perlayer_quantization_int8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<NDimSpatial,
                                                                NHWGC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                NHWGK,
                                                                int8_t,
                                                                int8_t,
                                                                Empty_Tuple,
                                                                int8_t,
                                                                PassThrough,
                                                                PassThrough,
                                                                Relu_Mul_Clamp>>>& instances)
{
    add_device_operation_instances(instances,
                                   device_grouped_conv2d_xdl_int8_instances<NHWGC,
                                                                            GKYXC,
                                                                            Empty_Tuple,
                                                                            NHWGK,
                                                                            Empty_Tuple,
                                                                            Relu_Mul_Clamp,
                                                                            ConvFwdDefault,
                                                                            16>{});
    add_device_operation_instances(instances,
                                   device_grouped_conv2d_xdl_int8_instances<NHWGC,
                                                                            GKYXC,
                                                                            Empty_Tuple,
                                                                            NHWGK,
                                                                            Empty_Tuple,
                                                                            Relu_Mul_Clamp,
                                                                            ConvFwd1x1P0,
                                                                            16>{});
    add_device_operation_instances(instances,
                                   device_grouped_conv2d_xdl_int8_instances<NHWGC,
                                                                            GKYXC,
                                                                            Empty_Tuple,
                                                                            NHWGK,
                                                                            Empty_Tuple,
                                                                            Relu_Mul_Clamp,
                                                                            ConvFwd1x1S1P0,
                                                                            16>{});
}
} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
