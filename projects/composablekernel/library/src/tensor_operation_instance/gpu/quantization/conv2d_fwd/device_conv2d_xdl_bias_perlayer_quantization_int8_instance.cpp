// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "device_conv2d_xdl_int8_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
void add_device_conv2d_xdl_bias_perlayer_quantization_int8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<NDimSpatial,
                                                                NHWGC,
                                                                GKYXC,
                                                                GK_Tuple,
                                                                NHWGK,
                                                                int8_t,
                                                                int8_t,
                                                                I32_Tuple,
                                                                int8_t,
                                                                PassThrough,
                                                                PassThrough,
                                                                Add_Mul_Clamp>>>& instances)
{
    add_device_operation_instances(instances,
                                   device_grouped_conv2d_xdl_int8_instances<NHWGC,
                                                                            GKYXC,
                                                                            GK_Tuple,
                                                                            NHWGK,
                                                                            I32_Tuple,
                                                                            Add_Mul_Clamp,
                                                                            ConvFwdDefault,
                                                                            8>{});
    add_device_operation_instances(instances,
                                   device_grouped_conv2d_xdl_int8_instances<NHWGC,
                                                                            GKYXC,
                                                                            GK_Tuple,
                                                                            NHWGK,
                                                                            I32_Tuple,
                                                                            Add_Mul_Clamp,
                                                                            ConvFwd1x1P0,
                                                                            8>{});
    add_device_operation_instances(instances,
                                   device_grouped_conv2d_xdl_int8_instances<NHWGC,
                                                                            GKYXC,
                                                                            GK_Tuple,
                                                                            NHWGK,
                                                                            I32_Tuple,
                                                                            Add_Mul_Clamp,
                                                                            ConvFwd1x1S1P0,
                                                                            8>{});
}

void add_device_conv2d_xdl_bias_relu_perlayer_quantization_int8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<NDimSpatial,
                                                                NHWGC,
                                                                GKYXC,
                                                                GK_Tuple,
                                                                NHWGK,
                                                                int8_t,
                                                                int8_t,
                                                                I32_Tuple,
                                                                int8_t,
                                                                PassThrough,
                                                                PassThrough,
                                                                Add_Relu_Mul_Clamp>>>& instances)
{
    add_device_operation_instances(instances,
                                   device_grouped_conv2d_xdl_int8_instances<NHWGC,
                                                                            GKYXC,
                                                                            GK_Tuple,
                                                                            NHWGK,
                                                                            I32_Tuple,
                                                                            Add_Relu_Mul_Clamp,
                                                                            ConvFwdDefault,
                                                                            8>{});

    add_device_operation_instances(instances,
                                   device_grouped_conv2d_xdl_int8_instances<NHWGC,
                                                                            GKYXC,
                                                                            GK_Tuple,
                                                                            NHWGK,
                                                                            I32_Tuple,
                                                                            Add_Relu_Mul_Clamp,
                                                                            ConvFwd1x1P0,
                                                                            8>{});

    add_device_operation_instances(instances,
                                   device_grouped_conv2d_xdl_int8_instances<NHWGC,
                                                                            GKYXC,
                                                                            GK_Tuple,
                                                                            NHWGK,
                                                                            I32_Tuple,
                                                                            Add_Relu_Mul_Clamp,
                                                                            ConvFwd1x1S1P0,
                                                                            8>{});
}

void add_device_conv2d_xdl_bias_tanh_perlayer_quantization_int8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<NDimSpatial,
                                                                NHWGC,
                                                                GKYXC,
                                                                GK_Tuple,
                                                                NHWGK,
                                                                int8_t,
                                                                int8_t,
                                                                I32_Tuple,
                                                                int8_t,
                                                                PassThrough,
                                                                PassThrough,
                                                                Add_Mul_TanH_Mul_Clamp>>>&
        instances)
{
    add_device_operation_instances(instances,
                                   device_grouped_conv2d_xdl_int8_instances<NHWGC,
                                                                            GKYXC,
                                                                            GK_Tuple,
                                                                            NHWGK,
                                                                            I32_Tuple,
                                                                            Add_Mul_TanH_Mul_Clamp,
                                                                            ConvFwdDefault,
                                                                            8>{});

    add_device_operation_instances(instances,
                                   device_grouped_conv2d_xdl_int8_instances<NHWGC,
                                                                            GKYXC,
                                                                            GK_Tuple,
                                                                            NHWGK,
                                                                            I32_Tuple,
                                                                            Add_Mul_TanH_Mul_Clamp,
                                                                            ConvFwd1x1P0,
                                                                            8>{});

    add_device_operation_instances(instances,
                                   device_grouped_conv2d_xdl_int8_instances<NHWGC,
                                                                            GKYXC,
                                                                            GK_Tuple,
                                                                            NHWGK,
                                                                            I32_Tuple,
                                                                            Add_Mul_TanH_Mul_Clamp,
                                                                            ConvFwd1x1S1P0,
                                                                            8>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
