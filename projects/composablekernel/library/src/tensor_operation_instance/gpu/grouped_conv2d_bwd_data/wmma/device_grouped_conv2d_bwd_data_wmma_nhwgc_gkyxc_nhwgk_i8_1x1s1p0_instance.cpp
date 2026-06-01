// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_data/device_grouped_conv_bwd_data_wmma_i8_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
void add_device_grouped_conv2d_bwd_data_wmma_nhwgk_gkyxc_nhwgc_i8_1x1s1p0_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NHWGK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  NHWGC,
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
        device_grouped_conv_bwd_data_wmma_i8_instances<2,
                                                       NHWGK,
                                                       GKYXC,
                                                       Empty_Tuple,
                                                       NHWGC,
                                                       Empty_Tuple,
                                                       PassThrough,
                                                       ConvBwdData1x1S1P0>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
