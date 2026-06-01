// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_data/device_grouped_conv_bwd_data_xdl_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_data/device_grouped_conv_bwd_data_transpose_xdl_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_grouped_conv2d_bwd_data_xdl_ngkhw_gkcyx_ngchw_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NGKHW,
                                                                  GKCYX,
                                                                  Empty_Tuple,
                                                                  NGCHW,
                                                                  BF16,
                                                                  BF16,
                                                                  Empty_Tuple,
                                                                  BF16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_data_xdl_bf16_instances<2,
                                                        NGKHW,
                                                        GKCYX,
                                                        Empty_Tuple,
                                                        NGCHW,
                                                        ConvBwdDataDefault>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
