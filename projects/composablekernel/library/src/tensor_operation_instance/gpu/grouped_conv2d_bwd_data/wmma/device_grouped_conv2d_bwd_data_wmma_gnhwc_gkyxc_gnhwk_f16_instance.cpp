// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_data/device_grouped_conv_bwd_data_wmma_f16_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
void add_device_grouped_conv2d_bwd_data_wmma_gnhwk_gkyxc_gnhwc_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  GNHWK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  GNHWC,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_data_wmma_f16_instances<2,
                                                        GNHWK,
                                                        GKYXC,
                                                        Empty_Tuple,
                                                        GNHWC,
                                                        Empty_Tuple,
                                                        PassThrough,
                                                        ConvBwdDataDefault>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
