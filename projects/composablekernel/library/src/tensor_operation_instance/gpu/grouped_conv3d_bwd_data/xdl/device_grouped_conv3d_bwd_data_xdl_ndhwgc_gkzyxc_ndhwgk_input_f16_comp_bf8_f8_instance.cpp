// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_data/device_grouped_conv_bwd_data_xdl_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_grouped_conv3d_bwd_data_xdl_ndhwgk_gkzyxc_ndhwgc_input_f16_comp_bf8f8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NDHWGK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  NDHWGC,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  BF8,
                                                                  F8>>>& instances)
{
#if CK_BUILD_DEPRECATED
#pragma message "These instances are getting deprecated"
    // 1. Default
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_data_xdl_input_fp16_comp_bf8f8_instances<3,
                                                                         NDHWGK,
                                                                         GKZYXC,
                                                                         Empty_Tuple,
                                                                         NDHWGC,
                                                                         ConvBwdDataDefault>{});
    // 2. Filter1x1Stride1Pad0
    add_device_operation_instances(instances,
                                   device_grouped_conv_bwd_data_xdl_input_fp16_comp_bf8f8_instances<
                                       3,
                                       NDHWGK,
                                       GKZYXC,
                                       Empty_Tuple,
                                       NDHWGC,
                                       ConvBwdDataFilter1x1Stride1Pad0>{});
#else
#pragma message "These instances were deprecated"
    std::ignore = instances;
#endif
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
