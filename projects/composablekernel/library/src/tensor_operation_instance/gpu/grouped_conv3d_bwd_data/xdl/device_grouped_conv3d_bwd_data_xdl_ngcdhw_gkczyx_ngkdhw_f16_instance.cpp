// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_data/device_grouped_conv_bwd_data_xdl_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_data/device_grouped_conv_bwd_data_transpose_xdl_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_grouped_conv3d_bwd_data_xdl_ngkdhw_gkczyx_ngcdhw_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NGKDHW,
                                                                  GKCZYX,
                                                                  Empty_Tuple,
                                                                  NGCDHW,
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
        device_grouped_conv_bwd_data_xdl_f16_instances<3,
                                                       NGKDHW,
                                                       GKCZYX,
                                                       Empty_Tuple,
                                                       NGCDHW,
                                                       ConvBwdDataDefault>{});
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_data_xdl_f16_nchw_instances<3,
                                                            NGKDHW,
                                                            GKCZYX,
                                                            Empty_Tuple,
                                                            NGCDHW,
                                                            ConvBwdDataFilter1x1Stride1Pad0>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
