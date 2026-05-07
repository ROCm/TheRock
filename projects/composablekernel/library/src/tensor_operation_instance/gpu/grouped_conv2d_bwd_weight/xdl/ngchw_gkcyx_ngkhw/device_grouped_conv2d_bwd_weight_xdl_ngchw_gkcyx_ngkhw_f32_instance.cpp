// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_xdl_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// Compilation parameters for in[n, hi, wi, g, c] * wei[g, k, y, x, c] = out[n, ho, wo, g, k]
void add_device_grouped_conv2d_bwd_weight_xdl_ngchw_gkcyx_ngkhw_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<2,
                                                           NGCHW,
                                                           GKCYX,
                                                           NGKHW,
                                                           F32,
                                                           F32,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances)
{
    // 1. Default
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_weight_xdl_c_shuffle_f32_instances<2,
                                                                   NGCHW,
                                                                   GKCYX,
                                                                   NGKHW,
                                                                   ConvBwdWeightDefault,
                                                                   1,
                                                                   1>{});
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_weight_xdl_c_shuffle_f32_instances<2,
                                                                   NGCHW,
                                                                   GKCYX,
                                                                   NGKHW,
                                                                   ConvBwdWeightDefault,
                                                                   4,
                                                                   4>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
