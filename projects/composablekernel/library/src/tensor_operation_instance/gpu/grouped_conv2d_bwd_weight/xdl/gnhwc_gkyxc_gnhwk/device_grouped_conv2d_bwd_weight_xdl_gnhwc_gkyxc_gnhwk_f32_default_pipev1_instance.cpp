// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_v3_xdl_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// Compilation parameters for in[g, n, hi, wi, c] * wei[g, k, y, x, c] = out[g, n, ho, wo, k]
void add_device_grouped_conv2d_bwd_weight_xdl_gnhwc_gkyxc_gnhwk_f32_default_pipev1_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<2,
                                                           GNHWC,
                                                           GKYXC,
                                                           GNHWK,
                                                           F32,
                                                           F32,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances)
{
    add_device_operation_instances(instances,
                                   device_grouped_conv_bwd_weight_v3_xdl_c_shuffle_f32_instances<
                                       2,
                                       GNHWC,
                                       GKYXC,
                                       GNHWK,
                                       ConvBwdWeightDefault,
                                       BlockGemmPipelineScheduler::Intrawave,
                                       BlockGemmPipelineVersion::v1>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
