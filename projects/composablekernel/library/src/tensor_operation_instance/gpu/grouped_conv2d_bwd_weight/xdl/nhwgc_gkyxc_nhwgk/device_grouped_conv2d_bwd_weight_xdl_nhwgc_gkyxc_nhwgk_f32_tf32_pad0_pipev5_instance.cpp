// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_v3_xdl_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// Compilation parameters for in[n, hi, wi, g, c] * wei[g, k, y, x, c] = out[n, ho, wo, g, k]
void add_device_grouped_conv2d_bwd_weight_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_pad0_pipev5_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<2,
                                                           NHWGC,
                                                           GKYXC,
                                                           NHWGK,
                                                           F32,
                                                           F32,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough,
                                                           TF32,
                                                           TF32>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_weight_v3_xdl_c_shuffle_f32_tf32_instances<
            2,
            NHWGC,
            GKYXC,
            NHWGK,
            ConvBwdWeightFilter1x1Stride1Pad0,
            BlockGemmPipelineScheduler::Intrawave,
            BlockGemmPipelineVersion::v5>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
