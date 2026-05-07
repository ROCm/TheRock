// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_two_stage_xdl_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// Compilation parameters for in[n, hi, wi, g, c] * wei[g, k, y, x, c] = out[n, ho, wo, g, k]
void add_device_grouped_conv2d_bwd_weight_two_stage_xdl_ngchw_gkcyx_ngkhw_bf16_pipev2_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<2,
                                                           NGCHW,
                                                           GKCYX,
                                                           NGKHW,
                                                           BF16,
                                                           BF16,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances)
{
    // 1. Default
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_weight_two_stage_ngchw_xdl_c_shuffle_bf16_instances<
            2,
            NGCHW,
            GKCYX,
            NGKHW,
            ConvBwdWeightDefault,
            BlockGemmPipelineScheduler::Intrawave,
            BlockGemmPipelineVersion::v2>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
