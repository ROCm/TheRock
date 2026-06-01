// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_xdl_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// Compilation parameters for in[n, hi, wi, g, c] * wei[g, k, y, x, c] = out[n, ho, wo, g, k]
void add_device_grouped_conv3d_bwd_weight_xdl_ngcdhw_gkczyx_ngkdhw_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<3,
                                                           NGCDHW,
                                                           GKCZYX,
                                                           NGKDHW,
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
        device_grouped_conv_bwd_weight_xdl_c_shuffle_bf16_instances<3,
                                                                    NGCDHW,
                                                                    GKCZYX,
                                                                    NGKDHW,
                                                                    ConvBwdWeightDefault,
                                                                    1,
                                                                    1>{});
    add_device_operation_instances(
        instances,
        device_grouped_conv_bwd_weight_xdl_c_shuffle_bf16_instances<3,
                                                                    NGCDHW,
                                                                    GKCZYX,
                                                                    NGKDHW,
                                                                    ConvBwdWeightDefault,
                                                                    4,
                                                                    4>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
