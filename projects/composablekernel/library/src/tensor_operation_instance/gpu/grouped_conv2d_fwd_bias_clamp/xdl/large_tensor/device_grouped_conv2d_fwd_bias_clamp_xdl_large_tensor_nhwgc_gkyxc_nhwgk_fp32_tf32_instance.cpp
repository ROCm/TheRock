// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_large_tensor_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
// Compilation parameters for in[n, hi, wi, g, c] * wei[g, k, y, x, c] = out[n, ho, wo, g, k]
void add_device_grouped_conv2d_fwd_bias_clamp_xdl_large_tensor_nhwgc_gkyxc_nhwgk_f32_tf32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<NHWGK>,
                                                                NHWGK,
                                                                F32,
                                                                F32,
                                                                Tuple<F32>,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                AddClamp,
                                                                TF32,
                                                                TF32>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_large_tensor_f32_tf32_instances<2,
                                                                    NHWGC,
                                                                    GKYXC,
                                                                    Tuple<NHWGK>,
                                                                    NHWGK,
                                                                    ConvFwdDefault,
                                                                    Tuple<F32>,
                                                                    AddClamp>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
