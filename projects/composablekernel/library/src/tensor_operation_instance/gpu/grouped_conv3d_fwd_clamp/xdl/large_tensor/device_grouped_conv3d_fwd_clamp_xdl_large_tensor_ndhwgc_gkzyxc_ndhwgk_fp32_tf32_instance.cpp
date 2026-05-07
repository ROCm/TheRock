// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_large_tensor_instance.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_grouped_conv3d_fwd_clamp_xdl_large_tensor_ndhwgc_gkzyxc_ndhwgk_f32_tf32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Tuple<>,
                                                                NDHWGK,
                                                                F32,
                                                                F32,
                                                                Tuple<>,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp,
                                                                TF32,
                                                                TF32>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_large_tensor_f32_tf32_instances<3,
                                                                    NDHWGC,
                                                                    GKZYXC,
                                                                    Tuple<>,
                                                                    NDHWGK,
                                                                    ConvFwdDefault,
                                                                    Tuple<>,
                                                                    Clamp>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
