// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
// Compilation parameters for in[n, hi, wi, g, c] * wei[g, k, y, x, c] = out[n, ho, wo, g, k]
void add_device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_f16_16x16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NGCDHW,
                                                                GKCZYX,
                                                                Empty_Tuple,
                                                                NGKDHW,
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
        device_grouped_conv_fwd_xdl_f16_16x16_instances<3,
                                                        NGCDHW,
                                                        GKCZYX,
                                                        Empty_Tuple,
                                                        NGKDHW,
                                                        ConvFwdDefault>{});

    add_device_operation_instances(instances,
                                   device_grouped_conv_fwd_xdl_f16_16x16_instances<3,
                                                                                   NGCDHW,
                                                                                   GKCZYX,
                                                                                   Empty_Tuple,
                                                                                   NGKDHW,
                                                                                   ConvFwd1x1P0>{});
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_f16_16x16_instances<3,
                                                        NGCDHW,
                                                        GKCZYX,
                                                        Empty_Tuple,
                                                        NGKDHW,
                                                        ConvFwd1x1S1P0>{});

    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_f16_nchw_instances<3,
                                                       NGCDHW,
                                                       GKCZYX,
                                                       Empty_Tuple,
                                                       NGKDHW,
                                                       ConvFwd1x1S1P0>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
