// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_bilinear_instance.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_grouped_conv3d_fwd_xdl_bilinear_ndhwgc_gkzyxc_ndhwgk_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<NDHWGK>,
                                                                NDHWGK,
                                                                BF16,
                                                                BF16,
                                                                ck::Tuple<BF16>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Bilinear>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_bilinear_bf16_instances<3,
                                                            NDHWGC,
                                                            GKZYXC,
                                                            Tuple<NDHWGK>,
                                                            NDHWGK,
                                                            ConvFwdDefault>{});
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_bilinear_bf16_instances<3,
                                                            NDHWGC,
                                                            GKZYXC,
                                                            Tuple<NDHWGK>,
                                                            NDHWGK,
                                                            ConvFwd1x1P0>{});
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_bilinear_bf16_instances<3,
                                                            NDHWGC,
                                                            GKZYXC,
                                                            Tuple<NDHWGK>,
                                                            NDHWGK,
                                                            ConvFwd1x1S1P0>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
