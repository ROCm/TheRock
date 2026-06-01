// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_grouped_conv3d_fwd_wmma_cshufflev3_ndhwgc_gkzyxc_ndhwgk_f16_instances_part4(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                NDHWGK,
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
        device_grouped_conv_fwd_wmma_cshufflev3_f16_instances_part4<3,
                                                                    NDHWGC,
                                                                    GKZYXC,
                                                                    Empty_Tuple,
                                                                    NDHWGK,
                                                                    ConvFwdDefault>{});
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_wmma_cshufflev3_f16_instances_part4<3,
                                                                    NDHWGC,
                                                                    GKZYXC,
                                                                    Empty_Tuple,
                                                                    NDHWGK,
                                                                    ConvFwd1x1P0>{});
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_wmma_cshufflev3_f16_instances_part4<3,
                                                                    NDHWGC,
                                                                    GKZYXC,
                                                                    Empty_Tuple,
                                                                    NDHWGK,
                                                                    ConvFwd1x1S1P0>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
