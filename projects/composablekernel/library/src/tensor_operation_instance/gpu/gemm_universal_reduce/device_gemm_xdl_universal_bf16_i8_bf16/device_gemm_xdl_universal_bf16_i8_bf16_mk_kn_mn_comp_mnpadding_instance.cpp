// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "device_gemm_xdl_universal_bf16_i8_bf16_mk_kn_mn.hpp"
#include "ck/host_utility/device_prop.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_xdl_universal_reduce_bf16_i8_bf16_mk_kn_mn_comp_mnpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmV2R1<Row,
                                               Row,
                                               DsLayout,
                                               Row,
                                               BF16,
                                               I8,
                                               DsDataType,
                                               BF16,
                                               PassThrough,
                                               PassThrough,
                                               PassThrough>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_gemm_xdl_universal_reduce_bf16_i8_bf16_mk_kn_mn_comp_instances<GemmMNPadding>{});

    if(ck::get_device_name() == "gfx950")
    {
        add_device_operation_instances(
            instances,
            device_gemm_xdl_universal_reduce_bf16_i8_bf16_mk_kn_mn_comp_instances_2x<
                GemmMNPadding>{});
    }
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
