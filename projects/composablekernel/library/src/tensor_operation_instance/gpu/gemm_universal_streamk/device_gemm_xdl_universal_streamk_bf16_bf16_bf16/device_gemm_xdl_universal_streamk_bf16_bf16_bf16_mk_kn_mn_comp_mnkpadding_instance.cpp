// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "device_gemm_xdl_universal_streamk_bf16_bf16_bf16_mk_kn_mn.hpp"
#include "ck/host_utility/device_prop.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_xdl_universal_streamk_bf16_bf16_bf16_mk_kn_mn_comp_mnkpadding_instances(
    std::vector<std::unique_ptr<DeviceGemm_Streamk_V2<Row,
                                                      Row,
                                                      Row,
                                                      BF16,
                                                      BF16,
                                                      BF16,
                                                      PassThrough,
                                                      PassThrough,
                                                      PassThrough>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_gemm_xdl_universal_streamk_bf16_bf16_bf16_mk_kn_mn_comp_instances<GemmMNKPadding>{});

    if(ck::get_device_name() != "gfx950")
    {
        add_device_operation_instances(
            instances,
            device_gemm_xdl_universal_streamk_bf16_bf16_bf16_mk_kn_mn_comp_instances_part2<
                GemmMNKPadding>{});
    }
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
