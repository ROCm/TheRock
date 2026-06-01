// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "device_gemm_xdl_universal_f16_f16_f16_mk_nk_mn.hpp"
#include "ck/host_utility/device_prop.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_xdl_universal_f16_f16_f16_mk_nk_mn_mem_v1_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances)
{
    add_device_operation_instances(
        instances,
        device_gemm_xdl_universal_f16_f16_f16_mk_nk_mn_mem_instances<Intrawave, GemmDefault>{});

    if(ck::get_device_name() != "gfx950")
    {
        add_device_operation_instances(
            instances,
            device_gemm_xdl_universal_f16_f16_f16_mk_nk_mn_mem_instances_part2<Intrawave,
                                                                               GemmDefault>{});
    }
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
