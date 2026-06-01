// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "device_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn_mem_v2_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Col, Row, F8, F8, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances)
{
    add_device_operation_instances(
        instances,
        device_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn_mem_instances<Interwave, GemmDefault>{});
    if(ck::get_device_name() == "gfx950")
    {
        add_device_operation_instances(
            instances,
            device_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn_mem_instances_dr<Interwave,
                                                                           GemmDefault>{});
    }
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
