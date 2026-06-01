// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "device_gemm_wmma_universal_f8_f8_bf16_mk_kn_mn.hpp"
#include "ck/host_utility/device_prop.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_wmma_universal_f8_f8_bf16_mk_kn_mn_comp_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Row, Row, F8, F8, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances)
{
    if(ck::is_gfx11_supported())
        return;

    add_device_operation_instances(
        instances, device_gemm_wmma_universal_f8_f8_bf16_mk_kn_mn_comp_instances<GemmDefault>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
