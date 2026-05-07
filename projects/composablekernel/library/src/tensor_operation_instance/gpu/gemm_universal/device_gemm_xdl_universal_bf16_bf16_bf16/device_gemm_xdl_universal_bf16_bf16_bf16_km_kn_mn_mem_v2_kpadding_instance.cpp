// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "device_gemm_xdl_universal_bf16_bf16_bf16_km_kn_mn.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_xdl_universal_bf16_bf16_bf16_km_kn_mn_mem_v2_kpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Col, Row, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances)
{
    add_device_operation_instances(
        instances,
        device_gemm_xdl_universal_bf16_bf16_bf16_km_kn_mn_mem_instances<Interwave, GemmKPadding>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
