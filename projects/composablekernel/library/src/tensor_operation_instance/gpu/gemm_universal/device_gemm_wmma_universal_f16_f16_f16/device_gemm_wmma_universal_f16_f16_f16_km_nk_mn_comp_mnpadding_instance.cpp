// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "device_gemm_wmma_universal_f16_f16_f16_km_nk_mn.hpp"
#include "ck/host_utility/device_prop.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_wmma_universal_f16_f16_f16_km_nk_mn_comp_mnpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Col, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances)
{
    add_device_operation_instances(
        instances, device_gemm_wmma_universal_f16_f16_f16_km_nk_mn_comp_instances<GemmMNPadding>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
