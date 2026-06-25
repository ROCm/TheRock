// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "device_gemm_xdl_universal_streamk_f8_f16_f16_mk_nk_mn.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_xdl_universal_streamk_f8_f16_f16_mk_nk_mn_comp_mnpadding_instances(
    std::vector<std::unique_ptr<
        DeviceGemm_Streamk_V2<Row, Col, Row, F8, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances)
{
    add_device_operation_instances(
        instances,
        device_gemm_xdl_universal_streamk_f8_f16_f16_mk_nk_mn_comp_instances<GemmMNPadding>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
