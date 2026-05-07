// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "device_gemm_b_scale_xdl_f16_i4_f16_mk_nk_mn.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
void add_device_gemm_b_scale_xdl_f16_i4_f16_mk_nk_mn_mem_v2_default_instances(
    std::vector<std::unique_ptr<DeviceGemmV2BScale<Row,
                                                   Col,
                                                   Row,
                                                   F16,
                                                   I4,
                                                   F16,
                                                   F16,
                                                   1,
                                                   128,
                                                   PassThrough,
                                                   PassThrough,
                                                   PassThrough>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_gemm_b_scale_xdl_f16_i4_f16_mk_nk_mn_mem_instances<Intrawave, GemmDefault>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
