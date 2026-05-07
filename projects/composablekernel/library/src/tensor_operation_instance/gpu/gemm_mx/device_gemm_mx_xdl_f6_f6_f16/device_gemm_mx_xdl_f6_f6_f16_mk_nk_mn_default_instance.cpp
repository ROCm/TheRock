// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "device_gemm_mx_xdl_f6_f6_f16_mk_nk_mn.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_mx_xdl_f6_f6_f16_mk_nk_mn_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMX<Row,
                                             Col,
                                             Row,
                                             F6,
                                             E8M0PK,
                                             F6,
                                             E8M0PK,
                                             F16,
                                             32,
                                             PassThrough,
                                             PassThrough,
                                             PassThrough>>>& instances)
{
    add_device_operation_instances(
        instances, device_gemm_mx_xdl_f6_f6_f16_mk_nk_mn_instances<Intrawave, GemmDefault>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
