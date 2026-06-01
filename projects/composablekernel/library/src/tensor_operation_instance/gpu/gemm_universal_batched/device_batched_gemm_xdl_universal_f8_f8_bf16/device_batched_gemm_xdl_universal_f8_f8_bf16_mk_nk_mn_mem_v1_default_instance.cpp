// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "device_batched_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_batched_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn_mem_v1_default_instances(
    std::vector<std::unique_ptr<DeviceBatchedGemmV2MultiD<Row,
                                                          Col,
                                                          ck::Tuple<>,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          ck::Tuple<>,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          PassThrough>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_batched_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn_mem_instances<Intrawave,
                                                                            GemmDefault>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
