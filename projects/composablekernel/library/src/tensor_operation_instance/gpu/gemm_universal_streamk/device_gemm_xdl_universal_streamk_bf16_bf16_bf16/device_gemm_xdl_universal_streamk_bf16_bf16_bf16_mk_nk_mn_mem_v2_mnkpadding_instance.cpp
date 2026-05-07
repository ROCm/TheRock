// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "device_gemm_xdl_universal_streamk_bf16_bf16_bf16_mk_nk_mn.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_xdl_universal_streamk_bf16_bf16_bf16_mk_nk_mn_mem_v2_mnkpadding_instances(
    std::vector<std::unique_ptr<DeviceGemm_Streamk_V2<Row,
                                                      Col,
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
        device_gemm_xdl_universal_streamk_bf16_bf16_bf16_mk_nk_mn_mem_instances<Interwave,
                                                                                GemmMNKPadding>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
