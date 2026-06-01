// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "device_gemm_multiply_multiply_xdl_i8_i8_bf16_mk_nk_mn.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_multiply_multiply_xdl_i8_i8_bf16_mk_nk_mn_mem_v2_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          I8,
                                                          I8,
                                                          Tuple<F32, F32>,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_gemm_multiply_multiply_xdl_i8_i8_bf16_mk_nk_mn_mem_instances<Interwave,
                                                                            GemmKPadding>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
