// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "device_gemm_multiply_multiply_wp_xdl_f8_f8_bf16_mk_mfma_mn.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma_mn_p5_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitKBPreShuffle<Row,
                                                                     Col,
                                                                     Tuple<Row, Col>,
                                                                     Row,
                                                                     F8,
                                                                     F8,
                                                                     Tuple<F32, F32>,
                                                                     BF16,
                                                                     PassThrough,
                                                                     PassThrough,
                                                                     MultiplyMultiply>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma_mn_p5_instances<
            v1,
            GemmDefault>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
