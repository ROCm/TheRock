// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "device_gemm_multiply_multiply_wp_xdl_f8_f8_f16_mk_mfma_mn.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma16x16_mn_compute_default_instances_p5(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitKBPreShuffle<Row,
                                                                     Col,
                                                                     Tuple<Row, Col>,
                                                                     Row,
                                                                     F8,
                                                                     F8,
                                                                     Tuple<F32, F32>,
                                                                     F16,
                                                                     PassThrough,
                                                                     PassThrough,
                                                                     MultiplyMultiply>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma16x16_mn_compute_instances_p5<
            GemmDefault>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
