// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "device_gemm_quantization_wmma_c_shuffle_i8_i8_i8_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_quantization_wmma_c_shuffle_i8_i8_i8_mk_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Row,
                                                          Empty_Tuple,
                                                          Row,
                                                          int8_t,
                                                          int8_t,
                                                          Empty_Tuple,
                                                          int8_t,
                                                          PassThrough,
                                                          PassThrough,
                                                          Mul_Clamp>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_gemm_quantization_wmma_c_shuffle_i8_i8_i8_mk_kn_mn_instances<
            Mul_Clamp,
            BlockGemmPipelineScheduler::Intrawave,
            BlockGemmPipelineVersion::v3>{});
    add_device_operation_instances(
        instances,
        device_gemm_quantization_wmma_c_shuffle_i8_i8_i8_mk_kn_mn_instances<
            Mul_Clamp,
            BlockGemmPipelineScheduler::Intrawave,
            BlockGemmPipelineVersion::v1>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
