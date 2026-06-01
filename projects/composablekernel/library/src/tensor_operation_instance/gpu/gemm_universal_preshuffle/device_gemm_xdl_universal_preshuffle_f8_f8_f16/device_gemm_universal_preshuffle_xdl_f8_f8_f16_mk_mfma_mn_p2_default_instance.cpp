// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma_mn.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p2_default_instances(
    std::vector<std::unique_ptr<DeviceGemmV2BPreshuffle<Row,
                                                        Col,
                                                        Row,
                                                        F8,
                                                        F8,
                                                        F16,
                                                        PassThrough,
                                                        PassThrough,
                                                        PassThrough>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p2_instances<v1, GemmDefault>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
