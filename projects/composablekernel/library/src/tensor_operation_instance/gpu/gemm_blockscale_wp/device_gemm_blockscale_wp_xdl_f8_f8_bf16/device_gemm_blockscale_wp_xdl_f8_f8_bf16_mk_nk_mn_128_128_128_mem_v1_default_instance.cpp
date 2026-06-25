// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "device_gemm_blockscale_wp_xdl_f8_f8_bf16_mk_nk_mn_128_128_128.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_blockscale_wp_xdl_f8_f8_bf16_mk_nk_mn_1_128_128_mem_v1_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_BlockScale_BPreshuffle<Row,
                                                                           Col,
                                                                           Tuple<>,
                                                                           Row,
                                                                           F8,
                                                                           F32,
                                                                           F8,
                                                                           F32,
                                                                           Tuple<>,
                                                                           BF16,
                                                                           1,
                                                                           128,
                                                                           128,
                                                                           PassThrough,
                                                                           PassThrough,
                                                                           PassThrough>>>&
        instances)
{
    add_device_operation_instances(
        instances,
        device_gemm_blockscale_wp_xdl_f8_f8_bf16_mk_nk_mn_1_128_128_mem_instances<Intrawave,
                                                                                  GemmDefault>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
