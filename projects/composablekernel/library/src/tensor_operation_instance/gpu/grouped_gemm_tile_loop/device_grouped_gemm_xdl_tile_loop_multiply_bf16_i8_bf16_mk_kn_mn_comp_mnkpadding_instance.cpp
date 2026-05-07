// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "device_grouped_gemm_xdl_tile_loop_multiply_bf16_i8_bf16_mk_kn_mn.hpp"
#include "ck/host_utility/device_prop.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_grouped_gemm_xdl_tile_loop_multiply_bf16_i8_bf16_mk_kn_mn_comp_mnkpadding_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmTileLoop<Row,
                                                          Row,
                                                          ck::Tuple<Row>,
                                                          Row,
                                                          BF16,
                                                          I8,
                                                          ck::Tuple<BF16>,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          Multiply>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_gemm_xdl_tile_loop_bf16_i8_bf16_mk_kn_mn_comp_instances<ck::Tuple<Row>,
                                                                               ck::Tuple<BF16>,
                                                                               Multiply,
                                                                               GemmMNKPadding>{});

    if(ck::get_device_name() == "gfx950")
    {
        add_device_operation_instances(
            instances,
            device_grouped_gemm_xdl_tile_loop_bf16_i8_bf16_mk_kn_mn_comp_instances_2x<
                ck::Tuple<Row>,
                ck::Tuple<BF16>,
                Multiply,
                GemmMNKPadding>{});
    }
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
