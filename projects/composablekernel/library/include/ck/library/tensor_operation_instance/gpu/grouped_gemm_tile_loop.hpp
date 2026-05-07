// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <memory>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm_tile_loop.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

#ifdef CK_ENABLE_FP16
// fp16_output
void add_device_grouped_gemm_xdl_tile_loop_f16_f16_f16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmTileLoop<Row,
                                                          Row,
                                                          Empty_Tuple,
                                                          Row,
                                                          F16,
                                                          F16,
                                                          Empty_Tuple,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          PassThrough>>>& instances);

void add_device_grouped_gemm_xdl_tile_loop_f16_f16_f16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmTileLoop<Row,
                                                          Col,
                                                          Empty_Tuple,
                                                          Row,
                                                          F16,
                                                          F16,
                                                          Empty_Tuple,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          PassThrough>>>& instances);
#endif

template <typename ALayout,
          typename BLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename EDataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGroupedGemmTileLoop<ALayout,
                                                            BLayout,
                                                            Empty_Tuple,
                                                            ELayout,
                                                            ADataType,
                                                            BDataType,
                                                            Empty_Tuple,
                                                            EDataType,
                                                            PassThrough,
                                                            PassThrough,
                                                            PassThrough>>
{
    using DeviceOp = DeviceGroupedGemmTileLoop<ALayout,
                                               BLayout,
                                               Empty_Tuple,
                                               ELayout,
                                               ADataType,
                                               BDataType,
                                               Empty_Tuple,
                                               EDataType,
                                               PassThrough,
                                               PassThrough,
                                               PassThrough>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

#ifdef CK_ENABLE_FP16
        // fp16_output
        if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, half_t> &&
                     is_same_v<EDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_tile_loop_f16_f16_f16_mk_kn_mn_instances(op_ptrs);
            }
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_tile_loop_f16_f16_f16_mk_nk_mn_instances(op_ptrs);
            }
        }
#endif
        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
