// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_streamk.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
#ifdef CK_ENABLE_FP16
void add_device_gemm_xdl_streamk_f16_f16_f16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemmStreamK<Row, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
struct DeviceOperationInstanceFactory<ck::tensor_operation::device::DeviceGemmStreamK<
    ALayout,
    BLayout,
    CLayout,
    ADataType,
    BDataType,
    CDataType,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough>>
{
    using DeviceOp = DeviceGemmStreamK<ALayout,
                                       BLayout,
                                       CLayout,
                                       ADataType,
                                       BDataType,
                                       CDataType,
                                       ck::tensor_operation::element_wise::PassThrough,
                                       ck::tensor_operation::element_wise::PassThrough,
                                       ck::tensor_operation::element_wise::PassThrough>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;
#if 0
        if constexpr(is_same_v<ADataType, float> && is_same_v<BDataType, float> &&
                     is_same_v<CDataType, float>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_splitk_f32_f32_f32_mk_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_splitk_f32_f32_f32_mk_nk_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_splitk_f32_f32_f32_km_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_splitk_f32_f32_f32_km_nk_mn_instances(op_ptrs);
            }
        }
        else if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, half_t> &&
                          is_same_v<CDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_v1_instances(op_ptrs);
                add_device_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_v1_irregular_instances(op_ptrs);
                add_device_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_v1_interwave_instances(op_ptrs);
                add_device_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_v1_interwave_irregular_instances(op_ptrs);
                add_device_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_v2_instances(op_ptrs);
                add_device_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_v2_irregular_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_v1_instances(op_ptrs);
                add_device_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_v1_irregular__instances(op_ptrs);
                add_device_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_v1_interwave_instances(op_ptrs);
                add_device_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_v1_interwave_irregular_instances(op_ptrs);
                add_device_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_v2_instances(op_ptrs);
                add_device_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_v2_irregular_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_splitk_f16_f16_f16_km_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_splitk_f16_f16_f16_km_nk_mn_instances(op_ptrs);
            }
        }
#endif
        if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, half_t> &&
                     is_same_v<CDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_streamk_f16_f16_f16_mk_kn_mn_instances(op_ptrs);
            }
        }
        return op_ptrs;
    }
};
#endif
} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
