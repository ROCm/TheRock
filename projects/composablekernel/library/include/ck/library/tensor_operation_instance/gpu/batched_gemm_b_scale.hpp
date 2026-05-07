// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_xdl_fpAintB_b_scale.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_wmma_cshuffle_v3_b_scale.hpp"

#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include <memory>
#include <vector>

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

#if defined(CK_USE_XDL)
#if(defined(CK_ENABLE_FP16) || defined(CK_ENABLE_FP8))
void add_device_batched_gemm_b_scale_xdl_f16_i4_f16_mk_nk_mn_mem_v2_default_instances(
    std::vector<std::unique_ptr<DeviceBatchedGemmV2BScale<Row,
                                                          Col,
                                                          Row,
                                                          F16,
                                                          I4,
                                                          F16,
                                                          F16,
                                                          1,
                                                          128,
                                                          PassThrough,
                                                          PassThrough,
                                                          PassThrough>>>& instances);
#endif
#endif // CK_USE_XDL

#if defined(CK_USE_WMMA)
#if(defined(CK_ENABLE_FP16) || defined(CK_ENABLE_FP8)) // TODO: really, or?
void add_device_batched_gemm_b_scale_wmma_f16_i4_f16_mk_nk_mn_mem_default_instances(
    std::vector<std::unique_ptr<DeviceBatchedGemmV2BScale<Row,
                                                          Col,
                                                          Row,
                                                          F16,
                                                          I4,
                                                          F16,
                                                          F16,
                                                          1,
                                                          128,
                                                          PassThrough,
                                                          PassThrough,
                                                          PassThrough>>>& instances);
#endif // CK_ENABLE_FP16 || CK_ENABLE_FP8
#endif // CK_USE_WMMA

template <typename ADataType,
          typename BDataType,
          typename BScaleDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          index_t ScaleBlockK>

struct DeviceOperationInstanceFactory<ck::tensor_operation::device::DeviceBatchedGemmV2BScale<
    ALayout,
    BLayout,
    CLayout,
    ADataType,
    BDataType,
    BScaleDataType,
    CDataType,
    1,
    ScaleBlockK,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough>>
{
    using DeviceOp = DeviceBatchedGemmV2BScale<ALayout,
                                               BLayout,
                                               CLayout,
                                               ADataType,
                                               BDataType,
                                               BScaleDataType,
                                               CDataType,
                                               1,
                                               ScaleBlockK,
                                               ck::tensor_operation::element_wise::PassThrough,
                                               ck::tensor_operation::element_wise::PassThrough,
                                               ck::tensor_operation::element_wise::PassThrough>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, pk_i4_t> &&
                     is_same_v<CDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                         is_same_v<CLayout, Row>)
            {
#if defined(CK_USE_XDL)
                add_device_batched_gemm_b_scale_xdl_f16_i4_f16_mk_nk_mn_mem_v2_default_instances(
                    op_ptrs);
#endif // CK_USE_XDL
#if defined(CK_USE_WMMA)
                add_device_batched_gemm_b_scale_wmma_f16_i4_f16_mk_nk_mn_mem_default_instances(
                    op_ptrs);
#endif // CK_USE_WMMA
            }
        }

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
