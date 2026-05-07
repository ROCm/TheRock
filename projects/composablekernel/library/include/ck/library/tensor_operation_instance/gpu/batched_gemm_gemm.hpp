// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <memory>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_gemm.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

#ifdef CK_USE_WMMA
#ifdef CK_ENABLE_BF16
void add_device_batched_gemm_gemm_wmma_cshuffle_v3_bf16_bf16_bf16_bf16_gmk_gnk_gno_gmo_instance(
    std::vector<std::unique_ptr<DeviceBatchedGemmGemm<Row,
                                                      Col,
                                                      Row,
                                                      Row,
                                                      BF16,
                                                      BF16,
                                                      BF16,
                                                      BF16,
                                                      PassThrough,
                                                      PassThrough,
                                                      PassThrough,
                                                      PassThrough,
                                                      PassThrough>>>& instances);
void add_device_batched_gemm_gemm_wmma_cshuffle_v3_bf16_bf16_bf16_bf16_gmk_gnk_gon_gmo_instance(
    std::vector<std::unique_ptr<DeviceBatchedGemmGemm<Row,
                                                      Col,
                                                      Col,
                                                      Row,
                                                      BF16,
                                                      BF16,
                                                      BF16,
                                                      BF16,
                                                      PassThrough,
                                                      PassThrough,
                                                      PassThrough,
                                                      PassThrough,
                                                      PassThrough>>>& instances);
#endif // CK_ENABLE_BF16
#ifdef CK_ENABLE_FP16
void add_device_batched_gemm_gemm_wmma_cshuffle_v3_f16_f16_f16_f16_gmk_gnk_gno_gmo_instance(
    std::vector<std::unique_ptr<DeviceBatchedGemmGemm<Row,
                                                      Col,
                                                      Row,
                                                      Row,
                                                      F16,
                                                      F16,
                                                      F16,
                                                      F16,
                                                      PassThrough,
                                                      PassThrough,
                                                      PassThrough,
                                                      PassThrough,
                                                      PassThrough>>>& instances);
void add_device_batched_gemm_gemm_wmma_cshuffle_v3_f16_f16_f16_f16_gmk_gnk_gon_gmo_instance(
    std::vector<std::unique_ptr<DeviceBatchedGemmGemm<Row,
                                                      Col,
                                                      Col,
                                                      Row,
                                                      F16,
                                                      F16,
                                                      F16,
                                                      F16,
                                                      PassThrough,
                                                      PassThrough,
                                                      PassThrough,
                                                      PassThrough,
                                                      PassThrough>>>& instances);
#endif // CK_ENABLE_FP16
#endif // CK_USE_WMMA
#ifdef CK_USE_XDL
#ifdef CK_ENABLE_FP16
void add_device_batched_gemm_gemm_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instance(
    std::vector<std::unique_ptr<DeviceBatchedGemmGemm<Row,
                                                      Col,
                                                      Row,
                                                      Row,
                                                      F16,
                                                      F16,
                                                      F16,
                                                      F16,
                                                      PassThrough,
                                                      PassThrough,
                                                      PassThrough,
                                                      PassThrough,
                                                      PassThrough>>>& instances);

void add_device_batched_gemm_gemm_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gon_gmo_instance(
    std::vector<std::unique_ptr<DeviceBatchedGemmGemm<Row,
                                                      Col,
                                                      Col,
                                                      Row,
                                                      F16,
                                                      F16,
                                                      F16,
                                                      F16,
                                                      PassThrough,
                                                      PassThrough,
                                                      PassThrough,
                                                      PassThrough,
                                                      PassThrough>>>& instances);
#endif // CK_ENABLE_FP16
#endif // CK_USE_XDL
template <typename ALayout,
          typename B0Layout,
          typename B1Layout,
          typename CLayout,
          typename ADataType,
          typename B0DataType,
          typename B1DataType,
          typename CDataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceBatchedGemmGemm<ALayout,
                                                        B0Layout,
                                                        B1Layout,
                                                        CLayout,
                                                        ADataType,
                                                        B0DataType,
                                                        B1DataType,
                                                        CDataType,
                                                        PassThrough,
                                                        PassThrough,
                                                        PassThrough,
                                                        PassThrough,
                                                        PassThrough>>
{
    using DeviceOp = DeviceBatchedGemmGemm<ALayout,
                                           B0Layout,
                                           B1Layout,
                                           CLayout,
                                           ADataType,
                                           B0DataType,
                                           B1DataType,
                                           CDataType,
                                           PassThrough,
                                           PassThrough,
                                           PassThrough,
                                           PassThrough,
                                           PassThrough>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;
#ifdef CK_USE_WMMA
#ifdef CK_ENABLE_BF16
        if constexpr(is_same_v<ADataType, bhalf_t> && is_same_v<B0DataType, bhalf_t> &&
                     is_same_v<B1DataType, bhalf_t> && is_same_v<CDataType, bhalf_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<B0Layout, Col> &&
                         is_same_v<B1Layout, Row> && is_same_v<CLayout, Row>)
            {
                add_device_batched_gemm_gemm_wmma_cshuffle_v3_bf16_bf16_bf16_bf16_gmk_gnk_gno_gmo_instance(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<B0Layout, Col> &&
                              is_same_v<B1Layout, Col> && is_same_v<CLayout, Row>)
            {
                add_device_batched_gemm_gemm_wmma_cshuffle_v3_bf16_bf16_bf16_bf16_gmk_gnk_gon_gmo_instance(
                    op_ptrs);
            }
        }
#endif // CK_ENABLE_BF16
#ifdef CK_ENABLE_FP16
        if constexpr(is_same_v<ADataType, half_t> && is_same_v<B0DataType, half_t> &&
                     is_same_v<B1DataType, half_t> && is_same_v<CDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<B0Layout, Col> &&
                         is_same_v<B1Layout, Row> && is_same_v<CLayout, Row>)
            {
                add_device_batched_gemm_gemm_wmma_cshuffle_v3_f16_f16_f16_f16_gmk_gnk_gno_gmo_instance(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<B0Layout, Col> &&
                              is_same_v<B1Layout, Col> && is_same_v<CLayout, Row>)
            {
                add_device_batched_gemm_gemm_wmma_cshuffle_v3_f16_f16_f16_f16_gmk_gnk_gon_gmo_instance(
                    op_ptrs);
            }
        }
#endif // CK_ENABLE_FP16
#endif // CK_USE_WMMA
#ifdef CK_USE_XDL
#ifdef CK_ENABLE_FP16
        if constexpr(is_same_v<ADataType, half_t> && is_same_v<B0DataType, half_t> &&
                     is_same_v<B1DataType, half_t> && is_same_v<CDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<B0Layout, Col> &&
                         is_same_v<B1Layout, Row> && is_same_v<CLayout, Row>)
            {
                add_device_batched_gemm_gemm_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instance(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<B0Layout, Col> &&
                              is_same_v<B1Layout, Col> && is_same_v<CLayout, Row>)
            {
                add_device_batched_gemm_gemm_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gon_gmo_instance(
                    op_ptrs);
            }
        }
#endif // CK_ENABLE_FP16
#endif // CK_USE_XDL
        return op_ptrs;
    }
};
} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
