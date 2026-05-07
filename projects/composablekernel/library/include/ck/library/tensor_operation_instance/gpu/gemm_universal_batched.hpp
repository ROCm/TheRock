// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <memory>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_xdl_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

#ifdef CK_ENABLE_BF16
void add_device_batched_gemm_xdl_universal_bf16_bf16_bf16_mk_nk_mn_comp_default_instances(
    std::vector<std::unique_ptr<DeviceBatchedGemmV2MultiD<Row,
                                                          Col,
                                                          Empty_Tuple,
                                                          Row,
                                                          BF16,
                                                          BF16,
                                                          Empty_Tuple,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          PassThrough>>>& instances);

void add_device_batched_gemm_xdl_universal_bf16_bf16_bf16_mk_nk_mn_mem_v1_default_instances(
    std::vector<std::unique_ptr<DeviceBatchedGemmV2MultiD<Row,
                                                          Col,
                                                          Empty_Tuple,
                                                          Row,
                                                          BF16,
                                                          BF16,
                                                          Empty_Tuple,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          PassThrough>>>& instances);

void add_device_batched_gemm_xdl_universal_bf16_bf16_bf16_mk_nk_mn_mem_v2_default_instances(
    std::vector<std::unique_ptr<DeviceBatchedGemmV2MultiD<Row,
                                                          Col,
                                                          Empty_Tuple,
                                                          Row,
                                                          BF16,
                                                          BF16,
                                                          Empty_Tuple,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          PassThrough>>>& instances);

#endif

#ifdef CK_ENABLE_FP8
void add_device_batched_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn_comp_default_instances(
    std::vector<std::unique_ptr<DeviceBatchedGemmV2MultiD<Row,
                                                          Col,
                                                          Empty_Tuple,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          Empty_Tuple,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          PassThrough>>>& instances);

void add_device_batched_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn_mem_v1_default_instances(
    std::vector<std::unique_ptr<DeviceBatchedGemmV2MultiD<Row,
                                                          Col,
                                                          Empty_Tuple,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          Empty_Tuple,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          PassThrough>>>& instances);

void add_device_batched_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn_mem_v2_default_instances(
    std::vector<std::unique_ptr<DeviceBatchedGemmV2MultiD<Row,
                                                          Col,
                                                          Empty_Tuple,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          Empty_Tuple,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          PassThrough>>>& instances);

#endif

template <typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout>
struct DeviceOperationInstanceFactory<ck::tensor_operation::device::DeviceBatchedGemmV2MultiD<
    ALayout,
    BLayout,
    DsLayout,
    CLayout,
    ADataType,
    BDataType,
    DsDataType,
    CDataType,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough>>
{
    using DeviceOp = DeviceBatchedGemmV2MultiD<ALayout,
                                               BLayout,
                                               DsLayout,
                                               CLayout,
                                               ADataType,
                                               BDataType,
                                               DsDataType,
                                               CDataType,
                                               ck::tensor_operation::element_wise::PassThrough,
                                               ck::tensor_operation::element_wise::PassThrough,
                                               ck::tensor_operation::element_wise::PassThrough>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

#ifdef CK_ENABLE_BF16
        if constexpr(is_same_v<ADataType, bhalf_t> && is_same_v<BDataType, bhalf_t> &&
                     is_same_v<CDataType, bhalf_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_batched_gemm_xdl_universal_bf16_bf16_bf16_mk_nk_mn_comp_default_instances(
                    op_ptrs);

                add_device_batched_gemm_xdl_universal_bf16_bf16_bf16_mk_nk_mn_mem_v1_default_instances(
                    op_ptrs);

                add_device_batched_gemm_xdl_universal_bf16_bf16_bf16_mk_nk_mn_mem_v2_default_instances(
                    op_ptrs);
            }
        }
#endif

#ifdef CK_ENABLE_FP8
        if constexpr(is_same_v<ADataType, f8_t> && is_same_v<BDataType, f8_t> &&
                     is_same_v<CDataType, bhalf_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_batched_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn_comp_default_instances(
                    op_ptrs);

                add_device_batched_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn_mem_v1_default_instances(
                    op_ptrs);

                add_device_batched_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn_mem_v2_default_instances(
                    op_ptrs);
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
