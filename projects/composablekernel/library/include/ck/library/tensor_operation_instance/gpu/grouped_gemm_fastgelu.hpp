// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <vector>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

#if defined(CK_USE_WMMA)
#if defined(CK_ENABLE_FP16)
void add_device_grouped_gemm_fastgelu_wmma_f16_f16_f16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemm<Row,
                                                  Row,
                                                  Empty_Tuple,
                                                  Row,
                                                  F16,
                                                  F16,
                                                  Empty_Tuple,
                                                  F16,
                                                  PassThrough,
                                                  PassThrough,
                                                  FastGelu>>>& instances);

void add_device_grouped_gemm_fastgelu_wmma_f16_f16_f16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemm<Row,
                                                  Col,
                                                  Empty_Tuple,
                                                  Row,
                                                  F16,
                                                  F16,
                                                  Empty_Tuple,
                                                  F16,
                                                  PassThrough,
                                                  PassThrough,
                                                  FastGelu>>>& instances);

void add_device_grouped_gemm_fastgelu_wmma_f16_f16_f16_km_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemm<Col,
                                                  Row,
                                                  Empty_Tuple,
                                                  Row,
                                                  F16,
                                                  F16,
                                                  Empty_Tuple,
                                                  F16,
                                                  PassThrough,
                                                  PassThrough,
                                                  FastGelu>>>& instances);

void add_device_grouped_gemm_fastgelu_wmma_f16_f16_f16_km_nk_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemm<Col,
                                                  Col,
                                                  Empty_Tuple,
                                                  Row,
                                                  F16,
                                                  F16,
                                                  Empty_Tuple,
                                                  F16,
                                                  PassThrough,
                                                  PassThrough,
                                                  FastGelu>>>& instances);
#endif // CK_ENABLE_FP16
#endif // CK_USE_WMMA

#if defined(CK_USE_XDL)
#if defined(CK_ENABLE_FP16)
void add_device_grouped_gemm_fastgelu_xdl_f16_f16_f16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemm<Row,
                                                  Row,
                                                  Empty_Tuple,
                                                  Row,
                                                  F16,
                                                  F16,
                                                  Empty_Tuple,
                                                  F16,
                                                  PassThrough,
                                                  PassThrough,
                                                  FastGelu>>>& instances);

void add_device_grouped_gemm_fastgelu_xdl_f16_f16_f16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemm<Row,
                                                  Col,
                                                  Empty_Tuple,
                                                  Row,
                                                  F16,
                                                  F16,
                                                  Empty_Tuple,
                                                  F16,
                                                  PassThrough,
                                                  PassThrough,
                                                  FastGelu>>>& instances);

void add_device_grouped_gemm_fastgelu_xdl_f16_f16_f16_km_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemm<Col,
                                                  Row,
                                                  Empty_Tuple,
                                                  Row,
                                                  F16,
                                                  F16,
                                                  Empty_Tuple,
                                                  F16,
                                                  PassThrough,
                                                  PassThrough,
                                                  FastGelu>>>& instances);

void add_device_grouped_gemm_fastgelu_xdl_f16_f16_f16_km_nk_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemm<Col,
                                                  Col,
                                                  Empty_Tuple,
                                                  Row,
                                                  F16,
                                                  F16,
                                                  Empty_Tuple,
                                                  F16,
                                                  PassThrough,
                                                  PassThrough,
                                                  FastGelu>>>& instances);
#endif // CK_ENABLE_FP16
#endif // CK_USE_XDL

// GroupedGEMM + GELU
template <typename ALayout,
          typename BLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename EDataType>
struct DeviceOperationInstanceFactory<ck::tensor_operation::device::DeviceGroupedGemm<ALayout,
                                                                                      BLayout,
                                                                                      Empty_Tuple,
                                                                                      ELayout,
                                                                                      ADataType,
                                                                                      BDataType,
                                                                                      Empty_Tuple,
                                                                                      EDataType,
                                                                                      PassThrough,
                                                                                      PassThrough,
                                                                                      FastGelu>>
{
    using DeviceOp = DeviceGroupedGemm<ALayout,
                                       BLayout,
                                       Empty_Tuple,
                                       ELayout,
                                       ADataType,
                                       BDataType,
                                       Empty_Tuple,
                                       EDataType,
                                       PassThrough,
                                       PassThrough,
                                       FastGelu>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

#if defined(CK_ENABLE_FP16)
        if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, half_t> &&
                     is_same_v<EDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<ELayout, Row>)
            {
#if defined(CK_USE_XDL)
                add_device_grouped_gemm_fastgelu_xdl_f16_f16_f16_mk_kn_mn_instances(op_ptrs);
#endif
#if defined(CK_USE_WMMA)
                add_device_grouped_gemm_fastgelu_wmma_f16_f16_f16_mk_kn_mn_instances(op_ptrs);
#endif
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<ELayout, Row>)
            {
#if defined(CK_USE_XDL)
                add_device_grouped_gemm_fastgelu_xdl_f16_f16_f16_mk_nk_mn_instances(op_ptrs);
#endif
#if defined(CK_USE_WMMA)
                add_device_grouped_gemm_fastgelu_wmma_f16_f16_f16_mk_nk_mn_instances(op_ptrs);
#endif
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<ELayout, Row>)
            {
#if defined(CK_USE_XDL)
                add_device_grouped_gemm_fastgelu_xdl_f16_f16_f16_km_kn_mn_instances(op_ptrs);
#endif
#if defined(CK_USE_WMMA)
                add_device_grouped_gemm_fastgelu_wmma_f16_f16_f16_km_kn_mn_instances(op_ptrs);
#endif
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<ELayout, Row>)
            {
#if defined(CK_USE_XDL)
                add_device_grouped_gemm_fastgelu_xdl_f16_f16_f16_km_nk_mn_instances(op_ptrs);
#endif
#if defined(CK_USE_WMMA)
                add_device_grouped_gemm_fastgelu_wmma_f16_f16_f16_km_nk_mn_instances(op_ptrs);
#endif
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
