// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <memory>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_d_layernorm.hpp"
#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"
#ifdef CK_ENABLE_FP16
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

#if defined(CK_USE_XDL)
void add_device_gemm_add_relu_add_xdl_c_shuffle_layernorm_f16_mk_kn_mn_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDLayernorm<Row,
                                                             Row,
                                                             Row_Row_Tuple,
                                                             Row,
                                                             F16,
                                                             F16,
                                                             F16_F16_Tuple,
                                                             F16,
                                                             F16,
                                                             F16,
                                                             PassThrough,
                                                             PassThrough,
                                                             AddReluAdd,
                                                             PassThrough>>>&);

void add_device_gemm_add_relu_add_xdl_c_shuffle_layernorm_f16_mk_nk_mn_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDLayernorm<Row,
                                                             Col,
                                                             Row_Row_Tuple,
                                                             Row,
                                                             F16,
                                                             F16,
                                                             F16_F16_Tuple,
                                                             F16,
                                                             F16,
                                                             F16,
                                                             PassThrough,
                                                             PassThrough,
                                                             AddReluAdd,
                                                             PassThrough>>>&);

void add_device_gemm_add_relu_add_xdl_c_shuffle_layernorm_f16_km_kn_mn_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDLayernorm<Col,
                                                             Row,
                                                             Row_Row_Tuple,
                                                             Row,
                                                             F16,
                                                             F16,
                                                             F16_F16_Tuple,
                                                             F16,
                                                             F16,
                                                             F16,
                                                             PassThrough,
                                                             PassThrough,
                                                             AddReluAdd,
                                                             PassThrough>>>&);

void add_device_gemm_add_relu_add_xdl_c_shuffle_layernorm_f16_km_nk_mn_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDLayernorm<Col,
                                                             Col,
                                                             Row_Row_Tuple,
                                                             Row,
                                                             F16,
                                                             F16,
                                                             F16_F16_Tuple,
                                                             F16,
                                                             F16,
                                                             F16,
                                                             PassThrough,
                                                             PassThrough,
                                                             AddReluAdd,
                                                             PassThrough>>>&);
#endif // CK_USE_XDL

#if defined(CK_USE_WMMA)
void add_device_gemm_add_relu_add_wmma_c_shuffle_layernorm_f16_mk_kn_mn_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDLayernorm<Row,
                                                             Row,
                                                             Row_Row_Tuple,
                                                             Row,
                                                             F16,
                                                             F16,
                                                             F16_F16_Tuple,
                                                             F16,
                                                             F16,
                                                             F16,
                                                             PassThrough,
                                                             PassThrough,
                                                             AddReluAdd,
                                                             PassThrough>>>&);

void add_device_gemm_add_relu_add_wmma_c_shuffle_layernorm_f16_mk_nk_mn_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDLayernorm<Row,
                                                             Col,
                                                             Row_Row_Tuple,
                                                             Row,
                                                             F16,
                                                             F16,
                                                             F16_F16_Tuple,
                                                             F16,
                                                             F16,
                                                             F16,
                                                             PassThrough,
                                                             PassThrough,
                                                             AddReluAdd,
                                                             PassThrough>>>&);

void add_device_gemm_add_relu_add_wmma_c_shuffle_layernorm_f16_km_kn_mn_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDLayernorm<Col,
                                                             Row,
                                                             Row_Row_Tuple,
                                                             Row,
                                                             F16,
                                                             F16,
                                                             F16_F16_Tuple,
                                                             F16,
                                                             F16,
                                                             F16,
                                                             PassThrough,
                                                             PassThrough,
                                                             AddReluAdd,
                                                             PassThrough>>>&);

void add_device_gemm_add_relu_add_wmma_c_shuffle_layernorm_f16_km_nk_mn_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDLayernorm<Col,
                                                             Col,
                                                             Row_Row_Tuple,
                                                             Row,
                                                             F16,
                                                             F16,
                                                             F16_F16_Tuple,
                                                             F16,
                                                             F16,
                                                             F16,
                                                             PassThrough,
                                                             PassThrough,
                                                             AddReluAdd,
                                                             PassThrough>>>&);
#endif

// GEMM + Add + Relu + Add + Layernorm
template <typename ALayout,
          typename BLayout,
          typename D0Layout,
          typename D1Layout,
          typename HLayout,
          typename ADataType,
          typename BDataType,
          typename D0DataType,
          typename D1DataType,
          typename GammaDataType,
          typename BetaDataType,
          typename HDataType>
struct DeviceOperationInstanceFactory<ck::tensor_operation::device::DeviceGemmMultipleDLayernorm<
    ALayout,
    BLayout,
    ck::Tuple<D0Layout, D1Layout>,
    HLayout,
    ADataType,
    BDataType,
    ck::Tuple<D0DataType, D1DataType>,
    GammaDataType,
    BetaDataType,
    HDataType,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::AddReluAdd,
    ck::tensor_operation::element_wise::PassThrough>>
{
    using DeviceOp = DeviceGemmMultipleDLayernorm<ALayout,
                                                  BLayout,
                                                  ck::Tuple<D0Layout, D1Layout>,
                                                  HLayout,
                                                  ADataType,
                                                  BDataType,
                                                  ck::Tuple<D0DataType, D1DataType>,
                                                  GammaDataType,
                                                  BetaDataType,
                                                  HDataType,
                                                  ck::tensor_operation::element_wise::PassThrough,
                                                  ck::tensor_operation::element_wise::PassThrough,
                                                  ck::tensor_operation::element_wise::AddReluAdd,
                                                  ck::tensor_operation::element_wise::PassThrough>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, half_t> &&
                     is_same_v<D0DataType, half_t> && is_same_v<D1DataType, half_t> &&
                     is_same_v<GammaDataType, half_t> && is_same_v<BetaDataType, half_t> &&
                     is_same_v<HDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<D0Layout, Row> && is_same_v<D1Layout, Row> &&
                         is_same_v<HLayout, Row>)
            {
#if defined(CK_USE_XDL)
                add_device_gemm_add_relu_add_xdl_c_shuffle_layernorm_f16_mk_kn_mn_mn_mn_instances(
                    op_ptrs);
#endif
#if defined(CK_USE_WMMA)
                add_device_gemm_add_relu_add_wmma_c_shuffle_layernorm_f16_mk_kn_mn_mn_mn_instances(
                    op_ptrs);
#endif
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<D0Layout, Row> && is_same_v<D1Layout, Row> &&
                              is_same_v<HLayout, Row>)
            {
#if defined(CK_USE_XDL)
                add_device_gemm_add_relu_add_xdl_c_shuffle_layernorm_f16_mk_nk_mn_mn_mn_instances(
                    op_ptrs);
#endif
#if defined(CK_USE_WMMA)
                add_device_gemm_add_relu_add_wmma_c_shuffle_layernorm_f16_mk_nk_mn_mn_mn_instances(
                    op_ptrs);
#endif
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<D0Layout, Row> && is_same_v<D1Layout, Row> &&
                              is_same_v<HLayout, Row>)
            {
#if defined(CK_USE_XDL)
                add_device_gemm_add_relu_add_xdl_c_shuffle_layernorm_f16_km_kn_mn_mn_mn_instances(
                    op_ptrs);
#endif
#if defined(CK_USE_WMMA)
                add_device_gemm_add_relu_add_wmma_c_shuffle_layernorm_f16_km_kn_mn_mn_mn_instances(
                    op_ptrs);
#endif
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<D0Layout, Row> && is_same_v<D1Layout, Row> &&
                              is_same_v<HLayout, Row>)
            {
#if defined(CK_USE_XDL)
                add_device_gemm_add_relu_add_xdl_c_shuffle_layernorm_f16_km_nk_mn_mn_mn_instances(
                    op_ptrs);
#endif
#if defined(CK_USE_WMMA)
                add_device_gemm_add_relu_add_wmma_c_shuffle_layernorm_f16_km_nk_mn_mn_mn_instances(
                    op_ptrs);
#endif
            }
        }

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
