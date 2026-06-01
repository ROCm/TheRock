// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <memory>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

#if defined(CK_USE_WMMA)
#if defined(CK_ENABLE_FP16)
void add_device_grouped_gemm_wmma_universal_f16_f16_f16_mk_nk_mn_instances(
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
                                                  PassThrough>>>& instances);

void add_device_grouped_gemm_wmma_universal_f16_f16_f16_mk_kn_mn_instances(
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
                                                  PassThrough>>>& instances);

void add_device_grouped_gemm_wmma_universal_f16_f16_f16_km_kn_mn_instances(
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
                                                  PassThrough>>>& instances);

void add_device_grouped_gemm_wmma_universal_f16_f16_f16_km_nk_mn_instances(
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
                                                  PassThrough>>>& instances);
#endif // CK_ENABLE_FP16
#if defined(CK_ENABLE_FP16) && defined(CK_ENABLE_FP8) && defined(__gfx12__)
void add_device_grouped_gemm_wmma_universal_f16_f8_f16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemm<Row,
                                                  Row,
                                                  Empty_Tuple,
                                                  Row,
                                                  F16,
                                                  F8,
                                                  Empty_Tuple,
                                                  F16,
                                                  PassThrough,
                                                  PassThrough,
                                                  PassThrough>>>& instances);

void add_device_grouped_gemm_wmma_universal_f8_f16_f16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemm<Row,
                                                  Row,
                                                  Empty_Tuple,
                                                  Row,
                                                  F8,
                                                  F16,
                                                  Empty_Tuple,
                                                  F16,
                                                  PassThrough,
                                                  PassThrough,
                                                  PassThrough>>>& instances);
#endif
#if defined(CK_ENABLE_BF16)
void add_device_grouped_gemm_wmma_universal_bf16_bf16_bf16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemm<Row,
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

void add_device_grouped_gemm_wmma_universal_bf16_bf16_bf16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemm<Row,
                                                  Row,
                                                  Empty_Tuple,
                                                  Row,
                                                  BF16,
                                                  BF16,
                                                  Empty_Tuple,
                                                  BF16,
                                                  PassThrough,
                                                  PassThrough,
                                                  PassThrough>>>& instances);

void add_device_grouped_gemm_wmma_universal_bf16_bf16_bf16_km_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemm<Col,
                                                  Row,
                                                  Empty_Tuple,
                                                  Row,
                                                  BF16,
                                                  BF16,
                                                  Empty_Tuple,
                                                  BF16,
                                                  PassThrough,
                                                  PassThrough,
                                                  PassThrough>>>& instances);

void add_device_grouped_gemm_wmma_universal_bf16_bf16_bf16_km_nk_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemm<Col,
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
#endif // CK_ENABLE_BF16
#endif // CK_USE_WMMA

#if defined(CK_USE_XDL)
#if defined(CK_ENABLE_FP16)
void add_device_grouped_gemm_xdl_f16_f16_f16_mk_kn_mn_instances(
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
                                                  PassThrough>>>& instances);

void add_device_grouped_gemm_xdl_f16_f16_f16_mk_nk_mn_instances(
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
                                                  PassThrough>>>& instances);

void add_device_grouped_gemm_xdl_f16_f16_f16_km_kn_mn_instances(
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
                                                  PassThrough>>>& instances);

void add_device_grouped_gemm_xdl_f16_f16_f16_km_nk_mn_instances(
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
                                                  PassThrough>>>& instances);

void add_device_grouped_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_instances(
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
                                                  PassThrough>>>& instances);

void add_device_grouped_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_instances(
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
                                                  PassThrough>>>& instances);

void add_device_grouped_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_irregular_pv1_inter_instances(
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
                                                  PassThrough>>>& instances);

void add_device_grouped_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_irregular_pv1_instances(
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
                                                  PassThrough>>>& instances);

void add_device_grouped_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_irregular_pv2_instances(
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
                                                  PassThrough>>>& instances);

void add_device_grouped_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_irregular_instances(
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
                                                  PassThrough>>>& instances);

void add_device_grouped_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_irregular_instances(
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
                                                  PassThrough>>>& instances);

void add_device_grouped_gemm_multiple_d_xdl_two_stage_f16_f16_f16_mk_kn_mn_instances(
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
                                                  PassThrough>>>& instances);
#endif

#if defined(CK_ENABLE_FP16) && defined(CK_ENABLE_FP8)
void add_device_grouped_gemm_xdl_splitk_f16_f8_f16_mk_kn_mn_irregular_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemm<Row,
                                                  Row,
                                                  Empty_Tuple,
                                                  Row,
                                                  F16,
                                                  F8,
                                                  Empty_Tuple,
                                                  F16,
                                                  PassThrough,
                                                  PassThrough,
                                                  PassThrough>>>& instances);

void add_device_grouped_gemm_xdl_splitk_f8_f16_f16_mk_kn_mn_irregular_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemm<Row,
                                                  Row,
                                                  Empty_Tuple,
                                                  Row,
                                                  F8,
                                                  F16,
                                                  Empty_Tuple,
                                                  F16,
                                                  PassThrough,
                                                  PassThrough,
                                                  PassThrough>>>& instances);
#endif

#if defined(CK_ENABLE_BF16)
void add_device_grouped_gemm_multiple_d_xdl_two_stage_bf16_bf16_bf16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemm<Row,
                                                  Row,
                                                  Empty_Tuple,
                                                  Row,
                                                  BF16,
                                                  BF16,
                                                  Empty_Tuple,
                                                  BF16,
                                                  PassThrough,
                                                  PassThrough,
                                                  PassThrough>>>& instances);

void add_device_grouped_gemm_multiple_d_xdl_two_stage_bf16_bf16_bf16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemm<Row,
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

void add_device_grouped_gemm_xdl_splitk_bf16_bf16_bf16_mk_kn_mn_irregular_pv1_inter_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemm<Row,
                                                  Row,
                                                  Empty_Tuple,
                                                  Row,
                                                  BF16,
                                                  BF16,
                                                  Empty_Tuple,
                                                  BF16,
                                                  PassThrough,
                                                  PassThrough,
                                                  PassThrough>>>& instances);

void add_device_grouped_gemm_xdl_splitk_bf16_bf16_bf16_mk_kn_mn_irregular_pv1_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemm<Row,
                                                  Row,
                                                  Empty_Tuple,
                                                  Row,
                                                  BF16,
                                                  BF16,
                                                  Empty_Tuple,
                                                  BF16,
                                                  PassThrough,
                                                  PassThrough,
                                                  PassThrough>>>& instances);

void add_device_grouped_gemm_xdl_splitk_bf16_bf16_bf16_mk_kn_mn_irregular_pv2_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemm<Row,
                                                  Row,
                                                  Empty_Tuple,
                                                  Row,
                                                  BF16,
                                                  BF16,
                                                  Empty_Tuple,
                                                  BF16,
                                                  PassThrough,
                                                  PassThrough,
                                                  PassThrough>>>& instances);

void add_device_grouped_gemm_xdl_splitk_bf16_bf16_bf16_mk_nk_mn_irregular_pv1_inter_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemm<Row,
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

void add_device_grouped_gemm_xdl_splitk_bf16_bf16_bf16_mk_nk_mn_irregular_pv1_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemm<Row,
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

void add_device_grouped_gemm_xdl_splitk_bf16_bf16_bf16_mk_nk_mn_irregular_pv2_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemm<Row,
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

void add_device_grouped_gemm_xdl_splitk_bf16_bf16_bf16_km_kn_mn_irregular_pv1_inter_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemm<Col,
                                                  Row,
                                                  Empty_Tuple,
                                                  Row,
                                                  BF16,
                                                  BF16,
                                                  Empty_Tuple,
                                                  BF16,
                                                  PassThrough,
                                                  PassThrough,
                                                  PassThrough>>>& instances);

void add_device_grouped_gemm_xdl_splitk_bf16_bf16_bf16_km_kn_mn_irregular_pv1_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemm<Col,
                                                  Row,
                                                  Empty_Tuple,
                                                  Row,
                                                  BF16,
                                                  BF16,
                                                  Empty_Tuple,
                                                  BF16,
                                                  PassThrough,
                                                  PassThrough,
                                                  PassThrough>>>& instances);

void add_device_grouped_gemm_xdl_splitk_bf16_bf16_bf16_km_kn_mn_irregular_pv2_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemm<Col,
                                                  Row,
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

#if defined(CK_ENABLE_BF16) && defined(CK_ENABLE_INT8)
void add_device_grouped_gemm_multiple_d_xdl_two_stage_bf16_i8_bf16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemm<Row,
                                                  Row,
                                                  Empty_Tuple,
                                                  Row,
                                                  BF16,
                                                  I8,
                                                  Empty_Tuple,
                                                  BF16,
                                                  PassThrough,
                                                  PassThrough,
                                                  PassThrough>>>& instances);

void add_device_grouped_gemm_multiple_d_xdl_two_stage_bf16_i8_bf16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemm<Row,
                                                  Col,
                                                  Empty_Tuple,
                                                  Row,
                                                  BF16,
                                                  I8,
                                                  Empty_Tuple,
                                                  BF16,
                                                  PassThrough,
                                                  PassThrough,
                                                  PassThrough>>>& instances);
#endif
#endif // CK_USE_XDL
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
                                                                                      PassThrough>>
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
                                       PassThrough>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;
#if defined(CK_USE_WMMA)
#if defined(CK_ENABLE_FP16)
        if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, half_t> &&
                     is_same_v<EDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_wmma_universal_f16_f16_f16_mk_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_wmma_universal_f16_f16_f16_mk_nk_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_wmma_universal_f16_f16_f16_km_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_wmma_universal_f16_f16_f16_km_nk_mn_instances(op_ptrs);
            }
        }
#endif // CK_ENABLE_FP16
#if defined(CK_ENABLE_FP16) && defined(CK_ENABLE_FP8) && defined(__gfx12__)
        if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, f8_t> &&
                     is_same_v<EDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_wmma_universal_f16_f8_f16_mk_kn_mn_instances(op_ptrs);
            }
        }
        else if constexpr(is_same_v<ADataType, f8_t> && is_same_v<BDataType, half_t> &&
                          is_same_v<EDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_wmma_universal_f8_f16_f16_mk_kn_mn_instances(op_ptrs);
            }
        }
#endif
#if defined(CK_ENABLE_BF16)
        if constexpr(is_same_v<ADataType, bhalf_t> && is_same_v<BDataType, bhalf_t> &&
                     is_same_v<EDataType, bhalf_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_wmma_universal_bf16_bf16_bf16_mk_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_wmma_universal_bf16_bf16_bf16_mk_nk_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_wmma_universal_bf16_bf16_bf16_km_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_wmma_universal_bf16_bf16_bf16_km_nk_mn_instances(op_ptrs);
            }
        }
#endif // CK_ENABLE_BF16
#endif // CK_USE_WMMA

#if defined(CK_USE_XDL)
#if defined(CK_ENABLE_FP16)
        if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, half_t> &&
                     is_same_v<EDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_f16_f16_f16_mk_kn_mn_instances(op_ptrs);
                add_device_grouped_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_instances(op_ptrs);
                add_device_grouped_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_irregular_pv1_inter_instances(
                    op_ptrs);
                add_device_grouped_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_irregular_pv1_instances(
                    op_ptrs);
                add_device_grouped_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_irregular_pv2_instances(
                    op_ptrs);
                add_device_grouped_gemm_multiple_d_xdl_two_stage_f16_f16_f16_mk_kn_mn_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_f16_f16_f16_mk_nk_mn_instances(op_ptrs);
                add_device_grouped_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_instances(op_ptrs);
                add_device_grouped_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_irregular_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_f16_f16_f16_km_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_f16_f16_f16_km_nk_mn_instances(op_ptrs);
            }
        }
#endif
#if defined(CK_ENABLE_FP16) && defined(CK_ENABLE_FP8)
        if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, f8_t> &&
                     is_same_v<EDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_splitk_f16_f8_f16_mk_kn_mn_irregular_instances(op_ptrs);
            }
        }
        else if constexpr(is_same_v<ADataType, f8_t> && is_same_v<BDataType, half_t> &&
                          is_same_v<EDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_splitk_f8_f16_f16_mk_kn_mn_irregular_instances(op_ptrs);
            }
        }
#endif
#if defined(CK_ENABLE_BF16) && defined(CK_ENABLE_INT8)
        if constexpr(is_same_v<ADataType, bhalf_t> && is_same_v<BDataType, int8_t> &&
                     is_same_v<EDataType, bhalf_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_multiple_d_xdl_two_stage_bf16_i8_bf16_mk_kn_mn_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_multiple_d_xdl_two_stage_bf16_i8_bf16_mk_nk_mn_instances(
                    op_ptrs);
            }
        }
#endif
#if defined(CK_ENABLE_BF16)
        if constexpr(is_same_v<ADataType, bhalf_t> && is_same_v<BDataType, bhalf_t> &&
                     is_same_v<EDataType, bhalf_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_multiple_d_xdl_two_stage_bf16_bf16_bf16_mk_kn_mn_instances(
                    op_ptrs);
                add_device_grouped_gemm_xdl_splitk_bf16_bf16_bf16_mk_kn_mn_irregular_pv1_inter_instances(
                    op_ptrs);
                add_device_grouped_gemm_xdl_splitk_bf16_bf16_bf16_mk_kn_mn_irregular_pv1_instances(
                    op_ptrs);
                add_device_grouped_gemm_xdl_splitk_bf16_bf16_bf16_mk_kn_mn_irregular_pv2_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_multiple_d_xdl_two_stage_bf16_bf16_bf16_mk_nk_mn_instances(
                    op_ptrs);
                add_device_grouped_gemm_xdl_splitk_bf16_bf16_bf16_mk_nk_mn_irregular_pv1_inter_instances(
                    op_ptrs);
                add_device_grouped_gemm_xdl_splitk_bf16_bf16_bf16_mk_nk_mn_irregular_pv1_instances(
                    op_ptrs);
                add_device_grouped_gemm_xdl_splitk_bf16_bf16_bf16_mk_nk_mn_irregular_pv2_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_splitk_bf16_bf16_bf16_km_kn_mn_irregular_pv1_inter_instances(
                    op_ptrs);
                add_device_grouped_gemm_xdl_splitk_bf16_bf16_bf16_km_kn_mn_irregular_pv1_instances(
                    op_ptrs);
                add_device_grouped_gemm_xdl_splitk_bf16_bf16_bf16_km_kn_mn_irregular_pv2_instances(
                    op_ptrs);
            }
        }
#endif
#endif // CK_USE_XDL
        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
