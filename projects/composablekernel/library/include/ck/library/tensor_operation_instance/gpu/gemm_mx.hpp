// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <memory>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3_mx.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_mx_xdl_f8_f8_f16_mk_nk_mn_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMX<Row,
                                             Col,
                                             Row,
                                             F8,
                                             E8M0PK,
                                             F8,
                                             E8M0PK,
                                             F16,
                                             32,
                                             PassThrough,
                                             PassThrough,
                                             PassThrough>>>& instances);

void add_device_gemm_mx_xdl_f8_f8_bf16_mk_nk_mn_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMX<Row,
                                             Col,
                                             Row,
                                             F8,
                                             E8M0PK,
                                             F8,
                                             E8M0PK,
                                             BF16,
                                             32,
                                             PassThrough,
                                             PassThrough,
                                             PassThrough>>>& instances);

void add_device_gemm_mx_xdl_f4_f4_f16_mk_nk_mn_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMX<Row,
                                             Col,
                                             Row,
                                             F4,
                                             I32,
                                             F4,
                                             I32,
                                             F16,
                                             32,
                                             PassThrough,
                                             PassThrough,
                                             PassThrough>>>& instances);

void add_device_gemm_mx_xdl_bf8_f8_f16_mk_kn_mn_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMX<Row,
                                             Row,
                                             Row,
                                             BF8,
                                             E8M0PK,
                                             F8,
                                             E8M0PK,
                                             F16,
                                             32,
                                             PassThrough,
                                             PassThrough,
                                             PassThrough>>>& instances);

void add_device_gemm_mx_xdl_f8_f8_bf16_km_nk_mn_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMX<Col,
                                             Col,
                                             Row,
                                             F8,
                                             E8M0PK,
                                             F8,
                                             E8M0PK,
                                             BF16,
                                             32,
                                             PassThrough,
                                             PassThrough,
                                             PassThrough>>>& instances);

void add_device_gemm_mx_xdl_f6_f6_f16_mk_nk_mn_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMX<Row,
                                             Col,
                                             Row,
                                             F6,
                                             E8M0PK,
                                             F6,
                                             E8M0PK,
                                             F16,
                                             32,
                                             PassThrough,
                                             PassThrough,
                                             PassThrough>>>& instances);

void add_device_gemm_mx_xdl_bf6_bf6_bf16_mk_nk_mn_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMX<Row,
                                             Col,
                                             Row,
                                             BF6,
                                             E8M0PK,
                                             BF6,
                                             E8M0PK,
                                             BF16,
                                             32,
                                             PassThrough,
                                             PassThrough,
                                             PassThrough>>>& instances);

template <typename ADataType,
          typename AScaleDataType,
          typename BDataType,
          typename BScaleDataType,
          typename CDataType,
          index_t ScaleBlockSize,
          typename ALayout,
          typename BLayout,
          typename CLayout>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGemmMX<ALayout,
                                               BLayout,
                                               CLayout,
                                               ADataType,
                                               AScaleDataType,
                                               BDataType,
                                               BScaleDataType,
                                               CDataType,
                                               ScaleBlockSize,
                                               ck::tensor_operation::element_wise::PassThrough,
                                               ck::tensor_operation::element_wise::PassThrough,
                                               ck::tensor_operation::element_wise::PassThrough>,
    enable_if_t<!is_same_v<BLayout, MFMA>>> // non-weight-pre-shuffle
{
    using DeviceOp = DeviceGemmMX<ALayout,
                                  BLayout,
                                  CLayout,
                                  ADataType,
                                  AScaleDataType,
                                  BDataType,
                                  BScaleDataType,
                                  CDataType,
                                  ScaleBlockSize,
                                  ck::tensor_operation::element_wise::PassThrough,
                                  ck::tensor_operation::element_wise::PassThrough,
                                  ck::tensor_operation::element_wise::PassThrough>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> && is_same_v<CLayout, Row>)
        {
            // Row-Col-Row -- one of the two currently supported layouts, another one is
            // Row-MFMA-Row
            if constexpr(is_same_v<ADataType, F8> && is_same_v<BDataType, F8> &&
                         is_same_v<CDataType, F16>)
            {

                add_device_gemm_mx_xdl_f8_f8_f16_mk_nk_mn_default_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ADataType, F8> && is_same_v<BDataType, F8> &&
                              is_same_v<CDataType, BF16>)
            {

                add_device_gemm_mx_xdl_f8_f8_bf16_mk_nk_mn_default_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ADataType, F4> && is_same_v<BDataType, F4> &&
                              is_same_v<CDataType, F16>)
            {
                add_device_gemm_mx_xdl_f4_f4_f16_mk_nk_mn_default_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ADataType, F6> && is_same_v<BDataType, F6> &&
                              is_same_v<CDataType, F16>)
            {
                add_device_gemm_mx_xdl_f6_f6_f16_mk_nk_mn_default_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ADataType, BF6> && is_same_v<BDataType, BF6> &&
                              is_same_v<CDataType, BF16>)
            {
                add_device_gemm_mx_xdl_bf6_bf6_bf16_mk_nk_mn_default_instances(op_ptrs);
            }
        }
        else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                          is_same_v<CLayout, Row>)
        {
            if constexpr(is_same_v<ADataType, BF8> && is_same_v<BDataType, F8> &&
                         is_same_v<CDataType, F16>)
            {

                add_device_gemm_mx_xdl_bf8_f8_f16_mk_kn_mn_default_instances(op_ptrs);
            }
        }
        else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                          is_same_v<CLayout, Row>)
        {
            if constexpr(is_same_v<ADataType, F8> && is_same_v<BDataType, F8> &&
                         is_same_v<CDataType, BF16>)
            {

                add_device_gemm_mx_xdl_f8_f8_bf16_km_nk_mn_default_instances(op_ptrs);
            }
        }

        return op_ptrs;
    }
};

void add_device_gemm_mx_xdl_f4_f4_f16_mk_mfma_mn_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMX<Row,
                                             MFMA,
                                             Row,
                                             F4,
                                             I32,
                                             F4,
                                             I32,
                                             F16,
                                             32,
                                             PassThrough,
                                             PassThrough,
                                             PassThrough>>>& instances);

template <typename ADataType,
          typename AScaleDataType,
          typename BDataType,
          typename BScaleDataType,
          typename CDataType,
          index_t ScaleBlockSize,
          typename ALayout,
          typename BLayout,
          typename CLayout>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGemmMX<ALayout,
                                               BLayout,
                                               CLayout,
                                               ADataType,
                                               AScaleDataType,
                                               BDataType,
                                               BScaleDataType,
                                               CDataType,
                                               ScaleBlockSize,
                                               ck::tensor_operation::element_wise::PassThrough,
                                               ck::tensor_operation::element_wise::PassThrough,
                                               ck::tensor_operation::element_wise::PassThrough>,
    enable_if_t<is_same_v<BLayout, MFMA>>>
{
    using DeviceOp = DeviceGemmMX<ALayout,
                                  BLayout,
                                  CLayout,
                                  ADataType,
                                  AScaleDataType,
                                  BDataType,
                                  BScaleDataType,
                                  CDataType,
                                  ScaleBlockSize,
                                  ck::tensor_operation::element_wise::PassThrough,
                                  ck::tensor_operation::element_wise::PassThrough,
                                  ck::tensor_operation::element_wise::PassThrough>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, MFMA> && is_same_v<CLayout, Row>)
        {
            if constexpr(is_same_v<ADataType, F4> && is_same_v<BDataType, F4> &&
                         is_same_v<CDataType, F16>)
            {
                add_device_gemm_mx_xdl_f4_f4_f16_mk_mfma_mn_default_instances(op_ptrs);
            }
        }

        return op_ptrs;
    }
};
} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
