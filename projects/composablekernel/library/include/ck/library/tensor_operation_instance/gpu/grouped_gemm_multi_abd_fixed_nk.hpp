// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <vector>
#include <memory>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm_multi_abd_fixed_nk.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multi_abd_xdl_fixed_nk.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using Multiply    = ck::tensor_operation::element_wise::Multiply;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using AddFastGelu = ck::tensor_operation::element_wise::AddFastGelu;

// RRR
void add_device_grouped_gemm_xdl_fixed_nk_multi_abd_bf16_i8_bf16_mk_kn_mn_bias_gelu_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmMultiABDFixedNK<ck::Tuple<Row>,
                                                                 ck::Tuple<Row, Row>,
                                                                 ck::Tuple<Row>,
                                                                 Row,
                                                                 ck::Tuple<BF16>,
                                                                 ck::Tuple<I8, BF16>,
                                                                 ck::Tuple<BF16>,
                                                                 BF16,
                                                                 PassThrough,
                                                                 Multiply,
                                                                 AddFastGelu>>>& instances);

void add_device_grouped_gemm_xdl_fixed_nk_multi_abd_bf16_i8_bf16_mk_kn_mn_bias_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmMultiABDFixedNK<ck::Tuple<Row>,
                                                                 ck::Tuple<Row, Row>,
                                                                 ck::Tuple<Row>,
                                                                 Row,
                                                                 ck::Tuple<BF16>,
                                                                 ck::Tuple<I8, BF16>,
                                                                 ck::Tuple<BF16>,
                                                                 BF16,
                                                                 PassThrough,
                                                                 Multiply,
                                                                 Add>>>& instances);

void add_device_grouped_gemm_xdl_fixed_nk_multi_abd_bf16_i8_bf16_mk_kn_mn_gelu_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmMultiABDFixedNK<ck::Tuple<Row>,
                                                                 ck::Tuple<Row, Row>,
                                                                 ck::Tuple<>,
                                                                 Row,
                                                                 ck::Tuple<BF16>,
                                                                 ck::Tuple<I8, BF16>,
                                                                 ck::Tuple<>,
                                                                 BF16,
                                                                 PassThrough,
                                                                 Multiply,
                                                                 FastGelu>>>& instances);

void add_device_grouped_gemm_xdl_fixed_nk_multi_abd_bf16_i8_bf16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmMultiABDFixedNK<ck::Tuple<Row>,
                                                                 ck::Tuple<Row, Row>,
                                                                 ck::Tuple<>,
                                                                 Row,
                                                                 ck::Tuple<BF16>,
                                                                 ck::Tuple<I8, BF16>,
                                                                 ck::Tuple<>,
                                                                 BF16,
                                                                 PassThrough,
                                                                 Multiply,
                                                                 PassThrough>>>& instances);

// RCR
void add_device_grouped_gemm_xdl_fixed_nk_multi_abd_bf16_i8_bf16_mk_nk_mn_bias_gelu_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmMultiABDFixedNK<ck::Tuple<Row>,
                                                                 ck::Tuple<Col, Col>,
                                                                 ck::Tuple<Row>,
                                                                 Row,
                                                                 ck::Tuple<BF16>,
                                                                 ck::Tuple<I8, BF16>,
                                                                 ck::Tuple<BF16>,
                                                                 BF16,
                                                                 PassThrough,
                                                                 Multiply,
                                                                 AddFastGelu>>>& instances);

void add_device_grouped_gemm_xdl_fixed_nk_multi_abd_bf16_i8_bf16_mk_nk_mn_bias_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmMultiABDFixedNK<ck::Tuple<Row>,
                                                                 ck::Tuple<Col, Col>,
                                                                 ck::Tuple<Row>,
                                                                 Row,
                                                                 ck::Tuple<BF16>,
                                                                 ck::Tuple<I8, BF16>,
                                                                 ck::Tuple<BF16>,
                                                                 BF16,
                                                                 PassThrough,
                                                                 Multiply,
                                                                 Add>>>& instances);

void add_device_grouped_gemm_xdl_fixed_nk_multi_abd_bf16_i8_bf16_mk_nk_mn_gelu_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmMultiABDFixedNK<ck::Tuple<Row>,
                                                                 ck::Tuple<Col, Col>,
                                                                 ck::Tuple<>,
                                                                 Row,
                                                                 ck::Tuple<BF16>,
                                                                 ck::Tuple<I8, BF16>,
                                                                 ck::Tuple<>,
                                                                 BF16,
                                                                 PassThrough,
                                                                 Multiply,
                                                                 FastGelu>>>& instances);

void add_device_grouped_gemm_xdl_fixed_nk_multi_abd_bf16_i8_bf16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmMultiABDFixedNK<ck::Tuple<Row>,
                                                                 ck::Tuple<Col, Col>,
                                                                 ck::Tuple<>,
                                                                 Row,
                                                                 ck::Tuple<BF16>,
                                                                 ck::Tuple<I8, BF16>,
                                                                 ck::Tuple<>,
                                                                 BF16,
                                                                 PassThrough,
                                                                 Multiply,
                                                                 PassThrough>>>& instances);

// CRR
void add_device_grouped_gemm_xdl_fixed_nk_multi_abd_bf16_i8_bf16_km_kn_mn_bias_gelu_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmMultiABDFixedNK<ck::Tuple<Col>,
                                                                 ck::Tuple<Row, Row>,
                                                                 ck::Tuple<Row>,
                                                                 Row,
                                                                 ck::Tuple<BF16>,
                                                                 ck::Tuple<I8, BF16>,
                                                                 ck::Tuple<BF16>,
                                                                 BF16,
                                                                 PassThrough,
                                                                 Multiply,
                                                                 AddFastGelu>>>& instances);

void add_device_grouped_gemm_xdl_fixed_nk_multi_abd_bf16_i8_bf16_km_kn_mn_bias_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmMultiABDFixedNK<ck::Tuple<Col>,
                                                                 ck::Tuple<Row, Row>,
                                                                 ck::Tuple<Row>,
                                                                 Row,
                                                                 ck::Tuple<BF16>,
                                                                 ck::Tuple<I8, BF16>,
                                                                 ck::Tuple<BF16>,
                                                                 BF16,
                                                                 PassThrough,
                                                                 Multiply,
                                                                 Add>>>& instances);

void add_device_grouped_gemm_xdl_fixed_nk_multi_abd_bf16_i8_bf16_km_kn_mn_gelu_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmMultiABDFixedNK<ck::Tuple<Col>,
                                                                 ck::Tuple<Row, Row>,
                                                                 ck::Tuple<>,
                                                                 Row,
                                                                 ck::Tuple<BF16>,
                                                                 ck::Tuple<I8, BF16>,
                                                                 ck::Tuple<>,
                                                                 BF16,
                                                                 PassThrough,
                                                                 Multiply,
                                                                 FastGelu>>>& instances);

void add_device_grouped_gemm_xdl_fixed_nk_multi_abd_bf16_i8_bf16_km_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmMultiABDFixedNK<ck::Tuple<Col>,
                                                                 ck::Tuple<Row, Row>,
                                                                 ck::Tuple<>,
                                                                 Row,
                                                                 ck::Tuple<BF16>,
                                                                 ck::Tuple<I8, BF16>,
                                                                 ck::Tuple<>,
                                                                 BF16,
                                                                 PassThrough,
                                                                 Multiply,
                                                                 PassThrough>>>& instances);

// GEMM + Add + Gelu
template <typename AsLayout,
          typename BsLayout,
          typename DsLayout,
          typename ELayout,
          typename AsDataType,
          typename BsDataType,
          typename DsDataType,
          typename EDataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGroupedGemmMultiABDFixedNK<AsLayout,
                                                                   BsLayout,
                                                                   DsLayout,
                                                                   ELayout,
                                                                   AsDataType,
                                                                   BsDataType,
                                                                   DsDataType,
                                                                   EDataType,
                                                                   PassThrough,
                                                                   Multiply,
                                                                   AddFastGelu>>
{
    using DeviceOp = DeviceGroupedGemmMultiABDFixedNK<AsLayout,
                                                      BsLayout,
                                                      DsLayout,
                                                      ELayout,
                                                      AsDataType,
                                                      BsDataType,
                                                      DsDataType,
                                                      EDataType,
                                                      PassThrough,
                                                      Multiply,
                                                      AddFastGelu>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(is_same_v<AsDataType, ck::Tuple<BF16>> &&
                     is_same_v<BsDataType, ck::Tuple<I8, BF16>> &&
                     is_same_v<DsDataType, ck::Tuple<BF16>> && is_same_v<EDataType, BF16>)
        {
            if constexpr(is_same_v<AsLayout, ck::Tuple<Row>> &&
                         is_same_v<BsLayout, ck::Tuple<Row, Row>> &&
                         is_same_v<DsLayout, ck::Tuple<Row>> && is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_fixed_nk_multi_abd_bf16_i8_bf16_mk_kn_mn_bias_gelu_instances(
                    op_ptrs);
            }

            if constexpr(is_same_v<AsLayout, ck::Tuple<Col>> &&
                         is_same_v<BsLayout, ck::Tuple<Row, Row>> &&
                         is_same_v<DsLayout, ck::Tuple<Row>> && is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_fixed_nk_multi_abd_bf16_i8_bf16_km_kn_mn_bias_gelu_instances(
                    op_ptrs);
            }

            if constexpr(is_same_v<AsLayout, ck::Tuple<Row>> &&
                         is_same_v<BsLayout, ck::Tuple<Col, Col>> &&
                         is_same_v<DsLayout, ck::Tuple<Row>> && is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_fixed_nk_multi_abd_bf16_i8_bf16_mk_nk_mn_bias_gelu_instances(
                    op_ptrs);
            }
        }

        return op_ptrs;
    }
};

// GEMM + Add
template <typename AsLayout,
          typename BsLayout,
          typename DsLayout,
          typename ELayout,
          typename AsDataType,
          typename BsDataType,
          typename DsDataType,
          typename EDataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGroupedGemmMultiABDFixedNK<AsLayout,
                                                                   BsLayout,
                                                                   DsLayout,
                                                                   ELayout,
                                                                   AsDataType,
                                                                   BsDataType,
                                                                   DsDataType,
                                                                   EDataType,
                                                                   PassThrough,
                                                                   Multiply,
                                                                   Add>>
{
    using DeviceOp = DeviceGroupedGemmMultiABDFixedNK<AsLayout,
                                                      BsLayout,
                                                      DsLayout,
                                                      ELayout,
                                                      AsDataType,
                                                      BsDataType,
                                                      DsDataType,
                                                      EDataType,
                                                      PassThrough,
                                                      Multiply,
                                                      Add>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(is_same_v<AsDataType, ck::Tuple<BF16>> &&
                     is_same_v<BsDataType, ck::Tuple<I8, BF16>> &&
                     is_same_v<DsDataType, ck::Tuple<BF16>> && is_same_v<EDataType, BF16>)
        {
            if constexpr(is_same_v<AsLayout, ck::Tuple<Row>> &&
                         is_same_v<BsLayout, ck::Tuple<Row, Row>> &&
                         is_same_v<DsLayout, ck::Tuple<Row>> && is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_fixed_nk_multi_abd_bf16_i8_bf16_mk_kn_mn_bias_instances(
                    op_ptrs);
            }

            if constexpr(is_same_v<AsLayout, ck::Tuple<Col>> &&
                         is_same_v<BsLayout, ck::Tuple<Row, Row>> &&
                         is_same_v<DsLayout, ck::Tuple<Row>> && is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_fixed_nk_multi_abd_bf16_i8_bf16_km_kn_mn_bias_instances(
                    op_ptrs);
            }

            if constexpr(is_same_v<AsLayout, ck::Tuple<Row>> &&
                         is_same_v<BsLayout, ck::Tuple<Col, Col>> &&
                         is_same_v<DsLayout, ck::Tuple<Row>> && is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_fixed_nk_multi_abd_bf16_i8_bf16_mk_nk_mn_bias_instances(
                    op_ptrs);
            }
        }

        return op_ptrs;
    }
};

// GEMM + Gelu
template <typename AsLayout,
          typename BsLayout,
          typename DsLayout,
          typename ELayout,
          typename AsDataType,
          typename BsDataType,
          typename DsDataType,
          typename EDataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGroupedGemmMultiABDFixedNK<AsLayout,
                                                                   BsLayout,
                                                                   DsLayout,
                                                                   ELayout,
                                                                   AsDataType,
                                                                   BsDataType,
                                                                   DsDataType,
                                                                   EDataType,
                                                                   PassThrough,
                                                                   Multiply,
                                                                   FastGelu>>
{
    using DeviceOp = DeviceGroupedGemmMultiABDFixedNK<AsLayout,
                                                      BsLayout,
                                                      DsLayout,
                                                      ELayout,
                                                      AsDataType,
                                                      BsDataType,
                                                      DsDataType,
                                                      EDataType,
                                                      PassThrough,
                                                      Multiply,
                                                      FastGelu>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(is_same_v<AsDataType, ck::Tuple<BF16>> &&
                     is_same_v<BsDataType, ck::Tuple<I8, BF16>> &&
                     is_same_v<DsDataType, ck::Tuple<>> && is_same_v<EDataType, BF16>)
        {
            if constexpr(is_same_v<AsLayout, ck::Tuple<Row>> &&
                         is_same_v<BsLayout, ck::Tuple<Row, Row>> &&
                         is_same_v<DsLayout, ck::Tuple<>> && is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_fixed_nk_multi_abd_bf16_i8_bf16_mk_kn_mn_gelu_instances(
                    op_ptrs);
            }

            if constexpr(is_same_v<AsLayout, ck::Tuple<Col>> &&
                         is_same_v<BsLayout, ck::Tuple<Row, Row>> &&
                         is_same_v<DsLayout, ck::Tuple<>> && is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_fixed_nk_multi_abd_bf16_i8_bf16_km_kn_mn_gelu_instances(
                    op_ptrs);
            }

            if constexpr(is_same_v<AsLayout, ck::Tuple<Row>> &&
                         is_same_v<BsLayout, ck::Tuple<Col, Col>> &&
                         is_same_v<DsLayout, ck::Tuple<>> && is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_fixed_nk_multi_abd_bf16_i8_bf16_mk_nk_mn_gelu_instances(
                    op_ptrs);
            }
        }

        return op_ptrs;
    }
};

// GEMM
template <typename AsLayout,
          typename BsLayout,
          typename DsLayout,
          typename ELayout,
          typename AsDataType,
          typename BsDataType,
          typename DsDataType,
          typename EDataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGroupedGemmMultiABDFixedNK<AsLayout,
                                                                   BsLayout,
                                                                   DsLayout,
                                                                   ELayout,
                                                                   AsDataType,
                                                                   BsDataType,
                                                                   DsDataType,
                                                                   EDataType,
                                                                   PassThrough,
                                                                   Multiply,
                                                                   PassThrough>>
{
    using DeviceOp = DeviceGroupedGemmMultiABDFixedNK<AsLayout,
                                                      BsLayout,
                                                      DsLayout,
                                                      ELayout,
                                                      AsDataType,
                                                      BsDataType,
                                                      DsDataType,
                                                      EDataType,
                                                      PassThrough,
                                                      Multiply,
                                                      PassThrough>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(is_same_v<AsDataType, ck::Tuple<BF16>> &&
                     is_same_v<BsDataType, ck::Tuple<I8, BF16>> &&
                     is_same_v<DsDataType, ck::Tuple<>> && is_same_v<EDataType, BF16>)
        {
            if constexpr(is_same_v<AsLayout, ck::Tuple<Row>> &&
                         is_same_v<BsLayout, ck::Tuple<Row, Row>> &&
                         is_same_v<DsLayout, ck::Tuple<>> && is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_fixed_nk_multi_abd_bf16_i8_bf16_mk_kn_mn_instances(
                    op_ptrs);
            }

            if constexpr(is_same_v<AsLayout, ck::Tuple<Col>> &&
                         is_same_v<BsLayout, ck::Tuple<Row, Row>> &&
                         is_same_v<DsLayout, ck::Tuple<>> && is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_fixed_nk_multi_abd_bf16_i8_bf16_km_kn_mn_instances(
                    op_ptrs);
            }

            if constexpr(is_same_v<AsLayout, ck::Tuple<Row>> &&
                         is_same_v<BsLayout, ck::Tuple<Col, Col>> &&
                         is_same_v<DsLayout, ck::Tuple<>> && is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_fixed_nk_multi_abd_bf16_i8_bf16_mk_nk_mn_instances(
                    op_ptrs);
            }
        }

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
