// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <vector>
#include <memory>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_multi_d.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

#ifdef DL_KERNELS
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
#ifdef CK_ENABLE_FP16
void add_device_batched_gemm_multi_d_dl_f16_f16_f16_gkm_gkn_gmn_instances(
    std::vector<std::unique_ptr<DeviceBatchedGemmMultiD<Col,
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

void add_device_batched_gemm_multi_d_dl_f16_f16_f16_gkm_gnk_gmn_instances(
    std::vector<std::unique_ptr<DeviceBatchedGemmMultiD<Col,
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

void add_device_batched_gemm_multi_d_dl_f16_f16_f16_gmk_gkn_gmn_instances(
    std::vector<std::unique_ptr<DeviceBatchedGemmMultiD<Row,
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

void add_device_batched_gemm_multi_d_dl_f16_f16_f16_gmk_gnk_gmn_instances(
    std::vector<std::unique_ptr<DeviceBatchedGemmMultiD<Row,
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

void add_device_batched_gemm_multi_d_dl_f16_f16_f16_gkm_gkn_gmn_irregular_instances(
    std::vector<std::unique_ptr<DeviceBatchedGemmMultiD<Col,
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

void add_device_batched_gemm_multi_d_dl_f16_f16_f16_gkm_gnk_gmn_irregular_instances(
    std::vector<std::unique_ptr<DeviceBatchedGemmMultiD<Col,
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

void add_device_batched_gemm_multi_d_dl_f16_f16_f16_gmk_gkn_gmn_irregular_instances(
    std::vector<std::unique_ptr<DeviceBatchedGemmMultiD<Row,
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

void add_device_batched_gemm_multi_d_dl_f16_f16_f16_gmk_gnk_gmn_irregular_instances(
    std::vector<std::unique_ptr<DeviceBatchedGemmMultiD<Row,
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
#ifdef CK_ENABLE_INT8
void add_device_batched_gemm_multi_d_dl_i8_i8_i8_gkm_gkn_gmn_instances(
    std::vector<std::unique_ptr<DeviceBatchedGemmMultiD<Col,
                                                        Row,
                                                        Empty_Tuple,
                                                        Row,
                                                        int8_t,
                                                        int8_t,
                                                        Empty_Tuple,
                                                        int8_t,
                                                        PassThrough,
                                                        PassThrough,
                                                        PassThrough>>>& instances);

void add_device_batched_gemm_multi_d_dl_i8_i8_i8_gkm_gnk_gmn_instances(
    std::vector<std::unique_ptr<DeviceBatchedGemmMultiD<Col,
                                                        Col,
                                                        Empty_Tuple,
                                                        Row,
                                                        int8_t,
                                                        int8_t,
                                                        Empty_Tuple,
                                                        int8_t,
                                                        PassThrough,
                                                        PassThrough,
                                                        PassThrough>>>& instances);

void add_device_batched_gemm_multi_d_dl_i8_i8_i8_gmk_gkn_gmn_instances(
    std::vector<std::unique_ptr<DeviceBatchedGemmMultiD<Row,
                                                        Row,
                                                        Empty_Tuple,
                                                        Row,
                                                        int8_t,
                                                        int8_t,
                                                        Empty_Tuple,
                                                        int8_t,
                                                        PassThrough,
                                                        PassThrough,
                                                        PassThrough>>>& instances);

void add_device_batched_gemm_multi_d_dl_i8_i8_i8_gmk_gnk_gmn_instances(
    std::vector<std::unique_ptr<DeviceBatchedGemmMultiD<Row,
                                                        Col,
                                                        Empty_Tuple,
                                                        Row,
                                                        int8_t,
                                                        int8_t,
                                                        Empty_Tuple,
                                                        int8_t,
                                                        PassThrough,
                                                        PassThrough,
                                                        PassThrough>>>& instances);

void add_device_batched_gemm_multi_d_dl_i8_i8_i8_gkm_gkn_gmn_irregular_instances(
    std::vector<std::unique_ptr<DeviceBatchedGemmMultiD<Col,
                                                        Row,
                                                        Empty_Tuple,
                                                        Row,
                                                        int8_t,
                                                        int8_t,
                                                        Empty_Tuple,
                                                        int8_t,
                                                        PassThrough,
                                                        PassThrough,
                                                        PassThrough>>>& instances);

void add_device_batched_gemm_multi_d_dl_i8_i8_i8_gkm_gnk_gmn_irregular_instances(
    std::vector<std::unique_ptr<DeviceBatchedGemmMultiD<Col,
                                                        Col,
                                                        Empty_Tuple,
                                                        Row,
                                                        int8_t,
                                                        int8_t,
                                                        Empty_Tuple,
                                                        int8_t,
                                                        PassThrough,
                                                        PassThrough,
                                                        PassThrough>>>& instances);

void add_device_batched_gemm_multi_d_dl_i8_i8_i8_gmk_gkn_gmn_irregular_instances(
    std::vector<std::unique_ptr<DeviceBatchedGemmMultiD<Row,
                                                        Row,
                                                        Empty_Tuple,
                                                        Row,
                                                        int8_t,
                                                        int8_t,
                                                        Empty_Tuple,
                                                        int8_t,
                                                        PassThrough,
                                                        PassThrough,
                                                        PassThrough>>>& instances);

void add_device_batched_gemm_multi_d_dl_i8_i8_i8_gmk_gnk_gmn_irregular_instances(
    std::vector<std::unique_ptr<DeviceBatchedGemmMultiD<Row,
                                                        Col,
                                                        Empty_Tuple,
                                                        Row,
                                                        int8_t,
                                                        int8_t,
                                                        Empty_Tuple,
                                                        int8_t,
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
struct DeviceOperationInstanceFactory<ck::tensor_operation::device::DeviceBatchedGemmMultiD<
    ALayout,
    BLayout,
    Empty_Tuple,
    ELayout,
    ADataType,
    BDataType,
    Empty_Tuple,
    EDataType,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough>>
{
    using DeviceOp = DeviceBatchedGemmMultiD<ALayout,
                                             BLayout,
                                             Empty_Tuple,
                                             ELayout,
                                             ADataType,
                                             BDataType,
                                             Empty_Tuple,
                                             EDataType,
                                             ck::tensor_operation::element_wise::PassThrough,
                                             ck::tensor_operation::element_wise::PassThrough,
                                             ck::tensor_operation::element_wise::PassThrough>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;
#ifdef CK_ENABLE_FP16
        if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, half_t> &&
                     is_same_v<EDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_batched_gemm_multi_d_dl_f16_f16_f16_gmk_gkn_gmn_instances(op_ptrs);
                add_device_batched_gemm_multi_d_dl_f16_f16_f16_gmk_gkn_gmn_irregular_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<ELayout, Row>)
            {
                add_device_batched_gemm_multi_d_dl_f16_f16_f16_gmk_gnk_gmn_instances(op_ptrs);
                add_device_batched_gemm_multi_d_dl_f16_f16_f16_gmk_gnk_gmn_irregular_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<ELayout, Row>)
            {
                add_device_batched_gemm_multi_d_dl_f16_f16_f16_gkm_gkn_gmn_instances(op_ptrs);
                add_device_batched_gemm_multi_d_dl_f16_f16_f16_gkm_gkn_gmn_irregular_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<ELayout, Row>)
            {
                add_device_batched_gemm_multi_d_dl_f16_f16_f16_gkm_gnk_gmn_instances(op_ptrs);
                add_device_batched_gemm_multi_d_dl_f16_f16_f16_gkm_gnk_gmn_irregular_instances(
                    op_ptrs);
            }
        }
#endif
#ifdef CK_ENABLE_INT8
        else if constexpr(is_same_v<ADataType, int8_t> && is_same_v<BDataType, int8_t> &&
                          is_same_v<EDataType, int8_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_batched_gemm_multi_d_dl_i8_i8_i8_gmk_gkn_gmn_instances(op_ptrs);
                add_device_batched_gemm_multi_d_dl_i8_i8_i8_gmk_gkn_gmn_irregular_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<ELayout, Row>)
            {
                add_device_batched_gemm_multi_d_dl_i8_i8_i8_gmk_gnk_gmn_instances(op_ptrs);
                add_device_batched_gemm_multi_d_dl_i8_i8_i8_gmk_gnk_gmn_irregular_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<ELayout, Row>)
            {
                add_device_batched_gemm_multi_d_dl_i8_i8_i8_gkm_gkn_gmn_instances(op_ptrs);
                add_device_batched_gemm_multi_d_dl_i8_i8_i8_gkm_gkn_gmn_irregular_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<ELayout, Row>)
            {
                add_device_batched_gemm_multi_d_dl_i8_i8_i8_gkm_gnk_gmn_instances(op_ptrs);
                add_device_batched_gemm_multi_d_dl_i8_i8_i8_gkm_gnk_gmn_irregular_instances(
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
#endif
