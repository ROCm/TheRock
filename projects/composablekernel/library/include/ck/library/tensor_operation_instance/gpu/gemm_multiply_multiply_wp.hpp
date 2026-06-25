// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <memory>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3_b_preshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

#if(defined(CK_ENABLE_F16) || defined(CK_ENABLE_FP8))
using TGemmMulMulF8F8F16Instances =
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitKBPreShuffle<Row,
                                                                     Col,
                                                                     Tuple<Row, Col>,
                                                                     Row,
                                                                     F8,
                                                                     F8,
                                                                     Tuple<F32, F32>,
                                                                     F16,
                                                                     PassThrough,
                                                                     PassThrough,
                                                                     MultiplyMultiply>>>;

#ifdef CK_USE_XDL
void add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_compute_default_instances_p1(
    TGemmMulMulF8F8F16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_compute_default_instances_p2(
    TGemmMulMulF8F8F16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p1_default_instances(
    TGemmMulMulF8F8F16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p2_default_instances(
    TGemmMulMulF8F8F16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p3_default_instances(
    TGemmMulMulF8F8F16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p4_default_instances(
    TGemmMulMulF8F8F16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p5_default_instances(
    TGemmMulMulF8F8F16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p1_default_instances_v2(
    TGemmMulMulF8F8F16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p2_default_instances_v2(
    TGemmMulMulF8F8F16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p3_default_instances_v2(
    TGemmMulMulF8F8F16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p4_default_instances_v2(
    TGemmMulMulF8F8F16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p5_default_instances_v2(
    TGemmMulMulF8F8F16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma16x16_mn_compute_default_instances_p1(
    TGemmMulMulF8F8F16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma16x16_mn_compute_default_instances_p2(
    TGemmMulMulF8F8F16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma16x16_mn_compute_default_instances_p3(
    TGemmMulMulF8F8F16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma16x16_mn_compute_default_instances_p4(
    TGemmMulMulF8F8F16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma16x16_mn_compute_default_instances_p5(
    TGemmMulMulF8F8F16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma16x16_mn_compute_default_instances_p6(
    TGemmMulMulF8F8F16Instances& instances);
#endif

#ifdef CK_USE_WMMA
void add_device_gemm_multiply_multiply_weight_preshuffle_wmma_f8_f8_f16_mk_wmma_mn_default_instances_p1(
    TGemmMulMulF8F8F16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_wmma_f8_f8_f16_mk_wmma_mn_default_instances_p2(
    TGemmMulMulF8F8F16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_wmma_f8_f8_f16_mk_wmma_mn_default_instances_p3(
    TGemmMulMulF8F8F16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_wmma_f8_f8_f16_mk_wmma_mn_default_instances_p4(
    TGemmMulMulF8F8F16Instances& instances);
#endif
#endif

#if(defined(CK_ENABLE_BF16) || defined(CK_ENABLE_FP8))
using TGemmMulMulF8F8BF16Instances =
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitKBPreShuffle<Row,
                                                                     Col,
                                                                     Tuple<Row, Col>,
                                                                     Row,
                                                                     F8,
                                                                     F8,
                                                                     Tuple<F32, F32>,
                                                                     BF16,
                                                                     PassThrough,
                                                                     PassThrough,
                                                                     MultiplyMultiply>>>;

#ifdef CK_USE_XDL
void add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma_mn_compute_default_instances_p1(
    TGemmMulMulF8F8BF16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma_mn_compute_default_instances_p2(
    TGemmMulMulF8F8BF16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma_mn_p1_default_instances(
    TGemmMulMulF8F8BF16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma_mn_p2_default_instances(
    TGemmMulMulF8F8BF16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma_mn_p3_default_instances(
    TGemmMulMulF8F8BF16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma_mn_p4_default_instances(
    TGemmMulMulF8F8BF16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma_mn_p5_default_instances(
    TGemmMulMulF8F8BF16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma_mn_p1_default_instances_v2(
    TGemmMulMulF8F8BF16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma_mn_p2_default_instances_v2(
    TGemmMulMulF8F8BF16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma_mn_p3_default_instances_v2(
    TGemmMulMulF8F8BF16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma_mn_p4_default_instances_v2(
    TGemmMulMulF8F8BF16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma_mn_p5_default_instances_v2(
    TGemmMulMulF8F8BF16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma16x16_mn_compute_default_instances_p1(
    TGemmMulMulF8F8BF16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma16x16_mn_compute_default_instances_p2(
    TGemmMulMulF8F8BF16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma16x16_mn_compute_default_instances_p3(
    TGemmMulMulF8F8BF16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma16x16_mn_compute_default_instances_p4(
    TGemmMulMulF8F8BF16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma16x16_mn_compute_default_instances_p5(
    TGemmMulMulF8F8BF16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma16x16_mn_compute_default_instances_p6(
    TGemmMulMulF8F8BF16Instances& instances);
#endif

#ifdef CK_USE_WMMA
void add_device_gemm_multiply_multiply_weight_preshuffle_wmma_f8_f8_bf16_mk_wmma_mn_default_instances_p1(
    TGemmMulMulF8F8BF16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_wmma_f8_f8_bf16_mk_wmma_mn_default_instances_p2(
    TGemmMulMulF8F8BF16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_wmma_f8_f8_bf16_mk_wmma_mn_default_instances_p3(
    TGemmMulMulF8F8BF16Instances& instances);

void add_device_gemm_multiply_multiply_weight_preshuffle_wmma_f8_f8_bf16_mk_wmma_mn_default_instances_p4(
    TGemmMulMulF8F8BF16Instances& instances);
#endif
#endif

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGemmMultipleDSplitKBPreShuffle<
        ALayout,
        BLayout,
        Tuple<Row, Col>,
        CLayout,
        ADataType,
        BDataType,
        Tuple<F32, F32>,
        CDataType,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::MultiplyMultiply>>
{
    using DeviceOp =
        DeviceGemmMultipleDSplitKBPreShuffle<ALayout,
                                             BLayout,
                                             Tuple<Row, Col>,
                                             CLayout,
                                             ADataType,
                                             BDataType,
                                             Tuple<F32, F32>,
                                             CDataType,
                                             ck::tensor_operation::element_wise::PassThrough,
                                             ck::tensor_operation::element_wise::PassThrough,
                                             ck::tensor_operation::element_wise::MultiplyMultiply>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;
// TODO: Add MFMA layout into tensor layout
#if(defined(CK_ENABLE_F16) || defined(CK_ENABLE_FP8))
        if constexpr(is_same_v<ADataType, f8_t> && is_same_v<BDataType, f8_t> &&
                     is_same_v<CDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                         is_same_v<CLayout, Row>)
            {
#ifdef CK_USE_XDL
                add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma16x16_mn_compute_default_instances_p1(
                    op_ptrs);
                add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma16x16_mn_compute_default_instances_p2(
                    op_ptrs);
                add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma16x16_mn_compute_default_instances_p3(
                    op_ptrs);
                add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma16x16_mn_compute_default_instances_p4(
                    op_ptrs);
                add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma16x16_mn_compute_default_instances_p5(
                    op_ptrs);
                add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma16x16_mn_compute_default_instances_p6(
                    op_ptrs);

                add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_compute_default_instances_p1(
                    op_ptrs);
                add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_compute_default_instances_p2(
                    op_ptrs);
                add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p1_default_instances(
                    op_ptrs);
                add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p2_default_instances(
                    op_ptrs);
                add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p3_default_instances(
                    op_ptrs);
                add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p4_default_instances(
                    op_ptrs);
                add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p5_default_instances(
                    op_ptrs);
                add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p1_default_instances_v2(
                    op_ptrs);
                add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p2_default_instances_v2(
                    op_ptrs);
                add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p3_default_instances_v2(
                    op_ptrs);
                add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p4_default_instances_v2(
                    op_ptrs);
                add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p5_default_instances_v2(
                    op_ptrs);
#endif
#ifdef CK_USE_WMMA
                add_device_gemm_multiply_multiply_weight_preshuffle_wmma_f8_f8_f16_mk_wmma_mn_default_instances_p1(
                    op_ptrs);
                add_device_gemm_multiply_multiply_weight_preshuffle_wmma_f8_f8_f16_mk_wmma_mn_default_instances_p2(
                    op_ptrs);
                add_device_gemm_multiply_multiply_weight_preshuffle_wmma_f8_f8_f16_mk_wmma_mn_default_instances_p3(
                    op_ptrs);
                add_device_gemm_multiply_multiply_weight_preshuffle_wmma_f8_f8_f16_mk_wmma_mn_default_instances_p4(
                    op_ptrs);
#endif
            }
        }
#endif

#if(defined(CK_ENABLE_BF16) || defined(CK_ENABLE_FP8))
        if constexpr(is_same_v<ADataType, f8_t> && is_same_v<BDataType, f8_t> &&
                     is_same_v<CDataType, bhalf_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                         is_same_v<CLayout, Row>)
            {
#ifdef CK_USE_XDL
                add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma16x16_mn_compute_default_instances_p1(
                    op_ptrs);
                add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma16x16_mn_compute_default_instances_p2(
                    op_ptrs);
                add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma16x16_mn_compute_default_instances_p3(
                    op_ptrs);
                add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma16x16_mn_compute_default_instances_p4(
                    op_ptrs);
                add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma16x16_mn_compute_default_instances_p5(
                    op_ptrs);
                add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma16x16_mn_compute_default_instances_p6(
                    op_ptrs);

                add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma_mn_compute_default_instances_p1(
                    op_ptrs);
                add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma_mn_compute_default_instances_p2(
                    op_ptrs);
                add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma_mn_p1_default_instances(
                    op_ptrs);
                add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma_mn_p2_default_instances(
                    op_ptrs);
                add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma_mn_p3_default_instances(
                    op_ptrs);
                add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma_mn_p4_default_instances(
                    op_ptrs);
                add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma_mn_p5_default_instances(
                    op_ptrs);
                add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma_mn_p1_default_instances_v2(
                    op_ptrs);
                add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma_mn_p2_default_instances_v2(
                    op_ptrs);
                add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma_mn_p3_default_instances_v2(
                    op_ptrs);
                add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma_mn_p4_default_instances_v2(
                    op_ptrs);
                add_device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma_mn_p5_default_instances_v2(
                    op_ptrs);
#endif
#ifdef CK_USE_WMMA
                add_device_gemm_multiply_multiply_weight_preshuffle_wmma_f8_f8_bf16_mk_wmma_mn_default_instances_p1(
                    op_ptrs);
                add_device_gemm_multiply_multiply_weight_preshuffle_wmma_f8_f8_bf16_mk_wmma_mn_default_instances_p2(
                    op_ptrs);
                add_device_gemm_multiply_multiply_weight_preshuffle_wmma_f8_f8_bf16_mk_wmma_mn_default_instances_p3(
                    op_ptrs);
                add_device_gemm_multiply_multiply_weight_preshuffle_wmma_f8_f8_bf16_mk_wmma_mn_default_instances_p4(
                    op_ptrs);
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
