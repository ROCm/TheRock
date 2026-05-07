// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <memory>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
#ifdef CK_USE_XDL
#ifdef CK_ENABLE_FP8
#ifdef CK_ENABLE_BF16
void add_device_gemm_multiply_multiply_xdl_f8_f8_bf16_mk_nk_mn_comp_default_instances_part1(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          Tuple<F32, F32>,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

void add_device_gemm_multiply_multiply_xdl_f8_f8_bf16_mk_nk_mn_comp_kpadding_instances_part1(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          Tuple<F32, F32>,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

void add_device_gemm_multiply_multiply_xdl_f8_f8_bf16_mk_nk_mn_comp_default_instances_part2(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          Tuple<F32, F32>,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

void add_device_gemm_multiply_multiply_xdl_f8_f8_bf16_mk_nk_mn_comp_kpadding_instances_part2(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          Tuple<F32, F32>,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

void add_device_gemm_multiply_multiply_xdl_f8_f8_bf16_mk_nk_mn_comp_mfma16x16_default_instances_part1(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          Tuple<F32, F32>,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

void add_device_gemm_multiply_multiply_xdl_f8_f8_bf16_mk_nk_mn_comp_mfma16x16_kpadding_instances_part1(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          Tuple<F32, F32>,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

void add_device_gemm_multiply_multiply_xdl_f8_f8_bf16_mk_nk_mn_comp_mfma16x16_default_instances_part2(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          Tuple<F32, F32>,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

void add_device_gemm_multiply_multiply_xdl_f8_f8_bf16_mk_nk_mn_comp_mfma16x16_kpadding_instances_part2(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          Tuple<F32, F32>,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

void add_device_gemm_multiply_multiply_xdl_f8_f8_bf16_mk_nk_mn_comp_mfma16x16_default_instances_part3(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          Tuple<F32, F32>,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

void add_device_gemm_multiply_multiply_xdl_f8_f8_bf16_mk_nk_mn_comp_mfma16x16_kpadding_instances_part3(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          Tuple<F32, F32>,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

void add_device_gemm_multiply_multiply_xdl_f8_f8_bf16_mk_nk_mn_mem_v1_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          Tuple<F32, F32>,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

void add_device_gemm_multiply_multiply_xdl_f8_f8_bf16_mk_nk_mn_mem_v1_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          Tuple<F32, F32>,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

void add_device_gemm_multiply_multiply_xdl_f8_f8_bf16_mk_nk_mn_mem_v2_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          Tuple<F32, F32>,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

void add_device_gemm_multiply_multiply_xdl_f8_f8_bf16_mk_nk_mn_mem_v2_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          Tuple<F32, F32>,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);
#endif // CK_ENABLE_BF16
#ifdef CK_ENABLE_FP16
void add_device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_comp_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          Tuple<F32, F32>,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

void add_device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_comp_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          Tuple<F32, F32>,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

void add_device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_mem_v1_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          Tuple<F32, F32>,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

void add_device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_mem_v1_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          Tuple<F32, F32>,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

void add_device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_mem_v2_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          Tuple<F32, F32>,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

void add_device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_mem_v2_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          Tuple<F32, F32>,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);
#endif // CK_ENABLE_FP16
#endif // CK_ENABLE_FP8

#ifdef CK_ENABLE_FP16
void add_device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_comp_default_instances_part1(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          Tuple<F32, F32>,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

void add_device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_comp_kpadding_instances_part1(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          Tuple<F32, F32>,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

void add_device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_comp_default_instances_part2(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          Tuple<F32, F32>,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

void add_device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_comp_kpadding_instances_part2(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          Tuple<F32, F32>,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

void add_device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_comp_mfma16x16_default_instances_part1(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          Tuple<F32, F32>,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

void add_device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_comp_mfma16x16_kpadding_instances_part1(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          Tuple<F32, F32>,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

void add_device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_comp_mfma16x16_default_instances_part2(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          Tuple<F32, F32>,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

void add_device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_comp_mfma16x16_kpadding_instances_part2(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          Tuple<F32, F32>,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

void add_device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_comp_mfma16x16_default_instances_part3(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          Tuple<F32, F32>,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

void add_device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_comp_mfma16x16_kpadding_instances_part3(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          Tuple<F32, F32>,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

void add_device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_mem_v1_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          Tuple<F32, F32>,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

void add_device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_mem_v1_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          Tuple<F32, F32>,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

void add_device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_mem_v2_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          Tuple<F32, F32>,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

void add_device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_mem_v2_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          Tuple<F32, F32>,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);
#endif // CK_ENABLE_FP16

#if(defined(CK_ENABLE_FP16) || defined(CK_ENABLE_INT8))
void add_device_gemm_multiply_multiply_xdl_i8_i8_f16_mk_nk_mn_comp_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          I8,
                                                          I8,
                                                          Tuple<F16, F16>,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

void add_device_gemm_multiply_multiply_xdl_i8_i8_f16_mk_nk_mn_comp_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          I8,
                                                          I8,
                                                          Tuple<F16, F16>,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

void add_device_gemm_multiply_multiply_xdl_i8_i8_f16_mk_nk_mn_mem_v1_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          I8,
                                                          I8,
                                                          Tuple<F16, F16>,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

void add_device_gemm_multiply_multiply_xdl_i8_i8_f16_mk_nk_mn_mem_v1_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          I8,
                                                          I8,
                                                          Tuple<F16, F16>,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

void add_device_gemm_multiply_multiply_xdl_i8_i8_f16_mk_nk_mn_mem_v2_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          I8,
                                                          I8,
                                                          Tuple<F16, F16>,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

void add_device_gemm_multiply_multiply_xdl_i8_i8_f16_mk_nk_mn_mem_v2_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col>,
                                                          Row,
                                                          I8,
                                                          I8,
                                                          Tuple<F16, F16>,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

#endif // CK_ENABLE_FP16 || CK_ENABLE_INT8
#endif // CK_USE_XDL

#ifdef CK_USE_WMMA
void add_device_gemm_multiply_multiply_wmma_c_shuffle_i8_i8_f16_km_nk_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Row_Col_Tuple,
                                                          Row,
                                                          I8,
                                                          I8,
                                                          F16_F16_Tuple,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

void add_device_gemm_multiply_multiply_wmma_c_shuffle_i8_i8_bf16_km_nk_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Row_Col_Tuple,
                                                          Row,
                                                          I8,
                                                          I8,
                                                          F32_F32_Tuple,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

void add_device_gemm_multiply_multiply_wmma_c_shuffle_f8_f8_f16_km_nk_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Row_Col_Tuple,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          F32_F32_Tuple,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);

void add_device_gemm_multiply_multiply_wmma_c_shuffle_f8_f8_bf16_km_nk_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Row_Col_Tuple,
                                                          Row,
                                                          F8,
                                                          F8,
                                                          F32_F32_Tuple,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances);
#endif // CK_USE_WMMA

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename DsDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
struct DeviceOperationInstanceFactory<DeviceGemmMultipleDSplitK<ALayout,
                                                                BLayout,
                                                                Tuple<Row, Col>,
                                                                CLayout,
                                                                ADataType,
                                                                BDataType,
                                                                DsDataType,
                                                                CDataType,
                                                                PassThrough,
                                                                PassThrough,
                                                                MultiplyMultiply>>
{
    using DeviceOp = DeviceGemmMultipleDSplitK<ALayout,
                                               BLayout,
                                               Tuple<Row, Col>,
                                               CLayout,
                                               ADataType,
                                               BDataType,
                                               DsDataType,
                                               CDataType,
                                               PassThrough,
                                               PassThrough,
                                               MultiplyMultiply>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

#ifdef CK_USE_XDL
#if defined(CK_USE_FP8_ON_UNSUPPORTED_ARCH) || CK_USE_OCP_FP8 || defined(CK_USE_GFX94)
#ifdef CK_ENABLE_BF16
        if constexpr(is_same_v<ADataType, f8_t> && is_same_v<BDataType, f8_t> &&
                     is_same_v<CDataType, bhalf_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_multiply_multiply_xdl_f8_f8_bf16_mk_nk_mn_comp_default_instances_part1(
                    op_ptrs);
                add_device_gemm_multiply_multiply_xdl_f8_f8_bf16_mk_nk_mn_comp_kpadding_instances_part1(
                    op_ptrs);
                add_device_gemm_multiply_multiply_xdl_f8_f8_bf16_mk_nk_mn_comp_default_instances_part2(
                    op_ptrs);
                add_device_gemm_multiply_multiply_xdl_f8_f8_bf16_mk_nk_mn_comp_kpadding_instances_part2(
                    op_ptrs);

                add_device_gemm_multiply_multiply_xdl_f8_f8_bf16_mk_nk_mn_comp_mfma16x16_default_instances_part1(
                    op_ptrs);
                add_device_gemm_multiply_multiply_xdl_f8_f8_bf16_mk_nk_mn_comp_mfma16x16_kpadding_instances_part1(
                    op_ptrs);
                add_device_gemm_multiply_multiply_xdl_f8_f8_bf16_mk_nk_mn_comp_mfma16x16_default_instances_part2(
                    op_ptrs);
                add_device_gemm_multiply_multiply_xdl_f8_f8_bf16_mk_nk_mn_comp_mfma16x16_kpadding_instances_part2(
                    op_ptrs);
                add_device_gemm_multiply_multiply_xdl_f8_f8_bf16_mk_nk_mn_comp_mfma16x16_default_instances_part3(
                    op_ptrs);
                add_device_gemm_multiply_multiply_xdl_f8_f8_bf16_mk_nk_mn_comp_mfma16x16_kpadding_instances_part3(
                    op_ptrs);

                add_device_gemm_multiply_multiply_xdl_f8_f8_bf16_mk_nk_mn_mem_v1_default_instances(
                    op_ptrs);
                add_device_gemm_multiply_multiply_xdl_f8_f8_bf16_mk_nk_mn_mem_v1_kpadding_instances(
                    op_ptrs);

                add_device_gemm_multiply_multiply_xdl_f8_f8_bf16_mk_nk_mn_mem_v2_default_instances(
                    op_ptrs);
                add_device_gemm_multiply_multiply_xdl_f8_f8_bf16_mk_nk_mn_mem_v2_kpadding_instances(
                    op_ptrs);
            }
        }
#endif // CK_ENABLE_BF16
#ifdef CK_ENABLE_FP16
        if constexpr(is_same_v<ADataType, f8_t> && is_same_v<BDataType, f8_t> &&
                     is_same_v<CDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_comp_default_instances_part1(
                    op_ptrs);
                add_device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_comp_kpadding_instances_part1(
                    op_ptrs);
                add_device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_comp_default_instances_part2(
                    op_ptrs);
                add_device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_comp_kpadding_instances_part2(
                    op_ptrs);

                add_device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_comp_mfma16x16_default_instances_part1(
                    op_ptrs);
                add_device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_comp_mfma16x16_kpadding_instances_part1(
                    op_ptrs);
                add_device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_comp_mfma16x16_default_instances_part2(
                    op_ptrs);
                add_device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_comp_mfma16x16_kpadding_instances_part2(
                    op_ptrs);
                add_device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_comp_mfma16x16_default_instances_part3(
                    op_ptrs);
                add_device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_comp_mfma16x16_kpadding_instances_part3(
                    op_ptrs);

                add_device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_mem_v1_default_instances(
                    op_ptrs);
                add_device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_mem_v1_kpadding_instances(
                    op_ptrs);

                add_device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_mem_v2_default_instances(
                    op_ptrs);
                add_device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_mem_v2_kpadding_instances(
                    op_ptrs);
            }
        }
#endif // CK_ENABLE_FP16
#endif // CK_ENABLE_FP8
#if(defined(CK_ENABLE_FP16) || defined(CK_ENABLE_INT8))
        if constexpr(is_same_v<ADataType, int8_t> && is_same_v<BDataType, int8_t> &&
                     is_same_v<CDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_multiply_multiply_xdl_i8_i8_f16_mk_nk_mn_comp_default_instances(
                    op_ptrs);
                add_device_gemm_multiply_multiply_xdl_i8_i8_f16_mk_nk_mn_comp_kpadding_instances(
                    op_ptrs);

                add_device_gemm_multiply_multiply_xdl_i8_i8_f16_mk_nk_mn_mem_v1_default_instances(
                    op_ptrs);
                add_device_gemm_multiply_multiply_xdl_i8_i8_f16_mk_nk_mn_mem_v1_kpadding_instances(
                    op_ptrs);

                add_device_gemm_multiply_multiply_xdl_i8_i8_f16_mk_nk_mn_mem_v2_default_instances(
                    op_ptrs);
                add_device_gemm_multiply_multiply_xdl_i8_i8_f16_mk_nk_mn_mem_v2_kpadding_instances(
                    op_ptrs);
            }
        }
#endif
#endif // CK_USE_XDL

#ifdef CK_USE_WMMA
        if constexpr(is_same_v<ADataType, int8_t> && is_same_v<BDataType, int8_t> &&
                     is_same_v<CDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_multiply_multiply_wmma_c_shuffle_i8_i8_f16_km_nk_mn_instances(
                    op_ptrs);
            }
        }
        if constexpr(is_same_v<ADataType, int8_t> && is_same_v<BDataType, int8_t> &&
                     is_same_v<CDataType, bhalf_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_multiply_multiply_wmma_c_shuffle_i8_i8_bf16_km_nk_mn_instances(
                    op_ptrs);
            }
        }
        if constexpr(is_same_v<ADataType, f8_t> && is_same_v<BDataType, f8_t> &&
                     is_same_v<CDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_multiply_multiply_wmma_c_shuffle_f8_f8_f16_km_nk_mn_instances(
                    op_ptrs);
            }
        }
        if constexpr(is_same_v<ADataType, f8_t> && is_same_v<BDataType, f8_t> &&
                     is_same_v<CDataType, bhalf_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_multiply_multiply_wmma_c_shuffle_f8_f8_bf16_km_nk_mn_instances(
                    op_ptrs);
            }
        }
#endif // CK_USE_WMMA

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
