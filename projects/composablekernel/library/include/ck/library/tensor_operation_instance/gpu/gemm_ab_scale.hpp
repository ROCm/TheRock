// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <memory>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3_ab_scale.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

#if(defined(CK_ENABLE_BF16) || defined(CK_ENABLE_FP8))
#ifdef CK_USE_WMMA_FP8
// Row, Col
void add_device_gemm_ab_scale_wmma_f8_f8_bf16_mk_nk_mn_1_128_128_comp_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_ABScaleSplitK<Row,
                                                                  Col,
                                                                  Tuple<>,
                                                                  Row,
                                                                  F8,
                                                                  F32,
                                                                  F8,
                                                                  F32,
                                                                  Tuple<>,
                                                                  BF16,
                                                                  1,
                                                                  128,
                                                                  128,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_gemm_ab_scale_wmma_f8_f8_bf16_mk_nk_mn_1_128_128_comp_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_ABScaleSplitK<Row,
                                                                  Col,
                                                                  Tuple<>,
                                                                  Row,
                                                                  F8,
                                                                  F32,
                                                                  F8,
                                                                  F32,
                                                                  Tuple<>,
                                                                  BF16,
                                                                  1,
                                                                  128,
                                                                  128,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_gemm_ab_scale_wmma_f8_f8_bf16_mk_nk_mn_1_128_128_mem_v1_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_ABScaleSplitK<Row,
                                                                  Col,
                                                                  Tuple<>,
                                                                  Row,
                                                                  F8,
                                                                  F32,
                                                                  F8,
                                                                  F32,
                                                                  Tuple<>,
                                                                  BF16,
                                                                  1,
                                                                  128,
                                                                  128,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_gemm_ab_scale_wmma_f8_f8_bf16_mk_nk_mn_1_128_128_mem_v1_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_ABScaleSplitK<Row,
                                                                  Col,
                                                                  Tuple<>,
                                                                  Row,
                                                                  F8,
                                                                  F32,
                                                                  F8,
                                                                  F32,
                                                                  Tuple<>,
                                                                  BF16,
                                                                  1,
                                                                  128,
                                                                  128,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

// Row, Row
void add_device_gemm_ab_scale_wmma_f8_f8_bf16_mk_kn_mn_1_128_128_comp_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_ABScaleSplitK<Row,
                                                                  Row,
                                                                  Tuple<>,
                                                                  Row,
                                                                  F8,
                                                                  F32,
                                                                  F8,
                                                                  F32,
                                                                  Tuple<>,
                                                                  BF16,
                                                                  1,
                                                                  128,
                                                                  128,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_gemm_ab_scale_wmma_f8_f8_bf16_mk_kn_mn_1_128_128_comp_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_ABScaleSplitK<Row,
                                                                  Row,
                                                                  Tuple<>,
                                                                  Row,
                                                                  F8,
                                                                  F32,
                                                                  F8,
                                                                  F32,
                                                                  Tuple<>,
                                                                  BF16,
                                                                  1,
                                                                  128,
                                                                  128,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_gemm_ab_scale_wmma_f8_f8_bf16_mk_kn_mn_1_128_128_mem_v1_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_ABScaleSplitK<Row,
                                                                  Row,
                                                                  Tuple<>,
                                                                  Row,
                                                                  F8,
                                                                  F32,
                                                                  F8,
                                                                  F32,
                                                                  Tuple<>,
                                                                  BF16,
                                                                  1,
                                                                  128,
                                                                  128,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_gemm_ab_scale_wmma_f8_f8_bf16_mk_kn_mn_1_128_128_mem_v1_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_ABScaleSplitK<Row,
                                                                  Row,
                                                                  Tuple<>,
                                                                  Row,
                                                                  F8,
                                                                  F32,
                                                                  F8,
                                                                  F32,
                                                                  Tuple<>,
                                                                  BF16,
                                                                  1,
                                                                  128,
                                                                  128,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

// Col, Row
void add_device_gemm_ab_scale_wmma_f8_f8_bf16_km_kn_mn_1_128_128_comp_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_ABScaleSplitK<Col,
                                                                  Row,
                                                                  Tuple<>,
                                                                  Row,
                                                                  F8,
                                                                  F32,
                                                                  F8,
                                                                  F32,
                                                                  Tuple<>,
                                                                  BF16,
                                                                  1,
                                                                  128,
                                                                  128,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_gemm_ab_scale_wmma_f8_f8_bf16_km_kn_mn_1_128_128_comp_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_ABScaleSplitK<Col,
                                                                  Row,
                                                                  Tuple<>,
                                                                  Row,
                                                                  F8,
                                                                  F32,
                                                                  F8,
                                                                  F32,
                                                                  Tuple<>,
                                                                  BF16,
                                                                  1,
                                                                  128,
                                                                  128,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_gemm_ab_scale_wmma_f8_f8_bf16_km_kn_mn_1_128_128_mem_v1_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_ABScaleSplitK<Col,
                                                                  Row,
                                                                  Tuple<>,
                                                                  Row,
                                                                  F8,
                                                                  F32,
                                                                  F8,
                                                                  F32,
                                                                  Tuple<>,
                                                                  BF16,
                                                                  1,
                                                                  128,
                                                                  128,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_gemm_ab_scale_wmma_f8_f8_bf16_km_kn_mn_1_128_128_mem_v1_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_ABScaleSplitK<Col,
                                                                  Row,
                                                                  Tuple<>,
                                                                  Row,
                                                                  F8,
                                                                  F32,
                                                                  F8,
                                                                  F32,
                                                                  Tuple<>,
                                                                  BF16,
                                                                  1,
                                                                  128,
                                                                  128,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#ifdef CK_USE_XDL
// Row, Col
void add_device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_1_128_128_comp_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_ABScale<Row,
                                                            Col,
                                                            Tuple<>,
                                                            Row,
                                                            F8,
                                                            F32,
                                                            F8,
                                                            F32,
                                                            Tuple<>,
                                                            BF16,
                                                            1,
                                                            128,
                                                            128,
                                                            PassThrough,
                                                            PassThrough,
                                                            PassThrough>>>& instances);

void add_device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_1_128_128_comp_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_ABScale<Row,
                                                            Col,
                                                            Tuple<>,
                                                            Row,
                                                            F8,
                                                            F32,
                                                            F8,
                                                            F32,
                                                            Tuple<>,
                                                            BF16,
                                                            1,
                                                            128,
                                                            128,
                                                            PassThrough,
                                                            PassThrough,
                                                            PassThrough>>>& instances);

void add_device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_1_128_128_mem_v1_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_ABScale<Row,
                                                            Col,
                                                            Tuple<>,
                                                            Row,
                                                            F8,
                                                            F32,
                                                            F8,
                                                            F32,
                                                            Tuple<>,
                                                            BF16,
                                                            1,
                                                            128,
                                                            128,
                                                            PassThrough,
                                                            PassThrough,
                                                            PassThrough>>>& instances);

void add_device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_1_128_128_mem_v1_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_ABScale<Row,
                                                            Col,
                                                            Tuple<>,
                                                            Row,
                                                            F8,
                                                            F32,
                                                            F8,
                                                            F32,
                                                            Tuple<>,
                                                            BF16,
                                                            1,
                                                            128,
                                                            128,
                                                            PassThrough,
                                                            PassThrough,
                                                            PassThrough>>>& instances);

// Row, Row
void add_device_gemm_ab_scale_xdl_f8_f8_bf16_mk_kn_mn_1_128_128_comp_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_ABScale<Row,
                                                            Row,
                                                            Tuple<>,
                                                            Row,
                                                            F8,
                                                            F32,
                                                            F8,
                                                            F32,
                                                            Tuple<>,
                                                            BF16,
                                                            1,
                                                            128,
                                                            128,
                                                            PassThrough,
                                                            PassThrough,
                                                            PassThrough>>>& instances);

void add_device_gemm_ab_scale_xdl_f8_f8_bf16_mk_kn_mn_1_128_128_comp_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_ABScale<Row,
                                                            Row,
                                                            Tuple<>,
                                                            Row,
                                                            F8,
                                                            F32,
                                                            F8,
                                                            F32,
                                                            Tuple<>,
                                                            BF16,
                                                            1,
                                                            128,
                                                            128,
                                                            PassThrough,
                                                            PassThrough,
                                                            PassThrough>>>& instances);

void add_device_gemm_ab_scale_xdl_f8_f8_bf16_mk_kn_mn_1_128_128_mem_v1_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_ABScale<Row,
                                                            Row,
                                                            Tuple<>,
                                                            Row,
                                                            F8,
                                                            F32,
                                                            F8,
                                                            F32,
                                                            Tuple<>,
                                                            BF16,
                                                            1,
                                                            128,
                                                            128,
                                                            PassThrough,
                                                            PassThrough,
                                                            PassThrough>>>& instances);

void add_device_gemm_ab_scale_xdl_f8_f8_bf16_mk_kn_mn_1_128_128_mem_v1_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_ABScale<Row,
                                                            Row,
                                                            Tuple<>,
                                                            Row,
                                                            F8,
                                                            F32,
                                                            F8,
                                                            F32,
                                                            Tuple<>,
                                                            BF16,
                                                            1,
                                                            128,
                                                            128,
                                                            PassThrough,
                                                            PassThrough,
                                                            PassThrough>>>& instances);

// Col, Row
void add_device_gemm_ab_scale_xdl_f8_f8_bf16_km_kn_mn_1_128_128_comp_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_ABScale<Col,
                                                            Row,
                                                            Tuple<>,
                                                            Row,
                                                            F8,
                                                            F32,
                                                            F8,
                                                            F32,
                                                            Tuple<>,
                                                            BF16,
                                                            1,
                                                            128,
                                                            128,
                                                            PassThrough,
                                                            PassThrough,
                                                            PassThrough>>>& instances);

void add_device_gemm_ab_scale_xdl_f8_f8_bf16_km_kn_mn_1_128_128_comp_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_ABScale<Col,
                                                            Row,
                                                            Tuple<>,
                                                            Row,
                                                            F8,
                                                            F32,
                                                            F8,
                                                            F32,
                                                            Tuple<>,
                                                            BF16,
                                                            1,
                                                            128,
                                                            128,
                                                            PassThrough,
                                                            PassThrough,
                                                            PassThrough>>>& instances);

void add_device_gemm_ab_scale_xdl_f8_f8_bf16_km_kn_mn_1_128_128_mem_v1_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_ABScale<Col,
                                                            Row,
                                                            Tuple<>,
                                                            Row,
                                                            F8,
                                                            F32,
                                                            F8,
                                                            F32,
                                                            Tuple<>,
                                                            BF16,
                                                            1,
                                                            128,
                                                            128,
                                                            PassThrough,
                                                            PassThrough,
                                                            PassThrough>>>& instances);

void add_device_gemm_ab_scale_xdl_f8_f8_bf16_km_kn_mn_1_128_128_mem_v1_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_ABScale<Col,
                                                            Row,
                                                            Tuple<>,
                                                            Row,
                                                            F8,
                                                            F32,
                                                            F8,
                                                            F32,
                                                            Tuple<>,
                                                            BF16,
                                                            1,
                                                            128,
                                                            128,
                                                            PassThrough,
                                                            PassThrough,
                                                            PassThrough>>>& instances);
#endif
#endif

template <typename A0DataType,
          typename A1DataType,
          typename B0DataType,
          typename B1DataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGemmMultipleD_ABScaleSplitK<ALayout,
                                                                    BLayout,
                                                                    Tuple<>,
                                                                    CLayout,
                                                                    A0DataType,
                                                                    A1DataType,
                                                                    B0DataType,
                                                                    B1DataType,
                                                                    Tuple<>,
                                                                    CDataType,
                                                                    1,
                                                                    128,
                                                                    128,
                                                                    PassThrough,
                                                                    PassThrough,
                                                                    PassThrough>>
{
    using DeviceOp = DeviceGemmMultipleD_ABScaleSplitK<ALayout,
                                                       BLayout,
                                                       Tuple<>,
                                                       CLayout,
                                                       A0DataType,
                                                       A1DataType,
                                                       B0DataType,
                                                       B1DataType,
                                                       Tuple<>,
                                                       CDataType,
                                                       1,
                                                       128,
                                                       128,
                                                       PassThrough,
                                                       PassThrough,
                                                       PassThrough>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

#if(defined(CK_ENABLE_BF16) || defined(CK_ENABLE_FP8))
#ifdef CK_USE_XDL
        // No XDL instances for DeviceGemmMultipleABDSplitK with Add at the moment
#endif
#ifdef CK_USE_WMMA_FP8
        if constexpr(is_same_v<A0DataType, f8_t> && is_same_v<B0DataType, f8_t> &&
                     is_same_v<CDataType, bhalf_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_ab_scale_wmma_f8_f8_bf16_mk_nk_mn_1_128_128_comp_default_instances(
                    op_ptrs);
                add_device_gemm_ab_scale_wmma_f8_f8_bf16_mk_nk_mn_1_128_128_comp_kpadding_instances(
                    op_ptrs);

                add_device_gemm_ab_scale_wmma_f8_f8_bf16_mk_nk_mn_1_128_128_mem_v1_default_instances(
                    op_ptrs);
                add_device_gemm_ab_scale_wmma_f8_f8_bf16_mk_nk_mn_1_128_128_mem_v1_kpadding_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_ab_scale_wmma_f8_f8_bf16_mk_kn_mn_1_128_128_comp_default_instances(
                    op_ptrs);
                add_device_gemm_ab_scale_wmma_f8_f8_bf16_mk_kn_mn_1_128_128_comp_kpadding_instances(
                    op_ptrs);

                add_device_gemm_ab_scale_wmma_f8_f8_bf16_mk_kn_mn_1_128_128_mem_v1_default_instances(
                    op_ptrs);
                add_device_gemm_ab_scale_wmma_f8_f8_bf16_mk_kn_mn_1_128_128_mem_v1_kpadding_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_ab_scale_wmma_f8_f8_bf16_km_kn_mn_1_128_128_comp_default_instances(
                    op_ptrs);
                add_device_gemm_ab_scale_wmma_f8_f8_bf16_km_kn_mn_1_128_128_comp_kpadding_instances(
                    op_ptrs);

                add_device_gemm_ab_scale_wmma_f8_f8_bf16_km_kn_mn_1_128_128_mem_v1_default_instances(
                    op_ptrs);
                add_device_gemm_ab_scale_wmma_f8_f8_bf16_km_kn_mn_1_128_128_mem_v1_kpadding_instances(
                    op_ptrs);
            }
        }
#endif
#endif
        return op_ptrs;
    }
};

template <typename A0DataType,
          typename A1DataType,
          typename B0DataType,
          typename B1DataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGemmMultipleD_ABScale<ALayout,
                                                              BLayout,
                                                              Tuple<>,
                                                              CLayout,
                                                              A0DataType,
                                                              A1DataType,
                                                              B0DataType,
                                                              B1DataType,
                                                              Tuple<>,
                                                              CDataType,
                                                              1,
                                                              128,
                                                              128,
                                                              PassThrough,
                                                              PassThrough,
                                                              PassThrough>>
{
    using DeviceOp = DeviceGemmMultipleD_ABScale<ALayout,
                                                 BLayout,
                                                 Tuple<>,
                                                 CLayout,
                                                 A0DataType,
                                                 A1DataType,
                                                 B0DataType,
                                                 B1DataType,
                                                 Tuple<>,
                                                 CDataType,
                                                 1,
                                                 128,
                                                 128,
                                                 PassThrough,
                                                 PassThrough,
                                                 PassThrough>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

#if(defined(CK_ENABLE_BF16) || defined(CK_ENABLE_FP8))
#ifdef CK_USE_XDL
        if constexpr(is_same_v<A0DataType, f8_t> && is_same_v<B0DataType, f8_t> &&
                     is_same_v<CDataType, bhalf_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_1_128_128_comp_default_instances(
                    op_ptrs);
                add_device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_1_128_128_comp_kpadding_instances(
                    op_ptrs);

                add_device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_1_128_128_mem_v1_default_instances(
                    op_ptrs);
                add_device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_1_128_128_mem_v1_kpadding_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_ab_scale_xdl_f8_f8_bf16_mk_kn_mn_1_128_128_comp_default_instances(
                    op_ptrs);
                add_device_gemm_ab_scale_xdl_f8_f8_bf16_mk_kn_mn_1_128_128_comp_kpadding_instances(
                    op_ptrs);

                add_device_gemm_ab_scale_xdl_f8_f8_bf16_mk_kn_mn_1_128_128_mem_v1_default_instances(
                    op_ptrs);
                add_device_gemm_ab_scale_xdl_f8_f8_bf16_mk_kn_mn_1_128_128_mem_v1_kpadding_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_ab_scale_xdl_f8_f8_bf16_km_kn_mn_1_128_128_comp_default_instances(
                    op_ptrs);
                add_device_gemm_ab_scale_xdl_f8_f8_bf16_km_kn_mn_1_128_128_comp_kpadding_instances(
                    op_ptrs);

                add_device_gemm_ab_scale_xdl_f8_f8_bf16_km_kn_mn_1_128_128_mem_v1_default_instances(
                    op_ptrs);
                add_device_gemm_ab_scale_xdl_f8_f8_bf16_km_kn_mn_1_128_128_mem_v1_kpadding_instances(
                    op_ptrs);
            }
        }
#endif

#ifdef CK_USE_WMMA_FP8
        using Wrapper = DeviceGemmMultipleD_ABScaleSplitKWrapper<ALayout,
                                                                 BLayout,
                                                                 Tuple<>,
                                                                 CLayout,
                                                                 A0DataType,
                                                                 A1DataType,
                                                                 B0DataType,
                                                                 B1DataType,
                                                                 Tuple<>,
                                                                 CDataType,
                                                                 1,
                                                                 128,
                                                                 128,
                                                                 PassThrough,
                                                                 PassThrough,
                                                                 PassThrough>;

        auto new_op_ptrs =
            DeviceOperationInstanceFactory<typename Wrapper::DeviceOp>::GetInstances();
        for(auto& op_ptr : new_op_ptrs)
        {
            op_ptrs.emplace_back(std::make_unique<Wrapper>(std::move(op_ptr)));
        }
#endif // CK_USE_WMMA_FP8
#endif
        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
