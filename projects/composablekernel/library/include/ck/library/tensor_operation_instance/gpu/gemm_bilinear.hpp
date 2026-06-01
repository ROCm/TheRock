// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <memory>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_d.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
#if defined(CK_USE_XDL)
#if defined(CK_ENABLE_FP16)
void add_device_gemm_bilinear_xdl_c_shuffle_f16_f16_f16_f16_km_kn_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Col,
                                                    Row,
                                                    Row_Tuple,
                                                    Row,
                                                    F16,
                                                    F16,
                                                    F16_Tuple,
                                                    F16,
                                                    PassThrough,
                                                    PassThrough,
                                                    Bilinear>>>& instances);

void add_device_gemm_bilinear_xdl_c_shuffle_f16_f16_f16_f16_km_nk_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Col,
                                                    Col,
                                                    Row_Tuple,
                                                    Row,
                                                    F16,
                                                    F16,
                                                    F16_Tuple,
                                                    F16,
                                                    PassThrough,
                                                    PassThrough,
                                                    Bilinear>>>& instances);

void add_device_gemm_bilinear_xdl_c_shuffle_f16_f16_f16_f16_mk_kn_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Row,
                                                    Row,
                                                    Row_Tuple,
                                                    Row,
                                                    F16,
                                                    F16,
                                                    F16_Tuple,
                                                    F16,
                                                    PassThrough,
                                                    PassThrough,
                                                    Bilinear>>>& instances);

void add_device_gemm_bilinear_xdl_c_shuffle_f16_f16_f16_f16_mk_nk_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Row,
                                                    Col,
                                                    Row_Tuple,
                                                    Row,
                                                    F16,
                                                    F16,
                                                    F16_Tuple,
                                                    F16,
                                                    PassThrough,
                                                    PassThrough,
                                                    Bilinear>>>& instances);
#endif // CK_ENABLE_FP16
#endif // CK_USE_XDL

#if defined(CK_USE_WMMA)
#if defined(CK_ENABLE_INT8)
void add_device_gemm_bilinear_wmma_c_shuffle_i8_i8_i8_i8_mk_kn_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Row,
                                                    Row,
                                                    Row_Tuple,
                                                    Row,
                                                    I8,
                                                    I8,
                                                    I8_Tuple,
                                                    I8,
                                                    PassThrough,
                                                    PassThrough,
                                                    Bilinear>>>& instances);

void add_device_gemm_bilinear_wmma_c_shuffle_i8_i8_i8_i8_mk_nk_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Row,
                                                    Col,
                                                    Row_Tuple,
                                                    Row,
                                                    I8,
                                                    I8,
                                                    I8_Tuple,
                                                    I8,
                                                    PassThrough,
                                                    PassThrough,
                                                    Bilinear>>>& instances);

void add_device_gemm_bilinear_wmma_c_shuffle_i8_i8_i8_i8_km_kn_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Col,
                                                    Row,
                                                    Row_Tuple,
                                                    Row,
                                                    I8,
                                                    I8,
                                                    I8_Tuple,
                                                    I8,
                                                    PassThrough,
                                                    PassThrough,
                                                    Bilinear>>>& instances);

void add_device_gemm_bilinear_wmma_c_shuffle_i8_i8_i8_i8_km_nk_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Col,
                                                    Col,
                                                    Row_Tuple,
                                                    Row,
                                                    I8,
                                                    I8,
                                                    I8_Tuple,
                                                    I8,
                                                    PassThrough,
                                                    PassThrough,
                                                    Bilinear>>>& instances);
#endif // CK_ENABLE_INT8

#if defined(CK_ENABLE_FP16)
void add_device_gemm_bilinear_wmma_c_shuffle_f16_f16_f16_f16_km_kn_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Col,
                                                          Row,
                                                          Row_Tuple,
                                                          Row,
                                                          F16,
                                                          F16,
                                                          F16_Tuple,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          Bilinear>>>& instances);

void add_device_gemm_bilinear_wmma_c_shuffle_f16_f16_f16_f16_km_nk_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Col,
                                                          Col,
                                                          Row_Tuple,
                                                          Row,
                                                          F16,
                                                          F16,
                                                          F16_Tuple,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          Bilinear>>>& instances);

void add_device_gemm_bilinear_wmma_c_shuffle_f16_f16_f16_f16_mk_kn_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Row,
                                                          Row_Tuple,
                                                          Row,
                                                          F16,
                                                          F16,
                                                          F16_Tuple,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          Bilinear>>>& instances);

void add_device_gemm_bilinear_wmma_c_shuffle_f16_f16_f16_f16_mk_nk_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Row_Tuple,
                                                          Row,
                                                          F16,
                                                          F16,
                                                          F16_Tuple,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          Bilinear>>>& instances);
#endif // CK_ENABLE_FP16
#endif // CK_USE_WMMA

// GEMM + Bilinear
template <typename ALayout,
          typename BLayout,
          typename DLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename DDataType,
          typename EDataType>
struct DeviceOperationInstanceFactory<DeviceGemmMultipleDSplitK<ALayout,
                                                                BLayout,
                                                                ck::Tuple<DLayout>,
                                                                ELayout,
                                                                ADataType,
                                                                BDataType,
                                                                ck::Tuple<DDataType>,
                                                                EDataType,
                                                                PassThrough,
                                                                PassThrough,
                                                                Bilinear>>
{
    using DeviceOp = DeviceGemmMultipleDSplitK<ALayout,
                                               BLayout,
                                               ck::Tuple<DLayout>,
                                               ELayout,
                                               ADataType,
                                               BDataType,
                                               ck::Tuple<DDataType>,
                                               EDataType,
                                               PassThrough,
                                               PassThrough,
                                               Bilinear>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

#if defined(CK_USE_XDL)
        // No XDL instances for DeviceGemmMultipleDSplitK with AddBilinear at the moment
#endif // CK_USE_XDL

#if defined(CK_USE_WMMA)
#if defined(CK_ENABLE_FP16)
        if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, half_t> &&
                     is_same_v<DDataType, half_t> && is_same_v<EDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<DLayout, Row> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_bilinear_wmma_c_shuffle_f16_f16_f16_f16_mk_kn_mn_mn_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<DLayout, Row> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_bilinear_wmma_c_shuffle_f16_f16_f16_f16_mk_nk_mn_mn_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<DLayout, Row> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_bilinear_wmma_c_shuffle_f16_f16_f16_f16_km_kn_mn_mn_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<DLayout, Row> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_bilinear_wmma_c_shuffle_f16_f16_f16_f16_km_nk_mn_mn_instances(
                    op_ptrs);
            }
        }
#endif // CK_ENABLE_FP16
#endif // CK_USE_WMMA

        return op_ptrs;
    }
};

// GEMM + Bilinear
template <typename ALayout,
          typename BLayout,
          typename DLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename DDataType,
          typename EDataType>
struct DeviceOperationInstanceFactory<DeviceGemmMultipleD<ALayout,
                                                          BLayout,
                                                          ck::Tuple<DLayout>,
                                                          ELayout,
                                                          ADataType,
                                                          BDataType,
                                                          ck::Tuple<DDataType>,
                                                          EDataType,
                                                          PassThrough,
                                                          PassThrough,
                                                          Bilinear>>
{
    using DeviceOp = DeviceGemmMultipleD<ALayout,
                                         BLayout,
                                         ck::Tuple<DLayout>,
                                         ELayout,
                                         ADataType,
                                         BDataType,
                                         ck::Tuple<DDataType>,
                                         EDataType,
                                         PassThrough,
                                         PassThrough,
                                         Bilinear>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;
#if defined(CK_USE_XDL)
#if defined(CK_ENABLE_FP16)
        if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, half_t> &&
                     is_same_v<DDataType, half_t> && is_same_v<EDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<DLayout, Row> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_bilinear_xdl_c_shuffle_f16_f16_f16_f16_mk_kn_mn_mn_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<DLayout, Row> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_bilinear_xdl_c_shuffle_f16_f16_f16_f16_mk_nk_mn_mn_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<DLayout, Row> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_bilinear_xdl_c_shuffle_f16_f16_f16_f16_km_kn_mn_mn_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<DLayout, Row> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_bilinear_xdl_c_shuffle_f16_f16_f16_f16_km_nk_mn_mn_instances(
                    op_ptrs);
            }
        }
#endif // CK_ENABLE_FP16
#endif // CK_USE_XDL

#if defined(CK_USE_WMMA)
        // Reuse DeviceGemmMultipleDSplitK instances
        using Wrapper = DeviceGemmMultipleDSplitKWrapper<ALayout,
                                                         BLayout,
                                                         ck::Tuple<DLayout>,
                                                         ELayout,
                                                         ADataType,
                                                         BDataType,
                                                         ck::Tuple<DDataType>,
                                                         EDataType,
                                                         PassThrough,
                                                         PassThrough,
                                                         Bilinear>;
        auto new_op_ptrs =
            DeviceOperationInstanceFactory<typename Wrapper::DeviceOp>::GetInstances();
        for(auto& op_ptr : new_op_ptrs)
        {
            op_ptrs.emplace_back(std::make_unique<Wrapper>(std::move(op_ptr)));
        }

        // Bilinear wmma i8 instances are using DeviceGemmMultipleD interface.
#if defined(CK_ENABLE_INT8)
        if constexpr(is_same_v<ADataType, std::int8_t> && is_same_v<BDataType, std::int8_t> &&
                     is_same_v<DDataType, std::int8_t> && is_same_v<EDataType, std::int8_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<DLayout, Row> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_bilinear_wmma_c_shuffle_i8_i8_i8_i8_mk_kn_mn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<DLayout, Row> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_bilinear_wmma_c_shuffle_i8_i8_i8_i8_mk_nk_mn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<DLayout, Row> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_bilinear_wmma_c_shuffle_i8_i8_i8_i8_km_kn_mn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<DLayout, Row> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_bilinear_wmma_c_shuffle_i8_i8_i8_i8_km_nk_mn_mn_instances(op_ptrs);
            }
        }
#endif // CK_ENABLE_INT8
#endif // CK_USE_WMMA
        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
