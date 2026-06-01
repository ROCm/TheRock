// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
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
void add_device_gemm_multiply_add_xdl_c_shuffle_f16_f16_f16_f16_f16_mk_kn_mn_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Row,
                                                    Row,
                                                    Row_Row_Tuple,
                                                    Row,
                                                    F16,
                                                    F16,
                                                    F16_F16_Tuple,
                                                    F16,
                                                    PassThrough,
                                                    PassThrough,
                                                    MultiplyAdd>>>&);

void add_device_gemm_multiply_add_xdl_c_shuffle_f16_f16_f16_f16_f16_mk_nk_mn_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Row,
                                                    Col,
                                                    Row_Row_Tuple,
                                                    Row,
                                                    F16,
                                                    F16,
                                                    F16_F16_Tuple,
                                                    F16,
                                                    PassThrough,
                                                    PassThrough,
                                                    MultiplyAdd>>>&);

#if defined CK_ENABLE_FP8
void add_device_gemm_multiply_add_xdl_c_shuffle_f16_f8_f32_f32_f16_mk_kn_mn_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Row,
                                                    Row,
                                                    Row_Row_Tuple,
                                                    Row,
                                                    F16,
                                                    F8,
                                                    F32_F32_Tuple,
                                                    F16,
                                                    PassThrough,
                                                    PassThrough,
                                                    MultiplyAdd>>>&);

void add_device_gemm_multiply_add_xdl_c_shuffle_f16_f8_f32_f32_f16_mk_nk_mn_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Row,
                                                    Col,
                                                    Row_Row_Tuple,
                                                    Row,
                                                    F16,
                                                    F8,
                                                    F32_F32_Tuple,
                                                    F16,
                                                    PassThrough,
                                                    PassThrough,
                                                    MultiplyAdd>>>&);
#endif // CK_ENABLE_FP8
#endif // CK_USE_XDL

#if defined(CK_USE_WMMA)
void add_device_gemm_multiply_add_wmma_c_shuffle_f16_f16_f16_f16_f16_mk_kn_mn_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Row,
                                                          Row_Row_Tuple,
                                                          Row,
                                                          F16,
                                                          F16,
                                                          F16_F16_Tuple,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyAdd>>>&);

void add_device_gemm_multiply_add_wmma_c_shuffle_f16_f16_f16_f16_f16_mk_nk_mn_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Row_Row_Tuple,
                                                          Row,
                                                          F16,
                                                          F16,
                                                          F16_F16_Tuple,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyAdd>>>&);
#ifdef CK_USE_WMMA_FP8
void add_device_gemm_multiply_add_wmma_c_shuffle_f16_f8_f32_f32_f16_mk_kn_mn_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Row,
                                                          Row_Row_Tuple,
                                                          Row,
                                                          F16,
                                                          F8,
                                                          F32_F32_Tuple,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyAdd>>>&);

void add_device_gemm_multiply_add_wmma_c_shuffle_f16_f8_f32_f32_f16_mk_nk_mn_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Row_Row_Tuple,
                                                          Row,
                                                          F16,
                                                          F8,
                                                          F32_F32_Tuple,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyAdd>>>&);
#endif // CK_USE_WMMA_FP8
#endif // CK_USE_WMMA

template <typename ALayout,
          typename BLayout,
          typename D0Layout,
          typename D1Layout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename D0DataType,
          typename D1DataType,
          typename EDataType>
struct DeviceOperationInstanceFactory<DeviceGemmMultipleDSplitK<ALayout,
                                                                BLayout,
                                                                ck::Tuple<D0Layout, D1Layout>,
                                                                ELayout,
                                                                ADataType,
                                                                BDataType,
                                                                ck::Tuple<D0DataType, D1DataType>,
                                                                EDataType,
                                                                PassThrough,
                                                                PassThrough,
                                                                MultiplyAdd>>
{
    using DeviceOp = DeviceGemmMultipleDSplitK<ALayout,
                                               BLayout,
                                               ck::Tuple<D0Layout, D1Layout>,
                                               ELayout,
                                               ADataType,
                                               BDataType,
                                               ck::Tuple<D0DataType, D1DataType>,
                                               EDataType,
                                               PassThrough,
                                               PassThrough,
                                               MultiplyAdd>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

#ifdef CK_USE_XDL
        // No XDL instances for DeviceGemmMultipleDSplitK with MultiplyAdd at the moment
#endif // CK_USE_XDL

#ifdef CK_USE_WMMA
        if constexpr(is_same_v<ADataType, F16> && is_same_v<BDataType, F16> &&
                     is_same_v<D0DataType, F16> && is_same_v<D1DataType, F16> &&
                     is_same_v<EDataType, F16>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<D0Layout, Row> && is_same_v<D1Layout, Row> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_gemm_multiply_add_wmma_c_shuffle_f16_f16_f16_f16_f16_mk_kn_mn_mn_mn_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<D0Layout, Row> && is_same_v<D1Layout, Row> &&
                              is_same_v<ELayout, Row>)
            {
                add_device_gemm_multiply_add_wmma_c_shuffle_f16_f16_f16_f16_f16_mk_nk_mn_mn_mn_instances(
                    op_ptrs);
            }
        }
#endif // CK_USE_WMMA
#ifdef CK_USE_WMMA_FP8
        if constexpr(is_same_v<ADataType, F16> && is_same_v<BDataType, F8> &&
                     is_same_v<D0DataType, F32> && is_same_v<D1DataType, F32> &&
                     is_same_v<EDataType, F16>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<D0Layout, Row> && is_same_v<D1Layout, Row> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_gemm_multiply_add_wmma_c_shuffle_f16_f8_f32_f32_f16_mk_kn_mn_mn_mn_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<D0Layout, Row> && is_same_v<D1Layout, Row> &&
                              is_same_v<ELayout, Row>)
            {
                add_device_gemm_multiply_add_wmma_c_shuffle_f16_f8_f32_f32_f16_mk_nk_mn_mn_mn_instances(
                    op_ptrs);
            }
        }
#endif // CK_USE_WMMA

        return op_ptrs;
    }
};

// DeviceGemmMultipleD specialization
template <typename ALayout,
          typename BLayout,
          typename D0Layout,
          typename D1Layout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename D0DataType,
          typename D1DataType,
          typename EDataType>
struct DeviceOperationInstanceFactory<DeviceGemmMultipleD<ALayout,
                                                          BLayout,
                                                          ck::Tuple<D0Layout, D1Layout>,
                                                          ELayout,
                                                          ADataType,
                                                          BDataType,
                                                          ck::Tuple<D0DataType, D1DataType>,
                                                          EDataType,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyAdd>>
{
    using DeviceOp = DeviceGemmMultipleD<ALayout,
                                         BLayout,
                                         ck::Tuple<D0Layout, D1Layout>,
                                         ELayout,
                                         ADataType,
                                         BDataType,
                                         ck::Tuple<D0DataType, D1DataType>,
                                         EDataType,
                                         PassThrough,
                                         PassThrough,
                                         MultiplyAdd>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

#ifdef CK_USE_XDL
        if constexpr(is_same_v<ADataType, F16> && is_same_v<BDataType, F16> &&
                     is_same_v<D0DataType, F16> && is_same_v<D1DataType, F16> &&
                     is_same_v<EDataType, F16>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<D0Layout, Row> && is_same_v<D1Layout, Row> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_gemm_multiply_add_xdl_c_shuffle_f16_f16_f16_f16_f16_mk_kn_mn_mn_mn_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<D0Layout, Row> && is_same_v<D1Layout, Row> &&
                              is_same_v<ELayout, Row>)
            {
                add_device_gemm_multiply_add_xdl_c_shuffle_f16_f16_f16_f16_f16_mk_nk_mn_mn_mn_instances(
                    op_ptrs);
            }
        }

#ifdef CK_ENABLE_FP8
        if constexpr(is_same_v<ADataType, F16> && is_same_v<BDataType, F8> &&
                     is_same_v<D0DataType, F32> && is_same_v<D1DataType, F32> &&
                     is_same_v<EDataType, F16>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<D0Layout, Row> && is_same_v<D1Layout, Row> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_gemm_multiply_add_xdl_c_shuffle_f16_f8_f32_f32_f16_mk_kn_mn_mn_mn_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<D0Layout, Row> && is_same_v<D1Layout, Row> &&
                              is_same_v<ELayout, Row>)
            {
                add_device_gemm_multiply_add_xdl_c_shuffle_f16_f8_f32_f32_f16_mk_nk_mn_mn_mn_instances(
                    op_ptrs);
            }
        }
#endif // CK_ENABLE_FP8
#endif // CK_USE_XDL

#ifdef CK_USE_WMMA
        // Reuse DeviceGemmMultipleDSplitK instances
        using Wrapper = DeviceGemmMultipleDSplitKWrapper<ALayout,
                                                         BLayout,
                                                         ck::Tuple<D0Layout, D1Layout>,
                                                         ELayout,
                                                         ADataType,
                                                         BDataType,
                                                         ck::Tuple<D0DataType, D1DataType>,
                                                         EDataType,
                                                         PassThrough,
                                                         PassThrough,
                                                         MultiplyAdd>;
        auto new_op_ptrs =
            DeviceOperationInstanceFactory<typename Wrapper::DeviceOp>::GetInstances();
        for(auto& op_ptr : new_op_ptrs)
        {
            op_ptrs.emplace_back(std::make_unique<Wrapper>(std::move(op_ptr)));
        }
#endif // CK_USE_WMMA

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
