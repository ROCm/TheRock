// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <memory>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_d.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

#if defined(CK_ENABLE_FP16)
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

#if defined(CK_USE_XDL)
void add_device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_f16_mk_kn_mn_mn_mn_instances(
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
                                                    AddAddFastGelu>>>&);

void add_device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_f16_mk_nk_mn_mn_mn_instances(
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
                                                    AddAddFastGelu>>>&);

void add_device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_f16_km_kn_mn_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Col,
                                                    Row,
                                                    Row_Row_Tuple,
                                                    Row,
                                                    F16,
                                                    F16,
                                                    F16_F16_Tuple,
                                                    F16,
                                                    PassThrough,
                                                    PassThrough,
                                                    AddAddFastGelu>>>&);

void add_device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_f16_km_nk_mn_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Col,
                                                    Col,
                                                    Row_Row_Tuple,
                                                    Row,
                                                    F16,
                                                    F16,
                                                    F16_F16_Tuple,
                                                    F16,
                                                    PassThrough,
                                                    PassThrough,
                                                    AddAddFastGelu>>>&);
#endif // CK_USE_XDL

#if defined(CK_USE_WMMA)
void add_device_gemm_add_add_fastgelu_wmma_c_shuffle_f16_f16_f16_f16_f16_mk_kn_mn_mn_mn_instances(
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
                                                          AddAddFastGelu>>>&);

void add_device_gemm_add_add_fastgelu_wmma_c_shuffle_f16_f16_f16_f16_f16_mk_nk_mn_mn_mn_instances(
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
                                                          AddAddFastGelu>>>&);

void add_device_gemm_add_add_fastgelu_wmma_c_shuffle_f16_f16_f16_f16_f16_km_kn_mn_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Col,
                                                          Row,
                                                          Row_Row_Tuple,
                                                          Row,
                                                          F16,
                                                          F16,
                                                          F16_F16_Tuple,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          AddAddFastGelu>>>&);

void add_device_gemm_add_add_fastgelu_wmma_c_shuffle_f16_f16_f16_f16_f16_km_nk_mn_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Col,
                                                          Col,
                                                          Row_Row_Tuple,
                                                          Row,
                                                          F16,
                                                          F16,
                                                          F16_F16_Tuple,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          AddAddFastGelu>>>&);
#endif // CK_USE_WMMA

// GEMM + Add + FastGelu
// DeviceGemmMultipleDSplitK specialization
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
                                                                AddAddFastGelu>>
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
                                               AddAddFastGelu>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

#if defined(CK_USE_XDL)
        // No XDL instances for DeviceGemmMultipleDSplitK with AddFastGelu at the moment
#endif // CK_USE_XDL

#if defined(CK_USE_WMMA)
        constexpr bool IsAllDRowLayout = is_same_v<D0Layout, Row> && is_same_v<D1Layout, Row>;
        constexpr bool IsAllDFloat16 =
            is_same_v<D0DataType, half_t> && is_same_v<D1DataType, half_t>;

        if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, half_t> &&
                     is_same_v<EDataType, half_t> && IsAllDRowLayout && IsAllDFloat16)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_gemm_add_add_fastgelu_wmma_c_shuffle_f16_f16_f16_f16_f16_mk_kn_mn_mn_mn_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<ELayout, Row>)
            {
                add_device_gemm_add_add_fastgelu_wmma_c_shuffle_f16_f16_f16_f16_f16_mk_nk_mn_mn_mn_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<ELayout, Row>)
            {
                add_device_gemm_add_add_fastgelu_wmma_c_shuffle_f16_f16_f16_f16_f16_km_kn_mn_mn_mn_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<ELayout, Row>)
            {
                add_device_gemm_add_add_fastgelu_wmma_c_shuffle_f16_f16_f16_f16_f16_km_nk_mn_mn_mn_instances(
                    op_ptrs);
            }
        }
#endif // CK_USE_WMMA

        return op_ptrs;
    }
};

// GEMM + Add + Add + FastGelu
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
                                                          AddAddFastGelu>>
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
                                         AddAddFastGelu>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

#if defined(CK_USE_XDL)
        constexpr bool IsAllDRowLayout = is_same_v<D0Layout, Row> && is_same_v<D1Layout, Row>;
        constexpr bool IsAllDFloat16 =
            is_same_v<D0DataType, half_t> && is_same_v<D1DataType, half_t>;

        if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, half_t> &&
                     is_same_v<EDataType, half_t> && IsAllDRowLayout && IsAllDFloat16)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_f16_mk_kn_mn_mn_mn_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<ELayout, Row>)
            {
                add_device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_f16_mk_nk_mn_mn_mn_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<ELayout, Row>)
            {
                add_device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_f16_km_kn_mn_mn_mn_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<ELayout, Row>)
            {
                add_device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_f16_km_nk_mn_mn_mn_instances(
                    op_ptrs);
            }
        }
#endif // CK_USE_XDL

#if defined(CK_USE_WMMA)
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
                                                         AddAddFastGelu>;
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
#endif // CK_ENABLE_FP16
