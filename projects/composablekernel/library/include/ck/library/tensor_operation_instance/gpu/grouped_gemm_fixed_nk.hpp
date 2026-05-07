// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <memory>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm_fixed_nk.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// fp16_output
void add_device_grouped_gemm_xdl_fixed_nk_f16_f16_f16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmFixedNK<Row,
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

void add_device_grouped_gemm_xdl_fixed_nk_f16_f16_f16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmFixedNK<Row,
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

// fp8_inputB
void add_device_grouped_gemm_xdl_fixed_nk_f16_f8_f16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmFixedNK<Row,
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

void add_device_grouped_gemm_xdl_fixed_nk_f16_f8_f16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmFixedNK<Row,
                                                         Col,
                                                         Empty_Tuple,
                                                         Row,
                                                         F16,
                                                         F8,
                                                         Empty_Tuple,
                                                         F16,
                                                         PassThrough,
                                                         PassThrough,
                                                         PassThrough>>>& instances);

// i8_inputB
void add_device_grouped_gemm_xdl_fixed_nk_f16_i8_f16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmFixedNK<Row,
                                                         Row,
                                                         Empty_Tuple,
                                                         Row,
                                                         F16,
                                                         I8,
                                                         Empty_Tuple,
                                                         F16,
                                                         PassThrough,
                                                         PassThrough,
                                                         PassThrough>>>& instances);

void add_device_grouped_gemm_xdl_fixed_nk_f16_i8_f16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmFixedNK<Row,
                                                         Col,
                                                         Empty_Tuple,
                                                         Row,
                                                         F16,
                                                         I8,
                                                         Empty_Tuple,
                                                         F16,
                                                         PassThrough,
                                                         PassThrough,
                                                         PassThrough>>>& instances);

// bf16_inputA i8_inputB
#if defined(CK_ENABLE_BF16) && defined(CK_ENABLE_INT8)
void add_device_grouped_gemm_xdl_fixed_nk_bf16_i8_bf16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmFixedNK<Row,
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

void add_device_grouped_gemm_xdl_fixed_nk_bf16_i8_bf16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmFixedNK<Row,
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

// bf16_inputA bf16_inputB
#if defined(CK_ENABLE_BF16)
void add_device_grouped_gemm_xdl_fixed_nk_bf16_bf16_bf16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmFixedNK<Row,
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

void add_device_grouped_gemm_xdl_fixed_nk_bf16_bf16_bf16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmFixedNK<Row,
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

template <typename ALayout,
          typename BLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename EDataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGroupedGemmFixedNK<ALayout,
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
    using DeviceOp = DeviceGroupedGemmFixedNK<ALayout,
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

        // fp16_output
        if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, half_t> &&
                     is_same_v<EDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_fixed_nk_f16_f16_f16_mk_kn_mn_instances(op_ptrs);
            }
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_fixed_nk_f16_f16_f16_mk_nk_mn_instances(op_ptrs);
            }
        }

        // fp8_input
        if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, f8_t> &&
                     is_same_v<EDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_fixed_nk_f16_f8_f16_mk_kn_mn_instances(op_ptrs);
            }
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_fixed_nk_f16_f8_f16_mk_nk_mn_instances(op_ptrs);
            }
        }

        // i8_input
        if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, int8_t> &&
                     is_same_v<EDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_fixed_nk_f16_i8_f16_mk_kn_mn_instances(op_ptrs);
            }
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_fixed_nk_f16_i8_f16_mk_nk_mn_instances(op_ptrs);
            }
        }

// bf16_i8_input
#if defined(CK_ENABLE_BF16) && defined(CK_ENABLE_INT8)
        if constexpr(is_same_v<ADataType, bhalf_t> && is_same_v<BDataType, int8_t> &&
                     is_same_v<EDataType, bhalf_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_fixed_nk_bf16_i8_bf16_mk_kn_mn_instances(op_ptrs);
            }
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_fixed_nk_bf16_i8_bf16_mk_nk_mn_instances(op_ptrs);
            }
        }
#endif

// bf16_inputA bf16_inputB
#if defined(CK_ENABLE_BF16)
        if constexpr(is_same_v<ADataType, bhalf_t> && is_same_v<BDataType, bhalf_t> &&
                     is_same_v<EDataType, bhalf_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_fixed_nk_bf16_bf16_bf16_mk_kn_mn_instances(op_ptrs);
            }
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_fixed_nk_bf16_bf16_bf16_mk_nk_mn_instances(op_ptrs);
            }
        }
#endif // CK_ENABLE_BF16

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
