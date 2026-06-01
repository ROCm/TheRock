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
void add_device_grouped_gemm_xdl_fixed_nk_bias_f16_f16_f16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmFixedNK<Row,
                                                         Row,
                                                         Row_Tuple,
                                                         Row,
                                                         F16,
                                                         F16,
                                                         F32_Tuple,
                                                         F16,
                                                         PassThrough,
                                                         PassThrough,
                                                         Add>>>& instances);

void add_device_grouped_gemm_xdl_fixed_nk_bias_f16_f16_f16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmFixedNK<Row,
                                                         Col,
                                                         Row_Tuple,
                                                         Row,
                                                         F16,
                                                         F16,
                                                         F32_Tuple,
                                                         F16,
                                                         PassThrough,
                                                         PassThrough,
                                                         Add>>>& instances);

// fp32_output
void add_device_grouped_gemm_xdl_fixed_nk_bias_f16_f16_f32_mk_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmFixedNK<Row,
                                                         Row,
                                                         Row_Tuple,
                                                         Row,
                                                         F16,
                                                         F16,
                                                         F32_Tuple,
                                                         F32,
                                                         PassThrough,
                                                         PassThrough,
                                                         Add>>>& instances);

void add_device_grouped_gemm_xdl_fixed_nk_bias_f16_f16_f32_mk_nk_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmFixedNK<Row,
                                                         Col,
                                                         Row_Tuple,
                                                         Row,
                                                         F16,
                                                         F16,
                                                         F32_Tuple,
                                                         F32,
                                                         PassThrough,
                                                         PassThrough,
                                                         Add>>>& instances);

template <typename ALayout,
          typename BLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename EDataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGroupedGemmFixedNK<ALayout,
                                                           BLayout,
                                                           Row_Tuple,
                                                           ELayout,
                                                           ADataType,
                                                           BDataType,
                                                           F32_Tuple,
                                                           EDataType,
                                                           PassThrough,
                                                           PassThrough,
                                                           Add>>
{
    using DeviceOp = DeviceGroupedGemmFixedNK<ALayout,
                                              BLayout,
                                              Row_Tuple,
                                              ELayout,
                                              ADataType,
                                              BDataType,
                                              F32_Tuple,
                                              EDataType,
                                              PassThrough,
                                              PassThrough,
                                              Add>;

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
                add_device_grouped_gemm_xdl_fixed_nk_bias_f16_f16_f16_mk_kn_mn_instances(op_ptrs);
            }
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_fixed_nk_bias_f16_f16_f16_mk_nk_mn_instances(op_ptrs);
            }
        }

        // fp32_output
        if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, half_t> &&
                     is_same_v<EDataType, float>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_fixed_nk_bias_f16_f16_f32_mk_kn_mn_instances(op_ptrs);
            }
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_fixed_nk_bias_f16_f16_f32_mk_nk_mn_instances(op_ptrs);
            }
        }
        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
