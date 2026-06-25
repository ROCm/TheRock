// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/device_elementwise.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

#ifdef CK_ENABLE_FP16
void add_device_batchnorm_infer_rank_4_f16_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceElementwise<
        ck::Tuple<F16, F32, F32, F16, F16>,
        ck::Tuple<F16>,
        ck::tensor_operation::element_wise::NormalizeInInfer,
        4>>>&);
#endif
#ifdef CK_ENABLE_FP32
void add_device_batchnorm_infer_rank_4_f32_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceElementwise<
        ck::Tuple<F32, F32, F32, F32, F32>,
        ck::Tuple<F32>,
        ck::tensor_operation::element_wise::NormalizeInInfer,
        4>>>&);
#endif
#ifdef CK_ENABLE_BF16
void add_device_batchnorm_infer_rank_4_bf16_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceElementwise<
        ck::Tuple<BF16, F32, F32, BF16, BF16>,
        ck::Tuple<BF16>,
        ck::tensor_operation::element_wise::NormalizeInInfer,
        4>>>&);
#endif
#ifdef CK_ENABLE_FP64
void add_device_batchnorm_infer_rank_4_f64_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceElementwise<
        ck::Tuple<F64, F64, F64, F64, F64>,
        ck::Tuple<F64>,
        ck::tensor_operation::element_wise::NormalizeInInfer,
        4>>>&);
#endif
template <typename XDataType,
          typename YDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename MeanVarDataType,
          index_t Rank>
struct DeviceOperationInstanceFactory<ck::tensor_operation::device::DeviceElementwise<
    ck::Tuple<XDataType, MeanVarDataType, MeanVarDataType, ScaleDataType, BiasDataType>,
    ck::Tuple<YDataType>,
    ck::tensor_operation::element_wise::NormalizeInInfer,
    Rank>>
{
    using DeviceOp = ck::tensor_operation::device::DeviceElementwise<
        ck::Tuple<XDataType, MeanVarDataType, MeanVarDataType, ScaleDataType, BiasDataType>,
        ck::Tuple<YDataType>,
        ck::tensor_operation::element_wise::NormalizeInInfer,
        Rank>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;
#ifdef CK_ENABLE_FP16
        if constexpr(is_same_v<XDataType, F16> && is_same_v<YDataType, F16> &&
                     is_same_v<ScaleDataType, F16> && is_same_v<BiasDataType, F16> &&
                     is_same_v<MeanVarDataType, F32>)
        {
            if constexpr(Rank == 4)
            {
                add_device_batchnorm_infer_rank_4_f16_instances(op_ptrs);
            }
        }
#endif
#ifdef CK_ENABLE_FP32
        if constexpr(is_same_v<XDataType, F32> && is_same_v<YDataType, F32> &&
                     is_same_v<ScaleDataType, F32> && is_same_v<BiasDataType, F32> &&
                     is_same_v<MeanVarDataType, F32>)
        {
            if constexpr(Rank == 4)
            {
                add_device_batchnorm_infer_rank_4_f32_instances(op_ptrs);
            }
        }
#endif
#ifdef CK_ENABLE_BF16
        if constexpr(is_same_v<XDataType, BF16> && is_same_v<YDataType, BF16> &&
                     is_same_v<ScaleDataType, BF16> && is_same_v<BiasDataType, BF16> &&
                     is_same_v<MeanVarDataType, F32>)
        {
            if constexpr(Rank == 4)
            {
                add_device_batchnorm_infer_rank_4_bf16_instances(op_ptrs);
            }
        }
#endif
#ifdef CK_ENABLE_FP64
        if constexpr(is_same_v<XDataType, F64> && is_same_v<YDataType, F64> &&
                     is_same_v<ScaleDataType, F64> && is_same_v<BiasDataType, F64> &&
                     is_same_v<MeanVarDataType, F64>)
        {
            if constexpr(Rank == 4)
            {
                add_device_batchnorm_infer_rank_4_f64_instances(op_ptrs);
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
