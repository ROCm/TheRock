// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_normalization_fwd.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// FP16
void add_device_normalization_fwd_rank_5_3_swish_f16_instances(
    std::vector<std::unique_ptr<DeviceNormalizationFwd<F16, F16, F16, F16, F16, Swish, 5, 3>>>&);

// FP32
void add_device_normalization_fwd_rank_5_3_swish_f32_instances(
    std::vector<std::unique_ptr<DeviceNormalizationFwd<F32, F32, F32, F32, F32, Swish, 5, 3>>>&);

// [x, gamma, beta, y] = [f16, f32, f32, f16]
void add_device_normalization_fwd_rank_5_3_swish_f16_f32_f32_f16_instances(
    std::vector<std::unique_ptr<DeviceNormalizationFwd<F16, F32, F32, F16, F32, Swish, 5, 3>>>&);

template <typename XDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename YDataType,
          typename SaveMeanInvStdDataType,
          index_t Rank,
          index_t NumReduceDim>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceNormalizationFwd<XDataType,
                                                         GammaDataType,
                                                         BetaDataType,
                                                         YDataType,
                                                         SaveMeanInvStdDataType,
                                                         ck::tensor_operation::element_wise::Swish,
                                                         Rank,
                                                         NumReduceDim>>
{
    using DeviceOp = DeviceNormalizationFwd<XDataType,
                                            GammaDataType,
                                            BetaDataType,
                                            YDataType,
                                            SaveMeanInvStdDataType,
                                            ck::tensor_operation::element_wise::Swish,
                                            Rank,
                                            NumReduceDim>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(is_same_v<XDataType, F16> && is_same_v<GammaDataType, F16> &&
                     is_same_v<BetaDataType, F16> && is_same_v<YDataType, F16> &&
                     is_same_v<SaveMeanInvStdDataType, F16>)
        {
            if constexpr(Rank == 5 && NumReduceDim == 3)
            {
                add_device_normalization_fwd_rank_5_3_swish_f16_instances(op_ptrs);
            }
        }
        else if constexpr(is_same_v<XDataType, F32> && is_same_v<GammaDataType, F32> &&
                          is_same_v<BetaDataType, F32> && is_same_v<YDataType, F32> &&
                          is_same_v<SaveMeanInvStdDataType, F32>)
        {
            if constexpr(Rank == 5 && NumReduceDim == 3)
            {
                add_device_normalization_fwd_rank_5_3_swish_f32_instances(op_ptrs);
            }
        }
        else if constexpr(is_same_v<XDataType, F16> && is_same_v<GammaDataType, F32> &&
                          is_same_v<BetaDataType, F32> && is_same_v<YDataType, F16> &&
                          is_same_v<SaveMeanInvStdDataType, F32>)
        {
            if constexpr(Rank == 5 && NumReduceDim == 3)
            {
                add_device_normalization_fwd_rank_5_3_swish_f16_f32_f32_f16_instances(op_ptrs);
            }
        }

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
