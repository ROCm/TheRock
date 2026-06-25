// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <memory>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_normalization_fwd.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
#ifdef CK_ENABLE_FP16
// FP16
void add_device_normalization_fwd_rank_2_1_f16_instances(
    std::vector<
        std::unique_ptr<DeviceNormalizationFwd<F16, F16, F16, F16, F16, PassThrough, 2, 1>>>&);

void add_device_normalization_fwd_rank_4_3_f16_instances(
    std::vector<
        std::unique_ptr<DeviceNormalizationFwd<F16, F16, F16, F16, F16, PassThrough, 4, 3>>>&);

void add_device_normalization_fwd_rank_5_3_f16_instances(
    std::vector<
        std::unique_ptr<DeviceNormalizationFwd<F16, F16, F16, F16, F16, PassThrough, 5, 3>>>&);
#endif
#ifdef CK_ENABLE_FP32
// FP32
void add_device_normalization_fwd_rank_2_1_f32_instances(
    std::vector<
        std::unique_ptr<DeviceNormalizationFwd<F32, F32, F32, F32, F32, PassThrough, 2, 1>>>&);

void add_device_normalization_fwd_rank_4_3_f32_instances(
    std::vector<
        std::unique_ptr<DeviceNormalizationFwd<F32, F32, F32, F32, F32, PassThrough, 4, 3>>>&);

void add_device_normalization_fwd_rank_5_3_f32_instances(
    std::vector<
        std::unique_ptr<DeviceNormalizationFwd<F32, F32, F32, F32, F32, PassThrough, 5, 3>>>&);
#endif
template <typename XDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename YDataType,
          typename SaveMeanInvStdDataType,
          index_t Rank,
          index_t NumReduceDim>
struct DeviceOperationInstanceFactory<ck::tensor_operation::device::DeviceNormalizationFwd<
    XDataType,
    GammaDataType,
    BetaDataType,
    YDataType,
    SaveMeanInvStdDataType,
    ck::tensor_operation::element_wise::PassThrough,
    Rank,
    NumReduceDim>>
{
    using DeviceOp = DeviceNormalizationFwd<XDataType,
                                            GammaDataType,
                                            BetaDataType,
                                            YDataType,
                                            SaveMeanInvStdDataType,
                                            ck::tensor_operation::element_wise::PassThrough,
                                            Rank,
                                            NumReduceDim>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;
#ifdef CK_ENABLE_FP16
        if constexpr(is_same_v<XDataType, F16> && is_same_v<GammaDataType, F16> &&
                     is_same_v<BetaDataType, F16> && is_same_v<YDataType, F16> &&
                     is_same_v<SaveMeanInvStdDataType, F16>)
        {
            if constexpr(Rank == 2 && NumReduceDim == 1)
            {
                add_device_normalization_fwd_rank_2_1_f16_instances(op_ptrs);
            }
            else if constexpr(Rank == 4 && NumReduceDim == 3)
            {
                add_device_normalization_fwd_rank_4_3_f16_instances(op_ptrs);
            }
            else if constexpr(Rank == 5 && NumReduceDim == 3)
            {
                add_device_normalization_fwd_rank_5_3_f16_instances(op_ptrs);
            }
        }
#endif
#ifdef CK_ENABLE_FP32
        if constexpr(is_same_v<XDataType, F32> && is_same_v<GammaDataType, F32> &&
                     is_same_v<BetaDataType, F32> && is_same_v<YDataType, F32> &&
                     is_same_v<SaveMeanInvStdDataType, F32>)
        {
            if constexpr(Rank == 2 && NumReduceDim == 1)
            {
                add_device_normalization_fwd_rank_2_1_f32_instances(op_ptrs);
            }
            else if constexpr(Rank == 4 && NumReduceDim == 3)
            {
                add_device_normalization_fwd_rank_4_3_f32_instances(op_ptrs);
            }
            else if constexpr(Rank == 5 && NumReduceDim == 3)
            {
                add_device_normalization_fwd_rank_5_3_f32_instances(op_ptrs);
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
