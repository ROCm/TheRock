// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <memory>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_normalization_bwd_data.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
#ifdef CK_ENABLE_FP16
// FP16
void add_device_layernorm2d_bwd_data_f16_instances(
    std::vector<std::unique_ptr<DeviceNormalizationBwdData<F16, F16, F16, F16, F16, 2, 1>>>&);
#endif
#ifdef CK_ENABLE_FP32
// FP32
void add_device_layernorm2d_bwd_data_f32_instances(
    std::vector<std::unique_ptr<DeviceNormalizationBwdData<F32, F32, F32, F32, F32, 2, 1>>>&);
#endif
template <typename DYDataType,
          typename XDataType,
          typename GammaDataType,
          typename MeanInvStdDataType,
          typename DXDataType,
          index_t Rank,
          index_t NumReduceDim>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceNormalizationBwdData<DYDataType,
                                                             XDataType,
                                                             GammaDataType,
                                                             MeanInvStdDataType,
                                                             DXDataType,
                                                             Rank,
                                                             NumReduceDim>>
{
    using DeviceOp = DeviceNormalizationBwdData<DYDataType,
                                                XDataType,
                                                GammaDataType,
                                                MeanInvStdDataType,
                                                DXDataType,
                                                Rank,
                                                NumReduceDim>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;
#ifdef CK_ENABLE_FP16
        if constexpr(is_same_v<DYDataType, F16> && is_same_v<XDataType, F16> &&
                     is_same_v<GammaDataType, F16> && is_same_v<MeanInvStdDataType, F16> &&
                     is_same_v<DXDataType, F16>)
        {
            if constexpr(Rank == 2 && NumReduceDim == 1)
            {
                add_device_layernorm2d_bwd_data_f16_instances(op_ptrs);
            }
        }
#endif
#ifdef CK_ENABLE_FP32
        if constexpr(is_same_v<DYDataType, F32> && is_same_v<XDataType, F32> &&
                     is_same_v<GammaDataType, F32> && is_same_v<MeanInvStdDataType, F32> &&
                     is_same_v<DXDataType, F32>)
        {
            if constexpr(Rank == 2 && NumReduceDim == 1)
            {
                add_device_layernorm2d_bwd_data_f32_instances(op_ptrs);
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
