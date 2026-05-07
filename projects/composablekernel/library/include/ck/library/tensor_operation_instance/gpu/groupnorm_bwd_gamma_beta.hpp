// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <memory>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_normalization_bwd_gamma_beta.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
#ifdef CK_ENABLE_FP32
// FP32
void add_device_groupnorm_bwd_gamma_beta_f32_instances(
    std::vector<std::unique_ptr<DeviceNormalizationBwdGammaBeta<F32, F32, F32, F32, F32, 5, 3>>>&);
#endif
template <typename DYDataType,
          typename XDataType,
          typename MeanInvStdDataType,
          typename DGammaDataType,
          typename DBetaDataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceNormalizationBwdGammaBeta<DYDataType,
                                                                  XDataType,
                                                                  MeanInvStdDataType,
                                                                  DGammaDataType,
                                                                  DBetaDataType,
                                                                  5,
                                                                  3>>
{
    using DeviceOp = DeviceNormalizationBwdGammaBeta<DYDataType,
                                                     XDataType,
                                                     MeanInvStdDataType,
                                                     DGammaDataType,
                                                     DBetaDataType,
                                                     5,
                                                     3>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

#ifdef CK_ENABLE_FP32
        if constexpr(is_same_v<DYDataType, F32> && is_same_v<XDataType, F32> &&
                     is_same_v<MeanInvStdDataType, F32> && is_same_v<DGammaDataType, F32> &&
                     is_same_v<DBetaDataType, F32>)
        {
            add_device_groupnorm_bwd_gamma_beta_f32_instances(op_ptrs);
        }
#endif
        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
