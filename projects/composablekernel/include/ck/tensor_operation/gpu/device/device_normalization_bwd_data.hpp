// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <vector>

#include "ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
template <typename DYDataType,
          typename XDataType,
          typename GammaDataType,
          typename MeanInvStdDataType,
          typename DXDataType,
          index_t Rank,
          index_t NumReduceDim>
struct DeviceNormalizationBwdData : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const std::vector<index_t> lengths,
                        const std::vector<index_t> dyStrides,
                        const std::vector<index_t> xStrides,
                        const std::vector<index_t> gammaStrides,
                        const std::vector<index_t> meanStrides,
                        const std::vector<index_t> invStdStrides,
                        const std::vector<index_t> dxStrides,
                        const std::vector<index_t> reduceDims,
                        const void* p_dy,
                        const void* p_x,
                        const void* p_gamma,
                        const void* p_mean,
                        const void* p_invStd,
                        void* p_dx) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename DYDataType,
          typename XDataType,
          typename GammaDataType,
          typename MeanInvStdDataType,
          typename DXDataType,
          index_t Rank,
          index_t NumReduceDim>
using DeviceNormalizationBwdDataPtr = std::unique_ptr<DeviceNormalizationBwdData<DYDataType,
                                                                                 XDataType,
                                                                                 GammaDataType,
                                                                                 MeanInvStdDataType,
                                                                                 DXDataType,
                                                                                 Rank,
                                                                                 NumReduceDim>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
