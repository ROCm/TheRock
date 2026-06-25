// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <memory>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename InDataType,
          typename DsDataType,
          typename AccDataType,
          typename OutDataType,
          index_t Rank,
          index_t NumReduceDim,
          typename ReduceOperation,
          typename InElementwiseOperation,
          typename OutElementwiseOperation>
struct DeviceReduceMultiD : public BaseOperator
{
    static constexpr index_t NumOutDim = (Rank - NumReduceDim == 0) ? 1 : Rank - NumReduceDim;

    static constexpr index_t NumDTensor = DsDataType::Size();

    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const std::array<index_t, Rank> inLengths,
                        const std::array<index_t, Rank> inStrides,
                        const std::array<std::array<index_t, NumOutDim>, NumDTensor> DsLengths,
                        const std::array<std::array<index_t, NumOutDim>, NumDTensor> DsStrides,
                        const std::array<index_t, NumOutDim> outLengths,
                        const std::array<index_t, NumOutDim> outStrides,
                        const std::array<int, NumReduceDim> reduceDims,
                        const void* in_dev,
                        const std::array<const void*, NumDTensor> ds_dev,
                        void* out_dev,
                        const InElementwiseOperation in_elementwise_op,
                        const OutElementwiseOperation out_elementwise_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename InDataType,
          typename DsDataType,
          typename AccDataType,
          typename OutDataType,
          index_t Rank,
          index_t NumReduceDim,
          typename ReduceOperation,
          typename InElementwiseOperation,
          typename OutElementwiseOperation>
using DeviceReduceMultiDPtr = std::unique_ptr<DeviceReduceMultiD<InDataType,
                                                                 DsDataType,
                                                                 AccDataType,
                                                                 OutDataType,
                                                                 Rank,
                                                                 NumReduceDim,
                                                                 ReduceOperation,
                                                                 InElementwiseOperation,
                                                                 OutElementwiseOperation>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
