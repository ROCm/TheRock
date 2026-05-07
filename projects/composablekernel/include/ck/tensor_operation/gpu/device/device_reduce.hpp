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
          typename AccDataType,
          typename OutDataType,
          index_t Rank,
          index_t NumReduceDim,
          typename ReduceOperation,
          typename InElementwiseOperation,
          typename AccElementwiseOperation,
          bool PropagateNan,
          bool OutputIndex>
struct DeviceReduce : public BaseOperator
{
    static constexpr index_t NumOutDim = (Rank - NumReduceDim == 0) ? 1 : Rank - NumReduceDim;

    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const std::array<index_t, Rank> inLengths,
                        const std::array<index_t, Rank> inStrides,
                        const std::array<index_t, NumOutDim> outLengths,
                        const std::array<index_t, NumOutDim> outStrides,
                        const std::array<int, NumReduceDim> reduceDims,
                        double alpha,
                        double beta,
                        const void* in_dev,
                        const void* in_index_dev,
                        void* out_dev,
                        void* out_index_dev,
                        const InElementwiseOperation in_elementwise_op,
                        const AccElementwiseOperation acc_elementwise_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename InDataType,
          typename AccDataType,
          typename OutDataType,
          index_t Rank,
          index_t NumReduceDim,
          typename ReduceOperation,
          typename InElementwiseOperation,
          typename AccElementwiseOperation,
          bool PropagateNan,
          bool OutputIndex>
using DeviceReducePtr = std::unique_ptr<DeviceReduce<InDataType,
                                                     AccDataType,
                                                     OutDataType,
                                                     Rank,
                                                     NumReduceDim,
                                                     ReduceOperation,
                                                     InElementwiseOperation,
                                                     AccElementwiseOperation,
                                                     PropagateNan,
                                                     OutputIndex>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
