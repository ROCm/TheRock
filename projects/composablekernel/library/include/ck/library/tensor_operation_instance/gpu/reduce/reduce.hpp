// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"
#include "ck/tensor_operation/gpu/device/device_reduce.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/library/tensor_operation_instance/gpu/reduce/device_reduce_instance.hpp"
#include "ck/utility/reduction_operator.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

template <typename InDataType,
          typename AccDataType,
          typename OutDataType,
          index_t Rank,
          index_t NumReduceDim,
          typename ReduceOperation,
          typename InElementwiseOp,
          typename AccElementwiseOp,
          bool PropagateNan,
          bool OutputIndex>
struct DeviceOperationInstanceFactory<ck::tensor_operation::device::DeviceReduce<InDataType,
                                                                                 AccDataType,
                                                                                 OutDataType,
                                                                                 Rank,
                                                                                 NumReduceDim,
                                                                                 ReduceOperation,
                                                                                 InElementwiseOp,
                                                                                 AccElementwiseOp,
                                                                                 PropagateNan,
                                                                                 OutputIndex>>
{
    using DeviceOp = DeviceReduce<InDataType,
                                  AccDataType,
                                  OutDataType,
                                  Rank,
                                  NumReduceDim,
                                  ReduceOperation,
                                  InElementwiseOp,
                                  AccElementwiseOp,
                                  PropagateNan,
                                  OutputIndex>;

    using DeviceOpPtr = DeviceReducePtr<InDataType,
                                        AccDataType,
                                        OutDataType,
                                        Rank,
                                        NumReduceDim,
                                        ReduceOperation,
                                        InElementwiseOp,
                                        AccElementwiseOp,
                                        PropagateNan,
                                        OutputIndex>;

    static auto GetInstances()
    {
        std::vector<DeviceOpPtr> op_ptrs;

        constexpr bool out_support_atomic_add =
            ck::reduce::InMemoryDataOperationSupportedOnDataType<
                InMemoryDataOperationEnum::AtomicAdd,
                OutDataType>::value;
        constexpr bool op_support_atomic_add =
            std::is_same<ReduceOperation, ReduceAdd>::value &&
            (std::is_same<AccElementwiseOp, PassThrough>::value ||
             std::is_same<AccElementwiseOp, UnaryDivide>::value);
        constexpr bool use_atomic_add = (out_support_atomic_add && op_support_atomic_add);

        add_device_reduce_instance_threadwise<InDataType,
                                              AccDataType,
                                              OutDataType,
                                              Rank,
                                              NumReduceDim,
                                              ReduceOperation,
                                              InElementwiseOp,
                                              AccElementwiseOp,
                                              PropagateNan,
                                              OutputIndex>(op_ptrs);

        add_device_reduce_instance_blockwise<InDataType,
                                             AccDataType,
                                             OutDataType,
                                             Rank,
                                             NumReduceDim,
                                             ReduceOperation,
                                             InElementwiseOp,
                                             AccElementwiseOp,
                                             PropagateNan,
                                             OutputIndex>(op_ptrs);

        if constexpr(use_atomic_add)
        {
            add_device_reduce_instance_multiblock_atomic_add<InDataType,
                                                             AccDataType,
                                                             OutDataType,
                                                             Rank,
                                                             NumReduceDim,
                                                             ReduceOperation,
                                                             InElementwiseOp,
                                                             AccElementwiseOp,
                                                             PropagateNan,
                                                             OutputIndex>(op_ptrs);
        };

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
