// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/utility/reduction_enums.hpp"
#include "ck/library/tensor_operation_instance/gpu/reduce/device_reduce_instance_blockwise.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// clang-format off
// InDataType | AccDataType | OutDataType | Rank | NumReduceDim | ReduceOperation | InElementwiseOp | AccElementwiseOp | PropagateNan | UseIndex 
template void add_device_reduce_instance_blockwise<F32, F32, F32, 4, 3, ReduceAdd, UnarySquare, UnarySqrt, false, false>(std::vector<DeviceReducePtr<F32, F32, F32, 4, 3, ReduceAdd, UnarySquare, UnarySqrt, false, false>>&); 
template void add_device_reduce_instance_blockwise<F32, F32, F32, 4, 4, ReduceAdd, UnarySquare, UnarySqrt, false, false>(std::vector<DeviceReducePtr<F32, F32, F32, 4, 4, ReduceAdd, UnarySquare, UnarySqrt, false, false>>&); 
template void add_device_reduce_instance_blockwise<F32, F32, F32, 4, 1, ReduceAdd, UnarySquare, UnarySqrt, false, false>(std::vector<DeviceReducePtr<F32, F32, F32, 4, 1, ReduceAdd, UnarySquare, UnarySqrt, false, false>>&); 
template void add_device_reduce_instance_blockwise<F32, F32, F32, 2, 1, ReduceAdd, UnarySquare, UnarySqrt, false, false>(std::vector<DeviceReducePtr<F32, F32, F32, 2, 1, ReduceAdd, UnarySquare, UnarySqrt, false, false>>&);
// clang-format on
// clang-format on

} // namespace instance
} // namespace device
} // namespace tensor_operation

} // namespace ck
