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
template void add_device_reduce_instance_blockwise<F16, F16, F16, 4, 3, ReduceAMax, UnaryAbs, PassThrough, false, false>(std::vector<DeviceReducePtr<F16, F16, F16, 4, 3, ReduceAMax, UnaryAbs, PassThrough, false, false>>&);
template void add_device_reduce_instance_blockwise<F16, F16, F16, 4, 4, ReduceAMax, UnaryAbs, PassThrough, false, false>(std::vector<DeviceReducePtr<F16, F16, F16, 4, 4, ReduceAMax, UnaryAbs, PassThrough, false, false>>&);
template void add_device_reduce_instance_blockwise<F16, F16, F16, 4, 1, ReduceAMax, UnaryAbs, PassThrough, false, false>(std::vector<DeviceReducePtr<F16, F16, F16, 4, 1, ReduceAMax, UnaryAbs, PassThrough, false, false>>&);
template void add_device_reduce_instance_blockwise<F16, F16, F16, 2, 1, ReduceAMax, UnaryAbs, PassThrough, false, false>(std::vector<DeviceReducePtr<F16, F16, F16, 2, 1, ReduceAMax, UnaryAbs, PassThrough, false, false>>&);
template void add_device_reduce_instance_blockwise<F16, F16, F16, 4, 3, ReduceAMax, UnaryAbs, PassThrough, false, true>(std::vector<DeviceReducePtr<F16, F16, F16, 4, 3, ReduceAMax, UnaryAbs, PassThrough, false, true>>&);
template void add_device_reduce_instance_blockwise<F16, F16, F16, 4, 4, ReduceAMax, UnaryAbs, PassThrough, false, true>(std::vector<DeviceReducePtr<F16, F16, F16, 4, 4, ReduceAMax, UnaryAbs, PassThrough, false, true>>&);
template void add_device_reduce_instance_blockwise<F16, F16, F16, 4, 1, ReduceAMax, UnaryAbs, PassThrough, false, true>(std::vector<DeviceReducePtr<F16, F16, F16, 4, 1, ReduceAMax, UnaryAbs, PassThrough, false, true>>&);
template void add_device_reduce_instance_blockwise<F16, F16, F16, 2, 1, ReduceAMax, UnaryAbs, PassThrough, false, true>(std::vector<DeviceReducePtr<F16, F16, F16, 2, 1, ReduceAMax, UnaryAbs, PassThrough, false, true>>&);
// clang-format on

} // namespace instance
} // namespace device
} // namespace tensor_operation

} // namespace ck
