// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/utility/reduction_enums.hpp"
#include "ck/library/tensor_operation_instance/gpu/reduce/device_reduce_instance_threadwise.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// clang-format off
// InDataType | AccDataType | OutDataType | Rank | NumReduceDim | ReduceOperation | InElementwiseOp | AccElementwiseOp | PropagateNan | UseIndex 
template void add_device_reduce_instance_threadwise<BF16, F32, BF16, 4, 3, ReduceMax, PassThrough, PassThrough, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 4, 3, ReduceMax, PassThrough, PassThrough, false, false>>&);
template void add_device_reduce_instance_threadwise<BF16, F32, BF16, 4, 4, ReduceMax, PassThrough, PassThrough, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 4, 4, ReduceMax, PassThrough, PassThrough, false, false>>&);
template void add_device_reduce_instance_threadwise<BF16, F32, BF16, 4, 1, ReduceMax, PassThrough, PassThrough, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 4, 1, ReduceMax, PassThrough, PassThrough, false, false>>&);
template void add_device_reduce_instance_threadwise<BF16, F32, BF16, 2, 1, ReduceMax, PassThrough, PassThrough, false, false>(std::vector<DeviceReducePtr<BF16, F32, BF16, 2, 1, ReduceMax, PassThrough, PassThrough, false, false>>&);
template void add_device_reduce_instance_threadwise<BF16, F32, BF16, 4, 3, ReduceMax, PassThrough, PassThrough, false, true>(std::vector<DeviceReducePtr<BF16, F32, BF16, 4, 3, ReduceMax, PassThrough, PassThrough, false, true>>&);
template void add_device_reduce_instance_threadwise<BF16, F32, BF16, 4, 4, ReduceMax, PassThrough, PassThrough, false, true>(std::vector<DeviceReducePtr<BF16, F32, BF16, 4, 4, ReduceMax, PassThrough, PassThrough, false, true>>&);
template void add_device_reduce_instance_threadwise<BF16, F32, BF16, 4, 1, ReduceMax, PassThrough, PassThrough, false, true>(std::vector<DeviceReducePtr<BF16, F32, BF16, 4, 1, ReduceMax, PassThrough, PassThrough, false, true>>&);
template void add_device_reduce_instance_threadwise<BF16, F32, BF16, 2, 1, ReduceMax, PassThrough, PassThrough, false, true>(std::vector<DeviceReducePtr<BF16, F32, BF16, 2, 1, ReduceMax, PassThrough, PassThrough, false, true>>&);
// clang-format on

} // namespace instance
} // namespace device
} // namespace tensor_operation

} // namespace ck
