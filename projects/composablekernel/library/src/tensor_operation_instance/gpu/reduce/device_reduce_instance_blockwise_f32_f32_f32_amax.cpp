// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/utility/reduction_enums.hpp"
#include "ck/library/tensor_operation_instance/gpu/reduce/device_reduce_instance_blockwise.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// clang-format off
//                                                 InDataType | AccDataType | OutDataType | Rank | NumReduceDim | ReduceOperation | InElementwiseOp | AccElementwiseOp | PropagateNan | UseIndex 
template void add_device_reduce_instance_blockwise<        F32,          F32,          F32,     4,             3,       ReduceAMax,         UnaryAbs,       PassThrough,        false,     false>(std::vector<DeviceReducePtr<F32, F32, F32, 4, 3, ReduceAMax, UnaryAbs, PassThrough, false, false>>&);
template void add_device_reduce_instance_blockwise<        F32,          F32,          F32,     4,             4,       ReduceAMax,         UnaryAbs,       PassThrough,        false,     false>(std::vector<DeviceReducePtr<F32, F32, F32, 4, 4, ReduceAMax, UnaryAbs, PassThrough, false, false>>&);
template void add_device_reduce_instance_blockwise<        F32,          F32,          F32,     4,             1,       ReduceAMax,         UnaryAbs,       PassThrough,        false,     false>(std::vector<DeviceReducePtr<F32, F32, F32, 4, 1, ReduceAMax, UnaryAbs, PassThrough, false, false>>&);
template void add_device_reduce_instance_blockwise<        F32,          F32,          F32,     2,             1,       ReduceAMax,         UnaryAbs,       PassThrough,        false,     false>(std::vector<DeviceReducePtr<F32, F32, F32, 2, 1, ReduceAMax, UnaryAbs, PassThrough, false, false>>&);
template void add_device_reduce_instance_blockwise<        F32,          F32,          F32,     4,             3,       ReduceAMax,         UnaryAbs,       PassThrough,        false,      true>(std::vector<DeviceReducePtr<F32, F32, F32, 4, 3, ReduceAMax, UnaryAbs, PassThrough, false, true>>&);
template void add_device_reduce_instance_blockwise<        F32,          F32,          F32,     4,             4,       ReduceAMax,         UnaryAbs,       PassThrough,        false,      true>(std::vector<DeviceReducePtr<F32, F32, F32, 4, 4, ReduceAMax, UnaryAbs, PassThrough, false, true>>&);
template void add_device_reduce_instance_blockwise<        F32,          F32,          F32,     4,             1,       ReduceAMax,         UnaryAbs,       PassThrough,        false,      true>(std::vector<DeviceReducePtr<F32, F32, F32, 4, 1, ReduceAMax, UnaryAbs, PassThrough, false, true>>&);
template void add_device_reduce_instance_blockwise<        F32,          F32,          F32,     2,             1,       ReduceAMax,         UnaryAbs,       PassThrough,        false,      true>(std::vector<DeviceReducePtr<F32, F32, F32, 2, 1, ReduceAMax, UnaryAbs, PassThrough, false, true>>&);
template void add_device_reduce_instance_blockwise<        F32,          F32,          F32,     6,             6,       ReduceAMax,         UnaryAbs,       PassThrough,         true,     false>(std::vector<DeviceReducePtr<F32, F32, F32, 6, 6, ReduceAMax, UnaryAbs, PassThrough, true, false>>&);
template void add_device_reduce_instance_blockwise<        F32,          F32,          F32,     5,             5,       ReduceAMax,         UnaryAbs,       PassThrough,         true,     false>(std::vector<DeviceReducePtr<F32, F32, F32, 5, 5, ReduceAMax, UnaryAbs, PassThrough, true, false>>&);
template void add_device_reduce_instance_blockwise<        F32,          F32,          F32,     4,             4,       ReduceAMax,         UnaryAbs,       PassThrough,         true,     false>(std::vector<DeviceReducePtr<F32, F32, F32, 4, 4, ReduceAMax, UnaryAbs, PassThrough, true, false>>&);
template void add_device_reduce_instance_blockwise<        F32,          F32,          F32,     6,             3,       ReduceAMax,         UnaryAbs,       PassThrough,         true,     false>(std::vector<DeviceReducePtr<F32, F32, F32, 6, 3, ReduceAMax, UnaryAbs, PassThrough, true, false>>&);
template void add_device_reduce_instance_blockwise<        F32,          F32,          F32,     5,             3,       ReduceAMax,         UnaryAbs,       PassThrough,         true,     false>(std::vector<DeviceReducePtr<F32, F32, F32, 5, 3, ReduceAMax, UnaryAbs, PassThrough, true, false>>&);
template void add_device_reduce_instance_blockwise<        F32,          F32,          F32,     4,             3,       ReduceAMax,         UnaryAbs,       PassThrough,         true,     false>(std::vector<DeviceReducePtr<F32, F32, F32, 4, 3, ReduceAMax, UnaryAbs, PassThrough, true, false>>&);
template void add_device_reduce_instance_blockwise<        F32,          F32,          F32,     3,             3,       ReduceAMax,      PassThrough,       PassThrough,         true,     false>(std::vector<DeviceReducePtr<F32, F32, F32, 3, 3, ReduceAMax, PassThrough, PassThrough, true, false>>&);
template void add_device_reduce_instance_blockwise<        F32,          F32,          F32,     2,             2,       ReduceAMax,      PassThrough,       PassThrough,         true,     false>(std::vector<DeviceReducePtr<F32, F32, F32, 2, 2, ReduceAMax, PassThrough, PassThrough, true, false>>&);
template void add_device_reduce_instance_blockwise<        F32,          F32,          F32,     1,             1,       ReduceAMax,      PassThrough,       PassThrough,         true,     false>(std::vector<DeviceReducePtr<F32, F32, F32, 1, 1, ReduceAMax, PassThrough, PassThrough, true, false>>&);
// clang-format on

} // namespace instance
} // namespace device
} // namespace tensor_operation

} // namespace ck
