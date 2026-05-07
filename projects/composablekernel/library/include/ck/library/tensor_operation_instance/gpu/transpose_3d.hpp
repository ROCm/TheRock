// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <memory>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/device_elementwise.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F16 = ck::half_t;
using F32 = float;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

void add_device_transpose_f16_instances(
    std::vector<std::unique_ptr<DeviceElementwise<ck::Tuple<F16>, ck::Tuple<F16>, PassThrough, 5>>>&
        instances);

void add_device_transpose_f32_instances(
    std::vector<std::unique_ptr<DeviceElementwise<ck::Tuple<F32>, ck::Tuple<F32>, PassThrough, 5>>>&
        instances);

template <typename InDataTypeTuple,
          typename OutDataTypeTuple,
          typename ElementwiseOperation,
          index_t NumDim>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::
        DeviceElementwise<InDataTypeTuple, OutDataTypeTuple, ElementwiseOperation, NumDim>>
{
    using DeviceOp =
        DeviceElementwise<InDataTypeTuple, OutDataTypeTuple, ElementwiseOperation, NumDim>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;
        if constexpr(is_same_v<InDataTypeTuple, ck::Tuple<F32>> &&
                     is_same_v<OutDataTypeTuple, ck::Tuple<F32>>)
        {
            add_device_transpose_f32_instances(op_ptrs);
        }
        else if constexpr(is_same_v<InDataTypeTuple, ck::Tuple<F16>> &&
                          is_same_v<OutDataTypeTuple, ck::Tuple<F16>>)
        {
            add_device_transpose_f16_instances(op_ptrs);
        }
        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
