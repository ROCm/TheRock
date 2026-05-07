// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>

#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/utility/reduction_enums.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// output[indices] = input
template <typename InDataType,
          typename IndexDataType,
          typename OutDataType,
          typename ElementwiseOperation,
          InMemoryDataOperationEnum Op>
struct DevicePutElement : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_input,
                        const void* p_indices,
                        void* p_output,
                        index_t input_length,
                        index_t output_length,
                        ElementwiseOperation elementwise_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
