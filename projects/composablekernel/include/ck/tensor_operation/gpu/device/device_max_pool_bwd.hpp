// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>

#include "ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// For pooling which used indexable operation, such as MaxPool, MinPool...etc
template <typename DOutDataType, typename IndexDataType, typename DInDataType>
struct DeviceMaxPoolBwd : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_dout,
                        const void* p_indices,
                        void* p_din,
                        index_t dout_length,
                        index_t din_length,
                        std::vector<ck::index_t> window_lengths,
                        std::vector<ck::index_t> window_strides,
                        std::vector<ck::index_t> window_dilations) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
