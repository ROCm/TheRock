// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <vector>

#include "device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename ComputeType = CDataType>
struct DeviceGemmSplitK : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                              const void* p_b,
                                                              void* p_c,
                                                              ck::index_t M,
                                                              ck::index_t N,
                                                              ck::index_t K,
                                                              ck::index_t StrideA,
                                                              ck::index_t StrideB,
                                                              ck::index_t StrideC,
                                                              AElementwiseOperation a_element_op,
                                                              BElementwiseOperation b_element_op,
                                                              CElementwiseOperation c_element_op,
                                                              ck::index_t KBatch) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename ComputeType = CDataType>
using DeviceGemmSplitKPtr = std::unique_ptr<DeviceGemmSplitK<ALayout,
                                                             BLayout,
                                                             CLayout,
                                                             ADataType,
                                                             BDataType,
                                                             CDataType,
                                                             AElementwiseOperation,
                                                             BElementwiseOperation,
                                                             CElementwiseOperation,
                                                             ComputeType>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
