// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename InDataType,
          typename AccDataType,
          typename OutDataType,
          typename InElementwiseOp,
          typename AccElementwiseOp,
          index_t Rank,
          index_t NumReduceDim>
struct DeviceSoftmax : public BaseOperator
{
    //
    // @brief      Makes a pointer to Argument class.
    //
    // @param[in]  inLengths           Input tensor extent(s) from high to low dimension
    // @param[in]  inStrides           Input tensor stride(s) from high to low dimension
    // @param[in]  reduceDims          The dimension(s) the normalization operation is applied
    // @param[in]  alpha               double type value
    // @param[in]  beta                double type value
    // @param[in]  in_dev              Typeless const pointer in device memory storing the input
    //                                 tensor
    // @param      out_dev             Typeless pointer in device memory storing the output tensor
    // @param[in]  in_elementwise_op   The input elementwise operation.
    // @param[in]  acc_elementwise_op  The accumulation elementwise operation.
    //
    // @return     Unique pointer to the Argument class.
    //
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const std::vector<index_t> inLengths,
                        const std::vector<index_t> inStrides,
                        const std::vector<int> reduceDims,
                        double alpha,
                        double beta,
                        const void* in_dev,
                        void* out_dev,
                        InElementwiseOp in_elementwise_op,
                        AccElementwiseOp acc_elementwise_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename InDataType,
          typename AccDataType,
          typename OutDataType,
          typename InElementwiseOp,
          typename AccElementwiseOp,
          index_t Rank,
          index_t NumReduceDim>
using DeviceSoftmaxPtr = std::unique_ptr<DeviceSoftmax<InDataType,
                                                       AccDataType,
                                                       OutDataType,
                                                       InElementwiseOp,
                                                       AccElementwiseOp,
                                                       Rank,
                                                       NumReduceDim>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
