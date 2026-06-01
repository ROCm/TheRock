// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"

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
          typename CElementwiseOperation>
struct DeviceGemm_Streamk_V2 : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument> MakeArgumentPointer(
        const void* p_a,
        const void* p_b,
        void* p_c,
        ck::index_t M,
        ck::index_t N,
        ck::index_t K,
        ck::index_t StrideA,
        ck::index_t StrideB,
        ck::index_t StrideC,
        ck::index_t Streamk_sel,
        ck::index_t Grid_size,
        AElementwiseOperation a_element_op,
        BElementwiseOperation b_element_op,
        CElementwiseOperation c_element_op,
        StreamKReductionStrategy reduction_strategy = StreamKReductionStrategy::Atomic) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
