// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>
#include <vector>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"
#include "ck/tensor_operation/gpu/device/device_softmax.hpp"
#include "ck/library/tensor_operation_instance/gpu/softmax/device_softmax_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

template <typename InDataType,
          typename AccDataType,
          typename OutDataType,
          index_t Rank,
          index_t NumReduceDim>
struct DeviceOperationInstanceFactory<ck::tensor_operation::device::DeviceSoftmax<InDataType,
                                                                                  AccDataType,
                                                                                  OutDataType,
                                                                                  PassThrough,
                                                                                  PassThrough,
                                                                                  Rank,
                                                                                  NumReduceDim>>
{
    using DeviceOp = DeviceSoftmax<InDataType,
                                   AccDataType,
                                   OutDataType,
                                   PassThrough,
                                   PassThrough,
                                   Rank,
                                   NumReduceDim>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;
#ifdef CK_ENABLE_FP16
        if constexpr(std::is_same_v<InDataType, F16> && std::is_same_v<AccDataType, F32> &&
                     std::is_same_v<OutDataType, F16>)
        {
            if constexpr(Rank == 3)
            {
                if constexpr(NumReduceDim == 1)
                    add_device_softmax_f16_f16_rank3_reduce1_instances(op_ptrs);
                else if constexpr(NumReduceDim == 2)
                    add_device_softmax_f16_f16_rank3_reduce2_instances(op_ptrs);
                else if constexpr(NumReduceDim == 3)
                    add_device_softmax_f16_f16_rank3_reduce3_instances(op_ptrs);
            }
            else if constexpr(Rank == 4)
            {
                if constexpr(NumReduceDim == 1)
                    add_device_softmax_f16_f16_rank4_reduce1_instances(op_ptrs);
                else if constexpr(NumReduceDim == 2)
                    add_device_softmax_f16_f16_rank4_reduce2_instances(op_ptrs);
                else if constexpr(NumReduceDim == 3)
                    add_device_softmax_f16_f16_rank4_reduce3_instances(op_ptrs);
                else if constexpr(NumReduceDim == 4)
                    add_device_softmax_f16_f16_rank4_reduce4_instances(op_ptrs);
            }
        }
#endif
#ifdef CK_ENABLE_FP32
        if constexpr(std::is_same_v<InDataType, F32> && std::is_same_v<AccDataType, F32> &&
                     std::is_same_v<OutDataType, F32>)
        {
            if constexpr(Rank == 3)
            {
                if constexpr(NumReduceDim == 1)
                    add_device_softmax_f32_f32_rank3_reduce1_instances(op_ptrs);
                else if constexpr(NumReduceDim == 2)
                    add_device_softmax_f32_f32_rank3_reduce2_instances(op_ptrs);
                else if constexpr(NumReduceDim == 3)
                    add_device_softmax_f32_f32_rank3_reduce3_instances(op_ptrs);
            }
            else if constexpr(Rank == 4)
            {
                if constexpr(NumReduceDim == 1)
                    add_device_softmax_f32_f32_rank4_reduce1_instances(op_ptrs);
                else if constexpr(NumReduceDim == 2)
                    add_device_softmax_f32_f32_rank4_reduce2_instances(op_ptrs);
                else if constexpr(NumReduceDim == 3)
                    add_device_softmax_f32_f32_rank4_reduce3_instances(op_ptrs);
                else if constexpr(NumReduceDim == 4)
                    add_device_softmax_f32_f32_rank4_reduce4_instances(op_ptrs);
            }
        }
#endif
        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
