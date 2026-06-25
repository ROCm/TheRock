// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_pool_fwd.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

static constexpr auto InOutRank  = 5;
static constexpr auto WindowRank = 3;

static constexpr auto MaxOp = ck::ReduceTensorOp::MAX;
static constexpr auto AvgOp = ck::ReduceTensorOp::AVG;

// FP16
void add_device_pool3d_fwd_ndhwc_f16_instances(
    std::vector<std::unique_ptr<
        DevicePoolFwd<InOutRank, WindowRank, F16, F16, I32, NDHWC, NDHWC, MaxOp, false>>>&);

void add_device_pool3d_fwd_ndhwc_f16_instances(
    std::vector<std::unique_ptr<
        DevicePoolFwd<InOutRank, WindowRank, F16, F16, I32, NDHWC, NDHWC, AvgOp, false>>>&);

// FP16 - return index
void add_device_pool3d_fwd_ndhwc_index_f16_instances(
    std::vector<std::unique_ptr<
        DevicePoolFwd<InOutRank, WindowRank, F16, F16, I32, NDHWC, NDHWC, MaxOp, true>>>&);

using F8 = ck::f8_t;
// F8
void add_device_pool3d_fwd_ndhwc_f8_instances(
    std::vector<std::unique_ptr<
        DevicePoolFwd<InOutRank, WindowRank, F8, F8, I32, NDHWC, NDHWC, MaxOp, false>>>&);

void add_device_pool3d_fwd_ndhwc_f8_instances(
    std::vector<std::unique_ptr<
        DevicePoolFwd<InOutRank, WindowRank, F8, F8, I32, NDHWC, NDHWC, AvgOp, false>>>&);

// FP8 - return index
void add_device_pool3d_fwd_ndhwc_index_f8_instances(
    std::vector<std::unique_ptr<
        DevicePoolFwd<InOutRank, WindowRank, F8, F8, I32, NDHWC, NDHWC, MaxOp, true>>>&);

// BF16
void add_device_pool3d_fwd_ndhwc_bf16_instances(
    std::vector<std::unique_ptr<
        DevicePoolFwd<InOutRank, WindowRank, BF16, BF16, I32, NDHWC, NDHWC, MaxOp, false>>>&);

void add_device_pool3d_fwd_ndhwc_bf16_instances(
    std::vector<std::unique_ptr<
        DevicePoolFwd<InOutRank, WindowRank, BF16, BF16, I32, NDHWC, NDHWC, AvgOp, false>>>&);

// BF16 - return index
void add_device_pool3d_fwd_ndhwc_index_bf16_instances(
    std::vector<std::unique_ptr<
        DevicePoolFwd<InOutRank, WindowRank, BF16, BF16, I32, NDHWC, NDHWC, MaxOp, true>>>&);

// FP32
void add_device_pool3d_fwd_ndhwc_f32_instances(
    std::vector<std::unique_ptr<
        DevicePoolFwd<InOutRank, WindowRank, F32, F32, I32, NDHWC, NDHWC, MaxOp, false>>>&);

void add_device_pool3d_fwd_ndhwc_f32_instances(
    std::vector<std::unique_ptr<
        DevicePoolFwd<InOutRank, WindowRank, F32, F32, I32, NDHWC, NDHWC, AvgOp, false>>>&);

// FP32 - return index
void add_device_pool3d_fwd_ndhwc_index_f32_instances(
    std::vector<std::unique_ptr<
        DevicePoolFwd<InOutRank, WindowRank, F32, F32, I32, NDHWC, NDHWC, MaxOp, true>>>&);

// I8
void add_device_pool3d_fwd_ndhwc_i8_instances(
    std::vector<std::unique_ptr<
        DevicePoolFwd<InOutRank, WindowRank, I8, I8, I32, NDHWC, NDHWC, MaxOp, false>>>&);

void add_device_pool3d_fwd_ndhwc_i8_instances(
    std::vector<std::unique_ptr<
        DevicePoolFwd<InOutRank, WindowRank, I8, I8, I32, NDHWC, NDHWC, AvgOp, false>>>&);

// I8 - return index
void add_device_pool3d_fwd_ndhwc_index_i8_instances(
    std::vector<std::unique_ptr<
        DevicePoolFwd<InOutRank, WindowRank, I8, I8, I32, NDHWC, NDHWC, MaxOp, true>>>&);

template <typename InDataType,
          typename OutDataType,
          typename IndexDataType,
          typename InLayout,
          typename OutLayout,
          ck::ReduceTensorOp ReduceOpId,
          bool OutputIndex>
struct DeviceOperationInstanceFactory<ck::tensor_operation::device::DevicePoolFwd<InOutRank,
                                                                                  WindowRank,
                                                                                  InDataType,
                                                                                  OutDataType,
                                                                                  IndexDataType,
                                                                                  InLayout,
                                                                                  OutLayout,
                                                                                  ReduceOpId,
                                                                                  OutputIndex>>
{
    using DeviceOp = DevicePoolFwd<InOutRank,
                                   WindowRank,
                                   InDataType,
                                   OutDataType,
                                   IndexDataType,
                                   InLayout,
                                   OutLayout,
                                   ReduceOpId,
                                   OutputIndex>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;
        if constexpr(is_same_v<InLayout, NDHWC> && is_same_v<OutLayout, NDHWC>)
        {
            if constexpr(is_same_v<InDataType, F16> && is_same_v<OutDataType, F16> &&
                         is_same_v<IndexDataType, I32>)
            {
                if constexpr(OutputIndex && ReduceOpId == MaxOp)
                {
                    add_device_pool3d_fwd_ndhwc_index_f16_instances(op_ptrs);
                }
                else
                {
                    add_device_pool3d_fwd_ndhwc_f16_instances(op_ptrs);
                }
            }
            else if constexpr(is_same_v<InDataType, BF16> && is_same_v<OutDataType, BF16> &&
                              is_same_v<IndexDataType, I32>)
            {
                if constexpr(OutputIndex && ReduceOpId == MaxOp)
                {
                    add_device_pool3d_fwd_ndhwc_index_bf16_instances(op_ptrs);
                }
                else
                {
                    add_device_pool3d_fwd_ndhwc_bf16_instances(op_ptrs);
                }
            }
            else if constexpr(is_same_v<InDataType, F32> && is_same_v<OutDataType, F32> &&
                              is_same_v<IndexDataType, I32>)
            {
                if constexpr(OutputIndex && ReduceOpId == MaxOp)
                {
                    add_device_pool3d_fwd_ndhwc_index_f32_instances(op_ptrs);
                }
                else
                {
                    add_device_pool3d_fwd_ndhwc_f32_instances(op_ptrs);
                }
            }
            else if constexpr(is_same_v<InDataType, F8> && is_same_v<OutDataType, F8> &&
                              is_same_v<IndexDataType, I32>)
            {
                if constexpr(OutputIndex && ReduceOpId == MaxOp)
                {
                    add_device_pool3d_fwd_ndhwc_index_f8_instances(op_ptrs);
                }
                else
                {
                    add_device_pool3d_fwd_ndhwc_f8_instances(op_ptrs);
                }
            }
            else if constexpr(is_same_v<InDataType, I8> && is_same_v<OutDataType, I8> &&
                              is_same_v<IndexDataType, I32>)
            {
                if constexpr(OutputIndex && ReduceOpId == MaxOp)
                {
                    add_device_pool3d_fwd_ndhwc_index_i8_instances(op_ptrs);
                }
                else
                {
                    add_device_pool3d_fwd_ndhwc_i8_instances(op_ptrs);
                }
            }
        }

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
