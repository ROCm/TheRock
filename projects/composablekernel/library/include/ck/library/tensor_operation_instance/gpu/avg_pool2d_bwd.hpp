// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/tensor_operation/gpu/device/device_avgpool_bwd.hpp"
#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

#ifdef CK_ENABLE_BF16
void add_device_avgpool_2D_bwd_nhwc_bf16_instances(
    std::vector<std::unique_ptr<DeviceAvgPoolBwd<2, BF16, BF16, NHWC, NHWC>>>&);
#endif

#ifdef CK_ENABLE_FP16
void add_device_avgpool_2D_bwd_nhwc_f16_instances(
    std::vector<std::unique_ptr<DeviceAvgPoolBwd<2, F16, F16, NHWC, NHWC>>>&);
#endif

#ifdef CK_ENABLE_FP8
void add_device_avgpool_2D_bwd_nhwc_f8_instances(
    std::vector<std::unique_ptr<DeviceAvgPoolBwd<2, F8, F8, NHWC, NHWC>>>&);
#endif

#ifdef CK_ENABLE_FP32
void add_device_avgpool_2D_bwd_nhwc_f32_instances(
    std::vector<std::unique_ptr<DeviceAvgPoolBwd<2, F32, F32, NHWC, NHWC>>>&);
#endif

#ifdef CK_ENABLE_INT8
void add_device_avgpool_2D_bwd_nhwc_int8_instances(
    std::vector<std::unique_ptr<DeviceAvgPoolBwd<2, I8, I8, NHWC, NHWC>>>&);
#endif

template <typename DOutDataType, typename DInDataType, typename InLayout, typename OutLayout>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::
        DeviceAvgPoolBwd<2, DOutDataType, DInDataType, InLayout, OutLayout>>
{
    using DeviceOp = DeviceAvgPoolBwd<2, DOutDataType, DInDataType, InLayout, OutLayout>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;
        if constexpr(is_same_v<InLayout, NHWC> && is_same_v<OutLayout, NHWC>)
        {
#ifdef CK_ENABLE_FP16
            if constexpr(is_same_v<DOutDataType, F16> && is_same_v<DInDataType, F16>)
                add_device_avgpool_2D_bwd_nhwc_f16_instances(op_ptrs);
#endif
#ifdef CK_ENABLE_BF16
            else if constexpr(is_same_v<DOutDataType, BF16> && is_same_v<DInDataType, BF16>)
                add_device_avgpool_2D_bwd_nhwc_bf16_instances(op_ptrs);
#endif
#ifdef CK_ENABLE_FP32
            else if constexpr(is_same_v<DOutDataType, F32> && is_same_v<DInDataType, F32>)
                add_device_avgpool_2D_bwd_nhwc_f32_instances(op_ptrs);
#endif
#ifdef CK_ENABLE_FP8
            else if constexpr(is_same_v<DOutDataType, F8> && is_same_v<DInDataType, F8>)
                add_device_avgpool_2D_bwd_nhwc_f8_instances(op_ptrs);
#endif
#ifdef CK_ENABLE_INT8
            else if constexpr(is_same_v<DOutDataType, I8> && is_same_v<DInDataType, I8>)
                add_device_avgpool_2D_bwd_nhwc_int8_instances(op_ptrs);
#endif
        }

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
