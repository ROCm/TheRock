// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/tensor_operation/gpu/device/device_max_pool_bwd.hpp"
#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

#ifdef CK_ENABLE_FP16
void add_device_maxpool_bwd_f16_instances(
    std::vector<std::unique_ptr<DeviceMaxPoolBwd<F16, I32, F16>>>&);
#endif
#ifdef CK_ENABLE_BF16
void add_device_maxpool_bwd_bf16_instances(
    std::vector<std::unique_ptr<DeviceMaxPoolBwd<BF16, I32, BF16>>>&);
#endif
#ifdef CK_ENABLE_FP32
void add_device_maxpool_bwd_f32_instances(
    std::vector<std::unique_ptr<DeviceMaxPoolBwd<F32, I32, F32>>>&);
#endif
#ifdef CK_ENABLE_FP8
void add_device_maxpool_bwd_f8_instances(
    std::vector<std::unique_ptr<DeviceMaxPoolBwd<F8, I32, F8>>>&);
#endif
#ifdef CK_ENABLE_INT8
void add_device_maxpool_bwd_int8_instances(
    std::vector<std::unique_ptr<DeviceMaxPoolBwd<I8, I32, I8>>>&);
#endif

template <typename DOutDataType, typename IndexDataType, typename DInDataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceMaxPoolBwd<DOutDataType, IndexDataType, DInDataType>>
{
    using DeviceOp = DeviceMaxPoolBwd<DOutDataType, IndexDataType, DInDataType>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

#ifdef CK_ENABLE_FP16
        if constexpr(is_same_v<DOutDataType, F16> && is_same_v<DInDataType, F16> &&
                     is_same_v<IndexDataType, I32>)
            add_device_maxpool_bwd_f16_instances(op_ptrs);
#endif
#ifdef CK_ENABLE_BF16
        else if constexpr(is_same_v<DOutDataType, BF16> && is_same_v<DInDataType, BF16> &&
                          is_same_v<IndexDataType, I32>)
            add_device_maxpool_bwd_bf16_instances(op_ptrs);
#endif
#ifdef CK_ENABLE_FP32
        else if constexpr(is_same_v<DOutDataType, F32> && is_same_v<DInDataType, F32> &&
                          is_same_v<IndexDataType, I32>)
            add_device_maxpool_bwd_f32_instances(op_ptrs);
#endif
#ifdef CK_ENABLE_FP8
        else if constexpr(is_same_v<DOutDataType, F8> && is_same_v<DInDataType, F8> &&
                          is_same_v<IndexDataType, I32>)
            add_device_maxpool_bwd_f8_instances(op_ptrs);
#endif
#ifdef CK_ENABLE_INT8
        else if constexpr(is_same_v<DOutDataType, I8> && is_same_v<DInDataType, I8> &&
                          is_same_v<IndexDataType, I32>)
            add_device_maxpool_bwd_int8_instances(op_ptrs);
#endif

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
