// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <memory>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_conv_fwd.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// conv2d forward
#ifdef CK_ENABLE_FP16
void add_device_conv2d_fwd_xdl_c_shuffle_nhwc_kyxc_nhwk_f16_instances(
    std::vector<std::unique_ptr<
        DeviceConvFwd<2, NHWC, KYXC, NHWK, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);
void add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f16_instances(
    std::vector<std::unique_ptr<
        DeviceConvFwd<2, NHWC, KYXC, NHWK, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);
#endif
#ifdef CK_ENABLE_BF16
void add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_bf16_instances(
    std::vector<std::unique_ptr<DeviceConvFwd<2,
                                              NHWC,
                                              KYXC,
                                              NHWK,
                                              BF16,
                                              BF16,
                                              BF16,
                                              PassThrough,
                                              PassThrough,
                                              PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP32
void add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f32_instances(
    std::vector<std::unique_ptr<
        DeviceConvFwd<2, NHWC, KYXC, NHWK, F32, F32, F32, PassThrough, PassThrough, PassThrough>>>&
        instances);
#endif
#ifdef CK_ENABLE_INT8
void add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_int8_instances(
    std::vector<std::unique_ptr<DeviceConvFwd<2,
                                              NHWC,
                                              KYXC,
                                              NHWK,
                                              int8_t,
                                              int8_t,
                                              int8_t,
                                              PassThrough,
                                              PassThrough,
                                              PassThrough>>>& instances);
#endif

template <ck::index_t NumDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceConvFwd<NumDimSpatial,
                                                InLayout,
                                                WeiLayout,
                                                OutLayout,
                                                InDataType,
                                                WeiDataType,
                                                OutDataType,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough>>
{
    using DeviceOp = DeviceConvFwd<NumDimSpatial,
                                   InLayout,
                                   WeiLayout,
                                   OutLayout,
                                   InDataType,
                                   WeiDataType,
                                   OutDataType,
                                   ck::tensor_operation::element_wise::PassThrough,
                                   ck::tensor_operation::element_wise::PassThrough,
                                   ck::tensor_operation::element_wise::PassThrough>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(NumDimSpatial == 2 && is_same_v<InLayout, NHWC> &&
                     is_same_v<WeiLayout, KYXC> && is_same_v<OutLayout, NHWK>)
        {
#ifdef CK_ENABLE_FP32
            if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                         is_same_v<OutDataType, float>)
            {
                add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f32_instances(op_ptrs);
            }
#endif
#ifdef CK_ENABLE_FP16
            if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                         is_same_v<OutDataType, half_t>)
            {
                add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f16_instances(op_ptrs);
                add_device_conv2d_fwd_xdl_c_shuffle_nhwc_kyxc_nhwk_f16_instances(op_ptrs);
            }
#endif
#ifdef CK_ENABLE_BF16
            if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
                         is_same_v<WeiDataType, ck::bhalf_t> && is_same_v<OutDataType, ck::bhalf_t>)
            {
                add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_bf16_instances(op_ptrs);
            }
#endif
#ifdef CK_ENABLE_INT8
            if constexpr(is_same_v<InDataType, int8_t> && is_same_v<WeiDataType, int8_t> &&
                         is_same_v<OutDataType, int8_t>)
            {
                add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_int8_instances(op_ptrs);
            }
#endif
        }

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
