// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <memory>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_conv_bwd_data.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
#ifdef CK_ENABLE_BF16
// conv1d backward data
void add_device_conv1d_bwd_data_xdl_nwc_kxc_nwk_bf16_instances(
    std::vector<std::unique_ptr<DeviceConvBwdData<1,
                                                  NWC,
                                                  KXC,
                                                  NWK,
                                                  BF16,
                                                  BF16,
                                                  BF16,
                                                  PassThrough,
                                                  PassThrough,
                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP16
void add_device_conv1d_bwd_data_xdl_nwc_kxc_nwk_f16_instances(
    std::vector<std::unique_ptr<
        DeviceConvBwdData<1, NWC, KXC, NWK, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);
#endif
#ifdef CK_ENABLE_FP32
void add_device_conv1d_bwd_data_xdl_nwc_kxc_nwk_f32_instances(
    std::vector<std::unique_ptr<
        DeviceConvBwdData<1, NWC, KXC, NWK, F32, F32, F32, PassThrough, PassThrough, PassThrough>>>&
        instances);
#endif
#ifdef CK_ENABLE_INT8
void add_device_conv1d_bwd_data_xdl_nwc_kxc_nwk_int8_instances(
    std::vector<std::unique_ptr<DeviceConvBwdData<1,
                                                  NWC,
                                                  KXC,
                                                  NWK,
                                                  int8_t,
                                                  int8_t,
                                                  int8_t,
                                                  PassThrough,
                                                  PassThrough,
                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_BF16
// conv2d backward data
void add_device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_bf16_instances(
    std::vector<std::unique_ptr<DeviceConvBwdData<2,
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
#ifdef CK_ENABLE_FP16
void add_device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_f16_instances(
    std::vector<std::unique_ptr<DeviceConvBwdData<2,
                                                  NHWC,
                                                  KYXC,
                                                  NHWK,
                                                  F16,
                                                  F16,
                                                  F16,
                                                  PassThrough,
                                                  PassThrough,
                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP32
void add_device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_f32_instances(
    std::vector<std::unique_ptr<DeviceConvBwdData<2,
                                                  NHWC,
                                                  KYXC,
                                                  NHWK,
                                                  F32,
                                                  F32,
                                                  F32,
                                                  PassThrough,
                                                  PassThrough,
                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_INT8
void add_device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_int8_instances(
    std::vector<std::unique_ptr<DeviceConvBwdData<2,
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
#ifdef DL_KERNELS
#ifdef CK_ENABLE_FP16
// conv2d dl
void add_device_conv2d_bwd_data_dl_nhwc_kyxc_nhwk_f16_instances(
    std::vector<std::unique_ptr<DeviceConvBwdData<2,
                                                  NHWC,
                                                  KYXC,
                                                  NHWK,
                                                  F16,
                                                  F16,
                                                  F16,
                                                  PassThrough,
                                                  PassThrough,
                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP32
void add_device_conv2d_bwd_data_dl_nhwc_kyxc_nhwk_f32_instances(
    std::vector<std::unique_ptr<DeviceConvBwdData<2,
                                                  NHWC,
                                                  KYXC,
                                                  NHWK,
                                                  F32,
                                                  F32,
                                                  F32,
                                                  PassThrough,
                                                  PassThrough,
                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_INT8
void add_device_conv2d_bwd_data_dl_nhwc_kyxc_nhwk_int8_instances(
    std::vector<std::unique_ptr<DeviceConvBwdData<2,
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
#endif
#ifdef CK_ENABLE_BF16
// conv3d backward data
void add_device_conv3d_bwd_data_xdl_ndhwc_kzyxc_ndhwk_bf16_instances(
    std::vector<std::unique_ptr<DeviceConvBwdData<3,
                                                  NDHWC,
                                                  KZYXC,
                                                  NDHWK,
                                                  BF16,
                                                  BF16,
                                                  BF16,
                                                  PassThrough,
                                                  PassThrough,
                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP16
void add_device_conv3d_bwd_data_xdl_ndhwc_kzyxc_ndhwk_f16_instances(
    std::vector<std::unique_ptr<DeviceConvBwdData<3,
                                                  NDHWC,
                                                  KZYXC,
                                                  NDHWK,
                                                  F16,
                                                  F16,
                                                  F16,
                                                  PassThrough,
                                                  PassThrough,
                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP32
void add_device_conv3d_bwd_data_xdl_ndhwc_kzyxc_ndhwk_f32_instances(
    std::vector<std::unique_ptr<DeviceConvBwdData<3,
                                                  NDHWC,
                                                  KZYXC,
                                                  NDHWK,
                                                  F32,
                                                  F32,
                                                  F32,
                                                  PassThrough,
                                                  PassThrough,
                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_INT8
void add_device_conv3d_bwd_data_xdl_ndhwc_kzyxc_ndhwk_int8_instances(
    std::vector<std::unique_ptr<DeviceConvBwdData<3,
                                                  NDHWC,
                                                  KZYXC,
                                                  NDHWK,
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
struct DeviceOperationInstanceFactory<ck::tensor_operation::device::DeviceConvBwdData<
    NumDimSpatial,
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
    using DeviceOp = DeviceConvBwdData<NumDimSpatial,
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

        if constexpr(NumDimSpatial == 1 && is_same_v<InLayout, NWC> && is_same_v<WeiLayout, KXC> &&
                     is_same_v<OutLayout, NWK>)
        {
#ifdef CK_ENABLE_FP32
            if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                         is_same_v<OutDataType, float>)
            {
                add_device_conv1d_bwd_data_xdl_nwc_kxc_nwk_f32_instances(op_ptrs);
            }
#endif
#ifdef CK_ENABLE_FP16
            if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                         is_same_v<OutDataType, half_t>)
            {
                add_device_conv1d_bwd_data_xdl_nwc_kxc_nwk_f16_instances(op_ptrs);
            }
#endif
#ifdef CK_ENABLE_BF16
            if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
                         is_same_v<WeiDataType, ck::bhalf_t> && is_same_v<OutDataType, ck::bhalf_t>)
            {
                add_device_conv1d_bwd_data_xdl_nwc_kxc_nwk_bf16_instances(op_ptrs);
            }
#endif
#ifdef CK_ENABLE_INT8
            if constexpr(is_same_v<InDataType, int8_t> && is_same_v<WeiDataType, int8_t> &&
                         is_same_v<OutDataType, int8_t>)
            {
                add_device_conv1d_bwd_data_xdl_nwc_kxc_nwk_int8_instances(op_ptrs);
            }
#endif
        }
        if constexpr(NumDimSpatial == 2 && is_same_v<InLayout, NHWC> &&
                     is_same_v<WeiLayout, KYXC> && is_same_v<OutLayout, NHWK>)
        {
#ifdef CK_ENABLE_FP32
            if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                         is_same_v<OutDataType, float>)
            {
                add_device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_f32_instances(op_ptrs);
            }
#endif
#if defined(DL_KERNELS) && defined(CK_ENABLE_FP32)
            if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                         is_same_v<OutDataType, float>)
            {
                add_device_conv2d_bwd_data_dl_nhwc_kyxc_nhwk_f32_instances(op_ptrs);
            }
#endif
#ifdef CK_ENABLE_FP16
            if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                         is_same_v<OutDataType, half_t>)
            {
                add_device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_f16_instances(op_ptrs);
#ifdef DL_KERNELS
                add_device_conv2d_bwd_data_dl_nhwc_kyxc_nhwk_f16_instances(op_ptrs);
#endif
            }
#endif
#ifdef CK_ENABLE_BF16
            if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
                         is_same_v<WeiDataType, ck::bhalf_t> && is_same_v<OutDataType, ck::bhalf_t>)
            {
                add_device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_bf16_instances(op_ptrs);
            }
#endif
#ifdef CK_ENABLE_INT8
            if constexpr(is_same_v<InDataType, int8_t> && is_same_v<WeiDataType, int8_t> &&
                         is_same_v<OutDataType, int8_t>)
            {
                add_device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_int8_instances(op_ptrs);
#ifdef DL_KERNELS
                add_device_conv2d_bwd_data_dl_nhwc_kyxc_nhwk_int8_instances(op_ptrs);
#endif
            }
#endif
        }
        if constexpr(NumDimSpatial == 3 && is_same_v<InLayout, NDHWC> &&
                     is_same_v<WeiLayout, KZYXC> && is_same_v<OutLayout, NDHWK>)
        {
#ifdef CK_ENABLE_FP32
            if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                         is_same_v<OutDataType, float>)
            {
                add_device_conv3d_bwd_data_xdl_ndhwc_kzyxc_ndhwk_f32_instances(op_ptrs);
            }
#endif
#ifdef CK_ENABLE_FP16
            if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                         is_same_v<OutDataType, half_t>)
            {
                add_device_conv3d_bwd_data_xdl_ndhwc_kzyxc_ndhwk_f16_instances(op_ptrs);
            }
#endif
#ifdef CK_ENABLE_BF16
            if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
                         is_same_v<WeiDataType, ck::bhalf_t> && is_same_v<OutDataType, ck::bhalf_t>)
            {
                add_device_conv3d_bwd_data_xdl_ndhwc_kzyxc_ndhwk_bf16_instances(op_ptrs);
            }
#endif
#ifdef CK_ENABLE_INT8
            if constexpr(is_same_v<InDataType, int8_t> && is_same_v<WeiDataType, int8_t> &&
                         is_same_v<OutDataType, int8_t>)
            {
                add_device_conv3d_bwd_data_xdl_ndhwc_kzyxc_ndhwk_int8_instances(op_ptrs);
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
