// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_bwd_data_multiple_d.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

#ifdef CK_USE_XDL
#include "grouped_convolution_backward_data_xdl.inc"
#endif
#ifdef CK_USE_WMMA
#include "grouped_convolution_backward_data_wmma.inc"
#endif

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

template <ck::index_t NumDimSpatial,
          typename OutLayout,
          typename WeiLayout,
          typename InLayout,
          typename OutDataType,
          typename WeiDataType,
          typename InDataType,
          typename ComputeTypeA,
          typename ComputeTypeB>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGroupedConvBwdDataMultipleD<
        NumDimSpatial,
        OutLayout,
        WeiLayout,
        Empty_Tuple,
        InLayout,
        OutDataType,
        WeiDataType,
        Empty_Tuple,
        InDataType,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::PassThrough,
        ComputeTypeA,
        ComputeTypeB>>
{
    using DeviceOp =
        DeviceGroupedConvBwdDataMultipleD<NumDimSpatial,
                                          OutLayout,
                                          WeiLayout,
                                          Empty_Tuple,
                                          InLayout,
                                          OutDataType,
                                          WeiDataType,
                                          Empty_Tuple,
                                          InDataType,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ComputeTypeA,
                                          ComputeTypeB>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

#ifdef CK_USE_XDL
        if constexpr(NumDimSpatial == 2)
        {
            if constexpr(is_same_v<InLayout, GNHWC> && is_same_v<WeiLayout, GKYXC> &&
                         is_same_v<OutLayout, GNHWK>)
            {
#ifdef CK_ENABLE_FP16
                if constexpr(is_same_v<InDataType, F16> && is_same_v<WeiDataType, F16> &&
                             is_same_v<OutDataType, F16> && is_same_v<ComputeTypeA, F16> &&
                             is_same_v<ComputeTypeB, F16>)
                {
                    add_device_grouped_conv2d_bwd_data_xdl_gnhwk_gkyxc_gnhwc_f16_instances(op_ptrs);
                }
#endif
#ifdef CK_ENABLE_FP32
                if constexpr(is_same_v<InDataType, F32> && is_same_v<WeiDataType, F32> &&
                             is_same_v<OutDataType, F32> && is_same_v<ComputeTypeA, F32> &&
                             is_same_v<ComputeTypeB, F32>)
                {
                    add_device_grouped_conv2d_bwd_data_xdl_gnhwk_gkyxc_gnhwc_f32_instances(op_ptrs);
                }
#endif
#ifdef CK_ENABLE_BF16
                if constexpr(is_same_v<InDataType, BF16> && is_same_v<WeiDataType, BF16> &&
                             is_same_v<OutDataType, BF16> && is_same_v<ComputeTypeA, BF16> &&
                             is_same_v<ComputeTypeB, BF16>)
                {
                    add_device_grouped_conv2d_bwd_data_xdl_gnhwk_gkyxc_gnhwc_bf16_instances(
                        op_ptrs);
                }
#endif
            }
            if constexpr(is_same_v<InLayout, NHWGC> && is_same_v<WeiLayout, GKYXC> &&
                         is_same_v<OutLayout, NHWGK>)
            {
#ifdef CK_ENABLE_FP16
                if constexpr(is_same_v<InDataType, F16> && is_same_v<WeiDataType, F16> &&
                             is_same_v<OutDataType, F16> && is_same_v<ComputeTypeA, F16> &&
                             is_same_v<ComputeTypeB, F16>)
                {
                    add_device_grouped_conv2d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_f16_instances(op_ptrs);
                    add_device_grouped_conv2d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_f16_16_16_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_f16_optimized_loads_instances(
                        op_ptrs);
                }
#endif
                if constexpr(is_same_v<InDataType, F32> && is_same_v<WeiDataType, F32> &&
                             is_same_v<OutDataType, F32>)
                {
                    static_assert(is_same_v<ComputeTypeA, ComputeTypeB>,
                                  "Error: this operator requires the same compute type");
#ifdef CK_ENABLE_TF32
                    if constexpr(is_same_v<ComputeTypeA, TF32>)
                    {
                        add_device_grouped_conv2d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_f32_tf32_instances(
                            op_ptrs);
                        add_device_grouped_conv2d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_f32_tf32_16_16_instances(
                            op_ptrs);
                        add_device_grouped_conv2d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_f32_tf32_optimized_loads_instances(
                            op_ptrs);
                    }
#endif
#ifdef CK_ENABLE_FP32
                    if constexpr(is_same_v<ComputeTypeA, F32>)
                    {
                        add_device_grouped_conv2d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_f32_instances(
                            op_ptrs);
                        add_device_grouped_conv2d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_f32_16_16_instances(
                            op_ptrs);
                        add_device_grouped_conv2d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_f32_optimized_loads_instances(
                            op_ptrs);
                    }
#endif
                }
#ifdef CK_ENABLE_BF16
                if constexpr(is_same_v<InDataType, BF16> && is_same_v<WeiDataType, BF16> &&
                             is_same_v<OutDataType, BF16> && is_same_v<ComputeTypeA, BF16> &&
                             is_same_v<ComputeTypeB, BF16>)
                {
                    add_device_grouped_conv2d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_bf16_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_bf16_16_16_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_bf16_optimized_loads_instances(
                        op_ptrs);
                }
#endif
            }
            if constexpr(is_same_v<InLayout, NGCHW> && is_same_v<WeiLayout, GKYXC> &&
                         is_same_v<OutLayout, NGKHW>)
            {
#ifdef CK_ENABLE_FP16
                if constexpr(is_same_v<InDataType, F16> && is_same_v<WeiDataType, F16> &&
                             is_same_v<OutDataType, F16> && is_same_v<ComputeTypeA, F16> &&
                             is_same_v<ComputeTypeB, F16>)
                {
                    add_device_grouped_conv2d_bwd_data_xdl_ngkhw_gkyxc_ngchw_f16_instances(op_ptrs);
                }
#endif
#ifdef CK_ENABLE_FP32
                if constexpr(is_same_v<InDataType, F32> && is_same_v<WeiDataType, F32> &&
                             is_same_v<OutDataType, F32> && is_same_v<ComputeTypeA, F32> &&
                             is_same_v<ComputeTypeB, F32>)
                {
                    add_device_grouped_conv2d_bwd_data_xdl_ngkhw_gkyxc_ngchw_f32_instances(op_ptrs);
                }
#endif
#ifdef CK_ENABLE_BF16
                if constexpr(is_same_v<InDataType, BF16> && is_same_v<WeiDataType, BF16> &&
                             is_same_v<OutDataType, BF16> && is_same_v<ComputeTypeA, BF16> &&
                             is_same_v<ComputeTypeB, BF16>)
                {
                    add_device_grouped_conv2d_bwd_data_xdl_ngkhw_gkyxc_ngchw_bf16_instances(
                        op_ptrs);
                }
#endif
            }
            if constexpr(is_same_v<InLayout, NGCHW> && is_same_v<WeiLayout, GKCYX> &&
                         is_same_v<OutLayout, NGKHW>)
            {
#ifdef CK_ENABLE_FP16
                if constexpr(is_same_v<InDataType, F16> && is_same_v<WeiDataType, F16> &&
                             is_same_v<OutDataType, F16> && is_same_v<ComputeTypeA, F16> &&
                             is_same_v<ComputeTypeB, F16>)
                {
                    add_device_grouped_conv2d_bwd_data_xdl_ngkhw_gkcyx_ngchw_f16_instances(op_ptrs);
                    add_device_grouped_conv2d_bwd_data_xdl_ngkhw_gkcyx_ngchw_f16_16_16_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_data_xdl_ngkhw_gkcyx_ngchw_f16_vec_transpose_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_FP32
                if constexpr(is_same_v<InDataType, F32> && is_same_v<WeiDataType, F32> &&
                             is_same_v<OutDataType, F32> && is_same_v<ComputeTypeA, F32> &&
                             is_same_v<ComputeTypeB, F32>)
                {
                    add_device_grouped_conv2d_bwd_data_xdl_ngkhw_gkcyx_ngchw_f32_instances(op_ptrs);
                    add_device_grouped_conv2d_bwd_data_xdl_ngkhw_gkcyx_ngchw_f32_16_16_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_data_xdl_ngkhw_gkcyx_ngchw_f32_vec_transpose_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_BF16
                if constexpr(is_same_v<InDataType, BF16> && is_same_v<WeiDataType, BF16> &&
                             is_same_v<OutDataType, BF16> && is_same_v<ComputeTypeA, BF16> &&
                             is_same_v<ComputeTypeB, BF16>)
                {
                    add_device_grouped_conv2d_bwd_data_xdl_ngkhw_gkcyx_ngchw_bf16_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_data_xdl_ngkhw_gkcyx_ngchw_bf16_16_16_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_data_xdl_ngkhw_gkcyx_ngchw_bf16_vec_transpose_instances(
                        op_ptrs);
                }
#endif
            }
        }
        if constexpr(NumDimSpatial == 3)
        {
            if constexpr(is_same_v<InLayout, GNDHWC> && is_same_v<WeiLayout, GKZYXC> &&
                         is_same_v<OutLayout, GNDHWK>)
            {
#ifdef CK_ENABLE_FP16
                if constexpr(is_same_v<InDataType, F16> && is_same_v<WeiDataType, F16> &&
                             is_same_v<OutDataType, F16> && is_same_v<ComputeTypeA, F16> &&
                             is_same_v<ComputeTypeB, F16>)
                {
                    add_device_grouped_conv3d_bwd_data_xdl_gndhwk_gkzyxc_gndhwc_f16_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_FP32
                if constexpr(is_same_v<InDataType, F32> && is_same_v<WeiDataType, F32> &&
                             is_same_v<OutDataType, F32> && is_same_v<ComputeTypeA, F32> &&
                             is_same_v<ComputeTypeB, F32>)
                {
                    add_device_grouped_conv3d_bwd_data_xdl_gndhwk_gkzyxc_gndhwc_f32_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_BF16
                if constexpr(is_same_v<InDataType, BF16> && is_same_v<WeiDataType, BF16> &&
                             is_same_v<OutDataType, BF16> && is_same_v<ComputeTypeA, BF16> &&
                             is_same_v<ComputeTypeB, BF16>)
                {
                    add_device_grouped_conv3d_bwd_data_xdl_gndhwk_gkzyxc_gndhwc_bf16_instances(
                        op_ptrs);
                }
#endif
            }
            if constexpr(is_same_v<InLayout, NDHWGC> && is_same_v<WeiLayout, GKZYXC> &&
                         is_same_v<OutLayout, NDHWGK>)
            {
#ifdef CK_ENABLE_FP16
                if constexpr(is_same_v<InDataType, F16> && is_same_v<WeiDataType, F16> &&
                             is_same_v<OutDataType, F16> && is_same_v<ComputeTypeA, F16> &&
                             is_same_v<ComputeTypeB, F16>)
                {
                    add_device_grouped_conv3d_bwd_data_xdl_ndhwgk_gkzyxc_ndhwgc_f16_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_data_xdl_ndhwgk_gkzyxc_ndhwgc_f16_16_16_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_data_xdl_ndhwgk_gkzyxc_ndhwgc_f16_optimized_loads_instances(
                        op_ptrs);
                }
#endif
#if defined CK_ENABLE_FP16 && defined CK_ENABLE_FP8 && defined CK_ENABLE_BF8
                if constexpr(is_same_v<InDataType, F16> && is_same_v<WeiDataType, F16> &&
                             is_same_v<OutDataType, F16> && is_same_v<ComputeTypeA, bf8_t> &&
                             is_same_v<ComputeTypeB, f8_t>)
                {
                    add_device_grouped_conv3d_bwd_data_xdl_ndhwgk_gkzyxc_ndhwgc_input_f16_comp_bf8f8_instances(
                        op_ptrs);
                }
#endif
                if constexpr(is_same_v<InDataType, F32> && is_same_v<WeiDataType, F32> &&
                             is_same_v<OutDataType, F32>)
                {
                    static_assert(is_same_v<ComputeTypeA, ComputeTypeB>,
                                  "Error: this operator requires the same compute type");
#ifdef CK_ENABLE_FP32
                    if constexpr(is_same_v<ComputeTypeA, F32>)
                    {
                        add_device_grouped_conv3d_bwd_data_xdl_ndhwgk_gkzyxc_ndhwgc_f32_instances(
                            op_ptrs);
                        add_device_grouped_conv3d_bwd_data_xdl_ndhwgk_gkzyxc_ndhwgc_f32_16_16_instances(
                            op_ptrs);
                        add_device_grouped_conv3d_bwd_data_xdl_ndhwgk_gkzyxc_ndhwgc_f32_optimized_loads_instances(
                            op_ptrs);
                    }
#endif
#ifdef CK_ENABLE_TF32
                    if constexpr(is_same_v<ComputeTypeA, TF32>)
                    {
                        add_device_grouped_conv3d_bwd_data_xdl_ndhwgk_gkzyxc_ndhwgc_f32_tf32_instances(
                            op_ptrs);
                        add_device_grouped_conv3d_bwd_data_xdl_ndhwgk_gkzyxc_ndhwgc_f32_tf32_16_16_instances(
                            op_ptrs);
                        add_device_grouped_conv3d_bwd_data_xdl_ndhwgk_gkzyxc_ndhwgc_f32_tf32_optimized_loads_instances(
                            op_ptrs);
                    }
#endif
                }
#ifdef CK_ENABLE_BF16
                if constexpr(is_same_v<InDataType, BF16> && is_same_v<WeiDataType, BF16> &&
                             is_same_v<OutDataType, BF16> && is_same_v<ComputeTypeA, BF16> &&
                             is_same_v<ComputeTypeB, BF16>)
                {
                    add_device_grouped_conv3d_bwd_data_xdl_ndhwgk_gkzyxc_ndhwgc_bf16_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_data_xdl_ndhwgk_gkzyxc_ndhwgc_bf16_16_16_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_data_xdl_ndhwgk_gkzyxc_ndhwgc_bf16_optimized_loads_instances(
                        op_ptrs);
                }
#endif
            }
            if constexpr(is_same_v<InLayout, NGCDHW> && is_same_v<WeiLayout, GKZYXC> &&
                         is_same_v<OutLayout, NGKDHW>)
            {
#ifdef CK_ENABLE_FP16
                if constexpr(is_same_v<InDataType, F16> && is_same_v<WeiDataType, F16> &&
                             is_same_v<OutDataType, F16> && is_same_v<ComputeTypeA, F16> &&
                             is_same_v<ComputeTypeB, F16>)
                {
                    add_device_grouped_conv3d_bwd_data_xdl_ngkdhw_gkzyxc_ngcdhw_f16_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_FP32
                if constexpr(is_same_v<InDataType, F32> && is_same_v<WeiDataType, F32> &&
                             is_same_v<OutDataType, F32> && is_same_v<ComputeTypeA, F32> &&
                             is_same_v<ComputeTypeB, F32>)
                {
                    add_device_grouped_conv3d_bwd_data_xdl_ngkdhw_gkzyxc_ngcdhw_f32_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_BF16
                if constexpr(is_same_v<InDataType, BF16> && is_same_v<WeiDataType, BF16> &&
                             is_same_v<OutDataType, BF16> && is_same_v<ComputeTypeA, BF16> &&
                             is_same_v<ComputeTypeB, BF16>)
                {
                    add_device_grouped_conv3d_bwd_data_xdl_ngkdhw_gkzyxc_ngcdhw_bf16_instances(
                        op_ptrs);
                }
#endif
            }
            if constexpr(is_same_v<InLayout, NGCDHW> && is_same_v<WeiLayout, GKCZYX> &&
                         is_same_v<OutLayout, NGKDHW>)
            {
#ifdef CK_ENABLE_FP16
                if constexpr(is_same_v<InDataType, F16> && is_same_v<WeiDataType, F16> &&
                             is_same_v<OutDataType, F16> && is_same_v<ComputeTypeA, F16> &&
                             is_same_v<ComputeTypeB, F16>)
                {
                    add_device_grouped_conv3d_bwd_data_xdl_ngkdhw_gkczyx_ngcdhw_f16_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_data_xdl_ngkdhw_gkczyx_ngcdhw_f16_16_16_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_data_xdl_ngkdhw_gkczyx_ngcdhw_f16_vec_transpose_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_FP32
                if constexpr(is_same_v<InDataType, F32> && is_same_v<WeiDataType, F32> &&
                             is_same_v<OutDataType, F32> && is_same_v<ComputeTypeA, F32> &&
                             is_same_v<ComputeTypeB, F32>)
                {
                    add_device_grouped_conv3d_bwd_data_xdl_ngkdhw_gkczyx_ngcdhw_f32_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_data_xdl_ngkdhw_gkczyx_ngcdhw_f32_16_16_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_data_xdl_ngkdhw_gkczyx_ngcdhw_f32_vec_transpose_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_BF16
                if constexpr(is_same_v<InDataType, BF16> && is_same_v<WeiDataType, BF16> &&
                             is_same_v<OutDataType, BF16> && is_same_v<ComputeTypeA, BF16> &&
                             is_same_v<ComputeTypeB, BF16>)
                {
                    add_device_grouped_conv3d_bwd_data_xdl_ngkdhw_gkczyx_ngcdhw_bf16_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_data_xdl_ngkdhw_gkczyx_ngcdhw_bf16_16_16_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_data_xdl_ngkdhw_gkczyx_ngcdhw_bf16_vec_transpose_instances(
                        op_ptrs);
                }
#endif
            }
        }
#endif

#ifdef CK_USE_WMMA
        if constexpr(NumDimSpatial == 2)
        {
            if constexpr(is_same_v<InLayout, GNHWC> && is_same_v<WeiLayout, GKYXC> &&
                         is_same_v<OutLayout, GNHWK>)
            {
#ifdef CK_ENABLE_FP16
                if constexpr(is_same_v<InDataType, F16> && is_same_v<WeiDataType, F16> &&
                             is_same_v<OutDataType, F16> && is_same_v<ComputeTypeA, F16> &&
                             is_same_v<ComputeTypeB, F16>)
                {
                    add_device_grouped_conv2d_bwd_data_wmma_gnhwk_gkyxc_gnhwc_f16_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_data_wmma_gnhwk_gkyxc_gnhwc_f16_1x1s1p0_instances(
                        op_ptrs);
                }
#endif

#ifdef CK_ENABLE_INT8
                if constexpr(is_same_v<InDataType, int8_t> && is_same_v<WeiDataType, int8_t> &&
                             is_same_v<OutDataType, int8_t> && is_same_v<ComputeTypeA, int8_t> &&
                             is_same_v<ComputeTypeB, int8_t>)
                {
                    add_device_grouped_conv2d_bwd_data_wmma_gnhwk_gkyxc_gnhwc_i8_instances(op_ptrs);
                    add_device_grouped_conv2d_bwd_data_wmma_gnhwk_gkyxc_gnhwc_i8_1x1s1p0_instances(
                        op_ptrs);
                }
#endif
            }
            if constexpr(is_same_v<InLayout, NHWGC> && is_same_v<WeiLayout, GKYXC> &&
                         is_same_v<OutLayout, NHWGK>)
            {
#ifdef CK_ENABLE_FP16
                if constexpr(is_same_v<InDataType, F16> && is_same_v<WeiDataType, F16> &&
                             is_same_v<OutDataType, F16> && is_same_v<ComputeTypeA, F16> &&
                             is_same_v<ComputeTypeB, F16>)
                {
                    add_device_grouped_conv2d_bwd_data_wmma_v3_nhwgk_gkyxc_nhwgc_f16_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_data_wmma_v3_nhwgk_gkyxc_nhwgc_f16_16_16_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_data_wmma_nhwgk_gkyxc_nhwgc_f16_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_data_wmma_nhwgk_gkyxc_nhwgc_f16_1x1s1p0_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_BF16
                if constexpr(is_same_v<InDataType, BF16> && is_same_v<WeiDataType, BF16> &&
                             is_same_v<OutDataType, BF16> && is_same_v<ComputeTypeA, BF16> &&
                             is_same_v<ComputeTypeB, BF16>)
                {
                    add_device_grouped_conv2d_bwd_data_wmma_v3_nhwgk_gkyxc_nhwgc_bf16_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_data_wmma_v3_nhwgk_gkyxc_nhwgc_bf16_16_16_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_INT8
                if constexpr(is_same_v<InDataType, int8_t> && is_same_v<WeiDataType, int8_t> &&
                             is_same_v<OutDataType, int8_t> && is_same_v<ComputeTypeA, int8_t> &&
                             is_same_v<ComputeTypeB, int8_t>)
                {
                    add_device_grouped_conv2d_bwd_data_wmma_nhwgk_gkyxc_nhwgc_i8_instances(op_ptrs);
                    add_device_grouped_conv2d_bwd_data_wmma_nhwgk_gkyxc_nhwgc_i8_1x1s1p0_instances(
                        op_ptrs);
                }
#endif
            }
        }
        if constexpr(NumDimSpatial == 3)
        {
            if constexpr(is_same_v<InLayout, GNDHWC> && is_same_v<WeiLayout, GKZYXC> &&
                         is_same_v<OutLayout, GNDHWK>)
            {
#ifdef CK_ENABLE_FP16
                if constexpr(is_same_v<InDataType, F16> && is_same_v<WeiDataType, F16> &&
                             is_same_v<OutDataType, F16> && is_same_v<ComputeTypeA, F16> &&
                             is_same_v<ComputeTypeB, F16>)
                {
                    add_device_grouped_conv3d_bwd_data_wmma_gndhwk_gkzyxc_gndhwc_f16_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_data_wmma_gndhwk_gkzyxc_gndhwc_f16_1x1s1p0_instances(
                        op_ptrs);
                }
#endif

#ifdef CK_ENABLE_INT8
                if constexpr(is_same_v<InDataType, int8_t> && is_same_v<WeiDataType, int8_t> &&
                             is_same_v<OutDataType, int8_t> && is_same_v<ComputeTypeA, int8_t> &&
                             is_same_v<ComputeTypeB, int8_t>)
                {
                    add_device_grouped_conv3d_bwd_data_wmma_gndhwk_gkzyxc_gndhwc_i8_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_data_wmma_gndhwk_gkzyxc_gndhwc_i8_1x1s1p0_instances(
                        op_ptrs);
                }
#endif
            }
            else if constexpr(is_same_v<InLayout, NDHWGC> && is_same_v<WeiLayout, GKZYXC> &&
                              is_same_v<OutLayout, NDHWGK>)
            {
#ifdef CK_ENABLE_FP16
                if constexpr(is_same_v<InDataType, F16> && is_same_v<WeiDataType, F16> &&
                             is_same_v<OutDataType, F16> && is_same_v<ComputeTypeA, F16> &&
                             is_same_v<ComputeTypeB, F16>)
                {
                    add_device_grouped_conv3d_bwd_data_wmma_ndhwgk_gkzyxc_ndhwgc_f16_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_data_wmma_ndhwgk_gkzyxc_ndhwgc_f16_1x1s1p0_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_data_wmma_v3_ndhwgk_gkzyxc_ndhwgc_f16_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_data_wmma_v3_ndhwgk_gkzyxc_ndhwgc_f16_16_16_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_BF16
                if constexpr(is_same_v<InDataType, BF16> && is_same_v<WeiDataType, BF16> &&
                             is_same_v<OutDataType, BF16> && is_same_v<ComputeTypeA, BF16> &&
                             is_same_v<ComputeTypeB, BF16>)
                {
                    add_device_grouped_conv3d_bwd_data_wmma_v3_ndhwgk_gkzyxc_ndhwgc_bf16_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_data_wmma_v3_ndhwgk_gkzyxc_ndhwgc_bf16_16_16_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_INT8
                if constexpr(is_same_v<InDataType, int8_t> && is_same_v<WeiDataType, int8_t> &&
                             is_same_v<OutDataType, int8_t> && is_same_v<ComputeTypeA, int8_t> &&
                             is_same_v<ComputeTypeB, int8_t>)
                {
                    add_device_grouped_conv3d_bwd_data_wmma_ndhwgk_gkzyxc_ndhwgc_i8_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_data_wmma_ndhwgk_gkzyxc_ndhwgc_i8_1x1s1p0_instances(
                        op_ptrs);
                }
#endif
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
