// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_bwd_weight.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

#ifdef DL_KERNELS
#include "grouped_convolution_backward_weight_dl.inc"
#endif
#ifdef CK_USE_XDL
#include "grouped_convolution_backward_weight_xdl.inc"
#include "grouped_convolution_backward_weight_explicit_xdl.inc"
#endif
#ifdef CK_USE_WMMA
#include "grouped_convolution_backward_weight_wmma.inc"
#include "grouped_convolution_backward_weight_explicit_wmma.inc"
#endif
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

template <ck::index_t NumDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename ComputeTypeA,
          typename ComputeTypeB>
struct DeviceOperationInstanceFactory<ck::tensor_operation::device::DeviceGroupedConvBwdWeight<
    NumDimSpatial,
    InLayout,
    WeiLayout,
    OutLayout,
    InDataType,
    WeiDataType,
    OutDataType,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ComputeTypeA,
    ComputeTypeB>>
{
    using DeviceOp = DeviceGroupedConvBwdWeight<NumDimSpatial,
                                                InLayout,
                                                WeiLayout,
                                                OutLayout,
                                                InDataType,
                                                WeiDataType,
                                                OutDataType,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                ComputeTypeA,
                                                ComputeTypeB>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

#ifdef DL_KERNELS
        if constexpr(NumDimSpatial == 1)
        {
            if constexpr(is_same_v<InLayout, GNWC> && is_same_v<WeiLayout, GKXC> &&
                         is_same_v<OutLayout, GNWK>)
            {
#ifdef CK_ENABLE_FP32
                if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                             is_same_v<OutDataType, float> && is_same_v<ComputeTypeA, float> &&
                             is_same_v<ComputeTypeB, float>)
                {
                    add_device_grouped_conv1d_bwd_weight_dl_gnwc_gkxc_gnwk_f32_instances(op_ptrs);
                }
#endif
#ifdef CK_ENABLE_FP16
                if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                             is_same_v<OutDataType, half_t> && is_same_v<ComputeTypeA, half_t> &&
                             is_same_v<ComputeTypeB, half_t>)
                {
                    add_device_grouped_conv1d_bwd_weight_dl_gnwc_gkxc_gnwk_f16_instances(op_ptrs);
                }
#endif
#ifdef CK_ENABLE_BF16
                if constexpr(is_same_v<InDataType, ck::bhalf_t> && is_same_v<WeiDataType, float> &&
                             is_same_v<OutDataType, ck::bhalf_t> &&
                             is_same_v<ComputeTypeA, ck::bhalf_t> &&
                             is_same_v<ComputeTypeB, ck::bhalf_t>)
                {
                    add_device_grouped_conv1d_bwd_weight_dl_gnwc_gkxc_gnwk_bf16_f32_bf16_instances(
                        op_ptrs);
                }
#endif
            }
            if constexpr(is_same_v<InLayout, NWGC> && is_same_v<WeiLayout, GKXC> &&
                         is_same_v<OutLayout, NWGK>)
            {
#ifdef CK_ENABLE_FP32
                if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                             is_same_v<OutDataType, float> && is_same_v<ComputeTypeA, float> &&
                             is_same_v<ComputeTypeB, float>)
                {
                    add_device_grouped_conv1d_bwd_weight_dl_nwgc_gkxc_nwgk_f32_instances(op_ptrs);
                }
#endif
#ifdef CK_ENABLE_FP16
                if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                             is_same_v<OutDataType, half_t> && is_same_v<ComputeTypeA, half_t> &&
                             is_same_v<ComputeTypeB, half_t>)
                {
                    add_device_grouped_conv1d_bwd_weight_dl_nwgc_gkxc_nwgk_f16_instances(op_ptrs);
                }
#endif
#ifdef CK_ENABLE_BF16
                if constexpr(is_same_v<InDataType, ck::bhalf_t> && is_same_v<WeiDataType, float> &&
                             is_same_v<OutDataType, ck::bhalf_t> &&
                             is_same_v<ComputeTypeA, ck::bhalf_t> &&
                             is_same_v<ComputeTypeB, ck::bhalf_t>)
                {
                    add_device_grouped_conv1d_bwd_weight_dl_nwgc_gkxc_nwgk_bf16_f32_bf16_instances(
                        op_ptrs);
                }
#endif
            }
        }
        if constexpr(NumDimSpatial == 2)
        {
            if constexpr(is_same_v<InLayout, GNHWC> && is_same_v<WeiLayout, GKYXC> &&
                         is_same_v<OutLayout, GNHWK>)
            {
#ifdef CK_ENABLE_FP32
                if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                             is_same_v<OutDataType, float> && is_same_v<ComputeTypeA, float> &&
                             is_same_v<ComputeTypeB, float>)
                {
                    add_device_grouped_conv2d_bwd_weight_dl_gnhwc_gkyxc_gnhwk_f32_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_FP16
                if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                             is_same_v<OutDataType, half_t> && is_same_v<ComputeTypeA, half_t> &&
                             is_same_v<ComputeTypeB, half_t>)
                {
                    add_device_grouped_conv2d_bwd_weight_dl_gnhwc_gkyxc_gnhwk_f16_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_BF16
                if constexpr(is_same_v<InDataType, ck::bhalf_t> && is_same_v<WeiDataType, float> &&
                             is_same_v<OutDataType, ck::bhalf_t> &&
                             is_same_v<ComputeTypeA, ck::bhalf_t> &&
                             is_same_v<ComputeTypeB, ck::bhalf_t>)
                {
                    add_device_grouped_conv2d_bwd_weight_dl_gnhwc_gkyxc_gnhwk_bf16_f32_bf16_instances(
                        op_ptrs);
                }
#endif
            }
            if constexpr(is_same_v<InLayout, NHWGC> && is_same_v<WeiLayout, GKYXC> &&
                         is_same_v<OutLayout, NHWGK>)
            {
#ifdef CK_ENABLE_FP32
                if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                             is_same_v<OutDataType, float> && is_same_v<ComputeTypeA, float> &&
                             is_same_v<ComputeTypeB, float>)
                {
                    add_device_grouped_conv2d_bwd_weight_dl_nhwgc_gkyxc_nhwgk_f32_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_FP16
                if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                             is_same_v<OutDataType, half_t> && is_same_v<ComputeTypeA, half_t> &&
                             is_same_v<ComputeTypeB, half_t>)
                {
                    add_device_grouped_conv2d_bwd_weight_dl_nhwgc_gkyxc_nhwgk_f16_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_BF16
                if constexpr(is_same_v<InDataType, ck::bhalf_t> && is_same_v<WeiDataType, float> &&
                             is_same_v<OutDataType, ck::bhalf_t> &&
                             is_same_v<ComputeTypeA, ck::bhalf_t> &&
                             is_same_v<ComputeTypeB, ck::bhalf_t>)
                {
                    add_device_grouped_conv2d_bwd_weight_dl_nhwgc_gkyxc_nhwgk_bf16_f32_bf16_instances(
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
#ifdef CK_ENABLE_FP32
                if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                             is_same_v<OutDataType, float> && is_same_v<ComputeTypeA, float> &&
                             is_same_v<ComputeTypeB, float>)
                {
                    add_device_grouped_conv3d_bwd_weight_dl_gndhwc_gkzyxc_gndhwk_f32_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_FP16
                if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                             is_same_v<OutDataType, half_t> && is_same_v<ComputeTypeA, half_t> &&
                             is_same_v<ComputeTypeB, half_t>)
                {
                    add_device_grouped_conv3d_bwd_weight_dl_gndhwc_gkzyxc_gndhwk_f16_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_BF16
                if constexpr(is_same_v<InDataType, ck::bhalf_t> && is_same_v<WeiDataType, float> &&
                             is_same_v<OutDataType, ck::bhalf_t> &&
                             is_same_v<ComputeTypeA, ck::bhalf_t> &&
                             is_same_v<ComputeTypeB, ck::bhalf_t>)
                {
                    add_device_grouped_conv3d_bwd_weight_dl_gndhwc_gkzyxc_gndhwk_bf16_f32_bf16_instances(
                        op_ptrs);
                }
#endif
            }
            if constexpr(is_same_v<InLayout, NDHWGC> && is_same_v<WeiLayout, GKZYXC> &&
                         is_same_v<OutLayout, NDHWGK>)
            {
#ifdef CK_ENABLE_FP32
                if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                             is_same_v<OutDataType, float> && is_same_v<ComputeTypeA, float> &&
                             is_same_v<ComputeTypeB, float>)
                {
                    add_device_grouped_conv3d_bwd_weight_dl_ndhwgc_gkzyxc_ndhwgk_f32_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_FP16
                if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                             is_same_v<OutDataType, half_t> && is_same_v<ComputeTypeA, half_t> &&
                             is_same_v<ComputeTypeB, half_t>)
                {
                    add_device_grouped_conv3d_bwd_weight_dl_ndhwgc_gkzyxc_ndhwgk_f16_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_BF16
                if constexpr(is_same_v<InDataType, ck::bhalf_t> && is_same_v<WeiDataType, float> &&
                             is_same_v<OutDataType, ck::bhalf_t> &&
                             is_same_v<ComputeTypeA, ck::bhalf_t> &&
                             is_same_v<ComputeTypeB, ck::bhalf_t>)
                {
                    add_device_grouped_conv3d_bwd_weight_dl_ndhwgc_gkzyxc_ndhwgk_bf16_f32_bf16_instances(
                        op_ptrs);
                }
#endif
            }
        }
#endif // DL_KERNELS
#ifdef CK_USE_XDL
        if constexpr(NumDimSpatial == 1)
        {
            if constexpr(is_same_v<InLayout, GNWC> && is_same_v<WeiLayout, GKXC> &&
                         is_same_v<OutLayout, GNWK>)
            {
#ifdef CK_ENABLE_FP32
                if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                             is_same_v<OutDataType, float> && is_same_v<ComputeTypeA, float> &&
                             is_same_v<ComputeTypeB, float>)
                {
                    add_device_grouped_conv1d_bwd_weight_xdl_gnwc_gkxc_gnwk_f32_instances(op_ptrs);
                }
#endif
#ifdef CK_ENABLE_FP16
                if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                             is_same_v<OutDataType, half_t> && is_same_v<ComputeTypeA, half_t> &&
                             is_same_v<ComputeTypeB, half_t>)
                {
                    add_device_grouped_conv1d_bwd_weight_xdl_gnwc_gkxc_gnwk_f16_instances(op_ptrs);
                }
#endif
#ifdef CK_ENABLE_BF16
                if constexpr(is_same_v<InDataType, ck::bhalf_t> && is_same_v<WeiDataType, float> &&
                             is_same_v<OutDataType, ck::bhalf_t> &&
                             is_same_v<ComputeTypeA, ck::bhalf_t> &&
                             is_same_v<ComputeTypeB, ck::bhalf_t>)
                {
                    add_device_grouped_conv1d_bwd_weight_xdl_gnwc_gkxc_gnwk_bf16_f32_bf16_instances(
                        op_ptrs);
                }
#endif
            }
        }
        if constexpr(NumDimSpatial == 2)
        {
            if constexpr(is_same_v<InLayout, GNHWC> && is_same_v<WeiLayout, GKYXC> &&
                         is_same_v<OutLayout, GNHWK>)
            {
#ifdef CK_ENABLE_FP32
                if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                             is_same_v<OutDataType, float> && is_same_v<ComputeTypeA, float> &&
                             is_same_v<ComputeTypeB, float>)
                {
                    add_device_grouped_conv2d_bwd_weight_xdl_gnhwc_gkyxc_gnhwk_f32_instances(
                        op_ptrs);

                    add_device_grouped_conv2d_bwd_weight_xdl_gnhwc_gkyxc_gnhwk_f32_default_pipev1_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_weight_xdl_gnhwc_gkyxc_gnhwk_f32_pad0_pipev1_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_FP16
                if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                             is_same_v<OutDataType, half_t> && is_same_v<ComputeTypeA, half_t> &&
                             is_same_v<ComputeTypeB, half_t>)
                {
                    add_device_grouped_conv2d_bwd_weight_xdl_gnhwc_gkyxc_gnhwk_f16_instances(
                        op_ptrs);

                    add_device_grouped_conv2d_bwd_weight_xdl_gnhwc_gkyxc_gnhwk_f16_default_pipev1_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_weight_xdl_gnhwc_gkyxc_gnhwk_f16_pad0_pipev1_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_BF16
                if constexpr(is_same_v<InDataType, ck::bhalf_t> && is_same_v<WeiDataType, float> &&
                             is_same_v<OutDataType, ck::bhalf_t> &&
                             is_same_v<ComputeTypeA, ck::bhalf_t> &&
                             is_same_v<ComputeTypeB, ck::bhalf_t>)
                {
                    add_device_grouped_conv2d_bwd_weight_xdl_gnhwc_gkyxc_gnhwk_bf16_f32_bf16_instances(
                        op_ptrs);
                }
#endif
            }
            if constexpr(is_same_v<InLayout, NHWGC> && is_same_v<WeiLayout, GKYXC> &&
                         is_same_v<OutLayout, NHWGK>)
            {
                if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                             is_same_v<OutDataType, float>)
                {
                    static_assert(is_same_v<ComputeTypeA, ComputeTypeB>,
                                  "Error: ComputeTypeA and ComputeTypeB should be the same");
#ifdef CK_ENABLE_FP32
                    if constexpr(is_same_v<ComputeTypeA, float>)
                    {
                        add_device_grouped_conv2d_bwd_weight_xdl_nhwgc_gkyxc_nhwgk_f32_instances(
                            op_ptrs);

                        add_device_grouped_conv2d_bwd_weight_xdl_nhwgc_gkyxc_nhwgk_f32_default_pipev2_instances(
                            op_ptrs);
                        add_device_grouped_conv2d_bwd_weight_xdl_nhwgc_gkyxc_nhwgk_f32_default_pipev5_instances(
                            op_ptrs);
                        add_device_grouped_conv2d_bwd_weight_xdl_nhwgc_gkyxc_nhwgk_f32_pad0_pipev2_instances(
                            op_ptrs);
                        add_device_grouped_conv2d_bwd_weight_xdl_nhwgc_gkyxc_nhwgk_f32_pad0_pipev5_instances(
                            op_ptrs);
                    }
#endif
#ifdef CK_ENABLE_TF32
                    if constexpr(is_same_v<ComputeTypeA, TF32>)
                    {
                        add_device_grouped_conv2d_bwd_weight_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_instances(
                            op_ptrs);
                        add_device_grouped_conv2d_bwd_weight_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_default_pipev2_instances(
                            op_ptrs);
                        add_device_grouped_conv2d_bwd_weight_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_default_pipev5_instances(
                            op_ptrs);
                        add_device_grouped_conv2d_bwd_weight_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_pad0_pipev2_instances(
                            op_ptrs);
                        add_device_grouped_conv2d_bwd_weight_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_pad0_pipev5_instances(
                            op_ptrs);
                    }
#endif
                }
#ifdef CK_ENABLE_FP16
                if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                             is_same_v<OutDataType, half_t> && is_same_v<ComputeTypeA, half_t> &&
                             is_same_v<ComputeTypeB, half_t>)
                {
                    add_device_grouped_conv2d_bwd_weight_xdl_nhwgc_gkyxc_nhwgk_f16_instances(
                        op_ptrs);

                    add_device_grouped_conv2d_bwd_weight_xdl_nhwgc_gkyxc_nhwgk_f16_default_pipev2_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_weight_xdl_nhwgc_gkyxc_nhwgk_f16_default_pipev5_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_weight_xdl_nhwgc_gkyxc_nhwgk_f16_pad0_pipev2_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_weight_xdl_nhwgc_gkyxc_nhwgk_f16_pad0_pipev5_instances(
                        op_ptrs);

                    add_device_grouped_conv2d_bwd_weight_two_stage_xdl_nhwgc_gkyxc_nhwgk_f16_pipev1_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_weight_two_stage_xdl_nhwgc_gkyxc_nhwgk_f16_pipev1_part2_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_weight_two_stage_xdl_nhwgc_gkyxc_nhwgk_f16_pipev2_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_weight_two_stage_xdl_nhwgc_gkyxc_nhwgk_f16_pipev5_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_weight_two_stage_xdl_nhwgc_gkyxc_nhwgk_f16_pipev2_irregular_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_weight_two_stage_xdl_nhwgc_gkyxc_nhwgk_f16_pipev5_irregular_instances(
                        op_ptrs);
                    // Explicit GEMM
                    add_device_grouped_convnd_bwd_weight_xdl_f16_f16_f16_exp_comp_default_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_xdl_f16_f16_f16_exp_comp_mnkpadding_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_xdl_f16_f16_f16_exp_mem_v1_default_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_xdl_f16_f16_f16_exp_mem_v1_mnkpadding_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_xdl_f16_f16_f16_exp_mem_v2_default_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_xdl_f16_f16_f16_exp_mem_v2_mnkpadding_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_xdl_f16_f16_f16_exp_odd_mn_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_xdl_f16_f16_f16_exp_odd_n_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_xdl_f16_f16_f16_exp_odd_m_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_BF16
                if constexpr(is_same_v<InDataType, ck::bhalf_t> && is_same_v<WeiDataType, float> &&
                             is_same_v<OutDataType, ck::bhalf_t> &&
                             is_same_v<ComputeTypeA, ck::bhalf_t> &&
                             is_same_v<ComputeTypeB, ck::bhalf_t>)
                {
                    add_device_grouped_conv2d_bwd_weight_xdl_nhwgc_gkyxc_nhwgk_bf16_f32_bf16_instances(
                        op_ptrs);
                }
                if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
                             is_same_v<WeiDataType, ck::bhalf_t> &&
                             is_same_v<OutDataType, ck::bhalf_t> &&
                             is_same_v<ComputeTypeA, ck::bhalf_t> &&
                             is_same_v<ComputeTypeB, ck::bhalf_t>)
                {
                    add_device_grouped_conv2d_bwd_weight_xdl_nhwgc_gkyxc_nhwgk_bf16_instances(
                        op_ptrs);

                    add_device_grouped_conv2d_bwd_weight_xdl_nhwgc_gkyxc_nhwgk_bf16_default_pipev2_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_weight_xdl_nhwgc_gkyxc_nhwgk_bf16_default_pipev5_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_weight_xdl_nhwgc_gkyxc_nhwgk_bf16_pad0_pipev2_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_weight_xdl_nhwgc_gkyxc_nhwgk_bf16_pad0_pipev5_instances(
                        op_ptrs);

                    add_device_grouped_conv2d_bwd_weight_two_stage_xdl_nhwgc_gkyxc_nhwgk_bf16_pipev1_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_weight_two_stage_xdl_nhwgc_gkyxc_nhwgk_bf16_pipev1_part2_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_weight_two_stage_xdl_nhwgc_gkyxc_nhwgk_bf16_pipev2_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_weight_two_stage_xdl_nhwgc_gkyxc_nhwgk_bf16_pipev5_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_weight_two_stage_xdl_nhwgc_gkyxc_nhwgk_bf16_pipev2_irregular_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_weight_two_stage_xdl_nhwgc_gkyxc_nhwgk_bf16_pipev5_irregular_instances(
                        op_ptrs);
                    // Explicit GEMM
                    add_device_grouped_convnd_bwd_weight_xdl_bf16_bf16_bf16_exp_comp_default_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_xdl_bf16_bf16_bf16_exp_comp_mnkpadding_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_xdl_bf16_bf16_bf16_exp_mem_v1_default_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_xdl_bf16_bf16_bf16_exp_mem_v1_mnkpadding_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_xdl_bf16_bf16_bf16_exp_mem_v2_default_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_xdl_bf16_bf16_bf16_exp_mem_v2_mnkpadding_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_xdl_bf16_bf16_bf16_exp_odd_mn_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_xdl_bf16_bf16_bf16_exp_odd_m_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_xdl_bf16_bf16_bf16_exp_odd_n_instances(
                        op_ptrs);
                }
#endif
            }
            if constexpr(is_same_v<InLayout, NGCHW> && is_same_v<WeiLayout, GKCYX> &&
                         is_same_v<OutLayout, NGKHW>)
            {
#ifdef CK_ENABLE_FP16
                if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                             is_same_v<OutDataType, half_t> && is_same_v<ComputeTypeA, half_t> &&
                             is_same_v<ComputeTypeB, half_t>)
                {
                    add_device_grouped_conv2d_bwd_weight_two_stage_xdl_ngchw_gkcyx_ngkhw_f16_pipev1_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_weight_two_stage_xdl_ngchw_gkcyx_ngkhw_f16_pipev1_part2_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_weight_two_stage_xdl_ngchw_gkcyx_ngkhw_f16_pipev2_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_weight_two_stage_xdl_ngchw_gkcyx_ngkhw_f16_pipev5_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_weight_xdl_ngchw_gkcyx_ngkhw_f16_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_BF16
                if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
                             is_same_v<WeiDataType, ck::bhalf_t> &&
                             is_same_v<OutDataType, ck::bhalf_t> &&
                             is_same_v<ComputeTypeA, ck::bhalf_t> &&
                             is_same_v<ComputeTypeB, ck::bhalf_t>)
                {
                    add_device_grouped_conv2d_bwd_weight_two_stage_xdl_ngchw_gkcyx_ngkhw_bf16_pipev1_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_weight_two_stage_xdl_ngchw_gkcyx_ngkhw_bf16_pipev1_part2_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_weight_two_stage_xdl_ngchw_gkcyx_ngkhw_bf16_pipev2_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_weight_two_stage_xdl_ngchw_gkcyx_ngkhw_bf16_pipev5_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_weight_xdl_ngchw_gkcyx_ngkhw_bf16_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_FP32
                if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                             is_same_v<OutDataType, float> && is_same_v<ComputeTypeA, float> &&
                             is_same_v<ComputeTypeB, float>)
                {
                    add_device_grouped_conv2d_bwd_weight_xdl_ngchw_gkcyx_ngkhw_f32_instances(
                        op_ptrs);
                }
#endif
            }
            if constexpr(is_same_v<InLayout, NGCHW> && is_same_v<WeiLayout, GKYXC> &&
                         is_same_v<OutLayout, NGKHW>)
            {
#ifdef CK_ENABLE_FP16
                if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                             is_same_v<OutDataType, half_t> && is_same_v<ComputeTypeA, half_t> &&
                             is_same_v<ComputeTypeB, half_t>)
                {
                    add_device_grouped_conv2d_bwd_weight_two_stage_xdl_ngchw_gkyxc_ngkhw_f16_pipev1_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_BF16
                if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
                             is_same_v<WeiDataType, ck::bhalf_t> &&
                             is_same_v<OutDataType, ck::bhalf_t> &&
                             is_same_v<ComputeTypeA, ck::bhalf_t> &&
                             is_same_v<ComputeTypeB, ck::bhalf_t>)
                {
                    add_device_grouped_conv2d_bwd_weight_two_stage_xdl_ngchw_gkyxc_ngkhw_bf16_pipev1_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_FP32
                if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                             is_same_v<OutDataType, float> && is_same_v<ComputeTypeA, float> &&
                             is_same_v<ComputeTypeB, float>)
                {
                    add_device_grouped_conv2d_bwd_weight_xdl_ngchw_gkyxc_ngkhw_f32_instances(
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
#ifdef CK_ENABLE_FP32
                if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                             is_same_v<OutDataType, float> && is_same_v<ComputeTypeA, float> &&
                             is_same_v<ComputeTypeB, float>)
                {
                    add_device_grouped_conv3d_bwd_weight_xdl_gndhwc_gkzyxc_gndhwk_f32_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_FP16
                if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                             is_same_v<OutDataType, half_t> && is_same_v<ComputeTypeA, half_t> &&
                             is_same_v<ComputeTypeB, half_t>)
                {
                    add_device_grouped_conv3d_bwd_weight_xdl_gndhwc_gkzyxc_gndhwk_f16_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_BF16
                if constexpr(is_same_v<InDataType, ck::bhalf_t> && is_same_v<WeiDataType, float> &&
                             is_same_v<OutDataType, ck::bhalf_t> &&
                             is_same_v<ComputeTypeA, ck::bhalf_t> &&
                             is_same_v<ComputeTypeB, ck::bhalf_t>)
                {
                    add_device_grouped_conv3d_bwd_weight_xdl_gndhwc_gkzyxc_gndhwk_bf16_f32_bf16_instances(
                        op_ptrs);
                }
#endif
            }
            if constexpr(is_same_v<InLayout, NDHWGC> && is_same_v<WeiLayout, GKZYXC> &&
                         is_same_v<OutLayout, NDHWGK>)
            {
                if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                             is_same_v<OutDataType, float>)
                {
                    static_assert(is_same_v<ComputeTypeA, ComputeTypeB>,
                                  "Error: ComputeTypeA and ComputeTypeB should be the same");
#ifdef CK_ENABLE_FP32
                    if constexpr(is_same_v<ComputeTypeA, float>)
                    {
                        add_device_grouped_conv3d_bwd_weight_xdl_ndhwgc_gkzyxc_ndhwgk_f32_instances(
                            op_ptrs);
                        add_device_grouped_conv3d_bwd_weight_xdl_ndhwgc_gkzyxc_ndhwgk_f32_default_pipev2_instances(
                            op_ptrs);
                        add_device_grouped_conv3d_bwd_weight_xdl_ndhwgc_gkzyxc_ndhwgk_f32_default_pipev5_instances(
                            op_ptrs);
                        add_device_grouped_conv3d_bwd_weight_xdl_ndhwgc_gkzyxc_ndhwgk_f32_pad0_pipev2_instances(
                            op_ptrs);
                        add_device_grouped_conv3d_bwd_weight_xdl_ndhwgc_gkzyxc_ndhwgk_f32_pad0_pipev5_instances(
                            op_ptrs);
                    }
#endif
#ifdef CK_ENABLE_TF32
                    if constexpr(is_same_v<ComputeTypeA, TF32>)
                    {
                        add_device_grouped_conv3d_bwd_weight_xdl_ndhwgc_gkzyxc_ndhwgk_f32_tf32_instances(
                            op_ptrs);
                        add_device_grouped_conv3d_bwd_weight_xdl_ndhwgc_gkzyxc_ndhwgk_f32_tf32_default_pipev2_instances(
                            op_ptrs);
                        add_device_grouped_conv3d_bwd_weight_xdl_ndhwgc_gkzyxc_ndhwgk_f32_tf32_default_pipev5_instances(
                            op_ptrs);
                        add_device_grouped_conv3d_bwd_weight_xdl_ndhwgc_gkzyxc_ndhwgk_f32_tf32_pad0_pipev2_instances(
                            op_ptrs);
                        add_device_grouped_conv3d_bwd_weight_xdl_ndhwgc_gkzyxc_ndhwgk_f32_tf32_pad0_pipev5_instances(
                            op_ptrs);
                    }
#endif
                }
#ifdef CK_ENABLE_FP16
                if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                             is_same_v<OutDataType, half_t> && is_same_v<ComputeTypeA, half_t> &&
                             is_same_v<ComputeTypeB, half_t>)
                {
                    add_device_grouped_conv3d_bwd_weight_xdl_ndhwgc_gkzyxc_ndhwgk_f16_instances(
                        op_ptrs);

                    add_device_grouped_conv3d_bwd_weight_xdl_ndhwgc_gkzyxc_ndhwgk_f16_default_pipev2_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_weight_xdl_ndhwgc_gkzyxc_ndhwgk_f16_default_pipev5_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_weight_xdl_ndhwgc_gkzyxc_ndhwgk_f16_pad0_pipev2_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_weight_xdl_ndhwgc_gkzyxc_ndhwgk_f16_pad0_pipev5_instances(
                        op_ptrs);

                    add_device_grouped_conv3d_bwd_weight_two_stage_xdl_ndhwgc_gkzyxc_ndhwgk_f16_pipev1_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_weight_two_stage_xdl_ndhwgc_gkzyxc_ndhwgk_f16_pipev1_part2_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_weight_two_stage_xdl_ndhwgc_gkzyxc_ndhwgk_f16_pipev2_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_weight_two_stage_xdl_ndhwgc_gkzyxc_ndhwgk_f16_pipev5_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_weight_two_stage_xdl_ndhwgc_gkzyxc_ndhwgk_f16_pipev2_irregular_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_weight_two_stage_xdl_ndhwgc_gkzyxc_ndhwgk_f16_pipev5_irregular_instances(
                        op_ptrs);
                    // Explicit GEMM
                    add_device_grouped_convnd_bwd_weight_xdl_f16_f16_f16_exp_comp_default_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_xdl_f16_f16_f16_exp_comp_mnkpadding_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_xdl_f16_f16_f16_exp_mem_v1_default_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_xdl_f16_f16_f16_exp_mem_v1_mnkpadding_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_xdl_f16_f16_f16_exp_mem_v2_default_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_xdl_f16_f16_f16_exp_mem_v2_mnkpadding_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_xdl_f16_f16_f16_exp_odd_mn_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_xdl_f16_f16_f16_exp_odd_n_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_xdl_f16_f16_f16_exp_odd_m_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_BF16
                if constexpr(is_same_v<InDataType, ck::bhalf_t> && is_same_v<WeiDataType, float> &&
                             is_same_v<OutDataType, ck::bhalf_t> &&
                             is_same_v<ComputeTypeA, ck::bhalf_t> &&
                             is_same_v<ComputeTypeB, ck::bhalf_t>)
                {
                    add_device_grouped_conv3d_bwd_weight_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_f32_bf16_instances(
                        op_ptrs);
                }
                if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
                             is_same_v<WeiDataType, ck::bhalf_t> &&
                             is_same_v<OutDataType, ck::bhalf_t> &&
                             is_same_v<ComputeTypeA, ck::bhalf_t> &&
                             is_same_v<ComputeTypeB, ck::bhalf_t>)
                {
                    add_device_grouped_conv3d_bwd_weight_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_instances(
                        op_ptrs);

                    add_device_grouped_conv3d_bwd_weight_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_default_pipev2_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_weight_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_default_pipev5_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_weight_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_pad0_pipev2_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_weight_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_pad0_pipev5_instances(
                        op_ptrs);

                    add_device_grouped_conv3d_bwd_weight_two_stage_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_pipev1_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_weight_two_stage_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_pipev1_part2_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_weight_two_stage_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_pipev2_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_weight_two_stage_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_pipev5_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_weight_two_stage_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_pipev2_irregular_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_weight_two_stage_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_pipev5_irregular_instances(
                        op_ptrs);
                    // Explicit GEMM
                    add_device_grouped_convnd_bwd_weight_xdl_bf16_bf16_bf16_exp_comp_default_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_xdl_bf16_bf16_bf16_exp_comp_mnkpadding_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_xdl_bf16_bf16_bf16_exp_mem_v1_default_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_xdl_bf16_bf16_bf16_exp_mem_v1_mnkpadding_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_xdl_bf16_bf16_bf16_exp_mem_v2_default_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_xdl_bf16_bf16_bf16_exp_mem_v2_mnkpadding_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_xdl_bf16_bf16_bf16_exp_odd_mn_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_xdl_bf16_bf16_bf16_exp_odd_m_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_xdl_bf16_bf16_bf16_exp_odd_n_instances(
                        op_ptrs);
                }
#endif
#if defined CK_ENABLE_FP16 && defined CK_ENABLE_FP8 && defined CK_ENABLE_BF8
                if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                             is_same_v<OutDataType, half_t> && is_same_v<ComputeTypeA, bf8_t> &&
                             is_same_v<ComputeTypeB, f8_t>)
                {
                    add_device_grouped_conv3d_bwd_weight_xdl_ndhwgc_gkzyxc_ndhwgk_f16_comp_bf8_f8_instances(
                        op_ptrs);
                }
#endif
            }
            if constexpr(is_same_v<InLayout, NGCDHW> && is_same_v<WeiLayout, GKCZYX> &&
                         is_same_v<OutLayout, NGKDHW>)
            {
#ifdef CK_ENABLE_FP16
                if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                             is_same_v<OutDataType, half_t> && is_same_v<ComputeTypeA, half_t> &&
                             is_same_v<ComputeTypeB, half_t>)
                {
                    add_device_grouped_conv3d_bwd_weight_two_stage_xdl_ngcdhw_gkczyx_ngkdhw_f16_pipev1_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_weight_two_stage_xdl_ngcdhw_gkczyx_ngkdhw_f16_pipev1_part2_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_weight_two_stage_xdl_ngcdhw_gkczyx_ngkdhw_f16_pipev2_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_weight_two_stage_xdl_ngcdhw_gkczyx_ngkdhw_f16_pipev5_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_weight_xdl_ngcdhw_gkczyx_ngkdhw_f16_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_BF16
                if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
                             is_same_v<WeiDataType, ck::bhalf_t> &&
                             is_same_v<OutDataType, ck::bhalf_t> &&
                             is_same_v<ComputeTypeA, ck::bhalf_t> &&
                             is_same_v<ComputeTypeB, ck::bhalf_t>)
                {
                    add_device_grouped_conv3d_bwd_weight_two_stage_xdl_ngcdhw_gkczyx_ngkdhw_bf16_pipev1_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_weight_two_stage_xdl_ngcdhw_gkczyx_ngkdhw_bf16_pipev1_part2_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_weight_two_stage_xdl_ngcdhw_gkczyx_ngkdhw_bf16_pipev2_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_weight_two_stage_xdl_ngcdhw_gkczyx_ngkdhw_bf16_pipev5_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_weight_xdl_ngcdhw_gkczyx_ngkdhw_bf16_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_FP32
                if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                             is_same_v<OutDataType, float> && is_same_v<ComputeTypeA, float> &&
                             is_same_v<ComputeTypeB, float>)
                {
                    add_device_grouped_conv3d_bwd_weight_xdl_ngcdhw_gkczyx_ngkdhw_f32_instances(
                        op_ptrs);
                }
#endif
            }
            if constexpr(is_same_v<InLayout, NGCDHW> && is_same_v<WeiLayout, GKZYXC> &&
                         is_same_v<OutLayout, NGKDHW>)
            {
#ifdef CK_ENABLE_FP16
                if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                             is_same_v<OutDataType, half_t> && is_same_v<ComputeTypeA, half_t> &&
                             is_same_v<ComputeTypeB, half_t>)
                {
                    add_device_grouped_conv3d_bwd_weight_two_stage_xdl_ngcdhw_gkzyxc_ngkdhw_f16_pipev1_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_BF16
                if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
                             is_same_v<WeiDataType, ck::bhalf_t> &&
                             is_same_v<OutDataType, ck::bhalf_t> &&
                             is_same_v<ComputeTypeA, ck::bhalf_t> &&
                             is_same_v<ComputeTypeB, ck::bhalf_t>)
                {
                    add_device_grouped_conv3d_bwd_weight_two_stage_xdl_ngcdhw_gkzyxc_ngkdhw_bf16_pipev1_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_FP32
                if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                             is_same_v<OutDataType, float> && is_same_v<ComputeTypeA, float> &&
                             is_same_v<ComputeTypeB, float>)
                {
                    add_device_grouped_conv3d_bwd_weight_xdl_ngcdhw_gkzyxc_ngkdhw_f32_instances(
                        op_ptrs);
                }
#endif
            }
        }
#endif
#ifdef CK_USE_WMMA
        if constexpr(NumDimSpatial == 2)
        {
            if constexpr(is_same_v<InLayout, NHWGC> && is_same_v<WeiLayout, GKYXC> &&
                         is_same_v<OutLayout, NHWGK>)
            {
#ifdef CK_ENABLE_FP16
                if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                             is_same_v<OutDataType, half_t> && is_same_v<ComputeTypeA, half_t> &&
                             is_same_v<ComputeTypeB, half_t>)
                {
                    add_device_grouped_conv2d_bwd_weight_wmma_nhwgc_gkyxc_nhwgk_f16_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_weight_two_stage_wmma_nhwgc_gkyxc_nhwgk_f16_pipev1_instances(
                        op_ptrs);
                    // Explicit GEMM
                    add_device_grouped_convnd_bwd_weight_wmma_f16_f16_f16_exp_comp_default_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_wmma_f16_f16_f16_exp_comp_mnkpadding_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_wmma_f16_f16_f16_exp_odd_mn_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_BF16
                if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
                             is_same_v<WeiDataType, ck::bhalf_t> &&
                             is_same_v<OutDataType, ck::bhalf_t> &&
                             is_same_v<ComputeTypeA, ck::bhalf_t> &&
                             is_same_v<ComputeTypeB, ck::bhalf_t>)
                {
                    add_device_grouped_conv2d_bwd_weight_wmma_nhwgc_gkyxc_nhwgk_bf16_instances(
                        op_ptrs);
                    add_device_grouped_conv2d_bwd_weight_two_stage_wmma_nhwgc_gkyxc_nhwgk_bf16_pipev1_instances(
                        op_ptrs);
                    // Explicit GEMM
                    add_device_grouped_convnd_bwd_weight_wmma_bf16_bf16_bf16_exp_comp_default_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_wmma_bf16_bf16_bf16_exp_odd_mn_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_wmma_bf16_bf16_bf16_exp_comp_mnkpadding_instances(
                        op_ptrs);
                }
#endif
            }
        }
        if constexpr(NumDimSpatial == 3)
        {
            if constexpr(is_same_v<InLayout, NDHWGC> && is_same_v<WeiLayout, GKZYXC> &&
                         is_same_v<OutLayout, NDHWGK>)
            {
#ifdef CK_ENABLE_FP16
                if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                             is_same_v<OutDataType, half_t> && is_same_v<ComputeTypeA, half_t> &&
                             is_same_v<ComputeTypeB, half_t>)
                {
                    add_device_grouped_conv3d_bwd_weight_wmma_ndhwgc_gkzyxc_ndhwgk_f16_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_weight_two_stage_wmma_ndhwgc_gkzyxc_ndhwgk_f16_pipev1_instances(
                        op_ptrs);
                    // Explicit GEMM
                    add_device_grouped_convnd_bwd_weight_wmma_f16_f16_f16_exp_comp_default_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_wmma_f16_f16_f16_exp_comp_mnkpadding_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_wmma_f16_f16_f16_exp_odd_mn_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_BF16
                if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
                             is_same_v<WeiDataType, ck::bhalf_t> &&
                             is_same_v<OutDataType, ck::bhalf_t> &&
                             is_same_v<ComputeTypeA, ck::bhalf_t> &&
                             is_same_v<ComputeTypeB, ck::bhalf_t>)
                {
                    add_device_grouped_conv3d_bwd_weight_wmma_ndhwgc_gkzyxc_ndhwgk_bf16_instances(
                        op_ptrs);
                    add_device_grouped_conv3d_bwd_weight_two_stage_wmma_ndhwgc_gkzyxc_ndhwgk_bf16_pipev1_instances(
                        op_ptrs);
                    // Explicit GEMM
                    add_device_grouped_convnd_bwd_weight_wmma_bf16_bf16_bf16_exp_comp_default_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_wmma_bf16_bf16_bf16_exp_odd_mn_instances(
                        op_ptrs);
                    add_device_grouped_convnd_bwd_weight_wmma_bf16_bf16_bf16_exp_comp_mnkpadding_instances(
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
