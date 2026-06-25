// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <memory>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_fwd_multiple_abd.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Scale       = ck::tensor_operation::element_wise::Scale;

#ifdef CK_USE_XDL
#ifdef CK_ENABLE_BF16
// grouped conv3d forward, NDHWGC/GKZYXC/NDHWGK
void add_device_grouped_conv3d_fwd_xdl_scale_ndhwgc_gkzyxc_ndhwgk_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<>,
                                                                NDHWGK,
                                                                BF16,
                                                                BF16,
                                                                ck::Tuple<>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Scale>>>& instances);
#endif

#ifdef CK_ENABLE_FP16
void add_device_grouped_conv3d_fwd_xdl_scale_ndhwgc_gkzyxc_ndhwgk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<>,
                                                                NDHWGK,
                                                                F16,
                                                                F16,
                                                                ck::Tuple<>,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Scale>>>& instances);
#endif

#ifdef CK_ENABLE_FP32
void add_device_grouped_conv3d_fwd_xdl_scale_ndhwgc_gkzyxc_ndhwgk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<>,
                                                                NDHWGK,
                                                                F32,
                                                                F32,
                                                                ck::Tuple<>,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                Scale>>>& instances);
#endif

#ifdef CK_ENABLE_TF32
void add_device_grouped_conv3d_fwd_xdl_scale_ndhwgc_gkzyxc_ndhwgk_f32_tf32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<>,
                                                                NDHWGK,
                                                                F32,
                                                                F32,
                                                                ck::Tuple<>,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                Scale,
                                                                TF32,
                                                                TF32>>>& instances);
#endif

#ifdef CK_ENABLE_INT8
void add_device_grouped_conv3d_fwd_xdl_scale_ndhwgc_gkzyxc_ndhwgk_int8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<>,
                                                                NDHWGK,
                                                                int8_t,
                                                                int8_t,
                                                                ck::Tuple<>,
                                                                int8_t,
                                                                PassThrough,
                                                                PassThrough,
                                                                Scale>>>& instances);
#endif
#endif

#ifdef CK_USE_WMMA
#ifdef CK_ENABLE_BF16
// grouped conv3d forward, NDHWGC/GKZYXC/NDHWGK
void add_device_grouped_conv3d_fwd_wmma_cshufflev3_scale_ndhwgc_gkzyxc_ndhwgk_bf16_instances_part1(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<>,
                                                                NDHWGK,
                                                                BF16,
                                                                BF16,
                                                                ck::Tuple<>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Scale>>>& instances);
void add_device_grouped_conv3d_fwd_wmma_cshufflev3_scale_ndhwgc_gkzyxc_ndhwgk_bf16_instances_part2(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<>,
                                                                NDHWGK,
                                                                BF16,
                                                                BF16,
                                                                ck::Tuple<>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Scale>>>& instances);
void add_device_grouped_conv3d_fwd_wmma_cshufflev3_scale_ndhwgc_gkzyxc_ndhwgk_bf16_instances_part3(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<>,
                                                                NDHWGK,
                                                                BF16,
                                                                BF16,
                                                                ck::Tuple<>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Scale>>>& instances);
void add_device_grouped_conv3d_fwd_wmma_cshufflev3_scale_ndhwgc_gkzyxc_ndhwgk_bf16_instances_part4(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<>,
                                                                NDHWGK,
                                                                BF16,
                                                                BF16,
                                                                ck::Tuple<>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Scale>>>& instances);
#endif

#ifdef CK_ENABLE_FP16
void add_device_grouped_conv3d_fwd_wmma_cshufflev3_scale_ndhwgc_gkzyxc_ndhwgk_f16_instances_part1(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<>,
                                                                NDHWGK,
                                                                F16,
                                                                F16,
                                                                ck::Tuple<>,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Scale>>>& instances);
void add_device_grouped_conv3d_fwd_wmma_cshufflev3_scale_ndhwgc_gkzyxc_ndhwgk_f16_instances_part2(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<>,
                                                                NDHWGK,
                                                                F16,
                                                                F16,
                                                                ck::Tuple<>,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Scale>>>& instances);
void add_device_grouped_conv3d_fwd_wmma_cshufflev3_scale_ndhwgc_gkzyxc_ndhwgk_f16_instances_part3(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<>,
                                                                NDHWGK,
                                                                F16,
                                                                F16,
                                                                ck::Tuple<>,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Scale>>>& instances);
void add_device_grouped_conv3d_fwd_wmma_cshufflev3_scale_ndhwgc_gkzyxc_ndhwgk_f16_instances_part4(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<>,
                                                                NDHWGK,
                                                                F16,
                                                                F16,
                                                                ck::Tuple<>,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Scale>>>& instances);
#endif
#endif

template <ck::index_t NumDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename DLayouts,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename DDataTypes,
          typename OutDataType,
          typename ComputeType>
struct DeviceOperationInstanceFactory<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<
    NumDimSpatial,
    InLayout,
    WeiLayout,
    DLayouts,
    OutLayout,
    InDataType,
    WeiDataType,
    DDataTypes,
    OutDataType,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::Scale,
    ComputeType>>
{
    using DeviceOp =
        DeviceGroupedConvFwdMultipleABD<NumDimSpatial,
                                        InLayout,
                                        WeiLayout,
                                        DLayouts,
                                        OutLayout,
                                        InDataType,
                                        WeiDataType,
                                        DDataTypes,
                                        OutDataType,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::Scale,
                                        ComputeType>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;
        if constexpr(NumDimSpatial == 3 && is_same_v<InLayout, NDHWGC> &&
                     is_same_v<WeiLayout, GKZYXC> && is_same_v<OutLayout, NDHWGK> &&
                     DLayouts::Size() == 0)
        {
            if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                         is_same_v<OutDataType, float>)
            {
#ifdef CK_ENABLE_TF32
                if constexpr(is_same_v<ComputeType, TF32>)
                {
                    add_device_grouped_conv3d_fwd_xdl_scale_ndhwgc_gkzyxc_ndhwgk_f32_tf32_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_FP32
                if constexpr(is_same_v<ComputeType, float>)
                {
                    add_device_grouped_conv3d_fwd_xdl_scale_ndhwgc_gkzyxc_ndhwgk_f32_instances(
                        op_ptrs);
                }
#endif
            }
#ifdef CK_ENABLE_FP16
            if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                         is_same_v<OutDataType, half_t> && is_same_v<ComputeType, half_t>)
            {
#ifdef CK_USE_XDL
                add_device_grouped_conv3d_fwd_xdl_scale_ndhwgc_gkzyxc_ndhwgk_f16_instances(op_ptrs);
#endif
#ifdef CK_USE_WMMA
                add_device_grouped_conv3d_fwd_wmma_cshufflev3_scale_ndhwgc_gkzyxc_ndhwgk_f16_instances_part1(
                    op_ptrs);
                add_device_grouped_conv3d_fwd_wmma_cshufflev3_scale_ndhwgc_gkzyxc_ndhwgk_f16_instances_part2(
                    op_ptrs);
                add_device_grouped_conv3d_fwd_wmma_cshufflev3_scale_ndhwgc_gkzyxc_ndhwgk_f16_instances_part3(
                    op_ptrs);
                add_device_grouped_conv3d_fwd_wmma_cshufflev3_scale_ndhwgc_gkzyxc_ndhwgk_f16_instances_part4(
                    op_ptrs);
#endif
            }
#endif
#ifdef CK_ENABLE_BF16
            if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
                         is_same_v<WeiDataType, ck::bhalf_t> && is_same_v<OutDataType, ck::bhalf_t>)
            {
#ifdef CK_USE_XDL
                add_device_grouped_conv3d_fwd_xdl_scale_ndhwgc_gkzyxc_ndhwgk_bf16_instances(
                    op_ptrs);
#endif
#ifdef CK_USE_WMMA
                add_device_grouped_conv3d_fwd_wmma_cshufflev3_scale_ndhwgc_gkzyxc_ndhwgk_bf16_instances_part1(
                    op_ptrs);
                add_device_grouped_conv3d_fwd_wmma_cshufflev3_scale_ndhwgc_gkzyxc_ndhwgk_bf16_instances_part2(
                    op_ptrs);
                add_device_grouped_conv3d_fwd_wmma_cshufflev3_scale_ndhwgc_gkzyxc_ndhwgk_bf16_instances_part3(
                    op_ptrs);
                add_device_grouped_conv3d_fwd_wmma_cshufflev3_scale_ndhwgc_gkzyxc_ndhwgk_bf16_instances_part4(
                    op_ptrs);
#endif
            }
#endif
#ifdef CK_ENABLE_INT8
            if constexpr(is_same_v<InDataType, int8_t> && is_same_v<WeiDataType, int8_t> &&
                         is_same_v<OutDataType, int8_t>)
            {
#ifdef CK_USE_XDL
                add_device_grouped_conv3d_fwd_xdl_scale_ndhwgc_gkzyxc_ndhwgk_int8_instances(
                    op_ptrs);
#endif
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
