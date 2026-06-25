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
using ScaleAdd    = ck::tensor_operation::element_wise::ScaleAdd;

#ifdef CK_USE_XDL
#ifdef CK_ENABLE_BF16
// grouped conv3d forward multi AB scaleadd, NDHWGC/GKZYXC/NDHWGK
void add_device_grouped_conv3d_fwd_xdl_scaleadd_ab_ndhwgc_gkzyxc_ndhwgk_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<>,
                                                                NDHWGK,
                                                                ck::Tuple<BF16, BF16>,
                                                                ck::Tuple<BF16, BF16>,
                                                                ck::Tuple<>,
                                                                BF16,
                                                                ScaleAdd,
                                                                ScaleAdd,
                                                                PassThrough>>>& instances);
#endif

#ifdef CK_ENABLE_FP16
void add_device_grouped_conv3d_fwd_xdl_scaleadd_ab_ndhwgc_gkzyxc_ndhwgk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<>,
                                                                NDHWGK,
                                                                ck::Tuple<F16, F16>,
                                                                ck::Tuple<F16, F16>,
                                                                ck::Tuple<>,
                                                                F16,
                                                                ScaleAdd,
                                                                ScaleAdd,
                                                                PassThrough>>>& instances);
#endif

#ifdef CK_ENABLE_FP32
void add_device_grouped_conv3d_fwd_xdl_scaleadd_ab_ndhwgc_gkzyxc_ndhwgk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<>,
                                                                NDHWGK,
                                                                ck::Tuple<F32, F32>,
                                                                ck::Tuple<F32, F32>,
                                                                ck::Tuple<>,
                                                                F32,
                                                                ScaleAdd,
                                                                ScaleAdd,
                                                                PassThrough>>>& instances);
#endif

#ifdef CK_ENABLE_INT8
void add_device_grouped_conv3d_fwd_xdl_scaleadd_ab_ndhwgc_gkzyxc_ndhwgk_int8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<>,
                                                                NDHWGK,
                                                                ck::Tuple<int8_t, int8_t>,
                                                                ck::Tuple<int8_t, int8_t>,
                                                                ck::Tuple<>,
                                                                int8_t,
                                                                ScaleAdd,
                                                                ScaleAdd,
                                                                PassThrough>>>& instances);
#endif
#endif // CK_USE_XDL

#ifdef CK_USE_WMMA
#ifdef CK_ENABLE_BF16
// grouped conv3d forward multi AB scaleadd, NDHWGC/GKZYXC/NDHWGK
void add_device_grouped_conv3d_fwd_wmma_cshufflev3_scaleadd_ab_ndhwgc_gkzyxc_ndhwgk_bf16_instances_part1(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<>,
                                                                NDHWGK,
                                                                ck::Tuple<BF16, BF16>,
                                                                ck::Tuple<BF16, BF16>,
                                                                ck::Tuple<>,
                                                                BF16,
                                                                ScaleAdd,
                                                                ScaleAdd,
                                                                PassThrough>>>& instances);
void add_device_grouped_conv3d_fwd_wmma_cshufflev3_scaleadd_ab_ndhwgc_gkzyxc_ndhwgk_bf16_instances_part2(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<>,
                                                                NDHWGK,
                                                                ck::Tuple<BF16, BF16>,
                                                                ck::Tuple<BF16, BF16>,
                                                                ck::Tuple<>,
                                                                BF16,
                                                                ScaleAdd,
                                                                ScaleAdd,
                                                                PassThrough>>>& instances);
void add_device_grouped_conv3d_fwd_wmma_cshufflev3_scaleadd_ab_ndhwgc_gkzyxc_ndhwgk_bf16_instances_part3(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<>,
                                                                NDHWGK,
                                                                ck::Tuple<BF16, BF16>,
                                                                ck::Tuple<BF16, BF16>,
                                                                ck::Tuple<>,
                                                                BF16,
                                                                ScaleAdd,
                                                                ScaleAdd,
                                                                PassThrough>>>& instances);
void add_device_grouped_conv3d_fwd_wmma_cshufflev3_scaleadd_ab_ndhwgc_gkzyxc_ndhwgk_bf16_instances_part4(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<>,
                                                                NDHWGK,
                                                                ck::Tuple<BF16, BF16>,
                                                                ck::Tuple<BF16, BF16>,
                                                                ck::Tuple<>,
                                                                BF16,
                                                                ScaleAdd,
                                                                ScaleAdd,
                                                                PassThrough>>>& instances);
#endif

#ifdef CK_ENABLE_FP16
void add_device_grouped_conv3d_fwd_wmma_cshufflev3_scaleadd_ab_ndhwgc_gkzyxc_ndhwgk_f16_instances_part1(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<>,
                                                                NDHWGK,
                                                                ck::Tuple<F16, F16>,
                                                                ck::Tuple<F16, F16>,
                                                                ck::Tuple<>,
                                                                F16,
                                                                ScaleAdd,
                                                                ScaleAdd,
                                                                PassThrough>>>& instances);
void add_device_grouped_conv3d_fwd_wmma_cshufflev3_scaleadd_ab_ndhwgc_gkzyxc_ndhwgk_f16_instances_part2(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<>,
                                                                NDHWGK,
                                                                ck::Tuple<F16, F16>,
                                                                ck::Tuple<F16, F16>,
                                                                ck::Tuple<>,
                                                                F16,
                                                                ScaleAdd,
                                                                ScaleAdd,
                                                                PassThrough>>>& instances);
void add_device_grouped_conv3d_fwd_wmma_cshufflev3_scaleadd_ab_ndhwgc_gkzyxc_ndhwgk_f16_instances_part3(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<>,
                                                                NDHWGK,
                                                                ck::Tuple<F16, F16>,
                                                                ck::Tuple<F16, F16>,
                                                                ck::Tuple<>,
                                                                F16,
                                                                ScaleAdd,
                                                                ScaleAdd,
                                                                PassThrough>>>& instances);
void add_device_grouped_conv3d_fwd_wmma_cshufflev3_scaleadd_ab_ndhwgc_gkzyxc_ndhwgk_f16_instances_part4(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<>,
                                                                NDHWGK,
                                                                ck::Tuple<F16, F16>,
                                                                ck::Tuple<F16, F16>,
                                                                ck::Tuple<>,
                                                                F16,
                                                                ScaleAdd,
                                                                ScaleAdd,
                                                                PassThrough>>>& instances);
#endif
#endif // CK_USE_WMMA

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
    ck::tensor_operation::element_wise::ScaleAdd,
    ck::tensor_operation::element_wise::ScaleAdd,
    ck::tensor_operation::element_wise::PassThrough,
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
                                        ck::tensor_operation::element_wise::ScaleAdd,
                                        ck::tensor_operation::element_wise::ScaleAdd,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ComputeType>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;
#ifdef CK_USE_XDL
        if constexpr(NumDimSpatial == 3 && is_same_v<InLayout, NDHWGC> &&
                     is_same_v<WeiLayout, GKZYXC> && is_same_v<OutLayout, NDHWGK>)
        {
#ifdef CK_ENABLE_FP32
            if constexpr(is_same_v<InDataType, ck::Tuple<float, float>> &&
                         is_same_v<WeiDataType, ck::Tuple<float, float>> &&
                         is_same_v<OutDataType, float> && is_same_v<ComputeType, float>)
            {
                add_device_grouped_conv3d_fwd_xdl_scaleadd_ab_ndhwgc_gkzyxc_ndhwgk_f32_instances(
                    op_ptrs);
            }
#endif
#ifdef CK_ENABLE_FP16
            if constexpr(is_same_v<InDataType, ck::Tuple<half_t, half_t>> &&
                         is_same_v<WeiDataType, ck::Tuple<half_t, half_t>> &&
                         is_same_v<OutDataType, half_t> && is_same_v<ComputeType, half_t>)
            {
                add_device_grouped_conv3d_fwd_xdl_scaleadd_ab_ndhwgc_gkzyxc_ndhwgk_f16_instances(
                    op_ptrs);
            }
#endif
#ifdef CK_ENABLE_BF16
            if constexpr(is_same_v<InDataType, ck::Tuple<ck::bhalf_t, ck::bhalf_t>> &&
                         is_same_v<WeiDataType, ck::Tuple<ck::bhalf_t, ck::bhalf_t>> &&
                         is_same_v<OutDataType, ck::bhalf_t> && is_same_v<ComputeType, ck::bhalf_t>)
            {
                add_device_grouped_conv3d_fwd_xdl_scaleadd_ab_ndhwgc_gkzyxc_ndhwgk_bf16_instances(
                    op_ptrs);
            }
#endif
#ifdef CK_ENABLE_INT8
            if constexpr(is_same_v<InDataType, ck::Tuple<int8_t, int8_t>> &&
                         is_same_v<WeiDataType, ck::Tuple<int8_t, int8_t>> &&
                         is_same_v<OutDataType, int8_t> && is_same_v<ComputeType, int8_t>)
            {
                add_device_grouped_conv3d_fwd_xdl_scaleadd_ab_ndhwgc_gkzyxc_ndhwgk_int8_instances(
                    op_ptrs);
            }
#endif
        }
#endif // CK_USE_XDL

#ifdef CK_USE_WMMA
        if constexpr(NumDimSpatial == 3 && is_same_v<InLayout, NDHWGC> &&
                     is_same_v<WeiLayout, GKZYXC> && is_same_v<OutLayout, NDHWGK>)
        {
#ifdef CK_ENABLE_FP16
            if constexpr(is_same_v<InDataType, ck::Tuple<half_t, half_t>> &&
                         is_same_v<WeiDataType, ck::Tuple<half_t, half_t>> &&
                         is_same_v<OutDataType, half_t> && is_same_v<ComputeType, half_t>)
            {
                add_device_grouped_conv3d_fwd_wmma_cshufflev3_scaleadd_ab_ndhwgc_gkzyxc_ndhwgk_f16_instances_part1(
                    op_ptrs);
                add_device_grouped_conv3d_fwd_wmma_cshufflev3_scaleadd_ab_ndhwgc_gkzyxc_ndhwgk_f16_instances_part2(
                    op_ptrs);
                add_device_grouped_conv3d_fwd_wmma_cshufflev3_scaleadd_ab_ndhwgc_gkzyxc_ndhwgk_f16_instances_part3(
                    op_ptrs);
                add_device_grouped_conv3d_fwd_wmma_cshufflev3_scaleadd_ab_ndhwgc_gkzyxc_ndhwgk_f16_instances_part4(
                    op_ptrs);
            }
#endif
#ifdef CK_ENABLE_BF16
            if constexpr(is_same_v<InDataType, ck::Tuple<ck::bhalf_t, ck::bhalf_t>> &&
                         is_same_v<WeiDataType, ck::Tuple<ck::bhalf_t, ck::bhalf_t>> &&
                         is_same_v<OutDataType, ck::bhalf_t> && is_same_v<ComputeType, ck::bhalf_t>)
            {
                add_device_grouped_conv3d_fwd_wmma_cshufflev3_scaleadd_ab_ndhwgc_gkzyxc_ndhwgk_bf16_instances_part1(
                    op_ptrs);
                add_device_grouped_conv3d_fwd_wmma_cshufflev3_scaleadd_ab_ndhwgc_gkzyxc_ndhwgk_bf16_instances_part2(
                    op_ptrs);
                add_device_grouped_conv3d_fwd_wmma_cshufflev3_scaleadd_ab_ndhwgc_gkzyxc_ndhwgk_bf16_instances_part3(
                    op_ptrs);
                add_device_grouped_conv3d_fwd_wmma_cshufflev3_scaleadd_ab_ndhwgc_gkzyxc_ndhwgk_bf16_instances_part4(
                    op_ptrs);
            }
#endif
        }
#endif // CK_USE_WMMA

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
