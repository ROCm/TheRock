// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <memory>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_fwd_multiple_abd.hpp"
#include "ck/tensor_operation/gpu/element/combined_element_wise_operation.hpp"
#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using ConvScale   = ck::tensor_operation::element_wise::ConvScale;

#ifdef CK_ENABLE_FP8
void add_device_grouped_conv3d_fwd_xdl_convscale_ndhwgc_gkzyxc_ndhwgk_f8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<>,
                                                                NDHWGK,
                                                                F8,
                                                                F8,
                                                                ck::Tuple<>,
                                                                F8,
                                                                PassThrough,
                                                                PassThrough,
                                                                ConvScale,
                                                                F8,
                                                                F8>>>& instances);
#endif

#if(defined(CK_ENABLE_FP8) && defined(CK_ENABLE_BF8))
void add_device_grouped_conv3d_fwd_xdl_convscale_ndhwgc_gkzyxc_ndhwgk_bf8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<>,
                                                                NDHWGK,
                                                                BF8,
                                                                BF8,
                                                                ck::Tuple<>,
                                                                F8,
                                                                PassThrough,
                                                                PassThrough,
                                                                ConvScale,
                                                                BF8>>>& instances);

void add_device_grouped_conv3d_fwd_xdl_convscale_ndhwgc_gkzyxc_ndhwgk_f8_bf8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<>,
                                                                NDHWGK,
                                                                F8,
                                                                BF8,
                                                                ck::Tuple<>,
                                                                F8,
                                                                PassThrough,
                                                                PassThrough,
                                                                ConvScale,
                                                                F8,
                                                                BF8>>>& instances);

void add_device_grouped_conv3d_fwd_xdl_convscale_ndhwgc_gkzyxc_ndhwgk_bf8_f8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<>,
                                                                NDHWGK,
                                                                BF8,
                                                                F8,
                                                                ck::Tuple<>,
                                                                F8,
                                                                PassThrough,
                                                                PassThrough,
                                                                ConvScale,
                                                                BF8,
                                                                F8>>>& instances);
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
          typename AComputeType,
          typename BComputeType>
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
    ck::tensor_operation::element_wise::ConvScale,
    AComputeType,
    BComputeType>>
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
                                        ck::tensor_operation::element_wise::ConvScale,
                                        AComputeType,
                                        BComputeType>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;
        if constexpr(NumDimSpatial == 3 && is_same_v<InLayout, NDHWGC> &&
                     is_same_v<WeiLayout, GKZYXC> && is_same_v<OutLayout, NDHWGK>)
        {
#ifdef CK_ENABLE_FP8
            if constexpr(is_same_v<InDataType, f8_t> && is_same_v<WeiDataType, f8_t> &&
                         is_same_v<OutDataType, f8_t> && is_same_v<AComputeType, f8_t> &&
                         is_same_v<BComputeType, f8_t>)
            {
                add_device_grouped_conv3d_fwd_xdl_convscale_ndhwgc_gkzyxc_ndhwgk_f8_instances(
                    op_ptrs);
            }
#endif

#if(defined(CK_ENABLE_FP8) && defined(CK_ENABLE_BF8))
            if constexpr(is_same_v<InDataType, BF8> && is_same_v<WeiDataType, BF8> &&
                         is_same_v<OutDataType, F8> && is_same_v<AComputeType, BF8> &&
                         is_same_v<BComputeType, BF8>)
            {
                add_device_grouped_conv3d_fwd_xdl_convscale_ndhwgc_gkzyxc_ndhwgk_bf8_instances(
                    op_ptrs);
            }

            if constexpr(is_same_v<InDataType, f8_t> && is_same_v<WeiDataType, bf8_t> &&
                         is_same_v<OutDataType, f8_t> && is_same_v<AComputeType, f8_t> &&
                         is_same_v<BComputeType, bf8_t>)
            {
                add_device_grouped_conv3d_fwd_xdl_convscale_ndhwgc_gkzyxc_ndhwgk_f8_bf8_instances(
                    op_ptrs);
            }

            if constexpr(is_same_v<InDataType, bf8_t> && is_same_v<WeiDataType, f8_t> &&
                         is_same_v<OutDataType, f8_t> && is_same_v<AComputeType, bf8_t> &&
                         is_same_v<BComputeType, f8_t>)
            {
                add_device_grouped_conv3d_fwd_xdl_convscale_ndhwgc_gkzyxc_ndhwgk_bf8_f8_instances(
                    op_ptrs);
            }
#endif
        }
        return op_ptrs;
    }
};

using CombConvScale = ck::tensor_operation::element_wise::ScaleScalePass;

#ifdef CK_ENABLE_FP8
void add_device_grouped_conv3d_fwd_xdl_combconvscale_ndhwgc_gkzyxc_ndhwgk_f8_f8_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                ck::Tuple<>,
                                                                NDHWGK,
                                                                F8,
                                                                F8,
                                                                ck::Tuple<>,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                CombConvScale,
                                                                F8,
                                                                F8>>>& instances);
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
          typename AComputeType,
          typename BComputeType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<NumDimSpatial,
                                                                  InLayout,
                                                                  WeiLayout,
                                                                  DLayouts,
                                                                  OutLayout,
                                                                  InDataType,
                                                                  WeiDataType,
                                                                  DDataTypes,
                                                                  OutDataType,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  CombConvScale,
                                                                  AComputeType,
                                                                  BComputeType>>
{
    using DeviceOp = DeviceGroupedConvFwdMultipleABD<NumDimSpatial,
                                                     InLayout,
                                                     WeiLayout,
                                                     DLayouts,
                                                     OutLayout,
                                                     InDataType,
                                                     WeiDataType,
                                                     DDataTypes,
                                                     OutDataType,
                                                     PassThrough,
                                                     PassThrough,
                                                     CombConvScale,
                                                     AComputeType,
                                                     BComputeType>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;
        if constexpr(NumDimSpatial == 3 && is_same_v<InLayout, NDHWGC> &&
                     is_same_v<WeiLayout, GKZYXC> && is_same_v<OutLayout, NDHWGK>)
        {
#ifdef CK_ENABLE_FP8
            if constexpr(is_same_v<InDataType, f8_t> && is_same_v<WeiDataType, f8_t> &&
                         is_same_v<OutDataType, F32> && is_same_v<AComputeType, f8_t> &&
                         is_same_v<BComputeType, f8_t>)
            {
                add_device_grouped_conv3d_fwd_xdl_combconvscale_ndhwgc_gkzyxc_ndhwgk_f8_f8_f32_instances(
                    op_ptrs);
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
