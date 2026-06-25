// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_scaleadd_scaleadd_relu.hpp>

#include "testing_utils.hpp"

using ck_tile::test::InstanceSet;
using ck_tile::test::InstancesMatch;

namespace {

constexpr static auto NumDimSpatial = 3;
using InLayout                      = ck::tensor_layout::convolution::NDHWGC;
using WeiLayout                     = ck::tensor_layout::convolution::GKZYXC;
using OutLayout                     = ck::tensor_layout::convolution::NDHWGK;

using ck::tensor_layout::convolution::G_K;
using ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD;
using ck::tensor_operation::element_wise::PassThrough;
using ck::tensor_operation::element_wise::ScaleAddScaleAddRelu;

template <typename T, typename U = T>
using DeviceOp = DeviceGroupedConvFwdMultipleABD<NumDimSpatial,
                                                 InLayout,
                                                 WeiLayout,
                                                 ck::Tuple<OutLayout, G_K>, // DsLayout
                                                 OutLayout,
                                                 T,               // InDataType
                                                 T,               // WeiDataType
                                                 ck::Tuple<U, U>, // DsDataType
                                                 T,               // OutDataType
                                                 PassThrough,
                                                 PassThrough,
                                                 ScaleAddScaleAddRelu,
                                                 T>; // ComputeType

} // namespace

template <typename Case>
struct CkFactoryTestConvFwd : public testing::Test
{
    static auto get_actual_instances()
    {
        return InstanceSet::from_factory<typename Case::DeviceOp>();
    }

    static auto get_expected_instances() { return InstanceSet(Case::expected); }
};

struct F32
{
    using DeviceOp = ::DeviceOp<float>;

    constexpr static auto expected = {
        // clang-format off
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,fp32,fp32,fp32,fp32,Tuple(fp32,fp32),fp32,PassThrough,PassThrough,ScaleAddScaleAddRelu,Default,MNKPadding,1,256,128,128,16,4,4,32,32,2,2,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,4,true,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,4,true,1,1,Seq(1,16,1,16),4,fp32,fp32,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,fp32,fp32,fp32,fp32,Tuple(fp32,fp32),fp32,PassThrough,PassThrough,ScaleAddScaleAddRelu,Default,MNKPadding,1,256,256,128,16,4,4,32,32,4,2,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,4,4,true,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,4,4,true,1,1,Seq(1,16,1,16),4,fp32,fp32,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,fp32,fp32,fp32,fp32,Tuple(fp32,fp32),fp32,PassThrough,PassThrough,ScaleAddScaleAddRelu,Default,MNKPadding,1,64,64,32,16,4,4,32,32,2,1,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,4,4,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,4,4,true,1,1,Seq(1,8,1,8),1,fp32,fp32,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,fp32,fp32,fp32,fp32,Tuple(fp32,fp32),fp32,PassThrough,PassThrough,ScaleAddScaleAddRelu,Default,MNKPadding,1,64,64,64,16,4,4,32,32,2,2,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,4,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,4,true,1,1,Seq(1,8,1,8),1,fp32,fp32,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,fp32,fp32,fp32,fp32,Tuple(fp32,fp32),fp32,PassThrough,PassThrough,ScaleAddScaleAddRelu,Filter1x1Pad0,MNKPadding,1,256,128,128,16,4,4,32,32,2,2,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,4,true,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,4,true,1,1,Seq(1,16,1,16),4,fp32,fp32,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,fp32,fp32,fp32,fp32,Tuple(fp32,fp32),fp32,PassThrough,PassThrough,ScaleAddScaleAddRelu,Filter1x1Pad0,MNKPadding,1,256,256,128,16,4,4,32,32,4,2,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,4,4,true,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,4,4,true,1,1,Seq(1,16,1,16),4,fp32,fp32,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,fp32,fp32,fp32,fp32,Tuple(fp32,fp32),fp32,PassThrough,PassThrough,ScaleAddScaleAddRelu,Filter1x1Pad0,MNKPadding,1,64,64,32,16,4,4,32,32,2,1,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,4,4,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,4,4,true,1,1,Seq(1,8,1,8),1,fp32,fp32,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,fp32,fp32,fp32,fp32,Tuple(fp32,fp32),fp32,PassThrough,PassThrough,ScaleAddScaleAddRelu,Filter1x1Pad0,MNKPadding,1,64,64,64,16,4,4,32,32,2,2,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,4,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,4,true,1,1,Seq(1,8,1,8),1,fp32,fp32,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,fp32,fp32,fp32,fp32,Tuple(fp32,fp32),fp32,PassThrough,PassThrough,ScaleAddScaleAddRelu,Filter1x1Stride1Pad0,MNKPadding,1,256,128,128,16,4,4,32,32,2,2,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,4,true,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,4,true,1,1,Seq(1,16,1,16),4,fp32,fp32,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,fp32,fp32,fp32,fp32,Tuple(fp32,fp32),fp32,PassThrough,PassThrough,ScaleAddScaleAddRelu,Filter1x1Stride1Pad0,MNKPadding,1,256,256,128,16,4,4,32,32,4,2,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,4,4,true,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,4,4,true,1,1,Seq(1,16,1,16),4,fp32,fp32,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,fp32,fp32,fp32,fp32,Tuple(fp32,fp32),fp32,PassThrough,PassThrough,ScaleAddScaleAddRelu,Filter1x1Stride1Pad0,MNKPadding,1,64,64,32,16,4,4,32,32,2,1,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,4,4,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,4,4,true,1,1,Seq(1,8,1,8),1,fp32,fp32,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,fp32,fp32,fp32,fp32,Tuple(fp32,fp32),fp32,PassThrough,PassThrough,ScaleAddScaleAddRelu,Filter1x1Stride1Pad0,MNKPadding,1,64,64,64,16,4,4,32,32,2,2,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,4,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,4,true,1,1,Seq(1,8,1,8),1,fp32,fp32,Default,1>"
        // clang-format on
    };
};

struct F16
{
    using DeviceOp = ::DeviceOp<ck::half_t>;

    constexpr static auto expected = {
        // clang-format off
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,fp16,fp16,fp32,fp16,Tuple(fp16,fp16),fp16,PassThrough,PassThrough,ScaleAddScaleAddRelu,Default,MNKPadding,1,256,128,128,32,8,8,32,32,2,2,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,1,1,Seq(1,32,1,8),8,fp16,fp16,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,fp16,fp16,fp32,fp16,Tuple(fp16,fp16),fp16,PassThrough,PassThrough,ScaleAddScaleAddRelu,Default,MNKPadding,1,256,256,128,32,8,8,32,32,4,2,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,1,1,Seq(1,32,1,8),8,fp16,fp16,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,fp16,fp16,fp32,fp16,Tuple(fp16,fp16),fp16,PassThrough,PassThrough,ScaleAddScaleAddRelu,Default,MNKPadding,1,64,64,32,32,8,8,32,32,2,1,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,1,1,Seq(1,16,1,4),1,fp16,fp16,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,fp16,fp16,fp32,fp16,Tuple(fp16,fp16),fp16,PassThrough,PassThrough,ScaleAddScaleAddRelu,Default,MNKPadding,1,64,64,64,32,8,8,32,32,2,2,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,1,1,Seq(1,16,1,4),1,fp16,fp16,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,fp16,fp16,fp32,fp16,Tuple(fp16,fp16),fp16,PassThrough,PassThrough,ScaleAddScaleAddRelu,Filter1x1Pad0,MNKPadding,1,256,128,128,32,8,8,32,32,2,2,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,1,1,Seq(1,32,1,8),8,fp16,fp16,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,fp16,fp16,fp32,fp16,Tuple(fp16,fp16),fp16,PassThrough,PassThrough,ScaleAddScaleAddRelu,Filter1x1Pad0,MNKPadding,1,256,256,128,32,8,8,32,32,4,2,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,1,1,Seq(1,32,1,8),8,fp16,fp16,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,fp16,fp16,fp32,fp16,Tuple(fp16,fp16),fp16,PassThrough,PassThrough,ScaleAddScaleAddRelu,Filter1x1Pad0,MNKPadding,1,64,64,32,32,8,8,32,32,2,1,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,1,1,Seq(1,16,1,4),1,fp16,fp16,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,fp16,fp16,fp32,fp16,Tuple(fp16,fp16),fp16,PassThrough,PassThrough,ScaleAddScaleAddRelu,Filter1x1Pad0,MNKPadding,1,64,64,64,32,8,8,32,32,2,2,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,1,1,Seq(1,16,1,4),1,fp16,fp16,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,fp16,fp16,fp32,fp16,Tuple(fp16,fp16),fp16,PassThrough,PassThrough,ScaleAddScaleAddRelu,Filter1x1Stride1Pad0,MNKPadding,1,256,128,128,32,8,8,32,32,2,2,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,1,1,Seq(1,32,1,8),8,fp16,fp16,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,fp16,fp16,fp32,fp16,Tuple(fp16,fp16),fp16,PassThrough,PassThrough,ScaleAddScaleAddRelu,Filter1x1Stride1Pad0,MNKPadding,1,256,256,128,32,8,8,32,32,4,2,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,1,1,Seq(1,32,1,8),8,fp16,fp16,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,fp16,fp16,fp32,fp16,Tuple(fp16,fp16),fp16,PassThrough,PassThrough,ScaleAddScaleAddRelu,Filter1x1Stride1Pad0,MNKPadding,1,64,64,32,32,8,8,32,32,2,1,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,1,1,Seq(1,16,1,4),1,fp16,fp16,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,fp16,fp16,fp32,fp16,Tuple(fp16,fp16),fp16,PassThrough,PassThrough,ScaleAddScaleAddRelu,Filter1x1Stride1Pad0,MNKPadding,1,64,64,64,32,8,8,32,32,2,2,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,1,1,Seq(1,16,1,4),1,fp16,fp16,Default,1>"
        // clang-format on
    };
};

struct BF16
{
    using DeviceOp = ::DeviceOp<ck::bhalf_t>;

    constexpr static auto expected = {
        // clang-format off
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,bf16,bf16,fp32,bf16,Tuple(bf16,bf16),bf16,PassThrough,PassThrough,ScaleAddScaleAddRelu,Default,MNKPadding,1,256,128,128,32,8,8,32,32,2,2,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,1,1,Seq(1,32,1,8),8,bf16,bf16,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,bf16,bf16,fp32,bf16,Tuple(bf16,bf16),bf16,PassThrough,PassThrough,ScaleAddScaleAddRelu,Default,MNKPadding,1,256,256,128,32,8,8,32,32,4,2,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,1,1,Seq(1,32,1,8),8,bf16,bf16,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,bf16,bf16,fp32,bf16,Tuple(bf16,bf16),bf16,PassThrough,PassThrough,ScaleAddScaleAddRelu,Default,MNKPadding,1,64,64,32,32,8,8,32,32,2,1,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,1,1,Seq(1,16,1,4),1,bf16,bf16,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,bf16,bf16,fp32,bf16,Tuple(bf16,bf16),bf16,PassThrough,PassThrough,ScaleAddScaleAddRelu,Default,MNKPadding,1,64,64,64,32,8,8,32,32,2,2,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,1,1,Seq(1,16,1,4),1,bf16,bf16,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,bf16,bf16,fp32,bf16,Tuple(bf16,bf16),bf16,PassThrough,PassThrough,ScaleAddScaleAddRelu,Filter1x1Pad0,MNKPadding,1,256,128,128,32,8,8,32,32,2,2,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,1,1,Seq(1,32,1,8),8,bf16,bf16,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,bf16,bf16,fp32,bf16,Tuple(bf16,bf16),bf16,PassThrough,PassThrough,ScaleAddScaleAddRelu,Filter1x1Pad0,MNKPadding,1,256,256,128,32,8,8,32,32,4,2,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,1,1,Seq(1,32,1,8),8,bf16,bf16,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,bf16,bf16,fp32,bf16,Tuple(bf16,bf16),bf16,PassThrough,PassThrough,ScaleAddScaleAddRelu,Filter1x1Pad0,MNKPadding,1,64,64,32,32,8,8,32,32,2,1,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,1,1,Seq(1,16,1,4),1,bf16,bf16,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,bf16,bf16,fp32,bf16,Tuple(bf16,bf16),bf16,PassThrough,PassThrough,ScaleAddScaleAddRelu,Filter1x1Pad0,MNKPadding,1,64,64,64,32,8,8,32,32,2,2,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,1,1,Seq(1,16,1,4),1,bf16,bf16,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,bf16,bf16,fp32,bf16,Tuple(bf16,bf16),bf16,PassThrough,PassThrough,ScaleAddScaleAddRelu,Filter1x1Stride1Pad0,MNKPadding,1,256,128,128,32,8,8,32,32,2,2,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,1,1,Seq(1,32,1,8),8,bf16,bf16,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,bf16,bf16,fp32,bf16,Tuple(bf16,bf16),bf16,PassThrough,PassThrough,ScaleAddScaleAddRelu,Filter1x1Stride1Pad0,MNKPadding,1,256,256,128,32,8,8,32,32,4,2,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,1,1,Seq(1,32,1,8),8,bf16,bf16,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,bf16,bf16,fp32,bf16,Tuple(bf16,bf16),bf16,PassThrough,PassThrough,ScaleAddScaleAddRelu,Filter1x1Stride1Pad0,MNKPadding,1,64,64,32,32,8,8,32,32,2,1,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,1,1,Seq(1,16,1,4),1,bf16,bf16,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,bf16,bf16,fp32,bf16,Tuple(bf16,bf16),bf16,PassThrough,PassThrough,ScaleAddScaleAddRelu,Filter1x1Stride1Pad0,MNKPadding,1,64,64,64,32,8,8,32,32,2,2,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,1,1,Seq(1,16,1,4),1,bf16,bf16,Default,1>"
        // clang-format on
    };
};

struct S8
{
    using DeviceOp = ::DeviceOp<int8_t, float>;

    constexpr static auto expected = {
        // clang-format off
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,s8,s8,s32,s8,Tuple(fp32,fp32),s8,PassThrough,PassThrough,ScaleAddScaleAddRelu,Default,MNKPadding,1,256,128,128,32,8,8,32,32,2,2,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,1,1,Seq(1,32,1,8),8,s8,s8,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,s8,s8,s32,s8,Tuple(fp32,fp32),s8,PassThrough,PassThrough,ScaleAddScaleAddRelu,Default,MNKPadding,1,256,256,128,32,8,8,32,32,4,2,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,1,1,Seq(1,32,1,8),8,s8,s8,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,s8,s8,s32,s8,Tuple(fp32,fp32),s8,PassThrough,PassThrough,ScaleAddScaleAddRelu,Default,MNKPadding,1,64,64,32,32,8,8,32,32,2,1,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,1,1,Seq(1,16,1,4),1,s8,s8,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,s8,s8,s32,s8,Tuple(fp32,fp32),s8,PassThrough,PassThrough,ScaleAddScaleAddRelu,Default,MNKPadding,1,64,64,64,32,8,8,32,32,2,2,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,1,1,Seq(1,16,1,4),1,s8,s8,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,s8,s8,s32,s8,Tuple(fp32,fp32),s8,PassThrough,PassThrough,ScaleAddScaleAddRelu,Filter1x1Pad0,MNKPadding,1,256,128,128,32,8,8,32,32,2,2,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,1,1,Seq(1,32,1,8),8,s8,s8,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,s8,s8,s32,s8,Tuple(fp32,fp32),s8,PassThrough,PassThrough,ScaleAddScaleAddRelu,Filter1x1Pad0,MNKPadding,1,256,256,128,32,8,8,32,32,4,2,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,1,1,Seq(1,32,1,8),8,s8,s8,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,s8,s8,s32,s8,Tuple(fp32,fp32),s8,PassThrough,PassThrough,ScaleAddScaleAddRelu,Filter1x1Pad0,MNKPadding,1,64,64,32,32,8,8,32,32,2,1,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,1,1,Seq(1,16,1,4),1,s8,s8,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,s8,s8,s32,s8,Tuple(fp32,fp32),s8,PassThrough,PassThrough,ScaleAddScaleAddRelu,Filter1x1Pad0,MNKPadding,1,64,64,64,32,8,8,32,32,2,2,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,1,1,Seq(1,16,1,4),1,s8,s8,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,s8,s8,s32,s8,Tuple(fp32,fp32),s8,PassThrough,PassThrough,ScaleAddScaleAddRelu,Filter1x1Stride1Pad0,MNKPadding,1,256,128,128,32,8,8,32,32,2,2,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,1,1,Seq(1,32,1,8),8,s8,s8,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,s8,s8,s32,s8,Tuple(fp32,fp32),s8,PassThrough,PassThrough,ScaleAddScaleAddRelu,Filter1x1Stride1Pad0,MNKPadding,1,256,256,128,32,8,8,32,32,4,2,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,Seq(4,64,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,1,1,Seq(1,32,1,8),8,s8,s8,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,s8,s8,s32,s8,Tuple(fp32,fp32),s8,PassThrough,PassThrough,ScaleAddScaleAddRelu,Filter1x1Stride1Pad0,MNKPadding,1,64,64,32,32,8,8,32,32,2,1,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,8,8,true,1,1,Seq(1,16,1,4),1,s8,s8,Default,1>",
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<3,NDHWGC,GKZYXC,Tuple(NDHWGK,G_K),NDHWGK,s8,s8,s32,s8,Tuple(fp32,fp32),s8,PassThrough,PassThrough,ScaleAddScaleAddRelu,Filter1x1Stride1Pad0,MNKPadding,1,64,64,64,32,8,8,32,32,2,2,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,Seq(4,16,1),Seq(1,0,2),Seq(1,0,2),2,1,8,true,1,1,Seq(1,16,1,4),1,s8,s8,Default,1>"
        // clang-format on
    };
};

using TestTypes = ::testing::Types<F32, F16, BF16, S8>;

TYPED_TEST_SUITE(CkFactoryTestConvFwd, TestTypes);

TYPED_TEST(CkFactoryTestConvFwd, TestInstances)
{
    auto actual   = TestFixture::get_actual_instances();
    auto expected = TestFixture::get_expected_instances();

    EXPECT_THAT(actual, InstancesMatch(expected));
}
