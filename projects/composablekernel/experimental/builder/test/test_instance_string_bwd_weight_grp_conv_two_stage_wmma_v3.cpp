// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "ck_tile/builder/reflect/instance_traits.hpp"
#include "ck_tile/builder/reflect/conv_description.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_two_stage_wmma_instance.hpp"

namespace {

namespace ckr = ck_tile::reflect;

// Use the first instance from
// device_grouped_conv_bwd_weight_two_stage_nhwgc_wmma_c_shuffle_f16_instances
using InstanceTuple = ck::tensor_operation::device::instance::
    device_grouped_conv_bwd_weight_two_stage_nhwgc_wmma_c_shuffle_f16_instances<
        2,                                     // NDimSpatial
        ck::tensor_layout::convolution::GNHWC, // ALayout (InLayout)
        ck::tensor_layout::convolution::GKYXC, // BLayout (WeiLayout)
        ck::tensor_layout::convolution::GNHWK, // ELayout (OutLayout)
        ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::Default,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v1>;

using DeviceInstance = typename std::tuple_element<0, InstanceTuple>::type;

// Expected string based on the first instance (BlockSize=32, MPerBlock=16, NPerBlock=16, etc.)
std::string expected_str =
    "DeviceGroupedConvBwdWeightTwoStage_Wmma_CShuffleV3"
    "<2"            // NDimSpatial
    ",GNHWC"        // InLayout
    ",GKYXC"        // WeiLayout
    ",GNHWK"        // OutLayout
    ",fp16"         // InDataType
    ",fp16"         // WeiDataType
    ",fp16"         // OutDataType
    ",fp32"         // AccDataType
    ",PassThrough"  // InElementwiseOperation
    ",PassThrough"  // WeiElementwiseOperation
    ",PassThrough"  // OutElementwiseOperation
    ",Default"      // ConvBackwardWeightSpecialization
    ",32"           // BlockSize
    ",16"           // MPerBlock
    ",16"           // NPerBlock
    ",32"           // KPerBlock
    ",8"            // ABK1
    ",16"           // MPerWmma
    ",16"           // NPerWmma
    ",1"            // MRepeat
    ",1"            // NRepeat
    ",Seq(4,8,1)"   // ABlockTransferThreadClusterLengths_AK0_M_AK1
    ",Seq(2,0,1)"   // ABlockTransferThreadClusterArrangeOrder
    ",Seq(1,0,2)"   // ABlockTransferSrcAccessOrder
    ",1"            // ABlockTransferSrcVectorDim
    ",1"            // ABlockTransferSrcScalarPerVector
    ",4"            // ABlockTransferDstScalarPerVector_AK1
    ",false"        // ABlockLdsAddExtraM
    ",Seq(4,8,1)"   // BBlockTransferThreadClusterLengths_BK0_N_BK1
    ",Seq(2,0,1)"   // BBlockTransferThreadClusterArrangeOrder
    ",Seq(1,0,2)"   // BBlockTransferSrcAccessOrder
    ",1"            // BBlockTransferSrcVectorDim
    ",1"            // BBlockTransferSrcScalarPerVector
    ",4"            // BBlockTransferDstScalarPerVector_BK1
    ",false"        // BBlockLdsAddExtraN
    ",1"            // CShuffleMRepeatPerShuffle
    ",1"            // CShuffleNRepeatPerShuffle
    ",Seq(1,4,1,8)" // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
    ",1"            // CShuffleBlockTransferScalarPerVector_NPerBlock
    ",Intrawave"    // BlkGemmPipeSched
    ",v1"           // BlkGemmPipelineVer
    ",1"            // NumGroupsToMerge
    ",fp16"         // ComputeTypeA
    ",fp16"         // ComputeTypeB
    ",1"            // TransposeTransferSrcScalarPerVector
    ",1"            // TransposeTransferDstScalarPerVector
    ">";

TEST(InstanceString, DescribeReturnsCorrectValueForBwdWeightGrpConvTwoStageWmmaV3)
{
    using BaseClass = ck::tensor_operation::device::BaseOperator;
    DeviceInstance device_instance;
    BaseClass* base_ptr = &device_instance;

    auto desc = base_ptr->describe();
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->instance_string(), expected_str);
}

} // namespace
