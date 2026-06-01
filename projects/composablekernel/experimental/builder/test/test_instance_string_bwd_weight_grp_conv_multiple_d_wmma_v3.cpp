// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "ck_tile/builder/reflect/instance_traits.hpp"
#include "ck_tile/builder/reflect/conv_description.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_wmma_scale_instance.hpp"

namespace {

namespace ckr = ck_tile::reflect;

// Use the first instance from device_grouped_conv_bwd_weight_wmma_c_shuffle_f16_scale_instances
using InstanceTuple = ck::tensor_operation::device::instance::
    device_grouped_conv_bwd_weight_wmma_c_shuffle_f16_scale_instances<
        2,                                     // NDimSpatial
        ck::tensor_layout::convolution::GNHWC, // ALayout (InLayout)
        ck::tensor_layout::convolution::GKYXC, // BLayout (WeiLayout)
        ck::tensor_layout::convolution::GNHWK, // ELayout (OutLayout)
        ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::Default>;

using DeviceInstance = typename std::tuple_element<0, InstanceTuple>::type;

// Expected string based on the generic instance
std::string expected_str =
    "DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3"
    "<2"             // NDimSpatial
    ",GNHWC"         // InLayout
    ",GKYXC"         // WeiLayout
    ",GNHWK"         // OutLayout
    ",EmptyTuple"    // DsLayout
    ",fp16"          // InDataType
    ",fp16"          // WeiDataType
    ",fp16"          // OutDataType
    ",fp32"          // AccDataType
    ",EmptyTuple"    // DsDataType
    ",PassThrough"   // InElementwiseOperation
    ",Scale"         // WeiElementwiseOperation
    ",PassThrough"   // OutElementwiseOperation
    ",Default"       // ConvBackwardWeightSpecialization
    ",64"            // BlockSize
    ",64"            // MPerBlock
    ",64"            // NPerBlock
    ",32"            // KPerBlock
    ",8"             // ABK1
    ",16"            // MPerWmma
    ",16"            // NPerWmma
    ",4"             // MRepeat
    ",2"             // NRepeat
    ",Seq(4,8,1)"    // ABlockTransferThreadClusterLengths_AK0_M_AK1
    ",Seq(2,0,1)"    // ABlockTransferThreadClusterArrangeOrder
    ",Seq(1,0,2)"    // ABlockTransferSrcAccessOrder
    ",1"             // ABlockTransferSrcVectorDim
    ",2"             // ABlockTransferSrcScalarPerVector
    ",4"             // ABlockTransferDstScalarPerVector_AK1
    ",true"          // ABlockLdsAddExtraM
    ",Seq(4,8,1)"    // BBlockTransferThreadClusterLengths_BK0_N_BK1
    ",Seq(2,0,1)"    // BBlockTransferThreadClusterArrangeOrder
    ",Seq(1,0,2)"    // BBlockTransferSrcAccessOrder
    ",1"             // BBlockTransferSrcVectorDim
    ",2"             // BBlockTransferSrcScalarPerVector
    ",4"             // BBlockTransferDstScalarPerVector_BK1
    ",true"          // BBlockLdsAddExtraN
    ",1"             // CShuffleMRepeatPerShuffle
    ",1"             // CShuffleNRepeatPerShuffle
    ",Seq(1,16,1,4)" // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
    ",2"             // CShuffleBlockTransferScalarPerVector_NPerBlock
    ",Intrawave"     // BlkGemmPipeSched
    ",v1"            // BlkGemmPipelineVer
    ",fp16"          // ComputeTypeA
    ",fp16"          // ComputeTypeB
    ">";

TEST(InstanceString, DescribeReturnsCorrectValueForBwdWeightGrpConvMultipleDWmmaV3)
{
    using BaseClass = ck::tensor_operation::device::BaseOperator;
    DeviceInstance device_instance;
    BaseClass* base_ptr = &device_instance;

    auto desc = base_ptr->describe();
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->instance_string(), expected_str);
}

} // namespace
