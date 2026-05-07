// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// #ifdef _NOT_DEFINED_

#include <gtest/gtest.h>
#include "ck_tile/builder/reflect/instance_traits.hpp"
#include "ck_tile/builder/reflect/conv_description.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_v3_wmma_instance.hpp"

namespace {

namespace ckr = ck_tile::reflect;

using InstanceTuple = ck::tensor_operation::device::instance::
    device_grouped_conv_bwd_weight_v3_wmma_c_shuffle_f16_instances<
        2,                                             // NDimSpatial
        ck::tensor_operation::device::instance::NHWGC, // InLayout
        ck::tensor_operation::device::instance::GKYXC, // WeiLayout
        ck::tensor_operation::device::instance::NHWGK, // OutLayout
        ck::tensor_operation::device::instance::ConvBwdWeightDefault>;

// Expected complete instance string
std::string expected_str = "DeviceGroupedConvBwdWeight_Wmma_CShuffleV3"
                           "<2"            // NDimSpatial
                           ",NHWGC"        // InLayout
                           ",GKYXC"        // WeiLayout
                           ",NHWGK"        // OutLayout
                           ",fp16"         // InDataType
                           ",fp16"         // WeiDataType
                           ",fp16"         // OutDataType
                           ",fp32"         // AccDataType
                           ",PassThrough"  // InElementwiseOperation
                           ",PassThrough"  // WeiElementwiseOperation
                           ",PassThrough"  // OutElementwiseOperation
                           ",Default"      // ConvBackwardWeightSpecialization
                           ",64"           // BlockSize
                           ",32"           // MPerBlock
                           ",32"           // NPerBlock
                           ",32"           // KPerBlock
                           ",8"            // ABK1
                           ",16"           // MPerWmma
                           ",16"           // NPerWmma
                           ",2"            // MRepeat
                           ",1"            // NRepeat
                           ",Seq(4,8,1)"   // ABlockTransferThreadClusterLengths_AK0_M_AK1
                           ",Seq(2,0,1)"   // ABlockTransferThreadClusterArrangeOrder
                           ",Seq(1,0,2)"   // ABlockTransferSrcAccessOrder
                           ",1"            // ABlockTransferSrcVectorDim
                           ",2"            // ABlockTransferSrcScalarPerVector
                           ",2"            // ABlockTransferDstScalarPerVector_AK1
                           ",false"        // ABlockLdsAddExtraM
                           ",Seq(4,16,1)"  // BBlockTransferThreadClusterLengths_BK0_N_BK1
                           ",Seq(2,0,1)"   // BBlockTransferThreadClusterArrangeOrder
                           ",Seq(1,0,2)"   // BBlockTransferSrcAccessOrder
                           ",1"            // BBlockTransferSrcVectorDim
                           ",2"            // BBlockTransferSrcScalarPerVector
                           ",2"            // BBlockTransferDstScalarPerVector_BK1
                           ",false"        // BBlockLdsAddExtraN
                           ",1"            // CShuffleMRepeatPerShuffle
                           ",1"            // CShuffleNRepeatPerShuffle
                           ",Seq(1,8,1,8)" // CShuffleBlockTransferClusterLengths
                           ",2"            // CShuffleBlockTransferScalarPerVector_NPerBlock
                           ",Intrawave"    // BlkGemmPipeSched
                           ",v1"           // BlkGemmPipelineVer
                           ",fp16"         // ComputeTypeA
                           ",fp16"         // ComputeTypeB
                           ",1"            // MaxTransposeTransferSrcScalarPerVector
                           ",1"            // MaxTransposeTransferDstScalarPerVector
                           ">";

// Get the first instance from the tuple
using DeviceInstance = typename std::tuple_element<0, InstanceTuple>::type;

// Test describe() through base class pointer for WMMA V3 variant
TEST(InstanceString, DescribeReturnsCorrectValueForBwdWeightGrpConvWmmaV3)
{
    using BaseClass = ck::tensor_operation::device::BaseOperator;
    DeviceInstance device_instance;
    BaseClass* base_ptr = &device_instance;

    auto desc = base_ptr->describe();
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->instance_string(), expected_str);
}

} // namespace

// #endif
