// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <ck_tile/builder/reflect/instance_traits.hpp>
#include <ck_tile/builder/reflect/conv_description.hpp>
#include <ck/tensor_operation/gpu/device/device_base.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_instance.hpp>

namespace {

namespace ckr = ck_tile::reflect;

// Use the template helper to get a working instance configuration
using InstanceTuple =
    ck::tensor_operation::device::instance::device_grouped_conv_fwd_wmma_f16_instances<
        2,                                                       // NDimSpatial
        ck::tensor_operation::device::instance::GNHWC,           // ALayout
        ck::tensor_operation::device::instance::GKYXC,           // BLayout
        ck::tensor_operation::device::instance::Empty_Tuple,     // DsLayout
        ck::tensor_operation::device::instance::GNHWK,           // ELayout
        ck::Tuple<>,                                             // DsDatatype
        ck::tensor_operation::element_wise::PassThrough,         // CDEElementOp
        ck::tensor_operation::device::instance::ConvFwdDefault>; // ConvSpec

// Get the first instance from the tuple
using DeviceInstance = typename std::tuple_element<0, InstanceTuple>::type;

// Expected complete instance string based on the first instance from
// device_grouped_conv_fwd_wmma_f16_instances
std::string expected_str = "DeviceGroupedConvFwdMultipleD_Wmma_CShuffle"
                           "<2"             // NDimSpatial
                           ",GNHWC"         // ALayout
                           ",GKYXC"         // BLayout
                           ",EmptyTuple"    // DsLayout
                           ",GNHWK"         // ELayout
                           ",fp16"          // ADataType
                           ",fp16"          // BDataType
                           ",fp32"          // AccDataType
                           ",fp16"          // CShuffleDataType
                           ",EmptyTuple"    // DsDataType
                           ",fp16"          // EDataType
                           ",PassThrough"   // AElementwiseOperation
                           ",PassThrough"   // BElementwiseOperation
                           ",PassThrough"   // CDEElementwiseOperation
                           ",Default"       // ConvForwardSpecialization
                           ",MNKPadding"    // GemmSpec
                           ",1"             // NumGemmKPrefetchStage
                           ",128"           // BlockSize
                           ",64"            // MPerBlock
                           ",64"            // NPerBlock
                           ",32"            // KPerBlock
                           ",8"             // K1
                           ",16"            // MPerWmma
                           ",16"            // NPerWmma
                           ",2"             // MRepeat
                           ",2"             // NRepeat
                           ",Seq(4,32,1)"   // ABlockTransferThreadClusterLengths
                           ",Seq(1,0,2)"    // ABlockTransferThreadClusterArrangeOrder
                           ",Seq(1,0,2)"    // ABlockTransferSrcAccessOrder
                           ",2"             // ABlockTransferSrcVectorDim
                           ",1"             // ABlockTransferSrcScalarPerVector
                           ",8"             // ABlockTransferDstScalarPerVector_AK1
                           ",true"          // ABlockLdsExtraM
                           ",Seq(4,32,1)"   // BBlockTransferThreadClusterLengths
                           ",Seq(1,0,2)"    // BBlockTransferThreadClusterArrangeOrder
                           ",Seq(1,0,2)"    // BBlockTransferSrcAccessOrder
                           ",2"             // BBlockTransferSrcVectorDim
                           ",1"             // BBlockTransferSrcScalarPerVector
                           ",8"             // BBlockTransferDstScalarPerVector_BK1
                           ",true"          // BBlockLdsExtraN
                           ",1"             // CShuffleMRepeatPerShuffle
                           ",1"             // CShuffleNRepeatPerShuffle
                           ",Seq(1,32,1,4)" // CDEShuffleBlockTransferClusterLengths
                           ",1"             // CDEShuffleBlockTransferScalarPerVector_NPerBlock
                           ",Default"       // LoopSched
                           ",v1>";          // PipelineVer

// Test describe() through base class pointer for WMMA variant
TEST(InstanceString, DescribeReturnsCorrectValueForFwdGrpConvWmma)
{
    using BaseClass = ck::tensor_operation::device::BaseOperator;
    DeviceInstance device_instance;
    BaseClass* base_ptr = &device_instance;

    auto desc = base_ptr->describe();
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->instance_string(), expected_str);
}

// TODO: Add DescriptionReturnsCorrectValueForFwdGrpConvWmma test once ckr::describe supports WMMA
// variants The WMMA variant uses WMMA-specific parameters (MPerWmma, NPerWmma, MRepeat, NRepeat)
// and has a different tile transfer structure incompatible with current ConvTraits

} // namespace
