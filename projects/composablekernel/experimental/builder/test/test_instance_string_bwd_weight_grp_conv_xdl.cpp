// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "ck_tile/builder/reflect/instance_traits.hpp"
#include "ck_tile/builder/reflect/conv_description.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_xdl_instance.hpp"

namespace {

namespace ckr = ck_tile::reflect;

// Use the template helper to get a working instance configuration
using InstanceTuple = ck::tensor_operation::device::instance::
    device_grouped_conv_bwd_weight_xdl_c_shuffle_f16_instances<
        2,                                             // NDimSpatial
        ck::tensor_operation::device::instance::GNHWC, // InLayout
        ck::tensor_operation::device::instance::GKYXC, // WeiLayout
        ck::tensor_operation::device::instance::GNHWK, // OutLayout
        ck::tensor_operation::device::instance::
            ConvBwdWeightDefault>; // ConvBwdWeightSpecialization

// Get the first instance from the tuple
using DeviceInstance = typename std::tuple_element<0, InstanceTuple>::type;

// Expected complete instance string based on the first instance from
// device_grouped_conv_bwd_weight_xdl_c_shuffle_f16_instances
// This corresponds to the configuration with BlockSize=64, MPerBlock=64, NPerBlock=64, etc.
std::string expected_str = "DeviceGroupedConvBwdWeight_Xdl_CShuffle"
                           "<2"             // NDimSpatial
                           ",GNHWC"         // InLayout
                           ",GKYXC"         // WeiLayout
                           ",GNHWK"         // OutLayout
                           ",fp16"          // InDataType
                           ",fp16"          // WeiDataType
                           ",fp16"          // OutDataType
                           ",fp32"          // AccDataType
                           ",PassThrough"   // InElementwiseOperation
                           ",PassThrough"   // WeiElementwiseOperation
                           ",PassThrough"   // OutElementwiseOperation
                           ",Default"       // ConvBackwardWeightSpecialization
                           ",64"            // BlockSize
                           ",64"            // MPerBlock
                           ",64"            // NPerBlock
                           ",4"             // K0PerBlock
                           ",8"             // K1
                           ",32"            // MPerXDL
                           ",32"            // NPerXDL
                           ",2"             // MXdlPerWave
                           ",2"             // NXdlPerWave
                           ",Seq(1,4,8,2)"  // ABlockTransferThreadClusterLengths_K0_M_K1
                           ",Seq(0,3,1,2)"  // ABlockTransferThreadClusterArrangeOrder
                           ",Seq(0,2,1,3)"  // ABlockTransferSrcAccessOrder
                           ",2"             // ABlockTransferSrcVectorDim
                           ",2"             // ABlockTransferSrcScalarPerVector
                           ",4"             // ABlockTransferDstScalarPerVector_K1
                           ",true"          // ABlockLdsAddExtraM
                           ",Seq(1,4,8,2)"  // BBlockTransferThreadClusterLengths_K0_N_K1
                           ",Seq(0,3,1,2)"  // BBlockTransferThreadClusterArrangeOrder
                           ",Seq(0,2,1,3)"  // BBlockTransferSrcAccessOrder
                           ",2"             // BBlockTransferSrcVectorDim
                           ",2"             // BBlockTransferSrcScalarPerVector
                           ",4"             // BBlockTransferDstScalarPerVector_K1
                           ",true"          // BBlockLdsAddExtraN
                           ",1"             // CShuffleMXdlPerWavePerShuffle
                           ",1"             // CShuffleNXdlPerWavePerShuffle
                           ",Seq(1,16,1,4)" // CBlockTransferClusterLengths
                           ",2"             // CBlockTransferScalarPerVector_NWaveNPerXdl
                           ",fp16"          // ComputeTypeA
                           ",fp16"          // ComputeTypeB
                           ",1"             // MaxTransposeTransferSrcScalarPerVector
                           ",1>";           // MaxTransposeTransferDstScalarPerVector

// Test describe() through base class pointer for backward weight XDL variant
TEST(InstanceString, DescribeReturnsCorrectValueForBwdWeightGrpConvXdl)
{
    using BaseClass = ck::tensor_operation::device::BaseOperator;
    DeviceInstance device_instance;
    BaseClass* base_ptr = &device_instance;

    auto desc = base_ptr->describe();
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->instance_string(), expected_str);
}

// TODO: Add DescriptionReturnsCorrectValueForBwdWeightGrpConvXdl test once ckr::describe supports
// backward weight convolution The backward weight variant uses different layout naming (InLayout,
// WeiLayout, OutLayout) instead of (ALayout, BLayout, ELayout) and has
// ConvBackwardWeightSpecialization

} // namespace
