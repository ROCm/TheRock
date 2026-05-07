// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "ck_tile/builder/reflect/instance_traits.hpp"
#include "ck_tile/builder/reflect/conv_description.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_v3_xdl_instance.hpp"

namespace {

namespace ckr = ck_tile::reflect;

using InstanceTuple = ck::tensor_operation::device::instance::
    device_grouped_conv_bwd_weight_v3_xdl_c_shuffle_f16_instances<
        2,                                                            // NDimSpatial
        ck::tensor_operation::device::instance::GNHWC,                // InLayout
        ck::tensor_operation::device::instance::GKYXC,                // WeiLayout
        ck::tensor_operation::device::instance::GNHWK,                // OutLayout
        ck::tensor_operation::device::instance::ConvBwdWeightDefault, // ConvBwdWeightSpecialization
        ck::BlockGemmPipelineScheduler::Intrawave,                    // BlkGemmPipeSched
        ck::BlockGemmPipelineVersion::v1>;                            // BlkGemmPipelineVer

using DeviceInstance = typename std::tuple_element<0, InstanceTuple>::type;

// Expected complete instance string based on the generic instance
std::string expected_str =
    "DeviceGroupedConvBwdWeight_Xdl_CShuffleV3"
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
    ",64"           // BlockSize
    ",32"           // MPerBlock
    ",32"           // NPerBlock
    ",32"           // K0PerBlock
    ",8"            // K1
    ",32"           // MPerXDL
    ",32"           // NPerXDL
    ",1"            // MXdlPerWave
    ",1"            // NXdlPerWave
    ",Seq(4,8,1)"   // ABlockTransferThreadClusterLengths_K0_M_K1
    ",Seq(2,0,1)"   // ABlockTransferThreadClusterArrangeOrder
    ",Seq(1,0,2)"   // ABlockTransferSrcAccessOrder
    ",1"            // ABlockTransferSrcVectorDim
    ",2"            // ABlockTransferSrcScalarPerVector
    ",2"            // ABlockTransferDstScalarPerVector_K1
    ",false"        // ABlockLdsAddExtraM
    ",Seq(4,16,1)"  // BBlockTransferThreadClusterLengths_K0_N_K1
    ",Seq(2,0,1)"   // BBlockTransferThreadClusterArrangeOrder
    ",Seq(1,0,2)"   // BBlockTransferSrcAccessOrder
    ",1"            // BBlockTransferSrcVectorDim
    ",2"            // BBlockTransferSrcScalarPerVector
    ",2"            // BBlockTransferDstScalarPerVector_K1
    ",false"        // BBlockLdsAddExtraN
    ",1"            // CShuffleMXdlPerWavePerShuffle
    ",1"            // CShuffleNXdlPerWavePerShuffle
    ",Seq(1,8,1,8)" // CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
    ",2"            // CBlockTransferScalarPerVector_NWaveNPerXdl
    ",Intrawave"    // BlkGemmPipeSched
    ",v1"           // BlkGemmPipelineVer
    ",fp16"         // ComputeTypeA
    ",fp16"         // ComputeTypeB
    ">";

// Test describe() through base class pointer for XDL V3 variant
TEST(InstanceString, DescribeReturnsCorrectValueForBwdWeightGrpConvXdlV3)
{
    using BaseClass = ck::tensor_operation::device::BaseOperator;
    DeviceInstance device_instance;
    BaseClass* base_ptr = &device_instance;

    auto desc = base_ptr->describe();
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->instance_string(), expected_str);
}

} // namespace
