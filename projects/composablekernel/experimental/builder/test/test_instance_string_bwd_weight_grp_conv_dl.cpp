// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "ck_tile/builder/reflect/instance_traits.hpp"
#include "ck_tile/builder/reflect/conv_description.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_dl_instance.hpp"

namespace {

namespace ckr = ck_tile::reflect;

// Use the first instance from device_grouped_conv_bwd_weight_dl_f16_instances
using InstanceTuple =
    ck::tensor_operation::device::instance::device_grouped_conv_bwd_weight_dl_f16_instances<
        2,                                     // NDimSpatial
        ck::tensor_layout::convolution::GNHWC, // ALayout (InLayout)
        ck::tensor_layout::convolution::GKYXC, // BLayout (WeiLayout)
        ck::tensor_layout::convolution::GNHWK, // ELayout (OutLayout)
        ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::Default>;

using DeviceInstance = typename std::tuple_element<0, InstanceTuple>::type;

// Expected string based on the generic instance
std::string expected_str = "DeviceGroupedConvBwdWeight_Dl"
                           "<2"                // NDimSpatial
                           ",GNHWC"            // InLayout
                           ",GKYXC"            // WeiLayout
                           ",GNHWK"            // OutLayout
                           ",fp16"             // InDataType
                           ",fp16"             // WeiDataType
                           ",fp16"             // OutDataType
                           ",fp32"             // AccDataType
                           ",PassThrough"      // InElementwiseOperation
                           ",PassThrough"      // WeiElementwiseOperation
                           ",PassThrough"      // OutElementwiseOperation
                           ",Default"          // ConvBackwardWeightSpecialization
                           ",256"              // BlockSize
                           ",128"              // MPerBlock
                           ",128"              // NPerBlock
                           ",16"               // K0PerBlock
                           ",1"                // K1
                           ",4"                // M1PerThread
                           ",4"                // N1PerThread
                           ",1"                // KPerThread
                           ",Seq(8,2)"         // M1N1ThreadClusterM1Xs
                           ",Seq(8,2)"         // M1N1ThreadClusterN1Xs
                           ",Seq(1,8,1,1,1)"   // ABlockTransferThreadSliceLengths_K0_M0_M1_K1
                           ",Seq(1,2,1,128,1)" // ABlockTransferThreadClusterLengths_K0_M0_M1_K1
                           ",Seq(0,2,3,1,4)"   // ABlockTransferThreadClusterArrangeOrder
                           ",Seq(0,2,3,1,4)"   // ABlockTransferSrcAccessOrder
                           ",Seq(1,1,1,1,1)"   // ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1
                           ",Seq(0,2,3,1,4)"   // ABlockTransferSrcVectorTensorContiguousDimOrder
                           ",Seq(1,1,1,1,1)"   // ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1
                           ",Seq(1,1,1,8,1)"   // BBlockTransferThreadSliceLengths_K0_N0_N1_K1
                           ",Seq(1,16,1,16,1)" // BBlockTransferThreadClusterLengths_K0_N0_N1_K1
                           ",Seq(0,1,4,2,3)"   // BBlockTransferThreadClusterArrangeOrder
                           ",Seq(0,1,4,2,3)"   // BBlockTransferSrcAccessOrder
                           ",Seq(1,1,1,1,1)"   // BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1
                           ",Seq(0,1,4,2,3)"   // BBlockTransferSrcVectorTensorContiguousDimOrder
                           ",Seq(1,1,1,1,1)"   // BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1
                           ",Seq(0,1,2,3,4,5)" // CThreadTransferSrcDstAccessOrder
                           ",5"                // CThreadTransferSrcDstVectorDim
                           ",1"                // CThreadTransferDstScalarPerVector
                           ">";

TEST(InstanceString, DescribeReturnsCorrectValueForBwdWeightGrpConvDl)
{
    using BaseClass = ck::tensor_operation::device::BaseOperator;
    DeviceInstance device_instance;
    BaseClass* base_ptr = &device_instance;

    auto desc = base_ptr->describe();
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->instance_string(), expected_str);
}

} // namespace
