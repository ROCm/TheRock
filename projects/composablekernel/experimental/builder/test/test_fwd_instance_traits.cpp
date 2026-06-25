// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/utility/reduction_operator.hpp"
#include "ck_tile/builder/reflect/instance_traits.hpp"
#include "ck_tile/builder/reflect/instance_traits_device_grouped_conv_fwd_dl_multiple_d_nhwc_kyxc_nhwk.hpp"
#include "ck_tile/builder/reflect/instance_traits_device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp"
#include "ck_tile/builder/reflect/instance_traits_device_grouped_conv_fwd_multiple_abd_xdl_cshuffle_v3.hpp"
#include "ck_tile/builder/reflect/instance_traits_device_grouped_conv_fwd_multiple_d_wmma_cshuffle.hpp"
#include "ck_tile/builder/reflect/instance_traits_device_grouped_conv_fwd_multiple_d_xdl_large_tensor_cshuffle.hpp"
#include "ck_tile/builder/reflect/instance_traits_tile_grouped_convolution_forward.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"

namespace {

using ::testing::ElementsAre;

// NOTE: The V3ExtractsAllFieldsCorrectly test below performs detailed field extraction testing
// for the V3 variant as a reference implementation. For new InstanceTraits specializations,
// only the instance_string() functionality needs to be tested. Each new specialization should have:
// 1. A test using instance_string<T>() directly (in this file)
// 2. A test using GetInstanceString() through base class pointer (in separate
// test_get_instance_string_*.cpp file) This prevents test duplication while ensuring both access
// methods work correctly.
TEST(InstanceTraits, V3ExtractsAllFieldsCorrectly)
{
    // Define a concrete instance type with specific template parameters
    using DeviceInstance =
        ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<
            2,                                               // NDimSpatial
            ck::tensor_layout::convolution::GNHWC,           // ALayout
            ck::tensor_layout::convolution::GKYXC,           // BLayout
            ck::Tuple<>,                                     // DsLayout
            ck::tensor_layout::convolution::GNHWK,           // ELayout
            ck::half_t,                                      // ADataType
            ck::half_t,                                      // BDataType
            float,                                           // AccDataType
            ck::half_t,                                      // CShuffleDataType
            ck::Tuple<>,                                     // DsDataType
            ck::half_t,                                      // EDataType
            ck::tensor_operation::element_wise::PassThrough, // AElementwiseOperation
            ck::tensor_operation::element_wise::PassThrough, // BElementwiseOperation
            ck::tensor_operation::element_wise::PassThrough, // CDEElementwiseOperation
            ck::tensor_operation::device::ConvolutionForwardSpecialization::
                Default,                                               // ConvForwardSpecialization
            ck::tensor_operation::device::GemmSpecialization::Default, // GemmSpec
            256,                                                       // BlockSize
            128,                                                       // MPerBlock
            128,                                                       // NPerBlock
            16,                                                        // KPerBlock
            8,                                                         // AK1
            8,                                                         // BK1
            32,                                                        // MPerXDL
            32,                                                        // NPerXDL
            4,                                                         // MXdlPerWave
            4,                                                         // NXdlPerWave
            ck::Sequence<4, 64, 1>, // ABlockTransferThreadClusterLengths_AK0_M_AK1
            ck::Sequence<1, 0, 2>,  // ABlockTransferThreadClusterArrangeOrder
            ck::Sequence<1, 0, 2>,  // ABlockTransferSrcAccessOrder
            2,                      // ABlockTransferSrcVectorDim
            8,                      // ABlockTransferSrcScalarPerVector
            8,                      // ABlockTransferDstScalarPerVector_AK1
            1,                      // ABlockLdsExtraM
            ck::Sequence<4, 64, 1>, // BBlockTransferThreadClusterLengths_BK0_N_BK1
            ck::Sequence<1, 0, 2>,  // BBlockTransferThreadClusterArrangeOrder
            ck::Sequence<1, 0, 2>,  // BBlockTransferSrcAccessOrder
            2,                      // BBlockTransferSrcVectorDim
            8,                      // BBlockTransferSrcScalarPerVector
            8,                      // BBlockTransferDstScalarPerVector_BK1
            1,                      // BBlockLdsExtraN
            1,                      // CShuffleMXdlPerWavePerShuffle
            1,                      // CShuffleNXdlPerWavePerShuffle
            ck::Sequence<1,
                         32,
                         1,
                         8>, // CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
            8,               // CDEBlockTransferScalarPerVector_NPerBlock
            ck::BlockGemmPipelineScheduler::Intrawave, // BlkGemmPipeSched
            ck::BlockGemmPipelineVersion::v1,          // BlkGemmPipelineVer
            ck::half_t,                                // AComputeDataType
            ck::half_t,                                // BComputeDataType
            false>;

    // Use InstanceTraits to extract compile-time information
    using Traits = ck_tile::reflect::InstanceTraits<DeviceInstance>;

    // Verify spatial dimension
    EXPECT_EQ(Traits::kSpatialDim, 2);

    // Verify block configuration
    EXPECT_EQ(Traits::kBlockSize, 256);
    EXPECT_EQ(Traits::kMPerBlock, 128);
    EXPECT_EQ(Traits::kNPerBlock, 128);
    EXPECT_EQ(Traits::kKPerBlock, 16);

    // Verify tuning parameters
    EXPECT_EQ(Traits::kAK1, 8);
    EXPECT_EQ(Traits::kBK1, 8);
    EXPECT_EQ(Traits::kMPerXDL, 32);
    EXPECT_EQ(Traits::kNPerXDL, 32);
    EXPECT_EQ(Traits::kMXdlPerWave, 4);
    EXPECT_EQ(Traits::kNXdlPerWave, 4);

    // Verify A block transfer parameters
    EXPECT_EQ(Traits::kABlockTransferSrcVectorDim, 2);
    EXPECT_EQ(Traits::kABlockTransferSrcScalarPerVector, 8);
    EXPECT_EQ(Traits::kABlockTransferDstScalarPerVectorK1, 8);
    EXPECT_EQ(Traits::kABlockLdsExtraM, 1);

    // Verify B block transfer parameters
    EXPECT_EQ(Traits::kBBlockTransferSrcVectorDim, 2);
    EXPECT_EQ(Traits::kBBlockTransferSrcScalarPerVector, 8);
    EXPECT_EQ(Traits::kBBlockTransferDstScalarPerVectorK1, 8);
    EXPECT_EQ(Traits::kBBlockLdsExtraN, 1);

    // Verify C shuffle parameters
    EXPECT_EQ(Traits::kCShuffleMXdlPerWavePerShuffle, 1);
    EXPECT_EQ(Traits::kCShuffleNXdlPerWavePerShuffle, 1);
    EXPECT_EQ(Traits::kCBlockTransferScalarPerVector, 8);

    // Verify pipeline configuration
    EXPECT_EQ(Traits::kPipelineScheduler, ck::BlockGemmPipelineScheduler::Intrawave);
    EXPECT_EQ(Traits::kPipelineVersion, ck::BlockGemmPipelineVersion::v1);

    // Verify data types using std::is_same
    EXPECT_TRUE((std::is_same<Traits::ADataType, ck::half_t>::value));
    EXPECT_TRUE((std::is_same<Traits::BDataType, ck::half_t>::value));
    EXPECT_TRUE((std::is_same<Traits::AccDataType, float>::value));
    EXPECT_TRUE((std::is_same<Traits::EDataType, ck::half_t>::value));

    // Verify layout types
    EXPECT_TRUE((std::is_same<Traits::ALayout, ck::tensor_layout::convolution::GNHWC>::value));
    EXPECT_TRUE((std::is_same<Traits::BLayout, ck::tensor_layout::convolution::GKYXC>::value));
    EXPECT_TRUE((std::is_same<Traits::ELayout, ck::tensor_layout::convolution::GNHWK>::value));

    // Verify all array values for thread cluster lengths using googlemock matchers
    EXPECT_THAT(Traits::kAThreadClusterLengths, ElementsAre(4, 64, 1));
    EXPECT_THAT(Traits::kBThreadClusterLengths, ElementsAre(4, 64, 1));
    EXPECT_THAT(Traits::kCThreadClusterLengths, ElementsAre(1, 32, 1, 8));

    // Verify A block transfer arrange order and access order arrays
    EXPECT_THAT(Traits::kAThreadClusterArrangeOrder, ElementsAre(1, 0, 2));
    EXPECT_THAT(Traits::kABlockTransferSrcAccessOrder, ElementsAre(1, 0, 2));

    // Verify B block transfer arrange order and access order arrays
    EXPECT_THAT(Traits::kBThreadClusterArrangeOrder, ElementsAre(1, 0, 2));
    EXPECT_THAT(Traits::kBBlockTransferSrcAccessOrder, ElementsAre(1, 0, 2));

    // Verify additional data types
    EXPECT_TRUE((std::is_same<Traits::CShuffleDataType, ck::half_t>::value));
    EXPECT_TRUE((std::is_same<Traits::DsDataType, ck::Tuple<>>::value));
    EXPECT_TRUE((std::is_same<Traits::AComputeDataType, ck::half_t>::value));
    EXPECT_TRUE((std::is_same<Traits::BComputeDataType, ck::half_t>::value));

    // Verify additional layout types
    EXPECT_TRUE((std::is_same<Traits::DsLayout, ck::Tuple<>>::value));

    // Verify element-wise operations
    EXPECT_TRUE((std::is_same<Traits::AElementwiseOperation,
                              ck::tensor_operation::element_wise::PassThrough>::value));
    EXPECT_TRUE((std::is_same<Traits::BElementwiseOperation,
                              ck::tensor_operation::element_wise::PassThrough>::value));
    EXPECT_TRUE((std::is_same<Traits::CDEElementwiseOperation,
                              ck::tensor_operation::element_wise::PassThrough>::value));
}

TEST(InstanceTraits, V3InstanceStringReturnsCorrectFormat)
{
    // Define a concrete instance type with specific template parameters
    using DeviceInstance =
        ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<
            2,                                               // NDimSpatial
            ck::tensor_layout::convolution::GNHWC,           // ALayout
            ck::tensor_layout::convolution::GKYXC,           // BLayout
            ck::Tuple<>,                                     // DsLayout
            ck::tensor_layout::convolution::GNHWK,           // ELayout
            ck::half_t,                                      // ADataType
            ck::half_t,                                      // BDataType
            float,                                           // AccDataType
            ck::half_t,                                      // CShuffleDataType
            ck::Tuple<>,                                     // DsDataType
            ck::half_t,                                      // EDataType
            ck::tensor_operation::element_wise::PassThrough, // AElementwiseOperation
            ck::tensor_operation::element_wise::PassThrough, // BElementwiseOperation
            ck::tensor_operation::element_wise::PassThrough, // CDEElementwiseOperation
            ck::tensor_operation::device::ConvolutionForwardSpecialization::
                Default,                                               // ConvForwardSpecialization
            ck::tensor_operation::device::GemmSpecialization::Default, // GemmSpec
            256,                                                       // BlockSize
            128,                                                       // MPerBlock
            128,                                                       // NPerBlock
            16,                                                        // KPerBlock
            8,                                                         // AK1
            8,                                                         // BK1
            32,                                                        // MPerXDL
            32,                                                        // NPerXDL
            4,                                                         // MXdlPerWave
            4,                                                         // NXdlPerWave
            ck::Sequence<4, 64, 1>, // ABlockTransferThreadClusterLengths_AK0_M_AK1
            ck::Sequence<1, 0, 2>,  // ABlockTransferThreadClusterArrangeOrder
            ck::Sequence<1, 0, 2>,  // ABlockTransferSrcAccessOrder
            2,                      // ABlockTransferSrcVectorDim
            8,                      // ABlockTransferSrcScalarPerVector
            8,                      // ABlockTransferDstScalarPerVector_AK1
            1,                      // ABlockLdsExtraM
            ck::Sequence<4, 64, 1>, // BBlockTransferThreadClusterLengths_BK0_N_BK1
            ck::Sequence<1, 0, 2>,  // BBlockTransferThreadClusterArrangeOrder
            ck::Sequence<1, 0, 2>,  // BBlockTransferSrcAccessOrder
            2,                      // BBlockTransferSrcVectorDim
            8,                      // BBlockTransferSrcScalarPerVector
            8,                      // BBlockTransferDstScalarPerVector_BK1
            1,                      // BBlockLdsExtraN
            1,                      // CShuffleMXdlPerWavePerShuffle
            1,                      // CShuffleNXdlPerWavePerShuffle
            ck::Sequence<1,
                         32,
                         1,
                         8>, // CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
            8,               // CDEBlockTransferScalarPerVector_NPerBlock
            ck::BlockGemmPipelineScheduler::Intrawave, // BlkGemmPipeSched
            ck::BlockGemmPipelineVersion::v1,          // BlkGemmPipelineVer
            ck::half_t,                                // AComputeDataType
            ck::half_t,                                // BComputeDataType
            false>;                                    // DirectLoad

    std::string instance_str = ck_tile::reflect::instance_string<DeviceInstance>();

    std::string expected_str = "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3"
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
                               ",Default"       // GemmSpec
                               ",256"           // BlockSize
                               ",128"           // MPerBlock
                               ",128"           // NPerBlock
                               ",16"            // KPerBlock
                               ",8"             // AK1
                               ",8"             // BK1
                               ",32"            // MPerXDL
                               ",32"            // NPerXDL
                               ",4"             // MXdlPerWave
                               ",4"             // NXdlPerWave
                               ",Seq(4,64,1)"   // ABlockTransferThreadClusterLengths
                               ",Seq(1,0,2)"    // ABlockTransferThreadClusterArrangeOrder
                               ",Seq(1,0,2)"    // ABlockTransferSrcAccessOrder
                               ",2"             // ABlockTransferSrcVectorDim
                               ",8"             // ABlockTransferSrcScalarPerVector
                               ",8"             // ABlockTransferDstScalarPerVector_AK1
                               ",true"          // ABlockLdsExtraM
                               ",Seq(4,64,1)"   // BBlockTransferThreadClusterLengths
                               ",Seq(1,0,2)"    // BBlockTransferThreadClusterArrangeOrder
                               ",Seq(1,0,2)"    // BBlockTransferSrcAccessOrder
                               ",2"             // BBlockTransferSrcVectorDim
                               ",8"             // BBlockTransferSrcScalarPerVector
                               ",8"             // BBlockTransferDstScalarPerVector_BK1
                               ",true"          // BBlockLdsExtraN
                               ",1"             // CShuffleMXdlPerWavePerShuffle
                               ",1"             // CShuffleNXdlPerWavePerShuffle
                               ",Seq(1,32,1,8)" // CDEBlockTransferClusterLengths
                               ",8"             // CDEBlockTransferScalarPerVector_NPerBlock
                               ",Intrawave"     // BlkGemmPipeSched
                               ",v1"            // BlkGemmPipelineVer
                               ",fp16"          // AComputeDataType
                               ",fp16"          // BComputeDataType
                               ",false>";       // DirectLoad

    EXPECT_EQ(instance_str, expected_str);
}

TEST(InstanceTraits, BaseInstanceStringReturnsCorrectFormat)
{
    using DeviceInstance =
        ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<
            2,                                               // NDimSpatial
            ck::tensor_layout::convolution::GNHWC,           // ALayout
            ck::tensor_layout::convolution::GKYXC,           // BLayout
            ck::Tuple<>,                                     // DsLayout
            ck::tensor_layout::convolution::GNHWK,           // ELayout
            ck::half_t,                                      // ADataType
            ck::half_t,                                      // BDataType
            float,                                           // AccDataType
            ck::half_t,                                      // CShuffleDataType
            ck::Tuple<>,                                     // DsDataType
            ck::half_t,                                      // EDataType
            ck::tensor_operation::element_wise::PassThrough, // AElementwiseOperation
            ck::tensor_operation::element_wise::PassThrough, // BElementwiseOperation
            ck::tensor_operation::element_wise::PassThrough, // CDEElementwiseOperation
            ck::tensor_operation::device::ConvolutionForwardSpecialization::
                Default,                                               // ConvForwardSpecialization
            ck::tensor_operation::device::GemmSpecialization::Default, // GemmSpec
            1,                                                         // NumGemmKPrefetchStage
            256,                                                       // BlockSize
            128,                                                       // MPerBlock
            128,                                                       // NPerBlock
            16,                                                        // KPerBlock
            8,                                                         // AK1
            8,                                                         // BK1
            32,                                                        // MPerXDL
            32,                                                        // NPerXDL
            4,                                                         // MXdlPerWave
            4,                                                         // NXdlPerWave
            ck::Sequence<4, 64, 1>, // ABlockTransferThreadClusterLengths_AK0_M_AK1
            ck::Sequence<1, 0, 2>,  // ABlockTransferThreadClusterArrangeOrder
            ck::Sequence<1, 0, 2>,  // ABlockTransferSrcAccessOrder
            2,                      // ABlockTransferSrcVectorDim
            8,                      // ABlockTransferSrcScalarPerVector
            8,                      // ABlockTransferDstScalarPerVector_AK1
            1,                      // ABlockLdsExtraM
            ck::Sequence<4, 64, 1>, // BBlockTransferThreadClusterLengths_BK0_N_BK1
            ck::Sequence<1, 0, 2>,  // BBlockTransferThreadClusterArrangeOrder
            ck::Sequence<1, 0, 2>,  // BBlockTransferSrcAccessOrder
            2,                      // BBlockTransferSrcVectorDim
            8,                      // BBlockTransferSrcScalarPerVector
            8,                      // BBlockTransferDstScalarPerVector_BK1
            1,                      // BBlockLdsExtraN
            1,                      // CShuffleMXdlPerWavePerShuffle
            1,                      // CShuffleNXdlPerWavePerShuffle
            ck::Sequence<1,
                         32,
                         1,
                         8>, // CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
            8,               // CDEBlockTransferScalarPerVector_NPerBlock
            ck::half_t,      // AComputeDataType
            ck::half_t,      // BComputeDataType
            ck::LoopScheduler::Default, // LoopSched
            1>;                         // NumGroupsToMerge

    std::string instance_str = ck_tile::reflect::instance_string<DeviceInstance>();

    std::string expected_str = "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle"
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
                               ",Default"       // GemmSpec
                               ",1"             // NumGemmKPrefetchStage
                               ",256"           // BlockSize
                               ",128"           // MPerBlock
                               ",128"           // NPerBlock
                               ",16"            // KPerBlock
                               ",8"             // AK1
                               ",8"             // BK1
                               ",32"            // MPerXDL
                               ",32"            // NPerXDL
                               ",4"             // MXdlPerWave
                               ",4"             // NXdlPerWave
                               ",Seq(4,64,1)"   // ABlockTransferThreadClusterLengths
                               ",Seq(1,0,2)"    // ABlockTransferThreadClusterArrangeOrder
                               ",Seq(1,0,2)"    // ABlockTransferSrcAccessOrder
                               ",2"             // ABlockTransferSrcVectorDim
                               ",8"             // ABlockTransferSrcScalarPerVector
                               ",8"             // ABlockTransferDstScalarPerVector_AK1
                               ",true"          // ABlockLdsExtraM
                               ",Seq(4,64,1)"   // BBlockTransferThreadClusterLengths
                               ",Seq(1,0,2)"    // BBlockTransferThreadClusterArrangeOrder
                               ",Seq(1,0,2)"    // BBlockTransferSrcAccessOrder
                               ",2"             // BBlockTransferSrcVectorDim
                               ",8"             // BBlockTransferSrcScalarPerVector
                               ",8"             // BBlockTransferDstScalarPerVector_BK1
                               ",true"          // BBlockLdsExtraN
                               ",1"             // CShuffleMXdlPerWavePerShuffle
                               ",1"             // CShuffleNXdlPerWavePerShuffle
                               ",Seq(1,32,1,8)" // CDEBlockTransferClusterLengths
                               ",8"             // CDEBlockTransferScalarPerVector_NPerBlock
                               ",fp16"          // AComputeDataType
                               ",fp16"          // BComputeDataType
                               ",Default"       // LoopSched
                               ",1>";           // NumGroupsToMerge

    EXPECT_EQ(instance_str, expected_str);
}

TEST(InstanceTraits, LargeTensorInstanceStringReturnsCorrectFormat)
{
    using DeviceInstance =
        ck::tensor_operation::device::DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor<
            2,                                               // NDimSpatial
            ck::tensor_layout::convolution::GNHWC,           // ALayout
            ck::tensor_layout::convolution::GKYXC,           // BLayout
            ck::Tuple<>,                                     // DsLayout
            ck::tensor_layout::convolution::GNHWK,           // ELayout
            ck::half_t,                                      // ADataType
            ck::half_t,                                      // BDataType
            float,                                           // AccDataType
            ck::half_t,                                      // CShuffleDataType
            ck::Tuple<>,                                     // DsDataType
            ck::half_t,                                      // EDataType
            ck::tensor_operation::element_wise::PassThrough, // AElementwiseOperation
            ck::tensor_operation::element_wise::PassThrough, // BElementwiseOperation
            ck::tensor_operation::element_wise::PassThrough, // CDEElementwiseOperation
            ck::tensor_operation::device::ConvolutionForwardSpecialization::
                Default,                                               // ConvForwardSpecialization
            ck::tensor_operation::device::GemmSpecialization::Default, // GemmSpec
            1,                                                         // NumGemmKPrefetchStage
            256,                                                       // BlockSize
            128,                                                       // MPerBlock
            128,                                                       // NPerBlock
            16,                                                        // KPerBlock
            8,                                                         // AK1
            8,                                                         // BK1
            32,                                                        // MPerXDL
            32,                                                        // NPerXDL
            4,                                                         // MXdlPerWave
            4,                                                         // NXdlPerWave
            ck::Sequence<4, 64, 1>, // ABlockTransferThreadClusterLengths_AK0_M_AK1
            ck::Sequence<1, 0, 2>,  // ABlockTransferThreadClusterArrangeOrder
            ck::Sequence<1, 0, 2>,  // ABlockTransferSrcAccessOrder
            2,                      // ABlockTransferSrcVectorDim
            8,                      // ABlockTransferSrcScalarPerVector
            8,                      // ABlockTransferDstScalarPerVector_AK1
            1,                      // ABlockLdsExtraM
            ck::Sequence<4, 64, 1>, // BBlockTransferThreadClusterLengths_BK0_N_BK1
            ck::Sequence<1, 0, 2>,  // BBlockTransferThreadClusterArrangeOrder
            ck::Sequence<1, 0, 2>,  // BBlockTransferSrcAccessOrder
            2,                      // BBlockTransferSrcVectorDim
            8,                      // BBlockTransferSrcScalarPerVector
            8,                      // BBlockTransferDstScalarPerVector_BK1
            1,                      // BBlockLdsExtraN
            1,                      // CShuffleMXdlPerWavePerShuffle
            1,                      // CShuffleNXdlPerWavePerShuffle
            ck::Sequence<1,
                         32,
                         1,
                         8>,             // CDEBlockTransferClusterLengths
            8,                           // CDEBlockTransferScalarPerVector_NPerBlock
            ck::half_t,                  // AComputeDataType
            ck::half_t,                  // BComputeDataType
            ck::LoopScheduler::Default>; // LoopSched

    // Generate instance string
    std::string instance_str = ck_tile::reflect::instance_string<DeviceInstance>();

    // Expected string with all 48 template parameters
    std::string expected_str = "DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor"
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
                               ",Default"       // GemmSpec
                               ",1"             // NumGemmKPrefetchStage
                               ",256"           // BlockSize
                               ",128"           // MPerBlock
                               ",128"           // NPerBlock
                               ",16"            // KPerBlock
                               ",8"             // AK1
                               ",8"             // BK1
                               ",32"            // MPerXDL
                               ",32"            // NPerXDL
                               ",4"             // MXdlPerWave
                               ",4"             // NXdlPerWave
                               ",Seq(4,64,1)"   // ABlockTransferThreadClusterLengths
                               ",Seq(1,0,2)"    // ABlockTransferThreadClusterArrangeOrder
                               ",Seq(1,0,2)"    // ABlockTransferSrcAccessOrder
                               ",2"             // ABlockTransferSrcVectorDim
                               ",8"             // ABlockTransferSrcScalarPerVector
                               ",8"             // ABlockTransferDstScalarPerVector_AK1
                               ",true"          // ABlockLdsExtraM
                               ",Seq(4,64,1)"   // BBlockTransferThreadClusterLengths
                               ",Seq(1,0,2)"    // BBlockTransferThreadClusterArrangeOrder
                               ",Seq(1,0,2)"    // BBlockTransferSrcAccessOrder
                               ",2"             // BBlockTransferSrcVectorDim
                               ",8"             // BBlockTransferSrcScalarPerVector
                               ",8"             // BBlockTransferDstScalarPerVector_BK1
                               ",true"          // BBlockLdsExtraN
                               ",1"             // CShuffleMXdlPerWavePerShuffle
                               ",1"             // CShuffleNXdlPerWavePerShuffle
                               ",Seq(1,32,1,8)" // CDEBlockTransferClusterLengths
                               ",8"             // CDEBlockTransferScalarPerVector_NPerBlock
                               ",fp16"          // AComputeDataType
                               ",fp16"          // BComputeDataType
                               ",Default>";     // LoopSched

    // Verify the generated string matches exactly
    EXPECT_EQ(instance_str, expected_str);
}

TEST(InstanceTraits, WmmaInstanceStringReturnsCorrectFormat)
{
    using DeviceInstance =
        ck::tensor_operation::device::DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<
            2,                                               // NDimSpatial
            ck::tensor_layout::convolution::GNHWC,           // ALayout
            ck::tensor_layout::convolution::GKYXC,           // BLayout
            ck::Tuple<>,                                     // DsLayout
            ck::tensor_layout::convolution::GNHWK,           // ELayout
            ck::half_t,                                      // ADataType
            ck::half_t,                                      // BDataType
            float,                                           // AccDataType
            ck::half_t,                                      // CShuffleDataType
            ck::Tuple<>,                                     // DsDataType
            ck::half_t,                                      // EDataType
            ck::tensor_operation::element_wise::PassThrough, // AElementwiseOperation
            ck::tensor_operation::element_wise::PassThrough, // BElementwiseOperation
            ck::tensor_operation::element_wise::PassThrough, // CDEElementwiseOperation
            ck::tensor_operation::device::ConvolutionForwardSpecialization::
                Default, // ConvForwardSpecialization
            ck::tensor_operation::device::GemmSpecialization::MNKPadding, // GemmSpec
            1,                                                            // NumGemmKPrefetchStage
            128,                                                          // BlockSize
            64,                                                           // MPerBlock
            64,                                                           // NPerBlock
            32,                                                           // KPerBlock
            8,                                                            // K1
            16,                                                           // MPerWmma
            16,                                                           // NPerWmma
            2,                                                            // MRepeat
            2,                                                            // NRepeat
            ck::Sequence<4, 32, 1>, // ABlockTransferThreadClusterLengths_AK0_M_AK1
            ck::Sequence<1, 0, 2>,  // ABlockTransferThreadClusterArrangeOrder
            ck::Sequence<1, 0, 2>,  // ABlockTransferSrcAccessOrder
            2,                      // ABlockTransferSrcVectorDim
            1,                      // ABlockTransferSrcScalarPerVector
            8,                      // ABlockTransferDstScalarPerVector_AK1
            1,                      // ABlockLdsExtraM
            ck::Sequence<4, 32, 1>, // BBlockTransferThreadClusterLengths_BK0_N_BK1
            ck::Sequence<1, 0, 2>,  // BBlockTransferThreadClusterArrangeOrder
            ck::Sequence<1, 0, 2>,  // BBlockTransferSrcAccessOrder
            2,                      // BBlockTransferSrcVectorDim
            1,                      // BBlockTransferSrcScalarPerVector
            8,                      // BBlockTransferDstScalarPerVector_BK1
            1,                      // BBlockLdsExtraN
            1,                      // CShuffleMRepeatPerShuffle
            1,                      // CShuffleNRepeatPerShuffle
            ck::Sequence<1,
                         32,
                         1,
                         4>,            // CDEShuffleBlockTransferClusterLengths
            1,                          // CDEShuffleBlockTransferScalarPerVector_NPerBlock
            ck::LoopScheduler::Default, // LoopSched
            ck::PipelineVersion::v1>;   // PipelineVer

    // Generate instance string
    std::string instance_str = ck_tile::reflect::instance_string<DeviceInstance>();

    // Expected string with all 46 template parameters
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

    // Verify the generated string matches exactly
    EXPECT_EQ(instance_str, expected_str);
}

TEST(InstanceTraits, DlInstanceStringReturnsCorrectFormat)
{
    using DeviceInstance =
        ck::tensor_operation::device::DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK<
            2,                                               // NDimSpatial
            ck::half_t,                                      // ADataType
            ck::half_t,                                      // BDataType
            ck::Tuple<>,                                     // DsDataType
            ck::half_t,                                      // EDataType
            float,                                           // AccDataType
            ck::tensor_layout::convolution::GNHWC,           // ALayout
            ck::tensor_layout::convolution::GKYXC,           // BLayout
            ck::Tuple<>,                                     // DsLayout
            ck::tensor_layout::convolution::GNHWK,           // ELayout
            ck::tensor_operation::element_wise::PassThrough, // AElementwiseOperation
            ck::tensor_operation::element_wise::PassThrough, // BElementwiseOperation
            ck::tensor_operation::element_wise::PassThrough, // CDEElementwiseOperation
            ck::tensor_operation::device::ConvolutionForwardSpecialization::
                Default, // ConvForwardSpecialization
            ck::tensor_operation::device::GemmSpecialization::MNKPadding, // GemmSpec
            8,                                                            // BlockSize
            16,                                                           // MPerBlock
            4,                                                            // NPerBlock
            2,                                                            // K0PerBlock
            1,                                                            // K1
            1,                                                            // M1PerThread
            2,                                                            // N1PerThread
            1,                                                            // KPerThread
            ck::Sequence<4, 2>,                                           // M1N1ThreadClusterM1Xs
            ck::Sequence<1, 1>,                                           // M1N1ThreadClusterN1Xs
            ck::Sequence<2, 1, 2, 1>,       // ABlockTransferThreadSliceLengths_K0_M0_M1_K1
            ck::Sequence<1, 1, 8, 1>,       // ABlockTransferThreadClusterLengths_K0_M0_M1_K1
            ck::Sequence<1, 2, 0, 3>,       // ABlockTransferThreadClusterArrangeOrder
            ck::Sequence<1, 2, 0, 3>,       // ABlockTransferSrcAccessOrder
            ck::Sequence<1, 1, 1, 1>,       // ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1
            ck::Sequence<1, 2, 0, 3>,       // ABlockTransferSrcVectorTensorContiguousDimOrder
            ck::Sequence<1, 1, 1, 1>,       // ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1
            ck::Sequence<1, 1, 1, 1>,       // BBlockTransferThreadSliceLengths_K0_N0_N1_K1
            ck::Sequence<2, 1, 4, 1>,       // BBlockTransferThreadClusterLengths_K0_N0_N1_K1
            ck::Sequence<1, 2, 0, 3>,       // BBlockTransferThreadClusterArrangeOrder
            ck::Sequence<1, 2, 0, 3>,       // BBlockTransferSrcAccessOrder
            ck::Sequence<1, 1, 1, 1>,       // BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1
            ck::Sequence<1, 2, 0, 3>,       // BBlockTransferSrcVectorTensorContiguousDimOrder
            ck::Sequence<1, 1, 1, 1>,       // BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1
            ck::Sequence<0, 1, 2, 3, 4, 5>, // CThreadTransferSrcDstAccessOrder
            5,                              // CThreadTransferSrcDstVectorDim
            1>;                             // CThreadTransferDstScalarPerVector

    // Generate instance string
    std::string instance_str = ck_tile::reflect::instance_string<DeviceInstance>();

    // Expected string with all 42 template parameters
    std::string expected_str = "DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK"
                               "<2"            // NDimSpatial
                               ",fp16"         // ADataType
                               ",fp16"         // BDataType
                               ",EmptyTuple"   // DsDataType
                               ",fp16"         // EDataType
                               ",fp32"         // AccDataType
                               ",GNHWC"        // ALayout
                               ",GKYXC"        // BLayout
                               ",EmptyTuple"   // DsLayout
                               ",GNHWK"        // ELayout
                               ",PassThrough"  // AElementwiseOperation
                               ",PassThrough"  // BElementwiseOperation
                               ",PassThrough"  // CDEElementwiseOperation
                               ",Default"      // ConvForwardSpecialization
                               ",MNKPadding"   // GemmSpec
                               ",8"            // BlockSize
                               ",16"           // MPerBlock
                               ",4"            // NPerBlock
                               ",2"            // K0PerBlock
                               ",1"            // K1
                               ",1"            // M1PerThread
                               ",2"            // N1PerThread
                               ",1"            // KPerThread
                               ",Seq(4,2)"     // M1N1ThreadClusterM1Xs
                               ",Seq(1,1)"     // M1N1ThreadClusterN1Xs
                               ",Seq(2,1,2,1)" // ABlockTransferThreadSliceLengths_K0_M0_M1_K1
                               ",Seq(1,1,8,1)" // ABlockTransferThreadClusterLengths_K0_M0_M1_K1
                               ",Seq(1,2,0,3)" // ABlockTransferThreadClusterArrangeOrder
                               ",Seq(1,2,0,3)" // ABlockTransferSrcAccessOrder
                               ",Seq(1,1,1,1)" // ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1
                               ",Seq(1,2,0,3)" // ABlockTransferSrcVectorTensorContiguousDimOrder
                               ",Seq(1,1,1,1)" // ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1
                               ",Seq(1,1,1,1)" // BBlockTransferThreadSliceLengths_K0_N0_N1_K1
                               ",Seq(2,1,4,1)" // BBlockTransferThreadClusterLengths_K0_N0_N1_K1
                               ",Seq(1,2,0,3)" // BBlockTransferThreadClusterArrangeOrder
                               ",Seq(1,2,0,3)" // BBlockTransferSrcAccessOrder
                               ",Seq(1,1,1,1)" // BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1
                               ",Seq(1,2,0,3)" // BBlockTransferSrcVectorTensorContiguousDimOrder
                               ",Seq(1,1,1,1)" // BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1
                               ",Seq(0,1,2,3,4,5)" // CThreadTransferSrcDstAccessOrder
                               ",5"                // CThreadTransferSrcDstVectorDim
                               ",1>";              // CThreadTransferDstScalarPerVector

    // Verify the generated string matches exactly
    EXPECT_EQ(instance_str, expected_str);
}

TEST(InstanceTraits, TileInstanceStringReturnsCorrectFormat)
{
    using GroupedConvTraitsType =
        ck_tile::GroupedConvTraits<2 /*NDimSpatial*/,
                                   ck_tile::ConvolutionSpecialization::Default /*ConvSpec*/,
                                   ck_tile::tensor_layout::convolution::NHWGC /*InLayout*/,
                                   ck_tile::tensor_layout::convolution::GKYXC /*WeiLayout*/,
                                   ck_tile::tuple<> /*DsLayout*/,
                                   ck_tile::tensor_layout::convolution::NHWGK /*OutLayout*/,
                                   4 /*VectorSizeA*/,
                                   4 /*VectorSizeB*/,
                                   4 /*VectorSizeC*/,
                                   1 /*NumGroupsToMerge*/,
                                   false /*EnableSplitImage*/,
                                   false /*ExplicitGemm*/>;

    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<128 /*M_Tile*/, 128 /*N_Tile*/, 32 /*K_Tile*/>,
        ck_tile::sequence<4 /*M_Warp*/, 1 /*N_Warp*/, 1 /*K_Warp*/>,
        ck_tile::sequence<16 /*M_Warp_Tile*/, 16 /*N_Warp_Tile*/, 16 /*K_Warp_Tile*/>>;

    using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<
        GemmShape,
        GroupedConvTraitsType::FixedGemmParams::TilePartitionerGroupNum,
        GroupedConvTraitsType::FixedGemmParams::TilePartitionerM01>;

    using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<
        GroupedConvTraitsType::FixedGemmParams::kPadM,
        GroupedConvTraitsType::FixedGemmParams::kPadN,
        GroupedConvTraitsType::FixedGemmParams::kPadK,
        false /*DoubleSmemBuffer*/,
        typename GroupedConvTraitsType::AsLayoutFwd,
        typename GroupedConvTraitsType::BsLayoutFwd,
        typename GroupedConvTraitsType::CLayoutFwd,
        GroupedConvTraitsType::FixedGemmParams::TransposeC,
        GroupedConvTraitsType::FixedGemmParams::UseStructuredSparsity,
        GroupedConvTraitsType::FixedGemmParams::Persistent,
        1 /*NumWaveGroups*/>;

    using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<
        ck_tile::bf16_t /*InDataType*/,
        ck_tile::bf16_t /*WeiDataType*/,
        float /*AccDataType*/,
        GemmShape,
        GemmUniversalTraits,
        ck_tile::GemmPipelineScheduler::Intrawave /*scheduler*/,
        ck_tile::element_wise::PassThrough /*AElementwiseOperation*/,
        ck_tile::element_wise::PassThrough /*BElementwiseOperation*/,
        ck_tile::bf16_t /*OutDataType*/,
        GroupedConvTraitsType::FixedGemmParams::FixedVectorSize,
        GroupedConvTraitsType::VectorSizeA,
        GroupedConvTraitsType::VectorSizeB>;

    using GemmPipeline = typename ck_tile::GemmPipelineAgBgCrCompV3<UniversalGemmProblem>;

    using ConvEpilogue = ck_tile::CShuffleEpilogue<
        ck_tile::CShuffleEpilogueProblem<ck_tile::bf16_t /*InDataType*/,
                                         ck_tile::bf16_t /*WeiDataType*/,
                                         ck_tile::tuple<> /*DsDataType*/,
                                         float /*AccDataType*/,
                                         ck_tile::bf16_t /*OutDataType*/,
                                         typename GroupedConvTraitsType::ImplicitGemmDsLayout,
                                         typename GroupedConvTraitsType::FixedGemmParams::ELayout,
                                         ck_tile::element_wise::PassThrough /*CDElementWise*/,
                                         128 /*MPerBlock*/,
                                         128 /*NPerBlock*/,
                                         4 /*M_Warp*/,
                                         1 /*N_Warp*/,
                                         16 /*M_Warp_Tile*/,
                                         16 /*N_Warp_Tile*/,
                                         16 /*K_Warp_Tile*/,
                                         GroupedConvTraitsType::FixedGemmParams::TransposeC,
                                         1 /*kNumWaveGroups*/,
                                         GroupedConvTraitsType::FixedGemmParams::FixedVectorSize,
                                         GroupedConvTraitsType::VectorSizeC>>;

    using GroupedConvFwdKernel = ck_tile::GroupedConvolutionForwardKernel<GroupedConvTraitsType,
                                                                          TilePartitioner,
                                                                          GemmPipeline,
                                                                          ConvEpilogue>;

    std::string instance_str = ck_tile::reflect::instance_string<GroupedConvFwdKernel>();

    std::string expected_str = "GroupedConvolutionForwardKernel"
                               "<2"           // NDimSpatial
                               ",Default"     // ConvSpecialization
                               ",NHWGC"       // InLayout
                               ",GKYXC"       // WeiLayout
                               ",EmptyTuple"  // DsLayout
                               ",NHWGK"       // OutLayout
                               ",4"           // VectorSizeA
                               ",4"           // VectorSizeB
                               ",4"           // VectorSizeC
                               ",1"           // NumGroupsToMerge
                               ",0"           // EnableSplitImage
                               ",0"           // ExplicitGemm
                               ",128"         // MPerBlock
                               ",128"         // NPerBlock
                               ",32"          // KPerBlock
                               ",4"           // MWarp
                               ",1"           // NWarp
                               ",1"           // KWarp
                               ",16"          // MWarpTile
                               ",16"          // NWarpTile
                               ",16"          // KWarpTile
                               ",bf16"        // ADataType
                               ",bf16"        // BDataType
                               ",COMPUTE_V3"  // BlkGemmPipelineVer
                               ",Intrawave"   // BlkGemmPipeSched
                               ",0"           // DoubleSmemBuffer
                               ",1"           // NumWaveGroups
                               ",fp32"        // AccDataType
                               ",bf16"        // EDataType
                               ",EmptyTuple"  // DsDataType
                               ",PassThrough" // CDEElementwiseOperation
                               ">";

    EXPECT_EQ(instance_str, expected_str);
}

} // anonymous namespace
