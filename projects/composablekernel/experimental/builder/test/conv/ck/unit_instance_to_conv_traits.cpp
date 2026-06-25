// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// ============================================================================
// Unit Tests for InstanceTraits to ConvTraits Conversion
// ============================================================================
//
// PURPOSE:
// --------
// These tests verify the conversion layer between InstanceTraits (low-level
// template parameter extraction) and ConvTraits (high-level semantic traits).
// The conversion transforms raw CK kernel parameters into builder-friendly
// enums and structures.
//
// DESIGN RATIONALE:
// -----------------
// ConvTraits uses a single generic specialization that works with any Device
// class satisfying the IsXdlFwdConv concept. This use of concepts is fragile
// and introduces extra complexity. We want to refector to just use functions
// for this conversion.
//
// These tests are intentionally verbose and repetitive to provide maximum
// coverage during refactoring. Once the refactoring is complete and stable,
// they can be simplified or consolidated.
//
// TEST COVERAGE:
// --------------
// 1. Enum conversion functions (pipeline version, scheduler, etc.)
// 2. Signature extraction (direction, specialization, layout, data type)
// 3. Full transformation verification for each XDL Device class template:
//    - DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3
//    - DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle
//    - DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor
//
// NOTE: WMMA and DL (Direct Load) variants are not covered as they don't
// satisfy the IsXdlFwdConv concept (different tile parameter structure).
// ============================================================================

#include "ck/utility/scheduler_enum.hpp"
#include "ck_tile/builder/types.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>
#include <ck_tile/builder/reflect/conv_traits.hpp>
#include <ck_tile/builder/reflect/instance_traits_device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp>
#include <ck_tile/builder/reflect/instance_traits_device_grouped_conv_fwd_multiple_abd_xdl_cshuffle_v3.hpp>
#include <ck_tile/builder/reflect/instance_traits_device_grouped_conv_fwd_multiple_d_xdl_large_tensor_cshuffle.hpp>

namespace {

using ck_tile::builder::ConvDirection;
using ck_tile::builder::DataType;
using ck_tile::builder::ElementwiseOperation;
using ck_tile::builder::GemmPadding;
using ck_tile::builder::PipelineScheduler;
using ck_tile::builder::PipelineVersion;
using ck_tile::builder::TensorLayout;
using ::testing::ElementsAre;

// ============================================================================
// Test Enum Conversion Functions
// ============================================================================

TEST(InstanceToConvTraits, ConvertsBlockGemmPipelineVersion)
{
    using ck_tile::reflect::conv::convert_pipeline_version;
    using enum ::ck::BlockGemmPipelineVersion;
    using enum ::ck_tile::builder::PipelineVersion;

    EXPECT_EQ(convert_pipeline_version<v1>(), V1);
    EXPECT_EQ(convert_pipeline_version<v2>(), V2);
    EXPECT_EQ(convert_pipeline_version<v3>(), V3);
    EXPECT_EQ(convert_pipeline_version<v4>(), V4);
    EXPECT_EQ(convert_pipeline_version<v5>(), V5);
}

TEST(InstanceToConvTraits, ConvertsPipelineVersion)
{
    using ck_tile::reflect::conv::convert_pipeline_version;
    using enum ck::PipelineVersion;
    using enum PipelineVersion;

    EXPECT_EQ(convert_pipeline_version<v1>(), V1);
    EXPECT_EQ(convert_pipeline_version<v2>(), V2);
    EXPECT_EQ(convert_pipeline_version<v4>(), V4);
    EXPECT_EQ(convert_pipeline_version<weight_only>(), WEIGHT_ONLY);
}

TEST(InstanceToConvTraits, ConvertsBlockGemmPipelineScheduler)
{
    using ck_tile::reflect::conv::convert_pipeline_scheduler;
    using enum ck::BlockGemmPipelineScheduler;
    using enum PipelineScheduler;

    EXPECT_EQ(convert_pipeline_scheduler<Intrawave>(), INTRAWAVE);
    EXPECT_EQ(convert_pipeline_scheduler<Interwave>(), INTERWAVE);
}

TEST(InstanceToConvTraits, ConvertsLoopScheduler)
{
    using ck_tile::reflect::conv::convert_pipeline_scheduler;
    using enum ck::LoopScheduler;
    using enum PipelineScheduler;

    EXPECT_EQ(convert_pipeline_scheduler<Default>(), DEFAULT);
    EXPECT_EQ(convert_pipeline_scheduler<Interwave>(), INTERWAVE);
}

// ============================================================================
// Test Convolution Direction Detection
// ============================================================================

TEST(InstanceToConvTraits, DetectsForwardDirection)
{
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
            ck::tensor_operation::device::ConvolutionForwardSpecialization::Default,
            ck::tensor_operation::device::GemmSpecialization::Default,
            256,
            128,
            128,
            16,
            8,
            8,
            32,
            32,
            4,
            4,
            ck::Sequence<4, 64, 1>,
            ck::Sequence<1, 0, 2>,
            ck::Sequence<1, 0, 2>,
            2,
            8,
            8,
            1,
            ck::Sequence<4, 64, 1>,
            ck::Sequence<1, 0, 2>,
            ck::Sequence<1, 0, 2>,
            2,
            8,
            8,
            1,
            1,
            1,
            ck::Sequence<1, 32, 1, 8>,
            8,
            ck::BlockGemmPipelineScheduler::Intrawave,
            ck::BlockGemmPipelineVersion::v1,
            ck::half_t,
            ck::half_t,
            false>;

    using Traits = ck_tile::reflect::conv::ConvTraits<DeviceInstance>;

    EXPECT_EQ(Traits::direction, ConvDirection::FORWARD);
}

// ============================================================================
// Test Convolution Specialization Detection
// ============================================================================

TEST(InstanceToConvTraits, ExtractsDefaultSpecialization)
{
    using DeviceInstance =
        ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<
            2,
            ck::tensor_layout::convolution::GNHWC,
            ck::tensor_layout::convolution::GKYXC,
            ck::Tuple<>,
            ck::tensor_layout::convolution::GNHWK,
            ck::half_t,
            ck::half_t,
            float,
            ck::half_t,
            ck::Tuple<>,
            ck::half_t,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::device::ConvolutionForwardSpecialization::Default,
            ck::tensor_operation::device::GemmSpecialization::Default,
            256,
            128,
            128,
            16,
            8,
            8,
            32,
            32,
            4,
            4,
            ck::Sequence<4, 64, 1>,
            ck::Sequence<1, 0, 2>,
            ck::Sequence<1, 0, 2>,
            2,
            8,
            8,
            1,
            ck::Sequence<4, 64, 1>,
            ck::Sequence<1, 0, 2>,
            ck::Sequence<1, 0, 2>,
            2,
            8,
            8,
            1,
            1,
            1,
            ck::Sequence<1, 32, 1, 8>,
            8,
            ck::BlockGemmPipelineScheduler::Intrawave,
            ck::BlockGemmPipelineVersion::v1,
            ck::half_t,
            ck::half_t,
            false>;

    using Traits = ck_tile::reflect::conv::ConvTraits<DeviceInstance>;

    EXPECT_EQ(Traits::conv_specialization, ck_tile::builder::ConvFwdSpecialization::DEFAULT);
}

TEST(InstanceToConvTraits, ExtractsFilter1x1Pad0Specialization)
{
    using DeviceInstance =
        ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<
            2,
            ck::tensor_layout::convolution::GNHWC,
            ck::tensor_layout::convolution::GKYXC,
            ck::Tuple<>,
            ck::tensor_layout::convolution::GNHWK,
            ck::half_t,
            ck::half_t,
            float,
            ck::half_t,
            ck::Tuple<>,
            ck::half_t,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Pad0,
            ck::tensor_operation::device::GemmSpecialization::Default,
            256,
            128,
            128,
            16,
            8,
            8,
            32,
            32,
            4,
            4,
            ck::Sequence<4, 64, 1>,
            ck::Sequence<1, 0, 2>,
            ck::Sequence<1, 0, 2>,
            2,
            8,
            8,
            1,
            ck::Sequence<4, 64, 1>,
            ck::Sequence<1, 0, 2>,
            ck::Sequence<1, 0, 2>,
            2,
            8,
            8,
            1,
            1,
            1,
            ck::Sequence<1, 32, 1, 8>,
            8,
            ck::BlockGemmPipelineScheduler::Intrawave,
            ck::BlockGemmPipelineVersion::v1,
            ck::half_t,
            ck::half_t,
            false>;

    using Traits = ck_tile::reflect::conv::ConvTraits<DeviceInstance>;

    EXPECT_EQ(Traits::conv_specialization,
              ck_tile::builder::ConvFwdSpecialization::FILTER_1X1_PAD0);
}

// ============================================================================
// Test Layout Detection
// ============================================================================

TEST(InstanceToConvTraits, ExtractsGnhwcLayout)
{
    using DeviceInstance =
        ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<
            2,
            ck::tensor_layout::convolution::GNHWC,
            ck::tensor_layout::convolution::GKYXC,
            ck::Tuple<>,
            ck::tensor_layout::convolution::GNHWK,
            ck::half_t,
            ck::half_t,
            float,
            ck::half_t,
            ck::Tuple<>,
            ck::half_t,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::device::ConvolutionForwardSpecialization::Default,
            ck::tensor_operation::device::GemmSpecialization::Default,
            256,
            128,
            128,
            16,
            8,
            8,
            32,
            32,
            4,
            4,
            ck::Sequence<4, 64, 1>,
            ck::Sequence<1, 0, 2>,
            ck::Sequence<1, 0, 2>,
            2,
            8,
            8,
            1,
            ck::Sequence<4, 64, 1>,
            ck::Sequence<1, 0, 2>,
            ck::Sequence<1, 0, 2>,
            2,
            8,
            8,
            1,
            1,
            1,
            ck::Sequence<1, 32, 1, 8>,
            8,
            ck::BlockGemmPipelineScheduler::Intrawave,
            ck::BlockGemmPipelineVersion::v1,
            ck::half_t,
            ck::half_t,
            false>;

    using Traits = ck_tile::reflect::conv::ConvTraits<DeviceInstance>;

    EXPECT_THAT(Traits::layout,
                ElementsAre(TensorLayout::GNHWC, TensorLayout::GKYXC, TensorLayout::GNHWK));
}

TEST(InstanceToConvTraits, ExtractsNhwgcLayout)
{
    using DeviceInstance =
        ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<
            2,
            ck::tensor_layout::convolution::NHWGC,
            ck::tensor_layout::convolution::GKYXC,
            ck::Tuple<>,
            ck::tensor_layout::convolution::NHWGK,
            ck::half_t,
            ck::half_t,
            float,
            ck::half_t,
            ck::Tuple<>,
            ck::half_t,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::device::ConvolutionForwardSpecialization::Default,
            ck::tensor_operation::device::GemmSpecialization::Default,
            256,
            128,
            128,
            16,
            8,
            8,
            32,
            32,
            4,
            4,
            ck::Sequence<4, 64, 1>,
            ck::Sequence<1, 0, 2>,
            ck::Sequence<1, 0, 2>,
            2,
            8,
            8,
            1,
            ck::Sequence<4, 64, 1>,
            ck::Sequence<1, 0, 2>,
            ck::Sequence<1, 0, 2>,
            2,
            8,
            8,
            1,
            1,
            1,
            ck::Sequence<1, 32, 1, 8>,
            8,
            ck::BlockGemmPipelineScheduler::Intrawave,
            ck::BlockGemmPipelineVersion::v1,
            ck::half_t,
            ck::half_t,
            false>;

    using Traits = ck_tile::reflect::conv::ConvTraits<DeviceInstance>;

    EXPECT_THAT(Traits::layout,
                ElementsAre(TensorLayout::NHWGC, TensorLayout::GKYXC, TensorLayout::NHWGK));
}

TEST(InstanceToConvTraits, ExtractsNgchwGkyxcLayout)
{
    using DeviceInstance =
        ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<
            2,
            ck::tensor_layout::convolution::NGCHW,
            ck::tensor_layout::convolution::GKYXC,
            ck::Tuple<>,
            ck::tensor_layout::convolution::NGKHW,
            ck::half_t,
            ck::half_t,
            float,
            ck::half_t,
            ck::Tuple<>,
            ck::half_t,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::device::ConvolutionForwardSpecialization::Default,
            ck::tensor_operation::device::GemmSpecialization::Default,
            256,
            128,
            128,
            16,
            8,
            8,
            32,
            32,
            4,
            4,
            ck::Sequence<4, 64, 1>,
            ck::Sequence<1, 0, 2>,
            ck::Sequence<1, 0, 2>,
            2,
            8,
            8,
            1,
            ck::Sequence<4, 64, 1>,
            ck::Sequence<1, 0, 2>,
            ck::Sequence<1, 0, 2>,
            2,
            8,
            8,
            1,
            1,
            1,
            ck::Sequence<1, 32, 1, 8>,
            8,
            ck::BlockGemmPipelineScheduler::Intrawave,
            ck::BlockGemmPipelineVersion::v1,
            ck::half_t,
            ck::half_t,
            false>;

    using Traits = ck_tile::reflect::conv::ConvTraits<DeviceInstance>;

    EXPECT_THAT(Traits::layout,
                ElementsAre(TensorLayout::NGCHW, TensorLayout::GKYXC, TensorLayout::NGKHW));
}

TEST(InstanceToConvTraits, ExtractsNgchwGkcyxLayout)
{
    using DeviceInstance =
        ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<
            2,
            ck::tensor_layout::convolution::NGCHW,
            ck::tensor_layout::convolution::GKCYX,
            ck::Tuple<>,
            ck::tensor_layout::convolution::NGKHW,
            ck::half_t,
            ck::half_t,
            float,
            ck::half_t,
            ck::Tuple<>,
            ck::half_t,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::device::ConvolutionForwardSpecialization::Default,
            ck::tensor_operation::device::GemmSpecialization::Default,
            256,
            128,
            128,
            16,
            8,
            8,
            32,
            32,
            4,
            4,
            ck::Sequence<4, 64, 1>,
            ck::Sequence<1, 0, 2>,
            ck::Sequence<1, 0, 2>,
            2,
            8,
            8,
            1,
            ck::Sequence<4, 64, 1>,
            ck::Sequence<1, 0, 2>,
            ck::Sequence<1, 0, 2>,
            2,
            8,
            8,
            1,
            1,
            1,
            ck::Sequence<1, 32, 1, 8>,
            8,
            ck::BlockGemmPipelineScheduler::Intrawave,
            ck::BlockGemmPipelineVersion::v1,
            ck::half_t,
            ck::half_t,
            false>;

    using Traits = ck_tile::reflect::conv::ConvTraits<DeviceInstance>;

    EXPECT_THAT(Traits::layout,
                ElementsAre(TensorLayout::NGCHW, TensorLayout::GKCYX, TensorLayout::NGKHW));
}

// ============================================================================
// Test Data Type Detection
// ============================================================================

TEST(InstanceToConvTraits, ExtractsFp16DataType)
{
    using DeviceInstance =
        ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<
            2,
            ck::tensor_layout::convolution::GNHWC,
            ck::tensor_layout::convolution::GKYXC,
            ck::Tuple<>,
            ck::tensor_layout::convolution::GNHWK,
            ck::half_t,
            ck::half_t,
            float,
            ck::half_t,
            ck::Tuple<>,
            ck::half_t,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::device::ConvolutionForwardSpecialization::Default,
            ck::tensor_operation::device::GemmSpecialization::Default,
            256,
            128,
            128,
            16,
            8,
            8,
            32,
            32,
            4,
            4,
            ck::Sequence<4, 64, 1>,
            ck::Sequence<1, 0, 2>,
            ck::Sequence<1, 0, 2>,
            2,
            8,
            8,
            1,
            ck::Sequence<4, 64, 1>,
            ck::Sequence<1, 0, 2>,
            ck::Sequence<1, 0, 2>,
            2,
            8,
            8,
            1,
            1,
            1,
            ck::Sequence<1, 32, 1, 8>,
            8,
            ck::BlockGemmPipelineScheduler::Intrawave,
            ck::BlockGemmPipelineVersion::v1,
            ck::half_t,
            ck::half_t,
            false>;

    using Traits = ck_tile::reflect::conv::ConvTraits<DeviceInstance>;

    EXPECT_EQ(Traits::data_type, DataType::FP16);
}

TEST(InstanceToConvTraits, ExtractsBf16DataType)
{
    using DeviceInstance =
        ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<
            2,
            ck::tensor_layout::convolution::GNHWC,
            ck::tensor_layout::convolution::GKYXC,
            ck::Tuple<>,
            ck::tensor_layout::convolution::GNHWK,
            ck::bhalf_t,
            ck::bhalf_t,
            float,
            ck::bhalf_t,
            ck::Tuple<>,
            ck::bhalf_t,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::device::ConvolutionForwardSpecialization::Default,
            ck::tensor_operation::device::GemmSpecialization::Default,
            256,
            128,
            128,
            16,
            8,
            8,
            32,
            32,
            4,
            4,
            ck::Sequence<4, 64, 1>,
            ck::Sequence<1, 0, 2>,
            ck::Sequence<1, 0, 2>,
            2,
            8,
            8,
            1,
            ck::Sequence<4, 64, 1>,
            ck::Sequence<1, 0, 2>,
            ck::Sequence<1, 0, 2>,
            2,
            8,
            8,
            1,
            1,
            1,
            ck::Sequence<1, 32, 1, 8>,
            8,
            ck::BlockGemmPipelineScheduler::Intrawave,
            ck::BlockGemmPipelineVersion::v1,
            ck::bhalf_t,
            ck::bhalf_t,
            false>;

    using Traits = ck_tile::reflect::conv::ConvTraits<DeviceInstance>;

    EXPECT_EQ(Traits::data_type, DataType::BF16);
}

TEST(InstanceToConvTraits, ExtractsFp32DataType)
{
    using DeviceInstance =
        ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<
            2,
            ck::tensor_layout::convolution::GNHWC,
            ck::tensor_layout::convolution::GKYXC,
            ck::Tuple<>,
            ck::tensor_layout::convolution::GNHWK,
            float,
            float,
            float,
            float,
            ck::Tuple<>,
            float,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::device::ConvolutionForwardSpecialization::Default,
            ck::tensor_operation::device::GemmSpecialization::Default,
            256,
            128,
            128,
            16,
            8,
            8,
            32,
            32,
            4,
            4,
            ck::Sequence<4, 64, 1>,
            ck::Sequence<1, 0, 2>,
            ck::Sequence<1, 0, 2>,
            2,
            8,
            8,
            1,
            ck::Sequence<4, 64, 1>,
            ck::Sequence<1, 0, 2>,
            ck::Sequence<1, 0, 2>,
            2,
            8,
            8,
            1,
            1,
            1,
            ck::Sequence<1, 32, 1, 8>,
            8,
            ck::BlockGemmPipelineScheduler::Intrawave,
            ck::BlockGemmPipelineVersion::v1,
            float,
            float,
            false>;

    using Traits = ck_tile::reflect::conv::ConvTraits<DeviceInstance>;

    EXPECT_EQ(Traits::data_type, DataType::FP32);
}

TEST(InstanceToConvTraits, ExtractsI8DataType)
{
    using DeviceInstance =
        ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<
            2,
            ck::tensor_layout::convolution::GNHWC,
            ck::tensor_layout::convolution::GKYXC,
            ck::Tuple<>,
            ck::tensor_layout::convolution::GNHWK,
            int8_t,
            int8_t,
            int32_t,
            int8_t,
            ck::Tuple<>,
            int8_t,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::device::ConvolutionForwardSpecialization::Default,
            ck::tensor_operation::device::GemmSpecialization::Default,
            256,
            128,
            128,
            16,
            8,
            8,
            32,
            32,
            4,
            4,
            ck::Sequence<4, 64, 1>,
            ck::Sequence<1, 0, 2>,
            ck::Sequence<1, 0, 2>,
            2,
            8,
            8,
            1,
            ck::Sequence<4, 64, 1>,
            ck::Sequence<1, 0, 2>,
            ck::Sequence<1, 0, 2>,
            2,
            8,
            8,
            1,
            1,
            1,
            ck::Sequence<1, 32, 1, 8>,
            8,
            ck::BlockGemmPipelineScheduler::Intrawave,
            ck::BlockGemmPipelineVersion::v1,
            int8_t,
            int8_t,
            false>;

    using Traits = ck_tile::reflect::conv::ConvTraits<DeviceInstance>;

    EXPECT_EQ(Traits::data_type, DataType::I8);
}

// ============================================================================
// Test GEMM Padding Detection
// ============================================================================

TEST(InstanceToConvTraits, ExtractsDefaultGemmPadding)
{
    using DeviceInstance =
        ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<
            2,
            ck::tensor_layout::convolution::GNHWC,
            ck::tensor_layout::convolution::GKYXC,
            ck::Tuple<>,
            ck::tensor_layout::convolution::GNHWK,
            ck::half_t,
            ck::half_t,
            float,
            ck::half_t,
            ck::Tuple<>,
            ck::half_t,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::device::ConvolutionForwardSpecialization::Default,
            ck::tensor_operation::device::GemmSpecialization::Default,
            256,
            128,
            128,
            16,
            8,
            8,
            32,
            32,
            4,
            4,
            ck::Sequence<4, 64, 1>,
            ck::Sequence<1, 0, 2>,
            ck::Sequence<1, 0, 2>,
            2,
            8,
            8,
            1,
            ck::Sequence<4, 64, 1>,
            ck::Sequence<1, 0, 2>,
            ck::Sequence<1, 0, 2>,
            2,
            8,
            8,
            1,
            1,
            1,
            ck::Sequence<1, 32, 1, 8>,
            8,
            ck::BlockGemmPipelineScheduler::Intrawave,
            ck::BlockGemmPipelineVersion::v1,
            ck::half_t,
            ck::half_t,
            false>;

    using Traits = ck_tile::reflect::conv::ConvTraits<DeviceInstance>;

    EXPECT_EQ(Traits::gemm_padding, GemmPadding::DEFAULT);
}

TEST(InstanceToConvTraits, ExtractsMnkGemmPadding)
{
    using DeviceInstance =
        ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<
            2,
            ck::tensor_layout::convolution::GNHWC,
            ck::tensor_layout::convolution::GKYXC,
            ck::Tuple<>,
            ck::tensor_layout::convolution::GNHWK,
            ck::half_t,
            ck::half_t,
            float,
            ck::half_t,
            ck::Tuple<>,
            ck::half_t,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::device::ConvolutionForwardSpecialization::Default,
            ck::tensor_operation::device::GemmSpecialization::MNKPadding,
            256,
            128,
            128,
            16,
            8,
            8,
            32,
            32,
            4,
            4,
            ck::Sequence<4, 64, 1>,
            ck::Sequence<1, 0, 2>,
            ck::Sequence<1, 0, 2>,
            2,
            8,
            8,
            1,
            ck::Sequence<4, 64, 1>,
            ck::Sequence<1, 0, 2>,
            ck::Sequence<1, 0, 2>,
            2,
            8,
            8,
            1,
            1,
            1,
            ck::Sequence<1, 32, 1, 8>,
            8,
            ck::BlockGemmPipelineScheduler::Intrawave,
            ck::BlockGemmPipelineVersion::v1,
            ck::half_t,
            ck::half_t,
            false>;

    using Traits = ck_tile::reflect::conv::ConvTraits<DeviceInstance>;

    EXPECT_EQ(Traits::gemm_padding, GemmPadding::MNK_PADDING);
}

// ============================================================================
// Comprehensive Transformation Tests - Per Device Class Template
// ============================================================================
// These tests verify the complete InstanceTraits → ConvTraits transformation
// for each forward convolution Device class template. They are verbose to
// provide maximum safety during refactoring.
// ============================================================================

TEST(InstanceToConvTraits, TransformsFwdMultipleAbdXdlCShuffleV3)
{
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
            ck::tensor_operation::device::ConvolutionForwardSpecialization::Default,
            ck::tensor_operation::device::GemmSpecialization::Default,
            256,                                       // BlockSize
            128,                                       // MPerBlock
            128,                                       // NPerBlock
            16,                                        // KPerBlock
            8,                                         // AK1
            8,                                         // BK1
            32,                                        // MPerXDL
            32,                                        // NPerXDL
            4,                                         // MXdlPerWave
            4,                                         // NXdlPerWave
            ck::Sequence<4, 64, 1>,                    // ABlockTransferThreadClusterLengths
            ck::Sequence<1, 0, 2>,                     // ABlockTransferThreadClusterArrangeOrder
            ck::Sequence<1, 0, 2>,                     // ABlockTransferSrcAccessOrder
            2,                                         // ABlockTransferSrcVectorDim
            8,                                         // ABlockTransferSrcScalarPerVector
            8,                                         // ABlockTransferDstScalarPerVector_AK1
            1,                                         // ABlockLdsExtraM
            ck::Sequence<4, 64, 1>,                    // BBlockTransferThreadClusterLengths
            ck::Sequence<1, 0, 2>,                     // BBlockTransferThreadClusterArrangeOrder
            ck::Sequence<1, 0, 2>,                     // BBlockTransferSrcAccessOrder
            2,                                         // BBlockTransferSrcVectorDim
            8,                                         // BBlockTransferSrcScalarPerVector
            8,                                         // BBlockTransferDstScalarPerVector_BK1
            1,                                         // BBlockLdsExtraN
            1,                                         // CShuffleMXdlPerWavePerShuffle
            1,                                         // CShuffleNXdlPerWavePerShuffle
            ck::Sequence<1, 32, 1, 8>,                 // CDEBlockTransferClusterLengths
            8,                                         // CDEBlockTransferScalarPerVector_NPerBlock
            ck::BlockGemmPipelineScheduler::Intrawave, // BlkGemmPipeSched
            ck::BlockGemmPipelineVersion::v1,          // BlkGemmPipelineVer
            ck::half_t,                                // AComputeDataType
            ck::half_t,                                // BComputeDataType
            false>;                                    // DirectLoad

    using InstTraits = ck_tile::reflect::InstanceTraits<DeviceInstance>;
    using ConvTraits = ck_tile::reflect::conv::ConvTraits<DeviceInstance>;

    // Verify signature information
    EXPECT_EQ(ConvTraits::spatial_dim, InstTraits::kSpatialDim);
    EXPECT_EQ(ConvTraits::direction, ConvDirection::FORWARD);
    EXPECT_EQ(ConvTraits::data_type, DataType::FP16);
    EXPECT_EQ(ConvTraits::gemm_padding, GemmPadding::DEFAULT);

    // Verify tile dimensions
    EXPECT_EQ(ConvTraits::tile_dims.m, InstTraits::kMPerBlock);
    EXPECT_EQ(ConvTraits::tile_dims.n, InstTraits::kNPerBlock);
    EXPECT_EQ(ConvTraits::tile_dims.k, InstTraits::kKPerBlock);

    // Verify pipeline configuration
    EXPECT_EQ(ConvTraits::pipeline_scheduler, PipelineScheduler::INTRAWAVE);
    EXPECT_EQ(ConvTraits::pipeline_version, PipelineVersion::V1);
}

TEST(InstanceToConvTraits, TransformsFwdMultipleAbdXdlCShuffle)
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
            ck::tensor_operation::device::ConvolutionForwardSpecialization::Default,
            ck::tensor_operation::device::GemmSpecialization::Default,
            1,                          // NumGemmKPrefetchStage
            256,                        // BlockSize
            128,                        // MPerBlock
            128,                        // NPerBlock
            16,                         // KPerBlock
            8,                          // AK1
            8,                          // BK1
            32,                         // MPerXDL
            32,                         // NPerXDL
            4,                          // MXdlPerWave
            4,                          // NXdlPerWave
            ck::Sequence<4, 64, 1>,     // ABlockTransferThreadClusterLengths
            ck::Sequence<1, 0, 2>,      // ABlockTransferThreadClusterArrangeOrder
            ck::Sequence<1, 0, 2>,      // ABlockTransferSrcAccessOrder
            2,                          // ABlockTransferSrcVectorDim
            8,                          // ABlockTransferSrcScalarPerVector
            8,                          // ABlockTransferDstScalarPerVector_AK1
            1,                          // ABlockLdsExtraM
            ck::Sequence<4, 64, 1>,     // BBlockTransferThreadClusterLengths
            ck::Sequence<1, 0, 2>,      // BBlockTransferThreadClusterArrangeOrder
            ck::Sequence<1, 0, 2>,      // BBlockTransferSrcAccessOrder
            2,                          // BBlockTransferSrcVectorDim
            8,                          // BBlockTransferSrcScalarPerVector
            8,                          // BBlockTransferDstScalarPerVector_BK1
            1,                          // BBlockLdsExtraN
            1,                          // CShuffleMXdlPerWavePerShuffle
            1,                          // CShuffleNXdlPerWavePerShuffle
            ck::Sequence<1, 32, 1, 8>,  // CDEBlockTransferClusterLengths
            8,                          // CDEBlockTransferScalarPerVector_NPerBlock
            ck::half_t,                 // AComputeDataType
            ck::half_t,                 // BComputeDataType
            ck::LoopScheduler::Default, // LoopSched
            1>;                         // NumGroupsToMerge

    using InstTraits = ck_tile::reflect::InstanceTraits<DeviceInstance>;
    using ConvTraits = ck_tile::reflect::conv::ConvTraits<DeviceInstance>;

    // Verify signature information
    EXPECT_EQ(ConvTraits::spatial_dim, InstTraits::kSpatialDim);
    EXPECT_EQ(ConvTraits::direction, ConvDirection::FORWARD);
    EXPECT_EQ(ConvTraits::data_type, DataType::FP16);
    EXPECT_EQ(ConvTraits::gemm_padding, GemmPadding::DEFAULT);

    // Verify tile dimensions
    EXPECT_EQ(ConvTraits::tile_dims.m, InstTraits::kMPerBlock);
    EXPECT_EQ(ConvTraits::tile_dims.n, InstTraits::kNPerBlock);
    EXPECT_EQ(ConvTraits::tile_dims.k, InstTraits::kKPerBlock);

    // Verify pipeline configuration (uses LoopScheduler instead of BlockGemmPipelineScheduler)
    EXPECT_EQ(ConvTraits::pipeline_scheduler, PipelineScheduler::DEFAULT);
    EXPECT_EQ(ConvTraits::pipeline_version, PipelineVersion::V1);
}

TEST(InstanceToConvTraits, TransformsFwdMultipleDXdlLargeTensor)
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
            ck::tensor_operation::device::ConvolutionForwardSpecialization::Default,
            ck::tensor_operation::device::GemmSpecialization::Default,
            1,                           // NumGemmKPrefetchStage
            256,                         // BlockSize
            128,                         // MPerBlock
            128,                         // NPerBlock
            16,                          // KPerBlock
            8,                           // AK1
            8,                           // BK1
            32,                          // MPerXDL
            32,                          // NPerXDL
            4,                           // MXdlPerWave
            4,                           // NXdlPerWave
            ck::Sequence<4, 64, 1>,      // ABlockTransferThreadClusterLengths
            ck::Sequence<1, 0, 2>,       // ABlockTransferThreadClusterArrangeOrder
            ck::Sequence<1, 0, 2>,       // ABlockTransferSrcAccessOrder
            2,                           // ABlockTransferSrcVectorDim
            8,                           // ABlockTransferSrcScalarPerVector
            8,                           // ABlockTransferDstScalarPerVector_AK1
            1,                           // ABlockLdsExtraM
            ck::Sequence<4, 64, 1>,      // BBlockTransferThreadClusterLengths
            ck::Sequence<1, 0, 2>,       // BBlockTransferThreadClusterArrangeOrder
            ck::Sequence<1, 0, 2>,       // BBlockTransferSrcAccessOrder
            2,                           // BBlockTransferSrcVectorDim
            8,                           // BBlockTransferSrcScalarPerVector
            8,                           // BBlockTransferDstScalarPerVector_BK1
            1,                           // BBlockLdsExtraN
            1,                           // CShuffleMXdlPerWavePerShuffle
            1,                           // CShuffleNXdlPerWavePerShuffle
            ck::Sequence<1, 32, 1, 8>,   // CDEBlockTransferClusterLengths
            8,                           // CDEBlockTransferScalarPerVector_NPerBlock
            ck::half_t,                  // AComputeDataType
            ck::half_t,                  // BComputeDataType
            ck::LoopScheduler::Default>; // LoopSched

    using InstTraits = ck_tile::reflect::InstanceTraits<DeviceInstance>;
    using ConvTraits = ck_tile::reflect::conv::ConvTraits<DeviceInstance>;

    // Verify signature information
    EXPECT_EQ(ConvTraits::spatial_dim, InstTraits::kSpatialDim);
    EXPECT_EQ(ConvTraits::direction, ConvDirection::FORWARD);
    EXPECT_EQ(ConvTraits::data_type, DataType::FP16);
    EXPECT_EQ(ConvTraits::gemm_padding, GemmPadding::DEFAULT);

    // Verify tile dimensions
    EXPECT_EQ(ConvTraits::tile_dims.m, InstTraits::kMPerBlock);
    EXPECT_EQ(ConvTraits::tile_dims.n, InstTraits::kNPerBlock);
    EXPECT_EQ(ConvTraits::tile_dims.k, InstTraits::kKPerBlock);

    // Verify pipeline configuration
    EXPECT_EQ(ConvTraits::pipeline_scheduler, PipelineScheduler::DEFAULT);
    EXPECT_EQ(ConvTraits::pipeline_version, PipelineVersion::V1);
}

} // anonymous namespace
