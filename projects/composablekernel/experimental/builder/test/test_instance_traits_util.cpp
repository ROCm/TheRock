// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/utility/scheduler_enum.hpp"
#include "ck/utility/sequence.hpp"
#include "ck_tile/builder/reflect/instance_traits_util.hpp"

namespace ck_tile::reflect::detail {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;

TEST(InstanceTraitsUtil, SequenceToArrayReturnsEmptyArrayForEmptySequence)
{
    EXPECT_THAT(SequenceToArray<ck::Sequence<>>::value, IsEmpty());
}

TEST(InstanceTraitsUtil, SequenceToArrayReturnsArrayWithSingleElement)
{
    EXPECT_THAT(SequenceToArray<ck::Sequence<42>>::value, ElementsAre(42));
}

TEST(InstanceTraitsUtil, SequenceToArrayReturnsArrayWithMultipleElements)
{
    EXPECT_THAT((SequenceToArray<ck::Sequence<1, 2, 3, 4, 5>>::value), ElementsAre(1, 2, 3, 4, 5));
}

TEST(InstanceTraitsUtil, TypeNameReturnsCorrectStrings)
{
    EXPECT_THAT((std::vector<std::string_view>{type_name<ck::half_t>(),
                                               type_name<float>(),
                                               type_name<double>(),
                                               type_name<int8_t>(),
                                               type_name<int32_t>(),
                                               type_name<ck::bhalf_t>(),
                                               type_name<ck::f8_t>(),
                                               type_name<ck::bf8_t>()}),
                ElementsAre("fp16", "fp32", "fp64", "s8", "s32", "bf16", "fp8", "bf8"));
}

TEST(InstanceTraitsUtil, LayoutNameReturnsCorrectStringsForGemmLayouts)
{
    namespace gemm = ck::tensor_layout::gemm;
    EXPECT_THAT((std::vector<std::string_view>{layout_name<gemm::RowMajor>(),
                                               layout_name<gemm::ColumnMajor>(),
                                               layout_name<gemm::MFMA>()}),
                ElementsAre("RowMajor", "ColumnMajor", "MFMA"));
}

TEST(InstanceTraitsUtil, LayoutNameReturnsCorrectStringsForConvLayouts)
{
    namespace conv = ck::tensor_layout::convolution;
    EXPECT_THAT((std::vector<std::string_view>{
                    // Input tensor layouts
                    // TODO(deprecated): Remove non-grouped layouts once instances are removed.
                    layout_name<conv::NCHW>(),
                    layout_name<conv::NHWC>(),
                    layout_name<conv::NCDHW>(),
                    layout_name<conv::NDHWC>(),
                    // Grouped input layouts
                    layout_name<conv::GNCHW>(),
                    layout_name<conv::GNHWC>(),
                    // Weight tensor layouts
                    layout_name<conv::KCYX>(),
                    layout_name<conv::KYXC>(),
                    layout_name<conv::GKCYX>(),
                    layout_name<conv::GKYXC>(),
                    // Output tensor layouts
                    layout_name<conv::NKHW>(),
                    layout_name<conv::NHWK>(),
                    layout_name<conv::GNKHW>(),
                    layout_name<conv::GNHWK>(),
                    // Strided layouts
                    // TODO(deprecated): Remove strided layouts once instances are removed.
                    layout_name<conv::G_NHW_C>(),
                    layout_name<conv::G_K_YX_C>(),
                    layout_name<conv::G_NHW_K>(),
                    // Bias layouts
                    layout_name<conv::G_C>(),
                    layout_name<conv::G_K>()}),
                ElementsAre("NCHW",
                            "NHWC",
                            "NCDHW",
                            "NDHWC",
                            "GNCHW",
                            "GNHWC",
                            "KCYX",
                            "KYXC",
                            "GKCYX",
                            "GKYXC",
                            "NKHW",
                            "NHWK",
                            "GNKHW",
                            "GNHWK",
                            "G_NHW_C",
                            "G_K_YX_C",
                            "G_NHW_K",
                            "G_C",
                            "G_K"));
}

TEST(InstanceTraitsUtil, ElementwiseOpNameReturnsCorrectStrings)
{
    namespace element_wise = ck::tensor_operation::element_wise;
    EXPECT_THAT((std::vector<std::string_view>{
                    elementwise_op_name<element_wise::PassThrough>(),
                    elementwise_op_name<element_wise::Scale>(),
                    elementwise_op_name<element_wise::Bilinear>(),
                    elementwise_op_name<element_wise::Add>(),
                    elementwise_op_name<element_wise::AddRelu>(),
                    elementwise_op_name<element_wise::Relu>(),
                    elementwise_op_name<element_wise::BiasNormalizeInInferClamp>(),
                    elementwise_op_name<element_wise::Clamp>(),
                    elementwise_op_name<element_wise::AddClamp>()}),
                ElementsAre("PassThrough",
                            "Scale",
                            "Bilinear",
                            "Add",
                            "AddRelu",
                            "Relu",
                            "BiasNormalizeInInferClamp",
                            "Clamp",
                            "AddClamp"));
}

TEST(InstanceTraitsUtil, ConvFwdSpecNameReturnsCorrectStrings)
{
    using enum ck::tensor_operation::device::ConvolutionForwardSpecialization;
    EXPECT_THAT(
        (std::vector<std::string_view>{conv_fwd_spec_name(Default),
                                       conv_fwd_spec_name(Filter1x1Stride1Pad0),
                                       conv_fwd_spec_name(Filter1x1Pad0),
                                       conv_fwd_spec_name(Filter3x3),
                                       conv_fwd_spec_name(OddC)}),
        ElementsAre("Default", "Filter1x1Stride1Pad0", "Filter1x1Pad0", "Filter3x3", "OddC"));
}

TEST(InstanceTraitsUtil, GemmSpecNameReturnsCorrectStrings)
{
    using enum ck::tensor_operation::device::GemmSpecialization;
    EXPECT_THAT((std::vector<std::string_view>{gemm_spec_name(Default),
                                               gemm_spec_name(MPadding),
                                               gemm_spec_name(NPadding),
                                               gemm_spec_name(KPadding),
                                               gemm_spec_name(MNPadding),
                                               gemm_spec_name(MKPadding),
                                               gemm_spec_name(NKPadding),
                                               gemm_spec_name(MNKPadding),
                                               gemm_spec_name(OPadding),
                                               gemm_spec_name(MOPadding),
                                               gemm_spec_name(NOPadding),
                                               gemm_spec_name(KOPadding),
                                               gemm_spec_name(MNOPadding),
                                               gemm_spec_name(MKOPadding),
                                               gemm_spec_name(NKOPadding),
                                               gemm_spec_name(MNKOPadding)}),
                ElementsAre("Default",
                            "MPadding",
                            "NPadding",
                            "KPadding",
                            "MNPadding",
                            "MKPadding",
                            "NKPadding",
                            "MNKPadding",
                            "OPadding",
                            "MOPadding",
                            "NOPadding",
                            "KOPadding",
                            "MNOPadding",
                            "MKOPadding",
                            "NKOPadding",
                            "MNKOPadding"));
}

TEST(InstanceTraitsUtil, PipelineSchedulerNameReturnsCorrectStrings)
{
    using enum ck::BlockGemmPipelineScheduler;
    EXPECT_THAT((std::vector<std::string_view>{pipeline_scheduler_name(Intrawave),
                                               pipeline_scheduler_name(Interwave)}),
                ElementsAre("Intrawave", "Interwave"));
}

TEST(InstanceTraitsUtil, PipelineVersionNameReturnsCorrectStrings)
{
    using enum ck::BlockGemmPipelineVersion;
    EXPECT_THAT((std::vector<std::string_view>{pipeline_version_name(v1),
                                               pipeline_version_name(v2),
                                               pipeline_version_name(v3),
                                               pipeline_version_name(v4),
                                               pipeline_version_name(v5)}),
                ElementsAre("v1", "v2", "v3", "v4", "v5"));
}

TEST(InstanceTraitsUtil, LoopSchedulerNameReturnsCorrectStrings)
{
    using enum ck::LoopScheduler;
    EXPECT_THAT((std::vector<std::string_view>{loop_scheduler_name(Default),
                                               loop_scheduler_name(Interwave)}),
                ElementsAre("Default", "Interwave"));
}

TEST(InstanceTraitsUtil, TupleNameReturnsEmptyTupleForEmptyTuple)
{
    EXPECT_EQ(tuple_name<ck::Tuple<>>(), "EmptyTuple");
}

TEST(InstanceTraitsUtil, TupleNameReturnsTupleStringForSingleLayout)
{
    EXPECT_EQ(tuple_name<ck::Tuple<ck::tensor_layout::convolution::NCHW>>(), "Tuple(NCHW)");
}

TEST(InstanceTraitsUtil, TupleNameReturnsTupleStringForTwoLayouts)
{
    EXPECT_EQ((tuple_name<ck::Tuple<ck::tensor_layout::convolution::NCHW,
                                    ck::tensor_layout::convolution::NHWC>>()),
              "Tuple(NCHW,NHWC)");
}

TEST(InstanceTraitsUtil, TupleNameReturnsTupleStringForThreeLayouts)
{
    EXPECT_EQ((tuple_name<ck::Tuple<ck::tensor_layout::convolution::NCHW,
                                    ck::tensor_layout::convolution::NHWC,
                                    ck::tensor_layout::convolution::NKHW>>()),
              "Tuple(NCHW,NHWC,NKHW)");
}

TEST(InstanceTraitsUtil, TupleNameReturnsTupleStringForSingleDataType)
{
    EXPECT_EQ(tuple_name<ck::Tuple<ck::half_t>>(), "Tuple(fp16)");
}

TEST(InstanceTraitsUtil, TupleNameReturnsTupleStringForTwoDataTypes)
{
    EXPECT_EQ((tuple_name<ck::Tuple<ck::half_t, float>>()), "Tuple(fp16,fp32)");
}

TEST(InstanceTraitsUtil, TupleNameReturnsTupleStringForThreeDataTypes)
{
    EXPECT_EQ((tuple_name<ck::Tuple<ck::half_t, float, double>>()), "Tuple(fp16,fp32,fp64)");
}

TEST(InstanceTraitsUtil, SequenceNameReturnsSeqStringForEmptySequence)
{
    EXPECT_EQ(sequence_name<ck::Sequence<>>(), "Seq()");
}

TEST(InstanceTraitsUtil, SequenceNameReturnsSeqStringForSingleValueSequence)
{
    EXPECT_EQ(sequence_name<ck::Sequence<42>>(), "Seq(42)");
}

TEST(InstanceTraitsUtil, SequenceNameReturnsSeqStringForTwoValueSequence)
{
    EXPECT_EQ((sequence_name<ck::Sequence<1, 2>>()), "Seq(1,2)");
}

TEST(InstanceTraitsUtil, SequenceNameReturnsSeqStringForMultipleValueSequence)
{
    EXPECT_EQ((sequence_name<ck::Sequence<256, 128, 64, 32, 16>>()), "Seq(256,128,64,32,16)");
}

TEST(InstanceTraitsUtil, TypeOrTypeTupleNameReturnsCorrectStringForScalarDataType)
{
    EXPECT_EQ(type_or_type_tuple_name<float>(), "fp32");
}

TEST(InstanceTraitsUtil, TypeOrTypeTupleNameReturnsCorrectStringForTupleOfDataTypes)
{
    EXPECT_EQ((type_or_type_tuple_name<ck::Tuple<ck::half_t, float>>()), "Tuple(fp16,fp32)");
}

} // namespace
} // namespace ck_tile::reflect::detail
