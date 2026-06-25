// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <limits>

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <miopen/miopen.h>

#include "MiopenConvDescriptor.hpp"

using namespace miopen_legacy_plugin;

TEST(TestMiopenConvDescriptor, CreateValidDescriptorFwd)
{
    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 0, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_data_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreateConvolutionFwdAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::ConvolutionFwdAttributes>(
        builder.GetBufferPointer());

    MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1);

    miopenStatus_t status;
    int returnedSpatialDimCount = 0;
    status = miopenGetConvolutionSpatialDim(convDesc.convDescriptor(), &returnedSpatialDimCount);
    EXPECT_EQ(status, miopenStatusSuccess);
    EXPECT_EQ(returnedSpatialDimCount, spatialDimCount);

    std::vector<int> returnedPadding(spatialDimCount);
    std::vector<int> returnedStride(spatialDimCount);
    std::vector<int> returnedDilation(spatialDimCount);
    status = miopenGetConvolutionNdDescriptor(convDesc.convDescriptor(),
                                              static_cast<int>(spatialDimCount),
                                              nullptr,
                                              returnedPadding.data(),
                                              returnedStride.data(),
                                              returnedDilation.data(),
                                              nullptr);
    EXPECT_EQ(status, miopenStatusSuccess);
    EXPECT_TRUE(std::equal(returnedPadding.begin(), returnedPadding.end(), prePadding.begin()));
    EXPECT_TRUE(std::equal(returnedStride.begin(), returnedStride.end(), stride.begin()));
    EXPECT_TRUE(std::equal(returnedDilation.begin(), returnedDilation.end(), dilation.begin()));
}

TEST(TestMiopenConvDescriptor, ThrowsOnWrongSpatialDimCountFwd)
{
    // MIOpen flags an ASAN error if improperly creating a conv descriptor.
    SKIP_IF_ASAN();

    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 0, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_data_sdk::data_objects::ConvMode::CROSS_CORRELATION;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreateConvolutionFwdAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::ConvolutionFwdAttributes>(
        builder.GetBufferPointer());

    EXPECT_THROW(MiopenConvDescriptor convDesc(
                     static_cast<size_t>(std::numeric_limits<int>::max()) + 1, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_THROW(MiopenConvDescriptor convDesc(prePadding.size() - 1, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_NO_THROW(MiopenConvDescriptor convDesc(prePadding.size(), *attrPtr, 1));
    EXPECT_THROW(MiopenConvDescriptor convDesc(prePadding.size() + 1, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenConvDescriptor, ThrowsOnWrongConvModeFwd)
{
    // MIOpen flags an ASAN error if improperly creating a conv descriptor.
    SKIP_IF_ASAN();

    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 0, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_data_sdk::data_objects::ConvMode::CONVOLUTION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreateConvolutionFwdAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::ConvolutionFwdAttributes>(
        builder.GetBufferPointer());

    EXPECT_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenConvDescriptor, ThrowsOnAsymmetricPaddingFwd)
{
    // MIOpen flags an ASAN error if improperly creating a conv descriptor.
    SKIP_IF_ASAN();

    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 1, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_data_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreateConvolutionFwdAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::ConvolutionFwdAttributes>(
        builder.GetBufferPointer());

    EXPECT_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenConvDescriptor, ThrowsOnWrongPaddingFwd)
{
    // MIOpen flags an ASAN error if improperly creating a conv descriptor.
    SKIP_IF_ASAN();

    const std::vector<int64_t> prePadding{0, -1, 0};
    const std::vector<int64_t> postPadding{0, -1, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_data_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreateConvolutionFwdAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::ConvolutionFwdAttributes>(
        builder.GetBufferPointer());

    EXPECT_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenConvDescriptor, ThrowsOnWrongStrideFwd)
{
    // MIOpen flags an ASAN error if improperly creating a conv descriptor.
    SKIP_IF_ASAN();

    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 0, 0};
    const std::vector<int64_t> stride{1, 0, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_data_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreateConvolutionFwdAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::ConvolutionFwdAttributes>(
        builder.GetBufferPointer());

    EXPECT_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenConvDescriptor, ThrowsOnWrongDilationFwd)
{
    // MIOpen flags an ASAN error if improperly creating a conv descriptor.
    SKIP_IF_ASAN();

    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 0, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 0, 1};
    const auto convMode = hipdnn_data_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreateConvolutionFwdAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::ConvolutionFwdAttributes>(
        builder.GetBufferPointer());

    EXPECT_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenConvDescriptor, CreateValidDescriptorBwd)
{
    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 0, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_data_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreateConvolutionBwdAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::ConvolutionBwdAttributes>(
        builder.GetBufferPointer());

    MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1);

    miopenStatus_t status;
    int returnedSpatialDimCount = 0;
    status = miopenGetConvolutionSpatialDim(convDesc.convDescriptor(), &returnedSpatialDimCount);
    EXPECT_EQ(status, miopenStatusSuccess);
    EXPECT_EQ(returnedSpatialDimCount, spatialDimCount);

    std::vector<int> returnedPadding(spatialDimCount);
    std::vector<int> returnedStride(spatialDimCount);
    std::vector<int> returnedDilation(spatialDimCount);
    status = miopenGetConvolutionNdDescriptor(convDesc.convDescriptor(),
                                              static_cast<int>(spatialDimCount),
                                              nullptr,
                                              returnedPadding.data(),
                                              returnedStride.data(),
                                              returnedDilation.data(),
                                              nullptr);
    EXPECT_EQ(status, miopenStatusSuccess);
    EXPECT_TRUE(std::equal(returnedPadding.begin(), returnedPadding.end(), prePadding.begin()));
    EXPECT_TRUE(std::equal(returnedStride.begin(), returnedStride.end(), stride.begin()));
    EXPECT_TRUE(std::equal(returnedDilation.begin(), returnedDilation.end(), dilation.begin()));
}

TEST(TestMiopenConvDescriptor, ThrowsOnWrongSpatialDimCountBwd)
{
    // MIOpen flags an ASAN error if improperly creating a conv descriptor.
    SKIP_IF_ASAN();

    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 0, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_data_sdk::data_objects::ConvMode::CROSS_CORRELATION;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreateConvolutionBwdAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::ConvolutionBwdAttributes>(
        builder.GetBufferPointer());

    EXPECT_THROW(MiopenConvDescriptor convDesc(
                     static_cast<size_t>(std::numeric_limits<int>::max()) + 1, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_THROW(MiopenConvDescriptor convDesc(prePadding.size() - 1, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_NO_THROW(MiopenConvDescriptor convDesc(prePadding.size(), *attrPtr, 1));
    EXPECT_THROW(MiopenConvDescriptor convDesc(prePadding.size() + 1, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenConvDescriptor, ThrowsOnWrongConvModeBwd)
{
    // MIOpen flags an ASAN error if improperly creating a conv descriptor.
    SKIP_IF_ASAN();

    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 0, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_data_sdk::data_objects::ConvMode::CONVOLUTION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreateConvolutionBwdAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::ConvolutionBwdAttributes>(
        builder.GetBufferPointer());

    EXPECT_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenConvDescriptor, ThrowsOnAsymmetricPaddingBwd)
{
    // MIOpen flags an ASAN error if improperly creating a conv descriptor.
    SKIP_IF_ASAN();

    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 1, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_data_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreateConvolutionBwdAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::ConvolutionBwdAttributes>(
        builder.GetBufferPointer());

    EXPECT_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenConvDescriptor, ThrowsOnWrongPaddingBwd)
{
    // MIOpen flags an ASAN error if improperly creating a conv descriptor.
    SKIP_IF_ASAN();

    const std::vector<int64_t> prePadding{0, -1, 0};
    const std::vector<int64_t> postPadding{0, -1, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_data_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreateConvolutionBwdAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::ConvolutionBwdAttributes>(
        builder.GetBufferPointer());

    EXPECT_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenConvDescriptor, ThrowsOnWrongStrideBwd)
{
    // MIOpen flags an ASAN error if improperly creating a conv descriptor.
    SKIP_IF_ASAN();

    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 0, 0};
    const std::vector<int64_t> stride{1, 0, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_data_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreateConvolutionBwdAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::ConvolutionBwdAttributes>(
        builder.GetBufferPointer());

    EXPECT_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenConvDescriptor, ThrowsOnWrongDilationBwd)
{
    // MIOpen flags an ASAN error if improperly creating a conv descriptor.
    SKIP_IF_ASAN();

    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 0, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 0, 1};
    const auto convMode = hipdnn_data_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreateConvolutionBwdAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::ConvolutionBwdAttributes>(
        builder.GetBufferPointer());

    EXPECT_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenConvDescriptor, CreateValidDescriptorWrw)
{
    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 0, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_data_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreateConvolutionWrwAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::ConvolutionWrwAttributes>(
        builder.GetBufferPointer());

    MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1);

    miopenStatus_t status;
    int returnedSpatialDimCount = 0;
    status = miopenGetConvolutionSpatialDim(convDesc.convDescriptor(), &returnedSpatialDimCount);
    EXPECT_EQ(status, miopenStatusSuccess);
    EXPECT_EQ(returnedSpatialDimCount, spatialDimCount);

    std::vector<int> returnedPadding(spatialDimCount);
    std::vector<int> returnedStride(spatialDimCount);
    std::vector<int> returnedDilation(spatialDimCount);
    status = miopenGetConvolutionNdDescriptor(convDesc.convDescriptor(),
                                              static_cast<int>(spatialDimCount),
                                              nullptr,
                                              returnedPadding.data(),
                                              returnedStride.data(),
                                              returnedDilation.data(),
                                              nullptr);
    EXPECT_EQ(status, miopenStatusSuccess);
    EXPECT_TRUE(std::equal(returnedPadding.begin(), returnedPadding.end(), prePadding.begin()));
    EXPECT_TRUE(std::equal(returnedStride.begin(), returnedStride.end(), stride.begin()));
    EXPECT_TRUE(std::equal(returnedDilation.begin(), returnedDilation.end(), dilation.begin()));
}

TEST(TestMiopenConvDescriptor, ThrowsOnWrongSpatialDimCountWrw)
{
    // MIOpen flags an ASAN error if improperly creating a conv descriptor.
    SKIP_IF_ASAN();

    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 0, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_data_sdk::data_objects::ConvMode::CROSS_CORRELATION;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreateConvolutionWrwAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::ConvolutionWrwAttributes>(
        builder.GetBufferPointer());

    EXPECT_THROW(MiopenConvDescriptor convDesc(
                     static_cast<size_t>(std::numeric_limits<int>::max()) + 1, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_THROW(MiopenConvDescriptor convDesc(prePadding.size() - 1, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_NO_THROW(MiopenConvDescriptor convDesc(prePadding.size(), *attrPtr, 1));
    EXPECT_THROW(MiopenConvDescriptor convDesc(prePadding.size() + 1, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenConvDescriptor, ThrowsOnWrongConvModeWrw)
{
    // MIOpen flags an ASAN error if improperly creating a conv descriptor.
    SKIP_IF_ASAN();

    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 0, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_data_sdk::data_objects::ConvMode::CONVOLUTION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreateConvolutionWrwAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::ConvolutionWrwAttributes>(
        builder.GetBufferPointer());

    EXPECT_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenConvDescriptor, ThrowsOnAsymmetricPaddingWrw)
{
    // MIOpen flags an ASAN error if improperly creating a conv descriptor.
    SKIP_IF_ASAN();

    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 1, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_data_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreateConvolutionWrwAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::ConvolutionWrwAttributes>(
        builder.GetBufferPointer());

    EXPECT_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenConvDescriptor, ThrowsOnWrongPaddingWrw)
{
    // MIOpen flags an ASAN error if improperly creating a conv descriptor.
    SKIP_IF_ASAN();

    const std::vector<int64_t> prePadding{0, -1, 0};
    const std::vector<int64_t> postPadding{0, -1, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_data_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreateConvolutionWrwAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::ConvolutionWrwAttributes>(
        builder.GetBufferPointer());

    EXPECT_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenConvDescriptor, ThrowsOnWrongStrideWrw)
{
    // MIOpen flags an ASAN error if improperly creating a conv descriptor.
    SKIP_IF_ASAN();

    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 0, 0};
    const std::vector<int64_t> stride{1, 0, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_data_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreateConvolutionWrwAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::ConvolutionWrwAttributes>(
        builder.GetBufferPointer());

    EXPECT_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenConvDescriptor, ThrowsOnWrongDilationWrw)
{
    // MIOpen flags an ASAN error if improperly creating a conv descriptor.
    SKIP_IF_ASAN();

    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 0, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 0, 1};
    const auto convMode = hipdnn_data_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreateConvolutionWrwAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::ConvolutionWrwAttributes>(
        builder.GetBufferPointer());

    EXPECT_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenConvDescriptor, AcceptsValidGroupCount)
{
    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 0, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_data_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreateConvolutionFwdAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::ConvolutionFwdAttributes>(
        builder.GetBufferPointer());

    EXPECT_NO_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1));
    EXPECT_NO_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 2));
    EXPECT_NO_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 4));
}

TEST(TestMiopenConvDescriptor, VerifiesGroupCountSetCorrectly)
{
    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 0, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_data_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 3;
    int groupCount = 4;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreateConvolutionFwdAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::ConvolutionFwdAttributes>(
        builder.GetBufferPointer());

    MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, groupCount);

    int returnedGroupCount = 0;
    miopenStatus_t status
        = miopenGetConvolutionGroupCount(convDesc.convDescriptor(), &returnedGroupCount);
    EXPECT_EQ(status, miopenStatusSuccess);
    EXPECT_EQ(returnedGroupCount, groupCount);
}

TEST(TestMiopenConvDescriptor, ThrowsOnInvalidGroupCount)
{
    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 0, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_data_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreateConvolutionFwdAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::ConvolutionFwdAttributes>(
        builder.GetBufferPointer());

    EXPECT_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 0),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, -1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}
