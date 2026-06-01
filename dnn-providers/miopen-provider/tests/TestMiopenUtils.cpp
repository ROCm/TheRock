// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "MiopenUtils.hpp"
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

using namespace miopen_legacy_plugin;
using namespace miopen_utils;

TEST(TestMiopenUtils, FindDeviceBufferReturnsCorrectBuffer)
{
    std::vector<hipdnnPluginDeviceBuffer_t> buffers
        = {{42, reinterpret_cast<void*>(0x1234)}, {99, reinterpret_cast<void*>(0x5678)}};

    auto result = miopen_utils::findDeviceBuffer(99, buffers.data(), 2);
    EXPECT_EQ(result.uid, 99);
    EXPECT_EQ(result.ptr, reinterpret_cast<void*>(0x5678));
}

TEST(TestMiopenUtils, FindDeviceBufferThrowsIfNotFound)
{
    std::vector<hipdnnPluginDeviceBuffer_t> buffers = {{1, reinterpret_cast<void*>(0x1111)}};

    EXPECT_THROW(
        miopen_utils::findDeviceBuffer(2, buffers.data(), static_cast<uint32_t>(buffers.size())),
        hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenUtils, TensorDataTypeToMiopenDataType)
{
    using namespace hipdnn_data_sdk::data_objects;

    EXPECT_EQ(miopen_utils::tensorDataTypeToMiopenDataType(DataType::FLOAT), miopenFloat);
    EXPECT_EQ(miopen_utils::tensorDataTypeToMiopenDataType(DataType::HALF), miopenHalf);
    EXPECT_EQ(miopen_utils::tensorDataTypeToMiopenDataType(DataType::BFLOAT16), miopenBFloat16);
}

TEST(TestMiopenUtils, TensorDataTypeToMiopenDataTypeThrowsOnUnsupported)
{
    // Use a value not in the enum
    EXPECT_THROW(miopen_utils::tensorDataTypeToMiopenDataType(
                     static_cast<hipdnn_data_sdk::data_objects::DataType>(-1)),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenUtils, FindTensorAttributesReturnsCorrectValue)
{
    flatbuffers::FlatBufferBuilder builder1;
    auto attrOffset1 = hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(builder1, 1);
    builder1.Finish(attrOffset1);

    flatbuffers::FlatBufferBuilder builder2;
    auto attrOffset2 = hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(builder2, 2);
    builder2.Finish(attrOffset2);

    auto attrPtr1 = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::TensorAttributes>(
        builder1.GetBufferPointer());
    auto attrPtr2 = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::TensorAttributes>(
        builder2.GetBufferPointer());

    auto attrMap
        = std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>{
            {1, attrPtr1}, {2, attrPtr2}};

    EXPECT_EQ(miopen_utils::findTensorAttributes(attrMap, 1).uid(), 1);
    EXPECT_EQ(miopen_utils::findTensorAttributes(attrMap, 2).uid(), 2);
}

TEST(TestMiopenUtils, FindTensorAttributesThrowsIfNotFound)
{
    auto attrMap
        = std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>{};

    EXPECT_THROW(miopen_utils::findTensorAttributes(attrMap, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenUtils, GetSpatialDimCountReturnsCorrectValue)
{
    std::vector<int64_t> dims = {1, 1, 1, 1, 1};

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "", hipdnn_data_sdk::data_objects::DataType::UNSET, nullptr, &dims);
    builder.Finish(attrOffset);

    auto attrPtr1 = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::TensorAttributes>(
        builder.GetBufferPointer());

    EXPECT_EQ(miopen_utils::getSpatialDimCount(*attrPtr1), 3);
}

TEST(TestMiopenUtils, GetSpatialDimCountThrowsOnInvalidDims)
{
    std::vector<int64_t> dims = {1, 1};

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "", hipdnn_data_sdk::data_objects::DataType::UNSET, nullptr, &dims);
    builder.Finish(attrOffset);

    auto attrPtr1 = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::TensorAttributes>(
        builder.GetBufferPointer());

    EXPECT_THROW(miopen_utils::getSpatialDimCount(*attrPtr1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenUtils, MapPointwiseModeStandardRelu)
{
    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreatePointwiseAttributes(
        builder, hipdnn_data_sdk::data_objects::PointwiseMode::RELU_FWD);
    builder.Finish(attrOffset);
    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    ActivationParams result;
    ASSERT_NO_THROW(result = mapPointwiseModeToMiopenActivation(*attr));
    EXPECT_EQ(result.mode, miopenActivationRELU);
    EXPECT_DOUBLE_EQ(result.alpha, 0.0);
}

TEST(TestMiopenUtils, MapPointwiseModeReluNonZeroLowerClipOnly)
{
    const float lowerClip = 0.1f;
    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreatePointwiseAttributes(
        builder, hipdnn_data_sdk::data_objects::PointwiseMode::RELU_FWD, lowerClip);
    builder.Finish(attrOffset);
    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    ASSERT_THROW(mapPointwiseModeToMiopenActivation(*attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenUtils, MapPointwiseModeReluZeroLowerClipOnly)
{
    const float lowerClip = 0.0f;
    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreatePointwiseAttributes(
        builder, hipdnn_data_sdk::data_objects::PointwiseMode::RELU_FWD, lowerClip);
    builder.Finish(attrOffset);
    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    ActivationParams result;
    ASSERT_NO_THROW(result = mapPointwiseModeToMiopenActivation(*attr));

    EXPECT_DOUBLE_EQ(result.alpha, static_cast<double>(lowerClip));
}

TEST(TestMiopenUtils, MapPointwiseModeClippedRelu)
{
    const float upperClip = 6.0f;
    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_data_sdk::data_objects::PointwiseMode::RELU_BWD,
        flatbuffers::nullopt,
        upperClip);
    builder.Finish(attrOffset);
    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    ActivationParams result;
    ASSERT_NO_THROW(result = miopen_utils::mapPointwiseModeToMiopenActivation(*attr));
    EXPECT_EQ(result.mode, miopenActivationCLIPPEDRELU);
    EXPECT_DOUBLE_EQ(result.alpha, static_cast<double>(upperClip));
}

TEST(TestMiopenUtils, MapPointwiseModeLeakyRelu)
{
    const float slope = 0.01f;
    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_data_sdk::data_objects::PointwiseMode::RELU_FWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        slope);
    builder.Finish(attrOffset);
    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    ActivationParams result;
    ASSERT_NO_THROW(result = miopen_utils::mapPointwiseModeToMiopenActivation(*attr));
    EXPECT_EQ(result.mode, miopenActivationLEAKYRELU);
    EXPECT_DOUBLE_EQ(result.alpha, static_cast<double>(slope));
}

TEST(TestMiopenUtils, MapPointwiseModeClamp)
{
    const float lowerClip = -1.0f;
    const float upperClip = 6.0f;
    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreatePointwiseAttributes(
        builder, hipdnn_data_sdk::data_objects::PointwiseMode::RELU_BWD, lowerClip, upperClip);
    builder.Finish(attrOffset);
    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    ActivationParams result;
    ASSERT_NO_THROW(result = miopen_utils::mapPointwiseModeToMiopenActivation(*attr));
    EXPECT_EQ(result.mode, miopenActivationCLAMP);
    EXPECT_DOUBLE_EQ(result.alpha, static_cast<double>(lowerClip));
    EXPECT_DOUBLE_EQ(result.beta, static_cast<double>(upperClip));
}

TEST(TestMiopenUtils, MapPointwiseModeSigmoid)
{
    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreatePointwiseAttributes(
        builder, hipdnn_data_sdk::data_objects::PointwiseMode::SIGMOID_FWD);
    builder.Finish(attrOffset);
    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    ActivationParams result;
    ASSERT_NO_THROW(result = miopen_utils::mapPointwiseModeToMiopenActivation(*attr));
    EXPECT_EQ(result.mode, miopenActivationLOGISTIC);
}

TEST(TestMiopenUtils, MapPointwiseModeTanh)
{
    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreatePointwiseAttributes(
        builder, hipdnn_data_sdk::data_objects::PointwiseMode::TANH_BWD);
    builder.Finish(attrOffset);
    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    ActivationParams result;
    ASSERT_NO_THROW(result = miopen_utils::mapPointwiseModeToMiopenActivation(*attr));
    EXPECT_EQ(result.mode, miopenActivationTANH);
    EXPECT_DOUBLE_EQ(result.alpha, 1.0);
    EXPECT_DOUBLE_EQ(result.beta, 1.0);
}

TEST(TestMiopenUtils, MapPointwiseModeEluWithCustomAlpha)
{
    const float eluAlpha = 2.0f;
    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_data_sdk::data_objects::PointwiseMode::ELU_FWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        0,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        1,
        flatbuffers::nullopt,
        eluAlpha);
    builder.Finish(attrOffset);
    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    ActivationParams result;
    ASSERT_NO_THROW(result = miopen_utils::mapPointwiseModeToMiopenActivation(*attr));
    EXPECT_EQ(result.mode, miopenActivationELU);
    EXPECT_DOUBLE_EQ(result.alpha, static_cast<double>(eluAlpha));
}

TEST(TestMiopenUtils, MapPointwiseModeEluWithDefaultAlpha)
{
    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreatePointwiseAttributes(
        builder, hipdnn_data_sdk::data_objects::PointwiseMode::ELU_BWD);
    builder.Finish(attrOffset);
    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    ActivationParams result;
    ASSERT_NO_THROW(result = miopen_utils::mapPointwiseModeToMiopenActivation(*attr));
    EXPECT_EQ(result.mode, miopenActivationELU);
    EXPECT_DOUBLE_EQ(result.alpha, 1.0);
}

TEST(TestMiopenUtils, MapPointwiseModeSoftplusWithoutBeta)
{
    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreatePointwiseAttributes(
        builder, hipdnn_data_sdk::data_objects::PointwiseMode::SOFTPLUS_FWD);
    builder.Finish(attrOffset);
    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    ActivationParams result;
    ASSERT_NO_THROW(result = miopen_utils::mapPointwiseModeToMiopenActivation(*attr));
    EXPECT_EQ(result.mode, miopenActivationSOFTRELU);
}

TEST(TestMiopenUtils, MapPointwiseModeSoftplusWithBetaOne)
{
    const float beta = 1.0f;
    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_data_sdk::data_objects::PointwiseMode::SOFTPLUS_BWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        0,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        1,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        beta);
    builder.Finish(attrOffset);
    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    ActivationParams result;
    ASSERT_NO_THROW(result = miopen_utils::mapPointwiseModeToMiopenActivation(*attr));
    EXPECT_EQ(result.mode, miopenActivationSOFTRELU);
}

TEST(TestMiopenUtils, MapPointwiseModeSoftplusWithInvalidBeta)
{
    const float beta = 2.0f;
    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_data_sdk::data_objects::PointwiseMode::SOFTPLUS_FWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        0,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        1,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        beta);
    builder.Finish(attrOffset);
    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    ASSERT_THROW(miopen_utils::mapPointwiseModeToMiopenActivation(*attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenUtils, MapPointwiseModeAbs)
{
    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreatePointwiseAttributes(
        builder, hipdnn_data_sdk::data_objects::PointwiseMode::ABS);
    builder.Finish(attrOffset);
    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    ActivationParams result;
    ASSERT_NO_THROW(result = miopen_utils::mapPointwiseModeToMiopenActivation(*attr));
    EXPECT_EQ(result.mode, miopenActivationABS);
}

TEST(TestMiopenUtils, MapPointwiseModeIdentity)
{
    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreatePointwiseAttributes(
        builder, hipdnn_data_sdk::data_objects::PointwiseMode::IDENTITY);
    builder.Finish(attrOffset);
    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    ActivationParams result;
    ASSERT_NO_THROW(result = miopen_utils::mapPointwiseModeToMiopenActivation(*attr));
    EXPECT_EQ(result.mode, miopenActivationPASTHRU);
}

TEST(TestMiopenUtils, MapPointwiseModeUnsupported)
{
    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreatePointwiseAttributes(
        builder, hipdnn_data_sdk::data_objects::PointwiseMode::ADD);
    builder.Finish(attrOffset);
    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    ASSERT_THROW(miopen_utils::mapPointwiseModeToMiopenActivation(*attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}
