// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <miopen/miopen.h>

#include "MiopenActivationDescriptor.hpp"

using namespace miopen_legacy_plugin;

namespace
{

// Helper function to create PointwiseAttributes flatbuffer
flatbuffers::FlatBufferBuilder createPointwiseAttributesBuilder(
    hipdnn_data_sdk::data_objects::PointwiseMode mode,
    flatbuffers::Optional<float> reluLowerClip = flatbuffers::nullopt,
    flatbuffers::Optional<float> reluUpperClip = flatbuffers::nullopt,
    flatbuffers::Optional<float> reluLowerClipSlope = flatbuffers::nullopt,
    flatbuffers::Optional<float> eluAlpha = flatbuffers::nullopt,
    flatbuffers::Optional<float> softplusBeta = flatbuffers::nullopt)
{
    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_data_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        mode,
        reluLowerClip,
        reluUpperClip,
        reluLowerClipSlope,
        flatbuffers::nullopt, // axis
        0, // in_0_tensor_uid
        flatbuffers::nullopt, // in_1_tensor_uid
        flatbuffers::nullopt, // in_2_tensor_uid
        1, // out_0_tensor_uid
        flatbuffers::nullopt, // swish_beta
        eluAlpha,
        softplusBeta);
    builder.Finish(attrOffset);
    return builder;
}

} // namespace

TEST(TestMiopenActivationDescriptor, CreatesStandardRelu)
{
    auto builder
        = createPointwiseAttributesBuilder(hipdnn_data_sdk::data_objects::PointwiseMode::RELU_FWD);
    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    MiopenActivationDescriptor activDesc(*attr);

    miopenActivationMode_t mode;
    double alpha;
    double beta;
    double gamma;
    EXPECT_EQ(miopenGetActivationDescriptor(
                  activDesc.activationDescriptor(), &mode, &alpha, &beta, &gamma),
              miopenStatusSuccess);
    EXPECT_EQ(mode, miopenActivationRELU);
    EXPECT_DOUBLE_EQ(alpha, 0.0);
    EXPECT_DOUBLE_EQ(beta, 0.0);
    EXPECT_DOUBLE_EQ(gamma, 0.0);
}

TEST(TestMiopenActivationDescriptor, CreatesStandardReluBwd)
{
    auto builder
        = createPointwiseAttributesBuilder(hipdnn_data_sdk::data_objects::PointwiseMode::RELU_BWD);
    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    MiopenActivationDescriptor activDesc(*attr);

    miopenActivationMode_t mode;
    double alpha;
    double beta;
    double gamma;
    EXPECT_EQ(miopenGetActivationDescriptor(
                  activDesc.activationDescriptor(), &mode, &alpha, &beta, &gamma),
              miopenStatusSuccess);
    EXPECT_EQ(mode, miopenActivationRELU);
}

TEST(TestMiopenActivationDescriptor, CreatesClippedRelu)
{
    const float upperClip = 6.0f;
    auto builder = createPointwiseAttributesBuilder(
        hipdnn_data_sdk::data_objects::PointwiseMode::RELU_FWD, flatbuffers::nullopt, upperClip);
    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    MiopenActivationDescriptor activDesc(*attr);

    miopenActivationMode_t mode;
    double alpha;
    double beta;
    double gamma;
    EXPECT_EQ(miopenGetActivationDescriptor(
                  activDesc.activationDescriptor(), &mode, &alpha, &beta, &gamma),
              miopenStatusSuccess);
    EXPECT_EQ(mode, miopenActivationCLIPPEDRELU);
    EXPECT_DOUBLE_EQ(alpha, static_cast<double>(upperClip));
}

TEST(TestMiopenActivationDescriptor, CreatesLeakyRelu)
{
    const float slope = 0.01f;
    auto builder
        = createPointwiseAttributesBuilder(hipdnn_data_sdk::data_objects::PointwiseMode::RELU_BWD,
                                           flatbuffers::nullopt,
                                           flatbuffers::nullopt,
                                           slope);
    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    MiopenActivationDescriptor activDesc(*attr);

    miopenActivationMode_t mode;
    double alpha;
    double beta;
    double gamma;
    EXPECT_EQ(miopenGetActivationDescriptor(
                  activDesc.activationDescriptor(), &mode, &alpha, &beta, &gamma),
              miopenStatusSuccess);
    EXPECT_EQ(mode, miopenActivationLEAKYRELU);
    EXPECT_DOUBLE_EQ(alpha, static_cast<double>(slope));
}

TEST(TestMiopenActivationDescriptor, CreatesClamp)
{
    const float lowerClip = 1.0f;
    const float upperClip = 6.0f;
    auto builder = createPointwiseAttributesBuilder(
        hipdnn_data_sdk::data_objects::PointwiseMode::RELU_FWD, lowerClip, upperClip);
    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    MiopenActivationDescriptor activDesc(*attr);

    miopenActivationMode_t mode;
    double alpha;
    double beta;
    double gamma;
    EXPECT_EQ(miopenGetActivationDescriptor(
                  activDesc.activationDescriptor(), &mode, &alpha, &beta, &gamma),
              miopenStatusSuccess);
    EXPECT_EQ(mode, miopenActivationCLAMP);
    EXPECT_DOUBLE_EQ(alpha, static_cast<double>(lowerClip));
    EXPECT_DOUBLE_EQ(beta, static_cast<double>(upperClip));
}

TEST(TestMiopenActivationDescriptor, CreatesSigmoidFwd)
{
    auto builder = createPointwiseAttributesBuilder(
        hipdnn_data_sdk::data_objects::PointwiseMode::SIGMOID_FWD);
    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    MiopenActivationDescriptor activDesc(*attr);

    miopenActivationMode_t mode;
    double alpha;
    double beta;
    double gamma;
    EXPECT_EQ(miopenGetActivationDescriptor(
                  activDesc.activationDescriptor(), &mode, &alpha, &beta, &gamma),
              miopenStatusSuccess);
    EXPECT_EQ(mode, miopenActivationLOGISTIC);
}

TEST(TestMiopenActivationDescriptor, CreatesSigmoidBwd)
{
    auto builder = createPointwiseAttributesBuilder(
        hipdnn_data_sdk::data_objects::PointwiseMode::SIGMOID_BWD);
    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    MiopenActivationDescriptor activDesc(*attr);

    miopenActivationMode_t mode;
    double alpha;
    double beta;
    double gamma;
    EXPECT_EQ(miopenGetActivationDescriptor(
                  activDesc.activationDescriptor(), &mode, &alpha, &beta, &gamma),
              miopenStatusSuccess);
    EXPECT_EQ(mode, miopenActivationLOGISTIC);
}

TEST(TestMiopenActivationDescriptor, CreatesTanhFwd)
{
    auto builder
        = createPointwiseAttributesBuilder(hipdnn_data_sdk::data_objects::PointwiseMode::TANH_FWD);
    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    MiopenActivationDescriptor activDesc(*attr);

    miopenActivationMode_t mode;
    double alpha;
    double beta;
    double gamma;
    EXPECT_EQ(miopenGetActivationDescriptor(
                  activDesc.activationDescriptor(), &mode, &alpha, &beta, &gamma),
              miopenStatusSuccess);
    EXPECT_EQ(mode, miopenActivationTANH);
    EXPECT_DOUBLE_EQ(alpha, 1.0);
    EXPECT_DOUBLE_EQ(beta, 1.0);
}

TEST(TestMiopenActivationDescriptor, CreatesTanhBwd)
{
    auto builder
        = createPointwiseAttributesBuilder(hipdnn_data_sdk::data_objects::PointwiseMode::TANH_BWD);
    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    MiopenActivationDescriptor activDesc(*attr);

    miopenActivationMode_t mode;
    double alpha;
    double beta;
    double gamma;
    EXPECT_EQ(miopenGetActivationDescriptor(
                  activDesc.activationDescriptor(), &mode, &alpha, &beta, &gamma),
              miopenStatusSuccess);
    EXPECT_EQ(mode, miopenActivationTANH);
}

TEST(TestMiopenActivationDescriptor, CreatesEluWithCustomAlpha)
{
    const float eluAlpha = 2.0f;
    auto builder
        = createPointwiseAttributesBuilder(hipdnn_data_sdk::data_objects::PointwiseMode::ELU_FWD,
                                           flatbuffers::nullopt,
                                           flatbuffers::nullopt,
                                           flatbuffers::nullopt,
                                           eluAlpha);
    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    MiopenActivationDescriptor activDesc(*attr);

    miopenActivationMode_t mode;
    double alpha;
    double beta;
    double gamma;
    EXPECT_EQ(miopenGetActivationDescriptor(
                  activDesc.activationDescriptor(), &mode, &alpha, &beta, &gamma),
              miopenStatusSuccess);
    EXPECT_EQ(mode, miopenActivationELU);
    EXPECT_DOUBLE_EQ(alpha, static_cast<double>(eluAlpha));
}

TEST(TestMiopenActivationDescriptor, CreatesEluWithDefaultAlpha)
{
    auto builder
        = createPointwiseAttributesBuilder(hipdnn_data_sdk::data_objects::PointwiseMode::ELU_BWD);
    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    MiopenActivationDescriptor activDesc(*attr);

    miopenActivationMode_t mode;
    double alpha;
    double beta;
    double gamma;
    EXPECT_EQ(miopenGetActivationDescriptor(
                  activDesc.activationDescriptor(), &mode, &alpha, &beta, &gamma),
              miopenStatusSuccess);
    EXPECT_EQ(mode, miopenActivationELU);
    EXPECT_DOUBLE_EQ(alpha, 1.0);
}

TEST(TestMiopenActivationDescriptor, CreatesSoftplusFwdWithoutBeta)
{
    auto builder = createPointwiseAttributesBuilder(
        hipdnn_data_sdk::data_objects::PointwiseMode::SOFTPLUS_FWD);
    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    MiopenActivationDescriptor activDesc(*attr);

    miopenActivationMode_t mode;
    double alpha;
    double beta;
    double gamma;
    EXPECT_EQ(miopenGetActivationDescriptor(
                  activDesc.activationDescriptor(), &mode, &alpha, &beta, &gamma),
              miopenStatusSuccess);
    EXPECT_EQ(mode, miopenActivationSOFTRELU);
}

TEST(TestMiopenActivationDescriptor, CreatesSoftplusBwdWithoutBeta)
{
    auto builder = createPointwiseAttributesBuilder(
        hipdnn_data_sdk::data_objects::PointwiseMode::SOFTPLUS_BWD);
    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    MiopenActivationDescriptor activDesc(*attr);

    miopenActivationMode_t mode;
    double alpha;
    double beta;
    double gamma;
    EXPECT_EQ(miopenGetActivationDescriptor(
                  activDesc.activationDescriptor(), &mode, &alpha, &beta, &gamma),
              miopenStatusSuccess);
    EXPECT_EQ(mode, miopenActivationSOFTRELU);
}

TEST(TestMiopenActivationDescriptor, CreatesSoftplusWithBetaOne)
{
    const float softplusBeta = 1.0f;
    auto builder = createPointwiseAttributesBuilder(
        hipdnn_data_sdk::data_objects::PointwiseMode::SOFTPLUS_FWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        softplusBeta);
    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    MiopenActivationDescriptor activDesc(*attr);

    miopenActivationMode_t mode;
    double alpha;
    double beta;
    double gamma;
    EXPECT_EQ(miopenGetActivationDescriptor(
                  activDesc.activationDescriptor(), &mode, &alpha, &beta, &gamma),
              miopenStatusSuccess);
    EXPECT_EQ(mode, miopenActivationSOFTRELU);
}

TEST(TestMiopenActivationDescriptor, ThrowsOnSoftplusWithInvalidBeta)
{
    const float softplusBeta = 2.0f;
    auto builder = createPointwiseAttributesBuilder(
        hipdnn_data_sdk::data_objects::PointwiseMode::SOFTPLUS_BWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        softplusBeta);
    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    EXPECT_THROW(MiopenActivationDescriptor activDesc(*attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenActivationDescriptor, CreatesAbs)
{
    auto builder
        = createPointwiseAttributesBuilder(hipdnn_data_sdk::data_objects::PointwiseMode::ABS);
    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    MiopenActivationDescriptor activDesc(*attr);

    miopenActivationMode_t mode;
    double alpha;
    double beta;
    double gamma;
    EXPECT_EQ(miopenGetActivationDescriptor(
                  activDesc.activationDescriptor(), &mode, &alpha, &beta, &gamma),
              miopenStatusSuccess);
    EXPECT_EQ(mode, miopenActivationABS);
}

TEST(TestMiopenActivationDescriptor, CreatesIdentity)
{
    auto builder
        = createPointwiseAttributesBuilder(hipdnn_data_sdk::data_objects::PointwiseMode::IDENTITY);
    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    MiopenActivationDescriptor activDesc(*attr);

    miopenActivationMode_t mode;
    double alpha;
    double beta;
    double gamma;
    EXPECT_EQ(miopenGetActivationDescriptor(
                  activDesc.activationDescriptor(), &mode, &alpha, &beta, &gamma),
              miopenStatusSuccess);
    EXPECT_EQ(mode, miopenActivationPASTHRU);
}

TEST(TestMiopenActivationDescriptor, ThrowsOnUnsupportedMode)
{
    auto builder
        = createPointwiseAttributesBuilder(hipdnn_data_sdk::data_objects::PointwiseMode::ADD);
    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    EXPECT_THROW(MiopenActivationDescriptor activDesc(*attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenActivationDescriptor, ClampTakesPrecedenceOverClippedRelu)
{
    // When both lower and upper clips are present, should use CLAMP, not CLIPPED_RELU
    const float lowerClip = -1.0f;
    const float upperClip = 6.0f;
    auto builder = createPointwiseAttributesBuilder(
        hipdnn_data_sdk::data_objects::PointwiseMode::RELU_BWD, lowerClip, upperClip);
    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    MiopenActivationDescriptor activDesc(*attr);

    miopenActivationMode_t mode;
    double alpha;
    double beta;
    double gamma;
    EXPECT_EQ(miopenGetActivationDescriptor(
                  activDesc.activationDescriptor(), &mode, &alpha, &beta, &gamma),
              miopenStatusSuccess);
    EXPECT_EQ(mode, miopenActivationCLAMP);
    EXPECT_DOUBLE_EQ(alpha, static_cast<double>(lowerClip));
    EXPECT_DOUBLE_EQ(beta, static_cast<double>(upperClip));
}

TEST(TestMiopenActivationDescriptor, UpperClipWithoutLowerUsesClippedRelu)
{
    // Only upper clip present should use CLIPPED_RELU, not CLAMP
    const float upperClip = 6.0f;
    auto builder = createPointwiseAttributesBuilder(
        hipdnn_data_sdk::data_objects::PointwiseMode::RELU_FWD, flatbuffers::nullopt, upperClip);
    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    MiopenActivationDescriptor activDesc(*attr);

    miopenActivationMode_t mode;
    double alpha;
    double beta;
    double gamma;
    EXPECT_EQ(miopenGetActivationDescriptor(
                  activDesc.activationDescriptor(), &mode, &alpha, &beta, &gamma),
              miopenStatusSuccess);
    EXPECT_EQ(mode, miopenActivationCLIPPEDRELU);
    EXPECT_DOUBLE_EQ(alpha, static_cast<double>(upperClip));
}
