// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/plans/MiopenBatchnormFwdTrainingPlan.hpp"
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using namespace miopen_legacy_plugin;

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST(TestBatchnormFwdTrainingActivParams, InitializesRequiredTensorsFromValidGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingActivGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& bnNode = graph.getNode(0);
    auto* bnAttrs = bnNode.attributes_as_BatchnormAttributes();
    ASSERT_NE(bnAttrs, nullptr);

    const auto& activNode = graph.getNode(1);
    auto* activAttrs = activNode.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttrs, nullptr);

    BatchnormFwdTrainingParams params(*bnAttrs, *activAttrs, graph.getTensorMap());

    // All required tensors should be initialized
    EXPECT_NO_THROW(params.x());
    EXPECT_NO_THROW(params.y());
    EXPECT_NO_THROW(params.scale());
    EXPECT_NO_THROW(params.bias());
    EXPECT_TRUE(params.optActivation().has_value());
}

TEST(TestBatchnormFwdTrainingActivParams, ExtractsEpsilonValueCorrectly)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingActivGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& bnNode = graph.getNode(0);
    auto* bnAttrs = bnNode.attributes_as_BatchnormAttributes();
    ASSERT_NE(bnAttrs, nullptr);

    const auto& activNode = graph.getNode(1);
    auto* activAttrs = activNode.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttrs, nullptr);

    BatchnormFwdTrainingParams params(*bnAttrs, *activAttrs, graph.getTensorMap());

    // Epsilon should be extracted as double
    EXPECT_NEAR(params.epsilonValue(), 1e-5, 1e-10);
}

// ============================================================================
// Optional Feature Tests (Mean/Variance and Running Statistics)
// ============================================================================

TEST(TestBatchnormFwdTrainingActivParams, HandlesMeanVariancePresent)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingActivGraph(true);
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& bnNode = graph.getNode(0);
    auto* bnAttrs = bnNode.attributes_as_BatchnormAttributes();
    ASSERT_NE(bnAttrs, nullptr);

    const auto& activNode = graph.getNode(1);
    auto* activAttrs = activNode.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttrs, nullptr);

    BatchnormFwdTrainingParams params(*bnAttrs, *activAttrs, graph.getTensorMap());

    EXPECT_TRUE(params.hasSaveMeanVariance());
    EXPECT_NO_THROW(params.mean());
    EXPECT_NO_THROW(params.invVariance());
}

TEST(TestBatchnormFwdTrainingActivParams, HandlesMeanVarianceMissing)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingActivGraph(false);
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& bnNode = graph.getNode(0);
    auto* bnAttrs = bnNode.attributes_as_BatchnormAttributes();
    ASSERT_NE(bnAttrs, nullptr);

    const auto& activNode = graph.getNode(1);
    auto* activAttrs = activNode.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttrs, nullptr);

    BatchnormFwdTrainingParams params(*bnAttrs, *activAttrs, graph.getTensorMap());

    EXPECT_FALSE(params.hasSaveMeanVariance());
}

TEST(TestBatchnormFwdTrainingActivParams, ThrowsWhenRunningStatsProvided)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingActivGraph(
        true, true); // with mean/variance and running stats
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& bnNode = graph.getNode(0);
    auto* bnAttrs = bnNode.attributes_as_BatchnormAttributes();
    ASSERT_NE(bnAttrs, nullptr);

    const auto& activNode = graph.getNode(1);
    auto* activAttrs = activNode.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttrs, nullptr);

    // Should throw because running stats are provided
    EXPECT_THROW(BatchnormFwdTrainingParams(*bnAttrs, *activAttrs, graph.getTensorMap()),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestBatchnormFwdTrainingActivParams, HasRunningStatsReturnsFalseWhenNotProvided)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingActivGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& bnNode = graph.getNode(0);
    auto* bnAttrs = bnNode.attributes_as_BatchnormAttributes();
    ASSERT_NE(bnAttrs, nullptr);

    const auto& activNode = graph.getNode(1);
    auto* activAttrs = activNode.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttrs, nullptr);

    BatchnormFwdTrainingParams params(*bnAttrs, *activAttrs, graph.getTensorMap());

    EXPECT_FALSE(params.hasRunningStats());
}

// ============================================================================
// Missing Tensor Tests
// ============================================================================

TEST(TestBatchnormFwdTrainingActivParams, ThrowsStdOutOfRangeForMissingEpsilonTensor)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_data_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    std::vector<int64_t> strides = {1, 3, 14, 14};
    std::vector<int64_t> dims = {1, 3, 14, 14};
    std::vector<int64_t> derivedStrides = {1, 3, 1, 1};
    std::vector<int64_t> derivedDims = {1, 3, 1, 1};

    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", hipdnn_data_sdk::data_objects::DataType::FLOAT, &strides, &dims));
    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "y", hipdnn_data_sdk::data_objects::DataType::FLOAT, &strides, &dims, true));
    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        3,
        "scale",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));
    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        4,
        "bias",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    // Missing epsilon tensor (ID 5)

    std::vector<::flatbuffers::Offset<hipdnn_data_sdk::data_objects::Node>> nodes;

    // BN node referencing non-existent epsilon tensor
    auto bnormAttributes = hipdnn_data_sdk::data_objects::CreateBatchnormAttributes(
        builder,
        1,
        3,
        4,
        5, // Epsilon tensor that doesn't exist
        0,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        2,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt);
    nodes.push_back(hipdnn_data_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormAttributes,
        bnormAttributes.Union()));

    auto actAttr = hipdnn_data_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_data_sdk::data_objects::PointwiseMode::RELU_FWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        2,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        3);
    nodes.push_back(hipdnn_data_sdk::data_objects::CreateNodeDirect(
        builder,
        "act",
        hipdnn_data_sdk::data_objects::DataType::UNSET,
        hipdnn_data_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        actAttr.Union()));

    auto graphOffset = hipdnn_data_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::DataType::HALF,
        hipdnn_data_sdk::data_objects::DataType::BFLOAT16,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);

    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& bnNode = graph.getNode(0);
    auto* bnAttrs = bnNode.attributes_as_BatchnormAttributes();
    ASSERT_NE(bnAttrs, nullptr);

    const auto& actNode = graph.getNode(1);
    auto* activAttrs = actNode.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttrs, nullptr);

    // Constructor uses tensorMap.at() which throws std::out_of_range for missing key
    EXPECT_THROW(BatchnormFwdTrainingParams(*bnAttrs, *activAttrs, graph.getTensorMap()),
                 std::out_of_range);
}

// ============================================================================
// Tensor Connectivity Validation Tests
// ============================================================================

TEST(TestBatchnormFwdTrainingActivParams, ThrowsWhenBnOutputDoesNotMatchActivationInput)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_data_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    std::vector<int64_t> strides = {1, 3, 14, 14};
    std::vector<int64_t> dims = {1, 3, 14, 14};
    std::vector<int64_t> derivedStrides = {1, 3, 1, 1};
    std::vector<int64_t> derivedDims = {1, 3, 1, 1};

    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", hipdnn_data_sdk::data_objects::DataType::FLOAT, &strides, &dims));
    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "y_bn", hipdnn_data_sdk::data_objects::DataType::FLOAT, &strides, &dims, true));
    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        6,
        "y_wrong",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        &strides,
        &dims,
        true));
    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        3,
        "scale",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));
    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        4,
        "bias",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    hipdnn_data_sdk::data_objects::Float32Value epsilonVal(1e-5f);
    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributes(
        builder,
        5,
        builder.CreateString("epsilon"),
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        0,
        0,
        false,
        hipdnn_data_sdk::data_objects::TensorValue::Float32Value,
        builder.CreateStruct(epsilonVal).Union()));

    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 7, "final_out", hipdnn_data_sdk::data_objects::DataType::FLOAT, &strides, &dims));

    std::vector<::flatbuffers::Offset<hipdnn_data_sdk::data_objects::Node>> nodes;

    // BN node with output tensor 2
    auto bnormAttributes
        = hipdnn_data_sdk::data_objects::CreateBatchnormAttributes(builder,
                                                                   1,
                                                                   3,
                                                                   4,
                                                                   5,
                                                                   0,
                                                                   flatbuffers::nullopt,
                                                                   flatbuffers::nullopt,
                                                                   flatbuffers::nullopt,
                                                                   2, // BN output
                                                                   flatbuffers::nullopt,
                                                                   flatbuffers::nullopt,
                                                                   flatbuffers::nullopt,
                                                                   flatbuffers::nullopt);
    nodes.push_back(hipdnn_data_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormAttributes,
        bnormAttributes.Union()));

    // Activation with input tensor 6 (wrong - should be 2)
    auto actAttr = hipdnn_data_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_data_sdk::data_objects::PointwiseMode::RELU_FWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        6, // Wrong input - doesn't match BN output
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        7);
    nodes.push_back(hipdnn_data_sdk::data_objects::CreateNodeDirect(
        builder,
        "act",
        hipdnn_data_sdk::data_objects::DataType::UNSET,
        hipdnn_data_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        actAttr.Union()));

    auto graphOffset = hipdnn_data_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::DataType::HALF,
        hipdnn_data_sdk::data_objects::DataType::BFLOAT16,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);

    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& bnNode = graph.getNode(0);
    auto* bnAttrs = bnNode.attributes_as_BatchnormAttributes();
    ASSERT_NE(bnAttrs, nullptr);

    const auto& actNode = graph.getNode(1);
    auto* activAttrs = actNode.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttrs, nullptr);

    EXPECT_THROW(BatchnormFwdTrainingParams(*bnAttrs, *activAttrs, graph.getTensorMap()),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestBatchnormFwdTrainingActivParams, ThrowsForMissingRequiredXTensor)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_data_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    std::vector<int64_t> strides = {1, 3, 14, 14};
    std::vector<int64_t> dims = {1, 3, 14, 14};
    std::vector<int64_t> derivedStrides = {1, 3, 1, 1};
    std::vector<int64_t> derivedDims = {1, 3, 1, 1};

    // Missing x tensor (ID 1)
    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "y", hipdnn_data_sdk::data_objects::DataType::FLOAT, &strides, &dims, true));
    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        3,
        "scale",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));
    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        4,
        "bias",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    hipdnn_data_sdk::data_objects::Float32Value epsilonVal(1e-5f);
    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributes(
        builder,
        5,
        builder.CreateString("epsilon"),
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        0,
        0,
        false,
        hipdnn_data_sdk::data_objects::TensorValue::Float32Value,
        builder.CreateStruct(epsilonVal).Union()));

    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 7, "final_out", hipdnn_data_sdk::data_objects::DataType::FLOAT, &strides, &dims));

    std::vector<::flatbuffers::Offset<hipdnn_data_sdk::data_objects::Node>> nodes;

    // BN node referencing missing x tensor
    auto bnormAttributes
        = hipdnn_data_sdk::data_objects::CreateBatchnormAttributes(builder,
                                                                   1, // Missing x tensor
                                                                   3,
                                                                   4,
                                                                   5,
                                                                   0,
                                                                   flatbuffers::nullopt,
                                                                   flatbuffers::nullopt,
                                                                   flatbuffers::nullopt,
                                                                   2,
                                                                   flatbuffers::nullopt,
                                                                   flatbuffers::nullopt,
                                                                   flatbuffers::nullopt,
                                                                   flatbuffers::nullopt);
    nodes.push_back(hipdnn_data_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormAttributes,
        bnormAttributes.Union()));

    auto actAttr = hipdnn_data_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_data_sdk::data_objects::PointwiseMode::RELU_FWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        2,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        7);
    nodes.push_back(hipdnn_data_sdk::data_objects::CreateNodeDirect(
        builder,
        "act",
        hipdnn_data_sdk::data_objects::DataType::UNSET,
        hipdnn_data_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        actAttr.Union()));

    auto graphOffset = hipdnn_data_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::DataType::HALF,
        hipdnn_data_sdk::data_objects::DataType::BFLOAT16,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);

    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& bnNode = graph.getNode(0);
    auto* bnAttrs = bnNode.attributes_as_BatchnormAttributes();
    ASSERT_NE(bnAttrs, nullptr);

    const auto& actNode = graph.getNode(1);
    auto* activAttrs = actNode.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttrs, nullptr);

    EXPECT_THROW(BatchnormFwdTrainingParams(*bnAttrs, *activAttrs, graph.getTensorMap()),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestBatchnormFwdTrainingActivParams, ThrowsForMissingRequiredScaleTensor)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_data_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    std::vector<int64_t> strides = {1, 3, 14, 14};
    std::vector<int64_t> dims = {1, 3, 14, 14};
    std::vector<int64_t> derivedStrides = {1, 3, 1, 1};
    std::vector<int64_t> derivedDims = {1, 3, 1, 1};

    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", hipdnn_data_sdk::data_objects::DataType::FLOAT, &strides, &dims));
    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "y", hipdnn_data_sdk::data_objects::DataType::FLOAT, &strides, &dims, true));
    // Missing scale tensor (ID 3)
    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        4,
        "bias",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    hipdnn_data_sdk::data_objects::Float32Value epsilonVal(1e-5f);
    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributes(
        builder,
        5,
        builder.CreateString("epsilon"),
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        0,
        0,
        false,
        hipdnn_data_sdk::data_objects::TensorValue::Float32Value,
        builder.CreateStruct(epsilonVal).Union()));

    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 7, "final_out", hipdnn_data_sdk::data_objects::DataType::FLOAT, &strides, &dims));

    std::vector<::flatbuffers::Offset<hipdnn_data_sdk::data_objects::Node>> nodes;

    // BN node referencing missing scale tensor
    auto bnormAttributes
        = hipdnn_data_sdk::data_objects::CreateBatchnormAttributes(builder,
                                                                   1,
                                                                   3, // Missing scale tensor
                                                                   4,
                                                                   5,
                                                                   0,
                                                                   flatbuffers::nullopt,
                                                                   flatbuffers::nullopt,
                                                                   flatbuffers::nullopt,
                                                                   2,
                                                                   flatbuffers::nullopt,
                                                                   flatbuffers::nullopt,
                                                                   flatbuffers::nullopt,
                                                                   flatbuffers::nullopt);
    nodes.push_back(hipdnn_data_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormAttributes,
        bnormAttributes.Union()));

    auto actAttr = hipdnn_data_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_data_sdk::data_objects::PointwiseMode::RELU_FWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        2,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        7);
    nodes.push_back(hipdnn_data_sdk::data_objects::CreateNodeDirect(
        builder,
        "act",
        hipdnn_data_sdk::data_objects::DataType::UNSET,
        hipdnn_data_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        actAttr.Union()));

    auto graphOffset = hipdnn_data_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::DataType::HALF,
        hipdnn_data_sdk::data_objects::DataType::BFLOAT16,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);

    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& bnNode = graph.getNode(0);
    auto* bnAttrs = bnNode.attributes_as_BatchnormAttributes();
    ASSERT_NE(bnAttrs, nullptr);

    const auto& actNode = graph.getNode(1);
    auto* activAttrs = actNode.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttrs, nullptr);

    EXPECT_THROW(BatchnormFwdTrainingParams(*bnAttrs, *activAttrs, graph.getTensorMap()),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestBatchnormFwdTrainingActivParams, ThrowsForMissingRequiredBiasTensor)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_data_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    std::vector<int64_t> strides = {1, 3, 14, 14};
    std::vector<int64_t> dims = {1, 3, 14, 14};
    std::vector<int64_t> derivedStrides = {1, 3, 1, 1};
    std::vector<int64_t> derivedDims = {1, 3, 1, 1};

    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", hipdnn_data_sdk::data_objects::DataType::FLOAT, &strides, &dims));
    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "y", hipdnn_data_sdk::data_objects::DataType::FLOAT, &strides, &dims, true));
    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        3,
        "scale",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));
    // Missing bias tensor (ID 4)

    hipdnn_data_sdk::data_objects::Float32Value epsilonVal(1e-5f);
    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributes(
        builder,
        5,
        builder.CreateString("epsilon"),
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        0,
        0,
        false,
        hipdnn_data_sdk::data_objects::TensorValue::Float32Value,
        builder.CreateStruct(epsilonVal).Union()));

    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 7, "final_out", hipdnn_data_sdk::data_objects::DataType::FLOAT, &strides, &dims));

    std::vector<::flatbuffers::Offset<hipdnn_data_sdk::data_objects::Node>> nodes;

    // BN node referencing missing bias tensor
    auto bnormAttributes
        = hipdnn_data_sdk::data_objects::CreateBatchnormAttributes(builder,
                                                                   1,
                                                                   3,
                                                                   4, // Missing bias tensor
                                                                   5,
                                                                   0,
                                                                   flatbuffers::nullopt,
                                                                   flatbuffers::nullopt,
                                                                   flatbuffers::nullopt,
                                                                   2,
                                                                   flatbuffers::nullopt,
                                                                   flatbuffers::nullopt,
                                                                   flatbuffers::nullopt,
                                                                   flatbuffers::nullopt);
    nodes.push_back(hipdnn_data_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormAttributes,
        bnormAttributes.Union()));

    auto actAttr = hipdnn_data_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_data_sdk::data_objects::PointwiseMode::RELU_FWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        2,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        7);
    nodes.push_back(hipdnn_data_sdk::data_objects::CreateNodeDirect(
        builder,
        "act",
        hipdnn_data_sdk::data_objects::DataType::UNSET,
        hipdnn_data_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        actAttr.Union()));

    auto graphOffset = hipdnn_data_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::DataType::HALF,
        hipdnn_data_sdk::data_objects::DataType::BFLOAT16,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);

    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& bnNode = graph.getNode(0);
    auto* bnAttrs = bnNode.attributes_as_BatchnormAttributes();
    ASSERT_NE(bnAttrs, nullptr);

    const auto& actNode = graph.getNode(1);
    auto* activAttrs = actNode.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttrs, nullptr);

    EXPECT_THROW(BatchnormFwdTrainingParams(*bnAttrs, *activAttrs, graph.getTensorMap()),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestBatchnormFwdTrainingActivParams, AcceptsReluFwdActivation)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingActivGraph(
        false, false, hipdnn_data_sdk::data_objects::PointwiseMode::RELU_FWD);
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& bnNode = graph.getNode(0);
    auto* bnAttrs = bnNode.attributes_as_BatchnormAttributes();
    ASSERT_NE(bnAttrs, nullptr);

    const auto& activNode = graph.getNode(1);
    auto* activAttrs = activNode.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttrs, nullptr);

    EXPECT_NO_THROW(BatchnormFwdTrainingParams(*bnAttrs, *activAttrs, graph.getTensorMap()));
}
