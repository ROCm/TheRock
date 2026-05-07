// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/plans/MiopenBatchnormFwdTrainingPlan.hpp"
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using namespace miopen_legacy_plugin;

TEST(TestBatchnormFwdTrainingParams, InitializesRequiredTensorsFromValidGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormAttributes();
    ASSERT_NE(attrs, nullptr);

    BatchnormFwdTrainingParams params(*attrs, graph.getTensorMap());

    // All required tensors should be initialized
    EXPECT_NO_THROW(params.x());
    EXPECT_NO_THROW(params.y());
    EXPECT_NO_THROW(params.scale());
    EXPECT_NO_THROW(params.bias());
}

TEST(TestBatchnormFwdTrainingParams, ExtractsEpsilonValueCorrectly)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormAttributes();
    ASSERT_NE(attrs, nullptr);

    BatchnormFwdTrainingParams params(*attrs, graph.getTensorMap());

    // Epsilon should be extracted as double
    EXPECT_NEAR(params.epsilonValue(), 1e-5, 1e-10);
}

TEST(TestBatchnormFwdTrainingParams, HandlesMeanVariancePresent)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph(
        {1, 3, 14, 14}, {1, 3, 14, 14}, true);
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormAttributes();
    ASSERT_NE(attrs, nullptr);

    BatchnormFwdTrainingParams params(*attrs, graph.getTensorMap());

    EXPECT_TRUE(params.hasSaveMeanVariance());
    EXPECT_NO_THROW(params.mean());
    EXPECT_NO_THROW(params.invVariance());
}

TEST(TestBatchnormFwdTrainingParams, HandlesMeanVarianceMissing)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph(
        {1, 3, 14, 14}, {1, 3, 14, 14}, false);
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormAttributes();
    ASSERT_NE(attrs, nullptr);

    BatchnormFwdTrainingParams params(*attrs, graph.getTensorMap());

    EXPECT_FALSE(params.hasSaveMeanVariance());
}

TEST(TestBatchnormFwdTrainingParams, ThrowsWhenRunningStatsProvided)
{
    // Create graph with running stats (should be rejected by defensive check)
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_data_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    std::vector<int64_t> strides = {1, 3, 14, 14};
    std::vector<int64_t> dims = {1, 3, 14, 14};
    std::vector<int64_t> derivedStrides = {1, 3, 1, 1};
    std::vector<int64_t> derivedDims = {1, 3, 1, 1};

    // Required tensors
    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", hipdnn_data_sdk::data_objects::DataType::FLOAT, &strides, &dims));
    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "y", hipdnn_data_sdk::data_objects::DataType::FLOAT, &strides, &dims));
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

    // Epsilon
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

    // Running stats tensor
    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        8,
        "prev_running_mean",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    // Momentum
    hipdnn_data_sdk::data_objects::Float32Value momentumVal(0.1f);
    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributes(
        builder,
        12,
        builder.CreateString("momentum"),
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        0,
        0,
        false,
        hipdnn_data_sdk::data_objects::TensorValue::Float32Value,
        builder.CreateStruct(momentumVal).Union()));

    auto bnormAttributes = hipdnn_data_sdk::data_objects::CreateBatchnormAttributes(
        builder,
        1, // x_tensor_uid
        3, // scale_tensor_uid
        4, // bias_tensor_uid
        5, // epsilon_tensor_uid
        0, // peer_stats_tensor_uid
        flatbuffers::Optional<int64_t>(8), // prev_running_mean_tensor_uid
        flatbuffers::nullopt, // prev_running_variance_tensor_uid
        flatbuffers::Optional<int64_t>(12), // momentum_tensor_uid
        2, // y_tensor_uid
        flatbuffers::nullopt, // mean_tensor_uid
        flatbuffers::nullopt, // inv_variance_tensor_uid
        flatbuffers::nullopt, // next_running_mean_tensor_uid
        flatbuffers::nullopt // next_running_variance_tensor_uid
    );

    std::vector<::flatbuffers::Offset<hipdnn_data_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_data_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm_training",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormAttributes,
        bnormAttributes.Union());
    nodes.push_back(node);

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

    const auto& graphNode = graph.getNode(0);
    auto* graphAttrs = graphNode.attributes_as_BatchnormAttributes();
    ASSERT_NE(graphAttrs, nullptr);

    // Should throw because running stats are provided
    EXPECT_THROW(BatchnormFwdTrainingParams(*graphAttrs, graph.getTensorMap()),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestBatchnormFwdTrainingParams, HasRunningStatsReturnsFalseWhenNotProvided)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormAttributes();
    ASSERT_NE(attrs, nullptr);

    BatchnormFwdTrainingParams params(*attrs, graph.getTensorMap());

    EXPECT_FALSE(params.hasRunningStats());
}
