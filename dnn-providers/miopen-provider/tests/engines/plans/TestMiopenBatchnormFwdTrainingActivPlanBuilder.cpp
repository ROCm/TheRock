// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_plugin_sdk/EnginePluginApi.h>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>

#include "HipdnnEnginePluginHandle.hpp"
#include "engines/plans/MiopenBatchnormFwdTrainingPlanBuilder.hpp"

#include "mocks/MockHipdnnEnginePluginExecutionContext.hpp"

using namespace miopen_legacy_plugin;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_plugin_sdk;

class TestMiopenBatchnormFwdTrainingActivPlanBuilder : public ::testing::Test
{
protected:
    MiopenBatchnormFwdTrainingPlanBuilder _planBuilder;
    HipdnnEnginePluginHandle _dummyHandle;
};

TEST_F(TestMiopenBatchnormFwdTrainingActivPlanBuilder,
       IsApplicableReturnsTrueForValidSingleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    bool applicable = _planBuilder.isApplicable(_dummyHandle, graph);

    EXPECT_TRUE(applicable);
}

TEST_F(TestMiopenBatchnormFwdTrainingActivPlanBuilder, IsApplicableReturnsFalseForThreeNodeGraph)
{
    MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(3));

    bool applicable = _planBuilder.isApplicable(_dummyHandle, mockGraph);

    EXPECT_FALSE(applicable);
}

TEST_F(TestMiopenBatchnormFwdTrainingActivPlanBuilder, IsApplicableReturnsTrueForValidTwoNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingActivGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    bool applicable = _planBuilder.isApplicable(_dummyHandle, graph);

    EXPECT_TRUE(applicable);
}

TEST_F(TestMiopenBatchnormFwdTrainingActivPlanBuilder,
       IsApplicableReturnsFalseForGraphWithRunningStatistics)
{
    auto builder
        = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingActivGraph(true, true);
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    bool applicable = _planBuilder.isApplicable(_dummyHandle, graph);

    EXPECT_FALSE(applicable);
}

TEST_F(TestMiopenBatchnormFwdTrainingActivPlanBuilder,
       IsApplicableReturnsTrueForFusionWithoutRunningStatistics)
{
    // Create a fused batchnorm training + activation graph WITHOUT running statistics
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingActivGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    bool applicable = _planBuilder.isApplicable(_dummyHandle, graph);

    EXPECT_TRUE(applicable);
}

TEST_F(TestMiopenBatchnormFwdTrainingActivPlanBuilder, IsApplicableReturnsFalseForNonReluActivation)
{
    // Graph with SIGMOID_FWD instead of RELU_FWD
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingActivGraph(
        false, false, hipdnn_data_sdk::data_objects::PointwiseMode::SIGMOID_FWD);
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    bool applicable = _planBuilder.isApplicable(_dummyHandle, graph);

    EXPECT_FALSE(applicable);
}

TEST_F(TestMiopenBatchnormFwdTrainingActivPlanBuilder, GetWorkspaceSizeReturnsZero)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingActivGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    size_t workspaceSize = _planBuilder.getWorkspaceSize(_dummyHandle, graph);

    EXPECT_EQ(workspaceSize, 0u);
}

TEST_F(TestMiopenBatchnormFwdTrainingActivPlanBuilder, BuildPlanSetsPlanForValidGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingActivGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    HipdnnEnginePluginExecutionContext ctx;

    EXPECT_NO_THROW(_planBuilder.buildPlan(_dummyHandle, graph, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}

TEST_F(TestMiopenBatchnormFwdTrainingActivPlanBuilder,
       BuildPlanThrowsForMalformedBatchnormAttributes)
{
    // Create graph with batchnorm training node but null attributes
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_data_sdk::data_objects::TensorAttributes>>
        tensorAttributes;
    std::vector<::flatbuffers::Offset<hipdnn_data_sdk::data_objects::Node>> nodes;

    // Node with BatchnormAttributes type but null attributes pointer
    auto node = hipdnn_data_sdk::data_objects::CreateNodeDirect(
        builder,
        "malformed_bn_training",
        hipdnn_data_sdk::data_objects::DataType::UNSET,
        hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormAttributes,
        0);
    nodes.push_back(node);

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
    HipdnnEnginePluginExecutionContext ctx;

    EXPECT_THROW(_planBuilder.buildPlan(_dummyHandle, graph, ctx), std::invalid_argument);
    EXPECT_FALSE(ctx.hasValidPlan());
}

TEST_F(TestMiopenBatchnormFwdTrainingActivPlanBuilder,
       BuildPlanThrowsForMalformedActivationAttributes)
{
    // Create graph with valid batchnorm but malformed activation
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
        builder,
        2,
        "y_virtual",
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

    std::vector<::flatbuffers::Offset<hipdnn_data_sdk::data_objects::Node>> nodes;

    // Valid batchnorm training node
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
                                                                   2,
                                                                   flatbuffers::nullopt,
                                                                   flatbuffers::nullopt,
                                                                   flatbuffers::nullopt,
                                                                   flatbuffers::nullopt);
    nodes.push_back(hipdnn_data_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_training",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormAttributes,
        bnormAttributes.Union()));

    // Malformed activation (null attributes)
    nodes.push_back(hipdnn_data_sdk::data_objects::CreateNodeDirect(
        builder,
        "malformed_act",
        hipdnn_data_sdk::data_objects::DataType::UNSET,
        hipdnn_data_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        0));

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
    HipdnnEnginePluginExecutionContext ctx;

    EXPECT_THROW(_planBuilder.buildPlan(_dummyHandle, graph, ctx), std::invalid_argument);
    EXPECT_FALSE(ctx.hasValidPlan());
}

TEST_F(TestMiopenBatchnormFwdTrainingActivPlanBuilder, IsApplicableReturnsFalseForWrongNodeOrder)
{
    // Create graph with nodes in wrong order (activation → batchnorm instead of batchnorm → activation)
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_data_sdk::data_objects::TensorAttributes>>
        tensorAttributes;
    std::vector<::flatbuffers::Offset<hipdnn_data_sdk::data_objects::Node>> nodes;

    // Wrong order: activation first
    auto actAttr = hipdnn_data_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_data_sdk::data_objects::PointwiseMode::RELU_FWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        1,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        2);
    nodes.push_back(hipdnn_data_sdk::data_objects::CreateNodeDirect(
        builder,
        "act",
        hipdnn_data_sdk::data_objects::DataType::UNSET,
        hipdnn_data_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        actAttr.Union()));

    // Then batchnorm (wrong order!)
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

    bool applicable = _planBuilder.isApplicable(_dummyHandle, graph);

    EXPECT_FALSE(applicable);
}

TEST_F(TestMiopenBatchnormFwdTrainingActivPlanBuilder,
       IsApplicableReturnsFalseWhenBnOutputIsNotVirtualInFusion)
{
    // Create fusion graph where BN output tensor is non-virtual (should be virtual)
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_data_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    std::vector<int64_t> strides = {1, 3, 14, 14};
    std::vector<int64_t> dims = {1, 3, 14, 14};
    std::vector<int64_t> derivedStrides = {1, 3, 1, 1};
    std::vector<int64_t> derivedDims = {1, 3, 1, 1};

    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", hipdnn_data_sdk::data_objects::DataType::FLOAT, &strides, &dims));
    // BN output NOT virtual (should be virtual for fusion)
    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        2,
        "y_bn",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        &strides,
        &dims,
        false));
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

    bool applicable = _planBuilder.isApplicable(_dummyHandle, graph);

    EXPECT_FALSE(applicable);
}
