// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/plans/MiopenBatchnormFwdInferencePlan.hpp"
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using namespace miopen_legacy_plugin;

TEST(TestMiopenBatchnormFwdInferenceParams, InitializesAllTensorsFromValidGraph)
{
    // Create a valid batchnorm graph
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    // Get the batchnorm node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormInferenceAttributes();
    ASSERT_NE(attrs, nullptr);

    // Expect that params construction doesn't throw
    EXPECT_NO_THROW(BatchnormFwdInferenceParams(*attrs, graph.getTensorMap()));

    BatchnormFwdInferenceParams params(*attrs, graph.getTensorMap());
    // verify activation optional params are null when no activation is specified
    EXPECT_EQ(params.optActivation(), std::nullopt);
    EXPECT_EQ(params.activationOut(), std::nullopt);
}

TEST(TestMiopenBatchnormFwdInferenceParams, InitializesFusedActivationBiasWithAllTensors)
{
    // Create a fused batchnorm fwd + activation graph
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdInferActGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    // Get the two required nodes
    const auto& batchnormInfNode = graph.getNode(0);
    const auto& pointwiseNode = graph.getNode(1);

    auto* batchnormInfAttrs = batchnormInfNode.attributes_as_BatchnormInferenceAttributes();
    auto* pointwiseAttrs = pointwiseNode.attributes_as_PointwiseAttributes();

    ASSERT_NE(batchnormInfAttrs, nullptr);
    ASSERT_NE(pointwiseAttrs, nullptr);

    // Construct fused params
    BatchnormFwdInferenceParams params(*batchnormInfAttrs, *pointwiseAttrs, graph.getTensorMap());

    // All required tensors should be initialized
    EXPECT_NO_THROW(params.x());
    EXPECT_NO_THROW(params.y());
    EXPECT_NO_THROW(params.scale());
    EXPECT_NO_THROW(params.bias());
    EXPECT_NO_THROW(params.estMean());
    EXPECT_NO_THROW(params.invVariance());

    // Optional fusion-specific tensors should be present
    EXPECT_TRUE(params.optActivation().has_value());
    EXPECT_TRUE(params.activationOut().has_value());
}
