// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <miopen/miopen.h>

#include "HipdnnEnginePluginHandle.hpp"
#include "engines/plans/MiopenConvFwdBiasActivPlan.hpp"

using namespace miopen_legacy_plugin;
using namespace hipdnn_data_sdk::data_objects;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;

class TestGpuConvFwdBiasActivPlan : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        ASSERT_EQ(miopenCreate(&_handle.miopenHandle), miopenStatusSuccess);
    }

    void TearDown() override
    {
        if(_handle.miopenHandle != nullptr)
        {
            EXPECT_EQ(miopenDestroy(_handle.miopenHandle), miopenStatusSuccess);
        }
    }

    HipdnnEnginePluginHandle _handle;
};

TEST(TestConvFwdBiasActivParams, InitializeFromValidConvFwdActivGraph)
{
    auto builder = createValidConvFwdActivGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node0 = graph.getNode(0);
    const auto* convAttr = node0.attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(convAttr, nullptr);

    const auto& node1 = graph.getNode(1);
    const auto* activAttr = node1.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttr, nullptr);

    ConvFwdBiasActivParams params(*convAttr, nullptr, *activAttr, graph.getTensorMap());
}

TEST(TestConvFwdBiasActivParams, InitializeFromValidConvFwdBiasActivGraph)
{
    auto builder = createValidConvFwdBiasActivGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node0 = graph.getNode(0);
    const auto* convAttr = node0.attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(convAttr, nullptr);

    const auto& node1 = graph.getNode(1);
    const auto* biasAttr = node1.attributes_as_PointwiseAttributes();
    ASSERT_NE(biasAttr, nullptr);

    const auto& node2 = graph.getNode(2);
    const auto* activAttr = node2.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttr, nullptr);

    ConvFwdBiasActivParams params(*convAttr, biasAttr, *activAttr, graph.getTensorMap());
}

TEST(TestConvFwdBiasActivParams, ThrowOnWrongActivationMode)
{
    const std::vector<int64_t> xDims = {1, 1, 1, 1};
    const std::vector<int64_t> wDims = {1, 1, 1, 1};
    const std::vector<int64_t> yDims = {1, 1, 1, 1};
    const std::vector<int64_t> convPadding = {0, 0};
    const std::vector<int64_t> convStrides = {1, 1};
    const std::vector<int64_t> convDilation = {1, 1};

    auto builder = createValidConvFwdBiasActivGraph(xDims,
                                                    generateStrides(xDims),
                                                    wDims,
                                                    generateStrides(wDims),
                                                    yDims,
                                                    generateStrides(yDims),
                                                    convPadding,
                                                    convPadding,
                                                    convStrides,
                                                    convDilation,
                                                    PointwiseMode::ADD); // Invalid activation mode
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node0 = graph.getNode(0);
    const auto* convAttr = node0.attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(convAttr, nullptr);

    const auto& node1 = graph.getNode(1);
    const auto* biasAttr = node1.attributes_as_PointwiseAttributes();
    ASSERT_NE(biasAttr, nullptr);

    const auto& node2 = graph.getNode(2);
    const auto* activAttr = node2.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttr, nullptr);

    EXPECT_THROW(ConvFwdBiasActivParams(*convAttr, biasAttr, *activAttr, graph.getTensorMap()),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST_F(TestGpuConvFwdBiasActivPlan, CreatePlanWithValidConvFwdActivGraph)
{
    auto builder = createValidConvFwdActivGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node0 = graph.getNode(0);
    const auto* convAttr = node0.attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(convAttr, nullptr);

    const auto& node1 = graph.getNode(1);
    const auto* activAttr = node1.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttr, nullptr);

    ConvFwdBiasActivParams params(*convAttr, nullptr, *activAttr, graph.getTensorMap());

    ConvFwdBiasActivPlan(_handle, std::move(params));
}

TEST_F(TestGpuConvFwdBiasActivPlan, CreatePlanWithValidConvFwdActivGraphCompileOnly)
{
    auto builder = createValidConvFwdActivGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node0 = graph.getNode(0);
    const auto* convAttr = node0.attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(convAttr, nullptr);

    const auto& node1 = graph.getNode(1);
    const auto* activAttr = node1.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttr, nullptr);

    ConvFwdBiasActivParams params(*convAttr, nullptr, *activAttr, graph.getTensorMap());

    ConvFwdBiasActivPlan(_handle, std::move(params), true, false);
}

TEST_F(TestGpuConvFwdBiasActivPlan, CreatePlanWithValidConvFwdActivGraphWorkspaceOnly)
{
    auto builder = createValidConvFwdActivGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node0 = graph.getNode(0);
    const auto* convAttr = node0.attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(convAttr, nullptr);

    const auto& node1 = graph.getNode(1);
    const auto* activAttr = node1.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttr, nullptr);

    ConvFwdBiasActivParams params(*convAttr, nullptr, *activAttr, graph.getTensorMap());

    ConvFwdBiasActivPlan(_handle, std::move(params), false, true);
}

TEST_F(TestGpuConvFwdBiasActivPlan, CreatePlanWithValidConvFwdBiasActivGraph)
{
    auto builder = createValidConvFwdBiasActivGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node0 = graph.getNode(0);
    const auto* convAttr = node0.attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(convAttr, nullptr);

    const auto& node1 = graph.getNode(1);
    const auto* biasAttr = node1.attributes_as_PointwiseAttributes();
    ASSERT_NE(biasAttr, nullptr);

    const auto& node2 = graph.getNode(2);
    const auto* activAttr = node2.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttr, nullptr);

    ConvFwdBiasActivParams params(*convAttr, biasAttr, *activAttr, graph.getTensorMap());

    ConvFwdBiasActivPlan(_handle, std::move(params));
}

TEST_F(TestGpuConvFwdBiasActivPlan, CreatePlanWithValidConvFwdBiasActivGraphCompileOnly)
{
    auto builder = createValidConvFwdBiasActivGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node0 = graph.getNode(0);
    const auto* convAttr = node0.attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(convAttr, nullptr);

    const auto& node1 = graph.getNode(1);
    const auto* biasAttr = node1.attributes_as_PointwiseAttributes();
    ASSERT_NE(biasAttr, nullptr);

    const auto& node2 = graph.getNode(2);
    const auto* activAttr = node2.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttr, nullptr);

    ConvFwdBiasActivParams params(*convAttr, biasAttr, *activAttr, graph.getTensorMap());

    ConvFwdBiasActivPlan(_handle, std::move(params), true, false);
}

TEST_F(TestGpuConvFwdBiasActivPlan, CreatePlanWithValidConvFwdBiasActivGraphWorkspaceOnly)
{
    auto builder = createValidConvFwdBiasActivGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node0 = graph.getNode(0);
    const auto* convAttr = node0.attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(convAttr, nullptr);

    const auto& node1 = graph.getNode(1);
    const auto* biasAttr = node1.attributes_as_PointwiseAttributes();
    ASSERT_NE(biasAttr, nullptr);

    const auto& node2 = graph.getNode(2);
    const auto* activAttr = node2.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttr, nullptr);

    ConvFwdBiasActivParams params(*convAttr, biasAttr, *activAttr, graph.getTensorMap());

    ConvFwdBiasActivPlan(_handle, std::move(params), false, true);
}
