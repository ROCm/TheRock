// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <miopen/miopen.h>

#include "HipdnnEnginePluginHandle.hpp"
#include "engines/plans/MiopenConvFwdPlan.hpp"

using namespace miopen_legacy_plugin;

class TestGpuConvFwdPlan : public ::testing::Test
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

TEST(TestConvFwdParams, InitializesAllTensorsFromValidGraph)
{
    // Create a valid convolution graph
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    // Get the convolution node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(attrs, nullptr);

    // Construct params
    ConvFwdParams params(*attrs, graph.getTensorMap());

    // All required tensors should be initialized
    EXPECT_NO_THROW(params.x());
    EXPECT_NO_THROW(params.w());
    EXPECT_NO_THROW(params.y());
    EXPECT_NO_THROW(params.conv());
}

TEST(TestConvFwdParams, ThrowsOnAssymetricPadding)
{
    std::vector<int64_t> xDims = {1, 1, 1, 1};
    std::vector<int64_t> xStrides = {1, 1, 1, 1};
    std::vector<int64_t> wDims = {1, 1, 1, 1};
    std::vector<int64_t> wStrides = {1, 1, 1, 1};
    std::vector<int64_t> yDims = {1, 1, 1, 1};
    std::vector<int64_t> yStrides = {1, 1, 1, 1};
    std::vector<int64_t> convPrePadding = {0, 0}; // Asymmetic padding
    std::vector<int64_t> convPostPadding = {1, 1};
    std::vector<int64_t> convStrides = {1, 1};
    std::vector<int64_t> convDilation = {1, 1};
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph(xDims,
                                                                       xStrides,
                                                                       wDims,
                                                                       wStrides,
                                                                       yDims,
                                                                       yStrides,
                                                                       convPrePadding,
                                                                       convPostPadding,
                                                                       convStrides,
                                                                       convDilation);
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    // Get the convolution node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(attrs, nullptr);

    // Construct params and expect exception
    EXPECT_THROW(ConvFwdParams(*attrs, graph.getTensorMap()),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestConvFwdParams, ThrowsOnInvalidPostPaddingVectorSize)
{
    std::vector<int64_t> xDims = {1, 1, 1, 1};
    std::vector<int64_t> xStrides = {1, 1, 1, 1};
    std::vector<int64_t> wDims = {1, 1, 1, 1};
    std::vector<int64_t> wStrides = {1, 1, 1, 1};
    std::vector<int64_t> yDims = {1, 1, 1, 1};
    std::vector<int64_t> yStrides = {1, 1, 1, 1};
    std::vector<int64_t> convPrePadding = {0, 0};
    std::vector<int64_t> convPostPadding = {0, 0, 0}; // Invalid post padding vector size
    std::vector<int64_t> convStrides = {1, 1};
    std::vector<int64_t> convDilation = {1, 1};
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph(xDims,
                                                                       xStrides,
                                                                       wDims,
                                                                       wStrides,
                                                                       yDims,
                                                                       yStrides,
                                                                       convPrePadding,
                                                                       convPostPadding,
                                                                       convStrides,
                                                                       convDilation);
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    // Get the convolution node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(attrs, nullptr);

    // Construct params and expect exception
    EXPECT_THROW(ConvFwdParams(*attrs, graph.getTensorMap()),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestConvFwdParams, ThrowsOnInvalidPaddingVectorsSize)
{
    // Create a convolution graph with invalid conv dims
    std::vector<int64_t> xDims = {1, 1, 1, 1};
    std::vector<int64_t> xStrides = {1, 1, 1, 1};
    std::vector<int64_t> wDims = {1, 1, 1, 1};
    std::vector<int64_t> wStrides = {1, 1, 1, 1};
    std::vector<int64_t> yDims = {1, 1, 1, 1};
    std::vector<int64_t> yStrides = {1, 1, 1, 1};
    std::vector<int64_t> convPrePadding = {0, 0, 0}; // Invalid pre padding vector size
    std::vector<int64_t> convPostPadding = {0, 0, 0}; // Invalid post padding vector size
    std::vector<int64_t> convStrides = {1, 1};
    std::vector<int64_t> convDilation = {1, 1};
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph(xDims,
                                                                       xStrides,
                                                                       wDims,
                                                                       wStrides,
                                                                       yDims,
                                                                       yStrides,
                                                                       convPrePadding,
                                                                       convPostPadding,
                                                                       convStrides,
                                                                       convDilation);
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    // Get the convolution node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(attrs, nullptr);

    // Construct params and expect exception
    EXPECT_THROW(ConvFwdParams(*attrs, graph.getTensorMap()),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestConvFwdParams, ThrowsOnInvalidStrideVectorSize)
{
    std::vector<int64_t> xDims = {1, 1, 1, 1};
    std::vector<int64_t> xStrides = {1, 1, 1, 1};
    std::vector<int64_t> wDims = {1, 1, 1, 1};
    std::vector<int64_t> wStrides = {1, 1, 1, 1};
    std::vector<int64_t> yDims = {1, 1, 1, 1};
    std::vector<int64_t> yStrides = {1, 1, 1, 1};
    std::vector<int64_t> convPrePadding = {0, 0};
    std::vector<int64_t> convPostPadding = {0, 0};
    std::vector<int64_t> convStrides = {1}; // Invalid strides vector size
    std::vector<int64_t> convDilation = {1, 1};
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph(xDims,
                                                                       xStrides,
                                                                       wDims,
                                                                       wStrides,
                                                                       yDims,
                                                                       yStrides,
                                                                       convPrePadding,
                                                                       convPostPadding,
                                                                       convStrides,
                                                                       convDilation);
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    // Get the convolution node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(attrs, nullptr);

    // Construct params and expect exception
    EXPECT_THROW(ConvFwdParams(*attrs, graph.getTensorMap()),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestConvFwdParams, ThrowsOnInvalidDilationVectorSize)
{
    std::vector<int64_t> xDims = {1, 1, 1, 1};
    std::vector<int64_t> xStrides = {1, 1, 1, 1};
    std::vector<int64_t> wDims = {1, 1, 1, 1};
    std::vector<int64_t> wStrides = {1, 1, 1, 1};
    std::vector<int64_t> yDims = {1, 1, 1, 1};
    std::vector<int64_t> yStrides = {1, 1, 1, 1};
    std::vector<int64_t> convPrePadding = {0, 0};
    std::vector<int64_t> convPostPadding = {0, 0};
    std::vector<int64_t> convStrides = {1, 1};
    std::vector<int64_t> convDilation = {1}; // Invalid dilation vector size
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph(xDims,
                                                                       xStrides,
                                                                       wDims,
                                                                       wStrides,
                                                                       yDims,
                                                                       yStrides,
                                                                       convPrePadding,
                                                                       convPostPadding,
                                                                       convStrides,
                                                                       convDilation);
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    // Get the convolution node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(attrs, nullptr);

    // Construct params and expect exception
    EXPECT_THROW(ConvFwdParams(*attrs, graph.getTensorMap()),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST_F(TestGpuConvFwdPlan, CreatesPlanWithValidGraph)
{
    // Create a valid convolution graph
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    // Get the convolution node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(attrs, nullptr);

    // Construct params
    ConvFwdParams params(*attrs, graph.getTensorMap());

    // Create plan
    ConvFwdPlan(_handle, std::move(params));
}

TEST_F(TestGpuConvFwdPlan, ThrowsOnInvalidDims)
{
    // Create a convolution graph with invalid conv dims
    std::vector<int64_t> xDims = {1, 1, 1, 1};
    std::vector<int64_t> xStrides = {1, 1, 1, 1};
    std::vector<int64_t> wDims = {1, 1, 1}; // Invalid w tensor dims
    std::vector<int64_t> wStrides = {1, 1, 1};
    std::vector<int64_t> yDims = {1, 1, 1, 1};
    std::vector<int64_t> yStrides = {1, 1, 1, 1};
    std::vector<int64_t> convPrePadding = {0, 0};
    std::vector<int64_t> convPostPadding = {0, 0};
    std::vector<int64_t> convStrides = {1, 1};
    std::vector<int64_t> convDilation = {1, 1};
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph(xDims,
                                                                       xStrides,
                                                                       wDims,
                                                                       wStrides,
                                                                       yDims,
                                                                       yStrides,
                                                                       convPrePadding,
                                                                       convPostPadding,
                                                                       convStrides,
                                                                       convDilation);
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    // Get the convolution node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(attrs, nullptr);

    // Construct params
    ConvFwdParams params(*attrs, graph.getTensorMap());

    // Create plan and expect exception
    EXPECT_THROW(ConvFwdPlan(_handle, std::move(params)), hipdnn_plugin_sdk::HipdnnPluginException);
}
