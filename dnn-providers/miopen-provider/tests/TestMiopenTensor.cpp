// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "MiopenTensor.hpp"
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using namespace miopen_legacy_plugin;

TEST(TestMiopenTensor, CanCreateAndDestroy)
{
    // Use a real tensor attributes from a valid batchnorm graph
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    // Get the first tensor attributes from the tensor map
    const auto& tensorMap = graph.getTensorMap();
    ASSERT_FALSE(tensorMap.empty());
    const auto* tensorAttr = tensorMap.begin()->second;
    ASSERT_NE(tensorAttr, nullptr);

    // Construct and destroy MiopenTensor
    EXPECT_NO_THROW({
        MiopenTensor tensor(*tensorAttr);
        EXPECT_EQ(tensor.uid(), tensorAttr->uid());
        EXPECT_NE(tensor.tensorDescriptor(), nullptr);
    });
}

TEST(TestMiopenTensor, TensorDescriptorIsValid)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& tensorMap = graph.getTensorMap();
    ASSERT_FALSE(tensorMap.empty());
    const auto* tensorAttr = tensorMap.begin()->second;
    MiopenTensor tensor(*tensorAttr);

    // The descriptor should be non-null and can be used in MIOpen API calls
    EXPECT_NE(tensor.tensorDescriptor(), nullptr);
}
