// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <gtest/gtest.h>
#include <unordered_map>
#include <vector>

#include <MiopenLegacyPlugin.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>

#include <hipdnn_data_sdk/utilities/LoadGraphAndTensors.hpp>

#include "HipdnnEnginePluginExecutionContext.hpp"
#include "HipdnnEnginePluginHandle.hpp"

namespace test_helpers
{
class TestGoldenReferenceGpu : public testing::TestWithParam<std::filesystem::path>
{
protected:
    hipdnn_data_sdk::utilities::GraphAndTensorMap _graphAndTensors;
    hipdnnEnginePluginHandle_t _handle;
    flatbuffers::DetachedBuffer _engineConfigBuffer;
    std::unordered_map<int64_t, std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>>
        _referenceOutputTensors;

    // NOLINTNEXTLINE(readability-identifier-naming)
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        const auto& path = GetParam();

        // TODO: Temporary fix until reference data can be properly installed
        if(path.empty())
        {
            HIPDNN_LOG_WARN("Reference not found for Gpu golden reference test");
            GTEST_SKIP();
        }

        hipdnnPluginStatus_t status = hipdnnEnginePluginCreateImpl(&_handle);
        ASSERT_EQ(status, hipdnnPluginStatus_t::HIPDNN_PLUGIN_STATUS_SUCCESS);

        _engineConfigBuffer = hipdnn_test_sdk::utilities::createValidEngineConfig(1).Release();

        _graphAndTensors = hipdnn_test_sdk::utilities::loadGraphAndTensors(path);
        _referenceOutputTensors = _graphAndTensors.extractAndClearOutputTensorData();
    }

    void goldenReferenceTestSuite(float absoluteTolerance, float relativeTolerance)
    {
        SKIP_IF_WINDOWS();

        hipdnnPluginConstData_t opGraph
            = {_graphAndTensors.graphBuffer.data(), _graphAndTensors.graphBuffer.size()};

        hipdnnPluginConstData_t engineConfig
            = {_engineConfigBuffer.data(), _engineConfigBuffer.size()};

        hipdnnPluginStatus_t status;
        hipdnnEnginePluginExecutionContext_t executionContext;
        status = hipdnnEnginePluginCreateExecutionContextImpl(
            _handle, &engineConfig, &opGraph, &executionContext);
        ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
        auto deviceBuffers = _graphAndTensors.deviceBuffers();

        status = hipdnnEnginePluginExecuteOpGraphImpl(_handle,
                                                      executionContext,
                                                      nullptr,
                                                      deviceBuffers.data(),
                                                      static_cast<uint32_t>(deviceBuffers.size()));
        EXPECT_EQ(status, hipdnnPluginStatus_t::HIPDNN_PLUGIN_STATUS_SUCCESS);
        for(auto uid : _graphAndTensors.outputTensorUids)
        {
            _graphAndTensors.tensorMap.at(uid)->markDeviceModified();
        }

        EXPECT_TRUE(_graphAndTensors.validateTensors(
            _referenceOutputTensors, absoluteTolerance, relativeTolerance));
    }
};

auto getGoldenReferenceParams(const std::filesystem::path& subDirectory)
{
    return testing::ValuesIn(
        hipdnn_test_sdk::utilities::filesInDirectoryWithExtReturnEmptyPathOnThrow(
            hipdnn_data_sdk::utilities::getCurrentExecutableDirectory()
                / "../lib/hipdnn_reference_data" / subDirectory,
            ".json"));
}

}
