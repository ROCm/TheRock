/*
// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
*/

#pragma once

#include <gmock/gmock.h>

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

#include "engines/EngineInterface.hpp"

namespace miopen_legacy_plugin
{

class MockEngine : public IEngine
{
public:
    MOCK_METHOD(int64_t, id, (), (const, override));
    MOCK_METHOD(bool,
                isApplicable,
                (HipdnnEnginePluginHandle & handle, const hipdnn_plugin_sdk::IGraph& opGraph),
                (const, override));
    MOCK_METHOD(void,
                getDetails,
                (HipdnnEnginePluginHandle & handle, hipdnnPluginConstData_t& detailsOut),
                (const, override));
    MOCK_METHOD(size_t,
                getWorkspaceSize,
                (const HipdnnEnginePluginHandle& handle, const hipdnn_plugin_sdk::IGraph& opGraph),
                (const, override));

    MOCK_METHOD(void,
                initializeExecutionContext,
                (const HipdnnEnginePluginHandle& handle,
                 const hipdnn_plugin_sdk::IGraph& opGraph,
                 HipdnnEnginePluginExecutionContext& executionContext),
                (const, override));
};

} // namespace miopen_legacy_plugin
