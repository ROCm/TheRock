// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gmock/gmock.h>

#include <hipdnn_data_sdk/data_objects/graph_generated.h>

#include "engines/plans/PlanBuilderInterface.hpp"

namespace miopen_legacy_plugin
{

class MockPlanBuilder : public IPlanBuilder
{
public:
    MOCK_METHOD(bool,
                isApplicable,
                (const HipdnnEnginePluginHandle& handle, const hipdnn_plugin_sdk::IGraph& opGraph),
                (const, override));
    MOCK_METHOD(size_t,
                getWorkspaceSize,
                (const HipdnnEnginePluginHandle& handle, const hipdnn_plugin_sdk::IGraph& opGraph),
                (const, override));

    MOCK_METHOD(void,
                buildPlan,
                (const HipdnnEnginePluginHandle& handle,
                 const hipdnn_plugin_sdk::IGraph& opGraph,
                 HipdnnEnginePluginExecutionContext& executionContext),
                (const, override));
};

}
