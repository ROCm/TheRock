// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <stdint.h>

#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

#include "HipdnnEnginePluginExecutionContext.hpp"
#include "HipdnnEnginePluginHandle.hpp"

namespace miopen_legacy_plugin
{

class IPlanBuilder
{
public:
    virtual ~IPlanBuilder() = default;

    virtual bool isApplicable(const HipdnnEnginePluginHandle& handle,
                              const hipdnn_plugin_sdk::IGraph& opGraph) const
        = 0;

    virtual size_t getWorkspaceSize(const HipdnnEnginePluginHandle& handle,
                                    const hipdnn_plugin_sdk::IGraph& opGraph) const
        = 0;

    virtual void buildPlan(const HipdnnEnginePluginHandle& handle,
                           const hipdnn_plugin_sdk::IGraph& opGraph,
                           HipdnnEnginePluginExecutionContext& executionContext) const
        = 0;
};
}
