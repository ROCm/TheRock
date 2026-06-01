// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>

#include <hipdnn_data_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "engines/plans/PlanInterface.hpp"

struct HipdnnEnginePluginExecutionContext
{
public:
    virtual ~HipdnnEnginePluginExecutionContext() = default;

    bool hasValidPlan() const
    {
        return _plan != nullptr;
    }

    void setPlan(std::unique_ptr<miopen_legacy_plugin::IPlan> plan)
    {
        _plan = std::move(plan);
    }

    virtual miopen_legacy_plugin::IPlan& plan() const
    {
        if(!hasValidPlan())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "Cannot get plan in execution context, its not set");
        }
        return *_plan;
    }

private:
    std::unique_ptr<miopen_legacy_plugin::IPlan> _plan;
};
