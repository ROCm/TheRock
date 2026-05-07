// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>

#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "EngineManager.hpp"
#include "engines/MiopenEngine.hpp"

using namespace hipdnn_plugin_sdk;

namespace miopen_legacy_plugin
{

void EngineManager::addEngine(std::unique_ptr<IEngine> engine)
{
    _engines.emplace(engine->id(), std::move(engine));
}

std::vector<int64_t> EngineManager::getApplicableEngineIds(HipdnnEnginePluginHandle& handle,
                                                           const hipdnn_plugin_sdk::IGraph& opGraph)
{
    std::vector<int64_t> applicable;
    for(const auto& engine : _engines)
    {
        if(engine.second->isApplicable(handle, opGraph))
        {
            applicable.push_back(engine.second->id());
        }
    }
    return applicable;
}

void EngineManager::getEngineDetails(HipdnnEnginePluginHandle& handle,
                                     const hipdnn_plugin_sdk::IGraph& opGraph,
                                     int64_t engineId,
                                     hipdnnPluginConstData_t& engineDetailsOut)
{
    (void)opGraph; // Unused parameter
    auto& engine = getEngine(engineId);
    engine.getDetails(handle, engineDetailsOut);
}

size_t EngineManager::getWorkspaceSize(const HipdnnEnginePluginHandle& handle,
                                       int64_t engineId,
                                       const hipdnn_plugin_sdk::IGraph& opGraph) const
{
    auto& engine = getEngine(engineId);
    return engine.getWorkspaceSize(handle, opGraph);
}

void EngineManager::initializeExecutionContext(
    const HipdnnEnginePluginHandle& handle,
    const hipdnn_plugin_sdk::IGraph& opGraph,
    const hipdnn_plugin_sdk::IEngineConfig& engineConfig,
    HipdnnEnginePluginExecutionContext& executionContext) const
{
    auto& engine = getEngine(engineConfig.engineId());
    engine.initializeExecutionContext(handle, opGraph, executionContext);
}

IEngine& EngineManager::getEngine(int64_t engineId) const
{
    auto it = _engines.find(engineId);
    if(it == _engines.end())
    {
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                    "Engine with ID " + std::to_string(engineId) + " not found.");
    }
    return *it->second;
}

}
