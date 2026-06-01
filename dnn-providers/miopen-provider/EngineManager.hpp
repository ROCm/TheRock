// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

#include "engines/EngineInterface.hpp"

namespace miopen_legacy_plugin
{

class EngineManager
{
public:
    EngineManager() = default;
    ~EngineManager() = default;

    //disallow copy and assignment
    EngineManager(const EngineManager&) = delete;
    EngineManager& operator=(const EngineManager&) = delete;

    void addEngine(std::unique_ptr<IEngine> engine);

    std::vector<int64_t> getApplicableEngineIds(HipdnnEnginePluginHandle& handle,
                                                const hipdnn_plugin_sdk::IGraph& opGraph);

    void getEngineDetails(HipdnnEnginePluginHandle& handle,
                          const hipdnn_plugin_sdk::IGraph& opGraph,
                          int64_t engineId,
                          hipdnnPluginConstData_t& engineDetailsOut);

    size_t getWorkspaceSize(const HipdnnEnginePluginHandle& handle,
                            int64_t engineId,
                            const hipdnn_plugin_sdk::IGraph& opGraph) const;

    void initializeExecutionContext(const HipdnnEnginePluginHandle& handle,
                                    const hipdnn_plugin_sdk::IGraph& opGraph,
                                    const hipdnn_plugin_sdk::IEngineConfig& engineConfig,
                                    HipdnnEnginePluginExecutionContext& executionContext) const;

private:
    IEngine& getEngine(int64_t engineId) const;

    std::unordered_map<int64_t, std::unique_ptr<IEngine>> _engines;
};

}
