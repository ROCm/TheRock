// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "PlanBuilderInterface.hpp"
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

namespace miopen_legacy_plugin
{

class MiopenBatchnormFwdTrainingPlanBuilder : public IPlanBuilder
{
public:
    MiopenBatchnormFwdTrainingPlanBuilder() = default;
    ~MiopenBatchnormFwdTrainingPlanBuilder() override = default;

    // Disallow copy and assignment
    MiopenBatchnormFwdTrainingPlanBuilder(const MiopenBatchnormFwdTrainingPlanBuilder&) = delete;
    MiopenBatchnormFwdTrainingPlanBuilder& operator=(const MiopenBatchnormFwdTrainingPlanBuilder&)
        = delete;

    bool isApplicable(const HipdnnEnginePluginHandle& handle,
                      const hipdnn_plugin_sdk::IGraph& opGraph) const override;
    size_t getWorkspaceSize(const HipdnnEnginePluginHandle& handle,
                            const hipdnn_plugin_sdk::IGraph& opGraph) const override;

    void buildPlan(const HipdnnEnginePluginHandle& handle,
                   const hipdnn_plugin_sdk::IGraph& opGraph,
                   HipdnnEnginePluginExecutionContext& executionContext) const override;
};

}
