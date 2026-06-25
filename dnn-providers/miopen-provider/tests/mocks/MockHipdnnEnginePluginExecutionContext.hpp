// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>

#include "mocks/MockPlan.hpp"

#include "HipdnnEnginePluginExecutionContext.hpp"

struct MockHipdnnEnginePluginExecutionContext : public HipdnnEnginePluginExecutionContext
{
    MockHipdnnEnginePluginExecutionContext()
        : mockPlan(std::make_unique<miopen_legacy_plugin::MockPlan>())
    {
    }

    miopen_legacy_plugin::IPlan& plan() const override
    {
        return *mockPlan;
    }

    std::unique_ptr<miopen_legacy_plugin::MockPlan> mockPlan;
};
