// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "engines/plans/PlanInterface.hpp"
#include <gmock/gmock.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

namespace miopen_legacy_plugin
{

class MockPlan : public IPlan
{
public:
    MOCK_METHOD(size_t,
                getWorkspaceSize,
                (const HipdnnEnginePluginHandle& handle),
                (const, override));
    MOCK_METHOD(void,
                execute,
                (const HipdnnEnginePluginHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace),
                (const, override));
};

}
