// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

namespace miopen_legacy_plugin
{

class IPlan
{
public:
    virtual ~IPlan() = default;

    virtual size_t getWorkspaceSize(const HipdnnEnginePluginHandle& handle) const = 0;

    virtual void execute(const HipdnnEnginePluginHandle& handle,
                         const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                         uint32_t numDeviceBuffers,
                         void* workspace = nullptr) const
        = 0;
};

}
