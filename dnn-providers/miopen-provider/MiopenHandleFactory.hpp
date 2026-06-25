// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

namespace miopen_legacy_plugin
{

class MiopenHandleFactory
{
public:
    static void createMiopenHandle(hipdnnEnginePluginHandle_t* handle);

    static void destroyMiopenHandle(hipdnnEnginePluginHandle_t handle);
};

}
