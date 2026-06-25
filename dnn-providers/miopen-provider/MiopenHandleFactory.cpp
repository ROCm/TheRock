// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "HipdnnEnginePluginHandle.hpp"
#include "MiopenHandleFactory.hpp"

using namespace hipdnn_plugin_sdk;

namespace miopen_legacy_plugin
{

void MiopenHandleFactory::createMiopenHandle(hipdnnEnginePluginHandle_t* handle)
{
    if(handle == nullptr)
    {
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM, "handle is null");
    }

    *handle = new HipdnnEnginePluginHandle();

    miopenStatus_t status = miopenCreate(&(*handle)->miopenHandle);

    if(status != miopenStatusSuccess)
    {
        delete *handle;
        *handle = nullptr;
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                    "Failed to create MIOpen handle");
    }
}

void MiopenHandleFactory::destroyMiopenHandle(hipdnnEnginePluginHandle_t handle)
{
    if(handle == nullptr)
    {
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM, "handle is null");
    }

    miopenStatus_t status = miopenDestroy(handle->miopenHandle);

    if(status != miopenStatusSuccess)
    {
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                    "Failed to destroy MIOpen handle");
    }
}

} // namespace miopen_legacy_plugin
