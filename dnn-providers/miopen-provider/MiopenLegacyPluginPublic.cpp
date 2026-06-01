// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipdnn_plugin_sdk/EnginePluginApi.h>
#include <hipdnn_plugin_sdk/PluginApi.h>

#include "MiopenLegacyPlugin.hpp"

extern "C" {

hipdnnPluginStatus_t hipdnnPluginGetName(const char** name)
{
    return hipdnnPluginGetNameImpl(name);
}

hipdnnPluginStatus_t hipdnnPluginGetVersion(const char** version)
{
    return hipdnnPluginGetVersionImpl(version);
}

hipdnnPluginStatus_t hipdnnPluginGetType(hipdnnPluginType_t* type)
{
    return hipdnnPluginGetTypeImpl(type);
}

void hipdnnPluginGetLastErrorString(const char** errorStr)
{
    hipdnnPluginGetLastErrorStringImpl(errorStr);
}

// Once plugins are loaded via plugin manager then logging will work for them
hipdnnPluginStatus_t hipdnnPluginSetLoggingCallback(hipdnnCallback_t callback)
{
    return hipdnnPluginSetLoggingCallbackImpl(callback);
}

hipdnnPluginStatus_t
    hipdnnEnginePluginGetAllEngineIds(int64_t* engineIds, uint32_t maxEngines, uint32_t* numEngines)
{
    return hipdnnEnginePluginGetAllEngineIdsImpl(engineIds, maxEngines, numEngines);
}

hipdnnPluginStatus_t hipdnnEnginePluginCreate(hipdnnEnginePluginHandle_t* handle)
{
    return hipdnnEnginePluginCreateImpl(handle);
}

hipdnnPluginStatus_t hipdnnEnginePluginDestroy(hipdnnEnginePluginHandle_t handle)
{
    return hipdnnEnginePluginDestroyImpl(handle);
}

hipdnnPluginStatus_t hipdnnEnginePluginSetStream(hipdnnEnginePluginHandle_t handle,
                                                 hipStream_t stream)
{
    return hipdnnEnginePluginSetStreamImpl(handle, stream);
}

hipdnnPluginStatus_t
    hipdnnEnginePluginGetApplicableEngineIds(hipdnnEnginePluginHandle_t handle,
                                             const hipdnnPluginConstData_t* opGraph,
                                             int64_t* engineIds,
                                             uint32_t maxEngines,
                                             uint32_t* numEngines)
{
    return hipdnnEnginePluginGetApplicableEngineIdsImpl(
        handle, opGraph, engineIds, maxEngines, numEngines);
}

hipdnnPluginStatus_t hipdnnEnginePluginGetEngineDetails(hipdnnEnginePluginHandle_t handle,
                                                        int64_t engineId,
                                                        const hipdnnPluginConstData_t* opGraph,
                                                        hipdnnPluginConstData_t* engineDetails)
{
    return hipdnnEnginePluginGetEngineDetailsImpl(handle, engineId, opGraph, engineDetails);
}

hipdnnPluginStatus_t hipdnnEnginePluginDestroyEngineDetails(hipdnnEnginePluginHandle_t handle,
                                                            hipdnnPluginConstData_t* engineDetails)
{
    return hipdnnEnginePluginDestroyEngineDetailsImpl(handle, engineDetails);
}

hipdnnPluginStatus_t hipdnnEnginePluginGetWorkspaceSize(hipdnnEnginePluginHandle_t handle,
                                                        const hipdnnPluginConstData_t* engineConfig,
                                                        const hipdnnPluginConstData_t* opGraph,
                                                        size_t* workspaceSize)
{
    return hipdnnEnginePluginGetWorkspaceSizeImpl(handle, engineConfig, opGraph, workspaceSize);
}

hipdnnPluginStatus_t
    hipdnnEnginePluginCreateExecutionContext(hipdnnEnginePluginHandle_t handle,
                                             const hipdnnPluginConstData_t* engineConfig,
                                             const hipdnnPluginConstData_t* opGraph,
                                             hipdnnEnginePluginExecutionContext_t* executionContext)
{
    return hipdnnEnginePluginCreateExecutionContextImpl(
        handle, engineConfig, opGraph, executionContext);
}

hipdnnPluginStatus_t
    hipdnnEnginePluginDestroyExecutionContext(hipdnnEnginePluginHandle_t handle,
                                              hipdnnEnginePluginExecutionContext_t executionContext)
{
    return hipdnnEnginePluginDestroyExecutionContextImpl(handle, executionContext);
}

hipdnnPluginStatus_t hipdnnEnginePluginGetWorkspaceSizeFromExecutionContext(
    hipdnnEnginePluginHandle_t handle,
    hipdnnEnginePluginExecutionContext_t executionContext,
    size_t* workspaceSize)
{
    return hipdnnEnginePluginGetWorkspaceSizeFromExecutionContextImpl(
        handle, executionContext, workspaceSize);
}

hipdnnPluginStatus_t
    hipdnnEnginePluginExecuteOpGraph(hipdnnEnginePluginHandle_t handle,
                                     hipdnnEnginePluginExecutionContext_t executionContext,
                                     void* workspace,
                                     const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                     uint32_t numDeviceBuffers)
{
    return hipdnnEnginePluginExecuteOpGraphImpl(
        handle, executionContext, workspace, deviceBuffers, numDeviceBuffers);
}

} // extern "C"
