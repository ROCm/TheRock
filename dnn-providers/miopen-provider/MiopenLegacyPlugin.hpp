// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime.h>
#include <hipdnn_plugin_sdk/PluginApi.h>

extern "C" {

hipdnnPluginStatus_t hipdnnPluginGetNameImpl(const char** name);

hipdnnPluginStatus_t hipdnnPluginGetVersionImpl(const char** version);
hipdnnPluginStatus_t hipdnnPluginGetTypeImpl(hipdnnPluginType_t* type);
void hipdnnPluginGetLastErrorStringImpl(const char** errorStr);
hipdnnPluginStatus_t hipdnnPluginSetLoggingCallbackImpl(hipdnnCallback_t callback);
hipdnnPluginStatus_t hipdnnEnginePluginGetAllEngineIdsImpl(int64_t* engineIds,
                                                           uint32_t maxEngines,
                                                           uint32_t* numEngines);
hipdnnPluginStatus_t hipdnnEnginePluginCreateImpl(hipdnnEnginePluginHandle_t* handle);
hipdnnPluginStatus_t hipdnnEnginePluginDestroyImpl(hipdnnEnginePluginHandle_t handle);
hipdnnPluginStatus_t hipdnnEnginePluginSetStreamImpl(hipdnnEnginePluginHandle_t handle,
                                                     hipStream_t stream);
hipdnnPluginStatus_t
    hipdnnEnginePluginGetApplicableEngineIdsImpl(hipdnnEnginePluginHandle_t handle,
                                                 const hipdnnPluginConstData_t* opGraph,
                                                 int64_t* engineIds,
                                                 uint32_t maxEngines,
                                                 uint32_t* numEngines);
hipdnnPluginStatus_t hipdnnEnginePluginGetEngineDetailsImpl(hipdnnEnginePluginHandle_t handle,
                                                            int64_t engineId,
                                                            const hipdnnPluginConstData_t* opGraph,
                                                            hipdnnPluginConstData_t* engineDetails);
hipdnnPluginStatus_t
    hipdnnEnginePluginDestroyEngineDetailsImpl(hipdnnEnginePluginHandle_t handle,
                                               hipdnnPluginConstData_t* engineDetails);
hipdnnPluginStatus_t
    hipdnnEnginePluginGetWorkspaceSizeImpl(hipdnnEnginePluginHandle_t handle,
                                           const hipdnnPluginConstData_t* engineConfig,
                                           const hipdnnPluginConstData_t* opGraph,
                                           size_t* workspaceSize);
hipdnnPluginStatus_t hipdnnEnginePluginCreateExecutionContextImpl(
    hipdnnEnginePluginHandle_t handle,
    const hipdnnPluginConstData_t* engineConfig,
    const hipdnnPluginConstData_t* opGraph,
    hipdnnEnginePluginExecutionContext_t* executionContext);
hipdnnPluginStatus_t hipdnnEnginePluginDestroyExecutionContextImpl(
    hipdnnEnginePluginHandle_t handle, hipdnnEnginePluginExecutionContext_t executionContext);
hipdnnPluginStatus_t hipdnnEnginePluginGetWorkspaceSizeFromExecutionContextImpl(
    hipdnnEnginePluginHandle_t handle,
    hipdnnEnginePluginExecutionContext_t executionContext,
    size_t* workspaceSize);
hipdnnPluginStatus_t
    hipdnnEnginePluginExecuteOpGraphImpl(hipdnnEnginePluginHandle_t handle,
                                         hipdnnEnginePluginExecutionContext_t executionContext,
                                         void* workspace,
                                         const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                         uint32_t numDeviceBuffers);

} // extern "C"
