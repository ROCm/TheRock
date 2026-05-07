// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <miopen/miopen.h>

#include <hipdnn_data_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_plugin_sdk/EnginePluginApi.h>
#include <hipdnn_plugin_sdk/PluginApi.h>
#include <hipdnn_plugin_sdk/PluginDataTypeHelpers.hpp>
#include <hipdnn_plugin_sdk/PluginHelpers.hpp>
#include <hipdnn_plugin_sdk/PluginLastErrorManager.hpp>

#include "EngineManager.hpp"
#include "HipdnnEnginePluginExecutionContext.hpp"
#include "HipdnnEnginePluginHandle.hpp"
#include "MiopenContainer.hpp"
#include "MiopenHandleFactory.hpp"
#include "MiopenLegacyPlugin.hpp"

static const char* pluginName = "miopen_legacy_plugin";
static const char* pluginVersion = "1.0.0";

using namespace hipdnn_plugin_sdk;
using namespace miopen_legacy_plugin;

// NOLINTNEXTLINE
thread_local char PluginLastErrorManager::s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH] = "";

// Keep a weak pointer to the MiopenContainer thats made when we create a plugin handle.
// The original shared_ptr is then stored on the handle so that it can be used for the lifecycle
// of the handle.  If we create another handle, then we can use the weak pointer to get access
// to the existing MiopenContainer.  If all handles are destroyed, then this allows us to properly
// clean up the container without having to fully unload the plugin.
std::weak_ptr<MiopenContainer> miopenContainerLifecyclePtr;

extern "C" {

hipdnnPluginStatus_t hipdnnPluginGetNameImpl(const char** name)
{
    LOG_API_ENTRY("name_ptr={:p}", static_cast<void*>(name));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        throwIfNull(name);

        *name = pluginName;

        LOG_API_SUCCESS(apiName, "pluginName={:p}", static_cast<void*>(name));
    });
}

hipdnnPluginStatus_t hipdnnPluginGetVersionImpl(const char** version)
{
    LOG_API_ENTRY("versionPtr={:p}", static_cast<void*>(version));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        throwIfNull(version);

        *version = pluginVersion;

        LOG_API_SUCCESS(apiName, "version={:p}", static_cast<void*>(version));
    });
}

hipdnnPluginStatus_t hipdnnPluginGetTypeImpl(hipdnnPluginType_t* type)
{
    LOG_API_ENTRY("typePtr={:p}", static_cast<void*>(type));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        throwIfNull(type);

        *type = HIPDNN_PLUGIN_TYPE_ENGINE;

        LOG_API_SUCCESS(apiName, "type={}", *type);
    });
}

void hipdnnPluginGetLastErrorStringImpl(const char** errorStr)
{
    LOG_API_ENTRY("errorStrPtr={:p}", static_cast<void*>(errorStr));

    hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        throwIfNull(errorStr);

        *errorStr = PluginLastErrorManager::getLastError();

        LOG_API_SUCCESS(apiName, "errorStr={:p}", static_cast<void*>(errorStr));
    });
}

// Once plugins are loaded via plugin manager then logging will work for them
hipdnnPluginStatus_t hipdnnPluginSetLoggingCallbackImpl(hipdnnCallback_t callback)
{
    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        throwIfNull(callback);
        hipdnn::logging::initializeCallbackLogging(pluginName, callback);
        LOG_API_SUCCESS(apiName, "", "");
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginGetAllEngineIdsImpl(int64_t* engineIds,
                                                           uint32_t maxEngines,
                                                           uint32_t* numEngines)
{
    LOG_API_ENTRY("engineIds={:p}, maxEngines={}, numEngines={:p}",
                  static_cast<void*>(engineIds),
                  maxEngines,
                  static_cast<void*>(numEngines));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        if(maxEngines != 0)
        {
            throwIfNull(engineIds);
        }
        throwIfNull(numEngines);

        // For now, we will just return a single engine ID.
        auto allEngineIds = std::vector<int64_t>({1});
        if(allEngineIds.size() > std::numeric_limits<uint32_t>::max())
        {
            throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                        "Number of engines exceeds maximum uint32_t value.");
        }

        if(maxEngines == 0)
        {
            *numEngines = static_cast<uint32_t>(allEngineIds.size());
        }
        else
        {
            *numEngines = 0;
            for(auto engineId : allEngineIds)
            {
                if(*numEngines == maxEngines)
                {
                    *numEngines = static_cast<uint32_t>(allEngineIds.size());
                    HIPDNN_LOG_INFO("Maximum number of engines reached ({}), ignoring additional "
                                    "engines, numEngines count: {}",
                                    maxEngines,
                                    *numEngines);
                    break;
                }

                engineIds[*numEngines] = engineId;
                (*numEngines)++;
            }
        }

        LOG_API_SUCCESS(apiName, "numEngines={}", *numEngines);
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginCreateImpl(hipdnnEnginePluginHandle_t* handle)
{
    LOG_API_ENTRY("handle_ptr={:p}", static_cast<void*>(handle));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        throwIfNull(handle);

        miopen_legacy_plugin::MiopenHandleFactory::createMiopenHandle(handle);

        auto miopenContainerPtr = miopenContainerLifecyclePtr.lock();
        if(miopenContainerPtr != nullptr)
        {
            (*handle)->miopenContainer = miopenContainerPtr;
        }
        else
        {
            static std::mutex s_miopenContainerMutex;
            std::lock_guard<std::mutex> lock(s_miopenContainerMutex);

            // if we do have a race condition that results in threads getting locked, we want to
            // ensure that we only create one instance.  Therefore, the second thread to get
            // through will just read from the weak pointer rather than create a new instance.
            miopenContainerPtr = miopenContainerLifecyclePtr.lock();
            if(miopenContainerPtr != nullptr)
            {
                (*handle)->miopenContainer = miopenContainerPtr;
            }
            else
            {
                (*handle)->miopenContainer = std::make_shared<MiopenContainer>();
                miopenContainerLifecyclePtr = (*handle)->miopenContainer;
            }
        }

        LOG_API_SUCCESS(apiName, "createdHandle={:p}", static_cast<void*>(*handle));
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginDestroyImpl(hipdnnEnginePluginHandle_t handle)
{
    LOG_API_ENTRY("handle={:p}", static_cast<void*>(handle));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        throwIfNull(handle);

        miopen_legacy_plugin::MiopenHandleFactory::destroyMiopenHandle(handle);
        delete handle;
        handle = nullptr;

        LOG_API_SUCCESS(apiName, "", "");
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginSetStreamImpl(hipdnnEnginePluginHandle_t handle,
                                                     hipStream_t stream)
{
    LOG_API_ENTRY(
        "handle={:p}, stream_id={:p}", static_cast<void*>(handle), static_cast<void*>(stream));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        throwIfNull(handle);

        handle->setStream(stream);

        LOG_API_SUCCESS(apiName, "", "");
    });
}

hipdnnPluginStatus_t
    hipdnnEnginePluginGetApplicableEngineIdsImpl(hipdnnEnginePluginHandle_t handle,
                                                 const hipdnnPluginConstData_t* opGraph,
                                                 int64_t* engineIds,
                                                 uint32_t maxEngines,
                                                 uint32_t* numEngines)
{
    LOG_API_ENTRY("handle={:p}, opGraph={:p}, engineIds={:p}, maxEngines={}, numEngines={:p}",
                  static_cast<void*>(handle),
                  static_cast<const void*>(opGraph),
                  static_cast<void*>(engineIds),
                  maxEngines,
                  static_cast<void*>(numEngines));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        throwIfNull(handle);
        throwIfNull(opGraph);
        if(maxEngines != 0)
        {
            throwIfNull(engineIds);
        }
        throwIfNull(numEngines);

        auto& engineManager = handle->getEngineManager();
        GraphWrapper opGraphWrapper(opGraph->ptr, opGraph->size);

        auto applicableEngines = engineManager.getApplicableEngineIds(*handle, opGraphWrapper);

        *numEngines = 0;
        for(auto& engineId : applicableEngines)
        {
            if(*numEngines == maxEngines)
            {
                *numEngines = static_cast<uint32_t>(applicableEngines.size());
                HIPDNN_LOG_INFO("Maximum number of engines reached ({}), ignoring additional "
                                "engines, numEngines count: {}",
                                maxEngines,
                                *numEngines);
                break;
            }

            engineIds[*numEngines] = engineId;
            (*numEngines)++;
        }

        LOG_API_SUCCESS(apiName, "numEngines={}", *numEngines);
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginGetEngineDetailsImpl(hipdnnEnginePluginHandle_t handle,
                                                            int64_t engineId,
                                                            const hipdnnPluginConstData_t* opGraph,
                                                            hipdnnPluginConstData_t* engineDetails)
{
    LOG_API_ENTRY("handle={:p}, engineId={}, opGraph={:p}, engineDetails={:p}",
                  static_cast<void*>(handle),
                  engineId,
                  static_cast<const void*>(opGraph),
                  static_cast<void*>(engineDetails));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        throwIfNull(handle);
        throwIfNull(opGraph);
        throwIfNull(engineDetails);

        auto& engineManager = handle->getEngineManager();
        GraphWrapper opGraphWrapper(opGraph->ptr, opGraph->size);

        engineManager.getEngineDetails(*handle, opGraphWrapper, engineId, *engineDetails);

        LOG_API_SUCCESS(apiName, "engineDetails->ptr={:p}", engineDetails->ptr);
    });
}

hipdnnPluginStatus_t
    hipdnnEnginePluginDestroyEngineDetailsImpl(hipdnnEnginePluginHandle_t handle,
                                               hipdnnPluginConstData_t* engineDetails)
{
    LOG_API_ENTRY("handle={:p}, engineDetails={}",
                  static_cast<void*>(handle),
                  static_cast<void*>(engineDetails));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        throwIfNull(handle);
        throwIfNull(engineDetails);
        throwIfNull(engineDetails->ptr);

        handle->removeEngineDetailsDetachedBuffer(engineDetails->ptr);

        LOG_API_SUCCESS(apiName, "engineDetails->ptr={:p}", engineDetails->ptr);
    });
}

hipdnnPluginStatus_t
    hipdnnEnginePluginGetWorkspaceSizeImpl(hipdnnEnginePluginHandle_t handle,
                                           const hipdnnPluginConstData_t* engineConfig,
                                           const hipdnnPluginConstData_t* opGraph,
                                           size_t* workspaceSize)
{
    LOG_API_ENTRY("handle={:p}, engineConfig={:p}, opGraph={:p}, workspaceSize={:p}",
                  static_cast<void*>(handle),
                  static_cast<const void*>(engineConfig),
                  static_cast<const void*>(opGraph),
                  static_cast<void*>(workspaceSize));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        throwIfNull(handle);
        throwIfNull(engineConfig);
        throwIfNull(opGraph);
        throwIfNull(workspaceSize);

        auto& engineManager = handle->getEngineManager();

        EngineConfigWrapper engineConfigWrapper(engineConfig->ptr, engineConfig->size);
        GraphWrapper opGraphWrapper(opGraph->ptr, opGraph->size);
        *workspaceSize = engineManager.getWorkspaceSize(
            *handle, engineConfigWrapper.engineId(), opGraphWrapper);

        LOG_API_SUCCESS(apiName, "workspaceSize={}", *workspaceSize);
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginCreateExecutionContextImpl(
    hipdnnEnginePluginHandle_t handle,
    const hipdnnPluginConstData_t* engineConfig,
    const hipdnnPluginConstData_t* opGraph,
    hipdnnEnginePluginExecutionContext_t* executionContext)
{
    LOG_API_ENTRY("handle={:p}, engineConfig={:p}, opGraph={:p}, executionContext={:p}",
                  static_cast<void*>(handle),
                  static_cast<const void*>(engineConfig),
                  static_cast<const void*>(opGraph),
                  static_cast<void*>(executionContext));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        throwIfNull(handle);
        throwIfNull(engineConfig);
        throwIfNull(opGraph);
        throwIfNull(executionContext);

        GraphWrapper opGraphWrapper(opGraph->ptr, opGraph->size);
        EngineConfigWrapper engineConfigWrapper(engineConfig->ptr, engineConfig->size);

        auto& engineManager = handle->getEngineManager();

        auto context = new HipdnnEnginePluginExecutionContext;

        try
        {
            engineManager.initializeExecutionContext(
                *handle, opGraphWrapper, engineConfigWrapper, *context);
        }
        catch(...)
        {
            delete context;
            throw;
        }

        *executionContext = context;

        LOG_API_SUCCESS(
            apiName, "created_execution_context={:p}", static_cast<void*>(*executionContext));
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginDestroyExecutionContextImpl(
    hipdnnEnginePluginHandle_t handle, hipdnnEnginePluginExecutionContext_t executionContext)
{
    LOG_API_ENTRY("handle={:p}, executionContext={:p}",
                  static_cast<void*>(handle),
                  static_cast<void*>(executionContext));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        throwIfNull(handle);
        throwIfNull(executionContext);

        delete executionContext;

        LOG_API_SUCCESS(apiName, "destroyed executionContext", "");
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginGetWorkspaceSizeFromExecutionContextImpl(
    hipdnnEnginePluginHandle_t handle,
    hipdnnEnginePluginExecutionContext_t executionContext,
    size_t* workspaceSize)
{
    LOG_API_ENTRY("handle={:p}, executionContext={:p}, workspaceSize={:p}",
                  static_cast<void*>(handle),
                  static_cast<const void*>(executionContext),
                  static_cast<void*>(workspaceSize));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        throwIfNull(handle);
        throwIfNull(executionContext);
        throwIfNull(workspaceSize);

        *workspaceSize = executionContext->plan().getWorkspaceSize(*handle);

        LOG_API_SUCCESS(apiName, "workspaceSize={}", *workspaceSize);
    });
}

hipdnnPluginStatus_t
    hipdnnEnginePluginExecuteOpGraphImpl(hipdnnEnginePluginHandle_t handle,
                                         hipdnnEnginePluginExecutionContext_t executionContext,
                                         void* workspace,
                                         const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                         uint32_t numDeviceBuffers)
{
    LOG_API_ENTRY("handle={:p}, executionContext={:p}, workspace={:p}, deviceBuffers={:p}, "
                  "numDeviceBuffers={}",
                  static_cast<void*>(handle),
                  static_cast<void*>(executionContext),
                  workspace,
                  static_cast<const void*>(deviceBuffers),
                  numDeviceBuffers);

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        throwIfNull(handle);
        throwIfNull(executionContext);
        throwIfNull(deviceBuffers);

        executionContext->plan().execute(*handle, deviceBuffers, numDeviceBuffers, workspace);

        LOG_API_SUCCESS(apiName, "executed graph", "");
    });
}

} // extern "C"
