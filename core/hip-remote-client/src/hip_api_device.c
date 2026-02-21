/*
 * Copyright 2025 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file hip_api_device.c
 * @brief Device management API implementations for remote HIP
 */

#include "hip_remote/hip_remote_client.h"
#include "hip_remote/hip_remote_protocol.h"

#include <string.h>

/* ============================================================================
 * Device Management APIs
 * ============================================================================ */

hipError_t hipGetDeviceCount(int* count) {
    if (!count) {
        return hipErrorInvalidValue;
    }

    HipRemoteDeviceCountResponse resp;
    hipError_t err = hip_remote_request(
        HIP_OP_GET_DEVICE_COUNT,
        NULL, 0,
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *count = resp.count;
    }
    return err;
}

hipError_t hipSetDevice(int deviceId) {
    HipRemoteDeviceRequest req = { .device_id = deviceId };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_SET_DEVICE,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipGetDevice(int* deviceId) {
    if (!deviceId) {
        return hipErrorInvalidValue;
    }

    HipRemoteGetDeviceResponse resp;
    hipError_t err = hip_remote_request(
        HIP_OP_GET_DEVICE,
        NULL, 0,
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *deviceId = resp.device_id;
    }
    return err;
}

hipError_t hipDeviceSynchronize(void) {
    HipRemoteResponseHeader resp;
    return hip_remote_request(
        HIP_OP_DEVICE_SYNCHRONIZE,
        NULL, 0,
        &resp, sizeof(resp)
    );
}

hipError_t hipDeviceReset(void) {
    HipRemoteResponseHeader resp;
    return hip_remote_request(
        HIP_OP_DEVICE_RESET,
        NULL, 0,
        &resp, sizeof(resp)
    );
}

hipError_t hipDeviceGetAttribute(int* value, int attr, int deviceId) {
    if (!value) {
        return hipErrorInvalidValue;
    }

    HipRemoteDeviceAttributeRequest req = {
        .device_id = deviceId,
        .attribute = attr
    };
    HipRemoteDeviceAttributeResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_DEVICE_GET_ATTRIBUTE,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *value = resp.value;
    }
    return err;
}

/* ============================================================================
 * Device Properties
 *
 * Note: We define a minimal hipDeviceProp_t here for the remote client.
 * The full definition would come from HIP headers, but for the remote
 * client we only need to map the fields we care about.
 * ============================================================================ */

typedef struct {
    char name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    int memoryClockRate;
    int memoryBusWidth;
    int major;
    int minor;
    int multiProcessorCount;
    int l2CacheSize;
    int maxThreadsPerMultiProcessor;
    int computeMode;
    int pciBusId;
    int pciDeviceId;
    int pciDomainId;
    int integrated;
    int canMapHostMemory;
    int concurrentKernels;
    char gcnArchName[256];
    /* ... additional fields would go here ... */
} hipDeviceProp_t_Remote;

hipError_t hipGetDeviceProperties(void* prop, int deviceId) {
    if (!prop) {
        return hipErrorInvalidValue;
    }

    HipRemoteDeviceRequest req = { .device_id = deviceId };
    HipRemoteDevicePropertiesResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_GET_DEVICE_PROPERTIES,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        hipDeviceProp_t_Remote* p = (hipDeviceProp_t_Remote*)prop;
        memset(p, 0, sizeof(*p));

        strncpy(p->name, resp.name, sizeof(p->name) - 1);
        p->totalGlobalMem = resp.total_global_mem;
        p->sharedMemPerBlock = resp.shared_mem_per_block;
        p->regsPerBlock = resp.regs_per_block;
        p->warpSize = resp.warp_size;
        p->maxThreadsPerBlock = resp.max_threads_per_block;
        p->maxThreadsDim[0] = resp.max_threads_dim[0];
        p->maxThreadsDim[1] = resp.max_threads_dim[1];
        p->maxThreadsDim[2] = resp.max_threads_dim[2];
        p->maxGridSize[0] = resp.max_grid_size[0];
        p->maxGridSize[1] = resp.max_grid_size[1];
        p->maxGridSize[2] = resp.max_grid_size[2];
        p->clockRate = resp.clock_rate;
        p->memoryClockRate = resp.memory_clock_rate;
        p->memoryBusWidth = resp.memory_bus_width;
        p->major = resp.major;
        p->minor = resp.minor;
        p->multiProcessorCount = resp.multi_processor_count;
        p->l2CacheSize = resp.l2_cache_size;
        p->maxThreadsPerMultiProcessor = resp.max_threads_per_multi_processor;
        p->computeMode = resp.compute_mode;
        p->pciBusId = resp.pci_bus_id;
        p->pciDeviceId = resp.pci_device_id;
        p->pciDomainId = resp.pci_domain_id;
        p->integrated = resp.integrated;
        p->canMapHostMemory = resp.can_map_host_memory;
        p->concurrentKernels = resp.concurrent_kernels;
        strncpy(p->gcnArchName, resp.gcn_arch_name, sizeof(p->gcnArchName) - 1);
    }

    return err;
}

/* ============================================================================
 * Runtime/Driver Version
 * ============================================================================ */

hipError_t hipRuntimeGetVersion(int* runtimeVersion) {
    if (!runtimeVersion) {
        return hipErrorInvalidValue;
    }

    HipRemoteVersionResponse resp;
    hipError_t err = hip_remote_request(
        HIP_OP_RUNTIME_GET_VERSION,
        NULL, 0,
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *runtimeVersion = resp.version;
    }
    return err;
}

hipError_t hipDriverGetVersion(int* driverVersion) {
    if (!driverVersion) {
        return hipErrorInvalidValue;
    }

    HipRemoteVersionResponse resp;
    hipError_t err = hip_remote_request(
        HIP_OP_DRIVER_GET_VERSION,
        NULL, 0,
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *driverVersion = resp.version;
    }
    return err;
}

/* ============================================================================
 * Error Handling
 * ============================================================================ */

hipError_t hipGetLastError(void) {
    HipRemoteResponseHeader resp;
    return hip_remote_request(
        HIP_OP_GET_LAST_ERROR,
        NULL, 0,
        &resp, sizeof(resp)
    );
}

hipError_t hipPeekAtLastError(void) {
    HipRemoteResponseHeader resp;
    return hip_remote_request(
        HIP_OP_PEEK_AT_LAST_ERROR,
        NULL, 0,
        &resp, sizeof(resp)
    );
}

/*
 * Error string implementations.
 *
 * These provide the same functionality as the HIP runtime functions in
 * rocm-systems/projects/clr/hipamd/src/hip_error.cpp.
 *
 * We implement these locally because:
 * 1. The remote client runs on macOS where HIP libraries are not available
 * 2. The client needs to translate error codes returned from the worker
 *
 * Error codes match hip_runtime_api.h. Last synced: HIP 6.3 (ROCm 6.3)
 */
const char* hipGetErrorName(hipError_t error) {
    switch (error) {
        case hipSuccess: return "hipSuccess";
        case hipErrorInvalidValue: return "hipErrorInvalidValue";
        case hipErrorOutOfMemory: return "hipErrorOutOfMemory";
        case hipErrorNotInitialized: return "hipErrorNotInitialized";
        case hipErrorDeinitialized: return "hipErrorDeinitialized";
        case hipErrorProfilerDisabled: return "hipErrorProfilerDisabled";
        case hipErrorProfilerNotInitialized: return "hipErrorProfilerNotInitialized";
        case hipErrorProfilerAlreadyStarted: return "hipErrorProfilerAlreadyStarted";
        case hipErrorProfilerAlreadyStopped: return "hipErrorProfilerAlreadyStopped";
        case hipErrorInvalidConfiguration: return "hipErrorInvalidConfiguration";
        case hipErrorInvalidSymbol: return "hipErrorInvalidSymbol";
        case hipErrorInvalidDevicePointer: return "hipErrorInvalidDevicePointer";
        case hipErrorInvalidMemcpyDirection: return "hipErrorInvalidMemcpyDirection";
        case hipErrorInsufficientDriver: return "hipErrorInsufficientDriver";
        case hipErrorMissingConfiguration: return "hipErrorMissingConfiguration";
        case hipErrorPriorLaunchFailure: return "hipErrorPriorLaunchFailure";
        case hipErrorInvalidDeviceFunction: return "hipErrorInvalidDeviceFunction";
        case hipErrorNoDevice: return "hipErrorNoDevice";
        case hipErrorInvalidDevice: return "hipErrorInvalidDevice";
        case hipErrorInvalidPitchValue: return "hipErrorInvalidPitchValue";
        case hipErrorInvalidImage: return "hipErrorInvalidImage";
        case hipErrorInvalidContext: return "hipErrorInvalidContext";
        case hipErrorContextAlreadyCurrent: return "hipErrorContextAlreadyCurrent";
        case hipErrorMapFailed: return "hipErrorMapFailed";
        case hipErrorUnmapFailed: return "hipErrorUnmapFailed";
        case hipErrorArrayIsMapped: return "hipErrorArrayIsMapped";
        case hipErrorAlreadyMapped: return "hipErrorAlreadyMapped";
        case hipErrorNoBinaryForGpu: return "hipErrorNoBinaryForGpu";
        case hipErrorAlreadyAcquired: return "hipErrorAlreadyAcquired";
        case hipErrorNotMapped: return "hipErrorNotMapped";
        case hipErrorNotMappedAsArray: return "hipErrorNotMappedAsArray";
        case hipErrorNotMappedAsPointer: return "hipErrorNotMappedAsPointer";
        case hipErrorECCNotCorrectable: return "hipErrorECCNotCorrectable";
        case hipErrorUnsupportedLimit: return "hipErrorUnsupportedLimit";
        case hipErrorContextAlreadyInUse: return "hipErrorContextAlreadyInUse";
        case hipErrorPeerAccessUnsupported: return "hipErrorPeerAccessUnsupported";
        case hipErrorInvalidKernelFile: return "hipErrorInvalidKernelFile";
        case hipErrorInvalidGraphicsContext: return "hipErrorInvalidGraphicsContext";
        case hipErrorInvalidSource: return "hipErrorInvalidSource";
        case hipErrorFileNotFound: return "hipErrorFileNotFound";
        case hipErrorSharedObjectSymbolNotFound: return "hipErrorSharedObjectSymbolNotFound";
        case hipErrorSharedObjectInitFailed: return "hipErrorSharedObjectInitFailed";
        case hipErrorOperatingSystem: return "hipErrorOperatingSystem";
        case hipErrorInvalidHandle: return "hipErrorInvalidHandle";
        case hipErrorIllegalState: return "hipErrorIllegalState";
        case hipErrorNotFound: return "hipErrorNotFound";
        case hipErrorNotReady: return "hipErrorNotReady";
        case hipErrorIllegalAddress: return "hipErrorIllegalAddress";
        case hipErrorLaunchOutOfResources: return "hipErrorLaunchOutOfResources";
        case hipErrorLaunchTimeOut: return "hipErrorLaunchTimeOut";
        case hipErrorPeerAccessAlreadyEnabled: return "hipErrorPeerAccessAlreadyEnabled";
        case hipErrorPeerAccessNotEnabled: return "hipErrorPeerAccessNotEnabled";
        case hipErrorSetOnActiveProcess: return "hipErrorSetOnActiveProcess";
        case hipErrorContextIsDestroyed: return "hipErrorContextIsDestroyed";
        case hipErrorAssert: return "hipErrorAssert";
        case hipErrorHostMemoryAlreadyRegistered: return "hipErrorHostMemoryAlreadyRegistered";
        case hipErrorHostMemoryNotRegistered: return "hipErrorHostMemoryNotRegistered";
        case hipErrorLaunchFailure: return "hipErrorLaunchFailure";
        case hipErrorNotSupported: return "hipErrorNotSupported";
        case hipErrorUnknown: return "hipErrorUnknown";
        case hipErrorRuntimeMemory: return "hipErrorRuntimeMemory";
        case hipErrorRuntimeOther: return "hipErrorRuntimeOther";
        case hipErrorCooperativeLaunchTooLarge: return "hipErrorCooperativeLaunchTooLarge";
        case hipErrorStreamCaptureUnsupported: return "hipErrorStreamCaptureUnsupported";
        case hipErrorStreamCaptureInvalidated: return "hipErrorStreamCaptureInvalidated";
        case hipErrorStreamCaptureMerge: return "hipErrorStreamCaptureMerge";
        case hipErrorStreamCaptureUnmatched: return "hipErrorStreamCaptureUnmatched";
        case hipErrorStreamCaptureUnjoined: return "hipErrorStreamCaptureUnjoined";
        case hipErrorStreamCaptureIsolation: return "hipErrorStreamCaptureIsolation";
        case hipErrorStreamCaptureImplicit: return "hipErrorStreamCaptureImplicit";
        case hipErrorCapturedEvent: return "hipErrorCapturedEvent";
        case hipErrorStreamCaptureWrongThread: return "hipErrorStreamCaptureWrongThread";
        case hipErrorGraphExecUpdateFailure: return "hipErrorGraphExecUpdateFailure";
        case hipErrorInvalidChannelDescriptor: return "hipErrorInvalidChannelDescriptor";
        case hipErrorInvalidTexture: return "hipErrorInvalidTexture";
        case hipErrorTbd: return "hipErrorTbd";
        default: return "hipErrorUnknown";
    }
}

const char* hipGetErrorString(hipError_t error) {
    switch (error) {
        case hipSuccess: return "no error";
        case hipErrorInvalidValue: return "invalid argument";
        case hipErrorOutOfMemory: return "out of memory";
        case hipErrorNotInitialized: return "initialization error";
        case hipErrorDeinitialized: return "driver shutting down";
        case hipErrorProfilerDisabled: return "profiler disabled while using external profiling tool";
        case hipErrorProfilerNotInitialized: return "profiler is not initialized";
        case hipErrorProfilerAlreadyStarted: return "profiler already started";
        case hipErrorProfilerAlreadyStopped: return "profiler already stopped";
        case hipErrorInvalidConfiguration: return "invalid configuration argument";
        case hipErrorInvalidPitchValue: return "invalid pitch argument";
        case hipErrorInvalidSymbol: return "invalid device symbol";
        case hipErrorInvalidDevicePointer: return "invalid device pointer";
        case hipErrorInvalidMemcpyDirection: return "invalid copy direction for memcpy";
        case hipErrorInsufficientDriver: return "driver version is insufficient for runtime version";
        case hipErrorMissingConfiguration: return "__global__ function call is not configured";
        case hipErrorPriorLaunchFailure: return "unspecified launch failure in prior launch";
        case hipErrorInvalidDeviceFunction: return "invalid device function";
        case hipErrorNoDevice: return "no ROCm-capable device is detected";
        case hipErrorInvalidDevice: return "invalid device ordinal";
        case hipErrorInvalidImage: return "device kernel image is invalid";
        case hipErrorInvalidContext: return "invalid device context";
        case hipErrorContextAlreadyCurrent: return "context is already current context";
        case hipErrorMapFailed: return "mapping of buffer object failed";
        case hipErrorUnmapFailed: return "unmapping of buffer object failed";
        case hipErrorArrayIsMapped: return "array is mapped";
        case hipErrorAlreadyMapped: return "resource already mapped";
        case hipErrorNoBinaryForGpu: return "no kernel image is available for execution on the device";
        case hipErrorAlreadyAcquired: return "resource already acquired";
        case hipErrorNotMapped: return "resource not mapped";
        case hipErrorNotMappedAsArray: return "resource not mapped as array";
        case hipErrorNotMappedAsPointer: return "resource not mapped as pointer";
        case hipErrorECCNotCorrectable: return "uncorrectable ECC error encountered";
        case hipErrorUnsupportedLimit: return "limit is not supported on this architecture";
        case hipErrorContextAlreadyInUse: return "exclusive-thread device already in use by a different thread";
        case hipErrorPeerAccessUnsupported: return "peer access is not supported between these two devices";
        case hipErrorInvalidKernelFile: return "invalid kernel file";
        case hipErrorInvalidGraphicsContext: return "invalid OpenGL or DirectX context";
        case hipErrorInvalidSource: return "device kernel image is invalid";
        case hipErrorFileNotFound: return "file not found";
        case hipErrorSharedObjectSymbolNotFound: return "shared object symbol not found";
        case hipErrorSharedObjectInitFailed: return "shared object initialization failed";
        case hipErrorOperatingSystem: return "OS call failed or operation not supported on this OS";
        case hipErrorInvalidHandle: return "invalid resource handle";
        case hipErrorIllegalState: return "the operation cannot be performed in the present state";
        case hipErrorNotFound: return "named symbol not found";
        case hipErrorNotReady: return "device not ready";
        case hipErrorIllegalAddress: return "an illegal memory access was encountered";
        case hipErrorLaunchOutOfResources: return "too many resources requested for launch";
        case hipErrorLaunchTimeOut: return "the launch timed out and was terminated";
        case hipErrorPeerAccessAlreadyEnabled: return "peer access is already enabled";
        case hipErrorPeerAccessNotEnabled: return "peer access has not been enabled";
        case hipErrorSetOnActiveProcess: return "cannot set while device is active in this process";
        case hipErrorContextIsDestroyed: return "context is destroyed";
        case hipErrorAssert: return "device-side assert triggered";
        case hipErrorHostMemoryAlreadyRegistered: return "part or all of the requested memory range is already mapped";
        case hipErrorHostMemoryNotRegistered: return "pointer does not correspond to a registered memory region";
        case hipErrorLaunchFailure: return "unspecified launch failure";
        case hipErrorCooperativeLaunchTooLarge: return "too many blocks in cooperative launch";
        case hipErrorNotSupported: return "operation not supported";
        case hipErrorStreamCaptureUnsupported: return "operation not permitted when stream is capturing";
        case hipErrorStreamCaptureInvalidated: return "operation failed due to a previous error during capture";
        case hipErrorStreamCaptureMerge: return "operation would result in a merge of separate capture sequences";
        case hipErrorStreamCaptureUnmatched: return "capture was not ended in the same stream as it began";
        case hipErrorStreamCaptureUnjoined: return "capturing stream has unjoined work";
        case hipErrorStreamCaptureIsolation: return "dependency created on uncaptured work in another stream";
        case hipErrorStreamCaptureImplicit: return "operation would make the legacy stream depend on a capturing blocking stream";
        case hipErrorCapturedEvent: return "operation not permitted on an event last recorded in a capturing stream";
        case hipErrorStreamCaptureWrongThread: return "attempt to terminate a thread-local capture sequence from another thread";
        case hipErrorGraphExecUpdateFailure: return "the graph update was not performed because it included changes which violated constraints specific to instantiated graph update";
        case hipErrorInvalidChannelDescriptor: return "invalid channel descriptor";
        case hipErrorInvalidTexture: return "invalid texture";
        case hipErrorRuntimeMemory: return "runtime memory call returned error";
        case hipErrorRuntimeOther: return "runtime call other than memory returned error";
        case hipErrorUnknown:
        default: return "unknown error";
    }
}
