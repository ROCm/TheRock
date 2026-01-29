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

/* Local error string mapping (no remote call needed) */
const char* hipGetErrorString(hipError_t error) {
    switch (error) {
        case hipSuccess: return "no error";
        case hipErrorInvalidValue: return "invalid argument";
        case hipErrorOutOfMemory: return "out of memory";
        case hipErrorNotInitialized: return "driver not initialized";
        case hipErrorDeinitialized: return "driver deinitialized";
        case hipErrorInvalidDevice: return "invalid device ordinal";
        case hipErrorInvalidResourceHandle: return "invalid resource handle";
        case hipErrorNotReady: return "device not ready";
        case hipErrorNoDevice: return "no HIP-capable device";
        case hipErrorNotSupported: return "operation not supported";
        case hipErrorLaunchFailure: return "launch failure";
        case hipErrorLaunchOutOfResources: return "launch out of resources";
        case hipErrorLaunchTimeOut: return "launch timeout";
        case hipErrorHostMemoryAlreadyRegistered: return "host memory already registered";
        case hipErrorHostMemoryNotRegistered: return "host memory not registered";
        default: return "unknown error";
    }
}

const char* hipGetErrorName(hipError_t error) {
    switch (error) {
        case hipSuccess: return "hipSuccess";
        case hipErrorInvalidValue: return "hipErrorInvalidValue";
        case hipErrorOutOfMemory: return "hipErrorOutOfMemory";
        case hipErrorNotInitialized: return "hipErrorNotInitialized";
        case hipErrorDeinitialized: return "hipErrorDeinitialized";
        case hipErrorInvalidDevice: return "hipErrorInvalidDevice";
        case hipErrorInvalidResourceHandle: return "hipErrorInvalidResourceHandle";
        case hipErrorNotReady: return "hipErrorNotReady";
        case hipErrorNoDevice: return "hipErrorNoDevice";
        case hipErrorNotSupported: return "hipErrorNotSupported";
        case hipErrorLaunchFailure: return "hipErrorLaunchFailure";
        case hipErrorLaunchOutOfResources: return "hipErrorLaunchOutOfResources";
        case hipErrorLaunchTimeOut: return "hipErrorLaunchTimeOut";
        case hipErrorHostMemoryAlreadyRegistered: return "hipErrorHostMemoryAlreadyRegistered";
        case hipErrorHostMemoryNotRegistered: return "hipErrorHostMemoryNotRegistered";
        default: return "hipErrorUnknown";
    }
}
