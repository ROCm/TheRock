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
 * @file hip_worker_main.c
 * @brief HIP worker service for remote HIP execution
 *
 * This service runs on a Linux system with AMD GPUs and handles
 * HIP API requests from remote clients (e.g., macOS).
 */

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#include <pthread.h>

#include <hip/hip_runtime.h>

/* Include protocol from client */
#include "hip_remote/hip_remote_protocol.h"

/* ============================================================================
 * Configuration
 * ============================================================================ */

static int g_listen_port = HIP_REMOTE_DEFAULT_PORT;
static int g_default_device = 0;
static bool g_debug_enabled = false;
static volatile bool g_running = true;
static int g_server_fd = -1;

/* ============================================================================
 * Logging
 * ============================================================================ */

#define LOG_DEBUG(fmt, ...) do { \
    if (g_debug_enabled) { \
        fprintf(stderr, "[HIP-Worker] " fmt "\n", ##__VA_ARGS__); \
    } \
} while (0)

#define LOG_INFO(fmt, ...) \
    fprintf(stdout, "[HIP-Worker] " fmt "\n", ##__VA_ARGS__)

#define LOG_ERROR(fmt, ...) \
    fprintf(stderr, "[HIP-Worker ERROR] " fmt "\n", ##__VA_ARGS__)

/* ============================================================================
 * Network Helpers
 * ============================================================================ */

static int send_all(int fd, const void* data, size_t len) {
    const uint8_t* p = (const uint8_t*)data;
    while (len > 0) {
#ifdef MSG_NOSIGNAL
        ssize_t n = send(fd, p, len, MSG_NOSIGNAL);
#else
        ssize_t n = send(fd, p, len, 0);
#endif
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) {
            errno = EPIPE;
            return -1;
        }
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 0;
}

static int recv_all(int fd, void* data, size_t len) {
    uint8_t* p = (uint8_t*)data;
    while (len > 0) {
        ssize_t n = recv(fd, p, len, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) {
            errno = ECONNRESET;
            return -1;
        }
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 0;
}

/* ============================================================================
 * Response Helpers
 * ============================================================================ */

static int send_response(int fd, HipRemoteOpCode op_code, uint32_t request_id,
                         const void* payload, size_t payload_size) {
    HipRemoteHeader header;
    hip_remote_init_header(&header, op_code, request_id, (uint32_t)payload_size);
    header.flags |= HIP_REMOTE_FLAG_RESPONSE;

    if (send_all(fd, &header, sizeof(header)) != 0) return -1;
    if (payload && payload_size > 0) {
        if (send_all(fd, payload, payload_size) != 0) return -1;
    }
    return 0;
}

static int send_simple_response(int fd, HipRemoteOpCode op_code,
                                uint32_t request_id, hipError_t err) {
    HipRemoteResponseHeader resp = { .error_code = (int32_t)err };
    return send_response(fd, op_code, request_id, &resp, sizeof(resp));
}

/* ============================================================================
 * Request Handlers
 * ============================================================================ */

static void handle_init(int fd, uint32_t request_id) {
    hipError_t err = hipSetDevice(g_default_device);
    LOG_DEBUG("Init: device=%d, err=%d", g_default_device, err);
    send_simple_response(fd, HIP_OP_INIT, request_id, err);
}

static void handle_shutdown(int fd, uint32_t request_id) {
    LOG_DEBUG("Shutdown");
    send_simple_response(fd, HIP_OP_SHUTDOWN, request_id, hipSuccess);
}

static void handle_get_device_count(int fd, uint32_t request_id) {
    int count = 0;
    hipError_t err = hipGetDeviceCount(&count);
    LOG_DEBUG("GetDeviceCount: count=%d, err=%d", count, err);

    HipRemoteDeviceCountResponse resp = {
        .header = { .error_code = (int32_t)err },
        .count = count
    };
    send_response(fd, HIP_OP_GET_DEVICE_COUNT, request_id, &resp, sizeof(resp));
}

static void handle_set_device(int fd, uint32_t request_id,
                              const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteDeviceRequest)) {
        send_simple_response(fd, HIP_OP_SET_DEVICE, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteDeviceRequest* req = (const HipRemoteDeviceRequest*)payload;
    hipError_t err = hipSetDevice(req->device_id);
    LOG_DEBUG("SetDevice: device=%d, err=%d", req->device_id, err);
    send_simple_response(fd, HIP_OP_SET_DEVICE, request_id, err);
}

static void handle_get_device(int fd, uint32_t request_id) {
    int device = 0;
    hipError_t err = hipGetDevice(&device);
    LOG_DEBUG("GetDevice: device=%d, err=%d", device, err);

    HipRemoteGetDeviceResponse resp = {
        .header = { .error_code = (int32_t)err },
        .device_id = device
    };
    send_response(fd, HIP_OP_GET_DEVICE, request_id, &resp, sizeof(resp));
}

static void handle_device_synchronize(int fd, uint32_t request_id) {
    hipError_t err = hipDeviceSynchronize();
    LOG_DEBUG("DeviceSynchronize: err=%d", err);
    send_simple_response(fd, HIP_OP_DEVICE_SYNCHRONIZE, request_id, err);
}

static void handle_get_device_properties(int fd, uint32_t request_id,
                                         const void* payload, size_t payload_size) {
    int device = 0;
    if (payload && payload_size >= sizeof(HipRemoteDeviceRequest)) {
        const HipRemoteDeviceRequest* req = (const HipRemoteDeviceRequest*)payload;
        device = req->device_id;
    }

    hipDeviceProp_t props;
    memset(&props, 0, sizeof(props));
    hipError_t err = hipGetDeviceProperties(&props, device);
    LOG_DEBUG("GetDeviceProperties: device=%d, name=%s, err=%d",
              device, props.name, err);

    HipRemoteDevicePropertiesResponse resp;
    memset(&resp, 0, sizeof(resp));
    resp.header.error_code = (int32_t)err;

    if (err == hipSuccess) {
        strncpy(resp.name, props.name, sizeof(resp.name) - 1);
        resp.total_global_mem = props.totalGlobalMem;
        resp.shared_mem_per_block = props.sharedMemPerBlock;
        resp.regs_per_block = props.regsPerBlock;
        resp.warp_size = props.warpSize;
        resp.max_threads_per_block = props.maxThreadsPerBlock;
        resp.max_threads_dim[0] = props.maxThreadsDim[0];
        resp.max_threads_dim[1] = props.maxThreadsDim[1];
        resp.max_threads_dim[2] = props.maxThreadsDim[2];
        resp.max_grid_size[0] = props.maxGridSize[0];
        resp.max_grid_size[1] = props.maxGridSize[1];
        resp.max_grid_size[2] = props.maxGridSize[2];
        resp.clock_rate = props.clockRate;
        resp.memory_clock_rate = props.memoryClockRate;
        resp.memory_bus_width = props.memoryBusWidth;
        resp.major = props.major;
        resp.minor = props.minor;
        resp.multi_processor_count = props.multiProcessorCount;
        resp.l2_cache_size = props.l2CacheSize;
        resp.max_threads_per_multi_processor = props.maxThreadsPerMultiProcessor;
        resp.compute_mode = props.computeMode;
        resp.pci_bus_id = props.pciBusID;
        resp.pci_device_id = props.pciDeviceID;
        resp.pci_domain_id = props.pciDomainID;
        resp.integrated = props.integrated;
        resp.can_map_host_memory = props.canMapHostMemory;
        resp.concurrent_kernels = props.concurrentKernels;
        strncpy(resp.gcn_arch_name, props.gcnArchName, sizeof(resp.gcn_arch_name) - 1);
    }

    send_response(fd, HIP_OP_GET_DEVICE_PROPERTIES, request_id, &resp, sizeof(resp));
}

static void handle_malloc(int fd, uint32_t request_id,
                          const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteMallocRequest)) {
        send_simple_response(fd, HIP_OP_MALLOC, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteMallocRequest* req = (const HipRemoteMallocRequest*)payload;
    void* ptr = NULL;
    hipError_t err = hipMalloc(&ptr, req->size);
    LOG_DEBUG("Malloc: size=%lu, ptr=%p, err=%d", (unsigned long)req->size, ptr, err);

    HipRemoteMallocResponse resp = {
        .header = { .error_code = (int32_t)err },
        .device_ptr = (uint64_t)(uintptr_t)ptr
    };
    send_response(fd, HIP_OP_MALLOC, request_id, &resp, sizeof(resp));
}

static void handle_free(int fd, uint32_t request_id,
                        const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteFreeRequest)) {
        send_simple_response(fd, HIP_OP_FREE, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteFreeRequest* req = (const HipRemoteFreeRequest*)payload;
    void* ptr = (void*)(uintptr_t)req->device_ptr;
    hipError_t err = hipFree(ptr);
    LOG_DEBUG("Free: ptr=%p, err=%d", ptr, err);
    send_simple_response(fd, HIP_OP_FREE, request_id, err);
}

static void handle_memcpy(int fd, uint32_t request_id,
                          const void* payload, size_t payload_size,
                          bool has_inline_data) {
    if (!payload || payload_size < sizeof(HipRemoteMemcpyRequest)) {
        send_simple_response(fd, HIP_OP_MEMCPY, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteMemcpyRequest* req = (const HipRemoteMemcpyRequest*)payload;
    hipError_t err = hipSuccess;

    LOG_DEBUG("Memcpy: dst=%p, src=%p, size=%lu, kind=%d",
              (void*)(uintptr_t)req->dst, (void*)(uintptr_t)req->src,
              (unsigned long)req->size, req->kind);

    if (req->kind == hipMemcpyHostToDevice && has_inline_data) {
        /* Inline data follows request struct */
        const uint8_t* data = (const uint8_t*)payload + sizeof(HipRemoteMemcpyRequest);
        size_t data_available = payload_size - sizeof(HipRemoteMemcpyRequest);

        if (data_available >= req->size) {
            err = hipMemcpy((void*)(uintptr_t)req->dst, data, req->size,
                            hipMemcpyHostToDevice);
        } else {
            err = hipErrorInvalidValue;
        }
        send_simple_response(fd, HIP_OP_MEMCPY, request_id, err);

    } else if (req->kind == hipMemcpyDeviceToHost) {
        /* Need to send data back */
        void* buffer = malloc(req->size);
        if (!buffer) {
            send_simple_response(fd, HIP_OP_MEMCPY, request_id, hipErrorOutOfMemory);
            return;
        }

        err = hipMemcpy(buffer, (void*)(uintptr_t)req->src, req->size,
                        hipMemcpyDeviceToHost);

        if (err == hipSuccess) {
            /* Send response header + data */
            HipRemoteHeader header;
            hip_remote_init_header(&header, HIP_OP_MEMCPY, request_id,
                                   sizeof(HipRemoteMemcpyResponse) + req->size);
            header.flags |= HIP_REMOTE_FLAG_RESPONSE | HIP_REMOTE_FLAG_HAS_INLINE_DATA;

            HipRemoteMemcpyResponse resp = {
                .header = { .error_code = (int32_t)err }
            };

            send_all(fd, &header, sizeof(header));
            send_all(fd, &resp, sizeof(resp));
            send_all(fd, buffer, req->size);
        } else {
            send_simple_response(fd, HIP_OP_MEMCPY, request_id, err);
        }

        free(buffer);

    } else if (req->kind == hipMemcpyDeviceToDevice) {
        err = hipMemcpy((void*)(uintptr_t)req->dst, (void*)(uintptr_t)req->src,
                        req->size, hipMemcpyDeviceToDevice);
        send_simple_response(fd, HIP_OP_MEMCPY, request_id, err);

    } else {
        send_simple_response(fd, HIP_OP_MEMCPY, request_id, hipErrorInvalidValue);
    }
}

static void handle_memset(int fd, uint32_t request_id,
                          const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteMemsetRequest)) {
        send_simple_response(fd, HIP_OP_MEMSET, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteMemsetRequest* req = (const HipRemoteMemsetRequest*)payload;
    hipError_t err = hipMemset((void*)(uintptr_t)req->dst, req->value, req->size);
    LOG_DEBUG("Memset: dst=%p, value=%d, size=%lu, err=%d",
              (void*)(uintptr_t)req->dst, req->value, (unsigned long)req->size, err);
    send_simple_response(fd, HIP_OP_MEMSET, request_id, err);
}

static void handle_mem_get_info(int fd, uint32_t request_id) {
    size_t free_bytes = 0, total_bytes = 0;
    hipError_t err = hipMemGetInfo(&free_bytes, &total_bytes);
    LOG_DEBUG("MemGetInfo: free=%lu, total=%lu, err=%d",
              (unsigned long)free_bytes, (unsigned long)total_bytes, err);

    HipRemoteMemGetInfoResponse resp = {
        .header = { .error_code = (int32_t)err },
        .free_bytes = free_bytes,
        .total_bytes = total_bytes
    };
    send_response(fd, HIP_OP_MEM_GET_INFO, request_id, &resp, sizeof(resp));
}

static void handle_stream_create(int fd, uint32_t request_id,
                                 const void* payload, size_t payload_size) {
    unsigned int flags = 0;
    if (payload && payload_size >= sizeof(HipRemoteStreamCreateRequest)) {
        const HipRemoteStreamCreateRequest* req = (const HipRemoteStreamCreateRequest*)payload;
        flags = req->flags;
    }

    hipStream_t stream = NULL;
    hipError_t err = hipStreamCreateWithFlags(&stream, flags);
    LOG_DEBUG("StreamCreate: flags=%u, stream=%p, err=%d", flags, stream, err);

    HipRemoteStreamCreateResponse resp = {
        .header = { .error_code = (int32_t)err },
        .stream = (uint64_t)(uintptr_t)stream
    };
    send_response(fd, HIP_OP_STREAM_CREATE, request_id, &resp, sizeof(resp));
}

static void handle_stream_destroy(int fd, uint32_t request_id,
                                  const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteStreamRequest)) {
        send_simple_response(fd, HIP_OP_STREAM_DESTROY, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteStreamRequest* req = (const HipRemoteStreamRequest*)payload;
    hipStream_t stream = (hipStream_t)(uintptr_t)req->stream;
    hipError_t err = hipStreamDestroy(stream);
    LOG_DEBUG("StreamDestroy: stream=%p, err=%d", stream, err);
    send_simple_response(fd, HIP_OP_STREAM_DESTROY, request_id, err);
}

static void handle_stream_synchronize(int fd, uint32_t request_id,
                                      const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteStreamRequest)) {
        send_simple_response(fd, HIP_OP_STREAM_SYNCHRONIZE, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteStreamRequest* req = (const HipRemoteStreamRequest*)payload;
    hipStream_t stream = (hipStream_t)(uintptr_t)req->stream;
    hipError_t err = hipStreamSynchronize(stream);
    LOG_DEBUG("StreamSynchronize: stream=%p, err=%d", stream, err);
    send_simple_response(fd, HIP_OP_STREAM_SYNCHRONIZE, request_id, err);
}

static void handle_event_create(int fd, uint32_t request_id,
                                const void* payload, size_t payload_size) {
    unsigned int flags = 0;
    if (payload && payload_size >= sizeof(HipRemoteEventCreateRequest)) {
        const HipRemoteEventCreateRequest* req = (const HipRemoteEventCreateRequest*)payload;
        flags = req->flags;
    }

    hipEvent_t event = NULL;
    hipError_t err = hipEventCreateWithFlags(&event, flags);
    LOG_DEBUG("EventCreate: flags=%u, event=%p, err=%d", flags, event, err);

    HipRemoteEventCreateResponse resp = {
        .header = { .error_code = (int32_t)err },
        .event = (uint64_t)(uintptr_t)event
    };
    send_response(fd, HIP_OP_EVENT_CREATE, request_id, &resp, sizeof(resp));
}

static void handle_event_destroy(int fd, uint32_t request_id,
                                 const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteEventRequest)) {
        send_simple_response(fd, HIP_OP_EVENT_DESTROY, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteEventRequest* req = (const HipRemoteEventRequest*)payload;
    hipEvent_t event = (hipEvent_t)(uintptr_t)req->event;
    hipError_t err = hipEventDestroy(event);
    LOG_DEBUG("EventDestroy: event=%p, err=%d", event, err);
    send_simple_response(fd, HIP_OP_EVENT_DESTROY, request_id, err);
}

static void handle_event_record(int fd, uint32_t request_id,
                                const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteEventRecordRequest)) {
        send_simple_response(fd, HIP_OP_EVENT_RECORD, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteEventRecordRequest* req = (const HipRemoteEventRecordRequest*)payload;
    hipEvent_t event = (hipEvent_t)(uintptr_t)req->event;
    hipStream_t stream = (hipStream_t)(uintptr_t)req->stream;
    hipError_t err = hipEventRecord(event, stream);
    LOG_DEBUG("EventRecord: event=%p, stream=%p, err=%d", event, stream, err);
    send_simple_response(fd, HIP_OP_EVENT_RECORD, request_id, err);
}

static void handle_event_synchronize(int fd, uint32_t request_id,
                                     const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteEventRequest)) {
        send_simple_response(fd, HIP_OP_EVENT_SYNCHRONIZE, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteEventRequest* req = (const HipRemoteEventRequest*)payload;
    hipEvent_t event = (hipEvent_t)(uintptr_t)req->event;
    hipError_t err = hipEventSynchronize(event);
    LOG_DEBUG("EventSynchronize: event=%p, err=%d", event, err);
    send_simple_response(fd, HIP_OP_EVENT_SYNCHRONIZE, request_id, err);
}

static void handle_event_elapsed_time(int fd, uint32_t request_id,
                                      const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteEventElapsedTimeRequest)) {
        send_simple_response(fd, HIP_OP_EVENT_ELAPSED_TIME, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteEventElapsedTimeRequest* req = (const HipRemoteEventElapsedTimeRequest*)payload;
    hipEvent_t start = (hipEvent_t)(uintptr_t)req->start_event;
    hipEvent_t end = (hipEvent_t)(uintptr_t)req->end_event;
    float ms = 0.0f;
    hipError_t err = hipEventElapsedTime(&ms, start, end);
    LOG_DEBUG("EventElapsedTime: start=%p, end=%p, ms=%.3f, err=%d", start, end, ms, err);

    HipRemoteEventElapsedTimeResponse resp = {
        .header = { .error_code = (int32_t)err },
        .milliseconds = ms
    };
    send_response(fd, HIP_OP_EVENT_ELAPSED_TIME, request_id, &resp, sizeof(resp));
}

static void handle_runtime_get_version(int fd, uint32_t request_id) {
    int version = 0;
    hipError_t err = hipRuntimeGetVersion(&version);
    LOG_DEBUG("RuntimeGetVersion: version=%d, err=%d", version, err);

    HipRemoteVersionResponse resp = {
        .header = { .error_code = (int32_t)err },
        .version = version
    };
    send_response(fd, HIP_OP_RUNTIME_GET_VERSION, request_id, &resp, sizeof(resp));
}

static void handle_driver_get_version(int fd, uint32_t request_id) {
    int version = 0;
    hipError_t err = hipDriverGetVersion(&version);
    LOG_DEBUG("DriverGetVersion: version=%d, err=%d", version, err);

    HipRemoteVersionResponse resp = {
        .header = { .error_code = (int32_t)err },
        .version = version
    };
    send_response(fd, HIP_OP_DRIVER_GET_VERSION, request_id, &resp, sizeof(resp));
}

static void handle_get_last_error(int fd, uint32_t request_id) {
    hipError_t err = hipGetLastError();
    LOG_DEBUG("GetLastError: err=%d", err);
    send_simple_response(fd, HIP_OP_GET_LAST_ERROR, request_id, err);
}

static void handle_peek_at_last_error(int fd, uint32_t request_id) {
    hipError_t err = hipPeekAtLastError();
    LOG_DEBUG("PeekAtLastError: err=%d", err);
    send_simple_response(fd, HIP_OP_PEEK_AT_LAST_ERROR, request_id, err);
}

/* ============================================================================
 * Module and Kernel Handlers
 * ============================================================================ */

static void handle_module_load_data(int fd, uint32_t request_id,
                                     const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteModuleLoadRequest)) {
        send_simple_response(fd, HIP_OP_MODULE_LOAD_DATA, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteModuleLoadRequest* req = (const HipRemoteModuleLoadRequest*)payload;
    const void* code_data = (const uint8_t*)payload + sizeof(HipRemoteModuleLoadRequest);
    size_t code_size = payload_size - sizeof(HipRemoteModuleLoadRequest);

    if (code_size < req->data_size) {
        LOG_ERROR("ModuleLoadData: incomplete data (got %zu, expected %lu)", code_size, req->data_size);
        send_simple_response(fd, HIP_OP_MODULE_LOAD_DATA, request_id, hipErrorInvalidValue);
        return;
    }

    hipModule_t module = NULL;
    hipError_t err = hipModuleLoadData(&module, code_data);
    LOG_DEBUG("ModuleLoadData: size=%lu, module=%p, err=%d", req->data_size, (void*)module, err);

    HipRemoteModuleLoadResponse resp = {
        .header = { .error_code = (int32_t)err },
        .module = (uint64_t)(uintptr_t)module
    };
    send_response(fd, HIP_OP_MODULE_LOAD_DATA, request_id, &resp, sizeof(resp));
}

static void handle_module_unload(int fd, uint32_t request_id,
                                  const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteModuleUnloadRequest)) {
        send_simple_response(fd, HIP_OP_MODULE_UNLOAD, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteModuleUnloadRequest* req = (const HipRemoteModuleUnloadRequest*)payload;
    hipModule_t module = (hipModule_t)(uintptr_t)req->module;

    hipError_t err = hipModuleUnload(module);
    LOG_DEBUG("ModuleUnload: module=%p, err=%d", (void*)module, err);
    send_simple_response(fd, HIP_OP_MODULE_UNLOAD, request_id, err);
}

static void handle_module_get_function(int fd, uint32_t request_id,
                                        const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteModuleGetFunctionRequest)) {
        send_simple_response(fd, HIP_OP_MODULE_GET_FUNCTION, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteModuleGetFunctionRequest* req = (const HipRemoteModuleGetFunctionRequest*)payload;
    hipModule_t module = (hipModule_t)(uintptr_t)req->module;

    hipFunction_t function = NULL;
    hipError_t err = hipModuleGetFunction(&function, module, req->function_name);

    /* Query the number of kernel arguments.
     * Note: hipKernelGetParamInfo is available in newer ROCm versions.
     * For older versions, return 0 and client falls back to NULL termination. */
    uint32_t num_args = 0;
#ifdef HIP_KERNEL_GET_PARAM_INFO_AVAILABLE
    if (err == hipSuccess && function != NULL) {
        for (uint32_t i = 0; i < HIP_REMOTE_MAX_KERNEL_ARGS; i++) {
            size_t offset = 0;
            hipError_t param_err = hipKernelGetParamInfo((hipKernel_t)function, i, &offset, NULL);
            if (param_err != hipSuccess) {
                break;
            }
            num_args++;
        }
    }
#else
    /* hipKernelGetParamInfo not available - client will use NULL-terminated array */
    (void)function;  /* Suppress unused warning */
#endif

    LOG_DEBUG("ModuleGetFunction: module=%p, name=%s, function=%p, num_args=%u, err=%d",
              (void*)module, req->function_name, (void*)function, num_args, err);

    HipRemoteModuleGetFunctionResponse resp = {
        .header = { .error_code = (int32_t)err },
        .function = (uint64_t)(uintptr_t)function,
        .num_args = num_args,
        .reserved = 0
    };
    send_response(fd, HIP_OP_MODULE_GET_FUNCTION, request_id, &resp, sizeof(resp));
}

static void handle_launch_kernel(int fd, uint32_t request_id,
                                  const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteLaunchKernelRequest)) {
        send_simple_response(fd, HIP_OP_LAUNCH_KERNEL, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteLaunchKernelRequest* req = (const HipRemoteLaunchKernelRequest*)payload;

    /* Validate sizes */
    size_t expected_min = sizeof(HipRemoteLaunchKernelRequest) +
                          req->num_args * sizeof(HipRemoteKernelArg);
    if (payload_size < expected_min) {
        LOG_ERROR("LaunchKernel: payload too small (got %zu, expected %zu)", payload_size, expected_min);
        send_simple_response(fd, HIP_OP_LAUNCH_KERNEL, request_id, hipErrorInvalidValue);
        return;
    }

    hipFunction_t function = (hipFunction_t)(uintptr_t)req->function;
    hipStream_t stream = (hipStream_t)(uintptr_t)req->stream;

    LOG_DEBUG("LaunchKernel: func=%p, grid=(%u,%u,%u), block=(%u,%u,%u), shared=%u, stream=%p, args=%u",
              (void*)function, req->grid_dim_x, req->grid_dim_y, req->grid_dim_z,
              req->block_dim_x, req->block_dim_y, req->block_dim_z,
              req->shared_mem_bytes, (void*)stream, req->num_args);

    /* Extract kernel arguments */
    const HipRemoteKernelArg* arg_descs = (const HipRemoteKernelArg*)((const uint8_t*)payload +
                                           sizeof(HipRemoteLaunchKernelRequest));
    const uint8_t* arg_data = (const uint8_t*)(arg_descs + req->num_args);

    /* Build argument pointer array for hipModuleLaunchKernel */
    void* kernel_params[HIP_REMOTE_MAX_KERNEL_ARGS];
    for (uint32_t i = 0; i < req->num_args && i < HIP_REMOTE_MAX_KERNEL_ARGS; i++) {
        kernel_params[i] = (void*)(arg_data + arg_descs[i].offset);
    }

    hipError_t err = hipModuleLaunchKernel(
        function,
        req->grid_dim_x, req->grid_dim_y, req->grid_dim_z,
        req->block_dim_x, req->block_dim_y, req->block_dim_z,
        req->shared_mem_bytes,
        stream,
        kernel_params,
        NULL  /* extra */
    );

    LOG_DEBUG("LaunchKernel: err=%d", err);
    send_simple_response(fd, HIP_OP_LAUNCH_KERNEL, request_id, err);
}

/* ============================================================================
 * Client Handler
 * ============================================================================ */

static void handle_client(int client_fd) {
    LOG_INFO("Client connected");

    while (g_running) {
        /* Read header */
        HipRemoteHeader header;
        if (recv_all(client_fd, &header, sizeof(header)) != 0) {
            LOG_DEBUG("Client disconnected");
            break;
        }

        if (hip_remote_validate_header(&header) != 0) {
            LOG_ERROR("Invalid header from client");
            break;
        }

        LOG_DEBUG("Request: %s (id=%u, payload=%u)",
                  hip_remote_op_name((HipRemoteOpCode)header.op_code),
                  header.request_id, header.payload_length);

        /* Read payload if present */
        void* payload = NULL;
        if (header.payload_length > 0) {
            payload = malloc(header.payload_length);
            if (!payload) {
                LOG_ERROR("Failed to allocate payload buffer");
                break;
            }
            if (recv_all(client_fd, payload, header.payload_length) != 0) {
                LOG_ERROR("Failed to receive payload");
                free(payload);
                break;
            }
        }

        bool has_inline_data = (header.flags & HIP_REMOTE_FLAG_HAS_INLINE_DATA) != 0;

        /* Dispatch */
        switch ((HipRemoteOpCode)header.op_code) {
            case HIP_OP_INIT:
                handle_init(client_fd, header.request_id);
                break;
            case HIP_OP_SHUTDOWN:
                handle_shutdown(client_fd, header.request_id);
                free(payload);
                goto client_done;

            case HIP_OP_GET_DEVICE_COUNT:
                handle_get_device_count(client_fd, header.request_id);
                break;
            case HIP_OP_SET_DEVICE:
                handle_set_device(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_GET_DEVICE:
                handle_get_device(client_fd, header.request_id);
                break;
            case HIP_OP_DEVICE_SYNCHRONIZE:
                handle_device_synchronize(client_fd, header.request_id);
                break;
            case HIP_OP_GET_DEVICE_PROPERTIES:
                handle_get_device_properties(client_fd, header.request_id, payload, header.payload_length);
                break;

            case HIP_OP_MALLOC:
                handle_malloc(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_FREE:
                handle_free(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_MEMCPY:
            case HIP_OP_MEMCPY_ASYNC:
                handle_memcpy(client_fd, header.request_id, payload, header.payload_length, has_inline_data);
                break;
            case HIP_OP_MEMSET:
            case HIP_OP_MEMSET_ASYNC:
                handle_memset(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_MEM_GET_INFO:
                handle_mem_get_info(client_fd, header.request_id);
                break;

            case HIP_OP_STREAM_CREATE:
            case HIP_OP_STREAM_CREATE_WITH_FLAGS:
            case HIP_OP_STREAM_CREATE_WITH_PRIORITY:
                handle_stream_create(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_STREAM_DESTROY:
                handle_stream_destroy(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_STREAM_SYNCHRONIZE:
                handle_stream_synchronize(client_fd, header.request_id, payload, header.payload_length);
                break;

            case HIP_OP_EVENT_CREATE:
            case HIP_OP_EVENT_CREATE_WITH_FLAGS:
                handle_event_create(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_EVENT_DESTROY:
                handle_event_destroy(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_EVENT_RECORD:
                handle_event_record(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_EVENT_SYNCHRONIZE:
                handle_event_synchronize(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_EVENT_ELAPSED_TIME:
                handle_event_elapsed_time(client_fd, header.request_id, payload, header.payload_length);
                break;

            case HIP_OP_RUNTIME_GET_VERSION:
                handle_runtime_get_version(client_fd, header.request_id);
                break;
            case HIP_OP_DRIVER_GET_VERSION:
                handle_driver_get_version(client_fd, header.request_id);
                break;

            case HIP_OP_GET_LAST_ERROR:
                handle_get_last_error(client_fd, header.request_id);
                break;
            case HIP_OP_PEEK_AT_LAST_ERROR:
                handle_peek_at_last_error(client_fd, header.request_id);
                break;

            case HIP_OP_MODULE_LOAD_DATA:
            case HIP_OP_MODULE_LOAD_DATA_EX:
                handle_module_load_data(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_MODULE_UNLOAD:
                handle_module_unload(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_MODULE_GET_FUNCTION:
                handle_module_get_function(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_LAUNCH_KERNEL:
            case HIP_OP_MODULE_LAUNCH_KERNEL:
                handle_launch_kernel(client_fd, header.request_id, payload, header.payload_length);
                break;

            default:
                LOG_ERROR("Unknown opcode: 0x%04x", header.op_code);
                send_simple_response(client_fd, (HipRemoteOpCode)header.op_code,
                                     header.request_id, hipErrorNotSupported);
                break;
        }

        free(payload);
    }

client_done:
    close(client_fd);
    LOG_INFO("Client disconnected");
}

/* ============================================================================
 * Signal Handling
 * ============================================================================ */

static void signal_handler(int sig) {
    (void)sig;
    g_running = false;
    if (g_server_fd >= 0) {
        shutdown(g_server_fd, SHUT_RDWR);
        close(g_server_fd);
        g_server_fd = -1;
    }
}

/* ============================================================================
 * Main
 * ============================================================================ */

static void print_usage(const char* prog) {
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  -p PORT    Listen port (default: %d)\n", HIP_REMOTE_DEFAULT_PORT);
    fprintf(stderr, "  -d DEVICE  Default GPU device (default: 0)\n");
    fprintf(stderr, "  -v         Enable verbose logging\n");
    fprintf(stderr, "  -h         Show this help\n");
    fprintf(stderr, "\nEnvironment:\n");
    fprintf(stderr, "  TF_WORKER_PORT     Listen port\n");
    fprintf(stderr, "  TF_DEVICE_ID       Default device\n");
    fprintf(stderr, "  TF_DEBUG           Enable debug (1/0)\n");
}

int main(int argc, char** argv) {
    /* Parse environment */
    const char* port_str = getenv("TF_WORKER_PORT");
    if (port_str) g_listen_port = atoi(port_str);

    const char* device_str = getenv("TF_DEVICE_ID");
    if (device_str) g_default_device = atoi(device_str);

    const char* debug = getenv("TF_DEBUG");
    if (debug && strcmp(debug, "1") == 0) g_debug_enabled = true;

    /* Parse arguments */
    int opt;
    while ((opt = getopt(argc, argv, "p:d:vh")) != -1) {
        switch (opt) {
            case 'p':
                g_listen_port = atoi(optarg);
                break;
            case 'd':
                g_default_device = atoi(optarg);
                break;
            case 'v':
                g_debug_enabled = true;
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }

    /* Set up signal handlers */
    signal(SIGPIPE, SIG_IGN);
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    /* Initialize HIP */
    LOG_INFO("Initializing HIP on device %d...", g_default_device);
    hipError_t err = hipSetDevice(g_default_device);
    if (err != hipSuccess) {
        LOG_ERROR("Failed to set device %d: %s", g_default_device, hipGetErrorString(err));
        return 1;
    }

    /* Print device info */
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, g_default_device) == hipSuccess) {
        LOG_INFO("Device: %s", props.name);
        LOG_INFO("  Memory: %.1f GB", props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        LOG_INFO("  Compute: %d.%d", props.major, props.minor);
    }

    /* Create server socket */
    g_server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (g_server_fd < 0) {
        LOG_ERROR("Failed to create socket: %s", strerror(errno));
        return 1;
    }

    int opt_val = 1;
    setsockopt(g_server_fd, SOL_SOCKET, SO_REUSEADDR, &opt_val, sizeof(opt_val));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons((uint16_t)g_listen_port);

    if (bind(g_server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        LOG_ERROR("Failed to bind to port %d: %s", g_listen_port, strerror(errno));
        return 1;
    }

    if (listen(g_server_fd, 5) < 0) {
        LOG_ERROR("Failed to listen: %s", strerror(errno));
        return 1;
    }

    LOG_INFO("Listening on port %d", g_listen_port);

    /* Accept clients */
    while (g_running) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(g_server_fd, (struct sockaddr*)&client_addr, &client_len);

        if (client_fd < 0) {
            if (errno == EINTR) continue;
            if (!g_running) break;
            LOG_ERROR("Accept failed: %s", strerror(errno));
            continue;
        }

        /* Set socket options */
        int nodelay = 1;
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));

        struct timeval io_timeout = { .tv_sec = 60, .tv_usec = 0 };
        setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &io_timeout, sizeof(io_timeout));
        setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &io_timeout, sizeof(io_timeout));

        /* Handle client (single-threaded for simplicity) */
        handle_client(client_fd);
    }

    if (g_server_fd >= 0) {
        close(g_server_fd);
    }

    LOG_INFO("Shutting down");
    return 0;
}
