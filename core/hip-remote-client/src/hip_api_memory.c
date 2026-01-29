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
 * @file hip_api_memory.c
 * @brief Memory management API implementations for remote HIP
 */

#include "hip_remote/hip_remote_client.h"
#include "hip_remote/hip_remote_protocol.h"

#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Memory Allocation
 * ============================================================================ */

hipError_t hipMalloc(void** ptr, size_t size) {
    if (!ptr) {
        return hipErrorInvalidValue;
    }
    if (size == 0) {
        *ptr = NULL;
        return hipSuccess;
    }

    HipRemoteMallocRequest req = {
        .size = size,
        .flags = 0
    };
    HipRemoteMallocResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_MALLOC,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        /* Store the remote pointer as an opaque handle */
        *ptr = (void*)(uintptr_t)resp.device_ptr;
    } else {
        *ptr = NULL;
    }
    return err;
}

hipError_t hipFree(void* ptr) {
    if (!ptr) {
        return hipSuccess;  /* NULL free is a no-op */
    }

    HipRemoteFreeRequest req = {
        .device_ptr = (uint64_t)(uintptr_t)ptr
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_FREE,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipMallocHost(void** ptr, size_t size) {
    if (!ptr) {
        return hipErrorInvalidValue;
    }
    if (size == 0) {
        *ptr = NULL;
        return hipSuccess;
    }

    /* For host memory, we allocate locally but register with remote */
    /* This allows zero-copy transfers when possible */
    *ptr = malloc(size);
    if (!*ptr) {
        return hipErrorOutOfMemory;
    }

    /* Optionally notify remote about pinned memory (for optimization) */
    HipRemoteMallocRequest req = {
        .size = size,
        .flags = 0
    };
    HipRemoteMallocResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_MALLOC_HOST,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    /* If remote registration failed, still return the local allocation */
    if (err != hipSuccess) {
        hip_remote_log_debug("Host malloc registration failed, using local only");
    }

    return hipSuccess;
}

hipError_t hipFreeHost(void* ptr) {
    if (!ptr) {
        return hipSuccess;
    }

    /* Notify remote (best effort) */
    HipRemoteFreeRequest req = {
        .device_ptr = (uint64_t)(uintptr_t)ptr
    };
    HipRemoteResponseHeader resp;
    (void)hip_remote_request(
        HIP_OP_FREE_HOST,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    /* Free local memory */
    free(ptr);
    return hipSuccess;
}

hipError_t hipMallocManaged(void** ptr, size_t size, unsigned int flags) {
    if (!ptr) {
        return hipErrorInvalidValue;
    }
    if (size == 0) {
        *ptr = NULL;
        return hipSuccess;
    }

    HipRemoteMallocRequest req = {
        .size = size,
        .flags = flags
    };
    HipRemoteMallocResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_MALLOC_MANAGED,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *ptr = (void*)(uintptr_t)resp.device_ptr;
    } else {
        *ptr = NULL;
    }
    return err;
}

/* ============================================================================
 * Memory Copy
 * ============================================================================ */

hipError_t hipMemcpy(void* dst, const void* src, size_t size, hipMemcpyKind kind) {
    if (size == 0) {
        return hipSuccess;
    }
    if (!dst || !src) {
        return hipErrorInvalidValue;
    }

    HipRemoteMemcpyRequest req = {
        .dst = (uint64_t)(uintptr_t)dst,
        .src = (uint64_t)(uintptr_t)src,
        .size = size,
        .kind = (int32_t)kind,
        .stream = 0  /* Default stream */
    };

    switch (kind) {
        case hipMemcpyHostToDevice: {
            /* Send data to remote device */
            HipRemoteMemcpyResponse resp;
            return hip_remote_request_with_data(
                HIP_OP_MEMCPY,
                &req, sizeof(req),
                src, size,
                &resp, sizeof(resp)
            );
        }

        case hipMemcpyDeviceToHost: {
            /* Receive data from remote device */
            HipRemoteMemcpyResponse resp;
            return hip_remote_request_receive_data(
                HIP_OP_MEMCPY,
                &req, sizeof(req),
                &resp, sizeof(resp),
                dst, size
            );
        }

        case hipMemcpyDeviceToDevice: {
            /* Remote-to-remote copy, no data transfer */
            HipRemoteMemcpyResponse resp;
            return hip_remote_request(
                HIP_OP_MEMCPY,
                &req, sizeof(req),
                &resp, sizeof(resp)
            );
        }

        case hipMemcpyHostToHost: {
            /* Local copy */
            memmove(dst, src, size);
            return hipSuccess;
        }

        case hipMemcpyDefault: {
            /* Let remote figure out the direction */
            /* For now, assume D2D */
            HipRemoteMemcpyResponse resp;
            return hip_remote_request(
                HIP_OP_MEMCPY,
                &req, sizeof(req),
                &resp, sizeof(resp)
            );
        }

        default:
            return hipErrorInvalidValue;
    }
}

hipError_t hipMemcpyAsync(void* dst, const void* src, size_t size,
                          hipMemcpyKind kind, void* stream) {
    if (size == 0) {
        return hipSuccess;
    }
    if (!dst || !src) {
        return hipErrorInvalidValue;
    }

    HipRemoteMemcpyRequest req = {
        .dst = (uint64_t)(uintptr_t)dst,
        .src = (uint64_t)(uintptr_t)src,
        .size = size,
        .kind = (int32_t)kind,
        .stream = (uint64_t)(uintptr_t)stream
    };

    /* For async, we still block on the network but the GPU operation is async */
    switch (kind) {
        case hipMemcpyHostToDevice: {
            HipRemoteMemcpyResponse resp;
            return hip_remote_request_with_data(
                HIP_OP_MEMCPY_ASYNC,
                &req, sizeof(req),
                src, size,
                &resp, sizeof(resp)
            );
        }

        case hipMemcpyDeviceToHost: {
            HipRemoteMemcpyResponse resp;
            return hip_remote_request_receive_data(
                HIP_OP_MEMCPY_ASYNC,
                &req, sizeof(req),
                &resp, sizeof(resp),
                dst, size
            );
        }

        case hipMemcpyDeviceToDevice: {
            HipRemoteMemcpyResponse resp;
            return hip_remote_request(
                HIP_OP_MEMCPY_ASYNC,
                &req, sizeof(req),
                &resp, sizeof(resp)
            );
        }

        case hipMemcpyHostToHost: {
            memmove(dst, src, size);
            return hipSuccess;
        }

        default: {
            HipRemoteMemcpyResponse resp;
            return hip_remote_request(
                HIP_OP_MEMCPY_ASYNC,
                &req, sizeof(req),
                &resp, sizeof(resp)
            );
        }
    }
}

/* Convenience functions */
hipError_t hipMemcpyHtoD(void* dst, const void* src, size_t size) {
    return hipMemcpy(dst, src, size, hipMemcpyHostToDevice);
}

hipError_t hipMemcpyDtoH(void* dst, const void* src, size_t size) {
    return hipMemcpy(dst, src, size, hipMemcpyDeviceToHost);
}

hipError_t hipMemcpyDtoD(void* dst, const void* src, size_t size) {
    return hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice);
}

/* ============================================================================
 * Memory Set
 * ============================================================================ */

hipError_t hipMemset(void* dst, int value, size_t size) {
    if (size == 0) {
        return hipSuccess;
    }
    if (!dst) {
        return hipErrorInvalidValue;
    }

    HipRemoteMemsetRequest req = {
        .dst = (uint64_t)(uintptr_t)dst,
        .value = value,
        .size = size,
        .stream = 0
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_MEMSET,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipMemsetAsync(void* dst, int value, size_t size, void* stream) {
    if (size == 0) {
        return hipSuccess;
    }
    if (!dst) {
        return hipErrorInvalidValue;
    }

    HipRemoteMemsetRequest req = {
        .dst = (uint64_t)(uintptr_t)dst,
        .value = value,
        .size = size,
        .stream = (uint64_t)(uintptr_t)stream
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_MEMSET_ASYNC,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

/* ============================================================================
 * Memory Info
 * ============================================================================ */

hipError_t hipMemGetInfo(size_t* free_bytes, size_t* total_bytes) {
    if (!free_bytes || !total_bytes) {
        return hipErrorInvalidValue;
    }

    HipRemoteMemGetInfoResponse resp;
    hipError_t err = hip_remote_request(
        HIP_OP_MEM_GET_INFO,
        NULL, 0,
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *free_bytes = (size_t)resp.free_bytes;
        *total_bytes = (size_t)resp.total_bytes;
    }
    return err;
}
