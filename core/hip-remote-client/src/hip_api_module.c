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
 * @file hip_api_module.c
 * @brief Module loading and kernel launch API implementation for remote HIP
 */

#include "hip_remote/hip_remote_client.h"
#include "hip_remote/hip_remote_protocol.h"

#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Module Management
 * ============================================================================ */

hipError_t hipModuleLoadData(hipModule_t* module, const void* image) {
    if (!module || !image) {
        return hipErrorInvalidValue;
    }

    /* Determine code object size by reading ELF header */
    /* For AMDGPU code objects, we need to parse the ELF to get size */
    /* For now, we require the caller to use hipModuleLoadDataEx with size hint */
    /* or use a simpler approach: scan for reasonable size markers */

    /* ELF magic check */
    const unsigned char* data = (const unsigned char*)image;
    if (data[0] != 0x7f || data[1] != 'E' || data[2] != 'L' || data[3] != 'F') {
        hip_remote_log_error("hipModuleLoadData: invalid ELF magic");
        return hipErrorInvalidImage;
    }

    /* Parse ELF header to get size (simplified - assumes 64-bit ELF) */
    /* e_shoff + (e_shnum * e_shentsize) gives approximate end */
    uint64_t e_shoff = *(uint64_t*)(data + 40);     /* Section header offset */
    uint16_t e_shentsize = *(uint16_t*)(data + 58); /* Section header entry size */
    uint16_t e_shnum = *(uint16_t*)(data + 60);     /* Number of section headers */

    size_t approx_size = (size_t)(e_shoff + e_shnum * e_shentsize);
    if (approx_size < 64 || approx_size > HIP_REMOTE_MAX_PAYLOAD_SIZE) {
        /* Fallback: assume a reasonable max size and let server validate */
        approx_size = 16 * 1024 * 1024; /* 16MB max */
    }

    HipRemoteModuleLoadRequest req;
    req.data_size = approx_size;

    HipRemoteModuleLoadResponse resp;
    memset(&resp, 0, sizeof(resp));

    hipError_t err = hip_remote_request_with_data(
        HIP_OP_MODULE_LOAD_DATA,
        &req, sizeof(req),
        image, approx_size,
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *module = (hipModule_t)(uintptr_t)resp.module;
    }

    return err;
}

hipError_t hipModuleLoadDataEx(hipModule_t* module, const void* image,
                                unsigned int numOptions,
                                hipJitOption* options, void** optionValues) {
    /* For remote execution, JIT options are handled on the worker side */
    /* We just forward the request - options will be ignored for now */
    (void)numOptions;
    (void)options;
    (void)optionValues;

    return hipModuleLoadData(module, image);
}

hipError_t hipModuleUnload(hipModule_t module) {
    HipRemoteModuleUnloadRequest req;
    req.module = (uint64_t)(uintptr_t)module;

    HipRemoteResponseHeader resp;
    memset(&resp, 0, sizeof(resp));

    return hip_remote_request(
        HIP_OP_MODULE_UNLOAD,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module,
                                 const char* kname) {
    if (!function || !kname) {
        return hipErrorInvalidValue;
    }

    HipRemoteModuleGetFunctionRequest req;
    memset(&req, 0, sizeof(req));
    req.module = (uint64_t)(uintptr_t)module;
    strncpy(req.function_name, kname, sizeof(req.function_name) - 1);

    HipRemoteModuleGetFunctionResponse resp;
    memset(&resp, 0, sizeof(resp));

    hipError_t err = hip_remote_request(
        HIP_OP_MODULE_GET_FUNCTION,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *function = (hipFunction_t)(uintptr_t)resp.function;
    }

    return err;
}

/* ============================================================================
 * Kernel Launch
 * ============================================================================ */

hipError_t hipModuleLaunchKernel(hipFunction_t f,
                                  unsigned int gridDimX,
                                  unsigned int gridDimY,
                                  unsigned int gridDimZ,
                                  unsigned int blockDimX,
                                  unsigned int blockDimY,
                                  unsigned int blockDimZ,
                                  unsigned int sharedMemBytes,
                                  hipStream_t stream,
                                  void** kernelParams,
                                  void** extra) {
    if (!f) {
        return hipErrorInvalidHandle;
    }

    /* extra parameter is not supported in remote mode */
    if (extra && extra[0]) {
        hip_remote_log_error("hipModuleLaunchKernel: extra parameter not supported");
        return hipErrorNotSupported;
    }

    /* Count and validate arguments */
    uint32_t num_args = 0;
    size_t total_arg_size = 0;

    if (kernelParams) {
        /* Count arguments - kernelParams is NULL-terminated */
        while (kernelParams[num_args] != NULL && num_args < HIP_REMOTE_MAX_KERNEL_ARGS) {
            num_args++;
        }
    }

    /* For simplicity, we assume each argument is a pointer (8 bytes) */
    /* In a real implementation, we'd need argument metadata from the kernel */
    /* For now, treat each kernelParams entry as a pointer-sized value */
    size_t arg_size = sizeof(void*);
    total_arg_size = num_args * arg_size;

    /* Build the request */
    size_t request_size = sizeof(HipRemoteLaunchKernelRequest) +
                          num_args * sizeof(HipRemoteKernelArg) +
                          total_arg_size;

    uint8_t* buffer = (uint8_t*)malloc(request_size);
    if (!buffer) {
        return hipErrorOutOfMemory;
    }

    HipRemoteLaunchKernelRequest* req = (HipRemoteLaunchKernelRequest*)buffer;
    req->function = (uint64_t)(uintptr_t)f;
    req->grid_dim_x = gridDimX;
    req->grid_dim_y = gridDimY;
    req->grid_dim_z = gridDimZ;
    req->block_dim_x = blockDimX;
    req->block_dim_y = blockDimY;
    req->block_dim_z = blockDimZ;
    req->shared_mem_bytes = sharedMemBytes;
    req->stream = (uint64_t)(uintptr_t)stream;
    req->num_args = num_args;

    /* Fill in argument descriptors and data */
    HipRemoteKernelArg* args = (HipRemoteKernelArg*)(buffer + sizeof(HipRemoteLaunchKernelRequest));
    uint8_t* arg_data = (uint8_t*)(args + num_args);
    uint32_t offset = 0;

    for (uint32_t i = 0; i < num_args; i++) {
        args[i].size = (uint32_t)arg_size;
        args[i].offset = offset;
        memcpy(arg_data + offset, kernelParams[i], arg_size);
        offset += (uint32_t)arg_size;
    }

    HipRemoteResponseHeader resp;
    memset(&resp, 0, sizeof(resp));

    hipError_t err = hip_remote_request(
        HIP_OP_LAUNCH_KERNEL,
        buffer, request_size,
        &resp, sizeof(resp)
    );

    free(buffer);
    return err;
}

hipError_t hipLaunchKernel(const void* function_address,
                            dim3 numBlocks,
                            dim3 dimBlocks,
                            void** args,
                            size_t sharedMemBytes,
                            hipStream_t stream) {
    /*
     * hipLaunchKernel with a host function pointer is used when the kernel
     * is linked into the executable. For remote execution, we need the
     * module/function approach instead.
     *
     * This function cannot work directly in remote mode because the function
     * pointer is meaningless on the remote worker. Applications should use
     * hipModuleLaunchKernel instead.
     */
    hip_remote_log_error("hipLaunchKernel: host function pointers not supported in remote mode");
    hip_remote_log_error("Use hipModuleLoadData + hipModuleGetFunction + hipModuleLaunchKernel instead");
    (void)function_address;
    (void)numBlocks;
    (void)dimBlocks;
    (void)args;
    (void)sharedMemBytes;
    (void)stream;
    return hipErrorNotSupported;
}

/* ============================================================================
 * Cooperative Launch (stub)
 * ============================================================================ */

hipError_t hipLaunchCooperativeKernel(const void* f,
                                       dim3 gridDim,
                                       dim3 blockDim,
                                       void** kernelParams,
                                       unsigned int sharedMemBytes,
                                       hipStream_t stream) {
    /* Cooperative kernels require additional synchronization that's complex
     * to implement over the network. For now, fall back to regular launch. */
    hip_remote_log_debug("hipLaunchCooperativeKernel: using regular launch");

    return hipModuleLaunchKernel(
        (hipFunction_t)f,
        gridDim.x, gridDim.y, gridDim.z,
        blockDim.x, blockDim.y, blockDim.z,
        sharedMemBytes,
        stream,
        kernelParams,
        NULL
    );
}
